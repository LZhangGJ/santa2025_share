#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

# =========================
# Precision (contest-friendly)
# =========================
getcontext().prec = 25
scale_factor = Decimal("1e18")


# =========================
# Christmas Tree Geometry (fixed)
# =========================
class ChristmasTree:
    """Represents a single, rotatable Christmas tree polygon of a fixed size."""

    def __init__(self, center_x="0", center_y="0", angle="0"):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                (Decimal("0.0") * scale_factor, tip_y * scale_factor),
                (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
                (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
                (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
                (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
            ]
        )

        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor),
        )


def _strip_prefix_to_number_str(v) -> str:
    s = str(v)
    if len(s) >= 2 and s[0].isalpha():
        return s[1:]
    return s


def load_configuration_from_df(n: int, df: pd.DataFrame) -> list[ChristmasTree]:
    prefix = f"{n:03d}_"
    group_data = df[df["id"].astype(str).str.startswith(prefix)]
    trees: list[ChristmasTree] = []

    for _, row in group_data.iterrows():
        x = _strip_prefix_to_number_str(row["x"])
        y = _strip_prefix_to_number_str(row["y"])
        deg = _strip_prefix_to_number_str(row["deg"])
        if x and y and deg:
            trees.append(ChristmasTree(x, y, deg))
    return trees


def get_score(trees: list[ChristmasTree], n: int) -> float:
    """Score contribution for this N: S^2 / N."""
    if not trees:
        return 0.0

    xys = np.concatenate(
        [np.asarray(t.polygon.exterior.xy).T / float(scale_factor) for t in trees]
    )
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)

    side_length = max(max_x - min_x, max_y - min_y)
    return (side_length ** 2) / n


def has_overlap(trees: list[ChristmasTree]) -> bool:
    """Touching is allowed. Overlap means intersect but not touches."""
    if len(trees) <= 1:
        return False

    polygons = [t.polygon for t in trees]
    tree = STRtree(polygons)

    geom_to_idx = {id(g): i for i, g in enumerate(polygons)}

    for i, poly in enumerate(polygons):
        candidates = tree.query(poly)
        for cand in candidates:
            j = geom_to_idx.get(id(cand), None)
            if j is None:
                continue
            if j == i:
                continue
            if poly.intersects(polygons[j]) and not poly.touches(polygons[j]):
                return True
    return False


def read_submission(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    for col in ["id", "x", "y", "deg"]:
        if col not in df.columns:
            raise ValueError(f"Invalid submission file (missing col={col}): {fp}")
    return df


def extract_block(df: pd.DataFrame, n: int) -> pd.DataFrame:
    prefix = f"{n:03d}_"
    blk = df[df["id"].astype(str).str.startswith(prefix)]
    return blk.copy()


def validate_submission(fp: str, max_n: int = 200, verbose: bool = True) -> dict:
    df = read_submission(fp)

    total_score = 0.0
    failed_overlap_n: list[int] = []

    if verbose:
        print(f"\n--- Validate: {fp} ---")

    for n in range(1, max_n + 1):
        trees = load_configuration_from_df(n, df)
        if not trees:
            continue

        sc = get_score(trees, n)
        total_score += sc

        if has_overlap(trees):
            failed_overlap_n.append(n)
            if verbose:
                print(f"  ❌ N={n:03d}: OVERLAP (score={sc:.6f})")

    status = "SUCCESS" if not failed_overlap_n else "FAILED (Overlaps)"
    if verbose:
        if failed_overlap_n:
            print(f"❌ Overlap Ns: {failed_overlap_n}")
        else:
            print("✅ No overlaps.")
        print(f"Total Score Σ(S²/N) = {total_score:.6f}")

    return {
        "status": status,
        "total_score": float(total_score),
        "failed_overlap_n": failed_overlap_n,
    }


def choose_best_block(
    n: int,
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    tol: float = 1e-14,
) -> tuple[pd.DataFrame, dict]:
    """
    STRICT RULE (monotonic best):
      - If new overlaps -> use old
      - Else new only accepted if new_score < old_score - tol
      - Otherwise keep old
    """
    old_blk = extract_block(old_df, n)
    new_blk = extract_block(new_df, n)

    if old_blk.empty and new_blk.empty:
        return pd.DataFrame(), {"chosen": "none", "reason": "missing_both"}

    # If old missing, we can only use new (but still reject overlap)
    if old_blk.empty:
        new_trees = load_configuration_from_df(n, new_df)
        new_ov = has_overlap(new_trees) if new_trees else False
        new_sc = get_score(new_trees, n) if new_trees else 1e30
        if new_ov:
            # nothing else to use
            return new_blk, {"chosen": "new", "reason": "old_missing_but_new_overlap_forced", "new_score": new_sc}
        return new_blk, {"chosen": "new", "reason": "old_missing", "new_overlap": new_ov, "new_score": new_sc}

    # If new missing, keep old
    if new_blk.empty:
        old_trees = load_configuration_from_df(n, old_df)
        old_ov = has_overlap(old_trees) if old_trees else False
        old_sc = get_score(old_trees, n) if old_trees else 1e30
        return old_blk, {"chosen": "old", "reason": "new_missing", "old_overlap": old_ov, "old_score": old_sc}

    old_trees = load_configuration_from_df(n, old_df)
    new_trees = load_configuration_from_df(n, new_df)

    old_overlap = has_overlap(old_trees) if old_trees else False
    new_overlap = has_overlap(new_trees) if new_trees else False

    old_score = get_score(old_trees, n) if old_trees else 1e30
    new_score = get_score(new_trees, n) if new_trees else 1e30

    # If old itself overlaps, you can decide policy. Here: prefer non-overlap even if score worse.
    # But you said best should not get worse; usually old is valid so this won't trigger.
    if old_overlap and not new_overlap:
        return new_blk, {
            "chosen": "new",
            "reason": "old_overlap_fixed",
            "old_score": old_score,
            "new_score": new_score,
            "old_overlap": old_overlap,
            "new_overlap": new_overlap,
        }

    # If new overlaps -> keep old
    if new_overlap:
        return old_blk, {
            "chosen": "old",
            "reason": "new_overlap",
            "old_score": old_score,
            "new_score": new_score,
            "old_overlap": old_overlap,
            "new_overlap": new_overlap,
        }

    # Now both non-overlap (or old overlap already handled above)
    # STRICT: only accept if strictly better
    if new_score < old_score - tol:
        return new_blk, {
            "chosen": "new",
            "reason": "strictly_better",
            "old_score": old_score,
            "new_score": new_score,
            "old_overlap": old_overlap,
            "new_overlap": new_overlap,
        }
    else:
        return old_blk, {
            "chosen": "old",
            "reason": "not_better_keep_old",
            "old_score": old_score,
            "new_score": new_score,
            "old_overlap": old_overlap,
            "new_overlap": new_overlap,
        }


def merge_old_new(
    old_fp: str,
    new_fp: str,
    out_fp: str,
    max_n: int = 200,
    verbose: bool = True,
    per_n_tol: float = 1e-14,
) -> dict:
    old_df = read_submission(old_fp)
    new_df = read_submission(new_fp)

    blocks = []
    stats = {}

    chosen_new = 0
    chosen_old = 0
    forced_from_old = []

    for n in range(1, max_n + 1):
        blk, info = choose_best_block(n, old_df, new_df, tol=per_n_tol)
        if blk.empty:
            continue
        blocks.append(blk)
        stats[n] = info

        if info.get("chosen") == "new":
            chosen_new += 1
        elif info.get("chosen") == "old":
            chosen_old += 1

        if info.get("reason") == "new_overlap":
            forced_from_old.append(n)

    merged = pd.concat(blocks, ignore_index=True)
    merged.to_csv(out_fp, index=False)

    if verbose:
        print(f"\n[MERGE] old={old_fp}")
        print(f"[MERGE] new={new_fp}")
        print(f"[MERGE] out={out_fp}")
        print(f"[MERGE] chosen_new={chosen_new}, chosen_old={chosen_old}")
        if forced_from_old:
            print(f"[MERGE] forced_from_old due to new-overlap Ns: {forced_from_old}")

    final_val = validate_submission(out_fp, max_n=max_n, verbose=verbose)

    return {
        "merged_file": out_fp,
        "chosen_new": chosen_new,
        "chosen_old": chosen_old,
        "forced_from_old": forced_from_old,
        "validation": final_val,
        "stats": stats,
    }


def run_eazy(eazy_bin: str, workdir: str, timeout_sec: int | None = None) -> None:
    eazy_bin = str(Path(eazy_bin).resolve())
    workdir = str(Path(workdir).resolve())

    if not Path(eazy_bin).exists():
        raise FileNotFoundError(f"eazy binary not found: {eazy_bin}")
    if not Path(workdir).exists():
        raise FileNotFoundError(f"workdir not found: {workdir}")

    print(f"\n[RUN] {eazy_bin}  (cwd={workdir})")
    subprocess.run([eazy_bin], cwd=workdir, check=True, timeout=timeout_sec)


def safe_copy(src: str, dst: str) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Iteratively improve Santa submission: run eazy -> validate/merge -> loop (monotonic best)"
    )
    parser.add_argument("--workdir", type=str, default="./",
                        help="Working directory (where eazy runs and writes submission.csv)")
    parser.add_argument("--eazy", type=str, default="./eazy",
                        help="Path to eazy binary")
    parser.add_argument("--init", type=str, default="./submission.csv" ,
                        help="Initial best submission csv path (seed)")
    parser.add_argument("--best_name", type=str, default="best_submission.csv",
                        help="Filename of best submission in workdir")
    parser.add_argument("--iters", type=int, default=2000,
                        help="How many outer iterations to run")
    parser.add_argument("--max_n", type=int, default=200,
                        help="Max N")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Timeout seconds for eazy per iteration (0 = no timeout)")
    parser.add_argument("--archive", type=str, default="archive_iters",
                        help="Archive folder inside workdir")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose validation prints")
    parser.add_argument("--per_n_tol", type=float, default=1e-14,
                        help="Per-N strict improvement tolerance: accept new only if new_score < old_score - tol")
    parser.add_argument("--global_eps", type=float, default=1e-12,
                        help="Global monotonic guard: reject merged if total_score > old_total + eps")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    archive_dir = workdir / args.archive
    archive_dir.mkdir(parents=True, exist_ok=True)

    best_fp = workdir / args.best_name
    init_fp = Path(args.init).resolve()

    if not init_fp.exists():
        print(f"[ERROR] init submission not found: {init_fp}")
        sys.exit(1)

    # Seed best
    safe_copy(str(init_fp), str(best_fp))
    print(f"[INIT] best <- {init_fp}")

    old_val = validate_submission(str(best_fp), max_n=args.max_n, verbose=args.verbose)
    old_total = old_val["total_score"]

    timeout_sec = None if args.timeout <= 0 else int(args.timeout)

    for it in range(1, args.iters + 1):
        print(f"\n==============================")
        print(f" Iteration {it}/{args.iters}")
        print(f"==============================")

        # Backup current best
        best_backup = archive_dir / f"best_iter_{it:03d}.csv"
        safe_copy(str(best_fp), str(best_backup))

        # Run eazy (it should read best_submission.csv from workdir!)
        try:
            run_eazy(args.eazy, str(workdir), timeout_sec=timeout_sec)
        except subprocess.TimeoutExpired:
            print(f"[WARN] eazy timeout at iteration {it}, keep previous best.")
            continue
        except Exception as e:
            print(f"[ERROR] eazy failed: {e}")
            print("[ERROR] keep previous best and continue.")
            continue

        new_fp = workdir / "submission.csv"
        if not new_fp.exists():
            print(f"[ERROR] eazy produced no submission.csv at {new_fp}")
            print("[ERROR] keep previous best and continue.")
            continue

        # Archive raw eazy output
        raw_out = archive_dir / f"raw_eazy_iter_{it:03d}.csv"
        safe_copy(str(new_fp), str(raw_out))

        # Merge with STRICT per-N rule (never accept worse per N)
        merged_tmp = workdir / "best_tmp.csv"
        merge_info = merge_old_new(
            old_fp=str(best_fp),
            new_fp=str(new_fp),
            out_fp=str(merged_tmp),
            max_n=args.max_n,
            verbose=args.verbose,
            per_n_tol=args.per_n_tol,
        )

        # Global monotonic guard: merged total must not be worse
        merged_total = merge_info["validation"]["total_score"]
        if merged_total > old_total + args.global_eps:
            print(f"\n[ITER {it}] ❌ MERGED TOTAL WORSE! merged={merged_total:.6f} old={old_total:.6f}")
            print("[ITER {it}] Keep old best. (Discard this iteration)")
            # Keep previous best (do nothing)
            continue

        # Accept merged as new best
        safe_copy(str(merged_tmp), str(best_fp))
        old_total = merged_total

        print(f"\n[ITER {it}] ✅ ACCEPTED merged_total={merged_total:.6f}")

    print("\n==============================")
    print("DONE")
    print("==============================")
    print(f"[FINAL] Best submission: {best_fp}")

    final_val = validate_submission(str(best_fp), max_n=args.max_n, verbose=True)
    print(final_val)


if __name__ == "__main__":
    main()
