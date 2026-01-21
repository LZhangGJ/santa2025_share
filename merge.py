import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from pathlib import Path

# Set precision for Decimal (contest-friendly)
submission_files = [
        'submission.csv',
        'best_submission.csv',

    ]
getcontext().prec = 25
scale_factor = Decimal("1e18")


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

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

        # Define the 15 vertices of the tree polygon
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

        # Apply rotation and translation (Shapely is float-based internally)
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor),
        )


def _strip_prefix_to_number_str(v) -> str:
    """
    Your original code assumed values look like "s0.123" / "s-0.123".
    Keep that behavior, but be a bit safer:
    - if it's a string starting with a letter (e.g., 's'), drop first char
    - otherwise keep as-is
    """
    s = str(v)
    if len(s) >= 2 and s[0].isalpha():
        return s[1:]
    return s


def load_configuration_from_df(n: int, df: pd.DataFrame) -> list[ChristmasTree]:
    """Loads all trees for a given N from the submission DataFrame."""
    group_data = df[df["id"].astype(str).str.startswith(f"{n:03d}_")]
    trees: list[ChristmasTree] = []

    for _, row in group_data.iterrows():
        x = _strip_prefix_to_number_str(row["x"])
        y = _strip_prefix_to_number_str(row["y"])
        deg = _strip_prefix_to_number_str(row["deg"])

        if x and y and deg:
            trees.append(ChristmasTree(x, y, deg))

    return trees


def get_score(trees: list[ChristmasTree], n: int) -> float:
    """Score contribution for this N: S^2 / N (S is min bounding square side)."""
    if not trees:
        return 0.0

    xys = np.concatenate(
        [np.asarray(t.polygon.exterior.xy).T / float(scale_factor) for t in trees]
    )
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)

    side_length = max(max_x - min_x, max_y - min_y)
    return (side_length**2) / n


def has_overlap(trees: list[ChristmasTree]) -> bool:
    """Check if any two ChristmasTree polygons overlap (touching is allowed)."""
    if len(trees) <= 1:
        return False

    polygons = [t.polygon for t in trees]
    idx = STRtree(polygons)

    for i, poly in enumerate(polygons):
        candidates = idx.query(poly)
        for j in candidates:
            if j == i:
                continue
            if poly.intersects(polygons[j]) and not poly.touches(polygons[j]):
                return True
    return False


def score_and_validate_submission(file_path: str, max_n: int = 200) -> dict:
    """Reads a submission CSV, totals score, and checks overlaps for each N."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {"status": "FAILED", "error": "File Not Found"}
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {"status": "FAILED", "error": f"CSV Read Error: {e}"}

    total_score = 0.0
    failed_overlap_n: list[int] = []

    print(f"--- Scoring and Validation: {file_path} (N=1 to {max_n}) ---")
    for n in range(1, max_n + 1):
        trees = load_configuration_from_df(n, df)
        if not trees:
            continue

        current_score = get_score(trees, n)
        total_score += current_score

        if has_overlap(trees):
            failed_overlap_n.append(n)
            print(f"  ❌ N={n:03d}: OVERLAP DETECTED! (Score: {current_score:.6f})")

    print("\n--- Summary ---")
    if failed_overlap_n:
        print(f"❌ Validation FAILED: overlaps in N: {failed_overlap_n}")
        status = "FAILED (Overlaps)"
    else:
        print("✅ Validation SUCCESSFUL: No overlaps detected.")
        status = "SUCCESS"

    print(f"Total Submission Score (Σ S²/N): {total_score:.6f}")

    return {
        "status": status,
        "total_score": total_score,
        "failed_overlap_n": failed_overlap_n,
    }


def merge_best_submissions(
    file_paths: list[str], output_path: str, max_n: int = 200
) -> dict:
    """
    Merge by selecting the best (lowest score) NON-OVERLAPPING configuration for each N.
    If the best-scoring configuration overlaps, try the next best, etc.

    If *all* candidates overlap for some N, we still write the best-scoring one (so output is complete),
    but we record that N in `forced_overlap_n`.
    """
    print(f"--- Merging {len(file_paths)} submission files (overlap-aware) ---")

    # Load all dataframes
    dfs: list[tuple[str, pd.DataFrame]] = []
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
            dfs.append((fp, df))
            print(f"✓ Loaded: {fp} ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Failed to load {fp}: {e}")

    if not dfs:
        return {"status": "FAILED", "error": "No valid files loaded"}

    best_configs: list[pd.DataFrame] = []
    selection_stats: dict[int, dict] = {}
    forced_overlap_n: list[int] = []
    missing_n: list[int] = []

    for n in range(1, max_n + 1):
        candidates = []
        for file_path, df in dfs:
            # Fast prefilter: only attempt if this df has this N
            # (still O(rows) string ops, but okay; you can optimize with precomputed masks if needed)
            block = df[df["id"].astype(str).str.startswith(f"{n:03d}_")]
            if block.empty:
                continue

            trees = load_configuration_from_df(n, df)
            if not trees:
                continue

            score = get_score(trees, n)
            overlap = has_overlap(trees)
            candidates.append(
                {
                    "score": score,
                    "overlap": overlap,
                    "file_path": file_path,
                    "source": Path(file_path).name,
                    "block": block.copy(),
                }
            )

        if not candidates:
            missing_n.append(n)
            continue

        # Sort by score ascending (best first)
        candidates.sort(key=lambda x: x["score"])

        # Pick the first non-overlapping candidate; otherwise force best
        chosen = None
        for c in candidates:
            if not c["overlap"]:
                chosen = c
                break

        if chosen is None:
            chosen = candidates[0]
            forced_overlap_n.append(n)

        best_configs.append(chosen["block"])
        selection_stats[n] = {
            "score": float(chosen["score"]),
            "source": chosen["source"],
            "overlap": bool(chosen["overlap"]),
            "rank_used": int(candidates.index(chosen) + 1),  # 1-based rank among candidates
            "num_candidates": int(len(candidates)),
        }

        tag = "⚠️FORCED_OVERLAP" if chosen["overlap"] else "OK"
        print(
            f"N={n:03d}: score={chosen['score']:.6f} from {chosen['source']} "
            f"[{tag}, rank {selection_stats[n]['rank_used']}/{selection_stats[n]['num_candidates']}]"
        )

    if not best_configs:
        return {"status": "FAILED", "error": "No valid configurations found"}

    merged_df = pd.concat(best_configs, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    print(f"\n✓ Merged submission saved to: {output_path}")

    if missing_n:
        print(f"\n⚠️ Missing N blocks (not found in any file): {missing_n}")

    if forced_overlap_n:
        print(f"\n⚠️ Forced-overlap N blocks (all candidates overlapped): {forced_overlap_n}")

    # Final validation
    print("\n--- Validating Merged Submission ---")
    validation_result = score_and_validate_submission(output_path, max_n=max_n)

    # Source usage stats
    source_counts: dict[str, int] = {}
    for stats in selection_stats.values():
        src = stats["source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    print("\n--- Source File Usage ---")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count} configurations ({count/max_n*100:.1f}%)")

    return {
        "status": validation_result["status"],
        "total_score": validation_result.get("total_score"),
        "failed_overlap_n": validation_result.get("failed_overlap_n", []),
        "forced_overlap_n": forced_overlap_n,
        "missing_n": missing_n,
        "merged_file": output_path,
        "source_counts": source_counts,
        "selection_stats": selection_stats,
    }


# Example usage
if __name__ == "__main__":
    # Use your already-defined `submission_files` list from above (copy part kept separate as you asked)
    output_file = "submission_merged.csv"

    result = merge_best_submissions(
        file_paths=submission_files,
        output_path=output_file,
        max_n=200,
    )

    print("\n=== Final Result ===")
    print(result)
