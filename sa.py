# ==============================
# SA optimizer for Santa 2025
# - starts from /kaggle/working/submission.csv
# - proposes via ./bbox3 with random (n,r)
# - accepts by SA rule
# - fast eval: only recompute changed groups
# ==============================

import os, math, random, time, shutil, tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from shapely.geometry import Polygon
from shapely.strtree import STRtree

# ------------------------------
# CONFIG
# ------------------------------
WORK_SUB = "best_submission.csv"

WORK_SUB2 = "submission_best_sa.csv"
BBOX = "bbox3"   # or "./bbox3" if you prefer
MAX_HOURS = 7.5                 # stop by wall-clock
MAX_ITERS = 999999              # also a hard cap
BBOX_TIMEOUT_SEC = 1200         # per bbox3 run timeout (20min)

# SA schedule
T0 = 1e-4       # initial temperature (score units)
T_MIN = 1e-8
ALPHA = 0.995   # T *= ALPHA each iter

# Proposal parameter space (you can tweak)
N_GRID = list(range(1000, 2001, 100))     # typical sweep
R_GRID = list(range(10,  91,   10))

# Occasionally do a more "wild" move
P_WILD = 0.10
N_WILD = (50, 3000)   # randint range
R_WILD = (5, 200)

# Saving
SAVE_EVERY = 25
BEST_PATH = "submission_best_sa.csv"
LOG_PATH  = "sa_log.txt"

# ------------------------------
# Helpers
# ------------------------------
TREE_PTS = np.array([
    (0,0.8),(0.125,0.5),(0.0625,0.5),(0.2,0.25),(0.1,0.25),(0.35,0),
    (0.075,0),(0.075,-0.2),(-0.075,-0.2),(-0.075,0),(-0.35,0),
    (-0.1,0.25),(-0.2,0.25),(-0.0625,0.5),(-0.125,0.5)
], dtype=np.float64)

def _strip_prefix_num(s: str) -> float:
    """Parse 's1.23' / 'x1.23' / 'd90' etc -> float."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    # remove leading non-numeric chars
    i = 0
    while i < len(s) and (s[i] not in "+-0123456789."):
        i += 1
    return float(s[i:]) if i < len(s) else 0.0

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize types
    df["id"]  = df["id"].astype(str)
    df["x"]   = df["x"].astype(str)
    df["y"]   = df["y"].astype(str)
    df["deg"] = df["deg"].astype(str)
    df["group_id"] = df["id"].str.split("_", n=1, expand=True)[0]
    return df

def group_signature(gdf: pd.DataFrame) -> tuple:
    """
    A stable signature to detect if a group's (x,y,deg) changed.
    Keep it cheap: tuple of strings in id order.
    """
    gg = gdf.sort_values("id")[["x","y","deg"]]
    return (tuple(gg["x"].tolist()), tuple(gg["y"].tolist()), tuple(gg["deg"].tolist()))

def build_polys(xs, ys, degs):
    """
    Build shapely polygons for overlap test.
    xs, ys, degs are float arrays.
    """
    polys = []
    for x, y, d in zip(xs, ys, degs):
        r = np.deg2rad(d)
        c, s = np.cos(r), np.sin(r)
        pts = np.empty_like(TREE_PTS)
        pts[:, 0] = TREE_PTS[:, 0] * c - TREE_PTS[:, 1] * s + x
        pts[:, 1] = TREE_PTS[:, 0] * s + TREE_PTS[:, 1] * c + y
        polys.append(Polygon(pts))
    return polys

def has_overlap_polys(polys) -> bool:
    if len(polys) < 2:
        return False
    idx = STRtree(polys)
    for i, p in enumerate(polys):
        for j in idx.query(p):
            if i == j:
                continue
            # "intersects but not just touches" => overlap
            if p.intersects(polys[j]) and (not p.touches(polys[j])):
                return True
    return False

def eval_group(gdf: pd.DataFrame, n: int) -> float:
    """
    Return contribution (S^2/n) for this group.
    If overlap found -> return +inf (invalid).
    """
    xs = gdf["x"].map(_strip_prefix_num).to_numpy(np.float64)
    ys = gdf["y"].map(_strip_prefix_num).to_numpy(np.float64)
    ds = gdf["deg"].map(_strip_prefix_num).to_numpy(np.float64)

    # score via min/max of vertices (fast)
    minx = miny = 1e100
    maxx = maxy = -1e100
    for x, y, d in zip(xs, ys, ds):
        r = np.deg2rad(d)
        c, s = np.cos(r), np.sin(r)
        vx = TREE_PTS[:, 0] * c - TREE_PTS[:, 1] * s + x
        vy = TREE_PTS[:, 0] * s + TREE_PTS[:, 1] * c + y
        minx = min(minx, float(vx.min()))
        maxx = max(maxx, float(vx.max()))
        miny = min(miny, float(vy.min()))
        maxy = max(maxy, float(vy.max()))
    side = max(maxx - minx, maxy - miny)
    contrib = (side * side) / float(n)

    # overlap check (slower) - only on groups we actually evaluate
    polys = build_polys(xs, ys, ds)
    if has_overlap_polys(polys):
        return float("inf")

    return contrib

def init_cache(df: pd.DataFrame):
    """
    Build:
    - sigs[group_id]
    - contrib[group_id]
    - total score
    """
    sigs = {}
    contrib = {}
    total = 0.0
    for n in range(1, 201):
        gid = f"{n:03d}"
        gdf = df[df["group_id"] == gid]
        sigs[gid] = group_signature(gdf)
        c = eval_group(gdf, n)
        contrib[gid] = c
        total += c
    return sigs, contrib, total

def propose_bbox3(current_sub_path: str):
    """
    Run bbox3 on a temp copy of current submission.
    Return new_df (after bbox3) and the (n,r) used.
    """
    if random.random() < P_WILD:
        n = random.randint(*N_WILD)
        r = random.randint(*R_WILD)
    else:
        n = random.choice(N_GRID)
        r = random.choice(R_GRID)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_sub = os.path.join(tmp, "submission.csv")
        tmp_bbox = os.path.join(tmp, "bbox3")
        shutil.copy(current_sub_path, tmp_sub)
        shutil.copy(BBOX, tmp_bbox)
        os.chmod(tmp_bbox, 0o755)

        # run bbox3 (silence output)
        import subprocess
        subprocess.run(
            [tmp_bbox, "-n", str(n), "-r", str(r)],
            cwd=tmp,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=BBOX_TIMEOUT_SEC
        )

        new_df = load_df(tmp_sub)

    return new_df, n, r

def save_df(df: pd.DataFrame, path: str):
    df[["id","x","y","deg"]].to_csv(path, index=False)

# ------------------------------
# SA main
# ------------------------------
assert os.path.exists(WORK_SUB), f"Missing {WORK_SUB}"
assert os.path.exists(BBOX), f"Missing {BBOX}"

df = load_df(WORK_SUB)
sigs, contrib, current_score = init_cache(df)


df2 = load_df(WORK_SUB2)
sigs2, contrib2, score2 = init_cache(df2)

if score2 < current_score:
    df = df2

best_df = df.copy()
best_score = current_score
save_df(best_df, BEST_PATH)

T = T0
start = datetime.now()
deadline = start + timedelta(hours=MAX_HOURS)

with open(LOG_PATH, "a", encoding="utf-8") as f:
    f.write(f"\n==== SA START {start} | MAX_HOURS={MAX_HOURS} ====\n")
    f.write(f"Initial score: {current_score:.12f}\n")

print(f"🔥 SA start: score={current_score:.12f} | best={best_score:.12f}")

iters = 0
while iters < MAX_ITERS and datetime.now() < deadline:
    iters += 1

    # propose
    try:
        cand_df, n_param, r_param = propose_bbox3(WORK_SUB)
    except Exception as e:
        print(f"⚠️ bbox3 failed/timeout at iter {iters}: {e}")
        continue

    # detect changed groups
    changed = []
    for n in range(1, 201):
        gid = f"{n:03d}"
        gdf = cand_df[cand_df["group_id"] == gid]
        sig = group_signature(gdf)
        if sig != sigs[gid]:
            changed.append(gid)

    # if nothing changed, skip quickly
    if not changed:
        T = max(T * ALPHA, T_MIN)
        continue

    # compute candidate score using delta update
    cand_score = current_score
    cand_contrib_updates = {}
    invalid = False
    for gid in changed:
        n = int(gid)
        gdf = cand_df[cand_df["group_id"] == gid]
        new_c = eval_group(gdf, n)
        if not np.isfinite(new_c):
            invalid = True
            break
        cand_score = cand_score - contrib[gid] + new_c
        cand_contrib_updates[gid] = new_c

    if invalid:
        T = max(T * ALPHA, T_MIN)
        continue

    delta = cand_score - current_score

    # SA accept rule
    accept = (delta < 0) or (random.random() < math.exp(-delta / max(T, 1e-30)))
    if accept:
        # update working submission
        df = cand_df
        save_df(df, WORK_SUB)

        # update caches
        for gid in changed:
            gdf = df[df["group_id"] == gid]
            sigs[gid] = group_signature(gdf)
            contrib[gid] = cand_contrib_updates[gid]
        current_score = cand_score

        # update best
        if current_score < best_score:
            best_score = current_score
            best_df = df.copy()
            save_df(best_df, BEST_PATH)
            print(f"🏆 iter={iters} NEW BEST: {best_score:.12f}  (n={n_param}, r={r_param}, changed={len(changed)})")

    # cool down
    T = max(T * ALPHA, T_MIN)

    # logging / periodic save
    if iters % SAVE_EVERY == 0:
        msg = (f"iter={iters} T={T:.2e} current={current_score:.12f} best={best_score:.12f} "
               f"(last n={n_param}, r={r_param}, changed={len(changed)}, accept={accept})")
        print(msg)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

end = datetime.now()
print("\n✅ SA finished")
print(f"Time: {start} -> {end}")
print(f"Final current: {current_score:.12f}")
print(f"Best saved to: {BEST_PATH}  score={best_score:.12f}")
