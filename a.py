import os
import time
import math
import random
import shutil
import hashlib
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
BBOX3_PATH = "./bbox3"
SGO_PATH   = "./single_group_optimizer"   # single_group_optimizer
WORK_SUB   = "submission.csv"

# ---- Runtime control ----
MAX_HOURS = 6.0
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---- SA control ----
MAX_ITERS = 10_000_000  # 实际会被时间上限截断
T0 = 1e-4
T_MIN = 1e-7
ALPHA = 0.995

# 若连续这么多次没有 BEST 改进 -> reheating
NO_BEST_IMPROVE_REHEAT = 200

# 若连续这么多次 accept 很少/无提升 -> restart current = best
NO_PROGRESS_RESTART = 500

# ---- bbox3 parameter sampling ----
# 你之前用的是 n in [1000..2000 step100], r in [10..90 step10]
# 这里给更灵活的采样（同时保留“历史有效区域”）
N_CAND = list(range(800, 2401, 100))
R_CAND = list(range(10, 151, 10))
WILD_PROB = 0.10  # 10% 完全随机扰动

# ---- local refinement ----
ENABLE_ROT_SQUEEZE = True   # 旋转压缩（不动相对布局，只整体旋转）
ENABLE_SGO = True           # 单组精修
SGO_ITERS = 80_000          # 单组优化迭代数（建议 50k~200k）
SGO_RESTARTS = 60           # 单组优化重启数（建议 20~120）
SGO_TOPK_BY_CONTRIB = 20    # 每轮最多精修多少个组（按贡献从大到小挑）

# ---- Elite pool + merge ----
POOL_DIR = "elite_pool"
POOL_KEEP = 50              # 最多保留多少个候选csv
POOL_ADD_EVERY = 30         # 每多少次迭代，把当前(若被接受)加入 pool
MERGE_EVERY = 200           # 每多少次迭代，做一次“按组择优合并”
MERGE_APPLY_IF_BETTER = True

# ---- optional validation (慢) ----
VALIDATE_OVERLAP = False  # True 会很慢（需要 shapely）；默认 False


# =========================
# Utilities
# =========================
TREE_PTS = np.array([
    (0.0, 0.8),
    (0.125, 0.5), (0.0625, 0.5),
    (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0),
    (0.075, 0.0),
    (0.075, -0.2),
    (-0.075, -0.2),
    (-0.075, 0.0),
    (-0.35, 0.0),
    (-0.1, 0.25), (-0.2, 0.25),
    (-0.0625, 0.5), (-0.125, 0.5)
], dtype=np.float64)

def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def now_s() -> float:
    return time.time()

def fmt_sec(sec: float) -> str:
    sec = max(0.0, sec)
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"

def ensure_exec(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing executable: {path}")
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass

def parse_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 格式通常是 "s-1.234"
    df["x_f"] = df["x"].astype(str).str[1:].astype(np.float64)
    df["y_f"] = df["y"].astype(str).str[1:].astype(np.float64)
    df["deg_f"] = df["deg"].astype(str).str[1:].astype(np.float64)
    df["group_id"] = df["id"].astype(str).str.split("_", n=1, expand=True)[0]
    df["item_id"] = df["id"].astype(str).str.split("_", n=1, expand=True)[1].astype(int)
    return df

def group_ids() -> List[str]:
    return [f"{i:03d}" for i in range(1, 201)]

def build_group_arrays(df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    return:
      gid -> (item_id, x, y, deg) all float64 / int
    """
    out = {}
    for gid, g in df.groupby("group_id"):
        g = g.sort_values("item_id")
        out[gid] = (
            g["item_id"].to_numpy(np.int32),
            g["x_f"].to_numpy(np.float64),
            g["y_f"].to_numpy(np.float64),
            g["deg_f"].to_numpy(np.float64),
        )
    return out

def render_group_points(x: np.ndarray, y: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """
    Return all polygon vertices points (Ntrees*15, 2)
    """
    # rot matrices per tree
    r = np.deg2rad(deg)
    c = np.cos(r)
    s = np.sin(r)

    # TREE_PTS: (15,2)
    px = TREE_PTS[:, 0][None, :]
    py = TREE_PTS[:, 1][None, :]

    # rotate
    xs = (px * c[:, None] - py * s[:, None]) + x[:, None]
    ys = (px * s[:, None] + py * c[:, None]) + y[:, None]

    pts = np.stack([xs, ys], axis=-1).reshape(-1, 2)
    return pts

def group_side_and_contrib(gid: str, arrs) -> Tuple[float, float]:
    """
    returns side, contrib = side^2 / n
    """
    _, x, y, deg = arrs
    pts = render_group_points(x, y, deg)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    side = float(max(mx[0] - mn[0], mx[1] - mn[1]))
    n = int(gid)
    contrib = (side * side) / n
    return side, contrib

def compute_all_contrib(groups: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[Dict[str, float], float]:
    contrib = {}
    total = 0.0
    for gid in group_ids():
        _, c = group_side_and_contrib(gid, groups[gid])
        contrib[gid] = c
        total += c
    return contrib, total

def write_submission_from_groups(template_df: pd.DataFrame,
                                groups: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                                out_path: str):
    """
    template_df：用于保留 id 顺序等
    """
    df = template_df.copy()
    # 更新 x,y,deg 字符串
    # 用 map 快速写回
    for gid in group_ids():
        item_id, x, y, deg = groups[gid]
        mask = (df["group_id"] == gid)
        # 假设 item_id 完整覆盖并且排序一致
        # 保险：按 item_id 对齐
        gidx = df.loc[mask].sort_values("item_id").index.to_numpy()
        df.loc[gidx, "x"] = ["s" + f"{v:.12f}" for v in x]
        df.loc[gidx, "y"] = ["s" + f"{v:.12f}" for v in y]
        df.loc[gidx, "deg"] = ["s" + f"{v:.12f}" for v in deg]

    df[["id", "x", "y", "deg"]].to_csv(out_path, index=False)

def diff_changed_groups(df_before: pd.DataFrame, df_after: pd.DataFrame) -> List[str]:
    """
    找出 bbox3 运行后哪些 group 的 (x,y,deg) 有变化
    """
    # 只比较原始字符串，最快
    a = df_before[["id","x","y","deg"]].copy()
    b = df_after[["id","x","y","deg"]].copy()
    # 对齐
    a = a.sort_values("id")
    b = b.sort_values("id")
    changed = (a["x"].to_numpy() != b["x"].to_numpy()) | \
              (a["y"].to_numpy() != b["y"].to_numpy()) | \
              (a["deg"].to_numpy() != b["deg"].to_numpy())
    if not changed.any():
        return []
    gids = a.loc[changed, "id"].str.split("_", n=1, expand=True)[0].unique().tolist()
    return sorted(gids)

# =========================
# Rotation squeeze (整体旋转最小化外接正方形边长)
# =========================
def bbox_side_at_angle(angle_deg: float, pts: np.ndarray) -> float:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    # rotate by -angle (等价用转置)
    x = pts[:,0]*c + pts[:,1]*s
    y = -pts[:,0]*s + pts[:,1]*c
    side = max(x.max()-x.min(), y.max()-y.min())
    return float(side)

def rotate_group_centers(x, y, deg, angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    围绕 group 的 bbox 中心旋转所有 tree center，同时 tree 自身角度加上 angle
    """
    pts = render_group_points(x, y, deg)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = (mn + mx) / 2.0  # (2,)

    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s,  c]], dtype=np.float64)

    centers = np.stack([x, y], axis=1)
    shifted = centers - center[None, :]
    rotated = shifted @ R.T + center[None, :]

    x2 = rotated[:,0]
    y2 = rotated[:,1]
    deg2 = deg + angle_deg
    return x2, y2, deg2

def golden_search_min_angle(pts: np.ndarray, lo=0.0, hi=90.0, iters=40) -> Tuple[float, float]:
    """
    在 [lo, hi] 找使 bbox_side 最小的角度（黄金分割搜索）
    """
    gr = (math.sqrt(5) - 1) / 2
    c = hi - gr*(hi-lo)
    d = lo + gr*(hi-lo)
    fc = bbox_side_at_angle(c, pts)
    fd = bbox_side_at_angle(d, pts)
    for _ in range(iters):
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - gr*(hi-lo)
            fc = bbox_side_at_angle(c, pts)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + gr*(hi-lo)
            fd = bbox_side_at_angle(d, pts)
    best_a = (lo + hi) / 2.0
    best_s = bbox_side_at_angle(best_a, pts)
    return best_a, best_s

def rotation_squeeze_group(arrs, eps=1e-9) -> Tuple[bool, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    返回是否改进，以及新 arrs
    """
    item_id, x, y, deg = arrs
    pts = render_group_points(x, y, deg)
    s0 = bbox_side_at_angle(0.0, pts)
    a_best, s_best = golden_search_min_angle(pts, lo=0.0, hi=90.0, iters=40)

    if (s0 - s_best) > eps:
        x2, y2, deg2 = rotate_group_centers(x, y, deg, a_best)
        return True, (item_id, x2, y2, deg2)
    return False, arrs

# =========================
# External executables
# =========================
def run_bbox3(n_param: int, r_param: int, timeout_sec=1800) -> bool:
    """
    bbox3 会读写 WORK_SUB（submission.csv）
    """
    cmd = [BBOX3_PATH, "-n", str(n_param), "-r", str(r_param)]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=timeout_sec)
        # 你需要可视输出可改为 print(res.stdout)
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        return False

def run_single_group_optimizer(group_n: int, iters: int, restarts: int, out_path: str, timeout_sec=1800) -> bool:
    """
    兼容你展示的用法：
      GROUP_NUMBER=196 ./single_group_optimizer -n 444888 -r 333 -o submission.csv
    """
    env = os.environ.copy()
    env["GROUP_NUMBER"] = str(group_n)

    cmd = [SGO_PATH, "-n", str(iters), "-r", str(restarts), "-o", out_path]
    try:
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=timeout_sec)
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        return False

# =========================
# Elite pool
# =========================
def pool_init():
    os.makedirs(POOL_DIR, exist_ok=True)

def pool_list() -> List[str]:
    if not os.path.exists(POOL_DIR):
        return []
    files = [os.path.join(POOL_DIR, f) for f in os.listdir(POOL_DIR) if f.endswith(".csv")]
    files.sort()
    return files

def pool_add(csv_path: str, score: float):
    pool_init()
    tag = f"{score:.12f}".replace(".", "_")
    dst = os.path.join(POOL_DIR, f"elite_{tag}_{int(time.time())}.csv")
    shutil.copy(csv_path, dst)

    files = pool_list()
    # 保留 score 最小的前 POOL_KEEP 个
    scored = []
    for f in files:
        # 从文件名里粗略解析 score（解析失败就放后面）
        base = os.path.basename(f)
        try:
            s = float(base.split("_")[1] + "." + base.split("_")[2])
        except Exception:
            s = 1e18
        scored.append((s, f))
    scored.sort(key=lambda x: x[0])
    for _, f in scored[POOL_KEEP:]:
        try:
            os.remove(f)
        except Exception:
            pass

def merge_best_from_pool(base_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    从 pool 中，对每个 group 选择贡献最小的那份拼起来
    返回 merged_df 和 merged_total_score
    """
    candidates = pool_list()
    if not candidates:
        return None, None

    # 预先读取所有候选
    dfs = []
    for p in candidates:
        try:
            df = parse_submission(p)
            dfs.append((p, df))
        except Exception:
            pass
    if not dfs:
        return None, None

    # 对每个 group 选 best
    chosen_rows = []
    total = 0.0

    for gid in group_ids():
        best_c = 1e18
        best_gdf = None
        for _, df in dfs:
            gdf = df[df["group_id"] == gid].sort_values("item_id")
            item_id = gdf["item_id"].to_numpy(np.int32)
            x = gdf["x_f"].to_numpy(np.float64)
            y = gdf["y_f"].to_numpy(np.float64)
            deg = gdf["deg_f"].to_numpy(np.float64)
            side, contrib = group_side_and_contrib(gid, (item_id, x, y, deg))
            if contrib < best_c:
                best_c = contrib
                best_gdf = gdf[["id","x","y","deg"]].copy()
        chosen_rows.append(best_gdf)
        total += best_c

    merged = pd.concat(chosen_rows, ignore_index=True)
    # 按 base 的顺序排序（稳定）
    merged = merged.set_index("id").loc[base_df["id"].values].reset_index()
    return merged, total

# =========================
# Main SA loop
# =========================
@dataclass
class State:
    df: pd.DataFrame
    groups: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    contrib: Dict[str, float]
    total: float

def pick_bbox_params(stats_good: List[Tuple[int,int]]) -> Tuple[int,int]:
    # 90%：从历史成功组合里采样，10%：wild
    if stats_good and random.random() > WILD_PROB:
        return random.choice(stats_good)
    return (random.choice(N_CAND), random.choice(R_CAND))

def main():
    ensure_exec(BBOX3_PATH)
    ensure_exec(SGO_PATH)
    if not os.path.exists(WORK_SUB):
        raise FileNotFoundError(f"Missing {WORK_SUB} in current directory")

    start_t = now_s()
    deadline = start_t + MAX_HOURS * 3600

    # Load initial
    df0 = parse_submission(WORK_SUB)
    groups0 = build_group_arrays(df0)
    contrib0, total0 = compute_all_contrib(groups0)

    state = State(df=df0, groups=groups0, contrib=contrib0, total=total0)
    best_total = state.total
    best_path = "best_submission.csv"
    shutil.copy(WORK_SUB, best_path)

    print(f"Initial score: {state.total:.12f}")

    T = T0
    good_params = []  # 记录带来 best 改进或被 accept 的参数

    no_best = 0
    no_progress = 0

    pool_init()
    pool_add(WORK_SUB, state.total)

    it = 0
    while it < MAX_ITERS and now_s() < deadline:
        it += 1

        # ---- propose via bbox3 ----
        n_param, r_param = pick_bbox_params(good_params)
        before_df = parse_submission(WORK_SUB)
        before_hash = md5(WORK_SUB)

        ok = run_bbox3(n_param, r_param, timeout_sec=1800)
        if not ok:
            # 失败就跳过
            continue

        after_hash = md5(WORK_SUB)
        if after_hash == before_hash:
            # bbox3 没改东西
            no_progress += 1
            no_best += 1
            # 退火
            T = max(T * ALPHA, T_MIN)
            if no_best >= NO_BEST_IMPROVE_REHEAT:
                T = T0
                no_best = 0
            if no_progress >= NO_PROGRESS_RESTART:
                shutil.copy(best_path, WORK_SUB)
                no_progress = 0
            continue

        after_df = parse_submission(WORK_SUB)
        changed = diff_changed_groups(before_df, after_df)
        if not changed:
            # hash 变了但对齐后看不出来（极少）
            changed = []

        # 把 after_df -> groups
        prop_groups = build_group_arrays(after_df)

        # ---- deterministic post-fix: rotation squeeze only changed groups ----
        if ENABLE_ROT_SQUEEZE and changed:
            for gid in changed:
                improved, new_arrs = rotation_squeeze_group(prop_groups[gid])
                if improved:
                    prop_groups[gid] = new_arrs

        # ---- local refine: single_group_optimizer on top contributors among changed ----
        # 先估计 changed 的贡献，用来挑 TopK
        if ENABLE_SGO and changed:
            # 用 prop_groups 计算改动后的 contrib
            tmp_contrib = []
            for gid in changed:
                _, c = group_side_and_contrib(gid, prop_groups[gid])
                tmp_contrib.append((c, gid))
            tmp_contrib.sort(reverse=True)  # contrib 大的优先（对总分影响大）
            refine_gids = [gid for _, gid in tmp_contrib[:SGO_TOPK_BY_CONTRIB]]

            # 写一个临时 submission 给 SGO 用
            # 注意：SGO 读写 submission.csv，所以我们先把当前 prop 写回 WORK_SUB
            # 用 state.df 作为模板（含列），但我们要更新 group_id/item_id 等字段一致
            base_template = after_df.copy()
            write_submission_from_groups(base_template, prop_groups, WORK_SUB)

            for gid in refine_gids:
                run_single_group_optimizer(int(gid), SGO_ITERS, SGO_RESTARTS, WORK_SUB, timeout_sec=1800)

            # SGO 后再读取一次
            after_df2 = parse_submission(WORK_SUB)
            prop_groups = build_group_arrays(after_df2)
            # SGO 可能改变更多 group（理论上只改变 GROUP_NUMBER），这里保险再 diff 一次
            changed2 = diff_changed_groups(after_df, after_df2)
            if changed2:
                changed = sorted(set(changed) | set(changed2))

        # ---- compute new total quickly: only recompute changed groups ----
        new_total = state.total
        new_contrib = dict(state.contrib)

        for gid in changed:
            _, c_new = group_side_and_contrib(gid, prop_groups[gid])
            new_total += (c_new - state.contrib[gid])
            new_contrib[gid] = c_new

        delta = new_total - state.total

        # ---- SA accept/reject ----
        accept = (delta < 0.0) or (random.random() < math.exp(-delta / max(T, 1e-12)))

        if accept:
            # update state
            state.df = parse_submission(WORK_SUB)  # 已经在 WORK_SUB 里
            state.groups = prop_groups
            state.contrib = new_contrib
            state.total = new_total
            good_params.append((n_param, r_param))
            no_progress = 0

            if new_total < best_total - 1e-15:
                best_total = new_total
                shutil.copy(WORK_SUB, best_path)
                print(f"[{it}] 🏆 NEW BEST: {best_total:.12f}  (n={n_param}, r={r_param}, |changed|={len(changed)})")
                no_best = 0
            else:
                no_best += 1

            # pool add
            if it % POOL_ADD_EVERY == 0:
                pool_add(WORK_SUB, state.total)

        else:
            # reject -> rollback file to best current (state)
            # 注意 state 代表“当前解”，不是 best 解；我们需要回滚到 state 的文件
            # 最简单：写回 state.groups
            write_submission_from_groups(state.df, state.groups, WORK_SUB)
            no_progress += 1
            no_best += 1

        # ---- merge from pool periodically ----
        if it % MERGE_EVERY == 0:
            merged_df, merged_total = merge_best_from_pool(state.df)
            if merged_df is not None:
                if (not MERGE_APPLY_IF_BETTER) or (merged_total < state.total - 1e-15):
                    merged_df.to_csv(WORK_SUB, index=False)
                    # reload merged as new state
                    mdf = parse_submission(WORK_SUB)
                    mgroups = build_group_arrays(mdf)
                    mcontrib, mtotal = compute_all_contrib(mgroups)
                    state = State(df=mdf, groups=mgroups, contrib=mcontrib, total=mtotal)

                    if mtotal < best_total - 1e-15:
                        best_total = mtotal
                        shutil.copy(WORK_SUB, best_path)
                        print(f"[{it}] 🧩 MERGE NEW BEST: {best_total:.12f}")
                    else:
                        print(f"[{it}] 🧩 MERGE applied: {mtotal:.12f}")

        # ---- anneal / reheating / restart ----
        T = max(T * ALPHA, T_MIN)

        if no_best >= NO_BEST_IMPROVE_REHEAT:
            # 重热：提高接受劣解概率，跳出局部最优
            T = T0
            no_best = 0
            print(f"[{it}] 🔥 Reheat: T -> {T:.2e}")

        if no_progress >= NO_PROGRESS_RESTART:
            # 重启：把 current 拉回 best，避免在坏区域浪费时间
            shutil.copy(best_path, WORK_SUB)
            state.df = parse_submission(WORK_SUB)
            state.groups = build_group_arrays(state.df)
            state.contrib, state.total = compute_all_contrib(state.groups)
            no_progress = 0
            print(f"[{it}] ♻️ Restart current = best ({best_total:.12f})")

        # ---- print occasionally ----
        if it % 20 == 0:
            left = deadline - now_s()
            print(f"[{it}] cur={state.total:.12f} best={best_total:.12f} "
                  f"T={T:.2e} left={fmt_sec(left)}")

    # Finish
    shutil.copy(best_path, WORK_SUB)
    print("\nDONE")
    print(f"Best score: {best_total:.12f}")
    print(f"Saved to: {WORK_SUB} (also {best_path})")
    print(f"Pool dir: {POOL_DIR}")

if __name__ == "__main__":
    main()
