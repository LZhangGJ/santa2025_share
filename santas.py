DEBUG = False

MAX_HOURS = 7.7
from shutil import copy

#'bbox3'
#!chmod +x ./bbox3

import os
import time
import subprocess
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar

getcontext().prec = 30
scale_factor = 1


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor),
        )

    def clone(self) -> "ChristmasTree":
        return ChristmasTree(
            center_x=str(self.center_x),
            center_y=str(self.center_y),
            angle=str(self.angle),
        )


# -------------------------
# FAST bounds-based side length (no unary_union)
# -------------------------
def get_tree_list_side_lenght(tree_list: list[ChristmasTree]) -> Decimal:
    # bounds = (minx, miny, maxx, maxy)
    minx = float("inf")
    miny = float("inf")
    maxx = float("-inf")
    maxy = float("-inf")
    for t in tree_list:
        bx0, by0, bx1, by1 = t.polygon.bounds
        if bx0 < minx: minx = bx0
        if by0 < miny: miny = by0
        if bx1 > maxx: maxx = bx1
        if by1 > maxy: maxy = by1
    side = max(maxx - minx, maxy - miny)
    return Decimal(str(side)) / Decimal(str(scale_factor))


def get_total_score(dict_of_side_length: dict[str, Decimal]):
    score = Decimal("0")
    for k, v in dict_of_side_length.items():
        score += (v ** 2) / Decimal(k)
    return score


def parse_csv(csv_path):
    print(f'\nparse_csv: {csv_path=}')
    result = pd.read_csv(csv_path)

    # NOTE: 使用 strip('s') 会把两边所有 's' 都去掉；通常这里是前缀 's'
    # 这里用 lstrip 更安全也更快
    result['x'] = result['x'].astype(str).str.strip().str.lstrip('sS')
    result['y'] = result['y'].astype(str).str.strip().str.lstrip('sS')
    result['deg'] = result['deg'].astype(str).str.strip().str.lstrip('sS')
    result[['group_id', 'item_id']] = result['id'].astype(str).str.split('_', n=2, expand=True)

    dict_of_tree_list = {}
    dict_of_side_length = {}

    # groupby 用 sort=False 更快
    for group_id, group_data in result.groupby('group_id', sort=False):
        trees = []
        # itertuples 比 iterrows 快很多
        for row in group_data.itertuples(index=False):
            trees.append(ChristmasTree(center_x=row.x, center_y=row.y, angle=row.deg))
        dict_of_tree_list[group_id] = trees
        dict_of_side_length[group_id] = get_tree_list_side_lenght(trees)

    return dict_of_tree_list, dict_of_side_length


def calculate_bbox_side_at_angle(angle_deg, points):
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix_T = np.array([[c, s], [-s, c]])
    rotated_points = points.dot(rot_matrix_T)
    min_xy = np.min(rotated_points, axis=0)
    max_xy = np.max(rotated_points, axis=0)
    return max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])


def optimize_rotation(trees):
    # 收集所有顶点（float）
    all_points = []
    for tree in trees:
        all_points.extend(list(tree.polygon.exterior.coords))
    points_np = np.asarray(all_points, dtype=np.float64)

    # hull 降低点数
    hull_points = points_np[ConvexHull(points_np).vertices]

    initial_side = calculate_bbox_side_at_angle(0.0, hull_points)

    res = minimize_scalar(
        lambda a: calculate_bbox_side_at_angle(a, hull_points),
        bounds=(0.001, 89.999),
        method='bounded'
    )
    found_angle_deg = float(res.x)
    found_side = float(res.fun)

    improvement = initial_side - found_side
    EPSILON = 1e-8

    if improvement > EPSILON:
        best_angle_deg = found_angle_deg
        best_side = Decimal(str(found_side)) / Decimal(str(scale_factor))
    else:
        best_angle_deg = 0.0
        best_side = Decimal(str(initial_side)) / Decimal(str(scale_factor))

    return best_side, best_angle_deg


def apply_rotation(trees, angle_deg):
    if (not trees) or abs(angle_deg) < 1e-9:
        return [t.clone() for t in trees]

    bounds = [t.polygon.bounds for t in trees]
    min_x = min(b[0] for b in bounds)
    min_y = min(b[1] for b in bounds)
    max_x = max(b[2] for b in bounds)
    max_y = max(b[3] for b in bounds)
    rotation_center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=np.float64)

    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]], dtype=np.float64)

    points = np.array([[float(t.center_x), float(t.center_y)] for t in trees], dtype=np.float64)
    shifted = points - rotation_center
    rotated = shifted.dot(rot_matrix.T) + rotation_center

    rotated_trees = []
    add_angle = Decimal(str(angle_deg))
    for i in range(len(trees)):
        new_tree = ChristmasTree(
            Decimal(str(rotated[i, 0])),
            Decimal(str(rotated[i, 1])),
            Decimal(str(trees[i].angle + add_angle)),
        )
        rotated_trees.append(new_tree)
    return rotated_trees


def fix_direction(current_solution_path='submission.csv', out_file='submission.csv'):
    dict_of_tree_list, dict_of_side_length = parse_csv(current_solution_path)

    current_score = get_total_score(dict_of_side_length)
    print(f'{current_score=:0.12f}')

    # 逐组尝试找到一个更优旋转
    for group_id_main in range(200, 2, -1):
        group_id_main = f'{int(group_id_main):03n}'

        trees = dict_of_tree_list[group_id_main]
        best_side, best_angle_deg = optimize_rotation(trees)
        cur_side = dict_of_side_length[group_id_main]

        if best_side < cur_side:
            fixed_trees = apply_rotation(trees, best_angle_deg)
            print(f'n={int(group_id_main)}, {cur_side:0.8f} -> {best_side:0.8f} (Δ={cur_side - best_side:0.8f})')
            dict_of_tree_list[group_id_main] = fixed_trees
            dict_of_side_length[group_id_main] = best_side

    new_score = get_total_score(dict_of_side_length)
    diff_score = current_score - new_score
    print(f'    {new_score=:0.12f}\n'
          f'    {diff_score=:0.12f}\n')

    if diff_score > 0:
        print('Достигнут прогресс --> сохраняю результат')
        tree_data = []
        for group_name, tree_list in dict_of_tree_list.items():
            for item_id, tree in enumerate(tree_list):
                tree_data.append({
                    'id': f'{group_name}_{item_id}',
                    'x': f's{tree.center_x}',
                    'y': f's{tree.center_y}',
                    'deg': f's{tree.angle}'
                })
        pd.DataFrame(tree_data).to_csv(out_file, index=False)

    return current_score, new_score


def run_bbox_simple_with_timeout(debug=False):
    os.makedirs("bbox_sub", exist_ok=True)
    log_file = "bbox_experiments.log"

    start_time = datetime.now()
    timeout = timedelta(hours=MAX_HOURS)

    print(f"Начало экспериментов в: {start_time}")
    print(f"Таймаут через: {MAX_HOURS} часов (до {start_time + timeout})")

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Начало: {start_time}\n")
        f.write(f"Таймаут: {MAX_HOURS} часов\n")
        f.write('='*50 + '\n')

        n_min, n_max, n_step = 1000, 2000, 100
        r_min, r_max, r_step = 10, 90, 10
        n_values = list(range(n_min, n_max + 1, n_step))
        r_values = list(range(r_min, r_max + 1, r_step))
        total_runs = len(n_values) * len(r_values)

        print(f"Всего планируется запусков: {total_runs}")

        initial_score, final_score = fix_direction()

        completed_runs = 0
        for r_value in r_values:
            for n_value in n_values:
                current_time = datetime.now()
                elapsed = current_time - start_time
                if elapsed > timeout:
                    print(f"\n⏰ ВРЕМЯ ИСТЕКЛО! Прошло {elapsed}")
                    f.write(f"\n⏰ ВРЕМЯ ИСТЕКЛО! Прошло {elapsed}\n")
                    return

                completed_runs += 1
                progress = (completed_runs / total_runs) * 100.0
                # 粗略 ETA（可删，别影响速度）
                time_left = (timeout - elapsed) / max(completed_runs, 1) * (total_runs - completed_runs)

                print(f"[Прогресс: {progress:.1f}%] [Прошло: {elapsed}] [Осталось: ~{time_left}]")
                print(f"Итерация {completed_runs} - Параметры: n={n_value}, r={r_value}")

                f.write(f"\n[Время: {current_time}] [Прошло: {elapsed}]\n")
                f.write(f"Итерация {completed_runs} - Параметры: n={n_value}, r={r_value}\n")

                try:
                    result = subprocess.run(
                        ["./bbox3", "-n", str(n_value), "-r", str(r_value)],
                        capture_output=True,
                        text=True,
                        timeout=1200
                    )
                    if result.stdout:
                        print(result.stdout)
                        f.write(result.stdout + "\n")
                    if result.stderr:
                        print("Ошибки:", result.stderr)
                        f.write(f"Ошибки: {result.stderr}\n")

                except subprocess.TimeoutExpired:
                    msg = f"⚠ Таймаут команды (20 минут) для n={n_value}, r={r_value}, i={completed_runs}"
                    print(msg); f.write(msg + "\n")
                    continue
                except Exception as e:
                    msg = f"❌ Ошибка при запуске: {e}"
                    print(msg); f.write(msg + "\n")
                    continue

                # 保存 bbox3 输出的 submission.csv
                if os.path.exists("submission.csv"):
                    new_name = f"bbox_sub/submi-n{n_value}_r{r_value}_i{completed_runs}.csv"
                    try:
                        copy("submission.csv", new_name)
                        msg = f"✓ Сохранено: {new_name}"
                        print(msg); f.write(msg + "\n")
                    except Exception as e:
                        msg = f"❌ Ошибка при сохранении файла: {e}"
                        print(msg); f.write(msg + "\n")

                f.write("---\n")
                f.flush()

                # fix_direction（现在快很多）
                _, final_score = fix_direction()

                if debug:
                    break
            if debug:
                break

        end_time = datetime.now()
        total_elapsed = end_time - start_time
        print(f"\n{'='*50}")
        print(f"✅ ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
        print(f"Начало: {start_time}")
        print(f"Завершение: {end_time}")
        print(f"Общее время: {total_elapsed}")
        print(f"Выполнено запусков: {completed_runs} из {total_runs}")
        print(f"Начальная метрика: {initial_score:.12f}")
        print(f"Финальная метрика: {final_score:.12f}")
        print(f"Прирост метрики:    {initial_score - final_score:.12f}")
        print('='*50)

        f.write(f"\n{'='*50}\n")
        f.write(f"✅ ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!\n")
        f.write(f"Начало: {start_time}\n")
        f.write(f"Завершение: {end_time}\n")
        f.write(f"Общее время: {total_elapsed}\n")
        f.write(f"Выполнено запусков: {completed_runs} из {total_runs}\n")
        f.write(f"Начальная метрика: {initial_score:.12f}\n")
        f.write(f"Финальная метрика: {final_score:.12f}\n")
        f.write(f"Прирост метрики:   {initial_score - final_score:.12f}\n")
        f.write('='*50 + '\n')

    print("Эксперименты завершены!")


run_bbox_simple_with_timeout(debug=DEBUG)
