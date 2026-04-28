"""
数据预处理器 v3（合并版）：
1. data_org  → Task A k-NN 形状库（保存至 processed_data_a）
2. data_org  → Task B 双精度 OD 耗时矩阵（保存至 processed_data_b）
3. data_ds15 → Task B 43 维训练特征矩阵（保存至 processed_data_b）

OD 矩阵改进：
  - 同时构建 precision=6（精细，~1.2km）和 precision=5（粗粒度，~4.9km）两级 OD
  - 每个 OD 格子保存：均值、中位数、标准差、样本数
  - 推理时：精细 OD 命中 → 用精细；否则 fallback 到粗粒度；再否则用全局均值
"""
import os
import pickle
import numpy as np
from features_and_utils import extract_task_b_features_advanced, get_grid_id, lookup_od_time

# ── 配置 ──────────────────────────────────────────────────────────────────────
ORG_TRAIN_FILE  = os.path.join("data_org",  "train.pkl")
DS15_TRAIN_FILE = os.path.join("data_ds15", "train.pkl")
PROCESSED_DIR_A = "processed_data_a"   # Task A 输出目录
PROCESSED_DIR_B = "processed_data_b"   # Task B 输出目录
BATCH_SIZE      = 20000
# ─────────────────────────────────────────────────────────────────────────────


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _build_od_dict(coords_list, travel_times, precision):
    """从轨迹列表构建指定精度的 OD 统计字典"""
    od_raw = {}
    for coords, tt in zip(coords_list, travel_times):
        g_o = get_grid_id(coords[0][0],  coords[0][1],  precision=precision)
        g_d = get_grid_id(coords[-1][0], coords[-1][1], precision=precision)
        key = f"{g_o}_{g_d}"
        od_raw.setdefault(key, []).append(tt)

    od_stats = {}
    for key, times in od_raw.items():
        arr = np.array(times)
        od_stats[key] = {
            "mean":   float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std":    float(np.std(arr)),
            "count":  len(arr),
        }
    return od_stats


def build_knowledge_from_org():
    """
    阶段 1：从高频 data_org 提取先验知识。
    - Task A：k-NN 形状库 → processed_data_a/knn_db.pkl
    - Task B：双精度 OD 矩阵 → processed_data_b/od_matrix_org.pkl
    """
    print(f"[*] 正在读取高频原始数据 {ORG_TRAIN_FILE}...")
    if not os.path.exists(ORG_TRAIN_FILE):
        print("    [警告] 未找到 data_org/train.pkl！")
        return {}

    with open(ORG_TRAIN_FILE, "rb") as f:
        org_data = pickle.load(f)
    print(f"    共 {len(org_data)} 条轨迹。")

    knn_db = {}
    MAX_PER_KEY = 3
    coords_list, travel_times = [], []

    print("[*] 构建 Task A k-NN 形状库 + 收集 Task B OD 样本...")
    for idx, traj in enumerate(org_data):
        if idx % 20000 == 0:
            print(f"    -> 进度: {idx}/{len(org_data)}")

        coords     = traj["coords"]
        timestamps = traj["timestamps"]
        n = len(coords)
        if n < 10:
            continue

        # ── Task B：收集 OD 样本（过滤异常行程）────────────────
        travel_time = timestamps[-1] - timestamps[0]
        if 60 < travel_time <= 7200:
            coords_list.append(coords)
            travel_times.append(travel_time)

        # ── Task A：k-NN 形状库 ──────────────────────────────────
        if n >= 30:
            avg_dt  = (timestamps[-1] - timestamps[0]) / (n - 1) if n > 1 else 3.0
            ds_step = max(1, round(15.0 / avg_dt))
            ds_indices = list(range(0, n, ds_step))

            for gap in [8, 16]:
                for i in range(0, len(ds_indices) - gap, gap):
                    idx_start = ds_indices[i]
                    idx_end   = ds_indices[min(i + gap, len(ds_indices) - 1)]
                    if idx_end >= n:
                        break
                    g_o = get_grid_id(coords[idx_start][0], coords[idx_start][1])
                    g_d = get_grid_id(coords[idx_end][0],   coords[idx_end][1])
                    if g_o == g_d:
                        continue
                    key = (g_o, g_d)
                    if key not in knn_db:
                        knn_db[key] = []
                    if len(knn_db[key]) < MAX_PER_KEY:
                        segment = coords[idx_start:idx_end + 1]
                        if len(segment) >= 2:
                            knn_db[key].append(segment)

    # 保存 Task A k-NN 库
    ensure_dir(PROCESSED_DIR_A)
    knn_path = os.path.join(PROCESSED_DIR_A, "knn_db.pkl")
    save_pickle(knn_db, knn_path)
    print(f"    -> [完成] Task A k-NN 库：{len(knn_db)} 种 OD 对 → {knn_path}")

    # 构建双精度 OD 矩阵
    print(f"[*] 构建双精度 OD 耗时矩阵（共 {len(coords_list)} 条有效轨迹）...")
    od_fine   = _build_od_dict(coords_list, travel_times, precision=6)
    od_coarse = _build_od_dict(coords_list, travel_times, precision=5)

    all_times  = np.array(travel_times)
    global_avg = float(np.mean(all_times)) if len(all_times) > 0 else 1200.0
    global_med = float(np.median(all_times)) if len(all_times) > 0 else 1200.0

    knowledge_base = {
        "od_fine":    od_fine,
        "od_coarse":  od_coarse,
        "global_avg": global_avg,
        "global_med": global_med,
    }

    ensure_dir(PROCESSED_DIR_B)
    od_path_b = os.path.join(PROCESSED_DIR_B, "od_matrix_org.pkl")
    save_pickle(knowledge_base, od_path_b)
    print(f"    -> [完成] Task B OD 矩阵：精细 {len(od_fine)} 对 / 粗粒度 {len(od_coarse)} 对 → {od_path_b}")

    # 同时保存兼容旧版 od_avg 格式（供 task_b_main.py 等旧脚本使用）
    compat = {
        "od_avg":     {k: v["mean"] for k, v in od_fine.items()},
        "global_avg": global_avg,
    }
    save_pickle(compat, os.path.join(PROCESSED_DIR_A, "od_matrix.pkl"))

    return knowledge_base


def build_training_features_from_ds15(knowledge_base):
    """
    阶段 2：从 data_ds15 提取 Task B 训练特征（43 维），保存至 processed_data_b。
    同时保存分批格式（供旧版 task_b_main.py 使用）至 processed_data_a。
    """
    print(f"\n[*] 正在读取降采样训练数据 {DS15_TRAIN_FILE}...")
    if not os.path.exists(DS15_TRAIN_FILE):
        print("    [警告] 未找到 data_ds15/train.pkl！")
        return

    with open(DS15_TRAIN_FILE, "rb") as f:
        ds15_data = pickle.load(f)
    print(f"    共 {len(ds15_data)} 条轨迹。")

    X_train, y_train = [], []
    skipped = 0

    print("[*] 提取增强特征（39维基础 + 4维OD统计 = 43维）...")
    for idx, traj in enumerate(ds15_data):
        if idx % 20000 == 0:
            print(f"    -> 进度: {idx}/{len(ds15_data)}")

        coords      = traj["coords"]
        timestamps  = traj["timestamps"]
        travel_time = timestamps[-1] - timestamps[0]

        if len(coords) < 2 or travel_time <= 60 or travel_time > 7200:
            skipped += 1
            continue

        try:
            feat, (g_o_fine, g_d_fine), (g_o_coarse, g_d_coarse) = \
                extract_task_b_features_advanced(coords, timestamps[0])

            od_mean, od_med, od_std, od_cnt = lookup_od_time(
                knowledge_base, g_o_fine, g_d_fine, g_o_coarse, g_d_coarse)

            X_train.append(feat + [od_mean, od_med, od_std, float(od_cnt)])
            y_train.append(travel_time)
        except Exception:
            skipped += 1
            continue

    if not X_train:
        print("    [警告] 未生成任何训练样本。")
        return

    print(f"    -> 有效样本: {len(X_train)}，跳过: {skipped}")

    X_arr = np.array(X_train, dtype=np.float32)
    y_arr = np.array(y_train, dtype=np.float32)

    ensure_dir(PROCESSED_DIR_B)
    np.save(os.path.join(PROCESSED_DIR_B, "X_train_final.npy"), X_arr)
    np.save(os.path.join(PROCESSED_DIR_B, "y_train_final.npy"), y_arr)
    print(f"    -> [完成] Task B 新版训练数据 → {PROCESSED_DIR_B}  (特征维度: {X_arr.shape[1]})")

    # 同时保存分批格式至 processed_data_a（供旧版脚本使用）
    ensure_dir(PROCESSED_DIR_A)
    batch_idx = 0
    for start in range(0, len(X_arr), BATCH_SIZE):
        np.save(os.path.join(PROCESSED_DIR_A, f"X_batch_{batch_idx}.npy"),
                X_arr[start:start + BATCH_SIZE])
        np.save(os.path.join(PROCESSED_DIR_A, f"y_batch_{batch_idx}.npy"),
                y_arr[start:start + BATCH_SIZE])
        batch_idx += 1
    print(f"    -> [完成] Task B 分批数据 → {PROCESSED_DIR_A}  ({batch_idx} 个 batch)")


def main():
    print("=" * 60)
    print("数据处理管道 v3 — 双精度 OD + 增强特征（43维）")
    print(f"输出目录：Task A → {PROCESSED_DIR_A}  |  Task B → {PROCESSED_DIR_B}")
    print("=" * 60)

    kb = build_knowledge_from_org()
    if kb:
        build_training_features_from_ds15(kb)

    print("\n[✓] 数据预处理完毕！")
    print(f"    下一步：python task_b_main_new.py")


if __name__ == "__main__":
    main()
