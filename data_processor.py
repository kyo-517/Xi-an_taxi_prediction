"""
数据预处理器：
1. 用 data_org（高密度~3s）构建精细 k-NN 轨迹形状库（Task A）
2. 用 data_ds15 构建 OD 耗时矩阵 + 特征批次（Task B）
"""
import pickle
import numpy as np
import os
from features_and_utils import extract_task_b_features_advanced, get_grid_id

# ================= 配置区 =================
TRAIN_DS15 = os.path.join("data_ds15", "train.pkl")
TRAIN_ORG  = os.path.join("data_org",  "train.pkl")
BATCH_SIZE = 20000
PROCESSED_DIR = "processed_data"
# ==========================================


def build_knn_db_from_org():
    """
    用 data_org（~3s 采样）构建高密度 k-NN 轨迹形状库。
    data_ds15 是 ~15s 采样，即 data_org 每 ~5 个点对应 data_ds15 的 1 个点。
    对于 1/8 缺失率：已知点间隔 = 8 * 15s = 120s，在 data_org 中约 40 个点。
    对于 1/16 缺失率：已知点间隔 = 16 * 15s = 240s，在 data_org 中约 80 个点。

    策略：从 data_org 中按 ds15 的间隔（每5个点取1个）模拟降采样，
    然后按 gap=8 和 gap=16 的间隔采集片段，存储中间的高密度轨迹形状。
    """
    if not os.path.exists(TRAIN_ORG):
        print(f"[警告] 未找到 {TRAIN_ORG}，跳过高密度 k-NN 构建。")
        return {}

    print("[*] 正在加载 data_org/train.pkl（高密度轨迹）...")
    with open(TRAIN_ORG, 'rb') as f:
        train_org = pickle.load(f)
    print(f"    共 {len(train_org)} 条轨迹。")

    knn_db = {}
    MAX_PER_KEY = 3  # 每种 OD 对最多保留 3 条

    print("[*] 从高密度数据构建 k-NN 形状库...")
    for idx, traj in enumerate(train_org):
        if idx % 20000 == 0:
            print(f"    -> 进度: {idx}/{len(train_org)}")

        coords = traj['coords']
        timestamps = traj['timestamps']
        n = len(coords)
        if n < 50:
            continue

        # 计算平均采样间隔
        avg_dt = (timestamps[-1] - timestamps[0]) / (n - 1) if n > 1 else 3.0
        # ds15 的降采样步长（约 15s / avg_dt）
        ds_step = max(1, round(15.0 / avg_dt))

        # 模拟 ds15 降采样后的索引
        ds_indices = list(range(0, n, ds_step))

        # 对 gap=8 和 gap=16 分别采集
        for gap in [8, 16]:
            for i in range(0, len(ds_indices) - gap, gap):
                idx_start = ds_indices[i]
                idx_end   = ds_indices[min(i + gap, len(ds_indices) - 1)]

                if idx_end >= n:
                    break

                grid_o = get_grid_id(coords[idx_start][0], coords[idx_start][1], precision=6)
                grid_d = get_grid_id(coords[idx_end][0],   coords[idx_end][1],   precision=6)
                key = (grid_o, grid_d)

                if key not in knn_db:
                    knn_db[key] = []
                if len(knn_db[key]) < MAX_PER_KEY:
                    # 存储从 idx_start 到 idx_end 的完整高密度轨迹段
                    segment = coords[idx_start:idx_end + 1]
                    if len(segment) >= 2:
                        knn_db[key].append(segment)

    print(f"[*] k-NN 形状库构建完成，共 {len(knn_db)} 种 OD 对。")
    return knn_db


def build_offline_databases():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # ============================================================
    # 阶段 1：用 data_org 构建高密度 k-NN 库
    # ============================================================
    knn_db = build_knn_db_from_org()
    with open(os.path.join(PROCESSED_DIR, "knn_db.pkl"), 'wb') as f:
        pickle.dump(knn_db, f)
    print(f"    -> k-NN 库已保存（{len(knn_db)} 种 OD 对）")

    # ============================================================
    # 阶段 2：用 data_ds15 构建 OD 耗时矩阵 + 特征批次（Task B）
    # ============================================================
    print(f"\n[*] 正在加载 {TRAIN_DS15}...")
    with open(TRAIN_DS15, 'rb') as f:
        train_raw = pickle.load(f)
    total_records = len(train_raw)
    print(f"    共 {total_records} 条轨迹。")

    od_dict = {}
    print("[*] 构建 OD 耗时矩阵...")
    for traj in train_raw:
        coords = traj['coords']
        timestamps = traj['timestamps']
        travel_time = timestamps[-1] - timestamps[0]
        if travel_time > 0 and len(coords) >= 2:
            grid_start = get_grid_id(coords[0][0],  coords[0][1],  precision=6)
            grid_end   = get_grid_id(coords[-1][0], coords[-1][1], precision=6)
            od_key = f"{grid_start}_{grid_end}"
            if od_key not in od_dict:
                od_dict[od_key] = []
            od_dict[od_key].append(travel_time)

    od_avg_time = {k: np.mean(v) for k, v in od_dict.items()}
    global_avg  = np.mean([np.mean(v) for v in od_dict.values()])

    with open(os.path.join(PROCESSED_DIR, "od_matrix.pkl"), 'wb') as f:
        pickle.dump({'od_avg': od_avg_time, 'global_avg': global_avg}, f)
    print(f"    -> OD 矩阵已保存（{len(od_avg_time)} 种 OD 对）")

    print("\n[*] 分批提取 Task B 特征...")
    batch_idx = 0
    X_batch, y_batch = [], []

    for idx, traj in enumerate(train_raw):
        coords = traj['coords']
        dep_time = traj['timestamps'][0]
        travel_time = traj['timestamps'][-1] - traj['timestamps'][0]

        if len(coords) >= 2 and travel_time > 0:
            base_feat, grid_o, grid_d = extract_task_b_features_advanced(coords, dep_time)
            od_key = f"{grid_o}_{grid_d}"
            hist_time = od_avg_time.get(od_key, global_avg)
            X_batch.append(base_feat + [hist_time])
            y_batch.append(travel_time)

        if len(X_batch) >= BATCH_SIZE or idx == total_records - 1:
            if X_batch:
                np.save(os.path.join(PROCESSED_DIR, f"X_batch_{batch_idx}.npy"), np.array(X_batch))
                np.save(os.path.join(PROCESSED_DIR, f"y_batch_{batch_idx}.npy"), np.array(y_batch))
                print(f"    -> Batch {batch_idx}：{len(X_batch)} 样本")
                batch_idx += 1
                X_batch, y_batch = [], []

    print("[✓] 数据预处理完毕！")


if __name__ == "__main__":
    build_offline_databases()
