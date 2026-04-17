"""
数据预处理器（合并版）：
1. 用 data_org（高密度~3s）构建精细 k-NN 轨迹形状库（Task A）
2. 用 data_org 构建高精度 OD 耗时矩阵（Task B 先验知识）
3. 用 data_ds15 构建对齐特征空间的训练集（Task B）
"""
import pickle
import numpy as np
import os

try:
    import pygeohash as pgh
    HAS_GEOHASH = True
except ImportError:
    HAS_GEOHASH = False

# 配置文件路径
ORG_TRAIN_FILE  = os.path.join("data_org",  "train.pkl")
DS15_TRAIN_FILE = os.path.join("data_ds15", "train.pkl")
PROCESSED_DIR   = "processed_data"
BATCH_SIZE      = 20000


def get_grid_id(lon, lat, precision=6):
    """将经纬度转换为 Geohash 网格 ID，精度 6 约等于 1.2km x 0.6km 的街区"""
    if HAS_GEOHASH:
        return pgh.encode(lat, lon, precision=precision)
    else:
        return f"{int(lat * 100)}_{int(lon * 100)}"


def build_knowledge_from_org():
    """
    第一阶段：从高频 data_org 中提取物理先验知识。
    - Task A：用 gap=8 / gap=16 精确模拟缺失场景，构建高密度 k-NN 形状库
    - Task B：构建全局精准 OD 耗时矩阵
    """
    print(f"[*] 正在读取高频原始数据 {ORG_TRAIN_FILE} (约 13 万条)...")
    if not os.path.exists(ORG_TRAIN_FILE):
        print("    [警告] 未找到 data_org/train.pkl，请确保文件存在！")
        return {}

    with open(ORG_TRAIN_FILE, "rb") as f:
        org_data = pickle.load(f)
    print(f"    共 {len(org_data)} 条轨迹。")

    od_dict = {}
    knn_db  = {}
    MAX_PER_KEY = 3

    print("[*] 正在挖掘 OD 历史耗时矩阵 (Task B) 与物理转弯形状库 (Task A)...")

    for idx, traj in enumerate(org_data):
        if idx % 20000 == 0:
            print(f"    -> 进度: {idx}/{len(org_data)}")

        coords     = traj["coords"]
        timestamps = traj["timestamps"]
        n = len(coords)
        if n < 10:
            continue

        # ============================================================
        # 挖掘 1：Task A 高密度 k-NN 形状库
        # 用 gap=8 / gap=16 精确模拟缺失场景
        # ============================================================
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

        # ============================================================
        # 挖掘 2：Task B 全局精准 OD 耗时矩阵
        # 过滤超过 3 小时的异常订单
        # ============================================================
        travel_time = timestamps[-1] - timestamps[0]
        if 0 < travel_time <= 10800:
            grid_o = get_grid_id(coords[0][0],  coords[0][1])
            grid_d = get_grid_id(coords[-1][0], coords[-1][1])
            od_key = f"{grid_o}_{grid_d}"
            if od_key not in od_dict:
                od_dict[od_key] = []
            od_dict[od_key].append(travel_time)

    # 保存 Task A 的 k-NN 库
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    knn_path = os.path.join(PROCESSED_DIR, "knn_db.pkl")
    with open(knn_path, "wb") as f:
        pickle.dump(knn_db, f)
    print(f"    -> [完成] 高清轨迹形状库：{len(knn_db)} 种 OD 对，已保存至 {knn_path}")

    # 保存 Task B 的 OD 矩阵
    od_avg     = {k: np.mean(v) for k, v in od_dict.items()}
    global_avg = float(np.mean([np.mean(v) for v in od_dict.values()]))
    knowledge_base = {"od_avg": od_avg, "global_avg": global_avg}

    out_path = os.path.join(PROCESSED_DIR, "od_matrix.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(knowledge_base, f)
    print(f"    -> [完成] 高精度 OD 耗时矩阵：{len(od_avg)} 种 OD 对，已保存至 {out_path}")

    return knowledge_base


def build_training_features_from_ds15(knowledge_base):
    """
    第二阶段：从降采样 data_ds15 中提取特征，保证与现场考试数据分布完全对齐。
    """
    print(f"\n[*] 正在读取降采样训练数据 {DS15_TRAIN_FILE} 以对齐特征空间...")
    if not os.path.exists(DS15_TRAIN_FILE):
        print("    [警告] 未找到 data_ds15/train.pkl，请确保文件存在！")
        return

    with open(DS15_TRAIN_FILE, "rb") as f:
        ds15_data = pickle.load(f)

    od_avg     = knowledge_base.get("od_avg", {})
    global_avg = knowledge_base.get("global_avg", 1200)

    try:
        from features_and_utils import extract_task_b_features_advanced
    except ImportError:
        print("    [错误] 无法导入 features_and_utils，请确保特征工具库存在。")
        return

    X_train, y_train = [], []
    print("[*] 正在结合 Org 先验知识与 DS15 物理特征构建最终训练集...")

    for traj in ds15_data:
        coords      = traj["coords"]
        dep_time    = traj["timestamps"][0]
        travel_time = traj["timestamps"][-1] - traj["timestamps"][0]

        if len(coords) < 2 or travel_time <= 0:
            continue

        base_feat, grid_o, grid_d = extract_task_b_features_advanced(coords, dep_time)
        hist_time  = od_avg.get(f"{grid_o}_{grid_d}", global_avg)
        final_feat = base_feat + [hist_time]

        X_train.append(final_feat)
        y_train.append(travel_time)

    print(f"    -> 特征维度: {len(X_train[0])}，样本数: {len(X_train)}")

    # 保存分批格式（task_b_main.py 读取 X_batch_*.npy）
    batch_idx = 0
    for start in range(0, len(X_train), BATCH_SIZE):
        end = start + BATCH_SIZE
        np.save(os.path.join(PROCESSED_DIR, f"X_batch_{batch_idx}.npy"),
                np.array(X_train[start:end]))
        np.save(os.path.join(PROCESSED_DIR, f"y_batch_{batch_idx}.npy"),
                np.array(y_train[start:end]))
        batch_idx += 1
    print(f"    -> [完成] 分批格式已保存（{batch_idx} 个 batch）")


if __name__ == "__main__":
    print("=" * 60)
    print("启动双源数据处理管道 (Data Pipeline V2)")
    print("策略：data_org 提取高清形状库/先验耗时，data_ds15 约束特征空间")
    print("=" * 60)

    kb = build_knowledge_from_org()
    if kb:
        build_training_features_from_ds15(kb)

    print("\n[✓] 数据预处理完毕！")
