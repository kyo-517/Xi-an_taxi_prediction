"""
数据预处理器（合并版）：
1. 用 data_org 构建 Task A 所需的 k-NN 轨迹形状库
2. 用 data_org 构建 Task B 所需的高精度 OD 先验
3. 用 data_ds15 构建 Task B 所需的对齐特征训练集
4. 同时生成 Task B 新版（task_b_main_new.py）所需的 processed_data_b 输出
"""
import os
import pickle
import numpy as np

try:
    import pygeohash as pgh
    HAS_GEOHASH = True
except ImportError:
    HAS_GEOHASH = False

# 配置文件路径
ORG_TRAIN_FILE  = os.path.join("data_org",  "train.pkl")
DS15_TRAIN_FILE = os.path.join("data_ds15", "train.pkl")
PROCESSED_DIR_A = "processed_data_a"
PROCESSED_DIR_B = "processed_data_b"
BATCH_SIZE      = 20000


def get_grid_id(lon, lat, precision=6):
    """将经纬度转换为 Geohash 网格 ID，精度 6 约等于 1.2km x 0.6km 的街区"""
    if HAS_GEOHASH:
        return pgh.encode(lat, lon, precision=precision)
    return f"{int(lat * 100)}_{int(lon * 100)}"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def build_knowledge_from_org():
    """
    第一阶段：从高频 data_org 提取物理先验知识。
    - Task A：构建高密度 k-NN 轨迹形状库
    - Task B：构建高精度 OD 耗时矩阵
    """
    print(f"[*] 正在读取高频原始数据 {ORG_TRAIN_FILE}...")
    if not os.path.exists(ORG_TRAIN_FILE):
        print("    [警告] 未找到 data_org/train.pkl，请确保文件存在！")
        return {}

    with open(ORG_TRAIN_FILE, "rb") as f:
        org_data = pickle.load(f)
    print(f"    共 {len(org_data)} 条轨迹。")

    od_dict = {}
    knn_db  = {}
    MAX_PER_KEY = 3

    print("[*] 正在构建 Task A k-NN 轨迹库 和 Task B OD 先验...")

    for idx, traj in enumerate(org_data):
        if idx % 20000 == 0:
            print(f"    -> 进度: {idx}/{len(org_data)}")

        coords     = traj["coords"]
        timestamps = traj["timestamps"]
        n = len(coords)
        if n < 10:
            continue

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

        travel_time = timestamps[-1] - timestamps[0]
        if 0 < travel_time <= 10800:
            grid_o = get_grid_id(coords[0][0],  coords[0][1])
            grid_d = get_grid_id(coords[-1][0], coords[-1][1])
            od_key = f"{grid_o}_{grid_d}"
            od_dict.setdefault(od_key, []).append(travel_time)

    ensure_dir(PROCESSED_DIR_A)
    ensure_dir(PROCESSED_DIR_B)

    knn_path_a = os.path.join(PROCESSED_DIR_A, "knn_db.pkl")
    save_pickle(knn_db, knn_path_a)
    print(f"    -> [完成] Task A k-NN 库已保存至 {knn_path_a} ({len(knn_db)} 种 OD 对)")

    od_avg     = {k: np.mean(v) for k, v in od_dict.items()}
    global_avg = float(np.mean([np.mean(v) for v in od_dict.values()])) if od_dict else 1200.0
    knowledge_base = {"od_avg": od_avg, "global_avg": global_avg}

    od_path_a = os.path.join(PROCESSED_DIR_A, "od_matrix.pkl")
    save_pickle(knowledge_base, od_path_a)
    print(f"    -> [完成] Task B OD 先验已保存至 {od_path_a} ({len(od_avg)} 种 OD 对)")

    od_path_b = os.path.join(PROCESSED_DIR_B, "od_matrix_org.pkl")
    save_pickle(knowledge_base, od_path_b)
    print(f"    -> [完成] Task B 新版 OD 先验已保存至 {od_path_b}")

    return knowledge_base


def build_training_features_from_ds15(knowledge_base):
    """
    第二阶段：从降采样 data_ds15 提取 Task B 特征，构建训练集。
    """
    print(f"\n[*] 正在读取降采样训练数据 {DS15_TRAIN_FILE}...")
    if not os.path.exists(DS15_TRAIN_FILE):
        print("    [警告] 未找到 data_ds15/train.pkl，请确保文件存在！")
        return

    with open(DS15_TRAIN_FILE, "rb") as f:
        ds15_data = pickle.load(f)

    od_avg     = knowledge_base.get("od_avg", {})
    global_avg = knowledge_base.get("global_avg", 1200.0)

    try:
        from features_and_utils import extract_task_b_features_advanced
    except ImportError:
        print("    [错误] 无法导入 features_and_utils，请确保特征工具库存在。")
        return

    X_train, y_train = [], []
    print("[*] 正在构建 Task B 特征与标签...")

    for idx, traj in enumerate(ds15_data):
        if idx % 20000 == 0:
            print(f"    -> 进度: {idx}/{len(ds15_data)}")

        coords     = traj["coords"]
        timestamps = traj["timestamps"]
        if len(coords) < 2:
            continue

        travel_time = timestamps[-1] - timestamps[0]
        if travel_time <= 0:
            continue

        base_feat, grid_o, grid_d = extract_task_b_features_advanced(coords, timestamps[0])
        hist_time = od_avg.get(f"{grid_o}_{grid_d}", global_avg)
        X_train.append(base_feat + [hist_time])
        y_train.append(travel_time)

    if not X_train:
        print("    [警告] 未生成任何 Task B 训练样本。")
        return

    X_arr = np.array(X_train, dtype=np.float32)
    y_arr = np.array(y_train, dtype=np.float32)
    print(f"    -> 总样本数: {len(X_arr)}，特征维度: {X_arr.shape[1]}")

    ensure_dir(PROCESSED_DIR_A)
    ensure_dir(PROCESSED_DIR_B)

    batch_idx = 0
    for start in range(0, len(X_arr), BATCH_SIZE):
        np.save(os.path.join(PROCESSED_DIR_A, f"X_batch_{batch_idx}.npy"), X_arr[start:start+BATCH_SIZE])
        np.save(os.path.join(PROCESSED_DIR_A, f"y_batch_{batch_idx}.npy"), y_arr[start:start+BATCH_SIZE])
        batch_idx += 1
    print(f"    -> [完成] Task B 分批训练数据已保存至 {PROCESSED_DIR_A} ({batch_idx} 个 batch)")

    np.save(os.path.join(PROCESSED_DIR_B, "X_train_final.npy"), X_arr)
    np.save(os.path.join(PROCESSED_DIR_B, "y_train_final.npy"), y_arr)
    print(f"    -> [完成] Task B 新版训练数据已保存至 {PROCESSED_DIR_B}")


def main():
    print("=" * 60)
    print("启动合并数据预处理管道")
    print("生成 Task A + Task B 所需文件：processed_data_a 与 processed_data_b")
    print("=" * 60)

    kb = build_knowledge_from_org()
    if kb:
        build_training_features_from_ds15(kb)

    print("\n[✓] 数据预处理完毕！")


if __name__ == "__main__":
    main()
