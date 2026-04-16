import pickle
import numpy as np
import os
import math
from features_and_utils import extract_task_b_features_advanced, get_grid_id

# ================= 配置区 =================
TRAIN_FILE = os.path.join("data_ds15", "train.pkl")
BATCH_SIZE = 20000  # 每次处理并持久化的数据量
PROCESSED_DIR = "processed_data"
# ==========================================

def build_offline_databases():
    """
    分批次处理原始训练集：
    1. 构建供任务A使用的 k-NN 历史轨迹形状库 (k-NN Trajectory DB)
    2. 构建供任务B使用的 OD (起终点) 历史平均耗时矩阵
    3. 提取任务B的模型特征并分批保存 (Batch-wise Feature Extraction)
    """
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    print("[*] 正在加载全量原始训练集 (仅在内存中驻留一次)...")
    with open(TRAIN_FILE, 'rb') as f:
        train_raw = pickle.load(f)
        
    total_records = len(train_raw)
    print(f"[*] 训练集加载完毕，共 {total_records} 条轨迹。")

    knn_db = {}     # 用于 Task A: {(Grid_O, Grid_D): [历史轨迹片段序列]}
    od_dict = {}    # 用于 Task B: {Grid_O_Grid_D: [耗时列表]}

    print("[*] 第一阶段：扫描数据构建字典库 (k-NN DB & OD Matrix)...")
    for idx, traj in enumerate(train_raw):
        if idx % 20000 == 0:
            print(f"    -> 扫描进度: {idx}/{total_records}")
            
        coords = traj['coords']
        timestamps = traj['timestamps']
        if len(coords) < 16:
            continue
            
        # 1. 采集历史片段，供 Task A (1/8和1/16采样率) 的 k-NN 匹配使用
        # 采集间隔为 8 的轨迹段
        for i in range(0, len(coords) - 8, 8):
            grid_o = get_grid_id(coords[i][0], coords[i][1], precision=6)
            grid_d = get_grid_id(coords[i+8][0], coords[i+8][1], precision=6)
            key = (grid_o, grid_d)
            if key not in knn_db:
                knn_db[key] = []
            # 为了防止内存爆炸，每种 OD 对最多只保留 3 条历史转弯形状
            if len(knn_db[key]) < 3:
                knn_db[key].append(coords[i:i+9])

        # 2. 统计完整轨迹的 OD 耗时，供 Task B 使用
        travel_time = timestamps[-1] - timestamps[0]
        if travel_time > 0:
            grid_start = get_grid_id(coords[0][0], coords[0][1], precision=6)
            grid_end = get_grid_id(coords[-1][0], coords[-1][1], precision=6)
            od_key = f"{grid_start}_{grid_end}"
            if od_key not in od_dict:
                od_dict[od_key] = []
            od_dict[od_key].append(travel_time)

    # 聚合 OD 均值
    od_avg_time = {k: np.mean(v) for k, v in od_dict.items()}
    global_avg = np.mean([np.mean(v) for v in od_dict.values()])

    print("[*] 保存 k-NN 轨迹库与 OD 均值矩阵到本地...")
    with open(os.path.join(PROCESSED_DIR, "knn_db.pkl"), 'wb') as f:
        pickle.dump(knn_db, f)
    with open(os.path.join(PROCESSED_DIR, "od_matrix.pkl"), 'wb') as f:
        pickle.dump({'od_avg': od_avg_time, 'global_avg': global_avg}, f)

    print("\n[*] 第二阶段：分批提取特征矩阵 (Batch Feature Processing)...")
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
            
            final_feat = base_feat + [hist_time]
            X_batch.append(final_feat)
            y_batch.append(travel_time)
            
        # 触发批量保存
        if len(X_batch) >= BATCH_SIZE or idx == total_records - 1:
            X_arr = np.array(X_batch)
            y_arr = np.array(y_batch)
            np.save(os.path.join(PROCESSED_DIR, f"X_batch_{batch_idx}.npy"), X_arr)
            np.save(os.path.join(PROCESSED_DIR, f"y_batch_{batch_idx}.npy"), y_arr)
            print(f"    -> 已保存 Batch {batch_idx}，包含 {len(X_arr)} 个样本。")
            
            batch_idx += 1
            X_batch, y_batch = [], [] # 清空当前 Batch，释放内存
            
    print("[*] 数据预处理流水线执行完毕！")

if __name__ == "__main__":
    build_offline_databases()