import pickle
import numpy as np
import os
try:
    import pygeohash as pgh
    HAS_GEOHASH = True
except ImportError:
    HAS_GEOHASH = False

# 配置文件路径
ORG_TRAIN_FILE = os.path.join("data_org", "train.pkl")
DS15_TRAIN_FILE = os.path.join("data_ds15", "train.pkl")
PROCESSED_DIR = "processed_data"

def get_grid_id(lon, lat, precision=6):
    """将经纬度转换为Geohash网格ID，精度6约等于1.2km x 0.6km的街区"""
    if HAS_GEOHASH:
        return pgh.encode(lat, lon, precision=precision)
    else:
        return f"{int(lat*100)}_{int(lon*100)}"

def build_knowledge_from_org():
    """
    第一阶段：从高频 data_org 中提取绝对精准的物理先验知识 (Task A 的形状库 + Task B 的耗时矩阵)
    """
    print(f"[*] 正在读取高频原始数据 {ORG_TRAIN_FILE} (约 13 万条)...")
    if not os.path.exists(ORG_TRAIN_FILE):
        print("    [警告] 未找到 data_org/train.pkl，请确保文件存在！")
        return {}

    with open(ORG_TRAIN_FILE, 'rb') as f:
        org_data = pickle.load(f)

    od_dict = {}
    knn_db = {}
    print("[*] 正在挖掘极高精度 OD 历史耗时矩阵 (Task B) 与 物理转弯形状库 (Task A)...")
    
    for traj in org_data:
        coords = traj['coords']
        timestamps = traj['timestamps']
        if len(coords) < 10: continue
        
        # ============================================================
        # 挖掘 1：Task A 高清 k-NN 形状库 (3秒一滴的平滑曲线)
        # 截取约 90 秒的连续行驶片段作为形状特征
        # ============================================================
        if len(coords) >= 30:
            for i in range(0, len(coords) - 30, 15):
                k_start = coords[i]
                k_end = coords[i+30]
                g_o = get_grid_id(k_start[0], k_start[1])
                g_d = get_grid_id(k_end[0], k_end[1])
                
                # 仅保存跨越网格的片段，避免原地停留的无意义形状
                if g_o != g_d:
                    key = (g_o, g_d)
                    if key not in knn_db:
                        knn_db[key] = []
                    # 每个区域对最多保存 3 条高清路线作为知识补丁
                    if len(knn_db[key]) < 3:
                        knn_db[key].append(coords[i:i+31])

        # ============================================================
        # 挖掘 2：Task B 全局精准 OD 耗时矩阵
        # ============================================================
        travel_time = timestamps[-1] - timestamps[0]
        # 过滤异常极值(>3小时的订单视为司机中途休息)
        if travel_time > 0 and travel_time <= 10800: 
            grid_o = get_grid_id(coords[0][0], coords[0][1])
            grid_d = get_grid_id(coords[-1][0], coords[-1][1])
            od_key = f"{grid_o}_{grid_d}"

            if od_key not in od_dict:
                od_dict[od_key] = []
            od_dict[od_key].append(travel_time)

    # 保存 Task A 的 k-NN 库
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    knn_path = os.path.join(PROCESSED_DIR, "knn_db.pkl")
    with open(knn_path, 'wb') as f:
        pickle.dump(knn_db, f)
    print(f"    -> [完成] 成功提取 {len(knn_db)} 种区域跳转的高清轨迹形状库，已保存至 {knn_path}")

    # 保存 Task B 的 OD 矩阵
    od_avg = {k: np.mean(v) for k, v in od_dict.items()}
    global_avg = np.mean([np.mean(v) for v in od_dict.values()])
    knowledge_base = {'od_avg': od_avg, 'global_avg': global_avg}
    
    out_path = os.path.join(PROCESSED_DIR, "od_matrix_org.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(knowledge_base, f)
    print(f"    -> [完成] 成功提取 {len(od_avg)} 种区域跳转的高精度历史耗时，已保存至 {out_path}")
    
    return knowledge_base

def build_training_features_from_ds15(knowledge_base):
    """
    第二阶段：从降采样 data_ds15 中提取特征，保证与现场考试数据分布完全对齐
    """
    print(f"\n[*] 正在读取降采样训练数据 {DS15_TRAIN_FILE}以对齐特征空间...")
    if not os.path.exists(DS15_TRAIN_FILE):
        print("    [警告] 未找到 data_ds15/train.pkl，请确保文件存在！")
        return

    with open(DS15_TRAIN_FILE, 'rb') as f:
        ds15_data = pickle.load(f)

    od_avg = knowledge_base.get('od_avg', {})
    global_avg = knowledge_base.get('global_avg', 1200)

    # 动态导入 task_b 的特征提取函数
    try:
        from features_and_utils import extract_task_b_features_advanced
    except ImportError:
        print("    [错误] 无法导入 features_and_utils，请确保特征工具库存在。")
        return
    
    X_train, y_train = [], []
    print("[*] 正在结合 Org 先验知识与 DS15 物理特征构建最终训练集...")
    
    for traj in ds15_data:
        coords = traj['coords']
        dep_time = traj['timestamps'][0]
        travel_time = traj['timestamps'][-1] - traj['timestamps'][0]
        
        # 1. 获取 DS15 的基础物理特征
        base_feat, grid_o, grid_d = extract_task_b_features_advanced(coords, dep_time)
        
        # 2. 挂载从 Org 提取的高精度先验知识
        hist_time = od_avg.get(f"{grid_o}_{grid_d}", global_avg)
        final_feat = base_feat + [hist_time]
        
        X_train.append(final_feat)
        y_train.append(travel_time)

    # 保存最终用于模型训练的数据切片
    X_path = os.path.join(PROCESSED_DIR, "X_train_final.npy")
    y_path = os.path.join(PROCESSED_DIR, "y_train_final.npy")
    np.save(X_path, np.array(X_train))
    np.save(y_path, np.array(y_train))
    print(f"    -> [完成] 完美对齐的训练矩阵已保存！特征维度: {len(X_train[0])}，样本数: {len(X_train)}")

if __name__ == "__main__":
    print("="*60)
    print("启动双源数据处理管道 (Data Pipeline V2)")
    print("策略：利用 data_org 提取高清形状库/先验耗时，利用 data_ds15 约束特征空间")
    print("="*60)
    
    kb = build_knowledge_from_org()
    if kb:
        build_training_features_from_ds15(kb)