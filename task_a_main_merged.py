"""
Task A - 轨迹填补（合并版）
融合新旧版的优点：
- 新版的数据结构设计（预分配数组 + 直接赋值）
- 新版的 OSM 距离范围 100-1500m（86%+ 覆盖率）
- 旧版的提交文件生成逻辑
- 更清晰的异常处理
"""
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from features_and_utils import haversine, get_grid_id
from osm_map_matching import get_or_download_xian_graph, route_constrained_interpolation

# 配置
PROCESSED_DIR = "processed_data"
INPUT_DIR = "task_A_recovery"
SUBMISSION_DIR = "submissions"

def load_knn_db():
    """加载 k-NN 历史轨迹库"""
    knn_path = os.path.join(PROCESSED_DIR, "knn_db.pkl")
    if os.path.exists(knn_path):
        with open(knn_path, 'rb') as f:
            return pickle.load(f)
    return {}

def transplant_trajectory_shape(start_coord, end_coord, missing_timestamps, t_start, t_end, historical_segment):
    """
    轨迹形状移植：将历史轨迹段锚定到当前起终点，保留中间弯曲形状
    """
    predicted = []
    h0 = np.array(historical_segment[0])
    hm = np.array(historical_segment[-1])
    
    for ts in missing_timestamps:
        ratio = (ts - t_start) / (t_end - t_start) if t_end > t_start else 0.5
        idx = max(0, min(int(ratio * (len(historical_segment) - 1)), len(historical_segment) - 1))
        h_interp = np.array(historical_segment[idx])
        # 锚点锚定公式：结合历史形状与当前几何位置
        current_pred = h_interp + (np.array(start_coord) - h0) * (1 - ratio) + (np.array(end_coord) - hm) * ratio
        predicted.append(current_pred.tolist())
    
    return predicted

def recover_trajectory_hybrid(traj_item, knn_db, OSM_Graph=None):
    """
    混合轨迹恢复策略：k-NN → OSM → 时间插值
    --- 改进点 ---
    1. 用预分配数组 + 直接赋值（新版数据结构）
    2. OSM 距离范围 100-1500m（基于数据分析）
    3. 显式处理 None 值用于最终插值
    """
    coords = traj_item['coords']
    timestamps = traj_item['timestamps']
    mask = traj_item['mask']
    
    # 【新版改进】预分配数组，避免 append 顺序问题
    full_coords = [None] * len(mask)
    
    # 填充已知点
    for k in range(len(mask)):
        if mask[k]:
            full_coords[k] = coords[k]
    
    # 遍历缺失段
    i = 0
    while i < len(mask):
        if mask[i]:
            i += 1
        else:
            # 找到连续缺失段 [i, j)
            start_idx = i - 1
            j = i
            while j < len(mask) and not mask[j]:
                j += 1
            end_idx = j
            
            missing_ts = timestamps[i:j]
            t_s = timestamps[start_idx] if start_idx >= 0 else missing_ts[0] - 15
            t_e = timestamps[end_idx] if end_idx < len(mask) else missing_ts[-1] + 15
            
            s_c = coords[start_idx] if start_idx >= 0 else coords[end_idx]
            e_c = coords[end_idx] if end_idx < len(mask) else coords[start_idx]
            
            # ============ 混合策略决策树 ============
            grid_o = get_grid_id(s_c[0], s_c[1])
            grid_d = get_grid_id(e_c[0], e_c[1])
            
            # 1. 优先：k-NN 历史轨迹（最准确）
            if (grid_o, grid_d) in knn_db:
                preds = transplant_trajectory_shape(s_c, e_c, missing_ts, t_s, t_e, knn_db[(grid_o, grid_d)][0])
                for k, p in enumerate(preds):
                    full_coords[i + k] = p
            
            # 2. 其次：OSM 路网匹配（短距离约束）
            # 【基于数据分析】100-1500m 范围覆盖 86%+ 的缺失段，性能和精度最佳
            elif OSM_Graph is not None:
                dist = haversine(s_c[0], s_c[1], e_c[0], e_c[1])
                if 100 <= dist <= 1500:
                    try:
                        pred = route_constrained_interpolation(
                            OSM_Graph,
                            [[s_c[0], s_c[1], t_s], [e_c[0], e_c[1], t_e]],
                            missing_ts
                        )
                        if len(pred) == len(missing_ts):
                            for k, p in enumerate(pred):
                                full_coords[i + k] = p
                        # 否则进入第3策略（兜底）
                    except:
                        pass  # 失败时进入第3策略
            
            i = j
    
    # ============ 第3步：时间权重插值兜底 ============
    # 统计已填充和未填充的点
    known_mask = [p is not None for p in full_coords]
    known_indices = np.where(known_mask)[0]
    
    if len(known_indices) > 0:
        # 仅从已填充的点构建 DataFrame，确保插值有基础
        df = pd.DataFrame(index=pd.to_datetime(timestamps, unit='s'))
        df['lon'] = np.nan
        df['lat'] = np.nan
        
        for idx in known_indices:
            df.iloc[idx, 0] = full_coords[idx][0]
            df.iloc[idx, 1] = full_coords[idx][1]
        
        # 时间权重插值 + bfill/ffill
        df_interp = df.interpolate(method='time').bfill().ffill()
        final_coords = df_interp[['lon', 'lat']].values.tolist()
    else:
        # 极端情况：没有任何已知点（不应该出现）
        final_coords = [[np.nan, np.nan] for _ in timestamps]
    
    return final_coords

def evaluate_recovery(pred_data, gt_data, input_data):
    """评估预测精度（仅对有真值的数据）"""
    pred_dict = {item['traj_id']: item['coords'] for item in pred_data}
    gt_dict = {item['traj_id']: item['coords'] for item in gt_data}
    
    errors = []
    for item in input_data:
        tid = item['traj_id']
        mask = item['mask']
        
        if tid not in gt_dict:
            continue
        
        for k, is_known in enumerate(mask):
            if not is_known:
                try:
                    dist = haversine(
                        pred_dict[tid][k][0], pred_dict[tid][k][1],
                        gt_dict[tid][k][0], gt_dict[tid][k][1]
                    )
                    if not np.isnan(dist):
                        errors.append(dist)
                except:
                    pass
    
    if errors:
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        return mae, rmse
    return None, None

def run_task_a():
    """主流程"""
    knn_db = load_knn_db()
    print(f"[*] 已加载 k-NN 库，共 {len(knn_db)} 种 OD 对")
    
    OSM_Graph = get_or_download_xian_graph()
    
    # 创建提交目录
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    # 支持验证和测试输入
    for prefix in ["val", "test"]:
        for rate in [8, 16]:
            filename = f"{prefix}_input_{rate}.pkl"
            input_path = os.path.join(INPUT_DIR, filename)
            
            if not os.path.exists(input_path):
                continue
            
            print(f"\n[*] 处理文件: {filename}")
            with open(input_path, 'rb') as f:
                input_data = pickle.load(f)
            
            # 恢复轨迹
            pred_results = []
            for traj in tqdm(input_data, desc=f"混合恢复 {rate}", unit="条", colour="green"):
                pred_results.append({
                    'traj_id': traj['traj_id'],
                    'coords': recover_trajectory_hybrid(traj, knn_db, OSM_Graph)
                })
            
            # 评估（若有真值）
            gt_file = os.path.join(INPUT_DIR, "val_gt.pkl")
            if prefix == "val" and os.path.exists(gt_file):
                with open(gt_file, 'rb') as f:
                    gt_data = pickle.load(f)
                mae, rmse = evaluate_recovery(pred_results, gt_data, input_data)
                if mae is not None:
                    print(f"    → 评测结果: MAE={mae:.2f}m, RMSE={rmse:.2f}m")
            
            # 本地保存（用于查看）
            local_path = os.path.join(INPUT_DIR, filename.replace("input", "pred"))
            with open(local_path, 'wb') as f:
                pickle.dump(pred_results, f)
            print(f"    → 本地保存: {local_path}")
            
            # 生成提交文件
            difficulty = filename.split('_')[-1].split('.')[0]  # 8 或 16
            submit_name = f"task_A_pred_{difficulty}.pkl"
            submit_path = os.path.join(SUBMISSION_DIR, submit_name)
            with open(submit_path, 'wb') as f:
                pickle.dump(pred_results, f)
            print(f"    → 提交文件: {submit_path}")

if __name__ == "__main__":
    print("="*60)
    print("任务 A：轨迹填补（混合修复）")
    print("="*60)
    run_task_a()
    print("\n[✓] 任务完成！")
