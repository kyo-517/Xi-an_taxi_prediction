import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from features_and_utils import haversine, get_grid_id
from osm_map_matching import get_or_download_xian_graph, route_constrained_interpolation

# 配置文件路径
PROCESSED_DIR = "processed_data"
INPUT_DIR = "task_A_recovery"

def load_knn_db():
    knn_path = os.path.join(PROCESSED_DIR, "knn_db.pkl")
    if os.path.exists(knn_path):
        with open(knn_path, 'rb') as f:
            return pickle.load(f)
    return {}

def transplant_trajectory_shape(start_coord, end_coord, missing_timestamps, t_start, t_end, historical_segment):
    """历史形状移植，保持轨迹的自然弯曲"""
    predicted = []
    h0, hm = np.array(historical_segment[0]), np.array(historical_segment[-1])
    for ts in missing_timestamps:
        ratio = (ts - t_start) / (t_end - t_start) if t_end > t_start else 0.5
        idx = max(0, min(int(ratio * (len(historical_segment) - 1)), len(historical_segment) - 1))
        h_interp = np.array(historical_segment[idx])
        # 锚点锚定公式
        current_pred = h_interp + (np.array(start_coord) - h0) * (1 - ratio) + (np.array(end_coord) - hm) * ratio
        predicted.append(current_pred.tolist())
    return predicted

def recover_trajectory_hybrid(traj_item, knn_db, OSM_Graph=None):
    coords, timestamps, mask = traj_item['coords'], traj_item['timestamps'], traj_item['mask']
    full_coords = [None] * len(mask)
    
    # 填充已知点
    for k in range(len(mask)):
        if mask[k]: full_coords[k] = coords[k]

    i = 0
    while i < len(mask):
        if mask[i]:
            i += 1
        else:
            start_idx, j = i - 1, i
            while j < len(mask) and not mask[j]: j += 1
            end_idx = j
            
            missing_ts = timestamps[i:j]
            t_s = timestamps[start_idx] if start_idx >= 0 else missing_ts[0] - 15
            t_e = timestamps[end_idx] if end_idx < len(mask) else missing_ts[-1] + 15
            s_c = coords[start_idx] if start_idx >= 0 else coords[end_idx]
            e_c = coords[end_idx] if end_idx < len(mask) else coords[start_idx]

            # --- 混合策略决策树 ---
            grid_o, grid_d = get_grid_id(s_c[0], s_c[1]), get_grid_id(e_c[0], e_c[1])
            
            # 1. 优先使用 k-NN (历史经验最准)
            if (grid_o, grid_d) in knn_db:
                preds = transplant_trajectory_shape(s_c, e_c, missing_ts, t_s, t_e, knn_db[(grid_o, grid_d)][0])
                for k, p in enumerate(preds): full_coords[i+k] = p
                
            # 2. 其次使用 OSM (针对短距离缺失提供路网约束)
            elif OSM_Graph is not None and 100 < haversine(s_c[0], s_c[1], e_c[0], e_c[1]) < 1500:
                try:
                    # 限制距离：距离太长路网匹配会变慢且误差激增
                    pred = route_constrained_interpolation(OSM_Graph, [[s_c[0], s_c[1], t_s], [e_c[0], e_c[1], t_e]], missing_ts)
                    if len(pred) == len(missing_ts):
                        for k, p in enumerate(pred): full_coords[i+k] = p
                    else: raise ValueError()
                except: pass # 失败则进入下一步兜底
            
            i = j

    # --- 终极数学兜底：时间权重线性插值 (配合 bfill/ffill) ---
    # 先处理已知点和刚才由策略填充的点
    known_mask = [p is not None for p in full_coords]
    known_indices = np.where(known_mask)[0]
    
    # 提取所有已确定坐标的点用于数学插值
    existing_coords = np.array([full_coords[idx] for idx in known_indices])
    
    # 如果依然存在未填充点，使用时间序列插值
    # 将 timestamps 转换为 DatetimeIndex（从 Unix 秒数转换）
    df = pd.DataFrame(index=pd.to_datetime(timestamps, unit='s'))
    df['lon'] = np.nan
    df['lat'] = np.nan
    for idx, coord in zip(known_indices, existing_coords):
        df.iloc[idx, 0] = coord[0]
        df.iloc[idx, 1] = coord[1]
    
    # method='time' 现在可用，因为索引已是 DatetimeIndex
    df_interp = df.interpolate(method='time').bfill().ffill()
    
    return df_interp[['lon', 'lat']].values.tolist()

def evaluate_recovery(pred_data, gt_data, input_data):
    pred_dict = {item['traj_id']: item['coords'] for item in pred_data}
    gt_dict = {item['traj_id']: item['coords'] for item in gt_data}
    errors = []
    for item in input_data:
        tid, mask = item['traj_id'], item['mask']
        for k, is_known in enumerate(mask):
            if not is_known:
                dist = haversine(pred_dict[tid][k][0], pred_dict[tid][k][1], 
                                 gt_dict[tid][k][0], gt_dict[tid][k][1])
                if not np.isnan(dist): errors.append(dist)
    return np.mean(errors), np.sqrt(np.mean(np.array(errors)**2))

def run_task_a():
    knn_db = load_knn_db()
    OSM_Graph = get_or_download_xian_graph()
    
    for prefix in ["val", "test"]:
        for rate in [8, 16]:
            filename = f"{prefix}_input_{rate}.pkl"
            input_path = os.path.join(INPUT_DIR, filename)
            if not os.path.exists(input_path): continue
            
            print(f"\n[*] 正在处理: {filename}")
            with open(input_path, 'rb') as f: data = pickle.load(f)
            
            results = []
            # 性能优化：对于 1/16 缺失率，减少死板的路网匹配搜索
            for traj in tqdm(data, desc=f"进度 {rate}", unit="条", colour='cyan'):
                results.append({
                    'traj_id': traj['traj_id'],
                    'coords': recover_trajectory_hybrid(traj, knn_db, OSM_Graph)
                })
            
            # 自动评估
            gt_file = os.path.join(INPUT_DIR, "val_gt.pkl")
            if prefix == "val" and os.path.exists(gt_file):
                with open(gt_file, 'rb') as f: gt_data = pickle.load(f)
                mae, rmse = evaluate_recovery(results, gt_data, data)
                print(f"    -> [优化后成绩] MAE: {mae:.2f} 米, RMSE: {rmse:.2f} 米")
            
            out_path = os.path.join(INPUT_DIR, filename.replace("input", "pred"))
            with open(out_path, 'wb') as f: pickle.dump(results, f)
            print(f"[*] 结果已保存: {out_path}")

if __name__ == "__main__":
    run_task_a()