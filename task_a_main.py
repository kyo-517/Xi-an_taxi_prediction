import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm  # 引入实时进度条库
from features_and_utils import haversine, get_grid_id
from osm_map_matching import get_or_download_xian_graph, route_constrained_interpolation

USE_MAP_MATCHING = True 
PROCESSED_DIR = "processed_data"
SUBMISSION_DIR = "submissions"

def load_knn_db():
    knn_path = os.path.join(PROCESSED_DIR, "knn_db.pkl")
    if os.path.exists(knn_path):
        with open(knn_path, 'rb') as f:
            return pickle.load(f)
    return {}

def transplant_trajectory_shape(start_coord, end_coord, missing_timestamps, t_start, t_end, historical_segment):
    """
    [核心算法] 轨迹形状移植：将检索到的历史轨迹段锚定到当前的起始与终止点上，保留其中间的弯曲形状
    """
    predicted = []
    h0 = np.array(historical_segment[0])
    hm = np.array(historical_segment[-1])
    
    for ts in missing_timestamps:
        ratio = (ts - t_start) / (t_end - t_start) if t_end > t_start else 0.5
        idx = int(ratio * (len(historical_segment) - 1))
        idx = max(0, min(idx, len(historical_segment) - 1))
        
        h_interp = np.array(historical_segment[idx])
        current_pred = h_interp + (np.array(start_coord) - h0) * (1 - ratio) + (np.array(end_coord) - hm) * ratio
        predicted.append(current_pred.tolist())
        
    return predicted

def recover_trajectory_hybrid(traj_item, knn_db, OSM_Graph=None):
    coords = traj_item['coords']
    timestamps = traj_item['timestamps']
    mask = traj_item['mask']
    
    full_coords = []
    i = 0
    while i < len(mask):
        if mask[i]:
            full_coords.append(coords[i])
            i += 1
        else:
            start_idx = i - 1
            j = i
            while j < len(mask) and not mask[j]:
                j += 1
            end_idx = j
            
            missing_timestamps = timestamps[i:j]
            t_start = timestamps[start_idx] if start_idx >= 0 else missing_timestamps[0] - 15
            t_end = timestamps[end_idx] if end_idx < len(mask) else missing_timestamps[-1] + 15
            
            start_coord = coords[start_idx] if start_idx >= 0 else coords[end_idx]
            end_coord = coords[end_idx] if end_idx < len(mask) else coords[start_idx]

            # ==========================================
            #  策略 1: k-NN 历史轨迹检索
            # ==========================================
            grid_o = get_grid_id(start_coord[0], start_coord[1])
            grid_d = get_grid_id(end_coord[0], end_coord[1])
            
            if (grid_o, grid_d) in knn_db:
                hist_seg = knn_db[(grid_o, grid_d)][0]
                pred_coords = transplant_trajectory_shape(start_coord, end_coord, missing_timestamps, t_start, t_end, hist_seg)
                full_coords.extend(pred_coords)
                
            # ==========================================
            #  策略 2: OSM 路网匹配
            # ==========================================
            elif USE_MAP_MATCHING and OSM_Graph is not None:
                # [性能优化] 如果直线距离大于 3km，直接放弃路网匹配（因为图搜索会极其缓慢且容易错）
                dist_gap = haversine(start_coord[0], start_coord[1], end_coord[0], end_coord[1])
                if not np.isnan(dist_gap) and dist_gap < 3000:
                    try:
                        known_pts = [[start_coord[0], start_coord[1], t_start], 
                                     [end_coord[0], end_coord[1], t_end]]
                        pred_coords = route_constrained_interpolation(OSM_Graph, known_pts, missing_timestamps)
                        if len(pred_coords) == len(missing_timestamps):
                            full_coords.extend(pred_coords)
                        else:
                            raise ValueError("路网插值失败")
                    except:
                        for ts in missing_timestamps: full_coords.append([np.nan, np.nan])
                else:
                    for ts in missing_timestamps: full_coords.append([np.nan, np.nan])
            
            # ==========================================
            #  策略 3: 纯时间线性插值兜底
            # ==========================================
            else:
                for ts in missing_timestamps: full_coords.append([np.nan, np.nan])
                
            i = j
            
    # ==============================================================
    # --- 核心修复区：解决 Pandas set_index 的底层 Bug ---
    # ==============================================================
    fc_arr = np.array(full_coords, dtype=float)
    if fc_arr.shape[0] != len(timestamps):
        fc_arr = np.full((len(timestamps), 2), np.nan)
        for k in range(len(mask)):
            if mask[k]: fc_arr[k] = coords[k]
            
    df = pd.DataFrame(fc_arr, columns=['lon', 'lat'])
    df.index = pd.to_datetime(timestamps, unit='s')
    df_interpolated = df.interpolate(method='time').bfill().ffill()
    
    return df_interpolated[['lon', 'lat']].values.tolist()

def evaluate_recovery(pred_data, gt_data, input_data):
    pred_dict = {item['traj_id']: item['coords'] for item in pred_data}
    gt_dict = {item['traj_id']: item['coords'] for item in gt_data}
    errors = []
    for item in input_data:
        mask = item['mask']
        for i, is_known in enumerate(mask):
            if not is_known: 
                dist = haversine(pred_dict[item['traj_id']][i][0], pred_dict[item['traj_id']][i][1], 
                                 gt_dict[item['traj_id']][i][0], gt_dict[item['traj_id']][i][1])
                if not np.isnan(dist): errors.append(dist)
    return np.mean(errors), np.sqrt(np.mean(np.array(errors)**2))

def run_task_a():
    knn_db = load_knn_db()
    print(f"[*] 成功加载 k-NN 历史轨迹知识库，共包含 {len(knn_db)} 种区域跳转路径。")
    
    OSM_Graph = get_or_download_xian_graph() if USE_MAP_MATCHING else None

    base_dir = "task_A_recovery"
    # 支持验证集与课堂测试命名（val_input / test_input）
    for filename in ["val_input_8.pkl", "val_input_16.pkl", "test_input_8.pkl", "test_input_16.pkl"]:
        input_path = os.path.join(base_dir, filename)
        if not os.path.exists(input_path):
            continue
        with open(input_path, 'rb') as f:
            input_data = pickle.load(f)
            
        print(f"\n[*] 正在执行混合修补 (k-NN -> OSM -> 线性): {filename} ...")
        pred_results = []
        
        # ==================================================
        # 加入 tqdm 实时进度条包装 input_data
        # ==================================================
        for traj in tqdm(input_data, desc="修复进度", unit="条", colour="green"):
            pred_results.append({
                'traj_id': traj['traj_id'],
                'coords': recover_trajectory_hybrid(traj, knn_db, OSM_Graph)
            })
            
        gt_file = os.path.join(base_dir, "val_gt.pkl")
        if os.path.exists(gt_file):
            with open(gt_file, 'rb') as f: gt_data = pickle.load(f)
            mae, rmse = evaluate_recovery(pred_results, gt_data, input_data)
            print(f"\n    -> [混合模型最终成绩] MAE: {mae:.2f} 米, RMSE: {rmse:.2f} 米")
            
        # 本地保存预测结果（用于查看/评测）
        with open(os.path.join(base_dir, filename.replace("input", "pred")), 'wb') as f:
            pickle.dump(pred_results, f)
        # 生成提交文件（符合要求：list of {traj_id, coords}）
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        difficulty = filename.split('_')[-1].split('.')[0]  # 8 或 16
        submit_name = f"task_A_pred_{difficulty}.pkl"
        submit_path = os.path.join(SUBMISSION_DIR, submit_name)
        with open(submit_path, 'wb') as f:
            pickle.dump(pred_results, f)
        print(f"    -> 已生成提交文件: {submit_path}")

if __name__ == "__main__":
    run_task_a()