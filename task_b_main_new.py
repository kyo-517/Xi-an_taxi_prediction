import pickle
import numpy as np
import os
import xgboost as xgb
import warnings
from tqdm import tqdm
from features_and_utils import extract_task_b_features_advanced, evaluate_metrics

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
    USE_LGB = True
except ImportError:
    USE_LGB = False

PROCESSED_DIR = "processed_data_b"
INPUT_DIR = "task_B_tte"

def train_models_direct_mae(xgb_path, lgb_path):
    """
    基于双源数据的全新单次训练流程 (恢复全量模型参数容量)
    """
    x_file = os.path.join(PROCESSED_DIR, "X_train_final.npy")
    y_file = os.path.join(PROCESSED_DIR, "y_train_final.npy")
    
    if not os.path.exists(x_file):
        print(f"[错误] 未找到特征切片 {x_file}，请先运行 data_processor_v2.py。")
        return None, None

    print(f"[*] 启动双模型 MAE 直优训练流程 (基于 org 双源高精度数据)...")
    X_train = np.load(x_file)
    y_train = np.load(y_file)
    
    # 【核心修复】：恢复模型的深度学习容量，将树的数量从 45 提升至 350
    print(f"    -> 配置 XGBoost 引擎 (350 棵树)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=350,            # 恢复全量训练所需的树数量
        learning_rate=0.08, 
        max_depth=8, 
        subsample=0.85, 
        colsample_bytree=0.85, 
        objective='reg:absoluteerror', 
        gamma=0.2, 
        min_child_weight=3,
        random_state=42, 
        n_jobs=-1
    )
    
    lgb_model = None
    if USE_LGB:
        print(f"    -> 配置 LightGBM 引擎 (350 棵树)...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=350,        # 恢复全量训练所需的树数量
            learning_rate=0.08, 
            max_depth=8, 
            num_leaves=63, 
            subsample=0.85, 
            colsample_bytree=0.85,
            objective='mae',               
            min_split_gain=0.2, 
            min_child_samples=10,
            random_state=42, 
            n_jobs=-1
        )

    print(f"    -> 正在强力训练 XGBoost (样本数: {len(X_train)})...")
    xgb_model.fit(X_train, y_train)
        
    if USE_LGB:
        print(f"    -> 正在强力训练 LightGBM...")
        lgb_model.fit(X_train, y_train)
            
    print("[*] 训练完成！巅峰双模型已生成。")
    return xgb_model, lgb_model

def run_task_b():
    xgb_path = os.path.join(PROCESSED_DIR, "xgb_model.json")
    lgb_path = os.path.join(PROCESSED_DIR, "lgb_model.txt")
    
    xgb_model, lgb_model = None, None
    
    if os.path.exists(xgb_path):
        print("[*] 发现本地持久化模型，直接加载进入推理模式 (跳过训练防止过拟合)...")
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(xgb_path)
        if USE_LGB and os.path.exists(lgb_path):
            lgb_model = lgb.Booster(model_file=lgb_path)
    else:
        xgb_model, lgb_model = train_models_direct_mae(xgb_path, lgb_path)
        if xgb_model is None: return
        xgb_model.save_model(xgb_path)
        if USE_LGB and lgb_model is not None:
            lgb_model.booster_.save_model(lgb_path)

    od_path = os.path.join(PROCESSED_DIR, "od_matrix_org.pkl")
    if not os.path.exists(od_path):
         print(f"[错误] 未找到高精度知识库 {od_path}，请先运行 data_processor_v2.py。")
         return
         
    with open(od_path, 'rb') as f:
        od_data = pickle.load(f)
    od_avg, glob_avg = od_data['od_avg'], od_data['global_avg']

    for prefix in ["val", "test"]:
        filename = f"{prefix}_input.pkl"
        input_path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(input_path): continue
        
        print(f"\n[*] 正在为 {filename} 生成行程时间预测...")
        with open(input_path, 'rb') as f: data = pickle.load(f)
        
        X_feats, traj_ids = [], []
        for traj in tqdm(data, desc=f"提取特征", unit="条", colour="green"):
            # 兼容 extract_task_b_features_advanced 的返回值 (新旧版本均适用)
            ext_res = extract_task_b_features_advanced(traj['coords'], traj['departure_timestamp'])
            if len(ext_res) == 3:
                f, g_o, g_d = ext_res
            else:
                f = ext_res
                g_o, g_d = "unknown", "unknown"
                
            hist_t = od_avg.get(f"{g_o}_{g_d}", glob_avg)
            X_feats.append(f + [hist_t])
            traj_ids.append(traj['traj_id'])
            
        X_feats_arr = np.array(X_feats)
        
        xgb_pred = xgb_model.predict(X_feats_arr)
        
        if USE_LGB and lgb_model is not None:
            lgb_pred = lgb_model.predict(X_feats_arr)
            final_pred_time = 0.6 * xgb_pred + 0.4 * lgb_pred
        else:
            final_pred_time = xgb_pred
            
        results = []
        for tid, t_time in zip(traj_ids, final_pred_time):
            results.append({
                'traj_id': tid,
                'travel_time': max(1.0, round(float(t_time), 2)) 
            })
            
        gt_file = os.path.join(INPUT_DIR, "val_gt.pkl")
        if prefix == "val" and os.path.exists(gt_file):
            with open(gt_file, 'rb') as f: gt_data = pickle.load(f)
            gt_dict = {item['traj_id']: item['travel_time'] for item in gt_data}
            y_true = [gt_dict[tid] for tid in traj_ids]
            y_pred_eval = [item['travel_time'] for item in results]
            
            mae, rmse, mape = evaluate_metrics(y_true, y_pred_eval)
            print(f"    -> [高精度先验融合模型成绩] MAE: {mae:.2f} 秒 | RMSE: {rmse:.2f} 秒 | MAPE: {mape:.2f} %")
            
        out_path = os.path.join(INPUT_DIR, f"{prefix}_pred.pkl")
        with open(out_path, 'wb') as f: pickle.dump(results, f)
        print(f"[完成] 预测结果已保存至: {out_path}")

if __name__ == "__main__":
    run_task_b()