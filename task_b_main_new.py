import pickle
import numpy as np
import os
import glob
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

PROCESSED_DIR = "processed_data"
INPUT_DIR = "task_B_tte"

def train_models_in_batches_direct_mae(xgb_path, lgb_path):
    """
    直接使用 L1 Loss (MAE) 作为底层优化目标的训练流程
    """
    batch_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "X_batch_*.npy")))
    if not batch_files:
        print("[错误] 未找到特征切片，请先运行 data_processor.py。")
        return None, None

    print(f"[*] 启动双模型 MAE 直优训练流程 (抛弃 Log，直接优化绝对误差)...")
    
    # 【黑科技】：将 objective 设为 reg:absoluteerror (直接优化 MAE)
    xgb_model = xgb.XGBRegressor(
        n_estimators=45, learning_rate=0.08, max_depth=8, 
        subsample=0.85, colsample_bytree=0.85, 
        objective='reg:absoluteerror',  # 核心改动：底层直接优化 MAE
        gamma=0.2, min_child_weight=3,
        random_state=42, n_jobs=-1
    )
    
    lgb_model = None
    if USE_LGB:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=45, learning_rate=0.08, max_depth=8, num_leaves=63, 
            #树深度调回 8, 增加叶子数以提升拟合能力
            subsample=0.85, colsample_bytree=0.85,
            objective='mae',                # 核心改动：底层直接优化 MAE
            min_split_gain=0.2, min_child_samples=10,
            random_state=42, n_jobs=-1
        )

    for batch_idx, x_file in enumerate(batch_files):
        y_file = x_file.replace("X_batch", "y_batch")
        X_batch = np.load(x_file)
        y_batch = np.load(y_file) # 直接使用真实的秒，不加 log
        
        print(f"    -> 正在训练 Batch {batch_idx+1}/{len(batch_files)} (样本数: {len(X_batch)})...")
        
        if batch_idx == 0:
            xgb_model.fit(X_batch, y_batch)
        else:
            xgb_model.n_estimators += 45  
            xgb_model.fit(X_batch, y_batch, xgb_model=xgb_model.get_booster())
            
        if USE_LGB:
            if batch_idx == 0:
                lgb_model.fit(X_batch, y_batch)
            else:
                lgb_model.n_estimators += 45
                lgb_model.fit(X_batch, y_batch, init_model=lgb_model.booster_)
            
    print("[*] 训练完成！双模型已生成。")
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
        xgb_model, lgb_model = train_models_in_batches_direct_mae(xgb_path, lgb_path)
        if xgb_model is None: return
        xgb_model.save_model(xgb_path)
        if USE_LGB and lgb_model is not None:
            lgb_model.booster_.save_model(lgb_path)

    od_path = os.path.join(PROCESSED_DIR, "od_matrix.pkl")
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
            f, g_o, g_d = extract_task_b_features_advanced(traj['coords'], traj['departure_timestamp'])
            hist_t = od_avg.get(f"{g_o}_{g_d}", glob_avg)
            X_feats.append(f + [hist_t])
            traj_ids.append(traj['traj_id'])
            
        X_feats_arr = np.array(X_feats)
        
        xgb_pred = xgb_model.predict(X_feats_arr)
        
        if USE_LGB and lgb_model is not None:
            lgb_pred = lgb_model.predict(X_feats_arr)
            # 根据经验，在树较深时 XGBoost 抗过拟合能力稍强，权重调整为 0.6 : 0.4
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
            print(f"    -> [MAE直优融合模型成绩] MAE: {mae:.2f} 秒 | RMSE: {rmse:.2f} 秒 | MAPE: {mape:.2f} %")
            
        out_path = os.path.join(INPUT_DIR, f"{prefix}_pred.pkl")
        with open(out_path, 'wb') as f: pickle.dump(results, f)
        print(f"[完成] 预测结果已保存至: {out_path}")

if __name__ == "__main__":
    run_task_b()