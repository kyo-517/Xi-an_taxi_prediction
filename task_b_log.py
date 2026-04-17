import pickle
import numpy as np
import os
import glob
import xgboost as xgb
import warnings
from tqdm import tqdm
from features_and_utils import extract_task_b_features_advanced, evaluate_metrics

# 屏蔽无害的特征名称警告
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
    USE_LGB = True
except ImportError:
    USE_LGB = False

# 路径配置
PROCESSED_DIR = "processed_data"
INPUT_DIR = "task_B_tte"

def train_models_in_batches_with_log(xgb_path, lgb_path):
    """
    带对数变换 (Log Transform) 的分批次训练。
    """
    batch_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "X_batch_*.npy")))
    if not batch_files:
        print("[错误] 未找到特征切片，请先运行 data_processor.py。")
        return None, None

    print(f"[*] 未发现历史模型，正在启动双模型全新训练流程 (引入 Log1p 变换)...")
    
    # 【终极微调】：放宽剪枝枷锁，释放拟合潜力
    xgb_model = xgb.XGBRegressor(
        n_estimators=30, learning_rate=0.08, max_depth=8, # 深度调回 8
        subsample=0.85, colsample_bytree=0.85, 
        gamma=0.2, min_child_weight=3,                    # 放宽剪枝要求
        random_state=42, n_jobs=-1
    )
    
    lgb_model = None
    if USE_LGB:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=30, learning_rate=0.08, max_depth=8, num_leaves=63, # 深度调回 8, 增加叶子数
            subsample=0.85, colsample_bytree=0.85,
            min_split_gain=0.2, min_child_samples=10,                        # 放宽剪枝要求
            random_state=42, n_jobs=-1
        )

    for batch_idx, x_file in enumerate(batch_files):
        y_file = x_file.replace("X_batch", "y_batch")
        X_batch = np.load(x_file)
        y_batch = np.load(y_file)
        
        # 核心：对目标时间进行 log(1+x) 变换
        y_batch_log = np.log1p(y_batch)
        
        print(f"    -> 正在训练 Batch {batch_idx+1}/{len(batch_files)} (样本数: {len(X_batch)})...")
        
        # XGBoost 续批次
        if batch_idx == 0:
            xgb_model.fit(X_batch, y_batch_log)
        else:
            xgb_model.n_estimators += 30  
            xgb_model.fit(X_batch, y_batch_log, xgb_model=xgb_model.get_booster())
            
        # LightGBM 续批次
        if USE_LGB:
            if batch_idx == 0:
                lgb_model.fit(X_batch, y_batch_log)
            else:
                lgb_model.n_estimators += 30
                lgb_model.fit(X_batch, y_batch_log, init_model=lgb_model.booster_)
            
    print("[*] 训练完成！双模型已生成。")
    return xgb_model, lgb_model

def run_task_b():
    xgb_path = os.path.join(PROCESSED_DIR, "xgb_model.json")
    lgb_path = os.path.join(PROCESSED_DIR, "lgb_model.txt")
    
    xgb_model, lgb_model = None, None
    
    # 改进：如果存在模型，只读取不训练（防止重复跑导致过拟合）
    if os.path.exists(xgb_path):
        print("[*] 发现本地持久化模型，直接加载进入推理模式 (跳过训练防止过拟合)...")
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(xgb_path)
        if USE_LGB and os.path.exists(lgb_path):
            lgb_model = lgb.Booster(model_file=lgb_path)
    else:
        xgb_model, lgb_model = train_models_in_batches_with_log(xgb_path, lgb_path)
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
        
        xgb_pred_log = xgb_model.predict(X_feats_arr)
        
        if USE_LGB and lgb_model is not None:
            lgb_pred_log = lgb_model.predict(X_feats_arr)
            # 根据经验，在树较深时 XGBoost 抗过拟合能力稍强，权重调整为 0.7 : 0.3
            final_pred_log = 0.7 * xgb_pred_log + 0.3 * lgb_pred_log
        else:
            final_pred_log = xgb_pred_log
            
        # 还原对数变换
        final_pred_time = np.expm1(final_pred_log)
        
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
            print(f"    -> [终极融合模型成绩] MAE: {mae:.2f} 秒 | RMSE: {rmse:.2f} 秒 | MAPE: {mape:.2f} %")
            
        out_path = os.path.join(INPUT_DIR, f"{prefix}_pred.pkl")
        with open(out_path, 'wb') as f: pickle.dump(results, f)
        print(f"[完成] 预测结果已保存至: {out_path}")

if __name__ == "__main__":
    run_task_b()