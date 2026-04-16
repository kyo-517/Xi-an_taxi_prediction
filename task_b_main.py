import pickle
import numpy as np
import os
import glob
from features_and_utils import extract_task_b_features_advanced, evaluate_metrics
import xgboost as xgb

PROCESSED_DIR = "processed_data"

def train_model_in_batches():
    """
    [核心机制] 增量学习 (Incremental Learning):
    不需要一次性将百GB数据载入内存，通过设定较小的每批树数量(n_estimators)，
    利用 xgb_model=model_booster 参数将上一批训练好的残差树传递给下一批，实现不断进化。
    """
    batch_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "X_batch_*.npy")))
    if not batch_files:
        print("[错误] 未找到特征切片，请先运行 data_processor.py。")
        return None

    print(f"[*] 发现 {len(batch_files)} 个数据切片，准备启动分批次增量训练 (Incremental Learning)...")
    
    model = None
    for batch_idx, x_file in enumerate(batch_files):
        y_file = x_file.replace("X_batch", "y_batch")
        
        X_batch = np.load(x_file)
        y_batch = np.load(y_file)
        
        print(f"    -> 正在训练 Batch {batch_idx+1}/{len(batch_files)} (样本数: {len(X_batch)})...")
        
        if model is None:
            # 首次初始化模型：每批次追加 30 棵树
            model = xgb.XGBRegressor(
                n_estimators=30,  
                learning_rate=0.05, 
                max_depth=8,
                subsample=0.8, 
                colsample_bytree=0.8, 
                random_state=42, 
                n_jobs=-1
            )
            model.fit(X_batch, y_batch)
        else:
            # 增量训练：利用已训练的 booster 继续添加新的残差树
            model.n_estimators += 30  
            model.fit(X_batch, y_batch, xgb_model=model.get_booster())
            
    print("[*] 增量训练全部完成！模型已拟合全量数据。")
    return model

def run_task_b():
    # 1. 训练模型 (基于已保存的批次特征)
    model = train_model_in_batches()
    if model is None: return

    # 2. 获取在 Data Processor 中提前建好的全局 OD 知识库
    od_matrix_path = os.path.join(PROCESSED_DIR, "od_matrix.pkl")
    with open(od_matrix_path, 'rb') as f:
        od_data = pickle.load(f)
    od_avg_time = od_data['od_avg']
    global_avg = od_data['global_avg']
    
    # 3. 对验证集进行预测
    val_input_file = os.path.join("task_B_tte", "val_input.pkl")
    with open(val_input_file, 'rb') as f:
        val_input = pickle.load(f)
        
    X_val, val_traj_ids = [], []
    for traj in val_input:
        base_feat, grid_o, grid_d = extract_task_b_features_advanced(traj['coords'], traj['departure_timestamp'])
        od_key = f"{grid_o}_{grid_d}"
        hist_time = od_avg_time.get(od_key, global_avg)
        
        final_feat = base_feat + [hist_time]
        X_val.append(final_feat)
        val_traj_ids.append(traj['traj_id'])
        
    print("\n[*] 正在执行验证集推理预估...")
    y_pred = model.predict(np.array(X_val))
    
    pred_results = []
    for tid, pred_time in zip(val_traj_ids, y_pred):
        pred_results.append({
            'traj_id': tid,
            'travel_time': max(1.0, round(float(pred_time), 2)) 
        })
        
    # 4. 评估结果
    val_gt_file = os.path.join("task_B_tte", "val_gt.pkl")
    if os.path.exists(val_gt_file):
        with open(val_gt_file, 'rb') as f:
            gt_data = pickle.load(f)
        gt_dict = {item['traj_id']: item['travel_time'] for item in gt_data}
        y_true = [gt_dict[tid] for tid in val_traj_ids]
        y_pred_eval = [item['travel_time'] for item in pred_results]
        
        mae, rmse, mape = evaluate_metrics(y_true, y_pred_eval)
        print(f"    -> [高阶批次训练模型评测] MAE: {mae:.2f} 秒 | RMSE: {rmse:.2f} 秒 | MAPE: {mape:.2f} %")
        
    with open(os.path.join("task_B_tte", "val_pred.pkl"), 'wb') as f:
        pickle.dump(pred_results, f)

if __name__ == "__main__":
    print("="*50)
    print("启动任务 B：基于 Geohash-OD 与 分批次增量学习 (Incremental Learning)")
    print("="*50)
    run_task_b()