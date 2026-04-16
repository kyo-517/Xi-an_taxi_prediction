import pickle
import numpy as np
import os
import glob
from features_and_utils import extract_task_b_features_advanced, evaluate_metrics
import xgboost as xgb

try:
    import lightgbm as lgb
    USE_LGB = True
    print("[系統] 已成功檢測到 LightGBM，將啟動 XGBoost + LightGBM 雙模型加權融合 (Weighted Blending)。")
except ImportError:
    USE_LGB = False
    print("[警告] 未安裝 LightGBM (建議 pip install lightgbm)。本次運行將僅使用 XGBoost。")

PROCESSED_DIR = "processed_data"

def train_models_in_batches(xgb_path, lgb_path):
    """
    分批次增量訓練 XGBoost 和 LightGBM 雙模型 (支持斷點續訓與高級剪枝機制)
    """
    batch_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "X_batch_*.npy")))
    if not batch_files:
        print("[錯誤] 未找到特徵切片，請先運行 data_processor.py。")
        return None, None

    print(f"[*] 發現 {len(batch_files)} 個數據切片，準備啟動雙模型分批次增量訓練...")
    
    # ==============================================================
    # 引入 XGBoost 剪枝與正則化機制 (Pruning Mechanisms)
    # ==============================================================
    xgb_model = xgb.XGBRegressor(
        n_estimators=30, 
        learning_rate=0.05, 
        max_depth=8,
        subsample=0.8, 
        colsample_bytree=0.8, 
        # [後剪枝] 損失下降小於2.0的分支將被無情剪除，防止過擬合
        gamma=2.0, 
        # [預剪枝] 要求葉子節點的權重總和必須大於10，避免對異常值敏感
        min_child_weight=10, 
        # [L1/L2剪枝] 加入正則化懲罰項，使樹結構更簡潔稀疏
        reg_alpha=0.5, 
        reg_lambda=1.5,
        random_state=42, 
        n_jobs=-1
    )
    
    xgb_booster = None
    if os.path.exists(xgb_path):
        print("[*] 檢測到本地已保存的 XGBoost 模型，將讀取為初始權重繼續追加訓練...")
        xgb_loaded = xgb.XGBRegressor()
        xgb_loaded.load_model(xgb_path)
        xgb_booster = xgb_loaded.get_booster()
        # 設置正確的起始樹數量，防止 Scikit-learn API 覆蓋原有樹
        current_trees = len(xgb_booster.get_dump())
        xgb_model.n_estimators = current_trees + 30
        
    # ==============================================================
    # 引入 LightGBM 剪枝與正則化機制 (Pruning Mechanisms)
    # ==============================================================
    lgb_model = None
    lgb_init = None
    if USE_LGB:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=30, 
            learning_rate=0.05, 
            max_depth=8, 
            # 控制葉子數量，與 max_depth 配合使用防止樹過深
            num_leaves=63, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            # [後剪枝] 降低至 0.5 避免過度剪枝導致無法分裂
            min_split_gain=0.5, 
            # [預剪枝] 每個葉子節點最少需要包含 10 個樣本
            min_child_samples=10, 
            # [L1/L2剪枝] 降低正則化強度
            reg_alpha=0.1,
            reg_lambda=0.5,
            verbose=-1,  # 抑制日誌
            random_state=42, 
            n_jobs=-1
        )
        if os.path.exists(lgb_path):
            print("[*] 檢測到本地已保存的 LightGBM 模型，將讀取為初始權重繼續追加訓練...")
            booster = lgb.Booster(model_file=lgb_path)
            current_trees = booster.num_trees()
            lgb_model.n_estimators = current_trees + 30
            lgb_init = booster

    for batch_idx, x_file in enumerate(batch_files):
        y_file = x_file.replace("X_batch", "y_batch")
        X_batch = np.load(x_file)
        y_batch = np.load(y_file)
        
        print(f"    -> 正在訓練 Batch {batch_idx+1}/{len(batch_files)} (樣本數: {len(X_batch)})...")
        
        # =============== XGBoost 追加訓練 ===============
        if xgb_booster is not None:
            xgb_model.fit(X_batch, y_batch, xgb_model=xgb_booster)
        else:
            xgb_model.fit(X_batch, y_batch)
        xgb_booster = xgb_model.get_booster()
        xgb_model.n_estimators += 30  
            
        # =============== LightGBM 追加訓練 ===============
        if USE_LGB:
            if lgb_init is not None:
                lgb_model.fit(X_batch, y_batch, init_model=lgb_init)
            else:
                lgb_model.fit(X_batch, y_batch)
            lgb_init = lgb_model.booster_
            lgb_model.n_estimators += 30
            
    print("[*] 增量訓練與剪枝過程全部完成！雙模型已完美擬合全量數據。")
    return xgb_model, lgb_model

def run_task_b():
    xgb_path = os.path.join(PROCESSED_DIR, "xgb_model.json")
    lgb_path = os.path.join(PROCESSED_DIR, "lgb_model.txt")
    
    # 1. 每次運行都會加載本地模型（如果有）並繼續追加訓練
    xgb_model, lgb_model = train_models_in_batches(xgb_path, lgb_path)
    if xgb_model is None: return
    
    # 訓練完成後，持久化保存更新後的模型
    xgb_model.save_model(xgb_path)
    print(f"    -> 追加訓練後的 XGBoost 模型已保存覆蓋至: {xgb_path}")
    if USE_LGB and lgb_model is not None:
        lgb_model.booster_.save_model(lgb_path)
        print(f"    -> 追加訓練後的 LightGBM 模型已保存覆蓋至: {lgb_path}")

    # 2. 獲取在 Data Processor 中提前建好的全局 OD 知識庫
    od_matrix_path = os.path.join(PROCESSED_DIR, "od_matrix.pkl")
    with open(od_matrix_path, 'rb') as f:
        od_data = pickle.load(f)
    od_avg_time = od_data['od_avg']
    global_avg = od_data['global_avg']
    
    # 3. 對驗證集進行預測
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
        
    print("\n[*] 正在執行驗證集推理預估...")
    X_val_arr = np.array(X_val)
    
    # 【核心機制】：模型加權融合 (Weighted Blending)
    xgb_pred = xgb_model.predict(X_val_arr)
    
    if USE_LGB and lgb_model is not None:
        lgb_pred = lgb_model.predict(X_val_arr)
        # XGBoost 和 LightGBM 採用 6:4 的黃金分割權重
        final_pred = 0.6 * xgb_pred + 0.4 * lgb_pred
        print("    -> 成功應用雙模型融合 (XGB 0.6 + LGB 0.4)")
    else:
        final_pred = xgb_pred
        print("    -> 僅應用 XGBoost 單模型推理")
    
    pred_results = []
    for tid, pred_time in zip(val_traj_ids, final_pred):
        pred_results.append({
            'traj_id': tid,
            'travel_time': max(1.0, round(float(pred_time), 2)) 
        })
        
    # 4. 評估結果
    val_gt_file = os.path.join("task_B_tte", "val_gt.pkl")
    if os.path.exists(val_gt_file):
        with open(val_gt_file, 'rb') as f:
            gt_data = pickle.load(f)
        gt_dict = {item['traj_id']: item['travel_time'] for item in gt_data}
        y_true = [gt_dict[tid] for tid in val_traj_ids]
        y_pred_eval = [item['travel_time'] for item in pred_results]
        
        mae, rmse, mape = evaluate_metrics(y_true, y_pred_eval)
        print(f"    -> [高階融合模型評測] MAE: {mae:.2f} 秒 | RMSE: {rmse:.2f} 秒 | MAPE: {mape:.2f} %")
        
    with open(os.path.join("task_B_tte", "val_pred.pkl"), 'wb') as f:
        pickle.dump(pred_results, f)

if __name__ == "__main__":
    print("="*50)
    print("啟動任務 B：基於 Geohash-OD 與 雙模型加權融合 (引入剪枝機制)")
    print("="*50)
    run_task_b()