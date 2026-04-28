"""
Task B — 行程时间估计 v4
改进点：
  1. 修复 LightGBM 加载 Bug（Booster vs LGBMRegressor 不一致）
  2. 修复增量训练树数量无限增长 Bug
  3. 新增验证集最优融合权重搜索（替代硬编码 0.6/0.4）
  4. 新增完整异常捕获与兜底逻辑（Fallback）
  5. 对接新版 features_and_utils（43 维特征 + 双精度 OD）
  6. 兼容 traj_id / order_id 等多种字段名
  7. [新增] Optuna 超参搜索（--tune 模式）：树深/叶数/剪枝/正则化联合优化
  8. [新增] 融合权重优先使用 val_gt.pkl（更可靠），fallback 到训练集内部验证集
"""
import pickle
import numpy as np
import os
import sys
import warnings
from tqdm import tqdm
import xgboost as xgb
from features_and_utils import extract_task_b_features_advanced, evaluate_metrics, lookup_od_time

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
    USE_LGB = True
except ImportError:
    USE_LGB = False
    print("[警告] 未安装 LightGBM，将仅使用 XGBoost。")

PROCESSED_DIR = "processed_data_b"
INPUT_DIR     = "task_B_tte"


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def safe_get_traj_id(traj):
    """兼容多种 traj_id 字段名"""
    for key in ["traj_id", "order_id", "id", "trip_id"]:
        if key in traj:
            return traj[key]
    return str(id(traj))


def safe_extract_features(traj, knowledge_base):
    """
    带完整异常捕获的特征提取。
    任何异常都返回基于 num_points 的兜底特征（43 维）。
    """
    global_avg = knowledge_base.get("global_avg", 1200.0)
    global_med = knowledge_base.get("global_med", 1200.0)

    try:
        coords = traj.get("coords", [])
        dep_ts = traj.get("departure_timestamp", 0)

        feat, (g_o_fine, g_d_fine), (g_o_coarse, g_d_coarse) = \
            extract_task_b_features_advanced(coords, dep_ts)

        od_mean, od_med, od_std, od_cnt = lookup_od_time(
            knowledge_base, g_o_fine, g_d_fine, g_o_coarse, g_d_coarse)

        return feat + [od_mean, od_med, od_std, float(od_cnt)]

    except Exception as e:
        tid = safe_get_traj_id(traj)
        print(f"    [警告] 特征提取失败 (traj_id={tid}): {e}")
        n = len(traj.get("coords", []))
        fallback_time = max(float(n) * 15.0, global_avg)
        feat = [0.0] * 43
        feat[0]  = float(n)
        feat[4]  = float(n) * 15.0
        feat[39] = fallback_time
        feat[40] = global_med
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# 模型训练
# ─────────────────────────────────────────────────────────────────────────────

def train_models(xgb_path, lgb_path):
    """单次全量训练，直接优化 MAE（默认超参版本）"""
    x_file = os.path.join(PROCESSED_DIR, "X_train_final.npy")
    y_file = os.path.join(PROCESSED_DIR, "y_train_final.npy")

    if not os.path.exists(x_file):
        print(f"[错误] 未找到 {x_file}，请先运行 data_processor.py。")
        return None, None

    X_train = np.load(x_file)
    y_train = np.load(y_file)
    print(f"[*] 训练集：{X_train.shape[0]} 样本，{X_train.shape[1]} 维特征")

    n = len(X_train)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    split = int(n * 0.9)
    tr_idx, va_idx = idx[:split], idx[split:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train[va_idx], y_train[va_idx]

    print("[*] 训练 XGBoost (MAE 直优，early stopping)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:absoluteerror",
        gamma=0.1,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="mae",
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)
    xgb_model.save_model(xgb_path)
    print(f"    -> XGBoost 已保存（最优轮次: {xgb_model.best_iteration}）→ {xgb_path}")

    lgb_model = None
    if USE_LGB:
        print("[*] 训练 LightGBM (MAE 直优，early stopping)...")
        ds_tr = lgb.Dataset(X_tr, y_tr)
        ds_va = lgb.Dataset(X_va, y_va, reference=ds_tr)
        params = {
            "objective":         "regression_l1",
            "metric":            "mae",
            "boosting_type":     "gbdt",
            "num_leaves":        127,
            "learning_rate":     0.05,
            "feature_fraction":  0.85,
            "bagging_fraction":  0.85,
            "bagging_freq":      5,
            "min_child_samples": 10,
            "reg_alpha":         0.1,
            "reg_lambda":        1.0,
            "verbose":           -1,
            "n_jobs":            -1,
        }
        lgb_booster = lgb.train(
            params, ds_tr,
            num_boost_round=1000,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )
        lgb_booster.save_model(lgb_path)
        print(f"    -> LightGBM 已保存（最优轮次: {lgb_booster.best_iteration}）→ {lgb_path}")
        lgb_model = lgb_booster

    return xgb_model, lgb_model


def tune_hyperparams(n_trials=40):
    """
    使用 Optuna 对 XGBoost + LightGBM 超参进行贝叶斯搜索。
    结果保存到 processed_data_b/best_params.pkl，供 train_models_with_params 使用。
    用法：python task_b_main_new.py --tune [轮数]
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[错误] 请先安装 Optuna：pip install optuna")
        return None

    x_file = os.path.join(PROCESSED_DIR, "X_train_final.npy")
    y_file = os.path.join(PROCESSED_DIR, "y_train_final.npy")
    if not os.path.exists(x_file):
        print(f"[错误] 未找到 {x_file}，请先运行 data_processor.py。")
        return None

    X = np.load(x_file)
    y = np.load(y_file)
    n = len(X)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    split = int(n * 0.85)
    X_tr, y_tr = X[idx[:split]], y[idx[:split]]
    X_va, y_va = X[idx[split:]], y[idx[split:]]

    print(f"[*] Optuna 超参搜索：{n_trials} 轮，训练集 {len(X_tr)} / 验证集 {len(X_va)}")

    def xgb_objective(trial):
        p = dict(
            n_estimators          = 1000,
            learning_rate         = trial.suggest_float("xgb_lr",     0.02, 0.15, log=True),
            max_depth             = trial.suggest_int("xgb_depth",     4,   10),
            subsample             = trial.suggest_float("xgb_sub",     0.6,  1.0),
            colsample_bytree      = trial.suggest_float("xgb_col",     0.6,  1.0),
            gamma                 = trial.suggest_float("xgb_gamma",   0.0,  2.0),
            min_child_weight      = trial.suggest_int("xgb_mcw",       1,   20),
            reg_alpha             = trial.suggest_float("xgb_alpha",   0.0,  2.0),
            reg_lambda            = trial.suggest_float("xgb_lambda",  0.5,  5.0),
            objective             = "reg:absoluteerror",
            eval_metric           = "mae",
            early_stopping_rounds = 40,
            random_state          = 42,
            n_jobs                = -1,
        )
        m = xgb.XGBRegressor(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return np.mean(np.abs(y_va - m.predict(X_va)))

    print("[*] 搜索 XGBoost 超参...")
    xgb_study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=True)
    xgb_best = xgb_study.best_params
    print(f"    -> XGBoost 最优 MAE: {xgb_study.best_value:.2f}s  params: {xgb_best}")

    lgb_best = None
    if USE_LGB:
        def lgb_objective(trial):
            p = dict(
                objective         = "regression_l1",
                metric            = "mae",
                boosting_type     = "gbdt",
                learning_rate     = trial.suggest_float("lgb_lr",     0.02, 0.15, log=True),
                num_leaves        = trial.suggest_int("lgb_leaves",    31,  255),
                max_depth         = trial.suggest_int("lgb_depth",      4,   12),
                subsample         = trial.suggest_float("lgb_sub",      0.6,  1.0),
                feature_fraction  = trial.suggest_float("lgb_col",      0.6,  1.0),
                min_child_samples = trial.suggest_int("lgb_mcs",        5,   50),
                reg_alpha         = trial.suggest_float("lgb_alpha",    0.0,  2.0),
                reg_lambda        = trial.suggest_float("lgb_lambda",   0.5,  5.0),
                bagging_freq      = 5,
                verbose           = -1,
                n_jobs            = -1,
            )
            ds_tr2 = lgb.Dataset(X_tr, y_tr)
            ds_va2 = lgb.Dataset(X_va, y_va, reference=ds_tr2)
            b = lgb.train(p, ds_tr2, num_boost_round=1000, valid_sets=[ds_va2],
                          callbacks=[lgb.early_stopping(40, verbose=False),
                                     lgb.log_evaluation(-1)])
            return np.mean(np.abs(y_va - b.predict(X_va)))

        print("[*] 搜索 LightGBM 超参...")
        lgb_study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=42))
        lgb_study.optimize(lgb_objective, n_trials=n_trials, show_progress_bar=True)
        lgb_best = lgb_study.best_params
        print(f"    -> LightGBM 最优 MAE: {lgb_study.best_value:.2f}s  params: {lgb_best}")

    best_params = {"xgb": xgb_best, "lgb": lgb_best}
    params_path = os.path.join(PROCESSED_DIR, "best_params.pkl")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(params_path, "wb") as f:
        pickle.dump(best_params, f)
    print(f"[*] 最优超参已保存 → {params_path}")
    return best_params


def train_models_with_params(xgb_path, lgb_path, best_params=None):
    """使用 Optuna 最优超参（或默认值）训练模型"""
    x_file = os.path.join(PROCESSED_DIR, "X_train_final.npy")
    y_file = os.path.join(PROCESSED_DIR, "y_train_final.npy")
    if not os.path.exists(x_file):
        print(f"[错误] 未找到 {x_file}，请先运行 data_processor.py。")
        return None, None

    if best_params is None:
        params_path = os.path.join(PROCESSED_DIR, "best_params.pkl")
        if os.path.exists(params_path):
            with open(params_path, "rb") as f:
                best_params = pickle.load(f)
            print(f"[*] 已加载 Optuna 最优超参 → {params_path}")

    X_train = np.load(x_file)
    y_train = np.load(y_file)
    print(f"[*] 训练集：{X_train.shape[0]} 样本，{X_train.shape[1]} 维特征")

    n = len(X_train)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    split = int(n * 0.9)
    tr_idx, va_idx = idx[:split], idx[split:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train[va_idx], y_train[va_idx]

    xgb_p = (best_params or {}).get("xgb", {}) or {}
    print("[*] 训练 XGBoost (MAE 直优，early stopping)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators          = 1000,
        learning_rate         = xgb_p.get("xgb_lr",     0.05),
        max_depth             = xgb_p.get("xgb_depth",  8),
        subsample             = xgb_p.get("xgb_sub",    0.85),
        colsample_bytree      = xgb_p.get("xgb_col",    0.85),
        gamma                 = xgb_p.get("xgb_gamma",  0.1),
        min_child_weight      = xgb_p.get("xgb_mcw",    5),
        reg_alpha             = xgb_p.get("xgb_alpha",  0.1),
        reg_lambda            = xgb_p.get("xgb_lambda", 1.0),
        objective             = "reg:absoluteerror",
        eval_metric           = "mae",
        early_stopping_rounds = 50,
        random_state          = 42,
        n_jobs                = -1,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)
    xgb_model.save_model(xgb_path)
    print(f"    -> XGBoost 已保存（最优轮次: {xgb_model.best_iteration}）→ {xgb_path}")

    lgb_model = None
    if USE_LGB:
        lgb_p = (best_params or {}).get("lgb", {}) or {}
        print("[*] 训练 LightGBM (MAE 直优，early stopping)...")
        params = {
            "objective":         "regression_l1",
            "metric":            "mae",
            "boosting_type":     "gbdt",
            "learning_rate":     lgb_p.get("lgb_lr",     0.05),
            "num_leaves":        lgb_p.get("lgb_leaves", 127),
            "max_depth":         lgb_p.get("lgb_depth",  8),
            "feature_fraction":  lgb_p.get("lgb_col",    0.85),
            "bagging_fraction":  lgb_p.get("lgb_sub",    0.85),
            "bagging_freq":      5,
            "min_child_samples": lgb_p.get("lgb_mcs",    10),
            "reg_alpha":         lgb_p.get("lgb_alpha",  0.1),
            "reg_lambda":        lgb_p.get("lgb_lambda", 1.0),
            "verbose":           -1,
            "n_jobs":            -1,
        }
        ds_tr = lgb.Dataset(X_tr, y_tr)
        ds_va = lgb.Dataset(X_va, y_va, reference=ds_tr)
        lgb_model = lgb.train(
            params, ds_tr, num_boost_round=1000, valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )
        lgb_model.save_model(lgb_path)
        print(f"    -> LightGBM 已保存（最优轮次: {lgb_model.best_iteration}）→ {lgb_path}")

    return xgb_model, lgb_model


def find_best_blend_weight(xgb_model, lgb_model, X_val, y_val):
    """在验证集上搜索最优融合权重 w（xgb×w + lgb×(1-w)），步长 0.02"""
    xgb_pred = xgb_model.predict(X_val)
    lgb_pred = lgb_model.predict(X_val)

    best_w, best_mae = 0.5, float("inf")
    for w in np.arange(0.0, 1.01, 0.02):
        blended = w * xgb_pred + (1 - w) * lgb_pred
        mae = np.mean(np.abs(y_val - blended))
        if mae < best_mae:
            best_mae = mae
            best_w = round(float(w), 2)

    print(f"    -> 最优融合权重：XGB={best_w:.2f} / LGB={1-best_w:.2f}（MAE={best_mae:.2f}s）")
    return best_w


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_task_b():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    xgb_path = os.path.join(PROCESSED_DIR, "xgb_model_v3.json")
    lgb_path = os.path.join(PROCESSED_DIR, "lgb_model_v3.txt")

    # ── 1. 加载或训练模型 ────────────────────────────────────────
    xgb_model, lgb_model = None, None

    if os.path.exists(xgb_path):
        print("[*] 发现已保存模型，直接加载（跳过训练）...")
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(xgb_path)
        if USE_LGB and os.path.exists(lgb_path):
            lgb_model = lgb.Booster(model_file=lgb_path)
            print(f"    -> LightGBM Booster 已加载 → {lgb_path}")
    else:
        print("[*] 未发现已保存模型，开始训练（自动读取 Optuna 超参）...")
        xgb_model, lgb_model = train_models_with_params(xgb_path, lgb_path)
        if xgb_model is None:
            return

    # ── 2. 加载知识库 ────────────────────────────────────────────
    od_path = os.path.join(PROCESSED_DIR, "od_matrix_org.pkl")
    if not os.path.exists(od_path):
        print(f"[错误] 未找到 {od_path}，请先运行 data_processor.py。")
        return

    with open(od_path, "rb") as f:
        knowledge_base = pickle.load(f)
    print(f"[*] 知识库已加载：精细 OD {len(knowledge_base.get('od_fine', {}))} 对，"
          f"粗粒度 OD {len(knowledge_base.get('od_coarse', {}))} 对")

    # ── 3. 搜索最优融合权重 ──────────────────────────────────────
    blend_w = 0.5
    if USE_LGB and lgb_model is not None:
        val_input_path = os.path.join(INPUT_DIR, "val_input.pkl")
        val_gt_path    = os.path.join(INPUT_DIR, "val_gt.pkl")

        if os.path.exists(val_input_path) and os.path.exists(val_gt_path):
            print("[*] 使用 val_gt.pkl 搜索最优融合权重（最可靠）...")
            with open(val_input_path, "rb") as f:
                val_data = pickle.load(f)
            with open(val_gt_path, "rb") as f:
                gt_data = pickle.load(f)
            gt_dict = {item["traj_id"]: item["travel_time"] for item in gt_data}
            X_ws, y_ws = [], []
            for traj in val_data:
                tid = safe_get_traj_id(traj)
                if tid in gt_dict:
                    X_ws.append(safe_extract_features(traj, knowledge_base))
                    y_ws.append(gt_dict[tid])
            if X_ws:
                blend_w = find_best_blend_weight(
                    xgb_model, lgb_model,
                    np.array(X_ws, dtype=np.float32),
                    np.array(y_ws, dtype=np.float32),
                )
        else:
            x_file = os.path.join(PROCESSED_DIR, "X_train_final.npy")
            y_file = os.path.join(PROCESSED_DIR, "y_train_final.npy")
            if os.path.exists(x_file):
                X_all = np.load(x_file)
                y_all = np.load(y_file)
                n = len(X_all)
                X_ws = X_all[int(n * 0.9):]
                y_ws = y_all[int(n * 0.9):]
                print("[*] 使用训练集内部验证集搜索融合权重（fallback）...")
                blend_w = find_best_blend_weight(xgb_model, lgb_model, X_ws, y_ws)

    # ── 4. 推理 ──────────────────────────────────────────────────
    for prefix in ["val", "test"]:
        input_path = os.path.join(INPUT_DIR, f"{prefix}_input.pkl")
        if not os.path.exists(input_path):
            continue

        print(f"\n[*] 处理 {prefix}_input.pkl ...")
        with open(input_path, "rb") as f:
            data = pickle.load(f)

        X_feats, traj_ids = [], []
        for traj in tqdm(data, desc=f"提取特征({prefix})", unit="条", colour="green"):
            X_feats.append(safe_extract_features(traj, knowledge_base))
            traj_ids.append(safe_get_traj_id(traj))

        X_arr = np.array(X_feats, dtype=np.float32)
        xgb_pred = xgb_model.predict(X_arr)

        if USE_LGB and lgb_model is not None:
            lgb_pred = lgb_model.predict(X_arr)
            final_pred = blend_w * xgb_pred + (1 - blend_w) * lgb_pred
            print(f"    -> 双模型融合（XGB×{blend_w:.2f} + LGB×{1-blend_w:.2f}）")
        else:
            final_pred = xgb_pred
            print("    -> 单模型 XGBoost 推理")

        results = [{"traj_id": tid, "travel_time": max(60.0, round(float(t), 2))}
                   for tid, t in zip(traj_ids, final_pred)]

        # ── 5. 评估（仅 val 集）────────────────────────────────
        if prefix == "val":
            gt_path = os.path.join(INPUT_DIR, "val_gt.pkl")
            if os.path.exists(gt_path):
                with open(gt_path, "rb") as f:
                    gt_data = pickle.load(f)
                gt_dict = {item["traj_id"]: item["travel_time"] for item in gt_data}
                y_true = [gt_dict.get(tid, 0) for tid in traj_ids]
                y_pred_eval = [r["travel_time"] for r in results]
                mae, rmse, mape = evaluate_metrics(y_true, y_pred_eval)
                print(f"MAE : {mae:>8.2f} 秒，RMSE: {rmse:>8.2f} 秒，MAPE: {mape:>8.2f} %")

        # ── 6. 保存结果 ─────────────────────────────────────────
        out_path = os.path.join(INPUT_DIR, f"{prefix}_pred.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(results, f)

        os.makedirs("submissions", exist_ok=True)
        submit_path = os.path.join("submissions", f"task_B_{prefix}_pred.pkl")
        with open(submit_path, "wb") as f:
            pickle.dump(results, f)

        print(f"[完成] 预测结果已保存 → {out_path}")
        print(f"[完成] 提交文件已保存 → {submit_path}")


if __name__ == "__main__":
    print("=" * 55)
    print("Task B — 行程时间估计 v4（增强特征 + 双精度OD + Optuna调参）")
    print("=" * 55)
    print("用法：")
    print("  python task_b_main_new.py            # 正常训练/推理")
    print("  python task_b_main_new.py --tune     # Optuna 超参搜索后训练（默认40轮）")
    print("  python task_b_main_new.py --tune 60  # 指定搜索轮数")
    print("-" * 55)

    if "--tune" in sys.argv:
        n_trials = 40
        for arg in sys.argv[1:]:
            try:
                n_trials = int(arg)
                break
            except ValueError:
                pass
        print(f"[模式] Optuna 超参搜索（{n_trials} 轮）→ 保存最优超参 → 训练最优模型")
        best_params = tune_hyperparams(n_trials=n_trials)
        if best_params is not None:
            for p in [os.path.join(PROCESSED_DIR, "xgb_model_v3.json"),
                      os.path.join(PROCESSED_DIR, "lgb_model_v3.txt")]:
                if os.path.exists(p):
                    os.remove(p)
            run_task_b()
    else:
        run_task_b()
