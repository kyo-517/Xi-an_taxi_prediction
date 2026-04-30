"""
Task B — 行程时间估计 v4
所有 Task B 专用逻辑（特征提取、OD查询、模型训练）均自包含在本文件中。
不修改 features_and_utils.py，仅从中借用 haversine / get_grid_id / evaluate_metrics。
"""
import math
import pickle
import datetime
import numpy as np
import os
import sys
import warnings
from tqdm import tqdm
import xgboost as xgb
from features_and_utils import haversine, get_grid_id, evaluate_metrics

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
    USE_LGB = True
except ImportError:
    USE_LGB = False
    print("[警告] 未安装 LightGBM，将仅使用 XGBoost。")

PROCESSED_DIR = "processed_data_b"
INPUT_DIR     = "task_B_tte"

# 西安钟楼坐标
_XIAN_LON = 108.940
_XIAN_LAT = 34.265


# ─────────────────────────────────────────────────────────────────────────────
# Task B 专用工具函数（不污染 features_and_utils.py）
# ─────────────────────────────────────────────────────────────────────────────

def _bearing(lon1, lat1, lon2, lat2):
    """计算从点1到点2的方位角（弧度，[-π, π]）"""
    dl = math.radians(lon2 - lon1)
    p1, p2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.atan2(x, y)


def extract_features_v2(coords, departure_timestamp):
    """
    增强版特征提取（39维），完全自包含，不修改 features_and_utils.py。
    返回：(features_list, grid_o_fine, grid_d_fine, grid_o_coarse, grid_d_coarse)
    """
    coords = np.array(coords, dtype=np.float64)
    valid_mask = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
    coords = coords[valid_mask]
    num_points = len(coords)

    if num_points < 2:
        return [0.0] * 39, "unknown", "unknown", "unknown", "unknown"

    lon_start, lat_start = coords[0]
    lon_end,   lat_end   = coords[-1]

    # ── 基础距离 ──────────────────────────────────────────────
    straight_dist = haversine(lon_start, lat_start, lon_end, lat_end)
    seg_dists = []
    for i in range(1, num_points):
        d = haversine(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
        seg_dists.append(d if not np.isnan(d) else 0.0)
    total_dist = sum(seg_dists)
    sinuosity  = total_dist / straight_dist if straight_dist > 1.0 else 1.0

    # ── 速度特征（假设采样间隔约 15s）────────────────────────
    dt_step = 15.0
    speeds      = [d / dt_step for d in seg_dists]
    avg_speed   = float(np.mean(speeds)) if speeds else 0.0
    std_speed   = float(np.std(speeds))  if speeds else 0.0
    max_speed   = float(np.max(speeds))  if speeds else 0.0
    n_seg       = max(1, len(speeds) // 3)
    speed_first  = float(np.mean(speeds[:n_seg]))        if speeds[:n_seg]        else avg_speed
    speed_middle = float(np.mean(speeds[n_seg:2*n_seg])) if speeds[n_seg:2*n_seg] else avg_speed
    speed_last   = float(np.mean(speeds[2*n_seg:]))      if speeds[2*n_seg:]      else avg_speed
    speed_trend  = speed_last - speed_first

    # ── 转向角特征 ────────────────────────────────────────────
    bearings = [_bearing(coords[i-1][0], coords[i-1][1],
                         coords[i][0],   coords[i][1])
                for i in range(1, num_points)]
    turn_angles = []
    for i in range(1, len(bearings)):
        diff = abs(bearings[i] - bearings[i-1])
        diff = min(diff, 2 * math.pi - diff)
        turn_angles.append(diff)
    if turn_angles:
        mean_turn   = float(np.mean(turn_angles))
        std_turn    = float(np.std(turn_angles))
        sharp_turns = sum(1 for t in turn_angles if t > math.pi / 4)
    else:
        mean_turn, std_turn, sharp_turns = 0.0, 0.0, 0

    # ── 边界框特征 ────────────────────────────────────────────
    bbox_lon  = float(coords[:, 0].max() - coords[:, 0].min())
    bbox_lat  = float(coords[:, 1].max() - coords[:, 1].min())
    bbox_area = bbox_lon * bbox_lat

    # ── 时间特征 ──────────────────────────────────────────────
    dt_obj  = datetime.datetime.fromtimestamp(departure_timestamp)
    hour    = dt_obj.hour
    minute  = dt_obj.minute
    weekday = dt_obj.weekday()

    hour_sin    = math.sin(2 * math.pi * hour    / 24)
    hour_cos    = math.cos(2 * math.pi * hour    / 24)
    minute_sin  = math.sin(2 * math.pi * minute  / 60)
    minute_cos  = math.cos(2 * math.pi * minute  / 60)
    weekday_sin = math.sin(2 * math.pi * weekday / 7)
    weekday_cos = math.cos(2 * math.pi * weekday / 7)

    is_rush_hour    = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    is_night        = 1 if hour < 6 or hour >= 22 else 0
    is_weekend      = 1 if weekday >= 5 else 0
    is_national_day = 1 if (dt_obj.month == 10 and 1 <= dt_obj.day <= 7) else 0

    # ── 空间上下文 ────────────────────────────────────────────
    dist_center_start = haversine(lon_start, lat_start, _XIAN_LON, _XIAN_LAT)
    dist_center_end   = haversine(lon_end,   lat_end,   _XIAN_LON, _XIAN_LAT)
    od_bear           = _bearing(lon_start, lat_start, lon_end, lat_end)
    baseline_time_est = num_points * 15.0

    # ── OD 网格 ID（双精度）──────────────────────────────────
    g_o_fine   = get_grid_id(lon_start, lat_start, precision=6)
    g_d_fine   = get_grid_id(lon_end,   lat_end,   precision=6)
    g_o_coarse = get_grid_id(lon_start, lat_start, precision=5)
    g_d_coarse = get_grid_id(lon_end,   lat_end,   precision=5)

    features = [
        # 基础 (0-4)
        float(num_points), total_dist, straight_dist, sinuosity, baseline_time_est,
        # 速度 (5-11)
        avg_speed, std_speed, max_speed,
        speed_first, speed_middle, speed_last, speed_trend,
        # 转向角 (12-14)
        mean_turn, std_turn, float(sharp_turns),
        # 边界框 (15-17)
        bbox_lon, bbox_lat, bbox_area,
        # 时间原始值 (18-20)
        float(hour), float(weekday), float(minute),
        # 时间周期编码 (21-26)
        hour_sin, hour_cos, minute_sin, minute_cos, weekday_sin, weekday_cos,
        # 时间标志位 (27-30)
        float(is_rush_hour), float(is_night), float(is_weekend), float(is_national_day),
        # 空间上下文 (31-34)
        dist_center_start, dist_center_end, od_bear, math.sin(od_bear),
        # 坐标 (35-38)
        lon_start, lat_start, lon_end, lat_end,
    ]
    return features, g_o_fine, g_d_fine, g_o_coarse, g_d_coarse


def lookup_od_time(knowledge_base, g_o_fine, g_d_fine, g_o_coarse, g_d_coarse):
    """双层 OD 查询：精细 → 粗粒度 → 全局均值"""
    od_fine    = knowledge_base.get("od_fine",   {})
    od_coarse  = knowledge_base.get("od_coarse", {})
    global_avg = knowledge_base.get("global_avg", 1200.0)
    global_med = knowledge_base.get("global_med", 1200.0)

    key_fine   = f"{g_o_fine}_{g_d_fine}"
    key_coarse = f"{g_o_coarse}_{g_d_coarse}"

    if key_fine in od_fine:
        s = od_fine[key_fine]
        return s["mean"], s["median"], s["std"], s["count"]
    if key_coarse in od_coarse:
        s = od_coarse[key_coarse]
        return s["mean"], s["median"], s["std"], s["count"]
    return global_avg, global_med, 0.0, 0


def safe_get_traj_id(traj):
    for key in ["traj_id", "order_id", "id", "trip_id"]:
        if key in traj:
            return traj[key]
    return str(id(traj))


def safe_extract_features(traj, knowledge_base):
    """带完整异常捕获的特征提取，返回 43 维向量"""
    global_avg = knowledge_base.get("global_avg", 1200.0)
    global_med = knowledge_base.get("global_med", 1200.0)
    try:
        coords = traj.get("coords", [])
        dep_ts = traj.get("departure_timestamp", 0)
        feat, g_o_f, g_d_f, g_o_c, g_d_c = extract_features_v2(coords, dep_ts)
        od_mean, od_med, od_std, od_cnt = lookup_od_time(
            knowledge_base, g_o_f, g_d_f, g_o_c, g_d_c)
        return feat + [od_mean, od_med, od_std, float(od_cnt)]
    except Exception as e:
        tid = safe_get_traj_id(traj)
        print(f"    [警告] 特征提取失败 (traj_id={tid}): {e}")
        n = len(traj.get("coords", []))
        fallback = max(float(n) * 15.0, global_avg)
        feat = [0.0] * 43
        feat[0]  = float(n)
        feat[4]  = float(n) * 15.0
        feat[39] = fallback
        feat[40] = global_med
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# 模型训练
# ─────────────────────────────────────────────────────────────────────────────

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
    X_tr, y_tr = X_train[idx[:split]], y_train[idx[:split]]
    X_va, y_va = X_train[idx[split:]], y_train[idx[split:]]

    xgb_p = (best_params or {}).get("xgb", {}) or {}
    print("[*] 训练 XGBoost (MAE 直优，early stopping)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=xgb_p.get("xgb_lr",     0.05),
        max_depth=xgb_p.get("xgb_depth",       8),
        subsample=xgb_p.get("xgb_sub",         0.85),
        colsample_bytree=xgb_p.get("xgb_col",  0.85),
        gamma=xgb_p.get("xgb_gamma",           0.1),
        min_child_weight=xgb_p.get("xgb_mcw",  5),
        reg_alpha=xgb_p.get("xgb_alpha",       0.1),
        reg_lambda=xgb_p.get("xgb_lambda",     1.0),
        objective="reg:absoluteerror",
        eval_metric="mae",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
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


def tune_hyperparams(n_trials=40):
    """Optuna 超参搜索，结果保存到 processed_data_b/best_params.pkl"""
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
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    split = int(len(X) * 0.85)
    X_tr, y_tr = X[idx[:split]], y[idx[:split]]
    X_va, y_va = X[idx[split:]], y[idx[split:]]
    print(f"[*] Optuna 超参搜索：{n_trials} 轮，训练集 {len(X_tr)} / 验证集 {len(X_va)}")

    def xgb_obj(trial):
        m = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=trial.suggest_float("xgb_lr",    0.02, 0.15, log=True),
            max_depth=trial.suggest_int("xgb_depth",       4,   10),
            subsample=trial.suggest_float("xgb_sub",       0.6,  1.0),
            colsample_bytree=trial.suggest_float("xgb_col",0.6,  1.0),
            gamma=trial.suggest_float("xgb_gamma",         0.0,  2.0),
            min_child_weight=trial.suggest_int("xgb_mcw",  1,   20),
            reg_alpha=trial.suggest_float("xgb_alpha",     0.0,  2.0),
            reg_lambda=trial.suggest_float("xgb_lambda",   0.5,  5.0),
            objective="reg:absoluteerror", eval_metric="mae",
            early_stopping_rounds=40, random_state=42, n_jobs=-1,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return np.mean(np.abs(y_va - m.predict(X_va)))

    xgb_study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(xgb_obj, n_trials=n_trials, show_progress_bar=True)
    xgb_best = xgb_study.best_params
    print(f"    -> XGBoost 最优 MAE: {xgb_study.best_value:.2f}s")

    lgb_best = None
    if USE_LGB:
        def lgb_obj(trial):
            p = dict(
                objective="regression_l1", metric="mae", boosting_type="gbdt",
                learning_rate=trial.suggest_float("lgb_lr",    0.02, 0.15, log=True),
                num_leaves=trial.suggest_int("lgb_leaves",     31,  255),
                max_depth=trial.suggest_int("lgb_depth",        4,   12),
                subsample=trial.suggest_float("lgb_sub",        0.6,  1.0),
                feature_fraction=trial.suggest_float("lgb_col", 0.6,  1.0),
                min_child_samples=trial.suggest_int("lgb_mcs",  5,   50),
                reg_alpha=trial.suggest_float("lgb_alpha",      0.0,  2.0),
                reg_lambda=trial.suggest_float("lgb_lambda",    0.5,  5.0),
                bagging_freq=5, verbose=-1, n_jobs=-1,
            )
            b = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=1000,
                          valid_sets=[lgb.Dataset(X_va, y_va)],
                          callbacks=[lgb.early_stopping(40, verbose=False),
                                     lgb.log_evaluation(-1)])
            return np.mean(np.abs(y_va - b.predict(X_va)))

        lgb_study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=42))
        lgb_study.optimize(lgb_obj, n_trials=n_trials, show_progress_bar=True)
        lgb_best = lgb_study.best_params
        print(f"    -> LightGBM 最优 MAE: {lgb_study.best_value:.2f}s")

    best_params = {"xgb": xgb_best, "lgb": lgb_best}
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, "best_params.pkl"), "wb") as f:
        pickle.dump(best_params, f)
    print(f"[*] 最优超参已保存 → {PROCESSED_DIR}/best_params.pkl")
    return best_params


def find_best_blend_weight(xgb_model, lgb_model, X_val, y_val):
    """在验证集上搜索最优融合权重 w（xgb×w + lgb×(1-w)），步长 0.02"""
    xp = xgb_model.predict(X_val)
    lp = lgb_model.predict(X_val)
    best_w, best_mae = 0.5, float("inf")
    for w in np.arange(0.0, 1.01, 0.02):
        mae = np.mean(np.abs(y_val - (w * xp + (1 - w) * lp)))
        if mae < best_mae:
            best_mae, best_w = mae, round(float(w), 2)
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
    n_fine   = len(knowledge_base.get("od_fine",   {}))
    n_coarse = len(knowledge_base.get("od_coarse", {}))
    print(f"[*] 知识库已加载：精细 OD {n_fine} 对，粗粒度 OD {n_coarse} 对")

    # ── 3. 搜索最优融合权重 ──────────────────────────────────────
    blend_w = 0.5
    if USE_LGB and lgb_model is not None:
        val_in  = os.path.join(INPUT_DIR, "val_input.pkl")
        val_gt  = os.path.join(INPUT_DIR, "val_gt.pkl")
        if os.path.exists(val_in) and os.path.exists(val_gt):
            print("[*] 使用 val_gt.pkl 搜索最优融合权重...")
            with open(val_in, "rb") as f:
                val_data = pickle.load(f)
            with open(val_gt, "rb") as f:
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
                    np.array(y_ws, dtype=np.float32))
        else:
            x_file = os.path.join(PROCESSED_DIR, "X_train_final.npy")
            if os.path.exists(x_file):
                X_all = np.load(x_file)
                y_all = np.load(os.path.join(PROCESSED_DIR, "y_train_final.npy"))
                n = len(X_all)
                print("[*] 使用训练集内部验证集搜索融合权重（fallback）...")
                blend_w = find_best_blend_weight(
                    xgb_model, lgb_model,
                    X_all[int(n * 0.9):], y_all[int(n * 0.9):])

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

        X_arr    = np.array(X_feats, dtype=np.float32)
        xgb_pred = xgb_model.predict(X_arr)

        if USE_LGB and lgb_model is not None:
            lgb_pred   = lgb_model.predict(X_arr)
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
                y_true      = [gt_dict.get(tid, 0) for tid in traj_ids]
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
        best = tune_hyperparams(n_trials=n_trials)
        if best is not None:
            for p in [os.path.join(PROCESSED_DIR, "xgb_model_v3.json"),
                      os.path.join(PROCESSED_DIR, "lgb_model_v3.txt")]:
                if os.path.exists(p):
                    os.remove(p)
            run_task_b()
    else:
        run_task_b()
