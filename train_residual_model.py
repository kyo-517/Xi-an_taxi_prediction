"""
残差学习模型训练脚本 v2 —— LightGBM
从 data_ds15/train.pkl 模拟缺失，用 data_org/train.pkl 提供真实坐标作为 ground truth。
分别训练 lon 和 lat 两个 LightGBM 回归模型。

特征设计（每个缺失点 20+ 维）：
  基础特征：ratio, gap_n, dist_od, bearing_od, dt_gap
  速度特征：speed_start, speed_end, speed_avg
  方向特征：bearing_start, bearing_end, bearing_diff
  坐标特征：lon_start, lat_start, lon_end, lat_end
  全局上下文：traj_total_dist, traj_total_time, traj_avg_speed
  时间特征：hour_of_day, is_rush_hour
  位置特征：ratio_sin, ratio_cos（捕捉周期性偏移模式）
"""
import pickle
import numpy as np
import math
import os
import lightgbm as lgb
import joblib

DATA_DS15 = os.path.join("data_ds15", "train.pkl")
DATA_ORG  = os.path.join("data_org",  "train.pkl")
MODEL_DIR = "processed_data"


def _haversine(lon1, lat1, lon2, lat2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lon1, lat1, lon2, lat2):
    dl = math.radians(lon2 - lon1)
    p1, p2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.atan2(x, y)


def generate_training_data(max_trajs=80000):
    """
    用 data_ds15 模拟缺失，用 data_org 提供真实坐标。
    """
    print(f"[*] 加载 {DATA_DS15}...")
    with open(DATA_DS15, "rb") as f:
        ds15_data = pickle.load(f)

    print(f"[*] 加载 {DATA_ORG}...")
    with open(DATA_ORG, "rb") as f:
        org_data = pickle.load(f)

    n_use = min(max_trajs, len(ds15_data))
    print(f"[*] 从 {n_use} 条轨迹生成训练样本...")

    all_X, all_y_lon, all_y_lat = [], [], []

    for traj_idx in range(n_use):
        if traj_idx % 10000 == 0:
            print(f"    进度: {traj_idx}/{n_use}")

        ds15 = ds15_data[traj_idx]
        org  = org_data[traj_idx]

        coords_ds15 = ds15["coords"]
        ts_ds15     = ds15["timestamps"]
        coords_org  = np.array(org["coords"])
        ts_org      = np.array(org["timestamps"])
        n = len(coords_ds15)

        if n < 16:
            continue

        # 全局轨迹特征
        traj_dist = _haversine(
            coords_ds15[0][0], coords_ds15[0][1],
            coords_ds15[-1][0], coords_ds15[-1][1])
        traj_time = ts_ds15[-1] - ts_ds15[0]
        traj_avg_speed = traj_dist / max(traj_time, 1)

        # 出发时间
        from datetime import datetime
        dt_obj = datetime.utcfromtimestamp(ts_ds15[0] + 8*3600)
        hour = dt_obj.hour
        is_rush = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0

        # 对 gap=8 和 gap=16 模拟缺失
        for gap in [8, 16]:
            known_set = set(range(0, n, gap))
            known_set.add(n - 1)
            known_sorted = sorted(known_set)

            for ki in range(len(known_sorted) - 1):
                si = known_sorted[ki]
                ei = known_sorted[ki + 1]
                missing = [idx for idx in range(si+1, ei)]
                if not missing:
                    continue

                s_lon, s_lat = coords_ds15[si]
                e_lon, e_lat = coords_ds15[ei]
                t_s = float(ts_ds15[si])
                t_e = float(ts_ds15[ei])
                dt = t_e - t_s
                if dt <= 0:
                    continue

                dist = _haversine(s_lon, s_lat, e_lon, e_lat)
                bear = _bearing(s_lon, s_lat, e_lon, e_lat)
                gap_n = len(missing)

                # 起点速度/方向
                if si > 0:
                    dt_s = float(ts_ds15[si]) - float(ts_ds15[si-1])
                    if dt_s > 0:
                        sp_s = _haversine(coords_ds15[si-1][0], coords_ds15[si-1][1],
                                          s_lon, s_lat) / dt_s
                        br_s = _bearing(coords_ds15[si-1][0], coords_ds15[si-1][1],
                                        s_lon, s_lat)
                    else:
                        sp_s, br_s = 0.0, bear
                else:
                    sp_s, br_s = dist/max(dt,1), bear

                # 终点速度/方向
                if ei < n - 1:
                    dt_e = float(ts_ds15[ei+1]) - float(ts_ds15[ei])
                    if dt_e > 0:
                        sp_e = _haversine(e_lon, e_lat,
                                          coords_ds15[ei+1][0], coords_ds15[ei+1][1]) / dt_e
                        br_e = _bearing(e_lon, e_lat,
                                        coords_ds15[ei+1][0], coords_ds15[ei+1][1])
                    else:
                        sp_e, br_e = 0.0, bear
                else:
                    sp_e, br_e = dist/max(dt,1), bear

                bear_diff = br_e - br_s
                speed_avg = dist / max(dt, 1)

                for idx in missing:
                    ratio = (float(ts_ds15[idx]) - t_s) / dt
                    lin_lon = s_lon + ratio * (e_lon - s_lon)
                    lin_lat = s_lat + ratio * (e_lat - s_lat)

                    # 真实坐标：从 data_org 中找最近时间戳的点
                    target_ts = float(ts_ds15[idx])
                    org_idx = np.argmin(np.abs(ts_org - target_ts))
                    true_lon = float(coords_org[org_idx, 0])
                    true_lat = float(coords_org[org_idx, 1])

                    res_lon = true_lon - lin_lon
                    res_lat = true_lat - lin_lat

                    feat = [
                        ratio,                          # 0
                        gap_n,                          # 1
                        dist,                           # 2
                        bear,                           # 3
                        dt,                             # 4
                        sp_s,                           # 5
                        sp_e,                           # 6
                        speed_avg,                      # 7
                        br_s,                           # 8
                        br_e,                           # 9
                        bear_diff,                      # 10
                        s_lon, s_lat,                   # 11, 12
                        e_lon, e_lat,                   # 13, 14
                        traj_dist,                      # 15
                        traj_time,                      # 16
                        traj_avg_speed,                 # 17
                        hour,                           # 18
                        is_rush,                        # 19
                        math.sin(ratio * math.pi),      # 20
                        math.cos(ratio * math.pi),      # 21
                    ]
                    all_X.append(feat)
                    all_y_lon.append(res_lon)
                    all_y_lat.append(res_lat)

    X = np.array(all_X, dtype=np.float32)
    y_lon = np.array(all_y_lon, dtype=np.float32)
    y_lat = np.array(all_y_lat, dtype=np.float32)
    print(f"[*] 生成 {len(X)} 个训练样本，特征维度 {X.shape[1]}")
    return X, y_lon, y_lat


def train_model():
    X, y_lon, y_lat = generate_training_data(max_trajs=80000)

    # 划分训练/验证
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = idx[:split], idx[split:]

    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
    }

    print("\n[*] 训练 LightGBM 残差模型 (lon)...")
    ds_lon_train = lgb.Dataset(X[train_idx], y_lon[train_idx])
    ds_lon_val   = lgb.Dataset(X[val_idx],   y_lon[val_idx], reference=ds_lon_train)
    model_lon = lgb.train(
        params, ds_lon_train,
        num_boost_round=2000,
        valid_sets=[ds_lon_val],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100),
        ],
    )

    print("\n[*] 训练 LightGBM 残差模型 (lat)...")
    ds_lat_train = lgb.Dataset(X[train_idx], y_lat[train_idx])
    ds_lat_val   = lgb.Dataset(X[val_idx],   y_lat[val_idx], reference=ds_lat_train)
    model_lat = lgb.train(
        params, ds_lat_train,
        num_boost_round=2000,
        valid_sets=[ds_lat_val],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100),
        ],
    )

    # 保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_lon.save_model(os.path.join(MODEL_DIR, "lgb_residual_lon.txt"))
    model_lat.save_model(os.path.join(MODEL_DIR, "lgb_residual_lat.txt"))
    print(f"\n[✓] 模型已保存到 {MODEL_DIR}/")

    # 评估
    pred_lon = model_lon.predict(X[val_idx])
    pred_lat = model_lat.predict(X[val_idx])
    mae_lon = np.mean(np.abs(y_lon[val_idx] - pred_lon))
    mae_lat = np.mean(np.abs(y_lat[val_idx] - pred_lat))
    print(f"    验证集残差 MAE: lon={mae_lon:.6f}° ({mae_lon*111000:.1f}m)")
    print(f"                    lat={mae_lat:.6f}° ({mae_lat*111000:.1f}m)")


if __name__ == "__main__":
    train_model()
