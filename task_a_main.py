"""
任务 A：轨迹修复 —— LightGBM 残差修正 + PCHIP 兜底
策略：
  1. 若 LightGBM 残差模型可用 → 线性插值 + LightGBM 残差修正
  2. 否则 → 全局 PCHIP
  3. 兜底 → Pandas time 线性插值
"""
import pickle
import numpy as np
import pandas as pd
import math
import os
from tqdm import tqdm
from datetime import datetime
from scipy.interpolate import PchipInterpolator
from features_and_utils import haversine

PROCESSED_DIR  = "processed_data"
SUBMISSION_DIR = "submissions"
INPUT_DIR      = "task_A_recovery"


# ============================================================
# 模型加载
# ============================================================
def load_lgb_models():
    """加载 LightGBM 残差模型。"""
    try:
        import lightgbm as lgb
        lon_path = os.path.join(PROCESSED_DIR, "lgb_residual_lon.txt")
        lat_path = os.path.join(PROCESSED_DIR, "lgb_residual_lat.txt")
        if os.path.exists(lon_path) and os.path.exists(lat_path):
            model_lon = lgb.Booster(model_file=lon_path)
            model_lat = lgb.Booster(model_file=lat_path)
            return model_lon, model_lat
    except Exception:
        pass
    return None, None


# ============================================================
# 特征提取（与训练脚本 v2 完全一致，22 维）
# ============================================================
def _bearing(lon1, lat1, lon2, lat2):
    dl = math.radians(lon2 - lon1)
    p1, p2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.atan2(x, y)


def _haversine(lon1, lat1, lon2, lat2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def extract_gap_features(coords, timestamps, mask, gap_start, gap_end,
                          traj_dist, traj_time, traj_avg_speed, hour, is_rush):
    """
    对缺失段 [gap_start, gap_end) 提取 22 维特征矩阵。
    返回 (features_array, linear_coords) 或 None。
    """
    n = len(mask)
    left = gap_start - 1
    right = gap_end

    if left < 0 or right >= n:
        return None

    s_lon, s_lat = float(coords[left][0]), float(coords[left][1])
    e_lon, e_lat = float(coords[right][0]), float(coords[right][1])
    if np.isnan(s_lon) or np.isnan(e_lon):
        return None

    t_start = float(timestamps[left])
    t_end = float(timestamps[right])
    dt = t_end - t_start
    if dt <= 0:
        return None

    dist = _haversine(s_lon, s_lat, e_lon, e_lat)
    bear = _bearing(s_lon, s_lat, e_lon, e_lat)
    gap_n = gap_end - gap_start
    speed_avg = dist / max(dt, 1)

    # 起点速度/方向
    known_idx = [k for k in range(n) if mask[k]]
    left_pos = known_idx.index(left) if left in known_idx else -1
    if left_pos > 0:
        prev = known_idx[left_pos - 1]
        dt_s = float(timestamps[left]) - float(timestamps[prev])
        if dt_s > 0:
            sp_s = _haversine(float(coords[prev][0]), float(coords[prev][1]),
                               s_lon, s_lat) / dt_s
            br_s = _bearing(float(coords[prev][0]), float(coords[prev][1]),
                             s_lon, s_lat)
        else:
            sp_s, br_s = 0.0, bear
    else:
        sp_s, br_s = speed_avg, bear

    # 终点速度/方向
    right_pos = known_idx.index(right) if right in known_idx else -1
    if right_pos >= 0 and right_pos < len(known_idx) - 1:
        nxt = known_idx[right_pos + 1]
        dt_e = float(timestamps[nxt]) - float(timestamps[right])
        if dt_e > 0:
            sp_e = _haversine(e_lon, e_lat,
                               float(coords[nxt][0]), float(coords[nxt][1])) / dt_e
            br_e = _bearing(e_lon, e_lat,
                             float(coords[nxt][0]), float(coords[nxt][1]))
        else:
            sp_e, br_e = 0.0, bear
    else:
        sp_e, br_e = speed_avg, bear

    bear_diff = br_e - br_s

    features = []
    linear_coords = []
    for idx in range(gap_start, gap_end):
        ratio = (float(timestamps[idx]) - t_start) / dt
        lin_lon = s_lon + ratio * (e_lon - s_lon)
        lin_lat = s_lat + ratio * (e_lat - s_lat)
        linear_coords.append([lin_lon, lin_lat])
        features.append([
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
        ])

    return np.array(features, dtype=np.float32), linear_coords


# ============================================================
# 修复函数
# ============================================================
def global_pchip_fill(coords, timestamps, mask):
    known_idx = [k for k in range(len(mask)) if mask[k]]
    if len(known_idx) < 2:
        result = [[float("nan"), float("nan")] for _ in range(len(mask))]
        for k in known_idx:
            result[k] = list(coords[k])
        return result

    t_k = np.array([timestamps[k] for k in known_idx], dtype=float)
    lo_k = np.array([coords[k][0] for k in known_idx], dtype=float)
    la_k = np.array([coords[k][1] for k in known_idx], dtype=float)

    pchip_lon = PchipInterpolator(t_k, lo_k, extrapolate=True)
    pchip_lat = PchipInterpolator(t_k, la_k, extrapolate=True)
    t_all = np.array(timestamps, dtype=float)
    lo_all = pchip_lon(t_all)
    la_all = pchip_lat(t_all)
    for k in known_idx:
        lo_all[k] = coords[k][0]
        la_all[k] = coords[k][1]
    return [[float(lo), float(la)] for lo, la in zip(lo_all, la_all)]


def recover_trajectory(traj_item, model_lon, model_lat):
    coords = traj_item["coords"]
    timestamps = traj_item["timestamps"]
    mask = traj_item["mask"]
    traj_id = traj_item["traj_id"]

    try:
        # 计算全局轨迹特征
        known_idx = [k for k in range(len(mask)) if mask[k]]
        if len(known_idx) >= 2:
            first_k, last_k = known_idx[0], known_idx[-1]
            traj_dist = _haversine(
                float(coords[first_k][0]), float(coords[first_k][1]),
                float(coords[last_k][0]),  float(coords[last_k][1]))
            traj_time = float(timestamps[last_k]) - float(timestamps[first_k])
            traj_avg_speed = traj_dist / max(traj_time, 1)
        else:
            traj_dist, traj_time, traj_avg_speed = 0.0, 0.0, 0.0

        dt_obj = datetime.utcfromtimestamp(float(timestamps[0]) + 8*3600)
        hour = dt_obj.hour
        is_rush = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0

        # 策略 1：LightGBM 残差修正
        if model_lon is not None and model_lat is not None:
            full_coords = [None] * len(mask)
            for k in known_idx:
                full_coords[k] = list(coords[k])

            i = 0
            while i < len(mask):
                if mask[i]:
                    i += 1
                    continue
                j = i
                while j < len(mask) and not mask[j]:
                    j += 1

                result = extract_gap_features(
                    coords, timestamps, mask, i, j,
                    traj_dist, traj_time, traj_avg_speed, hour, is_rush)
                if result is not None:
                    feat_arr, lin_coords = result
                    res_lon = model_lon.predict(feat_arr)
                    res_lat = model_lat.predict(feat_arr)
                    for k in range(j - i):
                        full_coords[i + k] = [
                            lin_coords[k][0] + res_lon[k],
                            lin_coords[k][1] + res_lat[k],
                        ]
                else:
                    pchip_result = global_pchip_fill(coords, timestamps, mask)
                    for k in range(i, j):
                        full_coords[k] = pchip_result[k]
                i = j

            has_none = any(v is None for v in full_coords)
            if not has_none:
                return full_coords

            fc_arr = np.full((len(timestamps), 2), np.nan)
            for k in range(len(mask)):
                if full_coords[k] is not None:
                    fc_arr[k] = full_coords[k]
            df = pd.DataFrame(fc_arr, columns=["lon", "lat"])
            df.index = pd.to_datetime(timestamps, unit="s")
            df = df.interpolate(method="time").bfill().ffill()
            return df[["lon", "lat"]].values.tolist()

        # 策略 2：全局 PCHIP
        result = global_pchip_fill(coords, timestamps, mask)
        fc_arr = np.array(result, dtype=float)
        if np.any(np.isnan(fc_arr)):
            df = pd.DataFrame(fc_arr, columns=["lon", "lat"])
            df.index = pd.to_datetime(timestamps, unit="s")
            df = df.interpolate(method="time").bfill().ffill()
            return df[["lon", "lat"]].values.tolist()
        return result

    except Exception as e:
        # 终极兜底：线性插值
        print(f"    [WARN] traj {traj_id} 异常({e})，降级线性。")
        fc_arr = np.full((len(timestamps), 2), np.nan)
        for k in range(len(mask)):
            if mask[k]:
                fc_arr[k] = coords[k]
        df = pd.DataFrame(fc_arr, columns=["lon", "lat"])
        df.index = pd.to_datetime(timestamps, unit="s")
        df = df.interpolate(method="time").bfill().ffill()
        return df[["lon", "lat"]].values.tolist()


# ============================================================
# 评估 & 主入口
# ============================================================
def evaluate_recovery(pred_data, gt_data, input_data):
    pred_dict = {item["traj_id"]: item["coords"] for item in pred_data}
    gt_dict = {item["traj_id"]: item["coords"] for item in gt_data}
    errors = []
    for item in input_data:
        tid, mask = item["traj_id"], item["mask"]
        for k, is_known in enumerate(mask):
            if not is_known:
                dist = haversine(pred_dict[tid][k][0], pred_dict[tid][k][1],
                                 gt_dict[tid][k][0], gt_dict[tid][k][1])
                if not np.isnan(dist):
                    errors.append(dist)
    return np.mean(errors), np.sqrt(np.mean(np.array(errors) ** 2))


def run_task_a():
    model_lon, model_lat = load_lgb_models()
    if model_lon is not None:
        print("[*] LightGBM 残差模型已加载 ✓")
    else:
        print("[*] LightGBM 模型未找到，使用 PCHIP 兜底。")
        print("    提示：运行 python train_residual_model.py 训练模型。")

    for prefix in ["val", "test"]:
        for rate in [8, 16]:
            filename = f"{prefix}_input_{rate}.pkl"
            input_path = os.path.join(INPUT_DIR, filename)
            if not os.path.exists(input_path):
                continue

            with open(input_path, "rb") as f:
                input_data = pickle.load(f)

            print(f"\n[*] 处理: {filename} ...")
            pred_results = []

            for traj in tqdm(input_data, desc=f"修复 {rate}", unit="条", colour="green"):
                try:
                    recovered = recover_trajectory(traj, model_lon, model_lat)
                except Exception as e:
                    print(f"    [FATAL] {traj.get('traj_id','?')} 失败({e})")
                    recovered = [list(c) for c in traj["coords"]]
                pred_results.append({"traj_id": traj["traj_id"], "coords": recovered})

            gt_file = os.path.join(INPUT_DIR, "val_gt.pkl")
            if prefix == "val" and os.path.exists(gt_file):
                with open(gt_file, "rb") as f:
                    gt_data = pickle.load(f)
                mae, rmse = evaluate_recovery(pred_results, gt_data, input_data)
                print(f"    ✓ MAE={mae:.2f}m, RMSE={rmse:.2f}m")

            local_path = os.path.join(INPUT_DIR, filename.replace("input", "pred"))
            with open(local_path, "wb") as f:
                pickle.dump(pred_results, f)

            os.makedirs(SUBMISSION_DIR, exist_ok=True)
            submit_path = os.path.join(SUBMISSION_DIR, f"task_A_pred_{rate}.pkl")
            with open(submit_path, "wb") as f:
                pickle.dump(pred_results, f)
            print(f"    → {submit_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("任务 A：轨迹修复（LightGBM 残差修正 + PCHIP 兜底）")
    print("=" * 60)
    run_task_a()
