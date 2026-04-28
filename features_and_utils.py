import math
import datetime
import numpy as np

try:
    import pygeohash as pgh
    HAS_GEOHASH = True
except ImportError:
    HAS_GEOHASH = False

# 西安钟楼坐标
XIAN_CENTER_LON = 108.940
XIAN_CENTER_LAT = 34.265


def haversine(lon1, lat1, lon2, lat2):
    if np.isnan(lon1) or np.isnan(lat1) or np.isnan(lon2) or np.isnan(lat2):
        return np.nan
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lon1, lat1, lon2, lat2):
    """计算从点1到点2的方位角（弧度，[-π, π]）"""
    dl = math.radians(lon2 - lon1)
    p1, p2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.atan2(x, y)


def get_grid_id(lon, lat, precision=6):
    """将经纬度转换为 Geohash 网格 ID"""
    if HAS_GEOHASH:
        return pgh.encode(lat, lon, precision=precision)
    else:
        # fallback: 精度6约等于 0.01° ≈ 1km
        scale = 10 ** (precision - 4)
        return f"{int(lat * scale)}_{int(lon * scale)}"


def extract_task_b_features_advanced(coords, departure_timestamp):
    """
    增强版特征提取 v2：
    - 速度特征（均值、方差、分段速度）
    - 转向角特征（均值、标准差、急转弯次数）
    - 时间周期编码（sin/cos）
    - 边界框面积
    - 多层级 OD 网格 ID（precision 6 + 5）
    - 轨迹进度特征（前/中/后三段距离比）
    """
    coords = np.array(coords, dtype=np.float64)
    # 过滤 NaN 坐标点
    valid_mask = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
    coords = coords[valid_mask]
    num_points = len(coords)

    if num_points < 2:
        # 极端兜底：返回全零特征 + 两个 unknown grid
        return [0.0] * 38, "unknown", "unknown"

    lon_start, lat_start = coords[0]
    lon_end, lat_end = coords[-1]

    # ── 基础距离 ──────────────────────────────────────────────
    straight_dist = haversine(lon_start, lat_start, lon_end, lat_end)

    seg_dists = []
    for i in range(1, num_points):
        d = haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        seg_dists.append(d if not np.isnan(d) else 0.0)

    total_dist = sum(seg_dists)
    sinuosity = total_dist / straight_dist if straight_dist > 1.0 else 1.0

    # ── 速度特征（假设采样间隔约 15s）────────────────────────
    dt_per_step = 15.0
    speeds = [d / dt_per_step for d in seg_dists]  # m/s
    avg_speed = np.mean(speeds) if speeds else 0.0
    std_speed = np.std(speeds) if speeds else 0.0
    max_speed = np.max(speeds) if speeds else 0.0

    # 三段速度（前1/3、中1/3、后1/3）
    n_seg = max(1, len(speeds) // 3)
    speed_first  = np.mean(speeds[:n_seg]) if speeds[:n_seg] else avg_speed
    speed_middle = np.mean(speeds[n_seg:2 * n_seg]) if speeds[n_seg:2 * n_seg] else avg_speed
    speed_last   = np.mean(speeds[2 * n_seg:]) if speeds[2 * n_seg:] else avg_speed
    speed_trend  = speed_last - speed_first  # 正值=加速，负值=减速

    # ── 转向角特征 ────────────────────────────────────────────
    bearings = []
    for i in range(1, num_points):
        b = bearing(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        bearings.append(b)

    turn_angles = []
    for i in range(1, len(bearings)):
        diff = abs(bearings[i] - bearings[i - 1])
        # 归一化到 [0, π]
        diff = min(diff, 2 * math.pi - diff)
        turn_angles.append(diff)

    if turn_angles:
        mean_turn = np.mean(turn_angles)
        std_turn  = np.std(turn_angles)
        # 急转弯：转角 > 45°
        sharp_turns = sum(1 for t in turn_angles if t > math.pi / 4)
    else:
        mean_turn, std_turn, sharp_turns = 0.0, 0.0, 0

    # ── 边界框特征 ────────────────────────────────────────────
    lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
    lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()
    bbox_lon = lon_max - lon_min
    bbox_lat = lat_max - lat_min
    # 近似面积（度²，用于相对比较）
    bbox_area = bbox_lon * bbox_lat

    # ── 时间特征 ──────────────────────────────────────────────
    dt_obj = datetime.datetime.fromtimestamp(departure_timestamp)
    hour    = dt_obj.hour
    minute  = dt_obj.minute
    weekday = dt_obj.weekday()

    # 周期编码（避免 23→0 的跳变）
    hour_sin    = math.sin(2 * math.pi * hour / 24)
    hour_cos    = math.cos(2 * math.pi * hour / 24)
    minute_sin  = math.sin(2 * math.pi * minute / 60)
    minute_cos  = math.cos(2 * math.pi * minute / 60)
    weekday_sin = math.sin(2 * math.pi * weekday / 7)
    weekday_cos = math.cos(2 * math.pi * weekday / 7)

    is_rush_hour    = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    is_night        = 1 if hour < 6 or hour >= 22 else 0
    is_weekend      = 1 if weekday >= 5 else 0
    # 国庆黄金周（2016年10月1-7日）
    is_national_day = 1 if (dt_obj.month == 10 and 1 <= dt_obj.day <= 7) else 0

    # ── 空间上下文特征 ────────────────────────────────────────
    dist_to_center_start = haversine(lon_start, lat_start, XIAN_CENTER_LON, XIAN_CENTER_LAT)
    dist_to_center_end   = haversine(lon_end,   lat_end,   XIAN_CENTER_LON, XIAN_CENTER_LAT)
    # 行程方向（起点→终点的方位角）
    od_bearing = bearing(lon_start, lat_start, lon_end, lat_end)

    # ── 基准时间估计（最强先验）────────────────────────────────
    baseline_time_est = num_points * 15.0

    # ── OD 网格 ID（双精度，供外部查 OD 矩阵）────────────────
    grid_o_fine   = get_grid_id(lon_start, lat_start, precision=6)
    grid_d_fine   = get_grid_id(lon_end,   lat_end,   precision=6)
    grid_o_coarse = get_grid_id(lon_start, lat_start, precision=5)
    grid_d_coarse = get_grid_id(lon_end,   lat_end,   precision=5)

    # ── 组装特征向量（共 38 维，不含 OD 历史时间）────────────
    features = [
        # 基础轨迹特征 (0-4)
        num_points,
        total_dist,
        straight_dist,
        sinuosity,
        baseline_time_est,
        # 速度特征 (5-11)
        avg_speed,
        std_speed,
        max_speed,
        speed_first,
        speed_middle,
        speed_last,
        speed_trend,
        # 转向角特征 (12-14)
        mean_turn,
        std_turn,
        float(sharp_turns),
        # 边界框特征 (15-17)
        bbox_lon,
        bbox_lat,
        bbox_area,
        # 时间原始值 (18-20)
        float(hour),
        float(weekday),
        float(minute),
        # 时间周期编码 (21-26)
        hour_sin,
        hour_cos,
        minute_sin,
        minute_cos,
        weekday_sin,
        weekday_cos,
        # 时间标志位 (27-30)
        float(is_rush_hour),
        float(is_night),
        float(is_weekend),
        float(is_national_day),
        # 空间上下文 (31-34)
        dist_to_center_start,
        dist_to_center_end,
        od_bearing,
        math.sin(od_bearing),
        # 坐标 (35-38)  ← 保留原始坐标供树模型学习区域效应
        lon_start,
        lat_start,
        lon_end,
        lat_end,
    ]

    return features, (grid_o_fine, grid_d_fine), (grid_o_coarse, grid_d_coarse)


def lookup_od_time(knowledge_base, grid_o_fine, grid_d_fine, grid_o_coarse, grid_d_coarse):
    """
    双层 OD 查询：精细 → 粗粒度 → 全局均值
    返回 (历史均值, 历史中位数, 历史标准差, 样本数)
    """
    od_fine    = knowledge_base.get("od_fine",   {})
    od_coarse  = knowledge_base.get("od_coarse", {})
    global_avg = knowledge_base.get("global_avg", 1200.0)
    global_med = knowledge_base.get("global_med", 1200.0)

    key_fine   = f"{grid_o_fine}_{grid_d_fine}"
    key_coarse = f"{grid_o_coarse}_{grid_d_coarse}"

    if key_fine in od_fine:
        s = od_fine[key_fine]
        return s["mean"], s["median"], s["std"], s["count"]
    elif key_coarse in od_coarse:
        s = od_coarse[key_coarse]
        return s["mean"], s["median"], s["std"], s["count"]
    else:
        return global_avg, global_med, 0.0, 0


def evaluate_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    non_zero = y_true > 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    return mae, rmse, mape
