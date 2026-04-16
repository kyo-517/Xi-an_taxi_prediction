import math
import datetime
import numpy as np
try:
    import pygeohash as pgh
    HAS_GEOHASH = True
except ImportError:
    HAS_GEOHASH = False

def haversine(lon1, lat1, lon2, lat2):
    if np.isnan(lon1) or np.isnan(lat1) or np.isnan(lon2) or np.isnan(lat2):
        return np.nan
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_grid_id(lon, lat, precision=6):
    """
    将经纬度转换为离散的网格ID (Geohash)。
    精度6代表约 1.2km x 0.6km 的矩形区域，非常适合城市街区级别的特征提取。
    """
    if HAS_GEOHASH:
        return pgh.encode(lat, lon, precision=precision)
    else:
        # Fallback: 手动经纬度截断作为简单网格划分 (约1km)
        return f"{int(lat*100)}_{int(lon*100)}"

def extract_task_b_features_advanced(coords, departure_timestamp):
    """
    进阶特征提取，加入了空间网格化 (Grid Geohash) 和形态学约束
    """
    coords = np.array(coords)
    num_points = len(coords)
    
    lon_start, lat_start = coords[0]
    lon_end, lat_end = coords[-1]
    
    straight_dist = haversine(lon_start, lat_start, lon_end, lat_end)
    
    total_dist = 0.0
    for i in range(1, num_points):
        dist = haversine(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
        if not np.isnan(dist):
            total_dist += dist
            
    sinuosity = total_dist / straight_dist if straight_dist > 0 else 1.0
    
    dt = datetime.datetime.fromtimestamp(departure_timestamp)
    hour = dt.hour
    weekday = dt.weekday()
    is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    
    baseline_time_est = num_points * 15.0
    
    # 提取起终点的网格ID (供后续构建 OD 矩阵使用)
    grid_o = get_grid_id(lon_start, lat_start)
    grid_d = get_grid_id(lon_end, lat_end)
    
    return [
        num_points,           
        total_dist,           
        straight_dist,        
        sinuosity,            
        hour,                 
        weekday,
        is_rush_hour,         # 新增：是否早晚高峰
        baseline_time_est,    
        lon_start, lat_start, 
        lon_end, lat_end      
    ], grid_o, grid_d

def evaluate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    non_zero = y_true > 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    return mae, rmse, mape