"""
数据探索脚本：从数据层面分析任务A的优化空间
- 训练数据分析（data_org / data_ds15）
- 验证数据结构
- 缺失模式统计
- 轨迹特征分布
- PCHIP vs Linear 的失败案例分析
"""
import pickle
import numpy as np
import pandas as pd
import os
import math
from collections import defaultdict
from tqdm import tqdm

# ==========================================
# 基础工具函数
# ==========================================
def haversine(lon1, lat1, lon2, lat2):
    """计算两点球面距离（米）"""
    if np.isnan(lon1) or np.isnan(lat1) or np.isnan(lon2) or np.isnan(lat2):
        return np.nan
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ==========================================
# 1. 训练数据探索
# ==========================================
def explore_training_data():
    """分析原始和降采样训练数据"""
    print("\n" + "="*70)
    print("【训练数据分析】")
    print("="*70)
    
    for dataset_name in ["data_org", "data_ds15"]:
        print(f"\n[*] 分析 {dataset_name}/")
        
        pkl_files = [f for f in os.listdir(dataset_name) if f.endswith('.pkl')]
        if not pkl_files:
            print(f"  ⚠️  未找到 pkl 文件")
            continue
        
        # 随机抽样一个文件查看
        sample_file = pkl_files[0]
        sample_path = os.path.join(dataset_name, sample_file)
        
        with open(sample_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"  样本文件: {sample_file}")
        print(f"  轨迹数量: {len(data)}")
        
        # 统计轨迹基本特征
        coords_counts = []
        durations = []
        distances = []
        speeds = []
        
        for traj in data:
            coords = traj.get('coords', [])
            timestamps = traj.get('timestamps', [])
            
            coords_counts.append(len(coords))
            
            if len(timestamps) > 1:
                duration = timestamps[-1] - timestamps[0]
                durations.append(duration)
                
                # 计算总距离
                dist = 0
                for i in range(len(coords)-1):
                    if i+1 < len(coords):
                        d = haversine(coords[i][0], coords[i][1], 
                                    coords[i+1][0], coords[i+1][1])
                        if not np.isnan(d):
                            dist += d
                distances.append(dist)
                
                # 平均速度
                if duration > 0:
                    speeds.append(dist / duration)
        
        # 打印统计
        print(f"\n  轨迹点数:")
        print(f"    范围: {min(coords_counts):3d} ~ {max(coords_counts):3d}")
        print(f"    均值: {np.mean(coords_counts):6.1f}")
        print(f"    中位数: {np.median(coords_counts):6.1f}")
        
        print(f"\n  轨迹时长:")
        durations_min = [d/60 for d in durations]
        print(f"    范围: {min(durations_min):5.1f} ~ {max(durations_min):5.1f} 分钟")
        print(f"    均值: {np.mean(durations_min):6.1f} 分钟")
        
        print(f"\n  轨迹距离:")
        print(f"    范围: {min(distances)/1000:5.2f} ~ {max(distances)/1000:5.2f} km")
        print(f"    均值: {np.mean(distances)/1000:6.2f} km")
        
        print(f"\n  平均速度:")
        speeds_kmh = [s*3.6 for s in speeds]
        print(f"    范围: {min(speeds_kmh):5.1f} ~ {max(speeds_kmh):5.1f} km/h")
        print(f"    均值: {np.mean(speeds_kmh):6.1f} km/h")
        
        # 采样间隔分析
        if dataset_name == "data_ds15":
            all_intervals = []
            for traj in data[:100]:  # 采样100条分析
                timestamps = traj.get('timestamps', [])
                if len(timestamps) > 1:
                    intervals = np.diff(timestamps)
                    all_intervals.extend(intervals)
            
            if all_intervals:
                print(f"\n  采样间隔:")
                print(f"    范围: {min(all_intervals):5.1f} ~ {max(all_intervals):5.1f} 秒")
                print(f"    均值: {np.mean(all_intervals):6.1f} 秒")
                print(f"    中位数: {np.median(all_intervals):6.1f} 秒")

# ==========================================
# 2. 验证数据分析
# ==========================================
def explore_validation_data():
    """分析验证集的缺失模式"""
    print("\n" + "="*70)
    print("【验证数据分析】")
    print("="*70)
    
    base_dir = "task_A_recovery"
    
    for filename in ["val_input_8.pkl", "val_input_16.pkl"]:
        input_path = os.path.join(base_dir, filename)
        if not os.path.exists(input_path):
            print(f"  ⚠️  未找到 {filename}")
            continue
        
        print(f"\n[*] {filename}")
        with open(input_path, 'rb') as f:
            input_data = pickle.load(f)
        
        # 缺失段长度分析
        missing_lengths = []
        missing_fractions = []
        per_traj_missing = []
        
        for traj in input_data:
            mask = traj.get('mask', [])
            coords = traj.get('coords', [])
            
            # 轨迹中缺失比例
            if len(mask) > 0:
                missing_pct = (sum(1 for m in mask if not m) / len(mask)) * 100
                missing_fractions.append(missing_pct)
            
            # 找连续缺失段
            i = 0
            while i < len(mask):
                if not mask[i]:
                    j = i
                    while j < len(mask) and not mask[j]:
                        j += 1
                    missing_lengths.append(j - i)
                    i = j
                else:
                    i += 1
            
            per_traj_missing.append(sum(1 for m in mask if not m))
        
        print(f"  样本数: {len(input_data)}")
        print(f"\n  缺失段长度分布:")
        print(f"    总段数: {len(missing_lengths)}")
        print(f"    范围: {min(missing_lengths):3d} ~ {max(missing_lengths):3d} 点")
        print(f"    均值: {np.mean(missing_lengths):6.2f} 点")
        print(f"    中位数: {np.median(missing_lengths):6.1f} 点")
        
        # 长度分类
        short = sum(1 for x in missing_lengths if x <= 10)
        medium = sum(1 for x in missing_lengths if 10 < x <= 20)
        long = sum(1 for x in missing_lengths if x > 20)
        print(f"\n  按缺失长度分类:")
        print(f"    ≤10点 (短):   {short:5d} 段 ({100*short/len(missing_lengths):5.1f}%)")
        print(f"    11-20点(中):  {medium:5d} 段 ({100*medium/len(missing_lengths):5.1f}%)")
        print(f"    >20点 (长):   {long:5d} 段 ({100*long/len(missing_lengths):5.1f}%)")
        
        print(f"\n  每条轨迹缺失点数:")
        print(f"    范围: {min(per_traj_missing):5d} ~ {max(per_traj_missing):5d}")
        print(f"    均值: {np.mean(per_traj_missing):8.1f}")
        print(f"    缺失比例: {100*np.sum(per_traj_missing)/(len(input_data)*len(input_data[0]['mask'])):5.1f}%")

# ==========================================
# 3. 预测误差分析
# ==========================================
def analyze_prediction_errors():
    """分析PCHIP和线性的预测误差分布"""
    print("\n" + "="*70)
    print("【预测误差分析】")
    print("="*70)
    
    base_dir = "task_A_recovery"
    
    # 加载真值
    gt_file = os.path.join(base_dir, "val_gt.pkl")
    if not os.path.exists(gt_file):
        print("  ⚠️  未找到真值文件")
        return
    
    with open(gt_file, 'rb') as f:
        gt_data = pickle.load(f)
    gt_dict = {item['traj_id']: item['coords'] for item in gt_data}
    
    # 加载预测结果
    for filename, method_name in [("val_pred_8.pkl", "PCHIP_8"), 
                                   ("val_pred_16.pkl", "PCHIP_16")]:
        pred_path = os.path.join(base_dir, filename)
        input_path = os.path.join(base_dir, filename.replace("pred", "input"))
        
        if not os.path.exists(pred_path) or not os.path.exists(input_path):
            continue
        
        print(f"\n[*] {method_name}")
        
        with open(input_path, 'rb') as f:
            input_data = pickle.load(f)
        with open(pred_path, 'rb') as f:
            pred_data = pickle.load(f)
        
        pred_dict = {item['traj_id']: item['coords'] for item in pred_data}
        
        # 计算每个缺失点的误差
        errors_by_length = defaultdict(list)
        all_errors = []
        
        for traj_item, pred_item in zip(input_data, pred_data):
            traj_id = traj_item['traj_id']
            mask = traj_item['mask']
            
            if traj_id not in gt_dict:
                continue
            
            pred_coords = pred_dict[traj_id]
            gt_coords = gt_dict[traj_id]
            
            # 找缺失段，统计每个缺失点及其所在段长度
            i = 0
            while i < len(mask):
                if not mask[i]:
                    j = i
                    while j < len(mask) and not mask[j]:
                        j += 1
                    seg_len = j - i
                    
                    # 统计这一段的误差
                    for k in range(i, j):
                        try:
                            p_lon, p_lat = pred_coords[k]
                            g_lon, g_lat = gt_coords[k]
                            dist = haversine(p_lon, p_lat, g_lon, g_lat)
                            if not np.isnan(dist):
                                errors_by_length[seg_len].append(dist)
                                all_errors.append(dist)
                        except:
                            pass
                    i = j
                else:
                    i += 1
        
        # 按缺失段长度统计误差
        print(f"\n  按缺失段长度统计MAE:")
        for seg_len in sorted([x for x in errors_by_length.keys() if x <= 20]):
            errors = errors_by_length[seg_len]
            mae = np.mean(errors)
            count = len(errors)
            print(f"    {seg_len:2d}点缺失: MAE={mae:6.2f}m (n={count:6d})")
        
        # 误差百分位
        if all_errors:
            all_errors = np.array(all_errors)
            print(f"\n  误差分布:")
            print(f"    P10: {np.percentile(all_errors, 10):6.2f}m")
            print(f"    P25: {np.percentile(all_errors, 25):6.2f}m")
            print(f"    P50: {np.percentile(all_errors, 50):6.2f}m (中位数)")
            print(f"    P75: {np.percentile(all_errors, 75):6.2f}m")
            print(f"    P90: {np.percentile(all_errors, 90):6.2f}m")
            print(f"    P95: {np.percentile(all_errors, 95):6.2f}m")
            
            # 异常值检测
            outlier_threshold = np.percentile(all_errors, 99)
            outliers = sum(1 for e in all_errors if e > outlier_threshold)
            print(f"\n  异常值(>1%分位)：{outliers} 个 (阈值={outlier_threshold:.1f}m)")

# ==========================================
# 4. 坐标分布分析
# ==========================================
def analyze_coordinate_distribution():
    """分析坐标的地理分布"""
    print("\n" + "="*70)
    print("【坐标地理分布分析】")
    print("="*70)
    
    base_dir = "task_A_recovery"
    input_file = os.path.join(base_dir, "val_input_8.pkl")
    
    if not os.path.exists(input_file):
        print("  ⚠️  未找到验证数据")
        return
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    all_coords = []
    for traj in data:
        coords = traj.get('coords', [])
        for lon, lat in coords:
            if not (np.isnan(lon) or np.isnan(lat)):
                all_coords.append((lon, lat))
    
    if not all_coords:
        print("  ⚠️  无有效坐标")
        return
    
    coords_array = np.array(all_coords)
    lons, lats = coords_array[:, 0], coords_array[:, 1]
    
    print(f"\n[*] 坐标范围（西安市）")
    print(f"  经度范围: {lons.min():.6f} ~ {lons.max():.6f}°")
    print(f"  纬度范围: {lats.min():.6f} ~ {lats.max():.6f}°")
    print(f"  坐标点数: {len(all_coords):,}")
    
    # 栅格划分分析
    lon_bins = np.linspace(lons.min(), lons.max(), 11)
    lat_bins = np.linspace(lats.min(), lats.max(), 11)
    
    grid_counts = {}
    for lon, lat in all_coords:
        i = np.searchsorted(lon_bins, lon)
        j = np.searchsorted(lat_bins, lat)
        grid_counts[(i, j)] = grid_counts.get((i, j), 0) + 1
    
    print(f"\n[*] 地理栅格分布 (10x10 栅格)")
    mean_count = np.mean(list(grid_counts.values()))
    print(f"  平均每格: {mean_count:.0f} 点")
    print(f"  最密集: {max(grid_counts.values())} 点")
    print(f"  最稀疏: {min(grid_counts.values())} 点")
    print(f"  覆盖的栅格: {len(grid_counts)}/100")

# ==========================================
# 主函数
# ==========================================
if __name__ == "__main__":
    print("\n" + "🔍 "*35)
    print("任务A 数据探索分析")
    print("🔍 "*35)
    
    explore_training_data()
    explore_validation_data()
    analyze_prediction_errors()
    analyze_coordinate_distribution()
    
    print("\n" + "="*70)
    print("✓ 数据探索完成")
    print("="*70)
