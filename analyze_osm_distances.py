"""
分析脚本：查看真实验证数据中缺失轨迹段的距离分布
目的：进行 OSM 匹配的距离范围限制
"""
import pickle
import numpy as np
import os
from features_and_utils import haversine
from collections import defaultdict

INPUT_DIR = "task_A_recovery"

def analyze_distances(filename):
    """分析单个文件中所有缺失段的距离"""
    input_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(input_path):
        print(f"[警告] {input_path} 不存在，跳过")
        return None
    
    print(f"\n{'='*60}")
    print(f"分析文件: {filename}")
    print(f"{'='*60}")
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    all_distances = []
    segment_stats = defaultdict(list)
    
    total_trajs = len(data)
    total_segments = 0
    
    for traj_idx, traj in enumerate(data):
        coords = traj['coords']
        mask = traj['mask']
        
        # 遍历轨迹中的所有缺失段
        i = 0
        while i < len(mask):
            if not mask[i]:
                # 找到缺失段的起点和终点
                start_idx = i - 1
                j = i
                while j < len(mask) and not mask[j]:
                    j += 1
                end_idx = j
                
                if start_idx >= 0 and end_idx < len(mask):
                    start_coord = coords[start_idx]
                    end_coord = coords[end_idx]
                    
                    dist = haversine(start_coord[0], start_coord[1], 
                                    end_coord[0], end_coord[1])
                    
                    if not np.isnan(dist):
                        all_distances.append(dist)
                        
                        # 分类统计
                        if dist < 100:
                            segment_stats['< 100m'].append(dist)
                        elif dist < 500:
                            segment_stats['100-500m'].append(dist)
                        elif dist < 1000:
                            segment_stats['500-1000m'].append(dist)
                        elif dist < 1500:
                            segment_stats['1000-1500m'].append(dist)
                        elif dist < 2000:
                            segment_stats['1500-2000m'].append(dist)
                        elif dist < 3000:
                            segment_stats['2000-3000m'].append(dist)
                        else:
                            segment_stats['>= 3000m'].append(dist)
                        
                        total_segments += 1
                
                i = j
            else:
                i += 1
    
    # ===== 统计汇总 =====
    if not all_distances:
        print("[警告] 未找到任何缺失段")
        return None
    
    all_distances = np.array(all_distances)
    
    print(f"\n[样本统计]")
    print(f"  总轨迹数: {total_trajs}")
    print(f"  缺失段总数: {total_segments}")
    
    print(f"\n[距离分布统计]")
    print(f"  最小距离: {np.min(all_distances):.2f} 米")
    print(f"  最大距离: {np.max(all_distances):.2f} 米")
    print(f"  平均距离: {np.mean(all_distances):.2f} 米")
    print(f"  中位数: {np.median(all_distances):.2f} 米")
    print(f"  25分位数: {np.percentile(all_distances, 25):.2f} 米")
    print(f"  75分位数: {np.percentile(all_distances, 75):.2f} 米")
    
    print(f"\n[按范围分类]")
    for range_name in ['< 100m', '100-500m', '500-1000m', '1000-1500m', '1500-2000m', '2000-3000m', '>= 3000m']:
        count = len(segment_stats[range_name])
        pct = count / total_segments * 100
        print(f"  {range_name:15} : {count:6d} 个 ({pct:5.2f}%)")
    
    # 新版本范围（100-1500）的覆盖率
    in_range = np.sum((all_distances >= 100) & (all_distances <= 1500))
    in_range_pct = in_range / total_segments * 100
    print(f"\n[新版本范围 100-1500m 覆盖率] : {in_range:6d} 个 ({in_range_pct:5.2f}%)")
    
    # 旧版本范围（0-3000）的覆盖率
    old_range = np.sum((all_distances > 0) & (all_distances < 3000))
    old_range_pct = old_range / total_segments * 100
    print(f"[旧版本范围 0-3000m 覆盖率  ] : {old_range:6d} 个 ({old_range_pct:5.2f}%)")
    
    # 建议的范围（根据数据自适应）
    p90_dist = np.percentile(all_distances, 90)
    print(f"\n[自适应建议] 90分位数: {p90_dist:.2f} 米")
    print(f"  建议范围: 100-{int(p90_dist + 500)} 米")
    
    return {
        'filename': filename,
        'total_segments': total_segments,
        'distances': all_distances,
        'stats': {
            'min': np.min(all_distances),
            'max': np.max(all_distances),
            'mean': np.mean(all_distances),
            'median': np.median(all_distances),
            'p25': np.percentile(all_distances, 25),
            'p75': np.percentile(all_distances, 75),
            'p90': np.percentile(all_distances, 90),
            'coverage_100_1500': in_range_pct,
            'coverage_0_3000': old_range_pct
        }
    }

if __name__ == "__main__":
    print("[*] 开始分析 OSM 距离范围...")
    
    results = {}
    for filename in ["val_input_8.pkl", "val_input_16.pkl"]:
        result = analyze_distances(filename)
        if result:
            results[filename] = result
    
    # ===== 汇总建议 =====
    print(f"\n{'='*60}")
    print("总体分析与建议")
    print(f"{'='*60}")
    
    if results:
        # 合并所有数据
        all_data = np.concatenate([r['distances'] for r in results.values()])
        total_segments = sum(r['total_segments'] for r in results.values())
        
        print(f"\n[全体数据统计] 总缺失段数: {total_segments}")
        print(f"  距离范围: {np.min(all_data):.2f} - {np.max(all_data):.2f} 米")
        print(f"  平均: {np.mean(all_data):.2f} m, 中位数: {np.median(all_data):.2f} m")
        
        p90 = np.percentile(all_data, 90)
        in_new = np.sum((all_data >= 100) & (all_data <= 1500)) / total_segments * 100
        
        print(f"\n[推荐方案]")
        print(f"  新版效率 (100-1500m): {in_new:.2f}% 覆盖率")
        print(f"  根据 90分位数 ({p90:.0f}m)，可调整为: 100-{int(p90 + 300)}")
        
        if in_new > 80:
            print(f"\n✓ 新版范围合理，可采用 100-1500m")
        elif in_new > 60:
            print(f"\n△ 新版范围覆盖率 {in_new:.0f}%，略低但可接受")
            print(f"  建议调整为: 50-{int(p90 + 500)}")
        else:
            print(f"\n✗ 新版范围覆盖率只有 {in_new:.0f}%，建议调整为: 50-{int(p90 + 500)}")
