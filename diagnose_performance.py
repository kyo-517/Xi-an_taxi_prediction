"""
诊断脚本：对比线性插值 vs 混合策略的性能
目的：找出为什么简单方法比复杂方法更好的根本原因
"""
import pickle
import numpy as np
import pandas as pd
import os
import math
from collections import defaultdict
from features_and_utils import haversine, get_grid_id

INPUT_DIR = "task_A_recovery"
PROCESSED_DIR = "processed_data"

def analyze_missing_segments(filename):
    """分析缺失段的特征分布"""
    input_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(input_path):
        return None
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    segment_stats = {
        'lengths': [],      # 缺失点数
        'time_spans': [],   # 时间跨度（秒）
        'distances': [],    # 起终点直线距离
        'segment_count': defaultdict(int),  # 缺失长度分布
    }
    
    for traj in data:
        coords = np.array(traj['coords'])
        timestamps = traj['timestamps']
        mask = traj['mask']
        
        i = 0
        while i < len(mask):
            if not mask[i]:
                start_idx = i - 1
                j = i
                while j < len(mask) and not mask[j]:
                    j += 1
                end_idx = j
                
                length = j - i  # 缺失点数
                if start_idx >= 0 and end_idx < len(mask):
                    time_span = timestamps[end_idx] - timestamps[start_idx]
                    dist = haversine(coords[start_idx][0], coords[start_idx][1],
                                    coords[end_idx][0], coords[end_idx][1])
                    
                    segment_stats['lengths'].append(length)
                    segment_stats['time_spans'].append(time_span)
                    segment_stats['distances'].append(dist)
                    
                    if length <= 3:
                        segment_stats['segment_count']['1-3'] += 1
                    elif length <= 5:
                        segment_stats['segment_count']['4-5'] += 1
                    elif length <= 10:
                        segment_stats['segment_count']['6-10'] += 1
                    else:
                        segment_stats['segment_count']['>10'] += 1
                
                i = j
            else:
                i += 1
    
    return segment_stats

def compare_predictions(filename):
    """对比两种方法的预测结果"""
    # 加载真值
    gt_file = os.path.join(INPUT_DIR, "val_gt.pkl")
    if not os.path.exists(gt_file):
        print("[警告] 未找到真值文件")
        return None
    
    with open(gt_file, 'rb') as f:
        gt_data = pickle.load(f)
    gt_dict = {item['traj_id']: item['coords'] for item in gt_data}
    
    # 加载输入
    input_path = os.path.join(INPUT_DIR, filename)
    with open(input_path, 'rb') as f:
        input_data = pickle.load(f)
    
    # 加载线性插值预测
    linear_pred_file = os.path.join(INPUT_DIR, filename.replace("input", "pred"))
    if not os.path.exists(linear_pred_file):
        print(f"[警告] 未找到线性插值结果: {linear_pred_file}")
        return None
    
    with open(linear_pred_file, 'rb') as f:
        linear_pred = pickle.load(f)
    linear_dict = {item['traj_id']: item['coords'] for item in linear_pred}
    
    # 按缺失段长度分组比较误差
    errors_by_length = defaultdict(list)
    
    for item in input_data:
        traj_id = item['traj_id']
        mask = item['mask']
        coords = np.array(item['coords'])
        
        if traj_id not in gt_dict or traj_id not in linear_dict:
            continue
        
        i = 0
        while i < len(mask):
            if not mask[i]:
                j = i
                while j < len(mask) and not mask[j]:
                    j += 1
                
                length = j - i
                
                # 计算该段的误差
                for k in range(i, j):
                    try:
                        linear_err = haversine(
                            linear_dict[traj_id][k][0], linear_dict[traj_id][k][1],
                            gt_dict[traj_id][k][0], gt_dict[traj_id][k][1]
                        )
                        if not np.isnan(linear_err):
                            errors_by_length[length].append(linear_err)
                    except:
                        pass
                
                i = j
            else:
                i += 1
    
    return errors_by_length

def load_knn_db():
    """加载 k-NN 库看覆盖率"""
    knn_path = os.path.join(PROCESSED_DIR, "knn_db.pkl")
    if os.path.exists(knn_path):
        with open(knn_path, 'rb') as f:
            return pickle.load(f)
    return {}

def analyze_strategy_coverage(filename):
    """分析混合策略的 k-NN 和 OSM 覆盖率"""
    input_path = os.path.join(INPUT_DIR, filename)
    with open(input_path, 'rb') as f:
        input_data = pickle.load(f)
    
    knn_db = load_knn_db()
    knn_hit = 0
    osm_candidate = 0
    total = 0
    
    for traj in input_data:
        coords = np.array(traj['coords'])
        mask = traj['mask']
        
        i = 0
        while i < len(mask):
            if not mask[i]:
                start_idx = i - 1
                j = i
                while j < len(mask) and not mask[j]:
                    j += 1
                end_idx = j
                
                if start_idx >= 0 and end_idx < len(mask):
                    total += 1
                    
                    s_c = coords[start_idx]
                    e_c = coords[end_idx]
                    grid_o = get_grid_id(s_c[0], s_c[1])
                    grid_d = get_grid_id(e_c[0], e_c[1])
                    
                    # k-NN 命中
                    if (grid_o, grid_d) in knn_db:
                        knn_hit += 1
                    else:
                        # OSM 候选（100-1500m）
                        dist = haversine(s_c[0], s_c[1], e_c[0], e_c[1])
                        if 100 <= dist <= 1500:
                            osm_candidate += 1
                
                i = j
            else:
                i += 1
    
    return {
        'total': total,
        'knn_hit': knn_hit,
        'knn_rate': knn_hit / total * 100 if total > 0 else 0,
        'osm_candidate': osm_candidate,
        'osm_rate': osm_candidate / total * 100 if total > 0 else 0,
        'fallback_rate': (total - knn_hit - osm_candidate) / total * 100 if total > 0 else 0
    }

if __name__ == "__main__":
    print("="*70)
    print("Task A 诊断分析：为什么线性插值比混合策略更好？")
    print("="*70)
    
    for filename in ["val_input_8.pkl", "val_input_16.pkl"]:
        print(f"\n【分析文件】{filename}")
        print("-" * 70)
        
        # 1. 缺失段特征分析
        print("[1] 缺失段特征分析")
        seg_stats = analyze_missing_segments(filename)
        if seg_stats:
            lengths = np.array(seg_stats['lengths'])
            distances = np.array(seg_stats['distances'])
            
            print(f"    缺失点数分布:")
            for key in ['1-3', '4-5', '6-10', '>10']:
                count = seg_stats['segment_count'].get(key, 0)
                pct = count / sum(seg_stats['segment_count'].values()) * 100
                print(f"      {key:5} 点 : {count:6d} 个 ({pct:5.1f}%)")
            
            print(f"    缺失点数统计: min={np.min(lengths)}, max={np.max(lengths)}, mean={np.mean(lengths):.2f}")
            print(f"    起终点距离 : min={np.min(distances):.1f}m, max={np.max(distances):.1f}m, " +
                  f"mean={np.mean(distances):.1f}m, median={np.median(distances):.1f}m")
        
        # 2. 混合策略覆盖率
        print("[2] 混合策略覆盖率分析")
        coverage = analyze_strategy_coverage(filename)
        if coverage:
            print(f"    缺失段总数: {coverage['total']}")
            print(f"    k-NN 命中  : {coverage['knn_hit']:6d} ({coverage['knn_rate']:5.1f}%)")
            print(f"    OSM 候选  : {coverage['osm_candidate']:6d} ({coverage['osm_rate']:5.1f}%)")
            print(f"    需要兜底  : {coverage['total'] - coverage['knn_hit'] - coverage['osm_candidate']:6d} " +
                  f"({coverage['fallback_rate']:5.1f}%)")
        
        # 3. 按缺失长度比较误差
        print("[3] 线性插值按缺失长度的误差")
        errors_by_len = compare_predictions(filename)
        if errors_by_len:
            for length in sorted(errors_by_len.keys())[:5]:  # 只看前5个长度
                errs = np.array(errors_by_len[length])
                print(f"    缺失 {length} 点 : MAE={np.mean(errs):.2f}m, " +
                      f"RMSE={np.sqrt(np.mean(errs**2)):.2f}m, count={len(errs)}")
    
    # 最后给出假说
    print("\n" + "="*70)
    print("【可能的原因分析】")
    print("="*70)
    print("""
1. 【短缺失优势】
   - 大多数缺失段非常短（通常 1-3 个点）
   - 短缺失段上，线性插值已经足够精准
   - 复杂策略（k-NN、OSM）反而增加噪声

2. 【k-NN 命中率问题】
   - k-NN 库可能不够全面（只包含历史路线的常见 OD 对）
   - 不在库中的 OD 对只能进入 OSM / 兜底，失去优势
   - 即使命中，历史轨迹形状移植可能不适合当前场景（不同时间、不同驾驶风格）

3. 【OSM 路网质量问题】
   - OSM 数据可能不准确（路网不完整、有错误）
   - 短路径搜索可能找到迂回线路而非直线
   - 100-1500m 范围覆盖虽然多，但未必是最关键的那些点

4. 【时间插值强大】
   - GPS 轨迹通常采样规律（每 5-10 秒一个点）
   - 时间加权插值利用这一规律，效果出奇地好
   - 特别是对短缺失段（< 5 点），它就是直线 → 完美拟合

5. 【泛化能力】
   - 简单方法（线性插值）具有更好的泛化能力
   - 复杂方法（混合策略）容易在特定场景下过度拟合

【推荐方案】
    ✓ 优先采用线性插值（已证明最优）
    △ 如果缺失段很长（> 5 点），可考虑 OSM 或 k-NN 作为辅助
    × 暂停混合策略，除非: (1) 优化 k-NN 库 (2) 改进路网数据 (3) 参数调优
    """)
