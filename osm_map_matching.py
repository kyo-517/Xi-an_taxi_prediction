import os
import numpy as np
import networkx as nx
try:
    import osmnx as ox
except ImportError:
    print("[警告] 未安装 osmnx。请运行 pip install osmnx。路网功能将不可用。")

def get_or_download_xian_graph(filepath="xian_drive.graphml", local_osm="xian_map.osm"):
    """
    获取或加载西安市的机动车道路网
    """
    if os.path.exists(filepath):
        print(f"[*] 发现本地路网缓存 {filepath}，正在加载...")
        G = ox.load_graphml(filepath)
        print("[*] 路网加载成功！")
        return G
    else:
        print(f"[*] 未找到 graphml 缓存，尝试读取本地 OSM 源文件: {local_osm}")
        if not os.path.exists(local_osm):
            print(f"[错误] 找不到本地文件 {local_osm}！请去 BBBike 下载解压并重命名为 {local_osm} 放入该目录。")
            return None
            
        print("[*] 正在解析本地 OSM XML 文件 (这需要一到两分钟，请耐心等待)...")
        # 从你手动下载的本地 .osm 文件中解析路网
        G = ox.graph_from_xml(local_osm, simplify=True)
        
        print(f"[*] 路网解析完成！包含 {len(G.nodes)} 个路口，{len(G.edges)} 条路段。")
        # 保存为 graphml 格式，下次就能秒加载了
        ox.save_graphml(G, filepath)
        print(f"[*] 路网已缓存至本地: {filepath}")
        return G

def route_constrained_interpolation(G, known_points, missing_timestamps):
    """
    【核心算法：受路网拓扑约束的空间时间插值】
    """
    if len(known_points) < 2:
        return []

    # 1. 寻找最近的物理路网节点
    lons = [p[0] for p in known_points]
    lats = [p[1] for p in known_points]
    # 兼容最新版 osmnx
    nodes = ox.nearest_nodes(G, X=lons, Y=lats)
    
    predicted_coords = []
    
    # 遍历每两个相邻的已知点
    for i in range(len(nodes) - 1):
        u = nodes[i]
        v = nodes[i+1]
        t_start = known_points[i][2] # 时间戳
        t_end = known_points[i+1][2]
        
        target_ts = [ts for ts in missing_timestamps if t_start < ts < t_end]
        
        try:
            # 2. 核心：在有向图中寻找最合理的真实行驶路径
            path = nx.shortest_path(G, u, v, weight='length')
            
            path_coords = []
            for node in path:
                path_coords.append((G.nodes[node]['x'], G.nodes[node]['y']))
                
            cum_dist = [0.0]
            for j in range(1, len(path_coords)):
                dx = path_coords[j][0] - path_coords[j-1][0]
                dy = path_coords[j][1] - path_coords[j-1][1]
                dist = np.sqrt(dx**2 + dy**2)
                cum_dist.append(cum_dist[-1] + dist)
                
            total_dist = cum_dist[-1]
            
            # 3. 沿物理道路进行插值
            for ts in target_ts:
                time_ratio = (ts - t_start) / (t_end - t_start) if t_end > t_start else 0
                target_dist = total_dist * time_ratio
                
                for j in range(1, len(cum_dist)):
                    if cum_dist[j] >= target_dist:
                        segment_ratio = (target_dist - cum_dist[j-1]) / (cum_dist[j] - cum_dist[j-1] + 1e-9)
                        lon = path_coords[j-1][0] + segment_ratio * (path_coords[j][0] - path_coords[j-1][0])
                        lat = path_coords[j-1][1] + segment_ratio * (path_coords[j][1] - path_coords[j-1][1])
                        predicted_coords.append([lon, lat])
                        break
        except nx.NetworkXNoPath:
            # 如果两个节点之间由于数据缺失不连通，退化为普通直线插值
            for ts in target_ts:
                time_ratio = (ts - t_start) / (t_end - t_start) if t_end > t_start else 0
                lon = known_points[i][0] + time_ratio * (known_points[i+1][0] - known_points[i][0])
                lat = known_points[i][1] + time_ratio * (known_points[i+1][1] - known_points[i][1])
                predicted_coords.append([lon, lat])
                
    return predicted_coords

if __name__ == "__main__":
    print("=== 初始化路网环境 ===")
    G = get_or_download_xian_graph()