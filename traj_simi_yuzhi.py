import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff, euclidean
from fastdtw import fastdtw
from shapely.geometry import LineString
from pyproj import Transformer
from tqdm import tqdm


def latlon_to_utm(coords):
    transformer = Transformer.from_crs("epsg:4326", "epsg:32650", always_xy=True)  # 修改为你的UTM带
    utm_coords = np.array([transformer.transform(lon, lat) for lon, lat in coords])
    return utm_coords


def trajectory_length(coords_utm):
    if len(coords_utm) < 2:
        return 0
    distances = np.linalg.norm(coords_utm[1:] - coords_utm[:-1], axis=1)
    return np.sum(distances)



#LCSS 匹配长度越大 → 越相似，归一化值 ∈ [0, 1]（越大越相似）。
def lcss(P, Q, epsilon=10):
    """
    Longest Common SubSequence (LCSS) with classic normalization: divide by min(len(P), len(Q))
    """
    n, m = len(P), len(Q)
    dp = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if euclidean(P[i - 1], Q[j - 1]) <= epsilon:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcss_len = dp[n][m]
    norm = min(n, m) if min(n, m) > 0 else 1
    lcss_norm = lcss_len / norm
    return lcss_len, lcss_norm


def frechet_distance(P, Q):
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return float('inf')
    ca = np.zeros((n, m))
    ca[0, 0] = np.linalg.norm(P[0] - Q[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], np.linalg.norm(P[i] - Q[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], np.linalg.norm(P[0] - Q[j]))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                np.linalg.norm(P[i] - Q[j])
            )
    return ca[-1, -1]


def compute_thresholds_from_odd_even_tracks(folder_path, output_excel, log_path):
    results = []

    files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    for file in tqdm(files, desc="Processing trajectories"):
        df = pd.read_excel(os.path.join(folder_path, file))
        road_points = df[df['type'] == 0][['longitude', 'latitude']].to_numpy()
        if len(road_points) < 4:
            continue

        # 奇偶分轨迹
        odd_points = road_points[::2]
        even_points = road_points[1::2]

        # 转换为UTM坐标
        utm_all = latlon_to_utm(road_points)
        utm_odd = latlon_to_utm(odd_points)
        utm_even = latlon_to_utm(even_points)

        # 空间总长度
        total_length = trajectory_length(utm_all)
        if total_length == 0:
            continue





        frechet_dist = frechet_distance(utm_odd, utm_even)
        hausdorff_dist = max(
            directed_hausdorff(utm_odd, utm_even)[0],
            directed_hausdorff(utm_even, utm_odd)[0]
        )
        lcss_len, lcss_norm = lcss(utm_odd, utm_even, epsilon=10)

        # 归一化

        frechet_norm = frechet_dist / total_length
        hausdorff_norm = hausdorff_dist / total_length

        results.append({
            'file': file,
            'frechet': frechet_dist,
            'frechet_norm': frechet_norm,
            'hausdorff': hausdorff_dist,
            'hausdorff_norm': hausdorff_norm,
            'lcss': lcss_len,
            'lcss_norm': lcss_norm
        })

    df_result = pd.DataFrame(results)
    df_result.to_excel(output_excel, index=False)

    # 计算均值
    mean_values = df_result[[ 'frechet', 'frechet_norm','hausdorff', 'hausdorff_norm', 'lcss', 'lcss_norm']].mean()

    # 计算标准差
    std_values = df_result[['frechet', 'frechet_norm','hausdorff', 'hausdorff_norm', 'lcss', 'lcss_norm']].std()

    # 写入日志文件
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("各指标原始值与归一化值的均值：\n\n")
        for metric, value in mean_values.items():
            f.write(f"{metric}: {value:.4f}\n")

        f.write("\n各指标原始值与归一化值的标准差：\n\n")
        for metric, value in std_values.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"\n结果已保存：\nExcel表格：{output_excel}\n日志文件：{log_path}")
    return df_result


folder_path = '/home/ubuntu/Data/'
output_excel = '/home/ubuntu/Data/'
log_path = '/home/ubuntu/Data/hyw/'


compute_thresholds_from_odd_even_tracks(folder_path, output_excel, log_path)














# import os
# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw
# from scipy.spatial.distance import directed_hausdorff
# from pyproj import Transformer
# from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw
# from scipy.spatial.distance import directed_hausdorff, euclidean
# from pyproj import Transformer
# import os
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from scipy.spatial.distance import euclidean
# from scipy.spatial.distance import directed_hausdorff
# from pyproj import Proj, Transformer
# from fastdtw import fastdtw
#
#
# # 经纬度转UTM，返回二维numpy数组，单位米
# def latlon_to_utm(coords):
#     # coords: Nx2 numpy数组，列顺序 [longitude, latitude]
#     transformer = Transformer.from_crs("epsg:4326", "epsg:32650", always_xy=True)  # 这里以50N带为例，实际按区域调整
#     utm_coords = np.array([transformer.transform(lon, lat) for lon, lat in coords])
#     return utm_coords
#
# # 计算轨迹空间长度，单位米
# def trajectory_length(coords_utm):
#     if len(coords_utm) < 2:
#         return 0
#     distances = np.linalg.norm(coords_utm[1:] - coords_utm[:-1], axis=1)
#     return np.sum(distances)
#
# # EDR函数，返回原始值和归一化值（归一化除以较小轨迹点数）
# def edr(P, Q, epsilon=10):
#     n, m = len(P), len(Q)
#     dp = np.zeros((n+1, m+1))
#     for i in range(n+1):
#         dp[i][0] = i
#     for j in range(m+1):
#         dp[0][j] = j
#
#     for i in range(1, n+1):
#         for j in range(1, m+1):
#             if euclidean(P[i-1], Q[j-1]) <= epsilon:
#                 dp[i][j] = dp[i-1][j-1]
#             else:
#                 dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
#
#     edr_dist = dp[n][m]
#     edr_norm = edr_dist / min(n, m) if min(n, m) > 0 else 0
#     return edr_dist, edr_norm
#
# # LCSS函数，返回原始值和归一化值（归一化除以较小轨迹点数）
# def lcss(P, Q, epsilon=10):
#     n, m = len(P), len(Q)
#     dp = np.zeros((n+1, m+1))
#     for i in range(1, n+1):
#         for j in range(1, m+1):
#             if euclidean(P[i-1], Q[j-1]) <= epsilon:
#                 dp[i][j] = dp[i-1][j-1] + 1
#             else:
#                 dp[i][j] = max(dp[i-1][j], dp[i][j-1])
#
#     lcss_len = dp[n][m]
#     lcss_norm = lcss_len / min(n, m) if min(n, m) > 0 else 0
#     return lcss_len, lcss_norm
#
#
# def frechet_distance(P, Q):
#     """计算两条轨迹间的Fréchet距离"""
#     n, m = len(P), len(Q)
#     if n == 0 or m == 0:
#         return float('inf')
#
#     # 初始化距离矩阵
#     ca = np.zeros((n, m))
#     ca[0, 0] = np.linalg.norm(P[0] - Q[0])
#
#     # 填充第一列
#     for i in range(1, n):
#         ca[i, 0] = max(ca[i - 1, 0], np.linalg.norm(P[i] - Q[0]))
#
#     # 填充第一行
#     for j in range(1, m):
#         ca[0, j] = max(ca[0, j - 1], np.linalg.norm(P[0] - Q[j]))
#
#     # 填充剩余部分
#     for i in range(1, n):
#         for j in range(1, m):
#             ca[i, j] = max(
#                 min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
#                 np.linalg.norm(P[i] - Q[j])
#             )
#
#     return ca[-1, -1]
#
#
# # 路径设置
# folder_path = 'D:/Download/project/filed-road-wheatData/data/wheat_traj_simi_yuzhi/'
#
# # save_dir = os.path.join(folder_path, "clusters")  #cluster_save_dir 是聚类结果的保存路径
# # os.makedirs(save_dir, exist_ok=True)
#
# # 存储结果
# metric_list = []
#
# for filename in os.listdir(folder_path):
#     if filename.endswith('.xlsx'):
#         file_path = os.path.join(folder_path, filename)
#         data = pd.read_excel(file_path)
#         data['timestamp'] = pd.to_datetime(data['timestamp'])
#
#         road_data = data[data['type'] == 0].reset_index(drop=True)
#         coords = road_data[['longitude', 'latitude']].to_numpy()
#
#         if len(coords) < 4:
#             continue
#
#         odd_coords = coords[::2]
#         even_coords = coords[1::2]
#
#         # 坐标转 UTM
#         odd_utm = convert_to_utm(odd_coords)
#         even_utm = convert_to_utm(even_coords)
#
#         dtw_val, _ = fastdtw(odd_utm, even_utm, dist=euclidean)
#         dtw_norm = dtw_val / max(len(odd_utm), len(even_utm))
#
#         frechet = frechet_distance(odd_utm, even_utm)
#         haus = max(directed_hausdorff(odd_utm, even_utm)[0], directed_hausdorff(even_utm, odd_utm)[0])
#
#         edr_val, edr_norm = edr(odd_coords, even_coords, epsilon=15)
#         lcss_len, lcss_norm = lcss(odd_coords, even_coords, epsilon=15)
#
#         metric_list.append({
#             'filename': filename,
#             'dtw': dtw_val,
#             'dtw_norm': dtw_norm,
#             'frechet': frechet,
#             'hausdorff': haus,
#             'edr': edr_val,
#             'edr_norm': edr_norm,
#             'lcss': lcss_len,
#             'lcss_norm': lcss_norm
#         })
#
# # 生成 DataFrame
# metric_df = pd.DataFrame(metric_list)
#
# # 计算统计值
# stats = {}
# for col in ['dtw_norm', 'frechet', 'hausdorff', 'edr_norm', 'lcss_norm']:
#     stats[col + '_mean'] = metric_df[col].mean()
#     stats[col + '_std'] = metric_df[col].std()
#
# stats_df = pd.DataFrame([stats])
#
# # 保存结果
# metric_df.to_excel(os.path.join(folder_path, 'trajectory_pairwise_metrics.xlsx'), index=False)
# #metric_df.to_excel("trajectory_pairwise_metrics.xlsx", index=False)
# stats_df.to_excel(os.path.join(folder_path, 'trajectory_metrics_statistics.xlsx'), index=False)
#
# #stats_df.to_excel("trajectory_metrics_statistics.xlsx", index=False)
#
