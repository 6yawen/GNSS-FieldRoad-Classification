import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import directed_hausdorff

#from shapely.geometry import LineString

# from shapely.geometry import LineString
# from shapely.algorithms.frechet_distance import frechet_distance
from pyproj import Proj, Transformer
from fastdtw import fastdtw

#from similaritymeasures import frechet_dist





def latlon_to_utm(coords):

    transformer = Transformer.from_crs("epsg:4326", "epsg:32650", always_xy=True)  #
    utm_coords = np.array([transformer.transform(lon, lat) for lon, lat in coords])
    return utm_coords


def trajectory_length(coords_utm):
    if len(coords_utm) < 2:
        return 0
    distances = np.linalg.norm(coords_utm[1:] - coords_utm[:-1], axis=1)
    return np.sum(distances)






def lcss(P, Q, epsilon=10):

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
        ca[i, 0] = max(ca[i-1, 0], np.linalg.norm(P[i] - Q[0]))

    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], np.linalg.norm(P[0] - Q[j]))

    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]),
                np.linalg.norm(P[i] - Q[j])
            )
    
    return ca[-1, -1]

def evaluate_trajectory_quality(original_folder, enhanced_folder):
    results = []
    files = [f for f in os.listdir(original_folder) if f.endswith('.xlsx')]

    for file in tqdm(files, desc="Evaluating trajectories"):
        df_orig = pd.read_excel(os.path.join(original_folder, file))
        df_enh = pd.read_excel(os.path.join(enhanced_folder, file))


        road_orig = df_orig[df_orig['type'] == 0][['longitude', 'latitude']].to_numpy()
        road_enh = df_enh[df_enh['type'] == 0][['longitude', 'latitude']].to_numpy()

        if len(road_orig) < 2 or len(road_enh) < 2:
            continue

        utm_orig = latlon_to_utm(road_orig)
        utm_enh = latlon_to_utm(road_enh)

        orig_length = trajectory_length(utm_orig)
        if orig_length == 0:
            continue




        frechet_dist = frechet_distance(utm_orig, utm_enh)

        hausdorff_dist = max(
            directed_hausdorff(utm_orig, utm_enh)[0],
            directed_hausdorff(utm_enh, utm_orig)[0]
        )

        lcss_len, lcss_norm = lcss(utm_orig, utm_enh, epsilon=10)

        frechet_norm = frechet_dist / orig_length
        hausdorff_norm = hausdorff_dist / orig_length


        results.append({
            'file': file,
            'frechet': frechet_dist,
            'frechet_norm': frechet_norm,
            'hausdorff': hausdorff_dist,
            'hausdorff_norm': hausdorff_norm,
            'lcss': lcss_len,
            'lcss_norm': lcss_norm
        })

    return pd.DataFrame(results)


# 调用示例
original_folder = '/home/ubuntu/Data/'  #原始轨迹
enhanced_folder = '/home/ubuntu/Data//'  # 增强后的轨迹

df_result = evaluate_trajectory_quality(original_folder, enhanced_folder)


# 目标文件夹路径
output_folder = '/home/ubuntu/Datay/'

# 如果文件夹不存在则自动创建
os.makedirs(output_folder, exist_ok=True)

# 构造完整保存路径
output_path = os.path.join(output_folder, "轨迹增强质量评估结果.xlsx")

# 保存结果
df_result.to_excel(output_path, index=False)

print(f"评估结果已保存到：{output_path}")



# 计算每个指标的均值
mean_values = df_result[['dtw', 'dtw_norm', 'frechet', 'frechet_norm','hausdorff', 'hausdorff_norm','edr', 'edr_norm', 'lcss', 'lcss_norm']].mean()

# 计算每个指标的标准差
std_values = df_result[['dtw', 'dtw_norm', 'frechet', 'frechet_norm','hausdorff', 'hausdorff_norm','edr', 'edr_norm', 'lcss', 'lcss_norm']].std()

# 构造日志内容
log_lines = ["轨迹增强质量评估指标均值及标准差：\n"]
for metric in mean_values.index:
    mean = mean_values[metric]
    std = std_values[metric]
    log_lines.append(f"{metric}: 均值 = {mean:.4f}, 标准差 = {std:.4f}")

# 构造日志文件路径
log_path = os.path.join(output_folder, "评估均值和标准差日志.txt")

# 保存日志到文件
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print(f"指标均值和标准差已保存到：{log_path}")