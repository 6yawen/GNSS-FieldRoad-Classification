import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

# 参数
input_folder = '/home/ubuntu/Data/'
output_folder = '/home/ubuntu/Data/'

# 定义日志文件路径（可放在脚本开头）
log_path = os.path.join(input_folder, "adasyn_log.txt")
os.makedirs(output_folder, exist_ok=True)

feature_cols = ['speed', 'bearing']

# 遍历每个 Excel 文件
for file in os.listdir(input_folder):
    if not file.endswith(".xlsx"): ## 如果不是以 .xlsx 结尾的文件，就跳过（比如 .csv、.txt、临时文件等）
        continue

    file_path = os.path.join(input_folder, file)  ## 构造完整路径
    df = pd.read_excel(file_path)  # # 读取 Excel 文件为 DataFrame

    # 格式处理：时间戳转为 float 便于插值
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp_float'] = df['timestamp'].astype('int64') / 1e9

    # 类别检查
    if df['type'].nunique() < 2:
        with open(log_path, "a", encoding="utf-8-sig") as f:
            f.write(f"{file} 只包含单一类别（type={df['type'].unique()[0]}），跳过增强\n")
        df.drop(columns=['timestamp_float'], inplace=True)
        df['timestamp'] = df['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S')
        df.to_excel(os.path.join(output_folder, file), index=False)
        continue

    # 仅使用 speed 和 bearing 做 ADASYN 增强
    X = df[feature_cols].values
    y = df['type'].values

    # 只增强道路点（type=0）
    num_road = (y == 0).sum()
    num_field = (y == 1).sum()
    target_road = int(num_field * 0.7)

    # 道路点不足 2 个，无法插值增强
    if num_road < 2:
        with open(log_path, "a", encoding="utf-8-sig") as f:
            f.write(f"{file} 道路点太少（{num_road}），跳过增强并保存原始文件\n")
        # 删除临时列并保存原始文件
        df.drop(columns=['timestamp_float'], inplace=True)
        df['timestamp'] = df['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S')
        df.to_excel(os.path.join(output_folder, file), index=False)
        continue

    if target_road <= num_road:
        with open(log_path, "a", encoding="utf-8-sig") as f:
            f.write(f"{file} 道路点已足够（{num_road} >= {target_road}），不做增强\n")
        df.drop(columns=["timestamp_float"], inplace=True)
        df['timestamp'] = df['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S')
        df.to_excel(os.path.join(output_folder, file), index=False)
        continue

    # 缩放特征
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 调整 ADASYN 参数防止报错
    k_neighbors = min(5, num_road - 1)
    ada = ADASYN(sampling_strategy={0: target_road}, random_state=42, n_neighbors=k_neighbors)

    try:
        X_res, y_res = ada.fit_resample(X_scaled, y)
    except ValueError as e:
        with open(log_path, "a", encoding="utf-8-sig") as f:
            f.write(f"{file} ADASYN 报错：{str(e)}，已保存原始文件\n")

        # 删除临时列，格式化时间
        df.drop(columns=["timestamp_float"], inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['timestamp'] = df['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S')

        # 保存原始文件到输出目录
        df.to_excel(os.path.join(output_folder, file), index=False)

        continue

    X_res_orig = scaler.inverse_transform(X_res)
    new_rows_mask = (y_res == 0)[len(y):]
    new_features = X_res_orig[len(X):][new_rows_mask]

    # 原始道路点，用于空间插值
    road_df = df[df['type'] == 0].reset_index(drop=True)
    road_X = scaler.transform(road_df[feature_cols].values)

    # ❗如果原始道路点太少，无法插值，跳过保存增强结果
    if len(road_df) < 2:
        with open(log_path, "a", encoding="utf-8-sig") as f:
            f.write(f"{file} 插值失败，原始道路点太少（{len(road_df)}），跳过增强并保存原始轨迹\n")

        df.drop(columns=["timestamp_float"], inplace=True)
        df['timestamp'] = df['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S')
        df.to_excel(os.path.join(output_folder, file), index=False)
        continue

    # 插值新点的时间和位置
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(road_X)
    _, indices = nn.kneighbors(new_features)

    interpolated_points = []
    for i, (idx_pair, new_feat) in enumerate(zip(indices, new_features)):
        p1 = road_df.iloc[idx_pair[0]]
        p2 = road_df.iloc[idx_pair[1]]

        alpha = np.random.uniform(0.3, 0.7)
        timestamp_float = (1 - alpha) * p1['timestamp_float'] + alpha * p2['timestamp_float']
        new_point = {
            'timestamp': datetime.fromtimestamp(timestamp_float).strftime('%Y/%m/%d %H:%M:%S'),
            'latitude': (1 - alpha) * p1['latitude'] + alpha * p2['latitude'],
            'longitude': (1 - alpha) * p1['longitude'] + alpha * p2['longitude'],
            'speed': new_feat[0],
            'bearing': new_feat[1],
            'type': 0
        }
        interpolated_points.append(new_point)

    interpolated_df = pd.DataFrame(interpolated_points)
    df.drop(columns=["timestamp_float"], inplace=True)
    # 确保 timestamp 为 datetime 类型，然后格式化为字符串
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['timestamp'] = df['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S')

    # 合并原始数据和增强数据
    final_df = pd.concat([df, interpolated_df], ignore_index=True)

    # 再次确保合并后的 timestamp 是 datetime 类型（防止 interpolated_df 中是字符串）
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], errors='coerce')

    # 排序
    final_df = final_df.sort_values(by='timestamp').reset_index(drop=True)

    # 格式化为标准字符串
    final_df['timestamp'] = final_df['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S')

    # 保存为 Excel 文件
    final_df.to_excel(os.path.join(output_folder, file), index=False)

    # ✅ 写入日志
    num_field = (df['type'] == 1).sum()
    num_road = (df['type'] == 0).sum()
    num_new_road = len(interpolated_df)
    total_road = num_road + num_new_road


    with open(log_path, "a", encoding="utf-8-sig") as f:
        f.write(f"{file} 处理完成：\n")
        f.write(f"  原始田间点数量：{num_field}\n")
        f.write(f"  原始道路点数量：{num_road}\n")
        f.write(f"  插值增强的道路点数：{num_new_road}\n")
        f.write(f"  增强后的道路点总数：{final_df[final_df['type'] == 0].shape[0]}\n")
        f.write(f"  田间点总数：{final_df[final_df['type'] == 1].shape[0]}\n\n")

    #print(f"{file} 增强完成，新增 {len(interpolated_df)} 个道路点")
