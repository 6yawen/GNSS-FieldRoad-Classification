


import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

input_folder = '/home/ubuntu/Data/'
output_folder = '/home/ubuntu/Data/'
os.makedirs(output_folder, exist_ok=True)

# 定义日志文件路径（可放在脚本开头）
log_path = os.path.join(output_folder, "smote_log.txt")

for file in os.listdir(input_folder):
    if file.endswith('.xlsx'):
        file_path = os.path.join(input_folder, file)
        df = pd.read_excel(file_path)


        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp_float'] = df['timestamp'].astype('int64') / 1e9

        if df['type'].nunique() < 2:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"{file} 类别不足，跳过增强，仅保存原始文件\n")
            df.drop(columns=['timestamp_float'], inplace=True) # ✅ 删除临时列
            # ✅ 格式化时间戳为 "YYYY/MM/DD HH:MM:SS"
            df['timestamp'] = df['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S')
            df.to_excel(os.path.join(output_folder, file), index=False)
            continue

        feature_cols = ['speed', 'bearing']
        X = df[feature_cols].values
        y = df['type'].values


        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)


        num_field = (y == 1).sum()
        num_road = (y == 0).sum()


        target_road_num = int(num_field * 0.7)


        if target_road_num <= num_road:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"{file} 道路点已足够（{num_road} >= {target_road_num}），不做增强\n")
            df.drop(columns=['timestamp_float'], inplace=True) # ✅ 删除临时列

            # ✅ 格式化时间戳为 "YYYY/MM/DD HH:MM:SS"
            df['timestamp'] = df['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S')
            df.to_excel(os.path.join(output_folder, file), index=False)
            continue

        sm = SMOTE(sampling_strategy={0: target_road_num}, random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X_scaled, y)


        n_new = len(X_resampled) - len(X)
        if n_new == 0:
            print(f"文件 {file} 无需增强。")
            continue


        X_res_orig = scaler.inverse_transform(X_resampled)


        new_rows_mask = (y_resampled == 0)[len(y):]
        new_features = X_res_orig[len(X):][new_rows_mask]


        road_df = df[df['type'] == 0].reset_index(drop=True)
        road_X = scaler.transform(road_df[feature_cols].values)


        nn = NearestNeighbors(n_neighbors=3)
        nn.fit(road_X)
        _, indices = nn.kneighbors(new_features)


        interpolated_points = []
        for i, (idx_pair, new_feat) in enumerate(zip(indices, new_features)):
            p1 = road_df.iloc[idx_pair[0]]
            p2 = road_df.iloc[idx_pair[1]]
            alpha = np.random.uniform(0.3, 0.7)

            new_point = {
                'timestamp_float': (1 - alpha) * p1['timestamp_float'] + alpha * p2['timestamp_float'],
                'latitude': (1 - alpha) * p1['latitude'] + alpha * p2['latitude'],
                'longitude': (1 - alpha) * p1['longitude'] + alpha * p2['longitude'],
                'speed': new_feat[0],
                'bearing': new_feat[1],
                'type': 0
            }
            interpolated_points.append(new_point)

        interpolated_df = pd.DataFrame(interpolated_points)


        interpolated_df['timestamp'] = pd.to_datetime(interpolated_df['timestamp_float'], unit='s')
        interpolated_df.drop(columns=['timestamp_float'], inplace=True)


        df.drop(columns=['timestamp_float'], inplace=True)


        final_df = pd.concat([df, interpolated_df], ignore_index=True)


        final_df = final_df.sort_values(by='timestamp').reset_index(drop=True)


        final_df['timestamp'] = final_df['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S')


        output_path = os.path.join(output_folder, file)
        final_df.to_excel(output_path, index=False)

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"处理完成：{file}\n")
            log_file.write(f" 原始田间点数量：{num_field}\n")
            log_file.write(f"  插值增强的道路点数：{len(interpolated_df)}\n")
            log_file.write(f"  增强后的道路点总数：{len(final_df[final_df['type'] == 0])}\n")
            log_file.write(f"  田间点总数：{len(final_df[final_df['type'] == 1])}\n")
            log_file.write(f"{'-' * 40}\n")
