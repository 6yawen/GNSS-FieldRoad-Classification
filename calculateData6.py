import os
from math import sin, cos, sqrt, atan2, radians
import datetime
import pandas as pd
import numpy as np
from datetime import timedelta
from math import isclose
import pyproj

cit1 = 5  # 定义了两个常量 cit1 和 cit2
cit2 = 10


# 计算两点之间的地球表面距离（Haversine 公式）
def calculate_distance(previous_lat, previous_long, current_lat, current_long):
    previous_lat = float(previous_lat)
    previous_long = float(previous_long)
    current_lat = float(current_lat)
    current_long = float(current_long)

    R = 6371000.0  # 地球半径，单位：米

    previous_lat_rad = radians(previous_lat)
    previous_long_rad = radians(previous_long)
    current_lat_rad = radians(current_lat)
    current_long_rad = radians(current_long)

    dlong = current_long_rad - previous_long_rad
    dlat = current_lat_rad - previous_lat_rad

    a = sin(dlat / 2) ** 2 + cos(previous_lat_rad) * cos(current_lat_rad) * sin(dlong / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance


def convert_to_plane_coordinates(longitude, latitude):
    in_proj = pyproj.CRS("EPSG:4326")  # WGS84
    out_proj = pyproj.CRS("EPSG:32633")  # UTM Zone 33N
    transformer = pyproj.Transformer.from_crs(in_proj, out_proj, always_xy=True)

    try:
        point_x, point_y = transformer.transform(longitude, latitude)
        return point_x, point_y
    except Exception as e:
        print(f"Error in coordinate transformation: {e}")
        return None, None


def count_points_in_rectangle_vectorized(current_point, gnss_data, rectangle_length, rectangle_width):
    in_proj = pyproj.CRS("EPSG:4326")
    out_proj = pyproj.CRS("EPSG:32633")
    transformer = pyproj.Transformer.from_crs(in_proj, out_proj, always_xy=True)

    longitudes = np.array([point['longitude'] for point in gnss_data])
    latitudes = np.array([point['latitude'] for point in gnss_data])
    points_x, points_y = transformer.transform(longitudes, latitudes)

    bearing_rad = np.radians(current_point['bearing'])
    direction_vector = np.array([cos(bearing_rad), sin(bearing_rad)])
    perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])

    rectangle_center = np.array([current_point['longitude'], current_point['latitude']])
    rectangle_half_length = rectangle_length / 2
    rectangle_half_width = rectangle_width / 2

    points_vectors = np.vstack((points_x - rectangle_center[0], points_y - rectangle_center[1])).T
    distances = np.linalg.norm(points_vectors, axis=1)

    in_length = np.abs(np.dot(points_vectors, perpendicular_vector)) <= rectangle_half_length
    in_width = (np.dot(points_vectors, direction_vector) >= -rectangle_half_width) & (
                np.dot(points_vectors, direction_vector) <= rectangle_half_width)
    inside_rectangle = in_length & in_width & (distances <= rectangle_half_length)

    count = np.sum(inside_rectangle)

    return count


def calculate_bearing_diff(current_bearing, previous_bearing):
    if current_bearing is None or previous_bearing is None:
        bearingdiff = 0
    else:
        bearing_diff = abs(current_bearing - previous_bearing)
        if bearing_diff > 180:
            bearing_diff = 360 - bearing_diff
    return bearing_diff


def cumulative_sum(num, lst):
    cum_sum = [0 for x in range(len(lst))]
    cum_sum[0] = sum(lst[:1])

    for i in range(1, len(lst)):
        if i < num:
            cum_sum[i] = sum(lst[:i + 1])
        else:
            cum_sum[i] = sum(lst[i - num:i + 1])

    return cum_sum


def calculate_curvature(lat1, lon1, lat2, lon2, lat3, lon3):
    distance1 = calculate_distance(lat1, lon1, lat2, lon2)
    distance2 = calculate_distance(lat2, lon2, lat3, lon3)

    if distance1 == 0 or distance2 == 0:
        return 0

    average_distance = (distance1 + distance2) / 2

    cos_angle = (distance1 ** 2 + distance2 ** 2 - calculate_distance(lat1, lon1, lat3, lon3) ** 2) / (
                2 * distance1 * distance2)
    cos_angle = min(1, max(cos_angle, -1))
    angle = np.arccos(cos_angle)

    curvature = angle / average_distance if average_distance != 0 else 0
    return curvature


def calculate_parallel_feature(df, rectangle_length, rectangle_width):
    angle_std_list = []
    angle_mean_list = []

    # 提取UTM坐标
    df[['x', 'y']] = df.apply(lambda row: pd.Series(convert_to_plane_coordinates(row['longitude'], row['latitude'])),
                              axis=1)

    for i in range(len(df)):
        current_point = df.iloc[i]
        current_bearing = current_point['bearing']
        center_x, center_y = current_point['x'], current_point['y']

        # 定义矩形框范围
        x_min, x_max = center_x - rectangle_length / 2, center_x + rectangle_length / 2
        y_min, y_max = center_y - rectangle_width / 2, center_y + rectangle_width / 2

        # 选取在矩形框内的其他轨迹点
        points_in_rectangle = df[
            (df['x'] >= x_min) &
            (df['x'] <= x_max) &
            (df['y'] >= y_min) &
            (df['y'] <= y_max)
            ]

        angle_diffs = []
        for j, other_point in points_in_rectangle.iterrows():
            if j == i:
                continue  # 跳过自己
            other_bearing = other_point['bearing']
            angle_diff = calculate_bearing_diff(current_bearing, other_bearing)
            angle_diffs.append(angle_diff)

        if angle_diffs:
            angle_std = np.std(angle_diffs)
            angle_mean = np.mean(angle_diffs)
        else:
            angle_std = 0
            angle_mean = 0

        angle_std_list.append(angle_std)
        angle_mean_list.append(angle_mean)

    df['angle_std'] = angle_std_list
    df['angle_mean'] = angle_mean_list

    return df


def calculate_radian_and_mean_length(df):
    radian_list = []
    mean_length_list = []

    for i in range(len(df)):
        if i == 0:
            # 只有下一个点
            p1, p2 = df.iloc[i], df.iloc[i + 1]
            radian = calculate_angle(p1, p2)
            mean_length = calculate_distance(p1['latitude'], p1['longitude'], p2['latitude'], p2['longitude'])
        elif i == len(df) - 1:
            # 只有上一个点
            p1, p2 = df.iloc[i - 1], df.iloc[i]
            radian = calculate_angle(p1, p2)
            mean_length = calculate_distance(p1['latitude'], p1['longitude'], p2['latitude'], p2['longitude'])
        else:
            # 同时具有前一个和下一个点
            p0, p1, p2 = df.iloc[i - 1], df.iloc[i], df.iloc[i + 1]
            angle1 = calculate_angle(p0, p1)
            angle2 = calculate_angle(p1, p2)
            radian = abs(angle2 - angle1)
            length1 = calculate_distance(p0['latitude'], p0['longitude'], p1['latitude'], p1['longitude'])
            length2 = calculate_distance(p1['latitude'], p1['longitude'], p2['latitude'], p2['longitude'])
            mean_length = (length1 + length2) / 2

        radian_list.append(radian)
        mean_length_list.append(mean_length)

    df['radian'] = radian_list
    df['mean_length'] = mean_length_list

    return df


def calculate_angle(p1, p2):

    dlon = radians(p2['longitude'] - p1['longitude'])
    dlat = radians(p2['latitude'] - p1['latitude'])
    return atan2(dlat, dlon)



def calFeature(path, final_path):
    rectangle_length = 40
    rectangle_width = 20

    # 只获取 .xlsx 文件
    files = [f for f in os.listdir(path) if f.endswith('.xlsx')]

    for i in range(len(files)):
        data = pd.read_excel(os.path.join(path, files[i]))

        timestamp_list = data['timestamp']
        longitude_list = data['longitude']
        latitude_list = data['latitude']
        speed_list = data['speed']
        bearing_list = data['bearing']
        type_list = data['type']


        time_diff = [0] * len(speed_list)
        speed_diff = [0] * len(speed_list)
        acceleration_list = [0] * len(speed_list)
        bearing_speed = [0] * len(speed_list)
        bearing_acceleration = [0] * len(speed_list)
        bearing_diff = [0] * len(speed_list)
        bearing_speed_diff = [0] * len(speed_list)
        distance_list = [0] * len(speed_list)
        distribution_list = [0] * len(timestamp_list)
        gnss_data = [{'longitude': lon, 'latitude': lat} for lon, lat in zip(longitude_list, latitude_list)]
        curvature_list = [0] * len(speed_list)


        for a in range(len(speed_list)):
            if a == 0:
                distance_list[0] = 0
                time_diff[0] = 0
            else:
                distance_list[a] = round(
                    calculate_distance(latitude_list[a - 1], longitude_list[a - 1], latitude_list[a],
                                       longitude_list[a]), 2)
                d1 = datetime.datetime.strptime(str(timestamp_list[a - 1]), '%Y/%m/%d %H:%M:%S')
                d2 = datetime.datetime.strptime(str(timestamp_list[a]), '%Y/%m/%d %H:%M:%S')
                time_diff[a] = (d2 - d1).total_seconds()


        for c in range(len(speed_list)):
            if c == 0:
                speed_diff[c] = 0
                acceleration_list[c] = 0
                bearing_diff[c] = 0
            else:
                speed_diff[c] = speed_list[c] - speed_list[c - 1]
                acceleration_list[c] = round(float(speed_diff[c]) / time_diff[c], 3) if time_diff[c] != 0 else 0
                speed_diff[c] = abs(speed_diff[c])
                bearing_diff[c] = calculate_bearing_diff(bearing_list[c], bearing_list[c - 1])


        for d in range(len(speed_list)):
            if d == 0:
                bearing_speed[d] = 0
            else:
                if speed_list[d] == 0:
                    bearing_speed[d] = 0
                else:
                    bearing_speed[d] = round(bearing_list[d] / time_diff[d], 3) if time_diff[d] != 0 else 0


        for e in range(len(speed_list)):
            if e == 0:
                bearing_speed_diff[e] = 0
                bearing_acceleration[e] = 0
            else:
                bearing_speed_diff[e] = round(bearing_speed[e] - bearing_speed[e - 1], 4)
                bearing_acceleration[e] = round(bearing_speed_diff[e] / time_diff[e], 4) if time_diff[e] != 0 else 0
                bearing_speed_diff[e] = abs(bearing_speed_diff[e])


        for f in range(len(timestamp_list)):
            point_x = longitude_list[f]
            point_y = latitude_list[f]
            bearing1 = bearing_list[f]
            x, y = convert_to_plane_coordinates(point_x, point_y)
            if x is not None and y is not None:
                current_point = {'longitude': x, 'latitude': y, 'bearing': bearing1}
                distribution_list[f] = count_points_in_rectangle_vectorized(current_point, gnss_data, rectangle_length,
                                                                            rectangle_width)
            else:
                distribution_list[f] = 0
            print(f"{f}-{distribution_list[f]}")

        # 计算累积距离
        distance_five = cumulative_sum(cit1, distance_list)
        distance_ten = cumulative_sum(cit2, distance_list)


        columns = ['timestamp', 'timeDiff', 'longitude', 'latitude', 'distance', 'speed', 'speedDiff', 'acceleration',
                   'bearing', 'bearingDiff', 'bearingSpeed', 'bearingSpeedDiff', 'bearingAcceleration',
                   'type', 'curvature', 'distance_five', 'distance_ten', 'distribution']

        df = pd.DataFrame({
            "timestamp": timestamp_list,
            "timeDiff": time_diff,
            "longitude": longitude_list,
            "latitude": latitude_list,
            "distance": distance_list,
            "speed": speed_list,
            "speedDiff": speed_diff,
            "acceleration": acceleration_list,
            "bearing": bearing_list,
            "bearingDiff": bearing_diff,
            "bearingSpeed": bearing_speed,
            "bearingSpeedDiff": bearing_speed_diff,
            "bearingAcceleration": bearing_acceleration,
            "type": type_list,
            "distance_five": distance_five,
            "distance_ten": distance_ten,
            "distribution": distribution_list
        })

        # 计算曲率
        curvature_list = [0]
        for j in range(1, len(df) - 1):
            lat1, lon1 = df.iloc[j - 1]['latitude'], df.iloc[j - 1]['longitude']
            lat2, lon2 = df.iloc[j]['latitude'], df.iloc[j]['longitude']
            lat3, lon3 = df.iloc[j + 1]['latitude'], df.iloc[j + 1]['longitude']
            curvature = calculate_curvature(lat1, lon1, lat2, lon2, lat3, lon3)
            curvature = round(curvature, 2)
            curvature_list.append(curvature)
        curvature_list.append(0)
        df['curvature'] = curvature_list


        df = calculate_parallel_feature(df, rectangle_length, rectangle_width)


        df=calculate_radian_and_mean_length(df)



        df.drop(['x', 'y'], axis=1, inplace=True)

        # 保存结果到新的 Excel 文件
        df.to_excel(os.path.join(final_path, files[i]), index=False)


if __name__ == "__main__":
    path = "/home/ubuntu/Data//"
    final_path = "/home/ubuntu/Data/"

    calFeature(path, final_path)



