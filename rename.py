# -*- coding: gbk -*-
import os  #用于处理文件路径和文件夹的创建
import pandas as pd  #用于处理数据操作，如读取 Excel 文件和数据处理

input_folder = '/home/ubuntu/Data/'
output_folder = '/home/ubuntu/Data/'
os.makedirs(output_folder, exist_ok=True)

translation_dict = {'时间': 'timestamp',
                    '纬度': 'latitude',
                    '速度': 'speed',
                    '方向': 'bearing',
                    '标记': 'type'}

selected_columns = ['timestamp', 'longitude', 'latitude', 'speed', 'bearing', 'type']
date_format = "%Y/%m/%d %H:%M:%S"


for file_name in os.listdir(input_folder):
    if file_name.endswith('.xlsx'):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        df = pd.read_excel(input_file_path)
        df = df.rename(columns=translation_dict)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime(date_format)
        df.loc[df['bearing'] == 360, 'bearing'] = 0
        df = df.sort_values(by='timestamp')
        

        time_delete = []
        all_time_point = []
        timestamp = df['timestamp']
        for j in range(len(timestamp)):
            if timestamp[j] in all_time_point:
                time_delete.append(j)
            else:
                all_time_point.append(timestamp[j])


        df = df.drop(time_delete)
        df = df.reset_index(drop=True)
       

        longitude = df['longitude']
        latitude = df['latitude']
        speed = df['speed']      
        space_delete = []
        all_space_point = []
        for k in range(len(speed)):
            point = []
            point = [longitude[k], latitude[k], speed[k]]
            if point in all_space_point:
                space_delete.append(k)
            else:
                all_space_point.append(point)


        df = df.drop(space_delete)
        df = df.reset_index(drop=True)
        df[selected_columns].to_excel(output_file_path, index=False)

