import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import os
import glob
import datetime
import math
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import torch.nn.functional as F

def calculate_average_metrics(test_metrics_dir, fold_num):
   
    # 确定当前折叠编号对应的文件名
    target_file = f"evaluation_metrics_{fold_num}.xlsx"
    
    # 检查文件是否存在
    if target_file not in os.listdir(test_metrics_dir):
        print(f"文件 {target_file} 不存在于目录 {test_metrics_dir} 中，无法计算平均指标。")
        return

    # 读取指定文件的内容
    metrics_df = pd.read_excel(os.path.join(test_metrics_dir, target_file))
    
    # 初始化列表存储指标值
    accuracy_list = [metrics_df['Accuracy'].mean()]
    
    field_precision_list = [metrics_df['Field-Precision'].mean()]
    field_recall_list = [metrics_df['Field-Recall'].mean()]
    field_f1_list = [metrics_df['Field-F1 Score'].mean()]
    
    road_precision_list = [metrics_df['Road-Precision'].mean()]
    road_recall_list = [metrics_df['Road-Recall'].mean()]
    road_f1_list = [metrics_df['Road-F1 Score'].mean()]
    
    precision_macro_list = [metrics_df['Precision_macro'].mean()]
    recall_macro_list = [metrics_df['Recall_macro'].mean()]
    f1_macro_list = [metrics_df['F1 Score_macro'].mean()]

    # 创建一个新的数据框存储平均指标
    average_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision_macro', 'Recall_macro', 'F1 Score_macro', 
                   'Field-Precision', 'Field-Recall', 'Field-F1 Score', 
                   'Road-Precision', 'Road-Recall', 'Road-F1 Score'],
        'Average Value': [np.mean(accuracy_list), np.mean(precision_macro_list), 
                          np.mean(recall_macro_list), np.mean(f1_macro_list), 
                          np.mean(field_precision_list), np.mean(field_recall_list),
                          np.mean(field_f1_list), np.mean(road_precision_list), 
                          np.mean(road_recall_list), np.mean(road_f1_list)]
    })

    # 将“Average Value”列中的所有值四舍五入到小数点后四位
    average_metrics['Average Value'] = average_metrics['Average Value'].apply(lambda x: round(x, 4))

    # 保存结果到文件
    output_file = os.path.join(test_metrics_dir, f'average_metrics_{fold_num}.xlsx')
    average_metrics.to_excel(output_file, index=False)
    print(f"Average metrics for fold {fold_num} calculated and saved to {output_file}.")




def load_data(file_name):
    print(f"Loading data from file {file_name}...")
    df = pd.read_excel(file_name)


    feature_columns = ['distance', 'speed', 'speedDiff', 'acceleration', 'bearing', 'bearingDiff', 'bearingSpeed',
                       'bearingSpeedDiff', 'curvature', 'distance_five', 'distance_ten', 'distribution','angle_std','angle_mean']
    features = df[feature_columns].values  #
    features = features.reshape(-1, 1, len(feature_columns))
    print("Input features shape:", features.shape)
    target = df['type'].values

    features = torch.tensor(features).float().to('cuda:1')
    target = torch.tensor(target).long().to('cuda:1')
    print("Data loading completed.")
    return features, target


class GumbelActivation(nn.Module):
    def __init__(self):
        super(GumbelActivation, self).__init__()

    def forward(self, x):
        return torch.exp(-torch.exp(-x))


class VAE(nn.Module):
    def __init__(self, input_dim=14, latent_dim=14):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 28),
            nn.ReLU(),
            nn.Linear(28, 14),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(14, latent_dim)
        self.fc_logvar = nn.Linear(14, latent_dim)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-2, max=2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(x.size(0), -1)
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ConvTEModel(nn.Module):
    def __init__(self, num_classes, vae_latent_dim=14, input_dim=14):
        super(ConvTEModel, self).__init__()
        self.vae = VAE(input_dim=input_dim, latent_dim=vae_latent_dim)


        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size=input_dim if i == 0 else 28, hidden_size=14, num_layers=2,
                    batch_first=True, bidirectional=True) for i in range(2)
        ])

        self.residual_fc = nn.Linear(input_dim, 28)

        self.fc = nn.Linear(28, num_classes)
        self.gumbel_activation = GumbelActivation()

    def forward(self, x):

        x = torch.mean(x, dim=1)
        x, mu, logvar = self.vae(x)
        x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)
        residual = self.residual_fc(x)


        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = x + residual

        output = torch.mean(x, dim=0)
        output = self.fc(output)
        output = torch.sigmoid(output)
        return output


def test(model, fold_num):

    test_accuracy_list = []

    test_precision_macro_list = []
    test_recall_macro_list = []
    test_f1_macro_list = []

    field_precision_list = []
    field_recall_list = []
    field_f1_list = []

    road_precision_list = []
    road_recall_list = []
    road_f1_list = []

    # 定义存储测试结果、图像和热图的文件路径
    test_result_dir = '/home/ubuntu/Data/'
    test_image_dir = '/home/ubuntu/Data/'
    test_heatmap_dir = '/home/ubuntu/Data/'
    test_metrics_dir='/home/ubuntu/Data/'

    print("Starting model testing...")


    test_file_names = glob.glob(
        f"/home/ubuntu/Data/wheat/_10fold/{fold_num}/test/*.xlsx")
    for test_file_name in test_file_names:
        model.eval()
        features_test, target_test = load_data(test_file_name)
        print(f'features_test: {features_test}')
        print(f'target_test: {target_test}')
        with torch.no_grad():
            test_outputs = model(features_test)
            print(f'test_outputs: {test_outputs}')
            test_predicted = torch.argmax(test_outputs, dim=1).cpu().numpy()

            print(f'test_predicted: {test_predicted}')

        target_test_cpu = target_test.cpu().numpy()
        target_test_cpu = target_test_cpu.astype(int)
        print(f'target_test_cpu: {target_test_cpu}')

        cm = confusion_matrix(target_test_cpu, test_predicted)


        plt.figure(figsize=(6, 5), dpi=300)


        ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        annot_kws={'size': 14, 'weight':'bold'},
        linewidths=0.5,
        linecolor='gray'
        )


        ax.set_xlabel('Predicted Label', fontsize=14, weight='bold')
        ax.set_ylabel('True Label', fontsize=14, weight='bold')


        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


        output_path = f'{test_result_dir}{fold_num}/confusionMatrix_{os.path.basename(test_file_name)}'
        plt.savefig(f'{output_path}.svg', format='svg', bbox_inches='tight') # 矢量图
        plt.savefig(f'{output_path}.png', dpi=600, bbox_inches='tight') # 高分辨率PNG

        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight',pad_inches=0.05, dpi=300)  # 影响PDF中栅格元素的精度
        plt.close()


        field_precision = round(precision_score(target_test_cpu, test_predicted, pos_label=1), 5)
        field_recall = round(recall_score(target_test_cpu, test_predicted, pos_label=1), 5)
        field_f1 = round(f1_score(target_test_cpu, test_predicted, pos_label=1), 5)


        field_precision_list.append(field_precision)
        field_recall_list.append(field_recall)
        field_f1_list.append(field_f1)


        road_precision = round(precision_score(target_test_cpu, test_predicted, pos_label=0), 5)
        road_recall = round(recall_score(target_test_cpu, test_predicted, pos_label=0), 5)
        road_f1 = round(f1_score(target_test_cpu, test_predicted, pos_label=0), 5)


        road_precision_list.append(road_precision)
        road_recall_list.append(road_recall)
        road_f1_list.append(road_f1)

        test_accuracy = round(accuracy_score(target_test_cpu, test_predicted), 6)
        test_accuracy_list.append(test_accuracy)


        test_precision_macro = round(precision_score(target_test_cpu, test_predicted, average='macro'), 6)
        test_recall_macro = round(recall_score(target_test_cpu, test_predicted, average='macro'), 6)
        test_f1_macro = round(f1_score(target_test_cpu, test_predicted, average='macro'), 6)


        test_precision_macro_list.append(test_precision_macro)
        test_recall_macro_list.append(test_recall_macro)
        test_f1_macro_list.append(test_f1_macro)


        print(f'road - Precision: {road_precision}')
        print(f'road - Recall: {road_recall}')
        print(f'road - F1 Score: {road_f1}')


        print(f'field - Precision: {field_precision}')
        print(f'field - Recall: {field_recall}')
        print(f'field - F1 Score: {field_f1}')


        print(f'Test Accuracy: {test_accuracy}')
        print(f'Test macro Precision: {test_precision_macro}')
        print(f'Test macro Recall: {test_recall_macro}')
        print(f'Test macro F1 Score: {test_f1_macro}')

        df = pd.read_excel(test_file_name)
        longitude = df['longitude'].values
        latitude = df['latitude'].values
        fig, ax = plt.subplots(figsize=(10, 6), dpi=900)



        colors = np.where(test_predicted == 0, 'blue', 'green')  # 根据预测结果为每个点分配颜色，0 表示道路（蓝色），1 表示田地（绿色）。
        # ax.scatter(longitude, latitude, c=colors, s=5)

        # 在坐标轴上绘制散点图，显示道路和田地的分布。
        ax.scatter(longitude[test_predicted == 0], latitude[test_predicted == 0], c='blue', s=5, label='road')
        ax.scatter(longitude[test_predicted == 1], latitude[test_predicted == 1], c='green', s=5, label='field')

        ax.axis('off')  #
        plt.legend()  #
        plt.savefig(f'{test_image_dir}{fold_num}/{os.path.basename(test_file_name)}.jpg', dpi=900)
        plt.close()

    eval_df = pd.DataFrame({
        'File': [test_file_name.split('/')[-1].split('.')[0] for test_file_name in test_file_names],
        'Accuracy': test_accuracy_list,
        'Precision_macro': test_precision_macro_list,
        'Recall_macro': test_recall_macro_list,
        'F1 Score_macro': test_f1_macro_list,
        'Field-Precision': field_precision_list,
        'Field-Recall': field_recall_list,
        'Field-F1 Score': field_f1_list,
        'Road-Precision': road_precision_list,
        'Road-Recall': road_recall_list,
        'Road-F1 Score': road_f1_list
    })


    eval_df.to_excel(f'{test_metrics_dir}evaluation_metrics_{fold_num}.xlsx', index=False)
    calculate_average_metrics(test_metrics_dir, fold_num)
    print("Model testing completed.")


# 指定折数，加载对应的最佳模型。
fold_num =1
model = torch.load(f'/home/ubuntu/Data//best_model10_{fold_num}.pt')

model = nn.DataParallel(model, device_ids=[1])  #
model = model.to('cuda:1')

test(model, fold_num)  #
