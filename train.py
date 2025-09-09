import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import math
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import seaborn as sns
import random
import torch
import torch.nn as nn


def load_data(file_name):
    print(f"Loading data from file {file_name}...")
    df = pd.read_excel(file_name)

    feature_columns = ['distance', 'speed', 'speedDiff', 'acceleration', 'bearing', 'bearingDiff', 'bearingSpeed',
                       'bearingSpeedDiff', 'curvature', 'distance_five', 'distance_ten', 'distribution','angle_std','angle_mean']
    features = df[feature_columns].values

    features = features.reshape(-1, 1, len(feature_columns))

    print("Input features shape:", features.shape)
    target = df['type'].values

    features = torch.tensor(features).float().to('cuda:6')
    target = torch.tensor(target).long().to('cuda:6')

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


class FocalLossWithRegularization(nn.Module):

    def __init__(self, alpha, gamma, reduction, regularization_coeff=0.00001):
        super(FocalLossWithRegularization, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.regularization_coeff = regularization_coeff

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')


        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss


        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)


        regularization_term = 0.0
        for param in self.parameters():
            regularization_term += torch.sum(param.pow(2))


        total_loss = focal_loss + self.regularization_coeff * regularization_term

        return total_loss


def train(model, num_epochs, batch_size, patience=10):

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.00001)
    criterion = FocalLossWithRegularization(alpha=0.25, gamma=2, reduction='mean')

    best_model_state = None
    train_loss_history = []
    val_accuracy_history = []

    val_precision_macro_history = []
    val_recall_macro_history = []
    val_f1_macro_history = []

    train_loss_avg_history = []
    val_loss_avg_history = []


    print("Starting model training...")
    train_data = []
    train_file_names = glob.glob(
        f"/home/ubuntu/Data/_10fold/{fold_num}/train/*.xlsx")
    for train_file_name in train_file_names:
        features_train, target_train = load_data(train_file_name)
        train_data.append((features_train, target_train))


    early_stopping_counter = 0
    best_mix_result = 0

    for epoch in range(num_epochs):
        model.train()
        all_file_loss = 0.0
        for train_file_index, (features_train, target_train) in enumerate(train_data, 1):
            current_file_loss = 0.0
            batch_size = features_train.shape[0]

            for i in range(0, len(features_train), batch_size):
                batch_features = features_train.clone().detach().to('cuda:6').float()
                batch_target = target_train.clone().detach().to('cuda:6').long()


                optimizer.zero_grad()
                output = model(batch_features)

                loss = criterion(output, batch_target)  # setting
                loss.backward()
                optimizer.step()

                current_file_loss += loss.item()
            current_file_average_loss = current_file_loss


            print(
                f'Epoch [{epoch + 1}/{num_epochs}], File: {train_file_index}-Training Loss: {current_file_average_loss}')
            all_file_loss += current_file_average_loss
        all_file_average_loss = all_file_loss / len(train_data)
        print(f'Epoch [{epoch + 1}/{num_epochs}], all_file average training Loss: {all_file_average_loss}')

        if (epoch + 1) % 10 == 0:

            train_loss_avg_history.append(all_file_average_loss)

            val_loss_list = []
            val_accuracy_list = []
            val_precision_macro_list = []
            val_recall_macro_list = []
            val_f1_macro_list = []

            field_precision_list = []
            field_recall_list = []
            field_f1_list = []

            road_precision_list = []
            road_recall_list = []
            road_f1_list = []

            # 加载验证数据文件，设定模型为评估模式
            val_file_names = glob.glob(
                f"/home/ubuntu/Data//_10fold/{fold_num}/valid/*.xlsx")
            for val_file_name in val_file_names:
                model.eval()
                features_val, target_val = load_data(val_file_name)
                print(f'features_val: {features_val}')
                print(f'target_val: {target_val}')

                with torch.no_grad():
                    val_outputs = model(features_val)
                    print(f'val_outputs: {val_outputs}')
                    val_predicted = torch.argmax(val_outputs, dim=1).cpu().numpy()
                    print(f'val_predicted: {val_predicted}')
                target_val_cpu = target_val.cpu().numpy()
                target_val_cpu = target_val_cpu.astype(int)
                print(f'target_val_cpu: {target_val_cpu}')

                field_precision = round(precision_score(target_val_cpu, val_predicted, pos_label=1), 5)
                field_recall = round(recall_score(target_val_cpu, val_predicted, pos_label=1), 5)
                field_f1 = round(f1_score(target_val_cpu, val_predicted, pos_label=1), 5)

                field_precision_list.append(field_precision)
                field_recall_list.append(field_recall)
                field_f1_list.append(field_f1)

                road_precision = round(precision_score(target_val_cpu, val_predicted, pos_label=0), 5)
                road_recall = round(recall_score(target_val_cpu, val_predicted, pos_label=0), 5)
                road_f1 = round(f1_score(target_val_cpu, val_predicted, pos_label=0), 5)

                road_precision_list.append(road_precision)
                road_recall_list.append(road_recall)
                road_f1_list.append(road_f1)

                val_loss = criterion(val_outputs, target_val).item()  # setting
                val_accuracy = round(accuracy_score(target_val_cpu, val_predicted), 6)

                val_precision_macro = round(precision_score(target_val_cpu, val_predicted, average='macro'), 6)
                val_recall_macro = round(recall_score(target_val_cpu, val_predicted, average='macro'), 6)
                val_f1_macro = round(f1_score(target_val_cpu, val_predicted, average='macro'), 6)

                val_loss_list.append(val_loss)
                val_accuracy_list.append(val_accuracy)

                val_precision_macro_list.append(val_precision_macro)
                val_recall_macro_list.append(val_recall_macro)
                val_f1_macro_list.append(val_f1_macro)

                print(f'road Precision: {road_precision}')
                print(f'road Recall: {road_recall}')
                print(f'road F1 score: {road_f1}')

                print(f'field Precision: {field_precision}')
                print(f'field Recall: {field_recall}')
                print(f'field F1 score: {field_f1}')

                print(f'Validation Loss: {val_loss}')
                print(f'Validation Accuracy: {val_accuracy}')
                print(f'Validation macro Precision: {val_precision_macro}')
                print(f'Validation macro Recall: {val_recall_macro}')
                print(f'Validation macro F1 score: {val_f1_macro}')

                # 计算平均验证损失和其他指标，并输出。
            val_loss_avg = sum(val_loss_list) / len(val_loss_list)
            val_accuracy_avg = round(sum(val_accuracy_list) / len(val_accuracy_list), 6)
            val_precision_macro_avg = round(sum(val_precision_macro_list) / len(val_precision_macro_list), 7)
            val_recall_macro_avg = round(sum(val_recall_macro_list) / len(val_recall_macro_list), 7)
            val_f1_macro_avg = round(sum(val_f1_macro_list) / len(val_f1_macro_list), 7)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss_avg}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy_avg}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation macro Precision: {val_precision_macro_avg}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation macro Recall: {val_recall_macro_avg}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation macro F1 score: {val_f1_macro_avg}')

            val_accuracy_history.append(val_accuracy_avg)
            val_loss_avg_history.append(val_loss_avg)

            val_precision_macro_history.append(val_precision_macro_avg)
            val_recall_macro_history.append(val_recall_macro_avg)
            val_f1_macro_history.append(val_f1_macro_avg)


            if 0.2 * val_precision_macro_avg + 0.2 * val_recall_macro_avg + 0.3 * val_f1_macro_avg + 0.3 * val_accuracy_avg > best_mix_result:
                best_mix_result = 0.2 * val_precision_macro_avg + 0.2 * val_recall_macro_avg + 0.3 * val_f1_macro_avg + 0.3 * val_accuracy_avg
                best_model_state = model.state_dict()
                early_stopping_counter = 0
                torch.save(model.module,
                           f'/home/ubuntu/Data/best_model10_{fold_num}.pt')
                print(f'Epoch [{epoch + 1}/{num_epochs}], saved')
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print("Early stopping triggered! Training stopped.")
                break

    epochs = range(1, len(val_precision_macro_history) + 1)


    plt.figure(figsize=(20, 10))
    plt.plot(epochs, val_accuracy_history, label='Validation Accuracy')
    plt.plot(epochs, val_precision_macro_history, label='Validation macro Precision')
    plt.plot(epochs, val_recall_macro_history, label='Validation macro Recall')
    plt.plot(epochs, val_f1_macro_history, label='Validation macro F1 Score')

    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Evaluation Metrics')
    plt.legend()
    plt.savefig(
        f'/home/ubuntu/Data/10/metrics_plot1_{fold_num}.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(epochs, train_loss_avg_history, label='Average Training Loss')
    plt.plot(epochs, val_loss_avg_history, label='Average Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'/home/ubuntu/Data/10/Loss_Graph1_{fold_num}.png')
    plt.close()

    print("Model training completed.")


seed_id = 3407
torch.manual_seed(seed_id)
torch.cuda.manual_seed_all(seed_id)
random.seed(seed_id)
np.random.seed(seed_id)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#
model = ConvTEModel(num_classes=2)

model = nn.DataParallel(model, device_ids=[6, 7])


model = model.to('cuda:6')
fold_num =9 # the number of fold is 0-9


train(model, num_epochs=400, batch_size=None)
