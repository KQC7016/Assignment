#%% Import Libraries

import os
import numpy as np
from pandas import DataFrame
from pandas import Series
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


#%% Function Definition

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(input_data, lag=1):
    df = DataFrame(input_data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a difference series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert difference value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    # Convert the input data into a two-dimensional array
    train = np.array(train)
    test = np.array(test)

    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)

    # transform train
    train_scaled = scaler.transform(train)

    # transform test
    test_scaled = scaler.transform(test)

    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, x, value):
    # Make sure X is a list
    if isinstance(x, np.ndarray):
        x = x.tolist()

    new_row = [x for x in x] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        super(TimeSeriesDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_sequence, target_value = self.data[idx, :-1], self.data[idx, -1]
        input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(-1)
        target_value = torch.tensor(target_value, dtype=torch.float32)
        return input_sequence, target_value


# Define LSTM model using PyTorch
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Training function
def train_model(model, device, train_loader, criterion, optimizer, num_epochs, save_path='model'):
    model.train()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            inputs = inputs.view(-1, 1, inputs.size(-2))  # 改变inputs的维度
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update()

        # 保存模型
        torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch + 1}.pth'))


# make a one-step forecast
def forecast_lstm(model, device, x):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x).unsqueeze(0).unsqueeze(-1).float().to(device)
        x = x.view(1, 1, -1)  # 确保x的维度正确
        yhat = model(x)
    return yhat.item()
