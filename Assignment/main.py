#%% Import Libraries

# Standard Library
import os
import time

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data

# Custom Modules
import utils
import lstm

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Create a folder to save the images
if not os.path.exists('PNG'):
    os.makedirs('PNG')

plt.ioff()


#%% Import Dataset & Basic Data Analysis

# Import the dataset.
dataset = utils.import_data('dataset/T1.csv')

# Set option to show all columns.
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)


# Simple data analysis
utils.basic(dataset)

# Calculate and plot the correlation matrix between attributes.
utils.correlation(dataset)

# Plot windrose plots.
utils.windrose(dataset, 'Wind Direction (°)', 'Wind Speed (m/s)')

# Plot kernel density estimate plots.
utils.kde(dataset)


#%% Time Series Analysis

# Split column "Date/Time" into year, month.
dataset['Year'] = dataset['Date/Time'].apply(lambda x: time.strptime(x, "%d %m %Y %H:%M")[0])
dataset['Month'] = dataset['Date/Time'].apply(lambda x: time.strptime(x, "%d %m %Y %H:%M")[1])
# dataset['Day'] = dataset['Date/Time'].apply(lambda x: time.strptime(x, "%d %m %Y %H:%M")[2])
# dataset['Hour'] = dataset['Date/Time'].apply(lambda x: time.strptime(x, "%d %m %Y %H:%M")[3])
# dataset['Minute'] = dataset['Date/Time'].apply(lambda x: time.strptime(x, "%d %m %Y %H:%M")[4])
print('\n[log_Test]Split "Date/Time" into year, month')
print(dataset.head())


# Convert column "Date/Time" to proper format.
dataset["Date/Time"] = pd.to_datetime(dataset["Date/Time"], format="%d %m %Y %H:%M", errors='coerce')


# Plot power over time
utils.para_over_time(dataset, "Date/Time", 'Theoretical_Power_Curve (KWh)', 5)

# Plot wind speed over time
utils.para_over_time(dataset, "Date/Time", 'Wind Speed (m/s)', 6)


#%% LSTM - Hardcode All Variables

# Number of data passed to the program for training at a time.
batch_size = 1

# Number of epoch.
epoch = 15

# Number of neurons.
neurons = 10

# Number of future time steps wish to predict.
predict_values = 1000

# Number of lag time steps.
lag = 24

# List of predicted values.
predictions = list()

# List of actual values.
expectations = list()

# List of predicted values for plotting.
predictions_plot = list()

# List of actual values for plotting.
expectations_plot = list()

# List of test data for prediction.
test_pred = list()

# Print test log or not. Set to 1 for code testing.
log_Test = 0

# Path of pretrained model.
model_path = 'model/pretrained_model.pth'


#%% LSTM - Preprocessing Data

# Remove unnecessary columns.
del dataset['LV ActivePower (kW)']
del dataset['Wind Speed (m/s)']
del dataset['Wind Direction (°)']
columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']
for column in columns:
    if column in dataset.columns:
        del dataset[column]

if log_Test == 1:
    print('\n[log_Test]Delete unwanted columns')
    print(dataset.head())

# Ensure all columns are numeric, and drop rows with NaN values.
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset = dataset.dropna()

# Delete the last few rows of data.
for i in range(0, 10):
    dataset = dataset[:-1]

if log_Test == 1:
    print('\n[log_Test]Delete last ten columns')
    print(dataset.tail())

if log_Test == 1:
    print("\n[log_Test]Shape of dataset:", dataset.shape)
    print("\n[log_Test]First few dataset:\n", dataset[:5])

# Differential Operation.
raw_values = dataset['Theoretical_Power_Curve (KWh)'].values
if log_Test == 1:
    print("\n[log_Test]Shape of raw values:", raw_values.shape)
    print("\n[log_Test]First few rows of raw values:\n", raw_values[:5])
diff_values = lstm.difference(raw_values, 1)
if log_Test == 1:
    print("\n[log_Test]Shape of diff values:", diff_values.shape)
    print("\n[log_Test]First few rows of diff values:\n", diff_values[:5])

# Convert to supervised learning format.
supervised = lstm.timeseries_to_supervised(diff_values, lag)
supervised_values = np.array(supervised)

if log_Test == 1:
    print("\n[log_Test]Shape of supervised values:", supervised_values.shape)
    print("\n[log_Test]First few rows of supervised values:\n", supervised_values[:5])

# Divide the training set and test set.
train, test = supervised_values[0:-predict_values], supervised_values[-predict_values:]

# Standardizing Data.

if log_Test == 1:
    print("\n[log_Test]Train shape before scaling:", train.shape)
    print("\n[log_Test]Test shape before scaling:", test.shape)

scaler, train_scaled, test_scaled = lstm.scale(train, test)

if log_Test == 1:
    print("\n[log_Test]Train shape after scaling:", train_scaled.shape)
    print("\n[log_Test]Test shape after scaling:", test_scaled.shape)


#%% LSTM - Train the Model and Make Predictions

# Create the model
input_size = train_scaled.shape[1] - 1
model = lstm.LSTM(input_size, neurons, 1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#  Check if exist pretrained model
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Loaded pretrained model. Skipping training.")
else:
    # Train the model
    train_dataset = lstm.TimeSeriesDataset(train_scaled)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    lstm.train_model(model, device, train_loader, criterion, optimizer, epoch)

    # Save Trained model
    torch.save(model.state_dict(), model_path)

for i in range(len(test_scaled)):
    input_sequence, target_value = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = lstm.forecast_lstm(model, device, input_sequence)
    test_pred = [yhat] + test_pred
    if len(test_pred) > lag+1:
        test_pred = test_pred[:-1]
    if i+1 < len(test_scaled):
        if i+1 > lag+1:
            test_scaled[i+1] = test_pred
        else:
            test_scaled[i+1] = np.concatenate((test_pred, test_scaled[i+1, i+1:]), axis=0)

    yhat = lstm.invert_scale(scaler, input_sequence, yhat)
    yhat = lstm.inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    expected = raw_values[len(train) + i + 1]
    predictions_plot.append(yhat)
    expectations_plot.append(expected)
    if expected != 0:
        predictions.append(yhat)
        expectations.append(expected)
    print('\n[log_13]Hour=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))


#%% LSTM - Compute and Plot the Error

# Calculation error.
expectations = np.array(expectations)
predictions = np.array(predictions)
absolute_percent_errors = np.abs((expectations - predictions) / expectations)
mean_absolute_percent_errors = np.mean(absolute_percent_errors)

print("\n[log_14]Mean Absolute Percent Error (MAPE): ", mean_absolute_percent_errors)

# Plot APE and MAPE
utils.ape_mape(absolute_percent_errors, mean_absolute_percent_errors)

# Plot Predicted Values VS Actual Values.
utils.compare(expectations_plot, predictions_plot)


#%%
