#%% Import Libraries

# Standard Library
import os
import time

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Custom Modules
import utils
import lstm

# Check if GPU is available and set memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Using GPU: {physical_devices}")
else:
    print("No GPU available, using CPU.")


# Create a folder to save the images
if not os.path.exists('PNG'):
    os.makedirs('PNG')

plt.ioff()


#%% Import Dataset & Basic Data Analysis

# Import the dataset.
data = utils.import_data('dataset/T1.csv')

# Set option to show all columns.
pd.set_option('display.max_columns', None)


# Simple data analysis
utils.basic(data)

# Calculate and plot the correlation matrix between attributes.
utils.correlation(data)

# Plot windrose plots.
utils.windrose(data, 'Wind Direction (°)', 'Wind Speed (m/s)')

# Plot kernel density estimate plots.
utils.kde(data)


#%% Time Series Analysis

# Split column "Date/Time" into year, month, day, hour, and minute.
data['Year'] = data['Date/Time'].apply(lambda x: time.strptime(x, "%d %m %Y %H:%M")[0])
data['Month'] = data['Date/Time'].apply(lambda x: time.strptime(x, "%d %m %Y %H:%M")[1])
data['Day'] = data['Date/Time'].apply(lambda x: time.strptime(x, "%d %m %Y %H:%M")[2])
data['Hours'] = data['Date/Time'].apply(lambda x: time.strptime(x, "%d %m %Y %H:%M")[3])
data['Minute'] = data['Date/Time'].apply(lambda x: time.strptime(x, "%d %m %Y %H:%M")[4])
data.head(10)


# Convert column "Date/Time" to proper format.
data["Date/Time"] = pd.to_datetime(data["Date/Time"], format="%d %m %Y %H:%M", errors='coerce')


# Plot power over time
utils.para_over_time(data, "Date/Time", 'Theoretical_Power_Curve (KWh)', 5)

# Plot wind speed over time
utils.para_over_time(data, "Date/Time", 'Wind Speed (m/s)', 6)


#%% LSTM - Hardcode All Variables

# Number of data passed to the program for training at a time.
batch_size_exp = 1

# Number of epoch.
epoch_exp = 15

# Number of neurons.
neurons_exp = 10

# Number of future time steps wish to predict.
predict_values_exp = 1000

# Number of lag time steps.
lag_exp = 24

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


#%% LSTM - Preprocessing Data

# Remove unnecessary columns.
del data['LV ActivePower (kW)']
del data['Wind Speed (m/s)']
del data['Wind Direction (°)']
columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']
for column in columns:
    if column in data.columns:
        del data[column]
# print(data)

# Ensure all columns are numeric, and drop rows with NaN values.
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()

# Delete the last few rows of data.
for i in range(0, 10):
    data = data[:-1]
# print(data.tail())

# Differential Operation.
raw_values = data['Theoretical_Power_Curve (KWh)'].values
diff_values = lstm.difference(raw_values, 1)

# Convert to supervised learning format.
supervised = lstm.timeseries_to_supervised(diff_values, lag_exp)
supervised_values = np.array(supervised)

# print("\n[log_15]Shape of supervised values:", supervised_values.shape)
# print("\n[log_15]First few rows of supervised values:", supervised_values[:5])

# Divide the training set and test set.
train, test = supervised_values[0:-predict_values_exp], supervised_values[-predict_values_exp:]

# Standardizing Data.
# print("\n[log_15]Train shape before scaling:", train.shape)
# print("\n[log_15]Test shape before scaling:", test.shape)

scaler, train_scaled, test_scaled = lstm.scale(train, test)

# print("\n[log_15]Train shape after scaling:", train_scaled.shape)
# print("\n[log_15]Test shape after scaling:", test_scaled.shape)


#%% LSTM - Train the model and make predictions

# fit the model.
lstm_model = lstm.fit_lstm(train_scaled, batch_size_exp, epoch_exp, neurons_exp)

for i in range(len(test_scaled)):
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = lstm.forecast_lstm(lstm_model, 1, X)
    test_pred = [yhat] + test_pred
    if len(test_pred) > lag_exp+1:
        test_pred = test_pred[:-1]
    if i+1 < len(test_scaled):
        if i+1 > lag_exp+1:
            test_scaled[i+1] = test_pred
        else:
            test_scaled[i+1] = np.concatenate((test_pred, test_scaled[i+1, i+1:]), axis=0)

    yhat = lstm.invert_scale(scaler, X, yhat)
    yhat = lstm.inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    expected = raw_values[len(train) + i + 1]
    predictions_plot.append(yhat)
    expectations_plot.append(expected)
    if expected != 0:
        predictions.append(yhat)
        expectations.append(expected)
    print('\n[log_13]Hour=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

lstm_model.summary()


#%% LSTM - Compute and Plot the Error

# Calculation error.
expectations = np.array(expectations)
predictions = np.array(predictions)
print("\n[log_14]Mean Absolute Percent Error: ", (np.mean(np.abs((expectations - predictions) / expectations))))

# Plot Predicted Values VS Actual Values.
utils.compare(expectations_plot, predictions_plot)
