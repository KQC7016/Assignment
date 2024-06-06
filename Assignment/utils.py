#%% Import Libraries

import os
import re
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from windrose import WindroseAxes


#%% Function Definition
def import_data(relative_path):
    data_path = os.path.abspath(relative_path)
    imported_data = pd.read_csv(data_path)
    print('\nDataset Loaded')
    return imported_data


def basic(input_dataset):
    plt.ioff()

    # Show the first few rows and the last few rows of the dataset.
    print('\n[log_01] First few rows of dataset:')
    print(input_dataset.head())
    print('\n[log_02] Last few rows of dataset:')
    print(input_dataset.tail())

    # Show basic information about the dataset, including data type and whether there are missing values.
    print("\n[log_03] Basic Information about the dataset:")
    print(input_dataset.info())

    # Show descriptive statistics for the data, including the mean, standard deviation, minimum, quartiles, and maximum.
    print("\n[log_04] Statistical summary of the dataset:")
    print(input_dataset.describe())

    # Check the number of missing values in the data.
    print("\n[log_05] Check for missing values:")
    print(input_dataset.isnull().sum())

    # Plot Correlation graph between attributes.
    print("\n[log_06] Plot Correlation Graph between attributes")
    sns.pairplot(input_dataset)
    plt.savefig(f'PNG/[01]Correlation_Graph.png')  # Save the correlation coefficient plot
    plt.show()
    print("\n[log_06] Plot Correlation Graph Done")


def correlation(input_dataset):
    plt.ioff()

    # Select numeric columns.
    numeric_data = input_dataset.select_dtypes(include=['int64', 'float64'])

    # Calculate the correlation matrix.
    correlation_matrix = numeric_data.corr()

    # Print the correlation matrix.
    print("\n[log_07] Print Correlation Matrix between attributes")
    print(correlation_matrix)

    # Plot the correlation matrix.
    print("\n[log_08] Plot Correlation Matrix")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                linewidths=0.5, linecolor='black', annot_kws={"size": 10})
    plt.title('Correlation Matrix')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=45, fontsize=10)
    plt.savefig(f'PNG/[02]Correlation_Matrix.png')  # Save the correlation coefficient plot
    plt.show()
    print("\n[log_08] Plot Correlation Matrix Done")


def windrose(input_dataset, direction_colum, speed_colum):
    plt.ioff()

    print("\n[log_09] Plot Windrose")
    ax = WindroseAxes.from_ax()
    ax.bar(input_dataset[direction_colum], input_dataset[speed_colum], normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    plt.title("Wind Direction (°) VS Wind Speed (m/s)")
    plt.savefig(f'PNG/[03]Windrose.png')  # Save the windrose plot
    plt.show()
    print("\n[log_09] Plot Windrose Done")


def kde(input_dataset):
    plt.ioff()

    print("\n[log_10] Plot Kernel Density Estimate")
    plt.figure(figsize=(10, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        sns.kdeplot(input_dataset.iloc[:, i + 1], fill=True)
        plt.title(input_dataset.columns[i + 1])
    plt.tight_layout()
    plt.savefig(f'PNG/[04]Kernel_Density_Estimate.png')  # Save the kernel density estimate plot
    plt.show()
    print("\n[log_10] Plot Kernel Density Estimate Done")


def para_over_time(input_dataset, x_colum, y_colum, number: int):
    plt.ioff()
    log_num = 11 if number == 5 else 12
    y_save_name = re.sub(r'\(.*?\)', '', y_colum).strip()

    if 'Month' in input_dataset.columns:
        for months in input_dataset['Month'].unique():
            monthly_data = input_dataset[input_dataset['Month'] == months]

            print(f"\n[log_{log_num}] Plot {y_save_name}(Month {str(months).zfill(2)}) VS Time")
            plt.figure(figsize=(16, 9))
            plt.plot(monthly_data[x_colum], monthly_data[y_colum])
            plt.title(f'{y_save_name}(Month {str(months).zfill(2)}) VS Time')
            plt.xlabel(x_colum)
            plt.ylabel(y_colum)
            plt.xticks(rotation=45)
            plt.savefig(f'PNG/[{str(number).zfill(2)}]'
                        f'{y_save_name.replace(" ", "_")}(Month_{str(months).zfill(2)})_VS_Time.png')
            plt.show()
            print(f"\n[log_{log_num}] Plot {y_save_name}(Month {str(months).zfill(2)}) VS Time Done")
            time.sleep(0.1)

    else:
        print("Don't have the month column")

    print(f"\n[log_{log_num}] Plot {y_save_name} VS Time")
    plt.figure(figsize=(160, 10))
    plt.plot(input_dataset[x_colum], input_dataset[y_colum])
    plt.title(f'{y_save_name} VS Time')
    plt.xlabel(x_colum)
    plt.ylabel(y_colum)
    plt.xticks(rotation=45)
    plt.savefig(f'PNG/[{str(number).zfill(2)}]{y_save_name.replace(" ", "_")}_VS_Time.png')
    plt.show()
    print(f"\n[log_{log_num}] Plot {y_save_name} VS Time Done")


def compare(expectations_plot, predictions_plot):
    plt.ioff()

    sns.set_style("whitegrid")
    print(f"\n[log_15] Plot Predicted Values VS Actual Values Done")
    plt.figure(figsize=(20, 10))
    plt.plot(expectations_plot[0:100], label="True")
    plt.plot(predictions_plot[0:100], label="Predicted")
    plt.legend(loc='upper right')
    plt.xlabel("Number of hours")
    plt.ylabel("Power generated by system (kW)")
    plt.savefig(f'PNG/[07]Predicted_Values_VS_Actual_Values.png')
    plt.show()
    print(f"\n[log_15] Plot Predicted Values VS Actual Values Done")
