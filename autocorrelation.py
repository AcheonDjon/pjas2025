import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def plot_autocorrelation_per_second(df, column_name, time_column='timestamp'):
    """
    Plots the autocorrelation graph for each second in the dataset.

    Args:
        df (pd.DataFrame): The input dataset.
        column_name (str): The name of the column to compute autocorrelation for.
        time_column (str): The name of the time column. Default is 'timestamp'.
    """
    # Ensure the time column is in datetime format
    df[time_column] = pd.to_datetime(df[time_column])

    # Set the time column as the index
    df.set_index(time_column, inplace=True)

    # Resample the data to 1-second intervals
    resampled_df = df.resample('1s').mean()

    # Drop rows with NaN values
    resampled_df.dropna(inplace=True)

    # Plot autocorrelation for each second
    for second in resampled_df.index:
        # Extract the data for the current second
        data = resampled_df[column_name].loc[second:second]
        
        # Check if there are more than one data point
        # if len(data) > 1:
        plt.figure(figsize=(10, 6))
        plot_acf(data, lags=40)
        plt.title(f'Autocorrelation for {column_name} at {second}')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load the dataset
    input_csv = r"C:\Users\manoj\Downloads\ScienceProject\input\cleaneddata\1.mp4_cleaned.csv"
    df = pd.read_csv(input_csv)

    # Plot autocorrelation for the specified column
    plot_autocorrelation_per_second(df, 'x_velocity_cm_per_s')