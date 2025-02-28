import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_fourier_transform(df, column_name, time_column='timestamp'):
    """
    Runs a Fourier Transform on time series data and plots the frequency spectrum.

    Args:
        df (pd.DataFrame): The input dataset.
        column_name (str): The name of the column to perform Fourier Transform on.
        time_column (str): The name of the time column. Default is 'timestamp'.
    """
    # Ensure the time column is in datetime format
    df[time_column] = pd.to_datetime(df[time_column])

    # Set the time column as the index
    df.set_index(time_column, inplace=True)

    # Extract the time series data
    time_series = df[column_name].dropna()

    # Calculate time differences
    time_diffs = (time_series.index[1:] - time_series.index[:-1]).total_seconds()

    # Check for zero time difference to avoid division by zero
    # Drop zero time differences
    time_diffs = time_diffs[time_diffs != 0]

    # Ensure time_diffs is not empty
    if len(time_diffs) == 0:
        raise ValueError("Time differences array is empty. Ensure the time series has more than one data point.")

    # Perform Fourier Transform
    fft_result = np.fft.fft(time_series)
    fft_freq = np.fft.fftfreq(len(time_series), d=time_diffs[0])

    # Plot the frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(fft_freq, np.abs(fft_result))
    plt.title(f'Fourier Transform of {column_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load the dataset
    input_csv = r"C:\Users\manoj\Downloads\ScienceProject\input\cleaneddata\1.mp4_cleaned.csv"
    df = pd.read_csv(input_csv)

    # Run Fourier Transform on the specified column
    run_fourier_transform(df, 'x_velocity_cm_per_s')