import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def compute_lyapunov(csv_file1, csv_file2, threshold=0.1):
    """
    Compute the Lyapunov exponent from two CSV files containing position, time, 
    and velocity information. The phase space used is total velocity as a function of x and y.

    Parameters:
    - csv_file1: str, path to the first CSV file
    - csv_file2: str, path to the second CSV file
    - threshold: float, proximity threshold to identify nearby initial conditions

    Returns:
    - lyapunov_exponent: float, computed Lyapunov exponent
    """
    # Load the CSV files
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)

    # Ensure both files have the same structure and align timestamps
    df1 = df1.sort_values("timestamp").reset_index(drop=True)
    df2 = df2.sort_values("timestamp").reset_index(drop=True)

    # Extract required columns
    time1 = df1["timestamp"].values
    time2 = df2["timestamp"].values
    x1, y1 = df1["x_cm"].values, df1["y_cm"].values
    x2, y2 = df2["x_cm"].values, df2["y_cm"].values
    vx1, vy1 = df1["x_velocity_cm_per_s"].values, df1["y_velocity_cm_per_s"].values
    vx2, vy2 = df2["x_velocity_cm_per_s"].values, df2["y_velocity_cm_per_s"].values

    # Compute total velocity magnitude
    velocity1 = np.sqrt(vx1**2 + vy1**2)
    velocity2 = np.sqrt(vx2**2 + vy2**2)

    # Identify nearby initial conditions in phase space
    initial_distances = []
    for i in range(len(df1)):
        d0 = np.linalg.norm([x1[i] - x2[i], y1[i] - y2[i]])  # Phase space distance
        if d0 < threshold:
            initial_distances.append((i, d0))

    # Compute distance evolution
    time_series = []
    log_distances = []
    for idx, d0 in initial_distances:
        distances = []
        for t in range(idx, len(df1)):
            dt = np.linalg.norm([velocity1[t] - velocity2[t]])
            distances.append(dt)
        log_distances.append(np.log(np.array(distances) / d0))
        time_series.append(time1[idx:])  # Use time from the first CSV file

    # Average log-distance over all pairs
    log_distances = np.mean(np.array(log_distances), axis=0)

    # Fit to linear model: log(d(t)) = λt
    def linear_model(t, lambda_exp):
        return lambda_exp * t

    popt, _ = curve_fit(linear_model, time_series[0], log_distances)
    lyapunov_exponent = popt[0]

    # Print and plot the results
    print(f"Largest Lyapunov Exponent: {lyapunov_exponent:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(time_series[0], log_distances, label="Log Distance Growth", marker="o")
    plt.plot(time_series[0], linear_model(time_series[0], *popt), label=f"Fit (λ = {lyapunov_exponent:.4f})", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Log(Distance Growth)")
    plt.title("Lyapunov Exponent Calculation")
    plt.legend()
    plt.grid()
    plt.show()

    return lyapunov_exponent
compute_lyapunov(r'input\cleaneddata\1.mp4_cleaned.csv', r'input\cleaneddata\2.mp4_cleaned.csv')