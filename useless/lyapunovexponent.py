import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def lyapunov_exponent(phase_space, tau, m, n_neighbors=10):
    """
    Calculate the largest Lyapunov exponent using the Rosenstein method.

    Args:
        phase_space (np.ndarray): The input phase space data.
        tau (int): The time delay for embedding.
        m (int): The embedding dimension.
        n_neighbors (int): The number of nearest neighbors to consider. Default is 10.

    Returns:
        float: The estimated largest Lyapunov exponent.
    """
    # Embed the phase space in a higher-dimensional space
    N = len(phase_space)
    embedded = np.array([phase_space[i:i + m * tau:tau].flatten() for i in range(N - (m - 1) * tau)])

    print(embedded)

    # Find the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(embedded)
    print(nbrs)
    distances, indices = nbrs.kneighbors(embedded)

    # Calculate the divergence of nearby trajectories
    divergence = np.zeros((N - (m - 1) * tau, n_neighbors))
    for i in range(N - (m - 1) * tau):
        for j in range(n_neighbors):
            divergence[i, j] = np.linalg.norm(embedded[i] - embedded[indices[i, j]])

    # Calculate the average divergence over time
    avg_divergence = np.mean(np.log(divergence), axis=1)

    # print(avg_divergence)

    # Perform linear regression to estimate the Lyapunov exponent
    t = np.arange(len(avg_divergence))
    coeffs = np.polyfit(t, avg_divergence, 1)
    lyapunov_exp = coeffs[0]

    return lyapunov_exp

def calculate_lyapunov_exponents(datasets, x_velocity_col, y_velocity_col, tau, m):
    """
    Calculate the Lyapunov exponent for multiple datasets using total velocity.

    Args:
        datasets (list of pd.DataFrame): The list of input datasets.
        x_velocity_col (str): The name of the column for x velocity.
        y_velocity_col (str): The name of the column for y velocity.
        tau (int): The time delay for embedding.
        m (int): The embedding dimension.

    Returns:
        list of float: The list of estimated Lyapunov exponents for each dataset.
    """
    lyapunov_exponents = []
    for df in datasets:
        x_velocity = df[x_velocity_col].dropna().values
        y_velocity = df[y_velocity_col].dropna().values
        min_length = min(len(x_velocity), len(y_velocity))
        total_velocity = np.sqrt(x_velocity[:min_length]**2 + y_velocity[:min_length]**2)
        lyapunov_exp = lyapunov_exponent(total_velocity, tau, m)
        lyapunov_exponents.append(lyapunov_exp)
    return lyapunov_exponents

# Example usage
if __name__ == "__main__":
    # Load the datasets
    input_files = [
        r"C:\Users\manoj\Downloads\ScienceProject\input\cleaneddata\1.mp4_cleaned.csv",
        r"C:\Users\manoj\Downloads\ScienceProject\input\cleaneddata\2.mp4_cleaned.csv",
        # Add more file paths as needed
    ]
    datasets = [pd.read_csv(file) for file in input_files]

    # Calculate the Lyapunov exponent for each dataset
    tau = 1  # Time delay
    m = 3    # Embedding dimension
    x_velocity_col = 'x_velocity_cm_per_s'
    y_velocity_col = 'y_velocity_cm_per_s'
    lyapunov_exponents = calculate_lyapunov_exponents(datasets, x_velocity_col, y_velocity_col, tau, m)

    # Print the results
    for i, lyapunov_exp in enumerate(lyapunov_exponents):
        print(f"Lyapunov exponent for dataset {i + 1}: {lyapunov_exp}")