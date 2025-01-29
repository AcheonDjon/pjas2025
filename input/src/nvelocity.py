import pandas as pd
import numpy as np

def calculate_velocity_central_difference(dataframe, time_column, position_column):
    """
    Calculates velocity using the central difference method for interior points,
    the forward difference method for the first point, and the backward difference
    method for the last point.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing time and position data.
        time_column (str): The name of the column representing time.
        position_column (str): The name of the column representing position.

    Returns:
        pd.Series: A Pandas Series containing the calculated velocity values.
    """
    # Extract position and time data
    position = dataframe[position_column]
    time = dataframe[time_column]
    
    # Initialize an array for velocities
    velocities = np.zeros_like(position, dtype=float)
    
    # Forward difference for the first point
    velocities[0] = (position.iloc[1] - position.iloc[0]) / (time.iloc[1] - time.iloc[0])
    
    # Central difference for interior points
    for i in range(1, len(position) - 1):
        velocities[i] = (position.iloc[i + 1] - position.iloc[i - 1]) / (time.iloc[i + 1] - time.iloc[i - 1])
    
    # Backward difference for the last point
    velocities[-1] = (position.iloc[-1] - position.iloc[-2]) / (time.iloc[-1] - time.iloc[-2])
    
    return pd.Series(velocities, index=dataframe.index)
