import pandas as pd
import numpy as np

def remove_outliers_IQRscore(df, column_name):
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = abs(df[column_name]).quantile(0.25)
    Q3 = abs(df[column_name]).quantile(0.75)

    #print min and max values of the column
    print(f"Min: {df[column_name].min()}, Max: {df[column_name].max()}")

    print(f"Q1: {Q1}, Q3: {Q3}")
    
    # Calculate the IQR
    IQR = Q3 - Q1
    
    # Identify upper outliers
    upper_outliers = abs(df[column_name]) > (Q3 + (2 * IQR))
    
    # Remove upper outliers
    df_cleaned = df[~upper_outliers]
    
    print(f"Outliers removed from column '{column_name}'")
    #print the percentage of data removed
    print(f"Percentage of data removed: {100 * (1 - len(df_cleaned) / len(df)):.2f}%")
    return df_cleaned
# Example usage
# if __name__ == "__main__":
#     input_csv = "outputcsv\particle_tracking_1.csv"
#     output_csv = "cleaneddata\particle_tracking_1_cleaned.csv"
#     column_name = 'x_velocity_cm_per_s'
  
#     df = pd.read_csv(input_csv)
#     df_cleaned = remove_outliers_IQRscore(df, column_name)  
#     df_cleaned.to_csv(output_csv, index=False)