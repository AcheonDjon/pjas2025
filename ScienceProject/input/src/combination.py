import pandas as pd
import os

def merge_csv_folders(folder1_path, folder2_path, output_folder_path, column_to_copy):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # Get list of CSV files from both folders
    files1 = sorted([f for f in os.listdir(folder1_path) if f.endswith('.csv')])
    files2 = sorted([f for f in os.listdir(folder2_path) if f.endswith('.csv')])
    
    # Verify equal number of files
    if len(files1) != len(files2):
        raise ValueError(f"Folders contain different numbers of files. Folder 1: {len(files1)}, Folder 2: {len(files2)}")
    
    # Process each pair of files
    for file1, file2 in zip(files1, files2):
        try:
            # Read both CSVs
            df1 = pd.read_csv(os.path.join(folder1_path, file1))
            df2 = pd.read_csv(os.path.join(folder2_path, file2))
            
            # Verify the column exists in the second file
            if column_to_copy not in df2.columns:
                print(f"Warning: Column '{column_to_copy}' not found in {file2}. Skipping this pair.")
                continue
            
            # Get the length of the shorter dataframe
            min_length = min(len(df1), len(df2))
            
            # Truncate both dataframes to the shorter length
            df1 = df1.iloc[:min_length]
            df2 = df2.iloc[:min_length]
            
            # Add the column from df2 to df1
            df1["z_particle_velocity_cm_per_s"] = df2[column_to_copy]
            
            # Save the merged result
            output_path = os.path.join(output_folder_path, f"merged_{file1}")
            df1.to_csv(output_path, index=False)
            print(f"Successfully merged {file1} and {file2} -> merged_{file1}")
            print(f"Length of output file: {len(df1)} rows (truncated to shorter input file)")
            
        except Exception as e:
            print(f"Error processing {file1} and {file2}: {str(e)}")
            continue

# Example usage
folder1_path = r"C:\Users\manoj\Downloads\ScienceProject\input\src\dataside"
folder2_path = r"C:\Users\manoj\Downloads\ScienceProject\input\src\datafront"
output_folder_path = r"C:\Users\manoj\Downloads\ScienceProject\input\src\totaldata"
column_to_copy = "x_velocity_cm_per_s"  # Replace with the name of the column you want to copy

merge_csv_folders(folder1_path, folder2_path, output_folder_path, column_to_copy)