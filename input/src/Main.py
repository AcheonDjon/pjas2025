import particletracking
import outliers
import os
import re

def natural_sort_key(s):
    """
    Create a key for sorting strings with numbers naturally.
    So '2.mp4' comes before '10.mp4'
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

folder_pathleft = r"input\src\sideveiw"
# 1:27, 3:23 4:52,6:39.5,7:28, 8:90,
# starttimes = [27, 18, 52, 39.5, 30.6, 31, 23, 53, 42, 28, 90]
starttimes = [27, 25, 49.2, 41.7, 30, 87.3, 12,51.5,42.5,31,31.5]
# Get sorted filenames using natural sorting
filenames = sorted(os.listdir(folder_pathleft), key=natural_sort_key)
print("Files will be processed in this order:", filenames)

for idx, filename in enumerate(filenames):
    file_path = os.path.join(folder_pathleft, filename)
    if os.path.isfile(file_path):
        print(f"\nProcessing file {idx+1}/{len(filenames)}: {file_path}")
        try:
            df = particletracking.track_red_particle(file_path, starttimes[idx])
            df = outliers.remove_outliers_IQRscore(df, 'x_velocity_cm_per_s')
            df = outliers.remove_outliers_IQRscore(df, 'y_velocity_cm_per_s')
            output_filename = f"input\\src\\datafront\\{filename}ty.csv"
            df.to_csv(output_filename, index=False)
            
            print(f"Processing completed successfully!")
            print("First few rows of processed data:")
            print(df.head())
            
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {str(e)}")
# file_path = r"C:\Users\manoj\Downloads\side1.mp4"
# df = particletracking.track_red_particle(file_path, start_time=27.41)
# df = outliers.remove_outliers_IQRscore(df, 'x_velocity_cm_per_s')
# df = outliers.remove_outliers_IQRscore(df, 'y_velocity_cm_per_s')
# filename = 1
# df.to_csv(f"datafront\{filename}.csv", index=False)