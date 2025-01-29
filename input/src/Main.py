import particletracking
import outliers
import os
folder_pathleft = "sideviewfolder"
starttimes =[27,83,23,53,62,41,27.4,26,44,32,43.8,24,21,34, 47, 27, 40.2,18, 56.5, 68, 52,39.5,30.6,30.2]
for idx, filename in enumerate(os.listdir(folder_pathleft)):
    file_path = os.path.join(folder_pathleft, filename)
    if os.path.isfile(file_path):
        print(f"Processing file: {file_path}")
        try:
            df = particletracking.track_red_particle(file_path, starttimes[idx])
            df = outliers.remove_outliers_IQRscore(df, 'x_velocity_cm_per_s')
            df = outliers.remove_outliers_IQRscore(df, 'y_velocity_cm_per_s')
            df.to_csv(f"cleaneddata\{filename}_cleaned.csv", index=False)

            print("Processing completed successfully!")
            print(df.head())
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {str(e)}")
# file_path = r"C:\Users\manoj\Downloads\side1.mp4"
# df = particletracking.track_red_particle(file_path, start_time=27.41)
# df = outliers.remove_outliers_IQRscore(df, 'x_velocity_cm_per_s')
# df = outliers.remove_outliers_IQRscore(df, 'y_velocity_cm_per_s')
# filename = 1
# df.to_csv(f"cleaneddata\{filename}_cleaned.csv", index=False)