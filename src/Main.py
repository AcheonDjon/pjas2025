import particletracking
import Velocity
import os
folder_pathleft = r"C:\Users\manoj\Downloads\ScienceProject\videos"

for filename in os.listdir(folder_pathleft):
    file_path = os.path.join(folder_pathleft, filename)
    if os.path.isfile(file_path):
        print(f"Processing file: {file_path}")
        try:
            df = particletracking.track_red_particle(file_path, start_time=0)
            df = Velocity.vernier_velocity(df, 'x_cm', 'timestamp')
            df = Velocity.vernier_velocity(df, 'y_cm', 'timestamp')

            print("Tracking completed successfully!")
            print(df.head())
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {str(e)}")
