import particletracking
import os
import nvelocity
import cv2

video_path = r"C:\Users\manoj\Downloads\side1.mp4"

# Attempt to open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video file at {video_path}")
else:
    print(f"Successfully opened video file at {video_path}")
file_path = r"C:\Users\manoj\Downloads\ScienceProject\videos\side1.mp4"
df = particletracking.track_red_particle(video_path, start_time=26.8)
df = nvelocity.compute_velocity(df, 'x_cm', 'timestamp')
df1 = nvelocity.compute_velocity(df, 'y_cm', 'timestamp')
df1.to_csv(f'particle_tracking_1.csv', index=False)


        
