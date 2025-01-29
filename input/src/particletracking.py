import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from nvelocity import calculate_velocity_central_difference as dvelocity

class PixelToCentimeterConverter:
    def __init__(self, pixel_reference=1070, cm_reference=15.75):
        self.scale_factor = cm_reference / pixel_reference

    def to_cm(self, pixels):
        if isinstance(pixels, tuple):
            return (pixels[0] * self.scale_factor, pixels[1] * self.scale_factor)
        return pixels * self.scale_factor

def track_red_particle(video_path, start_time, output_path=None):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Error: Could not open video file")
    
    converter = PixelToCentimeterConverter()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    positions = []
    frame_numbers = []
    timestamps = []
    
    frame_count = start_frame
    last_sample_time = 0
    samples_per_second = 6
    sample_interval = fps / samples_per_second
    
    end_time = start_time + 60  # Run for one minute
    end_frame = int(end_time * fps)

    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_count / fps
        current_video_timestamp = current_time - start_time
        
        # Only sample 3 times per second
        if current_time - last_sample_time >= 1/samples_per_second:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = frame_height - int(M["m01"] / M["m00"])
                    
                    cx_cm, cy_cm = converter.to_cm((cx, cy))
                    
                    positions.append((cx_cm, cy_cm))
                    frame_numbers.append(frame_count)
                    timestamps.append(current_video_timestamp)
                    
                    cv2.circle(frame, (cx, frame_height - cy), 5, (0, 255, 0), -1)
                    
                    last_sample_time = current_time
             
        
        # Flip the frame horizontally
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)
            if len(positions) > 1:
                x_velocity = (positions[-1][0] - positions[-2][0]) / (timestamps[-1] - timestamps[-2])
                y_velocity = (positions[-1][1] - positions[-2][1]) / (timestamps[-1] - timestamps[-2])
                cv2.putText(frame, f'X: {positions[-1][0]:.2f} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f'Y: {positions[-1][1]:.2f} cm', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f'X Velocity: {x_velocity:.2f} cm/s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f'Y Velocity: {y_velocity:.2f} cm/s', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Tracking', frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate velocities
    df = pd.DataFrame({
        'frame': frame_numbers,
        'timestamp': timestamps,
        'x_cm': [pos[0] for pos in positions],
        'y_cm': [pos[1] for pos in positions]
    })
    
    # Calculate velocities
    df['x_velocity_cm_per_s'] = dvelocity(df, 'timestamp', 'x_cm')
    df['y_velocity_cm_per_s'] = dvelocity(df, 'timestamp', 'y_cm')
    
    
    return df

# if __name__ == "__main__":
#     video_path = r"C:\Users\manoj\Downloads\side1.mp4"
#     try:
#         df = track_red_particle(video_path, start_time=27.41)
#         print("Tracking completed successfully!")
#         print(df.head())
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         df.to_csv(f'particle_tracking_{timestamp}.csv', index=False)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")