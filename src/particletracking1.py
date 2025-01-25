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

def track_red_particle(video_path, start_time=27.41, output_path=None):
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
    
    # Initialize the tracker
    tracker = cv2.TrackerCSRT_create()
    initialized = False
    
    # Initialize the Kalman Filter
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_count / fps
        current_video_timestamp = current_time - start_time
        
        if not initialized:
            # Convert the frame to HSV and create a mask for the red color
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
                x, y, w, h = cv2.boundingRect(largest_contour)
                tracker.init(frame, (x, y, w, h))
                initialized = True
        else:
            # Update the tracker and get the new position
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cx = x + w // 2
                cy = y + h // 2
                
                cx_cm, cy_cm = converter.to_cm((cx, cy))
                
                positions.append((cx_cm, cy_cm))
                frame_numbers.append(frame_count)
                timestamps.append(current_video_timestamp)
                
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
                # Update Kalman Filter with the new measurement
                kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
                
                last_sample_time = current_time
            else:
                # Predict the new position using the Kalman Filter
                prediction = kalman.predict()
                cx, cy = int(prediction[0]), int(prediction[1])
                
                cx_cm, cy_cm = converter.to_cm((cx, cy))
                
                positions.append((cx_cm, cy_cm))
                frame_numbers.append(frame_count)
                timestamps.append(current_video_timestamp)
                
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Draw in red to indicate prediction
        
        # Flip the frame vertically
        frame = cv2.flip(frame, 0)
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Display coordinates and velocities
        if len(positions) > 1:
            x_velocity = (positions[-1][0] - positions[-2][0]) / (timestamps[-1] - timestamps[-2])
            y_velocity = (positions[-1][1] - positions[-2][1]) / (timestamps[-1] - timestamps[-2])
            cv2.putText(frame, f'X: {positions[-1][0]:.2f} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'Y: {positions[-1][1]:.2f} cm', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'X Velocity: {x_velocity:.2f} cm/s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'Y Velocity: {y_velocity:.2f} cm/s', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Tracking', frame)
    
        frame_count += 1
    
        # Add a delay to slow down the display rate
        if cv2.waitKey(100) & 0xFF == ord('q'):
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
    df = dvelocity(df, 'x_cm', 'timestamp')
    df = dvelocity(df, 'y_cm', 'timestamp')

    if output_path:
        df.to_csv(output_path, index=False)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'particle_tracking_{timestamp}.csv', index=False)
    
    return df

if __name__ == "__main__":
    video_path = r"C:\Users\manoj\Downloads\side1.mp4"
    try:
        df = track_red_particle(video_path, start_time=27.41)
        print("Tracking completed successfully!")
        print(df.head())
    except Exception as e:
        print(f"An error occurred: {str(e)}")