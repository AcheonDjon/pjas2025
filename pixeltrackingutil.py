import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert coordinates back to original scale
        original_x = int(x * (original_width / display_width))
        original_y = int(y * (original_height / display_height))
        print(f"Clicked coordinates in original scale: ({original_x}, {original_y})")
        # Draw a small circle at the clicked point
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('Click to find coordinates', frame)

def find_coordinates(video_path):
    global frame, original_width, original_height, display_width, display_height
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Error: Could not open video file")

    # Get original dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate new dimensions (maintaining aspect ratio)
    # Set maximum dimension to 800 pixels
    scale = 800 / max(original_width, original_height)
    display_width = int(original_width * scale)
    display_height = int(original_height * scale)

    # Create window and set mouse callback
    cv2.namedWindow('Click to find coordinates')
    cv2.setMouseCallback('Click to find coordinates', mouse_callback)

    print("Click on points to get coordinates. Press 'q' to quit.")
    print(f"Original dimensions: {original_width}x{original_height}")
    print(f"Display dimensions: {display_width}x{display_height}")
    
    while True:
        ret, original_frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to start
            continue
        
        # Resize frame for display
        frame = cv2.resize(original_frame, (display_width, display_height))
            
        # Draw crosshair lines following cursor
        def draw_crosshair(event, x, y, flags, param):
            img = frame.copy()
            h, w = img.shape[:2]
            # Draw vertical and horizontal lines
            cv2.line(img, (x, 0), (x, h), (255, 0, 0), 1)
            cv2.line(img, (0, y), (w, y), (255, 0, 0), 1)
            # Calculate original scale coordinates
            original_x = int(x * (original_width / display_width))
            original_y = int(y * (original_height / display_height))
            # Display coordinates near cursor
            cv2.putText(img, f'Original: ({original_x}, {original_y})', (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Click to find coordinates', img)
            
        cv2.setMouseCallback('Click to find coordinates', draw_crosshair)
        
        # Show the frame
        cv2.imshow('Click to find coordinates', frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your video path
    video_path = r"C:\Users\manoj\Downloads\ScienceProject\input\src\frontveiw\1f.mp4"
    find_coordinates(video_path)