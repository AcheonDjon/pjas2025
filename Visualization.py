import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline
import os

# Create a 3D plot
fig = plt.figure(figsize=(48, 32))
ax = fig.add_subplot(111, projection='3d')

# Define the folder path containing CSV files
folder_path = r'C:\Users\manoj\Downloads\ScienceProject\input\src\totaldata'

# Get list of CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
colors = plt.cm.rainbow(np.linspace(0, 1, len(csv_files)))

# Process each CSV file
for file_name, color in zip(csv_files, colors):
    try:
        # Read the CSV file
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path, delimiter=',')
        
        # Select only rows from index 200 to 300
        data = data.iloc[200:300].copy()  # 301 to include index 300
        
        print(f"\nFile: {file_name}")
        print(f"Number of points: {len(data)}")
        print(f"X range: {data['x_cm'].min():.2f} to {data['x_cm'].max():.2f}")
        print(f"Y range: {data['y_cm'].min():.2f} to {data['y_cm'].max():.2f}")
        
        # Calculate velocity magnitude
        data['Velocity'] = np.sqrt(data['x_velocity_cm_per_s']**2 + data['y_velocity_cm_per_s']**2 + data['z_particle_velocity_cm_per_s']**2)
        
        # Skip if too few points remain
        if len(data) < 4:  # Need at least 4 points for cubic spline
            print(f"Skipping {file_name}: Too few points")
            continue
        
        # Prepare data for plotting
        time = data['timestamp']
        x = data['x_cm']
        y = data['y_cm']
        velocity = data['Velocity']
        
        # Interpolate the data to create smooth curves
        num_points = 100
        
        try:
            spl_x = make_interp_spline(time, x, k=3)
            spl_y = make_interp_spline(time, y, k=3)
            spl_velocity = make_interp_spline(time, velocity, k=3)
            
            time_smooth = np.linspace(time.min(), time.max(), num_points)
            x_smooth = spl_x(time_smooth)
            y_smooth = spl_y(time_smooth)
            velocity_smooth = spl_velocity(time_smooth)
            
            # Plot the smooth curve
            ax.plot(x_smooth, y_smooth, velocity_smooth, 
                    color=color, 
                    alpha=0.8, 
                    linewidth=0.6 )
                    
        except Exception as e:
            print(f"Interpolation error for {file_name}: {str(e)}")
            # If interpolation fails, plot raw data points
            ax.plot(x, y, velocity, 
                    color=color, 
                    alpha=0.8, 
                    linewidth=0.6, 
                    label=f'Trajectory {file_name} (raw)')
                
    except Exception as e:
        print(f"Error processing file {file_name}: {str(e)}")
        continue

# Label axes
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Velocity Magnitude')
ax.set_title('Particle Velocities as Functions of X and Y Positions (Indices 200-300)')

# Add legend
ax.legend()

# Set positive minimum for z-axis
ax.set_zlim(bottom=0, top=40)
ax.set_xlim(left=0, right=30)
ax.set_ylim(bottom=-10, top=25)
# Show the plot
plt.show()