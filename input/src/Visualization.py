import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline

# Calculate velocity magnitude
data = pd.read_csv(r'input\cleaneddata\1.mp4_cleaned.csv', delimiter=',')
data['Velocity'] = np.sqrt(data['x_velocity_cm_per_s']**2 + data['y_velocity_cm_per_s']**2)  # Optional to add z component if needed

# Prepare data for plotting
time = data['timestamp']
x = data['x_cm']
y = data['y_cm']
velocity = data['Velocity']
# Create a 3D plot
fig = plt.figure(figsize=(48, 32))
ax = fig.add_subplot(111, projection='3d')
# Interpolate the data to create smooth curves
num_points = 600  # Number of points for the smooth curve
spl_x = make_interp_spline(time, x, k=3)
spl_y = make_interp_spline(time, y, k=3)
spl_velocity = make_interp_spline(time, velocity, k=3)

time_smooth = np.linspace(time.min(), time.max(), num_points)
x_smooth = spl_x(time_smooth)
y_smooth = spl_y(time_smooth)
velocity_smooth = spl_velocity(time_smooth)

# Plot a smooth curve connecting points in time order
ax.plot(x_smooth, y_smooth, velocity_smooth, color='blue', alpha=0.8, linewidth=0.6, label='Trajectory')


# Label axes
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Velocity Magnitude')
ax.set_title('Particle Velocity as a Function of X and Y Positions')

# Show the plot
plt.show()
