import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


# Calculate velocity magnitude
data = pd.read_csv('particle_tracking_1s.csv')
data['Velocity'] = np.sqrt(data['x_velocity_cm_per_s']**2 + data['y_velocity_cm_per_s']**2)  # Optional to add z component if needed

# Prepare data for plotting
time = data['timestamp']
x = data['x_cm']
y = data['y_cm']
velocity = data['Velocity']

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot a line connecting points in time order
ax.plot(x, y, velocity, color='blue', alpha=0.8, linewidth=0.4, label='Trajectory')

# Label axes
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Velocity Magnitude')
ax.set_title('Particle Velocity as a Function of X and Y Positions')

# Show the plot
plt.show()
