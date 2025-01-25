import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Calculate velocity magnitude
data = pd.read_csv('particle_tracking_1.csv')
data['Velocity'] = np.sqrt(data['x_velocity_cm_per_s']**2 + data['y_velocity_cm_per_s']**2)  # Optional to add z component if needed

# Prepare data for plotting
time = data['timestamp']+ 27.41
x = data['x_cm']
y = data['y_cm']
velocity = data['Velocity']


# Plot a line connecting points in time order
plt.plot(time, velocity, color='blue', alpha=0.8, linewidth=1.5, label='Trajectory')

# Label axes
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Particle Velocity as a Function of Time')

# Show the plot
plt.show()
