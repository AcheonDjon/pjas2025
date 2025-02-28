import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

def lorenz_system(state, t, sigma=10, rho=28, beta=8/3):
    """Calculate the Lorenz system derivatives."""
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def generate_lorenz_data(t_span=100, n_points=10000, initial_state=[1.0, 1.0, 1.0], 
                        params=(10, 28, 8/3)):
    """Generate data from Lorenz system."""
    t = np.linspace(0, t_span, n_points)
    trajectory = odeint(lorenz_system, initial_state, t, args=params)
    return trajectory, t

def plot_lorenz_attractor(trajectory, title="Lorenz Attractor"):
    """Plot the 3D Lorenz attractor."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            lw=0.5, alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

def calculate_lyapunov_with_viz(time_series1, time_series2, embedding_dimension=2, delay=1,
                               include_lorenz=True):
    """Calculate Lyapunov exponent with visualizations and optional Lorenz analysis."""
    
    # 1. Plot original phase space
    print("Step 1: Visualizing original phase space")
    plt.figure(figsize=(10, 8))
    plt.scatter(time_series1, time_series2, c=np.arange(len(time_series1)), 
                cmap='viridis', alpha=0.5)
    plt.colorbar(label='Time')
    plt.xlabel('Time Series 1')
    plt.ylabel('Time Series 2')
    plt.title('Original Phase Space')
    plt.show()

    # Optional: Generate and plot Lorenz data for comparison
    if include_lorenz:
        print("\nGenerating Lorenz system data for comparison...")
        lorenz_traj, _ = generate_lorenz_data()
        plot_lorenz_attractor(lorenz_traj)
        
        # Plot Lorenz x-y projection for comparison
        plt.figure(figsize=(10, 8))
        plt.scatter(lorenz_traj[:, 0], lorenz_traj[:, 1], 
                   c=np.arange(len(lorenz_traj)), cmap='viridis', alpha=0.5)
        plt.colorbar(label='Time')
        plt.xlabel('Lorenz X')
        plt.ylabel('Lorenz Y')
        plt.title('Lorenz System X-Y Projection')
        plt.show()
    
    # 2. Create and visualize embedding
    def create_embedding(data, dimension, tau):
        N = len(data) - (dimension - 1) * tau
        embedding = np.zeros((N, dimension))
        for i in range(dimension):
            embedding[:, i] = data[i * tau:i * tau + N]
        return embedding
    
    phase_space1 = create_embedding(time_series1, embedding_dimension, delay)
    phase_space2 = create_embedding(time_series2, embedding_dimension, delay)
    
    print("\nStep 2: Visualizing embedded phase spaces")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.scatter(phase_space1[:, 0], phase_space1[:, 1], 
               c=np.arange(len(phase_space1)), cmap='viridis', alpha=0.5)
    plt.colorbar(label='Time')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Time Series 1 Embedding')
    
    plt.subplot(122)
    plt.scatter(phase_space2[:, 0], phase_space2[:, 1],
               c=np.arange(len(phase_space2)), cmap='viridis', alpha=0.5)
    plt.colorbar(label='Time')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Time Series 2 Embedding')
    plt.tight_layout()
    plt.show()
    
    # Combine phase spaces
    phase_space = np.hstack((phase_space1, phase_space2))
    
    # 3. Find and visualize example neighbors
    def find_nearest_neighbors(point, points, exclude_radius):
        distances = np.linalg.norm(points - point, axis=1)
        distances[max(0, int(point[0]-exclude_radius)):min(len(distances), 
                 int(point[0]+exclude_radius))] = np.inf
        return np.argmin(distances)
    
    print("\nStep 3: Visualizing example neighboring trajectories")
    example_point = len(phase_space) // 3
    example_neighbor = find_nearest_neighbors(phase_space[example_point], 
                                           phase_space, exclude_radius=10)
    
    plt.figure(figsize=(12, 5))
    num_steps = 10
    
    # Plot trajectories
    reference = phase_space[example_point:example_point+num_steps]
    neighbor = phase_space[example_neighbor:example_neighbor+num_steps]
    
    plt.subplot(121)
    plt.plot(reference[:, 0], reference[:, 1], 'b-', label='Reference')
    plt.plot(neighbor[:, 0], neighbor[:, 1], 'r-', label='Neighbor')
    plt.scatter(reference[0, 0], reference[0, 1], c='b', marker='o', s=100)
    plt.scatter(neighbor[0, 0], neighbor[0, 1], c='r', marker='o', s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Example Trajectories')
    plt.legend()
    
    # Plot separation
    plt.subplot(122)
    separations = np.linalg.norm(reference - neighbor, axis=1)
    plt.plot(range(num_steps), separations, 'g-')
    plt.xlabel('Time Steps')
    plt.ylabel('Separation Distance')
    plt.title('Trajectory Separation')
    plt.show()
    
    # 4. Track divergence
    evolution_time = 20
    n_points = len(phase_space)
    divergences = []
    times = np.arange(evolution_time)
    
    print("\nStep 4: Calculating divergences...")
    for i in range(n_points - evolution_time):
        if i % 100 == 0:
            print(f"Processing point {i}/{n_points - evolution_time}")
            
        j = find_nearest_neighbors(phase_space[i], phase_space, exclude_radius=10)
        
        d0 = np.linalg.norm(phase_space[i] - phase_space[j])
        if d0 == 0:
            continue
            
        separations = []
        for t in range(evolution_time):
            dt = np.linalg.norm(phase_space[i + t] - phase_space[j + t])
            if dt == 0:
                break
            separations.append(np.log(dt/d0))
            
        if len(separations) == evolution_time:
            divergences.append(separations)
    
    if not divergences:
        return None
    
    # 5. Plot divergence curves
    print("\nStep 5: Visualizing divergence curves")
    plt.figure(figsize=(12, 6))
    divergences_array = np.array(divergences)
    
    # Individual curves
    for div in divergences_array:
        plt.plot(times, div, 'b-', alpha=0.1)
    
    # Mean divergence
    mean_divergence = np.mean(divergences_array, axis=0)
    plt.plot(times, mean_divergence, 'r-', linewidth=2, label='Mean divergence')
    plt.xlabel('Time steps')
    plt.ylabel('ln(dt/d0)')
    plt.title('All Divergence Curves with Mean')
    plt.legend()
    plt.show()
    
    # 6. Final calculation
    def linear_fit(t, lambda_exp, c):
        return lambda_exp * t + c
    
    popt, _ = curve_fit(linear_fit, times, mean_divergence)
    lambda_exp = popt[0]
    
    print("\nStep 6: Final Lyapunov exponent calculation")
    plt.figure(figsize=(10, 6))
    plt.scatter(times, mean_divergence, label='Average divergence')
    plt.plot(times, linear_fit(times, *popt), 'r-', 
             label=f'Fit: λ = {lambda_exp:.4f}')
    plt.xlabel('Time steps')
    plt.ylabel('ln(dt/d0)')
    plt.title('Lyapunov Exponent Calculation')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return lambda_exp

# Example usage
if __name__ == "__main__":
    # Generate example data
    t = np.linspace(0, 100, 1000)
    
    # Option 1: Use simple oscillatory data
    x = np.sin(t) + 0.1 * np.random.randn(len(t))
    y = np.cos(t) + 0.1 * np.random.randn(len(t))
    
    # Option 2: Use Lorenz data
    # lorenz_traj, _ = generate_lorenz_data()
    # x = lorenz_traj[:, 0]
    # y = lorenz_traj[:, 1]
    
    lyap_exp = calculate_lyapunov_with_viz(x, y)
    
    if lyap_exp is not None:
        print(f"\nFinal Lyapunov Exponent: {lyap_exp:.4f}")
        if lyap_exp > 0.01:
            print("System exhibits CHAOTIC behavior")
            print(f"Divergence time scale: {1/lyap_exp:.2f} time steps")
        elif lyap_exp < -0.01:
            print("System exhibits STABLE behavior")
            print(f"Convergence time scale: {1/abs(lyap_exp):.2f} time steps")
        else:
            print("System is at the EDGE OF CHAOS")