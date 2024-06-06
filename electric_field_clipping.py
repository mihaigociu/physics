import numpy as np
import matplotlib.pyplot as plt

# Coulomb's constant
k = 8.99e9

# Define a function to calculate the electric field
def calculate_electric_field(charges, grid_size, grid_resolution):
    # Create a grid of points
    x = np.linspace(-grid_size, grid_size, grid_resolution)
    y = np.linspace(-grid_size, grid_size, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # Initialize the electric field components
    Ex = np.zeros(X.shape)
    Ey = np.zeros(Y.shape)
    
    # Calculate the electric field contribution from each charge
    for charge in charges:
        qx, qy, q = charge
        # Calculate the distance components
        dx = X - qx
        dy = Y - qy
        # Calculate the distance squared
        r_squared = dx**2 + dy**2
        # Avoid division by zero
        r_squared[r_squared == 0] = np.inf
        # Calculate the electric field components
        E = k * q / r_squared
        Ex += E * dx / np.sqrt(r_squared)
        Ey += E * dy / np.sqrt(r_squared)
    
    return X, Y, Ex, Ey

# Define a function to apply logarithmic scaling and clip the magnitudes
def apply_log_scaling_and_clipping(Ex, Ey, max_magnitude):
    magnitude = np.sqrt(Ex**2 + Ey**2)
    log_magnitude = np.log1p(magnitude)  # Use log1p for better numerical stability
    clipped_magnitude = np.minimum(log_magnitude, max_magnitude)
    Ex_log_clipped = Ex / magnitude * clipped_magnitude
    Ey_log_clipped = Ey / magnitude * clipped_magnitude
    return Ex_log_clipped, Ey_log_clipped

# Define a function to visualize the electric field
def plot_electric_field(X, Y, Ex, Ey, max_magnitude, scale=50):
    Ex_log_clipped, Ey_log_clipped = apply_log_scaling_and_clipping(Ex, Ey, max_magnitude)
    plt.figure(figsize=(10, 10))
    plt.quiver(X, Y, Ex_log_clipped, Ey_log_clipped, color='r', scale=scale)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Electric Field Vector Field with Logarithmic Scaling and Clipping')
    plt.grid(True)
    plt.show()

# Example usage
charges = [(0, 0, 1e-9), (1, 1, -1e-9), (-1, -1, 1e-9)]
grid_size = 2
grid_resolution = 50
max_magnitude = 1.0

X, Y, Ex, Ey = calculate_electric_field(charges, grid_size, grid_resolution)
plot_electric_field(X, Y, Ex, Ey, max_magnitude, scale=50)
