import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Coulomb's constant
k = 8.99e9

def calculate_electric_field(charges, grid_size, grid_resolution):
    x = np.linspace(-grid_size, grid_size, grid_resolution)
    y = np.linspace(-grid_size, grid_size, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    Ex, Ey = np.zeros(X.shape), np.zeros(Y.shape)
    
    for qx, qy, q in charges:
        dx, dy = X - qx, Y - qy
        r_squared = dx**2 + dy**2
        r_squared[r_squared == 0] = np.inf
        r = np.sqrt(r_squared)
        E = k * q / r_squared
        Ex += E * dx / r
        Ey += E * dy / r
    
    return x, y, Ex, Ey

def apply_log_scaling(Ex, Ey):
    magnitude = np.sqrt(Ex**2 + Ey**2)
    log_magnitude = np.log1p(magnitude)  # Use log1p for better numerical stability
    Ex_log = Ex / magnitude * log_magnitude
    Ey_log = Ey / magnitude * log_magnitude
    return Ex_log, Ey_log


def electric_field(charge, point):
    x_charge, y_charge, q = charge
    px, py = point
    # Calculate the distance components
    dx = px - x_charge
    dy = py - y_charge
    # Calculate the distance r
    r = np.sqrt(dx**2 + dy**2)
    # Calculate the unit vector components
    if r != 0:
        ux, uy = dx / r, dy / r
    else:
        ux, uy = 0, 0
    # Calculate the electric field magnitude
    E_magnitude = k * q / r**2
    # Electric field components
    Ex = E_magnitude * ux
    Ey = E_magnitude * uy
    return Ex, Ey

def total_electric_field_at_point(charges, point):
    E_total_x, E_total_y = 0, 0
    for charge in charges:
        Ex, Ey = electric_field(charge, point)
        E_total_x += Ex
        E_total_y += Ey
    return E_total_x, E_total_y

def force_on_particle(charge, E_field_at_point):
    q = charge
    E_x, E_y = E_field_at_point
    F_x = q * E_x
    F_y = q * E_y
    return F_x, F_y

def simulate_trajectory(charges, particle, grid_size, grid_resolution, time_step, num_steps):
    x_grid, y_grid, Ex, Ey = calculate_electric_field(charges, grid_size, grid_resolution)
    
    x, y, q_particle, m_particle = particle
    vx, vy = 0, 0
    trajectory = [(x, y)]
    
    for step in range(num_steps):
        # Check if the particle is still within the grid
        if x < x_grid[0] or x > x_grid[-1] or y < y_grid[0] or y > y_grid[-1]:
            print(f"Step {step}: Particle out of bounds at position ({x}, {y}).")
            break

        Ex_at_point, Ey_at_point = total_electric_field_at_point(charges, (x, y))
        Fx, Fy = force_on_particle(q_particle, (Ex_at_point, Ey_at_point))
        ax = Fx / m_particle
        ay = Fy / m_particle

        vx += ax * time_step
        vy += ay * time_step
        x += vx * time_step
        y += vy * time_step
        
        trajectory.append((x, y))
        if step % 100 == 0:  # Print every 100 steps for brevity
            print(f"Step {step}: x={x}, y={y}, vx={vx}, vy={vy}, ax={ax}, ay={ay}, Fx={Fx}, Fy={Fy}, Ex={Ex_at_point}, Ey={Ey_at_point}")
    
    return x_grid, y_grid, Ex, Ey, trajectory

def plot_electric_field(X, Y, Ex, Ey):
    plt.figure(figsize=(10, 10))
    plt.quiver(X, Y, Ex, Ey, color='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Electric Field Vectors')
    plt.grid(True)
    plt.show()

def plot_trajectory(X, Y, Ex, Ey, trajectory, max_magnitude, scale=50):
    Ex_log_clipped, Ey_log_clipped = apply_log_scaling(Ex, Ey)
    plt.figure(figsize=(10, 10))
    plt.quiver(X, Y, Ex_log_clipped, Ey_log_clipped, color='r', scale=scale)
    
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Electric Field and Particle Trajectory')
    plt.grid(True)
    plt.show()

# Example usage
charges = [(0, 0, 1e-9), (1, 1, -1e-9), (-1, -1, 1e-9)]
particle = (1.5, 0.0, 1e-9, 1e-5)
grid_size = 2
grid_resolution = 50
time_step = 1e-1
num_steps = 5001
max_magnitude = 1.0

X, Y, Ex, Ey, trajectory = simulate_trajectory(charges, particle, grid_size, grid_resolution, time_step, num_steps)
plot_trajectory(X, Y, Ex, Ey, trajectory, max_magnitude, scale=50)
