import numpy as np
import matplotlib.pyplot as plt

# Parameters
k = 10  # Spring constant (N/m)
m = 2    # Mass (kg)
omega = np.sqrt(k / m)  # Angular frequency
initial_position = 1.0  # Initial displacement (m)
initial_velocity = 50.0  # Initial velocity (m/s)
time_step = 0.02        # Time step (s)
total_time = 100         # Total simulation time (s)

mu = 2

# Time array
num_steps = int(total_time / time_step)
times = np.linspace(0, total_time, num_steps)

# Arrays to store results
positions = np.zeros(num_steps)
velocities = np.zeros(num_steps)

# Initial conditions
positions[0] = initial_position
velocities[0] = initial_velocity

# Runge-Kutta Method (4th Order)
for i in range(1, num_steps):
    # Current state
    x = positions[i-1]
    v = velocities[i-1]
    
    # RK4 Steps
    k1x = time_step * v
    k1v = time_step * (-omega**2 * x - mu * v)
    
    k2x = time_step * (v + 0.5 * k1v)
    k2v = time_step * (-omega**2 * (x + 0.5 * k1x) - mu * v * (x + 0.5 * k1x))
    
    k3x = time_step * (v + 0.5 * k2v)
    k3v = time_step * (-omega**2 * (x + 0.5 * k2x) - mu *(x + 0.5 * k2x))
    
    k4x = time_step * (v + k3v)
    k4v = time_step * (-omega**2 * (x + k3x) - mu *  (x + k3x) )
    
    # Update position and velocity
    positions[i] = x + (k1x + 2*k2x + 2*k3x + k4x) / 6 
    velocities[i] = v + (k1v + 2*k2v + 2*k3v + k4v) / 6 

# Plotting phase space
plt.figure(figsize=(8, 8))
plt.plot(positions, velocities, label="Phase Space Trajectory")
plt.title("Phase Space Plot (Position vs. Velocity)")
plt.xlabel("Position (x)")
plt.ylabel("Velocity (v)")
plt.grid()
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.legend()
plt.show()
