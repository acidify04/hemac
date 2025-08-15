"""Simulation of HeMAC flight dynamics."""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_velocity = 0  # Initial velocity in m/s
time_duration = 25  # Time duration of the simulation in seconds
time_step = 1  # Time step in seconds
top_speed = 16
max_thrust = 5

# Time array
time = np.arange(0, time_duration + time_step, time_step)

# Initialize arrays
velocity = np.zeros(len(time))
position = np.zeros(len(time))
acceleration = np.zeros(len(time))

# Initial conditions
velocity[0] = initial_velocity
position[0] = 0


# Define the command function (e.g., a constant command or a function of time)
def command(t, v, a):
    """Output a command given time, speed and acceleration."""
    if t < 10:
        if v < top_speed:
            u = max_thrust - 0.022 * v**2
        else:
            u = top_speed - v
    elif t < 15:
        u = 0
    else:
        if v > max_thrust:
            u = -max_thrust - 0.022 * v**2
        else:
            u = -v
    print(f"timestep: {t}, velocity: {v}, final accel: {u}")

    return 0.5 * a + 0.5 * u


# Simulation loop
for i in range(1, len(time)):
    acceleration[i] = command(time[i], velocity[i - 1], acceleration[i - 1])
    velocity[i] = velocity[i - 1] + acceleration[i] * time_step
    position[i] = position[i - 1] + velocity[i - 1] * time_step + 0.5 * acceleration[i] * time_step**2

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot velocity
plt.subplot(2, 1, 1)
plt.plot(time, velocity, label="Velocity (m/s)", color="blue")
plt.title("Velocity and Position vs Time")
plt.ylabel("Velocity (m/s)")
plt.grid(True)
plt.legend()

# Plot position
plt.subplot(2, 1, 2)
plt.plot(time, position, label="Position (m)", color="green")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
