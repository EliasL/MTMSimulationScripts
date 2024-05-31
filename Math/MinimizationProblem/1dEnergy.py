import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import minimize

# Define the function to be animated
def func(x, t):
    return 1-x/7 + 0.5 * np.sin(2 * x) + np.sin(4 * x + t) * (1 + np.sin(t + 2)) / 2

# Set up the figure and axis
fig, ax = plt.subplots()
x = np.linspace(0, 10, 1000)
line, = ax.plot(x, func(x, 0))
point, = ax.plot([], [], 'ro')  # point to mark the minimum

# Set the axis limits
ax.set_xlim(0, 10)
ax.set_ylim(-3, 3)

# Initialize the starting point for the minimizer
previous_min_x = 9

# Function to capture specific frames
def save_frames():
    # List of specific t values for the frames
    t_values = [0,1,1]  # Adjust these as needed to capture the desired moments
    previous_min_x=9
    for i, t in enumerate(t_values):
        y = func(x, t)
        ax.clear()  # Clear previous plot
        ax.plot(x, y, label=f'Time = {t:.2f}')  # Plot curve for current t
        
        # Find and plot the minimum
        if i == 1:
            # Keep the same x with updated y-value
            min_x = previous_min_x
        else:
            result = minimize(lambda x0: func(x0, t), x0=previous_min_x, bounds=[(0, 10)])
            min_x = result.x
        
        min_y = func(min_x, t)
        ax.plot(min_x, min_y, 'ro')  # Plot the minimum point
        
        # Set the axis limits
        ax.set_xlim(0, 10)
        ax.set_ylim(-3, 3)
        
        # Save each frame as an image
        plt.savefig(f'Math/MinimizationProblem/frame_{i}.png', dpi=150)
        
        # Update previous minimum
        previous_min_x = min_x

# Call the function to save specific frames
save_frames()

# Define the update function for the animation
def update(t):
    global previous_min_x
    
    y = func(x, t)
    line.set_ydata(y)
    
    # Find the local minimum starting from the previous minimum
    result = minimize(lambda x0: func(x0, t), x0=previous_min_x, bounds=[(0, 10)])
    min_x = result.x
    min_y = func(min_x, t)
    
    # Update the point
    point.set_data(min_x, min_y)
    
    # Update the previous minimum
    previous_min_x = min_x
    
    return line, point

# Increase the frames and reduce the interval for smoother and faster animation
ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 6 * np.pi, 40), interval=50, blit=True)

# Save the animation as a high-resolution GIF with a high frame rate and transparency
fName='Math/MinimizationProblem/animated_wave_with_minimum_following.gif'
#ani.save(fName, writer='pillow', fps=60, dpi=150, savefig_kwargs={'transparent': True})

# Display the animation
#plt.show()