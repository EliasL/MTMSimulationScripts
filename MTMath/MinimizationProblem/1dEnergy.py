import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import minimize

# Ensure directories exist
base_output_dir = "MTMath/MinimizationProblem"
os.makedirs(base_output_dir, exist_ok=True)

# Define domain
x = np.linspace(0, 10, 1000)


# Define given functions
def func(x, t):
    # Function that will be plotted and minimized at time t
    return -x / 7 + 0.5 * np.sin(2 * x) + np.sin(4 * x + t) * (1 + np.sin(t + 2)) / 2


def simple(x, t):
    return -x / 7 + np.sin(x + t)


def make_animation(
    f,
    t_start,
    t_end,
    output_name,
    frames=400,
    initial_guess=6.0,
    label=r"$f(x,t)$",
):
    """
    Create an animation for the given function f(x,t), from t_start to t_end.
    Save as mp4 in the given output_name.
    frames: number of frames in the animation.
    initial_guess: initial guess for the minimizer.
    """
    # Create figure and axis for this animation
    fig, ax = plt.subplots()
    (line,) = ax.plot(x, f(x, t_start), label=label)
    (point,) = ax.plot([], [], "ro", label="")
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 1.7)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x,t)$")
    ax.legend()

    # Local variable to store the previous minimum x
    previous_min_x = initial_guess

    # Define the update function for FuncAnimation
    def update(t):
        nonlocal previous_min_x
        y = f(x, t)
        line.set_ydata(y)
        # Minimize to find local min
        res = minimize(lambda x0: f(x0, t), x0=previous_min_x, bounds=[(0, 10)])
        min_x = res.x
        min_y = f(min_x, t)
        point.set_data(min_x, min_y)
        previous_min_x = min_x
        # Avoid boundary locking
        if min_x < 1:
            previous_min_x = 9.0
        return line, point

    # Create animation
    t_values = np.linspace(t_start, t_end, frames)
    ani = animation.FuncAnimation(fig, update, frames=t_values, interval=30, blit=True)

    # Save as MP4 using FFMpegWriter
    Writer = animation.FFMpegWriter
    writer = Writer(fps=30, codec="h264", bitrate=-1)
    output_path = os.path.join(base_output_dir, output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer=writer, dpi=150)
    plt.close(fig)  # Close figure after saving


# ---------------------------
# Create the three videos
# ---------------------------

# 1. Forward time using func
make_animation(
    f=func,
    t_start=0,
    t_end=6 * np.pi,
    output_name="animated_wave_with_minimum_forward.mp4",
    frames=400,
    initial_guess=6.0,
    label=r"$f(x,t) = -x/7+\sin(2x)/2+\sin(4x+t)(1+\sin(t+2))/2$",
)

# 2. Backward time using func
make_animation(
    f=func,
    t_start=6 * np.pi,
    t_end=0,
    output_name="animated_wave_with_minimum_backward.mp4",
    frames=400,
    initial_guess=6.0,
    label=r"$f(x,t) = -x/7+\sin(2x)/2+\sin(4x+t)(1+\sin(t+2))/2$",
)

# 3. Forward time using simple
make_animation(
    f=simple,
    t_start=0,
    t_end=6 * np.pi,
    output_name="animated_wave_with_minimum_simple.mp4",
    frames=400,
    initial_guess=6.0,
    label=r"$g(x,t) = -x/7+sin(x+t)$",
)
