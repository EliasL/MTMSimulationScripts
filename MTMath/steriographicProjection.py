import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Generate hemisphere coordinates
def generate_hemisphere(n=100):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, 0.5 * np.pi, n)  # Only upper hemisphere
    u, v = np.meshgrid(u, v)

    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    return x, y, z


# Stereographic projection from north pole (0, 0, 1)
def stereographic_proj(x, y, z):
    denom = 1 - z
    denom[denom == 0] = np.nan  # Avoid division by zero
    x_proj = x / denom
    y_proj = y / denom
    return x_proj, y_proj


# Normalize to unit circle (hyperbolic disk)
def normalize_to_disk(x, y):
    r = np.sqrt(x**2 + y**2)
    mask = r > 1
    x[mask] /= r[mask]
    y[mask] /= r[mask]
    return x, y


# Setup plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([0, 1])
ax.set_box_aspect([1, 1, 1])
ax.axis("off")

# Hemisphere data
x, y, z = generate_hemisphere()
surf = ax.plot_surface(x, y, z, color="skyblue", alpha=0.6, linewidth=0)

# Projected data (initially same as original)
x_proj, y_proj = stereographic_proj(x, y, z)
x_disk, y_disk = normalize_to_disk(x_proj.copy(), y_proj.copy())
z_flat = np.zeros_like(z)
scatter = ax.plot_surface(x_disk, y_disk, z_flat, color="black", alpha=0.5)


# Animation
def update(frame):
    t = frame / 100.0  # Interpolation factor from 0 to 1
    x_interp = (1 - t) * x + t * x_disk
    y_interp = (1 - t) * y + t * y_disk
    z_interp = (1 - t) * z + t * z_flat

    ax.clear()  # Remove old plots
    ax.plot_surface(x, y, z, color="skyblue", alpha=0.3, linewidth=0)
    ax.plot_surface(
        x_interp, y_interp, z_interp, color="darkred", alpha=0.8, linewidth=0
    )

    return []


ani = FuncAnimation(fig, update, frames=101, interval=50, blit=False)
plt.show()
