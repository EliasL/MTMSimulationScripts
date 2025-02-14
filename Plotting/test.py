import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define a 4x4 matrix representing discrete parameter steps
epsR_values = np.logspace(-3, 0, 4)  # 4 values from 0.001 to 1 (log scale)
epsE_values = np.logspace(-3, 0, 4)  # Same for epsE

# Create a 4x4 matrix where each element corresponds to a (epsR, epsE) pair
color_matrix = np.zeros((4, 4, 4))  # RGBA matrix

# Normalize values between 0 and 1 for mapping
epsR_norm = (np.log10(epsR_values) - np.log10(epsR_values.min())) / (
    np.log10(epsR_values.max()) - np.log10(epsR_values.min())
)
epsE_norm = (np.log10(epsE_values) - np.log10(epsE_values.min())) / (
    np.log10(epsE_values.max()) - np.log10(epsE_values.min())
)

# Assign colors: Red for epsR, Blue for epsE, Black for min values
for i in range(4):
    for j in range(4):
        red = epsR_norm[j]  # Horizontal axis controls red intensity
        blue = epsE_norm[i]  # Vertical axis controls blue intensity
        color_matrix[i, j] = [
            red,
            abs(red - blue),
            blue,
            1 - max(red, blue) / 2,
        ]  # RGB composition

# Create figure with two subplots: one for the color matrix, one for the curves
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the color matrix on the left
cax = axs[0].matshow(color_matrix.transpose((0, 1, 2)))

# Set ticks and labels
axs[0].set_xticks(range(4))
axs[0].set_yticks(range(4))
axs[0].set_xticklabels([f"{val:.3f}" for val in epsR_values])
axs[0].set_yticklabels([f"{val:.3f}" for val in epsE_values])

# Flip y-axis to ensure smallest values are at the **bottom-left**
axs[0].invert_yaxis()
axs[0].xaxis.set_ticks_position("bottom")  # Move x-axis ticks to the bottom
axs[0].xaxis.set_label_position("bottom")  # Move x-axis label to the bottom

axs[0].set_xlabel("epsR (Increasing → Yello)")
axs[0].set_ylabel("epsE (Increasing → Cyan)")
axs[0].set_title("2D Parameter Color Map")

# Generate x values for curves
x = np.linspace(0, 10, 100)

# Plot the curves on the right
for i in range(4):
    for j in range(4):
        epsR = epsR_values[j]
        epsE = epsE_values[i]
        y = np.sin(x) * (epsR + 3 * epsE)  # Example function
        color = color_matrix[i, j]  # Get corresponding color
        axs[1].plot(x, y, color=color, label=f"epsR={epsR:.3f}, epsE={epsE:.3f}")

axs[1].set_xlabel("X-axis")
axs[1].set_ylabel("Y-axis")
axs[1].set_yscale("log")
axs[1].set_title("Curves with Corresponding Colors")
axs[1].legend(fontsize=8, loc="upper right", ncol=2)

plt.tight_layout()
plt.show()
