import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==== Mathematical model and helpers ====

# --- Define deformation gradient F (you can change this) ---
F = np.array([[1, 0.5], [0, 1]])

# --- Unit circle points in reference configuration ---
t = np.linspace(0, 2 * np.pi, 400)
base_circle = np.vstack((np.cos(t), np.sin(t)))  # shape (2, N)


def rotation(theta):
    """Return 2D rotation matrix for angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def simple_shear(gamma):
    """Return 2D simple shear matrix S with shear gamma in x-direction."""
    return np.array([[1.0, gamma], [0.0, 1.0]])


def predeformed_circle(gamma):
    """Return pre-deformed circle given shear gamma."""
    S = simple_shear(gamma)
    return S @ base_circle


def translate_circle(circle, shift):
    """Translate circle by shift in x-direction."""
    return circle + np.array([[shift], [0.0]])


def compute_RF(theta):
    """Return RF = R @ F for a given rotation angle theta (radians)."""
    R = rotation(theta)
    return R @ F


def compute_RFRT(theta):
    """Return RFR^T = R @ F @ R^T for a given rotation angle theta (radians)."""
    R = rotation(theta)
    return R @ F @ R.T


def apply_transform(transform_matrix, circle):
    """Apply transformation matrix to circle."""
    return transform_matrix @ circle


def get_columns(matrix):
    """Return columns of matrix as tuple (col1, col2)."""
    return matrix[:, 0], matrix[:, 1]


def compute_visualization_data(theta, gamma, shift):
    """
    Compute all data needed for visualization.

    Returns:
        dict with keys:
            - 'circle_ref': reference circle coordinates
            - 'circle_RF': RF transformed circle
            - 'circle_RFRT': RFRT transformed circle
            - 'RF_col1', 'RF_col2': columns of RF
            - 'RFRT_col1', 'RFRT_col2': columns of RFRT
    """
    # Compute pre-deformed and reference circle
    predef = predeformed_circle(gamma)
    circle_ref = translate_circle(predef, shift)

    # Compute transformation matrices
    RF = compute_RF(theta)
    RFRT = compute_RFRT(theta)

    # Apply transformations to circles
    circle_RF = translate_circle(apply_transform(RF, predef), shift)
    circle_RFRT = translate_circle(apply_transform(RFRT, predef), shift)

    # Get matrix columns
    RF_col1, RF_col2 = get_columns(RF)
    RFRT_col1, RFRT_col2 = get_columns(RFRT)

    return {
        "circle_ref": circle_ref,
        "circle_RF": circle_RF,
        "circle_RFRT": circle_RFRT,
        "RF_col1": RF_col1,
        "RF_col2": RF_col2,
        "RFRT_col1": RFRT_col1,
        "RFRT_col2": RFRT_col2,
    }


# ==== Plotting and interactive UI ====

# --- Initial parameters ---
theta0_deg = 0.0
theta0_rad = np.deg2rad(theta0_deg)
shift0 = 0.0
gamma0 = 0.0

# Get initial visualization data
viz_data = compute_visualization_data(theta0_rad, gamma0, shift0)

# --- Set up figure and axes ---
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
plt.subplots_adjust(
    left=0.06, right=0.97, bottom=0.15, top=0.95, wspace=0.1, hspace=0.1
)

ax11 = axes[0, 0]  # top-left: reference circle + RF(circle)
ax12 = axes[0, 1]  # top-right: columns of RF
ax21 = axes[1, 0]  # bottom-left: reference circle + RFR^T(circle)
ax22 = axes[1, 1]  # bottom-right: columns of RFR^T

for ax in [ax11, ax12, ax21, ax22]:
    ax.set_aspect("equal", "box")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True)

# --- Top-left: reference circle and RF acting on it ---
[line_unit_RF] = ax11.plot(
    viz_data["circle_ref"][0], viz_data["circle_ref"][1], "--", label="reference circle"
)
[line_RF_circle] = ax11.plot(
    viz_data["circle_RF"][0],
    viz_data["circle_RF"][1],
    "-",
    label=r"$\mathbf{R}\mathbf{F}$ (circle)",
)
ax11.set_title(r"$\mathbf{R}\mathbf{F}$ acting on circle")
ax11.legend(loc="upper right")

# --- Top-right: columns of RF as vectors ---
[line_RF_e1] = ax12.plot(
    [0, viz_data["RF_col1"][0]],
    [0, viz_data["RF_col1"][1]],
    "-",
    label=r"col 1 of $\mathbf{R}\mathbf{F}$",
)
[line_RF_e2] = ax12.plot(
    [0, viz_data["RF_col2"][0]],
    [0, viz_data["RF_col2"][1]],
    "-",
    label=r"col 2 of $\mathbf{R}\mathbf{F}$",
)
ax12.set_title(r"Columns of $\mathbf{R}\mathbf{F}$")
ax12.legend(loc="upper right")

# --- Bottom-left: reference circle and RFR^T acting on it ---
[line_unit_RFRT] = ax21.plot(
    viz_data["circle_ref"][0], viz_data["circle_ref"][1], "--", label="reference circle"
)
[line_RFRT_circle] = ax21.plot(
    viz_data["circle_RFRT"][0],
    viz_data["circle_RFRT"][1],
    "-",
    label=r"$\mathbf{R}\mathbf{F}\mathbf{R}^\mathsf{T}$ (circle)",
)
ax21.set_title(r"$\mathbf{R}\mathbf{F}\mathbf{R}^\mathsf{T}$ acting on circle")
ax21.legend(loc="upper right")

# --- Bottom-right: columns of RFR^T as vectors ---
[line_RFRT_e1] = ax22.plot(
    [0, viz_data["RFRT_col1"][0]],
    [0, viz_data["RFRT_col1"][1]],
    "-",
    label=r"col 1 of $\mathbf{R}\mathbf{F}\mathbf{R}^\mathsf{T}$",
)
[line_RFRT_e2] = ax22.plot(
    [0, viz_data["RFRT_col2"][0]],
    [0, viz_data["RFRT_col2"][1]],
    "-",
    label=r"col 2 of $\mathbf{R}\mathbf{F}\mathbf{R}^\mathsf{T}$",
)
ax22.set_title(r"Columns of $\mathbf{R}\mathbf{F}\mathbf{R}^\mathsf{T}$")
ax22.legend(loc="upper right")

# --- Sliders ---
slider_shift = 0.0
ax_theta = plt.axes([0.15, 0.09 + slider_shift, 0.7, 0.03])
theta_slider = Slider(
    ax=ax_theta,
    label=r"Rotation angle $\theta$ (deg)",
    valmin=-180.0,
    valmax=180.0,
    valinit=theta0_deg,
)
ax_shear = plt.axes([0.15, 0.03 + slider_shift, 0.7, 0.03])
shear_slider = Slider(
    ax=ax_shear,
    label=r"Shear $\gamma$ in $\mathbf{S}$",
    valmin=-1.5,
    valmax=1.5,
    valinit=gamma0,
)

# ax_shift = plt.axes([0.15, 0.03+slider_shift, 0.7, 0.03])
# shift_slider = Slider(
#     ax=ax_shift,
#     label=r"Shift of reference circle in $x$",
#     valmin=-1.5,
#     valmax=1.5,
#     valinit=shift0,
# )


# --- Update function ---
def update(val):
    """Update all plots based on current slider values."""
    theta_deg = theta_slider.val
    theta = np.deg2rad(theta_deg)
    shift = 0
    gamma = shear_slider.val

    # Compute all visualization data using math functions
    viz_data = compute_visualization_data(theta, gamma, shift)

    # Update reference circles
    line_unit_RF.set_data(viz_data["circle_ref"][0], viz_data["circle_ref"][1])
    line_unit_RFRT.set_data(viz_data["circle_ref"][0], viz_data["circle_ref"][1])

    # Update RF circle (top-left)
    line_RF_circle.set_data(viz_data["circle_RF"][0], viz_data["circle_RF"][1])

    # Update columns of RF (top-right)
    line_RF_e1.set_data([0, viz_data["RF_col1"][0]], [0, viz_data["RF_col1"][1]])
    line_RF_e2.set_data([0, viz_data["RF_col2"][0]], [0, viz_data["RF_col2"][1]])

    # Update RFR^T circle (bottom-left)
    line_RFRT_circle.set_data(viz_data["circle_RFRT"][0], viz_data["circle_RFRT"][1])

    # Update columns of RFR^T (bottom-right)
    line_RFRT_e1.set_data([0, viz_data["RFRT_col1"][0]], [0, viz_data["RFRT_col1"][1]])
    line_RFRT_e2.set_data([0, viz_data["RFRT_col2"][0]], [0, viz_data["RFRT_col2"][1]])

    fig.canvas.draw_idle()


theta_slider.on_changed(update)
# shift_slider.on_changed(update)
shear_slider.on_changed(update)

plt.show()
