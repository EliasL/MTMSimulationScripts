#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def equilateral_triangle(center=(0.0, 0.0), radius=1.0, rotation_deg=90.0):
    """Return 2x3 array of vertices (columns) for a centroid-centered equilateral triangle on a circle of given radius."""
    angles = np.deg2rad(rotation_deg + np.array([0.0, 120.0, 240.0]))
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    V = np.vstack([x, y])
    # Recenter to exact centroid and then shift to desired center
    centroid = V.mean(axis=1, keepdims=True)
    V = V - centroid + np.array(center, dtype=float).reshape(2, 1)
    return V


def shear_matrix(shx=0, shy=0):
    """Simple shear: x' = x + shx*y, y' = shy*x + y."""
    return np.array([[1.0, shx], [shy, 1.0]], dtype=float)


def apply_affine(M, V):
    """Apply 2x2 matrix M to vertices V (2xN)."""
    return M @ V


def polygon_bounds(V):
    """Axis-aligned bounds of 2xN vertices."""
    xmin, ymin = V.min(axis=1)
    xmax, ymax = V.max(axis=1)
    return xmin, xmax, ymin, ymax


def translate(V, dx=0.0, dy=0.0):
    return V + np.array([[dx], [dy]])


def plot_triangle(ax, V, **kwargs):
    """Plot closed polygon given 2xN vertices."""
    P = np.hstack([V, V[:, :1]])  # 2x(N+1), closes the loop
    ax.plot(P[0], P[1], **kwargs)


if __name__ == "__main__":
    # Parameters (feel free to adjust)
    kx = 1  # integer horizontal shear magnitude
    ky = 1  # integer vertical shear magnitude
    base_radius = 1.0
    line_kw = dict(linewidth=2)

    # Base (center) triangle
    V0 = equilateral_triangle(radius=base_radius, rotation_deg=90)

    # Sheared variants (still centered at origin initially)
    V_left = apply_affine(shear_matrix(shx=-kx, shy=0), V0)  # negative horizontal shear
    V_right = apply_affine(
        shear_matrix(shx=+kx, shy=0), V0
    )  # positive horizontal shear
    V_up = apply_affine(shear_matrix(shx=0, shy=+ky), V0)  # positive vertical shear
    V_down = apply_affine(shear_matrix(shx=0, shy=-ky), V0)  # negative vertical shear

    # Compute a spacing so triangles don’t overlap: use width/height of the largest variant
    triangles = [V0, V_left, V_right, V_up, V_down]
    widths = []
    heights = []
    for V in triangles:
        xmin, xmax, ymin, ymax = polygon_bounds(V)
        widths.append(xmax - xmin)
        heights.append(ymax - ymin)
    W = max(widths)
    H = max(heights)
    gap = 0.2  # extra padding factor

    # Place around the center: left/right along x, up/down along y
    V_left = translate(V_left, dx=-(W * (1 + gap)), dy=0)
    V_right = translate(V_right, dx=+(W * (1 + gap)), dy=0)
    V_up = translate(V_up, dx=0, dy=+(H * (1 + gap)))
    V_down = translate(V_down, dx=0, dy=-(H * (1 + gap)))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_triangle(ax, V0, color="black", **line_kw)
    # Plot V_left and its undeformed outline
    plot_triangle(ax, V_left, color="tab:blue", **line_kw)
    plot_triangle(
        ax,
        translate(V0, dx=-(W * (1 + gap)), dy=0),
        color="gray",
        alpha=0.3,
        linestyle="--",
    )
    # Plot V_right and its undeformed outline
    plot_triangle(ax, V_right, color="tab:orange", **line_kw)
    plot_triangle(
        ax,
        translate(V0, dx=+(W * (1 + gap)), dy=0),
        color="gray",
        alpha=0.3,
        linestyle="--",
    )
    # Plot V_up and its undeformed outline
    plot_triangle(ax, V_up, color="tab:green", **line_kw)
    plot_triangle(
        ax,
        translate(V0, dx=0, dy=+(H * (1 + gap))),
        color="gray",
        alpha=0.3,
        linestyle="--",
    )
    # Plot V_down and its undeformed outline
    plot_triangle(ax, V_down, color="tab:red", **line_kw)
    plot_triangle(
        ax,
        translate(V0, dx=0, dy=-(H * (1 + gap))),
        color="gray",
        alpha=0.3,
        linestyle="--",
    )

    # --- Arrows + matrix labels ---
    def _centroid(V):
        return V.mean(axis=1)

    def _matrix_label(shx, shy):
        return None
        # return rf"$m=\\begin{pmatrix}1 & {shx}\\\\ {shy} & 1\\end{pmatrix}$"

    def _arrow_with_label(ax, p0, p1, color, label):
        ax.annotate(
            "",
            xy=(p1[0], p1[1]),
            xytext=(p0[0], p0[1]),
            arrowprops=dict(arrowstyle="->", lw=1.5, color=color),
        )
        mid = 0.55 * p1 + 0.45 * p0
        ax.text(
            mid[0],
            mid[1],
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )

    c0 = _centroid(V0)
    cL = _centroid(V_left)
    cR = _centroid(V_right)
    cU = _centroid(V_up)
    cD = _centroid(V_down)

    _arrow_with_label(ax, c0, cL, "tab:blue", _matrix_label(shx=-kx, shy=0))
    _arrow_with_label(ax, c0, cR, "tab:orange", _matrix_label(shx=+kx, shy=0))
    _arrow_with_label(ax, c0, cU, "tab:green", _matrix_label(shx=0, shy=+ky))
    _arrow_with_label(ax, c0, cD, "tab:red", _matrix_label(shx=0, shy=-ky))

    # Optional: helpful labels
    def label_center(ax, V, text):
        c = V.mean(axis=1)
        ax.text(c[0], c[1], text, ha="center", va="center")

    # label_center(ax, V0, "center")
    # label_center(ax, V_left, f"shx = {-kx}")
    # label_center(ax, V_right, f"shx = {+kx}")
    # label_center(ax, V_up, f"shy = {+ky}")
    # label_center(ax, V_down, f"shy = {-ky}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Equilateral triangle with integer simple shears")

    # --- Remove ticks and tick labels, hide axes frame ---
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Nice bounds
    all_pts = np.hstack([V0, V_left, V_right, V_up, V_down])
    pad = 0.1 * max(W, H)
    ax.set_xlim(all_pts[0].min() - pad, all_pts[0].max() + pad)
    ax.set_ylim(all_pts[1].min() - pad, all_pts[1].max() + pad)

    plt.show()
