from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def shear(points: np.ndarray, gamma: float) -> np.ndarray:
    transform = np.array([[1.0, gamma], [0.0, 1.0]])
    return points @ transform.T


def closed(points: np.ndarray) -> np.ndarray:
    return np.vstack([points, points[0]])


def center(points: np.ndarray) -> np.ndarray:
    return points - points.mean(axis=0)


def draw_triangle(ax, points, *, color, label, linestyle="-", linewidth=2.2, alpha=1.0):
    pts = closed(points)
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )


def draw_patch(ax, points, diagonal, *, color, label, linestyle="-", alpha=1.0):
    square = np.array([points[0], points[1], points[3], points[2], points[0]])
    ax.plot(square[:, 0], square[:, 1], color=color, linewidth=2.0, alpha=alpha)
    a, b = diagonal
    ax.plot(
        [points[a, 0], points[b, 0]],
        [points[a, 1], points[b, 1]],
        color=color,
        linestyle=linestyle,
        linewidth=2.3,
        alpha=alpha,
        label=label,
    )


def set_equal_panel(ax, title, *, xlim=(-0.85, 1.05), ylim=(-0.82, 0.82)):
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    out_path = ROOT / "Plots" / "stiffness_reference_current_geometry.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reference_color = "#1f77b4"
    current_color = "#d95f02"
    old_color = "#777777"

    tri = center(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    square = center(
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.1), constrained_layout=True)

    ax = axes[0]
    draw_triangle(
        ax,
        tri,
        color=reference_color,
        label="reference X",
        linestyle="--",
    )
    draw_triangle(
        ax,
        shear(tri, 0.9),
        color=current_color,
        label="current x",
    )
    ax.text(-0.78, -0.74, "fixed reference\ncurrent sheared", fontsize=9)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    set_equal_panel(ax, "Distorted Current")

    ax = axes[1]
    draw_triangle(
        ax,
        shear(tri, 1.0),
        color=reference_color,
        label="reference X",
        linestyle="--",
    )
    draw_triangle(
        ax,
        shear(tri, 0.35),
        color=current_color,
        label="current x",
    )
    ax.text(-0.78, -0.74, "reference sheared by k\ncurrent loaded by g", fontsize=9)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    set_equal_panel(ax, "Distorted Reference")

    ax = axes[2]
    draw_patch(
        ax,
        square,
        (0, 3),
        color=old_color,
        label="old shared edge",
        linestyle=":",
        alpha=0.65,
    )
    draw_patch(
        ax,
        square,
        (1, 2),
        color=reference_color,
        label="new reference edge",
        linestyle="--",
    )
    draw_patch(
        ax,
        shear(square, 0.75) + np.array([0.15, 0.0]),
        (1, 2),
        color=current_color,
        label="new current edge",
        linestyle="-",
    )
    ax.text(-0.68, -0.74, "edge flip changes topology\nnew X and x use new element", fontsize=9)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    set_equal_panel(ax, "After Edge Flip", xlim=(-0.75, 1.25), ylim=(-0.82, 0.82))

    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
