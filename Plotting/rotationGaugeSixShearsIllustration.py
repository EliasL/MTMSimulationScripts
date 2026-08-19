"""Illustrate the six unit-shear moves that generate r^2 = -I."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Plotting.matrix_visualization import draw_matrix_columns
from Plotting.plasticReductionCorrectnessIllustration import (
    COLUMN_COLORS,
    COLUMN_LINESTYLES,
    KNOWN_PLASTIC_F,
    TOTAL_F,
)


OUT = ROOT / "Plots" / "plastic_reduction"
OUTPUT_STEM = OUT / "rotation_gauge_six_shears_illustration"
TEXT = "#20262E"
MUTED = "#66717B"
GRID = "#CBD2D8"
SPECIAL = "#2E7D32"
LIMITS = (-2.2, 2.2)


def polar_angle(matrix):
    left, _, right_transpose = np.linalg.svd(matrix)
    rotation = left @ right_transpose
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))


def shear_sequence():
    shear_x = np.array([[1.0, 1.0], [0.0, 1.0]])
    shear_y_minus = np.array([[1.0, 0.0], [-1.0, 1.0]])
    moves = (
        (r"$S_x$", shear_x),
        (r"$S_y^{-}$", shear_y_minus),
        (r"$S_x$", shear_x),
        (r"$S_x$", shear_x),
        (r"$S_y^{-}$", shear_y_minus),
        (r"$S_x$", shear_x),
    )
    cumulative = np.eye(2)
    states = [(0, r"$Q_0=I$", cumulative.copy(), 0.0, "")]
    for step, (move_label, shear) in enumerate(moves, start=1):
        cumulative = cumulative @ shear
        if step == 3:
            label = r"$Q_3=r=S_xS_y^{-}S_x$"
            move_label = r"$r$ checkpoint"
        elif step == 6:
            label = r"$Q_6=r^2=-I$"
            move_label = r"$r^2$ checkpoint"
        else:
            label = rf"$Q_{step}=Q_{{{step - 1}}}{move_label[1:-1]}$"
        states.append(
            (step, label, cumulative.copy(), polar_angle(cumulative), move_label)
        )
    return states


def make_figure():
    known_F_p = KNOWN_PLASTIC_F.copy()
    known_F_e = TOTAL_F @ np.linalg.inv(known_F_p)
    states = shear_sequence()
    fig = plt.figure(figsize=(12.2, 15.5))

    fig.text(
        0.5,
        0.985,
        r"Six unit shears generate $r^2=-I$",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="semibold",
        color=TEXT,
    )
    fig.text(
        0.5,
        0.962,
        r"$F_{e,k}=F_eQ_k$,  $F_{p,k}=Q_k^{-1}F_p$,  "
        r"$F=F_{e,k}F_{p,k}$,  $\det Q_k=+1$",
        ha="center",
        va="top",
        fontsize=11,
        color=TEXT,
    )
    fig.text(
        0.5,
        0.943,
        r"Only $Q_3=r$ and $Q_6=r^2$ are pure orthogonal rotations; "
        r"the other angles are polar-shear rotations.",
        ha="center",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )

    column_x = (0.22, 0.49, 0.76)
    for x, title in zip(
        column_x,
        (r"$Q_k$", r"$F_{e,k}=F_eQ_k$", r"$F_{p,k}=Q_k^{-1}F_p$"),
    ):
        fig.text(
            x + 0.055,
            0.915,
            title,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color=TEXT,
        )

    axis_size = 0.105
    row_bottoms = np.linspace(0.805, 0.085, len(states))
    for (step, label, Q, angle, move_label), row_bottom in zip(
        states, row_bottoms
    ):
        row_center = row_bottom + axis_size / 2
        row_color = SPECIAL if step in (3, 6) else TEXT
        fig.text(
            0.025,
            row_center,
            label,
            ha="left",
            va="center",
            fontsize=10.5,
            color=row_color,
        )
        if step > 0 and step not in (3, 6):
            fig.text(
                0.145,
                row_center,
                "apply " + move_label,
                ha="center",
                va="center",
                fontsize=8.5,
                color=MUTED,
            )

        for x, matrix in zip(
            column_x,
            (Q, known_F_e @ Q, np.linalg.inv(Q) @ known_F_p),
        ):
            draw_matrix_columns(
                fig.add_axes([x, row_bottom, axis_size, axis_size]),
                matrix,
                limits=LIMITS,
                colors=COLUMN_COLORS,
                linestyles=COLUMN_LINESTYLES,
                linewidth=2.0,
                mutation_scale=10,
                grid_color=GRID,
                show_ticks=False,
            )

        angle_text = r"$\theta_{\rm polar}=" + f"{angle:.1f}^\\circ$"
        fig.text(
            0.895,
            row_center,
            angle_text,
            ha="left",
            va="center",
            fontsize=9,
            color=row_color if step in (3, 6) else MUTED,
        )

    fig.text(
        0.5,
        0.025,
        r"At $k=6$: $(F_eQ_6,Q_6^{-1}F_p)=(-F_e,-F_p)$, "
        r"so $F$ and both metric tensors are unchanged.",
        ha="center",
        va="bottom",
        fontsize=10,
        color=TEXT,
    )
    return fig


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig = make_figure()
    OUT.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_STEM.with_suffix(".png")
    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
