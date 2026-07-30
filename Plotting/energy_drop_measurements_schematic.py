"""Draw the schematic definitions of four energy-drop measurements.

The figure is deliberately data-free: its coordinates only control the visual
layout.  Run this file directly to write PNG and PDF copies to ``Plots/``.
"""

import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


BLUE = "#0072B2"
PURPLE = "#7656A7"
RED = "#D1495B"
GREEN = "#009E73"
NODE_SIZE = 820
TEXT_SIZE = 19
CM = 1 / 2.54


def _configure_matplotlib() -> None:
    """Use LaTeX/Computer Modern consistently for text and mathematics."""

    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "Latin Modern Roman",
                "cmr10",
            ],
            "mathtext.fontset": "cm",
            "text.latex.preamble": r"\usepackage{amsmath}",
            "axes.linewidth": 0.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _hollow_node(
    ax,
    xy,
    *,
    size=NODE_SIZE,
    edgecolor="black",
    linewidth=3.0,
    linestyle="solid",
    zorder=8,
) -> None:
    """Draw a hollow node in point units so it is always a true circle."""

    ax.scatter(
        [xy[0]],
        [xy[1]],
        s=size,
        marker="o",
        facecolor="white",
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )


def _measurement_arrow(
    ax,
    x,
    y_bottom,
    y_top,
    *,
    color,
    label,
) -> None:
    """Draw a compact double-ended arrow and a direct measurement label."""

    ax.add_patch(
        FancyArrowPatch(
            (x, y_bottom),
            (x, y_top),
            arrowstyle="<->",
            mutation_scale=15,
            shrinkA=1,
            shrinkB=1,
            color=color,
            linewidth=2.5,
            capstyle="round",
            zorder=5,
        )
    )
    ax.text(
        x + 0.06,
        0.5 * (y_bottom + y_top),
        label,
        color=color,
        fontsize=TEXT_SIZE,
        ha="left",
        va="center",
    )


def make_figure() -> tuple[plt.Figure, plt.Axes]:
    _configure_matplotlib()

    # Specify the canvas in centimetres; Matplotlib converts to its required
    # internal unit at this single API boundary.
    fig, ax = plt.subplots(
        figsize=(25.0 * CM, 16.0 * CM),
        constrained_layout=False,
    )
    # Autoscaling does not reliably include text extents.  These compact data
    # limits keep every label visible while avoiding unused interior space.
    ax.set_xlim(0.18, 6.10)
    ax.set_ylim(0.36, 5.80)
    ax.set_aspect("auto")
    ax.axis("off")

    # Geometry of the loading/relaxation construction.
    first_strain = 1.20
    strain_step = 1.25
    second_strain = first_strain + strain_step
    third_strain = second_strain + strain_step

    first = (first_strain, 1.78)
    inter = (second_strain, 2.78)
    relax_top = (third_strain, 4.55)
    equilibrium = (third_strain, 1.38)

    # Extrapolate the straight stress-corrected line through the two previous
    # equilibria.  The true-drop construction lies directly on that line.
    stress_slope = (inter[1] - first[1]) / (inter[0] - first[0])
    stress_top = (
        third_strain,
        inter[1] + stress_slope * (third_strain - second_strain),
    )
    true_x = second_strain + 0.65 * strain_step
    stress_y_at_true = inter[1] + stress_slope * (true_x - second_strain)
    true_top = (true_x, stress_y_at_true)

    measurement_start = third_strain + 0.35
    measurement_spacing = 0.52
    measurement_x = {
        "inter": measurement_start,
        "true": measurement_start + measurement_spacing,
        "stress": measurement_start + 2 * measurement_spacing,
        "relax": measurement_start + 3 * measurement_spacing,
    }

    # Light dotted guide levels shared with the measurement arrows.
    guide_style = dict(
        color="black",
        linewidth=0.85,
        linestyle=(0, (1.2, 2.3)),
        alpha=0.58,
        zorder=0,
    )
    ax.plot(
        [relax_top[0], measurement_x["relax"]],
        [relax_top[1]] * 2,
        **guide_style,
    )
    ax.plot(
        [stress_top[0], measurement_x["stress"]],
        [stress_top[1]] * 2,
        **guide_style,
    )
    ax.plot(
        [true_top[0], measurement_x["true"]],
        [true_top[1]] * 2,
        **guide_style,
    )
    ax.plot(
        [inter[0], measurement_x["inter"]],
        [inter[1]] * 2,
        **guide_style,
    )
    ax.plot(
        [equilibrium[0], measurement_x["relax"]],
        [equilibrium[1]] * 2,
        **guide_style,
    )

    # Affine prediction from the current equilibrium to relaxation.
    ax.plot(
        [inter[0], relax_top[0]],
        [inter[1], relax_top[1]],
        color="black",
        linewidth=1.15,
        linestyle=(0, (5.5, 5.5)),
        zorder=1,
    )

    # Purple stress-correction construction, using the same line styling as
    # the affine-loading path.
    ax.plot(
        [first[0], stress_top[0]],
        [first[1], stress_top[1]],
        color=PURPLE,
        linewidth=1.15,
        linestyle=(0, (5.5, 5.5)),
        zorder=1,
    )

    # Measured and corrected energy paths.
    ax.plot(
        [inter[0], equilibrium[0]],
        [inter[1], equilibrium[1]],
        color=BLUE,
        linewidth=2.5,
        zorder=2,
    )
    ax.plot(
        [true_top[0], true_top[0]],
        [equilibrium[1], true_top[1]],
        color=RED,
        linewidth=2.6,
        zorder=2,
    )
    ax.plot(
        [relax_top[0], equilibrium[0]],
        [equilibrium[1], relax_top[1]],
        color=GREEN,
        linewidth=2.6,
        zorder=2,
    )

    # Filled states and hollow predicted states.  The relaxation state is
    # intentionally hollow, as requested.
    ax.scatter(
        [first[0], inter[0], equilibrium[0]],
        [first[1], inter[1], equilibrium[1]],
        s=NODE_SIZE,
        marker="o",
        facecolor="black",
        edgecolor="black",
        linewidth=1.0,
        zorder=7,
    )
    dashed_ring = (0, (1.2, 1.25))
    _hollow_node(
        ax,
        true_top,
        edgecolor=RED,
        linewidth=3.2,
        linestyle=dashed_ring,
    )
    _hollow_node(
        ax,
        (true_top[0], equilibrium[1]),
        edgecolor=RED,
        linewidth=3.2,
        linestyle=dashed_ring,
    )
    _hollow_node(
        ax,
        stress_top,
        edgecolor=PURPLE,
        linewidth=3.2,
        linestyle=dashed_ring,
    )
    _hollow_node(ax, relax_top, linewidth=3.0)

    # Direct measurement arrows replace the visually heavy curly braces.
    _measurement_arrow(
        ax,
        measurement_x["inter"],
        equilibrium[1],
        inter[1],
        color=BLUE,
        label=r"$\Delta E_I$",
    )
    _measurement_arrow(
        ax,
        measurement_x["true"],
        equilibrium[1],
        true_top[1],
        color=RED,
        label=r"$\Delta E_T$",
    )
    _measurement_arrow(
        ax,
        measurement_x["stress"],
        equilibrium[1],
        stress_top[1],
        color=PURPLE,
        label=r"$\Delta E_S$",
    )
    _measurement_arrow(
        ax,
        measurement_x["relax"],
        equilibrium[1],
        relax_top[1],
        color=GREEN,
        label=r"$\Delta E_R$",
    )

    # State and process labels.  Matplotlib transforms this data-space angle so
    # the text remains exactly parallel to the affine path on screen.
    affine_dx = relax_top[0] - inter[0]
    affine_dy = relax_top[1] - inter[1]
    affine_length = math.hypot(affine_dx, affine_dy)
    affine_midpoint = (
        0.5 * (inter[0] + relax_top[0]),
        0.5 * (inter[1] + relax_top[1]),
    )
    label_offset = 0.24
    affine_label = (
        affine_midpoint[0] - label_offset * affine_dy / affine_length,
        affine_midpoint[1] + label_offset * affine_dx / affine_length,
    )
    affine_angle = math.degrees(math.atan2(affine_dy, affine_dx))
    ax.text(
        affine_label[0],
        affine_label[1],
        r"Affine loading",
        fontsize=TEXT_SIZE,
        rotation=affine_angle,
        transform_rotates_text=True,
        rotation_mode="anchor",
        ha="center",
        va="center",
    )
    ax.text(
        relax_top[0] + 0.72,
        relax_top[1] + 0.12,
        r"Relaxation",
        fontsize=TEXT_SIZE,
        ha="center",
        va="center",
    )
    ax.text(
        equilibrium[0] + 0.22,
        equilibrium[1] - 0.27,
        r"Equilibrium state",
        fontsize=TEXT_SIZE,
        ha="left",
        va="center",
    )

    # Main energy and strain axes.
    arrow_kw = dict(
        arrowstyle="-|>",
        color="black",
        linewidth=4.0,
        mutation_scale=28,
        shrinkA=0,
        shrinkB=0,
        capstyle="butt",
        joinstyle="miter",
        zorder=10,
    )
    ax.add_patch(FancyArrowPatch((0.66, 1.56), (0.66, 4.78), **arrow_kw))
    strain_axis_y = 0.78
    strain_label_y = strain_axis_y - 0.36
    ax.add_patch(
        FancyArrowPatch((0.94, strain_axis_y), (5.35, strain_axis_y), **arrow_kw)
    )
    ax.text(
        0.31,
        3.15,
        r"Energy",
        fontsize=TEXT_SIZE,
        rotation=90,
        ha="center",
        va="center",
    )
    ax.text(
        5.02,
        strain_label_y,
        r"Strain",
        fontsize=TEXT_SIZE,
        ha="center",
        va="bottom",
    )

    # Strain-step ticks and labels.  The final mark is the critical strain
    # used for the true-drop construction, directly below the ΔE_T arrow.
    strain_marks = (
        (first[0], r"$\gamma_{n-1}$"),
        (inter[0], r"$\gamma_{n}$"),
        (equilibrium[0], r"$\gamma_{n+1}$"),
        (true_top[0], r"$\gamma_c$"),
    )
    for x_mark, label in strain_marks:
        ax.plot(
            [x_mark, x_mark],
            [strain_axis_y - 0.13, strain_axis_y + 0.13],
            color="black",
            linewidth=1.35,
            solid_capstyle="butt",
            zorder=11,
        )
        ax.text(
            x_mark,
            strain_label_y,
            label,
            fontsize=TEXT_SIZE,
            ha="center",
            va="bottom",
            zorder=11,
        )

    # Two aligned heading rows replace the staggered title arrangement.
    ax.text(
        first_strain,
        5.62,
        r"Inter-strain energy drop: $\Delta E_I$",
        color=BLUE,
        fontsize=TEXT_SIZE,
        ha="left",
        va="center",
    )
    ax.text(
        3.25,
        5.62,
        r"Stress-corrected energy drop: $\Delta E_S$",
        color=PURPLE,
        fontsize=TEXT_SIZE,
        ha="left",
        va="center",
    )
    ax.text(
        first_strain,
        5.27,
        r"True energy drop: $\Delta E_T$",
        color=RED,
        fontsize=TEXT_SIZE,
        ha="left",
        va="center",
    )
    ax.text(
        3.25,
        5.27,
        r"Relaxation energy drop: $\Delta E_R$",
        color=GREEN,
        fontsize=TEXT_SIZE,
        ha="left",
        va="center",
    )

    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.01, top=0.995)
    return fig, ax


def main() -> None:
    fig, _ = make_figure()
    output_dir = Path(__file__).resolve().parents[1] / "Plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / "energy_drop_measurements_schematic"
    save_options = {"bbox_inches": "tight", "pad_inches": 0}
    fig.savefig(output_stem.with_suffix(".png"), dpi=160, **save_options)
    fig.savefig(output_stem.with_suffix(".pdf"), **save_options)
    plt.close(fig)


if __name__ == "__main__":
    main()
