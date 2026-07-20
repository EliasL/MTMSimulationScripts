from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from Plotting.mesh_plotting import MeshFigure, MeshStyle


TRIANGLE_CONNECTIVITY = np.array([[0, 1, 2]], dtype=int)


def shear_matrix(gamma: float) -> np.ndarray:
    return np.array([[1.0, gamma], [0.0, 1.0]], dtype=float)


def shear(points: np.ndarray, gamma: float) -> np.ndarray:
    return points @ shear_matrix(gamma).T


def centered_unit_triangle() -> np.ndarray:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    return points - points.mean(axis=0)


def configure_axis(ax, *, title: str, xlim: tuple[float, float]) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(-1.08, 0.88)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_panel_text(ax, lines: list[str], *, xy: tuple[float, float]) -> None:
    ax.text(
        xy[0],
        xy[1],
        "\n".join(lines),
        fontsize=8.2,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.85", alpha=0.92),
    )


def schematic_title(integer_shear: int, local_shear: float) -> str:
    return (
        rf"$n={integer_shear}$, $s={local_shear:g}$, "
        rf"$F_{{12}}=\gamma_{{\mathrm{{cur}}}}-\gamma_{{\mathrm{{ref}}}}="
        rf"{integer_shear + local_shear:g}$"
    )


def schematic_handles(reference_color: str, current_color: str) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=reference_color,
            linestyle="--",
            linewidth=2.0,
            label=r"reference $X$",
        ),
        Line2D(
            [0],
            [0],
            color=current_color,
            linestyle="-",
            linewidth=2.0,
            label=r"current $x$",
        ),
    ]


def add_center_annotation(
    axes, *, integer_shear: int, local_shear: float, reference_color: str, current_color: str
) -> None:
    positions = [ax.get_position() for ax in axes]
    left = min(position.x0 for position in positions)
    right = max(position.x1 for position in positions)
    bottom = min(position.y0 for position in positions)
    top = max(position.y1 for position in positions)
    center_x = 0.5 * (left + right)
    height = top - bottom
    fig = axes[0].figure

    fig.text(
        center_x,
        bottom + 0.31 * height,
        schematic_title(integer_shear, local_shear),
        ha="center",
        va="center",
        fontsize=8.8,
    )
    fig.legend(
        handles=schematic_handles(reference_color, current_color),
        loc="center",
        bbox_to_anchor=(center_x, bottom + 0.16 * height),
        bbox_transform=fig.transFigure,
        fontsize=7.8,
        frameon=True,
        framealpha=0.92,
        borderpad=0.25,
        handlelength=1.8,
    )


def plot_schematic(axes, *, integer_shear: int, local_shear: float) -> None:
    axes = np.asarray(axes).ravel()
    if len(axes) != 2:
        raise ValueError(f"Expected two schematic axes, got {len(axes)}.")

    reference_color = "#1f77b4"
    current_color = "#d95f02"
    reference_style = MeshStyle(
        color=reference_color,
        linewidth=2.0,
        linestyle="--",
        node_size=18,
        node_linewidth=0.0,
    )
    current_style = MeshStyle(
        color=current_color,
        linewidth=2.0,
        linestyle="-",
        node_size=18,
        node_linewidth=0.0,
    )

    base = centered_unit_triangle()
    current_distorted_reference = base
    current_distorted_current = shear(base, integer_shear + local_shear)

    reference_distorted_reference = shear(base, -integer_shear)
    reference_distorted_current = shear(base, local_shear)

    ax = axes[0]
    mesh = MeshFigure(ax)
    mesh.draw_mesh(
        current_distorted_reference,
        TRIANGLE_CONNECTIVITY,
        style=reference_style,
    )
    mesh.draw_mesh(
        current_distorted_current,
        TRIANGLE_CONNECTIVITY,
        style=current_style,
    )
    add_panel_text(
        ax,
        [
            r"$\gamma_{\mathrm{ref}}=0$",
            rf"$\gamma_{{\mathrm{{cur}}}}={integer_shear + local_shear:g}$",
        ],
        xy=(-1.18, -1.0),
    )
    configure_axis(ax, title="distorted current geometry", xlim=(-1.35, 2.45))
    ax = axes[1]
    mesh = MeshFigure(ax)
    mesh.draw_mesh(
        reference_distorted_reference,
        TRIANGLE_CONNECTIVITY,
        style=reference_style,
    )
    mesh.draw_mesh(
        reference_distorted_current,
        TRIANGLE_CONNECTIVITY,
        style=current_style,
    )
    add_panel_text(
        ax,
        [
            rf"$\gamma_{{\mathrm{{ref}}}}=-{integer_shear}$",
            rf"$\gamma_{{\mathrm{{cur}}}}={local_shear:g}$",
        ],
        xy=(0.38, -1.0),
    )
    configure_axis(ax, title="distorted reference geometry", xlim=(-2.25, 1.45))
    add_center_annotation(
        axes,
        integer_shear=integer_shear,
        local_shear=local_shear,
        reference_color=reference_color,
        current_color=current_color,
    )


def make_figure(*, integer_shear: int, local_shear: float, out_pdf: Path, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.7, 3.1))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.92, bottom=0.08, wspace=-0.18)
    plot_schematic(axes, integer_shear=integer_shear, local_shear=local_shear)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw the current/reference element geometry for the two shear tests."
    )
    parser.add_argument("--integer-shear", type=int, default=2)
    parser.add_argument("--local-shear", type=float, default=0.5)
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("Plots/current_vs_reference_distortion_element_schematic.pdf"),
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("Plots/current_vs_reference_distortion_element_schematic.png"),
    )
    args = parser.parse_args()

    make_figure(
        integer_shear=args.integer_shear,
        local_shear=args.local_shear,
        out_pdf=args.out_pdf,
        out_png=args.out_png,
    )
    print(args.out_pdf)
    print(args.out_png)


if __name__ == "__main__":
    main()
