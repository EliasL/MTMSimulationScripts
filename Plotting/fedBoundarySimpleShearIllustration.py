"""Plot the four FED boundaries and the corresponding reduction shears.

The figure uses the determinant-one Poincare-disk coordinates already used
throughout the project.  Each colored face is annotated by its defining
equality, and a representative configuration just outside that face is shown
being reduced by the corresponding unit simple shear.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MTMath.poincareEnergy import C2PoincareDisk, drawPoincareGrid


OUTPUT_STEM = ROOT / "Plots" / "plastic_reduction" / "fed_boundary_simple_shear_illustration"

TEXT = "#20262E"
GRID = "#C7CED6"
DISK_EDGE = "#7A8793"
DISK_GRID_SIZE = 1000
BOUNDARY_COLORS = {
    "11+": "#0072B2",  # blue
    "11-": "#D55E00",  # vermillion
    "22+": "#009E73",  # green
    "22-": "#CC79A7",  # purple
}


@dataclass(frozen=True)
class BoundarySpec:
    """One of the four outer FED faces and its elementary shear convention."""

    key: str
    label: str
    diagonal: str
    sign: int
    shear_family: str
    color: str

    @property
    def equation(self) -> str:
        diagonal = rf"C_{{{self.diagonal}}}"
        other_diagonal = "22" if self.diagonal == "11" else "11"
        coefficient = "2" if self.sign > 0 else "-2"
        relation = rf"{coefficient}C_{{12}}={diagonal}"
        return relation + rf",\quad {diagonal}\leq C_{{{other_diagonal}}}"

    @property
    def shear_matrix_label(self) -> str:
        argument = -self.sign
        indices = "12" if self.shear_family == "U" else "21"
        return rf"$\mathbf{{M}}^{{({indices})}}({argument})$"

    @property
    def shear_indices(self) -> str:
        return "12" if self.shear_family == "U" else "21"


BOUNDARIES = (
    BoundarySpec("11+", r"$\Gamma_{11}^{+}$", "11", 1, "U", BOUNDARY_COLORS["11+"]),
    BoundarySpec("11-", r"$\Gamma_{11}^{-}$", "11", -1, "U", BOUNDARY_COLORS["11-"]),
    BoundarySpec("22+", r"$\Gamma_{22}^{+}$", "22", 1, "V", BOUNDARY_COLORS["22+"]),
    BoundarySpec("22-", r"$\Gamma_{22}^{-}$", "22", -1, "V", BOUNDARY_COLORS["22-"]),
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def boundary_metrics(
    spec: BoundarySpec,
    resolution: int = 700,
    a_min: float = 1.0e-5,
) -> np.ndarray:
    """Return determinant-one metrics along one complete FED boundary face."""

    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if a_min <= 0.0:
        raise ValueError("a_min must be positive")

    # On a boundary face, the smaller diagonal is a and |C12|=a/2.
    # det(C)=1 then fixes the larger diagonal to (a^2+4)/(4a).  The endpoint
    # where the two diagonals meet is a=2/sqrt(3).
    a = np.geomspace(a_min, 2.0 / np.sqrt(3.0), resolution)
    c12 = spec.sign * 0.5 * a
    larger = (1.0 + c12 * c12) / a

    C = np.zeros((resolution, 2, 2), dtype=float)
    if spec.diagonal == "11":
        C[:, 0, 0] = a
        C[:, 1, 1] = larger
    elif spec.diagonal == "22":
        C[:, 0, 0] = larger
        C[:, 1, 1] = a
    else:
        raise ValueError(f"Unknown smaller diagonal: {spec.diagonal}")
    C[:, 0, 1] = c12
    C[:, 1, 0] = c12
    return C


def simple_shear_metrics(
    spec: BoundarySpec,
    gamma: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return F and C along the matching unit-area simple-shear path."""

    gamma = np.asarray(gamma, dtype=float)
    F = np.zeros(gamma.shape + (2, 2), dtype=float)
    F[..., 0, 0] = 1.0
    F[..., 1, 1] = 1.0
    signed_gamma = spec.sign * gamma
    if spec.shear_family == "U":
        F[..., 0, 1] = signed_gamma
    elif spec.shear_family == "V":
        F[..., 1, 0] = signed_gamma
    else:
        raise ValueError(f"Unknown shear family: {spec.shear_family}")
    C = np.einsum("...ji,...jk->...ik", F, F)
    return F, C


def _poincare_xy(C: np.ndarray) -> np.ndarray:
    x, y = C2PoincareDisk(C)
    return np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))


def _native_to_plot(xy: np.ndarray) -> np.ndarray:
    """Map native disk coordinates in [-1, 1]^2 to the reusable grid scale."""

    return (np.asarray(xy, dtype=float) + 1.0) * (DISK_GRID_SIZE / 2.0)


def _format_tick(value: float) -> str:
    """Format a native disk coordinate without floating-point noise at zero."""

    if abs(value) < 1.0e-12:
        value = 0.0
    return f"{value:g}"


def _draw_arrow(
    ax,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(end),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.25,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=6,
        )
    )


def unit_shear_matrix(spec: BoundarySpec, argument: int) -> np.ndarray:
    """Return the horizontal/vertical unit simple-shear matrix M^(ij)(argument)."""

    if argument not in (-1, 1):
        raise ValueError("the reduction shear argument must be -1 or +1")
    M = np.eye(2)
    if spec.shear_family == "U":
        M[0, 1] = argument
    elif spec.shear_family == "V":
        M[1, 0] = argument
    else:
        raise ValueError(f"Unknown shear family: {spec.shear_family}")
    return M


def reduction_example(
    spec: BoundarySpec,
    gamma_out: float = 0.58,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an outside metric, its reduced metric, and the active shear M."""

    _, C_out = simple_shear_metrics(spec, gamma_out)
    M = unit_shear_matrix(spec, -spec.sign)
    C_reduced = M.T @ C_out @ M
    return C_out, C_reduced, M


def _draw_boundary_equation(
    ax,
    spec: BoundarySpec,
) -> None:
    """Place each face equation inside the FED near the horizontal axis."""

    anchor = {
        "11+": (-0.3, 0.2),
        "11-": (-0.3, -0.2),
        "22+": (0.3, 0.2),
        "22-": (0.3, -0.2),
    }[spec.key]
    anchor = _native_to_plot(np.array(anchor))
    ax.text(
        anchor[0],
        anchor[1],
        rf"${spec.equation}$",
        ha="right" if spec.diagonal == "11" else "left",
        va="center",
        color=spec.color,
        fontsize=8.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        zorder=8,
    )


def _draw_reduction_arrow(ax, spec: BoundarySpec) -> None:
    """Show a representative outside configuration being reduced into the FED."""

    C_out, C_reduced, _ = reduction_example(spec)
    start = _native_to_plot(_poincare_xy(C_out)[0])
    end = _native_to_plot(_poincare_xy(C_reduced)[0])

    _draw_arrow(ax, start, end, spec.color)
    ax.scatter(
        start[0],
        start[1],
        s=30,
        facecolor="white",
        edgecolor=spec.color,
        linewidth=1.25,
        zorder=12,
    )
    ax.scatter(
        end[0],
        end[1],
        s=27,
        facecolor=spec.color,
        edgecolor="white",
        linewidth=0.8,
        zorder=12,
    )

    # Keep the reduction labels near their outside starting points, leaving
    # the central part of the disk readable where the four arrows pass.
    offset = {
        "11+": (-3, 3),
        "11-": (-3, -4),
        "22+": (3, 3),
        "22-": (3, -4),
    }[spec.key]
    ax.annotate(
        spec.shear_matrix_label,
        xy=start,
        xytext=offset,
        textcoords="offset points",
        ha="right" if offset[0] < 0 else "left",
        va="bottom" if offset[1] > 0 else "top",
        color=spec.color,
        fontsize=9.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.4},
        zorder=9,
    )


def make_figure(
    *,
    output_stem: Path = OUTPUT_STEM,
    boundary_resolution: int = 700,
    dpi: int = 260,
) -> tuple[Path, Path]:
    """Create the annotated FED boundary figure."""

    boundary_xy_native = {
        spec.key: _poincare_xy(boundary_metrics(spec, boundary_resolution))
        for spec in BOUNDARIES
    }
    boundary_xy = {
        key: _native_to_plot(xy) for key, xy in boundary_xy_native.items()
    }

    fig, ax = plt.subplots(figsize=(6.016, 3.008), constrained_layout=True)

    drawPoincareGrid(
        ax=ax,
        grid_size=DISK_GRID_SIZE,
        zoom=1,
        depth=5,
        c=GRID,
        alpha=0.48,
        linewidth=0.45,
        zorder=0,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 800)
    ax.plot(
        _native_to_plot(np.cos(theta)),
        _native_to_plot(np.sin(theta)),
        color=DISK_EDGE,
        linewidth=0.85,
        alpha=0.65,
        zorder=2,
    )
    ax.axhline(
        _native_to_plot(0.0),
        color=GRID,
        linewidth=0.75,
        linestyle=(0, (2.0, 2.5)),
        zorder=3,
    )
    ax.axvline(
        _native_to_plot(0.0),
        color=GRID,
        linewidth=0.75,
        linestyle=(0, (2.0, 2.5)),
        zorder=3,
    )

    for spec in BOUNDARIES:
        xy = boundary_xy[spec.key]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=spec.color,
            linewidth=2.4,
            solid_capstyle="round",
            zorder=4,
        )
        _draw_boundary_equation(ax, spec)

    for spec in BOUNDARIES:
        _draw_reduction_arrow(ax, spec)

    ax.set_xlabel(r"$x_p$", color=TEXT)
    ax.set_ylabel(r"$y_p$", color=TEXT)
    ax.set_xlim(*_native_to_plot(np.array([-1.04, 1.04])))
    ax.set_ylim(*_native_to_plot(np.array([-0.36, 0.36])))
    x_ticks = np.arange(-1.0, 1.01, 0.25)
    y_ticks = np.arange(-0.3, 0.31, 0.1)
    ax.set_xticks(_native_to_plot(x_ticks))
    ax.set_xticklabels([_format_tick(value) for value in x_ticks])
    ax.set_yticks(_native_to_plot(y_ticks))
    ax.set_yticklabels([_format_tick(value) for value in y_ticks])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=OUTPUT_STEM,
        help="Path without extension for the PNG and PDF outputs.",
    )
    args = parser.parse_args()
    png_path, pdf_path = make_figure(output_stem=args.output_stem)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    _style()
    main()
