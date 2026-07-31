"""Illustrations of the forward--backward reversibility protocol.

The figures are intentionally schematic: they use the small mesh-drawing
helpers in :mod:`Plotting.mesh_plotting` rather than simulation output.  The
protocol panels show a saved initial mesh, forward loading and relaxation,
reverse loading and relaxation, and the comparison with the saved state.

Run this file directly to write PNG and PDF copies to ``Plots/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Keep Matplotlib's font cache in the project, even when the shell has a
# non-writable global MPLCONFIGDIR configured.
os.environ["MPLCONFIGDIR"] = str(ROOT / ".matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from MTMath.meshUtils import structured_triangular_mesh
from Plotting.mesh_plotting import MeshFigure, MeshStyle


PROTOCOL_GAMMA = 0.65
REVERSIBLE_COLOR = "C0"
IRREVERSIBLE_COLOR = "C1"


def simple_shear(points: np.ndarray, gamma: float) -> np.ndarray:
    """Apply horizontal simple shear to row-wise two-dimensional points."""

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (n_points, 2), got {points.shape}.")
    return points @ np.array([[1.0, gamma], [0.0, 1.0]]).T


def shift_upper_nodes(nodes: np.ndarray, *, shift: float = 1.0) -> np.ndarray:
    """Shift the two rightmost nodes on the upper boundary by one lattice step."""

    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"nodes must have shape (n_points, 2), got {nodes.shape}.")

    result = nodes.copy()
    ymax = np.max(result[:, 1])
    upper_ids = np.flatnonzero(np.isclose(result[:, 1], ymax))
    if len(upper_ids) < 2:
        raise ValueError("At least two nodes are required on the upper boundary.")
    result[upper_ids[-2:], 0] += shift
    return result


def build_protocol_states(
    gamma: float = PROTOCOL_GAMMA,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[np.ndarray]]]:
    """Return the common mesh and the two schematic protocol histories."""

    initial_nodes, connectivity = structured_triangular_mesh(
        (4, 4), diagonal="minor"
    )
    forward_nodes = simple_shear(initial_nodes, gamma)
    dislocated_initial = shift_upper_nodes(initial_nodes)
    dislocated_forward = simple_shear(dislocated_initial, gamma)
    reverse_reversible = simple_shear(forward_nodes, -gamma)
    reverse_irreversible = simple_shear(dislocated_forward, -gamma)

    states = {
        "reversible": [
            initial_nodes,
            forward_nodes,
            forward_nodes.copy(),
            reverse_reversible,
            initial_nodes.copy(),
        ],
        "irreversible": [
            initial_nodes,
            forward_nodes,
            dislocated_forward,
            reverse_irreversible,
            dislocated_initial,
        ],
    }
    return initial_nodes, connectivity, states


def _mesh_style(*, color: str = "0.25") -> MeshStyle:
    """Use the project mesh defaults, with only a small schematic adjustment."""

    return MeshStyle(
        color=color,
        face_alpha=0.10,
        linewidth=1.05,
        node_size=14.0,
        node_linewidth=0.6,
        zorder=2.0,
    )


def _draw_protocol_panel(
    ax: plt.Axes,
    nodes: np.ndarray,
    connectivity: np.ndarray,
) -> None:
    """Draw one clean schematic mesh state."""

    mesh = MeshFigure(ax)
    mesh.draw_mesh(nodes, connectivity, style=_mesh_style())
    mesh.configure_axis(
        xlim=(-0.20, 6.20),
        ylim=(-0.20, 3.20),
        equal_aspect=True,
        hide_axes=True,
    )


def make_protocol_figure(
    gamma: float = PROTOCOL_GAMMA,
) -> tuple[plt.Figure, np.ndarray]:
    """Create the reversible/irreversible mesh protocol schematic."""

    _, connectivity, states = build_protocol_states(gamma)
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(11.5, 4.35),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.06, "hspace": 0.38},
    )

    titles = (
        ("state 0", "saved; minimized"),
        ("state 1", rf"affine $+\gamma$"),
        ("state 2", "relaxed"),
        ("state 3", rf"affine $-\gamma$"),
        ("state 4", "relaxed; same as 0"),
    )
    primed_titles = (
        ("state 0", "saved; minimized"),
        ("state 1", rf"affine $+\gamma$"),
        (r"state $2'$", "relaxed"),
        (r"state $3'$", rf"affine $-\gamma$"),
        (r"state $4'$", "relaxed; differs from 0"),
    )

    for row, branch in enumerate(("reversible", "irreversible")):
        for column, nodes in enumerate(states[branch]):
            _draw_protocol_panel(
                axes[row, column],
                nodes,
                connectivity,
            )
            title, subtitle = (titles if row == 0 else primed_titles)[column]
            axes[row, column].set_title(
                f"{title}\n{subtitle}",
                loc="left",
                fontsize=8.0,
                linespacing=1.15,
                pad=3.0,
            )

        axes[row, 0].text(
            -0.24,
            0.50,
            "reversible" if row == 0 else "irreversible",
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.0,
        )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.83)
    return figure, axes


def _annotate_state(ax: plt.Axes, x: float, y: float, label: str, dx: float, dy: float) -> None:
    """Label one protocol point with an explicit offset from its marker."""

    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        textcoords="data",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=8.5,
        zorder=6,
    )


def make_energy_strain_figure() -> tuple[plt.Figure, plt.Axes]:
    """Create the schematic energy--strain version of the protocol."""

    reversible = np.array(
        [
            [0.00, 0.00],  # state 0
            [0.65, 0.46],  # state 1
            [0.65, 0.22],  # state 2
            [0.00, 0.34],  # state 3
            [0.00, 0.00],  # state 4
        ]
    )
    irreversible = np.array(
        [
            [0.00, 0.00],  # state 0
            [0.65, 0.46],  # state 1
            [0.65, 0.19],  # state 2'
            [0.00, 0.05],  # state 3'
            [0.00, -0.16],  # state 4'
        ]
    )

    figure, ax = plt.subplots(figsize=(6.0, 4.15))
    ax.plot(
        reversible[:, 0],
        reversible[:, 1],
        "o-",
        color=REVERSIBLE_COLOR,
        linewidth=1.7,
        markersize=5.5,
        label="elastic / reversible",
        zorder=3,
    )
    ax.plot(
        irreversible[:, 0],
        irreversible[:, 1],
        "o--",
        color=IRREVERSIBLE_COLOR,
        linewidth=1.7,
        markersize=5.5,
        label="plastic / irreversible",
        zorder=2,
    )

    # The shared states are labelled once.  States 2--4 receive primed labels
    # on the irreversible path so both histories remain readable.
    _annotate_state(ax, *reversible[0], "0", 0.014, 0.025)
    _annotate_state(ax, *reversible[1], "1", 0.014, 0.025)
    _annotate_state(ax, *reversible[2], "2", 0.014, 0.015)
    _annotate_state(ax, *irreversible[2], r"$2'$", 0.014, -0.100)

    # Put the return-state labels on the left of their nodes so they do not
    # collide with state 0 at the origin.
    _annotate_state(ax, *reversible[3], "3", -0.020, 0.018)
    _annotate_state(ax, *irreversible[3], r"$3'$", -0.020, 0.080)
    _annotate_state(ax, *reversible[4], "4", -0.020, -0.020)
    _annotate_state(ax, *irreversible[4], r"$4'$", -0.020, 0.000)

    ax.set_xlabel(r"strain $\gamma$")
    ax.set_ylabel(r"$E$")
    ax.set_xlim(-0.105, 0.76)
    ax.set_ylim(-0.225, 0.52)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    figure.tight_layout()
    return figure, ax


def make_figures(
    output_dir: Path = ROOT / "Plots",
) -> tuple[Path, Path, Path, Path]:
    """Write both figures and return their PNG/PDF paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_figure, _ = make_protocol_figure()
    energy_figure, _ = make_energy_strain_figure()

    protocol_stem = output_dir / "reversibility_protocol_schematic"
    energy_stem = output_dir / "reversibility_energy_strain_schematic"
    protocol_pdf = protocol_stem.with_suffix(".pdf")
    protocol_png = protocol_stem.with_suffix(".png")
    energy_pdf = energy_stem.with_suffix(".pdf")
    energy_png = energy_stem.with_suffix(".png")
    protocol_figure.savefig(protocol_pdf, bbox_inches="tight")
    protocol_figure.savefig(protocol_png, dpi=300, bbox_inches="tight")
    energy_figure.savefig(energy_pdf, bbox_inches="tight")
    energy_figure.savefig(energy_png, dpi=300, bbox_inches="tight")
    plt.close(protocol_figure)
    plt.close(energy_figure)
    return protocol_png, protocol_pdf, energy_png, energy_pdf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "Plots",
        help="Directory for the PNG and PDF figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for path in make_figures(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
