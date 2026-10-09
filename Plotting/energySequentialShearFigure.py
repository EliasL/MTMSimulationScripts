"""Draw the sequential-shear energy figure from its original Grace datasets.

Retain orientation II (G0.S1) and reconnection (G0.S2). The original energy
datasets are separate from the uploaded coarse-step mesh runs; A–D indicate
the common protocol displacement values, rather than identical trajectories.
The horizontal-first protocol ends at (u_x, u_y) = (1, 3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Plotting.meshLockingFigure import (
    DEFAULT_LOADS,
    ROOT,
    _configure_matplotlib,
    _nice_energy_max,
)


def read_grace_datasets(path: Path) -> dict[str, np.ndarray]:
    """Read the embedded XY datasets without altering their sampling."""
    datasets = {}
    target = None
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("@target "):
            target = line.split()[1]
            rows = []
        elif line.strip() == "&" and target is not None:
            datasets[target] = np.asarray(rows, dtype=float)
            target = None
        elif target is not None and line.strip() and not line.startswith(("@", "#")):
            rows.append([float(value) for value in line.split()])
    return datasets


def make_energy_sequential_shear_figure(
    source: Path,
    *,
    no_reconnection_run: Path | None = None,
    edge_flipping_run: Path | None = None,
):
    """Show orientation II and the complete protocol, with mesh snapshot guides."""
    if (no_reconnection_run is None) != (edge_flipping_run is None):
        raise ValueError("Supply both explicit run paths, or neither.")
    if no_reconnection_run is None:
        data = read_grace_datasets(source)
    else:
        data = {
            key: pd.read_csv(Path(run) / "macroData.csv")
            [["load", "total_energy"]].to_numpy(dtype=float)
            for key, run in (("G0.S1", no_reconnection_run),
                             ("G0.S2", edge_flipping_run))
        }
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    fig.subplots_adjust(left=0.12, right=0.965, bottom=0.19, top=0.91)
    curves = (
        ("G0.S1", "Fixed connectivity", "#222222", (0, (5, 3))),
        ("G0.S2", "Reconnection", "#0072B2", "-"),
    )
    for key, label, color, linestyle in curves:
        values = data[key]
        if values[0, 0] > 0 or values[-1, 0] < 4:
            raise ValueError(f"{key} does not cover the full protocol from load 0 to 4.")
        mask = (values[:, 0] >= 0) & (values[:, 0] <= 4 + 1e-9)
        ax.plot(values[mask, 0], values[mask, 1], label=label,
                color=color, linestyle=linestyle, linewidth=2)

    for letter, load in zip("ABCD", DEFAULT_LOADS):
        ax.axvline(load, color="#6B7077", linestyle=(0, (2, 3)),
                   linewidth=1, alpha=0.65, zorder=0)
        ax.text(load, 3, letter, ha="center", va="center", fontsize=17,
                fontweight="bold", zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        for key, _, color, _ in curves:
            values = data[key]
            index = int(np.argmin(np.abs(values[:, 0] - load)))
            if not np.isclose(values[index, 0], load, atol=1e-7, rtol=0):
                raise ValueError(f"{key} has no observation at snapshot {letter}.")
            ax.plot(load, values[index, 1], "o", markersize=5.5,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.2, clip_on=False, zorder=5)

    ax.set_xlim(0, 4)
    maximum = max(
        float(np.max(data[key][(data[key][:, 0] >= 0) & (data[key][:, 0] <= 4), 1]))
        for key, _, _, _ in curves
    )
    energy_max = max(3.5, _nice_energy_max(maximum))
    ax.set_ylim(0, energy_max)
    ax.set_xticks(range(5))
    ax.set_xticklabels([
        "$u_x=0$\n$u_y=0$", "$u_x=1$\n$u_y=0$",
        "$u_x=1$\n$u_y=1$", "$u_x=1$\n$u_y=2$",
        "$u_x=1$\n$u_y=3$",
    ], fontsize=15)
    ax.set_yticks(np.arange(0, energy_max + 1e-9, 1))
    ax.set_ylabel("Internal energy", fontsize=19)
    ax.tick_params(direction="in", top=True, right=True, length=6,
                   axis="y", labelsize=15)
    ax.tick_params(axis="x", direction="in", top=True, length=6, pad=9)
    ax.legend(loc="upper left", fontsize=16)
    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "Plots" / "energy_sequential_shear.agr")
    parser.add_argument("--output-stem", type=Path,
                        default=ROOT / "Plots" / "energy_sequential_shear")
    parser.add_argument("--no-reconnection-run", type=Path,
                        help="Use this run's energy instead of the Grace dataset.")
    parser.add_argument("--edge-flipping-run", type=Path,
                        help="Use this run's energy instead of the Grace dataset.")
    args = parser.parse_args()
    fig, _ = make_energy_sequential_shear_figure(
        args.source, no_reconnection_run=args.no_reconnection_run,
        edge_flipping_run=args.edge_flipping_run,
    )
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
