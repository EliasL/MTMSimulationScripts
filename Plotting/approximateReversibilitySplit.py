"""Approximate reversible/irreversible splits from :math:`\\Delta E_R`.

The split is deliberately exploratory.  It first computes the exact
log-space Otsu cut used elsewhere in the reversibility analysis, then starts
at the corresponding histogram bin and descends to a neighbouring local
minimum.  The search uses probability mass per bin, not density.  A
zero-count plateau is represented by its middle bin.

This is not the standard power-law workflow.  For the standard result, use
post-yield ``kappa_det = mu/(2 rho)`` with ``rho=N/V_0=2`` for the reversible/irreversible split,
fit only irreversible ``Delta E_S`` events, search all observed xmin
candidates, and then perform the maximum-likelihood fit.

Run the size-scaling diagnostic with::

    .venv/bin/python -m Plotting.approximateReversibilitySplit
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from MTMath.evaluatePowerlawFit import POWERLAW_STANDARD_WORKFLOW
from Plotting.numericalParameterJustification import unbinned_log_otsu_cut
from Plotting.sizeScalingCollapse import (
    REGIMES,
    completed_size_scaling_paths,
    pool_drops,
)


DEFAULT_SIZES = (50, 100, 150, 200, 250)
DEFAULT_OUTPUT_DIR = Path(
    "Plots/powerLaw/size_collapse/approximate_reversibility_split"
)
FIGURE_DPI = 250


@dataclass(frozen=True)
class ApproximateReversibilitySplitResult:
    """Binned probability mass and split diagnostics for one drop population."""

    cut: float
    otsu_cut: float
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    counts: np.ndarray
    probability: np.ndarray
    otsu_bin: int
    minimum_bin: int
    local_minimum_bins: np.ndarray
    descent_path: np.ndarray


def _positive_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 50:
        raise ValueError(
            "ApproximateReversibilitySplit needs at least 50 positive values; "
            f"got {values.size}."
        )
    return values


def _log_histogram(values: np.ndarray, bins_per_decade: int) -> tuple[np.ndarray, ...]:
    if bins_per_decade < 1:
        raise ValueError("bins_per_decade must be a positive integer.")
    low, high = np.log10(values.min()), np.log10(values.max())
    if np.isclose(low, high):
        low -= 0.5
        high += 0.5
    n_bins = max(10, int(np.ceil((high - low) * bins_per_decade)))
    edges = np.logspace(low, high, n_bins + 1)
    counts, edges = np.histogram(values, bins=edges)
    probability = counts / values.size
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers, counts.astype(np.int64), probability


def _descending_minimum(
    probability: np.ndarray,
    otsu_bin: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Descend from Otsu to an interior local minimum in bin probability."""
    if probability.size < 3:
        raise ValueError("At least three histogram bins are required.")

    # The two largest sides of the Otsu cut define the interval in which the
    # valley between the two bumps is sought.  This prevents the descent from
    # wandering into a low-count tail at either end of the histogram.
    left_peak = int(np.argmax(probability[: otsu_bin + 1]))
    right_peak = otsu_bin + int(np.argmax(probability[otsu_bin:]))
    lower = min(left_peak, otsu_bin)
    upper = max(right_peak, otsu_bin)
    lower = max(1, lower)
    upper = min(probability.size - 2, upper)
    if lower > upper:
        lower, upper = 1, probability.size - 2

    current = int(np.clip(otsu_bin, lower, upper))
    path = [current]
    while True:
        neighbours = [
            index
            for index in (current - 1, current + 1)
            if lower <= index <= upper
        ]
        if not neighbours:
            break
        lowest = min(probability[index] for index in neighbours)
        if lowest >= probability[current]:
            break
        # A tie is resolved toward the Otsu bin so the search remains local.
        next_index = min(
            (index for index in neighbours if probability[index] == lowest),
            key=lambda index: abs(index - otsu_bin),
        )
        current = next_index
        path.append(current)

    minimum_probability = probability[current]
    local_minimum_bins = np.flatnonzero(
        probability[lower : upper + 1] == minimum_probability
    ) + lower
    if minimum_probability == 0:
        # Prefer the zero plateau containing the descended minimum.  If the
        # histogram contains separated zero bins, retain only the local zero
        # plateau reached by the descent.
        plateau = [current]
        index = current - 1
        while index >= lower and probability[index] == 0:
            plateau.append(index)
            index -= 1
        index = current + 1
        while index <= upper and probability[index] == 0:
            plateau.append(index)
            index += 1
        local_minimum_bins = np.asarray(sorted(plateau), dtype=int)
        current = int(local_minimum_bins[len(local_minimum_bins) // 2])
    else:
        # For a non-zero minimum, use the midpoint of a flat local minimum.
        current = int(local_minimum_bins[len(local_minimum_bins) // 2])

    return current, local_minimum_bins, np.asarray(path, dtype=int)


def ApproximateReversibilitySplit(
    values: np.ndarray,
    *,
    bins_per_decade: int = 10,
    min_class_fraction: float = 0.02,
) -> ApproximateReversibilitySplitResult:
    """Estimate a reversible/irreversible split from positive ``Delta E_R``.

    Otsu's log-space cut supplies the starting bin.  A nearest-neighbour
    descent in the binned probability mass then finds the valley between the
    bumps.  If that valley contains zero-probability bins, the middle zero bin
    is selected.
    The returned ``cut`` is the geometric centre of the selected log bin.
    """
    values = _positive_values(values)
    otsu_cut, _ = unbinned_log_otsu_cut(
        values,
        min_class_fraction=min_class_fraction,
    )
    edges, centers, counts, probability = _log_histogram(values, bins_per_decade)
    otsu_bin = int(np.searchsorted(edges, otsu_cut, side="right") - 1)
    otsu_bin = int(np.clip(otsu_bin, 0, counts.size - 1))
    minimum_bin, local_minimum_bins, descent_path = _descending_minimum(
        probability,
        otsu_bin,
    )
    return ApproximateReversibilitySplitResult(
        cut=float(centers[minimum_bin]),
        otsu_cut=float(otsu_cut),
        bin_edges=edges,
        bin_centers=centers,
        counts=counts,
        probability=probability,
        otsu_bin=otsu_bin,
        minimum_bin=minimum_bin,
        local_minimum_bins=local_minimum_bins,
        descent_path=descent_path,
    )


approximate_reversibility_split = ApproximateReversibilitySplit


def _plot_split(
    size: int,
    results: dict[str, ApproximateReversibilitySplitResult],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=False)
    for ax, regime in zip(axes, REGIMES):
        result = results[regime]
        positive_probability = result.probability[result.probability > 0]
        if positive_probability.size == 0:
            raise RuntimeError(f"L={size}, {regime}: histogram probability is empty.")
        ax.stairs(
            result.probability,
            result.bin_edges,
            color="C0",
            linewidth=1.3,
            label=r"binned probability $P(\Delta E_R\in\mathrm{bin})$",
        )
        ax.axvline(
            result.otsu_cut,
            color="C1",
            linestyle="--",
            linewidth=1.2,
            label="Otsu cut",
        )
        ax.axvline(
            result.cut,
            color="black",
            linewidth=1.5,
            label="descending local minimum",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=float(positive_probability.min()) / 2.0)
        ax.set_xlabel(r"$\Delta E_R$")
        ax.set_ylabel(r"$P(\Delta E_R\in\mathrm{bin})$")
        ax.set_title("Pre-yield" if regime == "pre" else "Post-yield")
        ax.grid(alpha=0.2)
        ax.text(
            0.03,
            0.96,
            f"Otsu={result.otsu_cut:.3e}\n"
            f"split={result.cut:.3e}\n"
            f"zero bins={np.count_nonzero(result.counts == 0)}",
            transform=ax.transAxes,
            va="top",
            fontsize="small",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    axes[0].legend(fontsize="small")
    fig.suptitle(f"Approximate reversibility split; L={size}")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"L{size}.png"
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(output_dir / f"L{size}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}", flush=True)


def run_size_scaling_diagnostic(
    data_root: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    seeds_per_size: int = 6,
    bins_per_decade: int = 10,
) -> dict:
    """Generate one pre/post diagnostic figure for each requested size."""
    paths, inventory = completed_size_scaling_paths(
        data_root,
        seeds_per_size,
        REGIMES["post"][1],
    )
    missing = sorted(set(sizes) - set(paths))
    if missing:
        raise RuntimeError(f"Requested sizes are not available: {missing}")
    paths = {size: paths[size] for size in sizes}
    pooled = pool_drops(
        paths,
        REGIMES,
        output_dir / "cache",
    )
    summaries = {}
    for size in sizes:
        results = {}
        for regime in REGIMES:
            values = pooled["initial_guess_energy"][regime][size]
            print(
                f"Finding approximate reversibility split: {regime}-yield, "
                f"L={size}, n={values.size}",
                flush=True,
            )
            result = ApproximateReversibilitySplit(
                values,
                bins_per_decade=bins_per_decade,
            )
            results[regime] = result
        _plot_split(size, results, output_dir)
        summaries[str(size)] = {
            regime: {
                "cut": result.cut,
                "otsu_cut": result.otsu_cut,
                "otsu_bin": result.otsu_bin,
                "minimum_bin": result.minimum_bin,
                "local_minimum_bins": result.local_minimum_bins.tolist(),
                "descent_path": result.descent_path.tolist(),
                "sample_count": int(
                    pooled["initial_guess_energy"][regime][size].size
                ),
            }
            for regime, result in results.items()
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "splits.json").write_text(
        json.dumps(
            {
                "data_root": str(data_root),
                "sizes": list(sizes),
                "seeds_per_size": seeds_per_size,
                "bins_per_decade": bins_per_decade,
                "inventory": inventory,
                "protocol": "initial_guess_energy (Delta E_R)",
                "splits": summaries,
            },
            indent=2,
        )
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=POWERLAW_STANDARD_WORKFLOW,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Volumes/data/remoteData/macro"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds-per-size", type=int, default=6)
    parser.add_argument("--bins-per-decade", type=int, default=10)
    args = parser.parse_args()
    run_size_scaling_diagnostic(
        args.data_root,
        args.output_dir,
        seeds_per_size=args.seeds_per_size,
        bins_per_decade=args.bins_per_decade,
    )


if __name__ == "__main__":
    main()
