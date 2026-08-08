"""Make compact size-scaling plots from the completed power-law analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Plotting.sizeScalingCollapse import (
    PROTOCOL_LABELS,
    PROTOCOLS,
    REGIMES,
    completed_size_scaling_paths,
    fixed_xmin_parameter_fits,
    fit_xmins,
    histogram_curves,
    plot_raw_and_xmin,
    plot_xmin_vs_size,
    pool_drops,
)


SIZES = (50, 100, 150, 200, 250)
MARKERS = ("o", "s", "^")
COLORS = ("#0072B2", "#E69F00", "#009E73")


def _save(fig, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def parameter_records(
    data_root: Path,
    cache_root: Path,
    xmin_cache_root: Path,
    accuracy: float,
    parallel_xmin: bool,
    narrow_search: bool,
):
    paths, inventory = completed_size_scaling_paths(data_root, 6, REGIMES["post"][1])
    paths = {size: paths[size] for size in SIZES}
    pooled = pool_drops(
        paths,
        REGIMES,
        cache_root / "extracted",
    )
    selected_drops = {
        protocol: {
            regime: {size: pooled[protocol][regime][size] for size in SIZES}
            for regime in REGIMES
        }
        for protocol in PROTOCOLS
    }
    fits = {protocol: {} for protocol in PROTOCOLS}
    xmins = {
        "simple_drop": {protocol: {} for protocol in PROTOCOLS},
        "global_min": {protocol: {} for protocol in PROTOCOLS},
    }
    for protocol in PROTOCOLS:
        for regime in REGIMES:
            fits[protocol][regime] = fit_xmins(
                selected_drops[protocol][regime],
                parallel=parallel_xmin,
                cache_dir=xmin_cache_root / protocol / regime,
                description=f"{protocol}, {regime}-yield",
                narrow_search=narrow_search,
            )
            xmins["simple_drop"][protocol][regime] = {
                size: float(fits[protocol][regime][size].xmin_analysis["simple_drop_xmin"])
                for size in SIZES
            }
            xmins["global_min"][protocol][regime] = {
                size: float(fits[protocol][regime][size].xmin_analysis["global_min_xmin"])
                for size in SIZES
            }

    records = {method: {protocol: {} for protocol in PROTOCOLS} for method in xmins}
    for method in xmins:
        for protocol in PROTOCOLS:
            for regime in REGIMES:
                records[method][protocol][regime] = fixed_xmin_parameter_fits(
                    selected_drops[protocol][regime],
                    xmins[method][protocol][regime],
                    fit_cache_dir=cache_root / "fixed_xmin" / method / protocol / regime,
                    evaluation_cache_dir=cache_root / "evaluation" / method / protocol / regime,
                    uncertainty_accuracy=accuracy,
                    parallel=False,
                    description=f"{method}, {protocol}, {regime}-yield",
                )

    diagnostics = cache_root.parent / "cutoff_diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    for protocol in PROTOCOLS:
        for regime in REGIMES:
            plot_raw_and_xmin(
                histogram_curves(selected_drops[protocol][regime], bins_per_decade=10),
                fits[protocol][regime],
                protocol,
                regime,
                diagnostics / f"{protocol}_{regime}_xmin_choices.pdf",
            )
    return records, xmins, inventory


def plot_parameter_vs_size(records, parameter: str, output: Path):
    if parameter not in {"alpha", "Lambda"}:
        raise ValueError("parameter must be 'alpha' or 'Lambda'.")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True)
    for ax, regime in zip(axes, REGIMES):
        for marker, color, protocol in zip(MARKERS, COLORS, PROTOCOLS):
            values = records[protocol][regime]
            x = np.asarray(SIZES, dtype=float)
            y = np.asarray([values[size][parameter] for size in SIZES], dtype=float)
            errors = np.asarray(
                [values[size][f"{parameter}_std"] for size in SIZES], dtype=float
            )
            yerr = errors
            if parameter == "Lambda":
                yerr = np.vstack((np.minimum(errors, 0.99 * y), errors))
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=marker,
                color=color,
                capsize=3,
                linewidth=1.1,
                label=PROTOCOL_LABELS[protocol],
            )
        ax.set_xlabel("System size $L$")
        ax.grid(alpha=0.2)
        ax.set_xticks(SIZES)
        ax.set_title("Pre-yield" if regime == "pre" else "Post-yield")
        if parameter == "Lambda":
            ax.set_yscale("log")
    axes[0].set_ylabel(r"Cutoff rate $\lambda$" if parameter == "Lambda" else r"Exponent $\alpha$")
    axes[0].legend(fontsize="small")
    fig.tight_layout()
    _save(fig, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/Volumes/data/remoteData/macro"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Plots/powerLaw/size_collapse/xmin_plateau_accuracy_0.1/parameters_per_size_L50_L250"),
    )
    parser.add_argument(
        "--xmin-cache-root",
        type=Path,
        default=Path("Plots/powerLaw/size_collapse/cache/xmin"),
    )
    parser.add_argument("--uncertainty-accuracy", type=float, default=0.1)
    parser.add_argument("--parallel-xmin", action="store_true")
    parser.add_argument(
        "--narrow-search",
        action="store_true",
        help="Refine only the adjacent coarse-candidate interval around the "
        "steepest coarse KS decrease.",
    )
    args = parser.parse_args()

    records, xmins, inventory = parameter_records(
        args.data_root,
        args.output_dir / "cache",
        args.xmin_cache_root,
        args.uncertainty_accuracy,
        args.parallel_xmin,
        args.narrow_search,
    )
    (args.output_dir / "parameter_results.json").write_text(
        json.dumps(
            {
                "inventory": inventory,
                "sizes": SIZES,
                "protocols": PROTOCOLS,
                "narrow_search": args.narrow_search,
                "xmins": xmins,
                "records": records,
            },
            indent=2,
        )
    )
    for method in records:
        plot_xmin_vs_size(
            xmins,
            method,
            args.output_dir / f"xmin_vs_size_{method}.pdf",
        )
        plot_parameter_vs_size(
            records[method],
            "Lambda",
            args.output_dir / f"lambda_vs_size_{method}.pdf",
        )
        plot_parameter_vs_size(
            records[method],
            "alpha",
            args.output_dir / f"alpha_vs_size_{method}.pdf",
        )
    print(args.output_dir)


if __name__ == "__main__":
    main()
