"""Overlay energy-prediction-error distributions for every available size."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from Plotting.plot_energy_error_size_scaling import (
    DEFAULT_RUNS,
    RECONNECTING_RUNS,
    _csv_path,
)


def _collect_size(item):
    size, paths = item
    from Plotting.plot_reconnection_energy_error_distribution import _collect_values

    return size, _collect_values(paths, event_only=False)["second"]


def collect_distributions(
    data_root: Path,
    workers: int,
    runs=DEFAULT_RUNS,
    reconnecting: bool = False,
):
    items = []
    for size, (threads, seeds) in runs.items():
        paths = [
            _csv_path(
                data_root,
                size,
                threads,
                seed,
                reconnecting=reconnecting,
            )
            for seed in seeds
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing size-scaling CSV files: "
                + ", ".join(str(path) for path in missing)
            )
        items.append((size, paths))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(_collect_size, items))


def _format_count(count: int):
    exponent = int(np.floor(np.log10(count)))
    mantissa = count / 10.0**exponent
    return rf"$n={mantissa:.1f}\times 10^{{{exponent}}}$"


def make_plot(distributions, region: str, edges, output_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    sizes = sorted(distributions)
    colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")

    fig, ax = plt.subplots(figsize=(4.329, 2.808))
    repeated_edges = np.repeat(edges, 2)[1:-1]
    for size, color in zip(sizes, colors):
        sample = distributions[size][region]
        histogram, _ = np.histogram(sample, bins=edges)
        probability = histogram.astype(float) / sample.size
        ax.plot(
            repeated_edges,
            np.repeat(probability, 2),
            color=color,
            linewidth=1.2,
            label=rf"$L={size}$ ({_format_count(sample.size)})",
        )

    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"$|\Delta E_S|/V_0$")
    ax.set_ylabel("Probability per logarithmic bin")
    ax.legend(
        title="Pre-yield" if region == "pre" else "Post-yield",
        loc="upper left",
        fontsize=7.5,
        title_fontsize=8.0,
        handlelength=1.45,
        borderpad=0.26,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Plots/energy_prediction_normal_data"),
    )
    parser.add_argument("--pre-output", type=Path)
    parser.add_argument("--post-output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reconnecting", action="store_true")
    args = parser.parse_args()

    if args.reconnecting:
        runs = RECONNECTING_RUNS
        pre_output = args.pre_output or Path(
            "Plots/energy_error_cauchy_a_reconnecting_available_sizes_"
            "pre_yield_V0norm_120bins.pdf"
        )
        post_output = args.post_output or Path(
            "Plots/energy_error_cauchy_a_reconnecting_available_sizes_"
            "post_yield_V0norm_120bins.pdf"
        )
    else:
        runs = DEFAULT_RUNS
        pre_output = args.pre_output or Path(
            "Plots/energy_error_cauchy_a_all_sizes_no_recon_"
            "pre_yield_V0norm_120bins.pdf"
        )
        post_output = args.post_output or Path(
            "Plots/energy_error_cauchy_a_all_sizes_no_recon_"
            "post_yield_V0norm_120bins.pdf"
        )

    distributions = collect_distributions(
        args.data_root,
        args.workers,
        runs=runs,
        reconnecting=args.reconnecting,
    )
    all_values = np.concatenate(
        [
            distributions[size][region]
            for size in sorted(distributions)
            for region in ("pre", "post")
        ]
    )
    edges = np.geomspace(float(np.min(all_values)), float(np.max(all_values)), 121)

    make_plot(distributions, "pre", edges, pre_output)
    make_plot(distributions, "post", edges, post_output)
    print(f'Pre-yield plot saved at: "{pre_output}"')
    print(f'Post-yield plot saved at: "{post_output}"')


if __name__ == "__main__":
    main()
