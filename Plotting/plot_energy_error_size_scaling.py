"""Plot pooled pre- and post-yield energy-prediction errors versus system size."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path

import numpy as np


DEFAULT_RUNS = {
    50: (2, (0, 1, 2)),
    100: (3, (0, 1, 2)),
    150: (4, (0, 1, 2, 4, 5, 6)),
    200: (8, (5, 6, 7, 8)),
    250: (8, (0, 2, 3)),
}

RECONNECTING_RUNS = {
    50: (2, (0, 3, 4)),
    100: (3, (2, 8, 9)),
}


def _csv_path(
    data_root: Path,
    size: int,
    threads: int,
    seed: int,
    reconnecting: bool = False,
) -> Path:
    boundary_tag = "PBCedgeFlip" if reconnecting else "PBC"
    name = (
        f"simpleShear,s{size}x{size}l0.15,1e-05,1.0{boundary_tag}t{threads}"
        f"LBFGSEpsx1e-06s{seed}"
    )
    return data_root / name / "macroData.csv"


def _sample_statistics(item):
    size, seed, path = item
    from Plotting.plot_reconnection_energy_error_distribution import _collect_values

    values = _collect_values([path], event_only=False)["second"]
    result = {"size": size, "seed": seed}
    for region in ("pre", "post"):
        sample = values[region]
        if not sample.size:
            raise ValueError(f"No {region}-yield values found in {path}.")
        result[f"{region}_sum"] = float(np.sum(sample))
        result[f"{region}_count"] = int(sample.size)
        result[f"{region}_mean"] = float(np.mean(sample))
    return result


def collect_statistics(
    data_root: Path,
    workers: int,
    runs=DEFAULT_RUNS,
    reconnecting: bool = False,
):
    items = []
    for size, (threads, seeds) in runs.items():
        for seed in seeds:
            path = _csv_path(
                data_root,
                size,
                threads,
                seed,
                reconnecting=reconnecting,
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing size-scaling CSV: {path}")
            items.append((size, seed, path))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        sample_rows = list(executor.map(_sample_statistics, items))

    summaries = []
    for size in sorted(runs):
        size_rows = [row for row in sample_rows if row["size"] == size]
        summary = {"size": size, "runs": len(size_rows)}
        for region in ("pre", "post"):
            total = sum(row[f"{region}_sum"] for row in size_rows)
            count = sum(row[f"{region}_count"] for row in size_rows)
            run_means = np.array(
                [row[f"{region}_mean"] for row in size_rows], dtype=float
            )
            summary[f"{region}_mean"] = total / count
            summary[f"{region}_count"] = count
            summary[f"{region}_run_std"] = float(np.std(run_means, ddof=1))
        summaries.append(summary)
    return summaries, sample_rows


def save_summary(path: Path, summaries):
    fieldnames = [
        "size",
        "runs",
        "pre_mean",
        "post_mean",
        "pre_count",
        "post_count",
        "pre_run_std",
        "post_run_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def make_plot(summaries, sample_rows, output_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    sizes = np.array([row["size"] for row in summaries], dtype=float)
    pre = np.array([row["pre_mean"] for row in summaries], dtype=float)
    post = np.array([row["post_mean"] for row in summaries], dtype=float)

    fig, ax = plt.subplots(figsize=(4.329, 2.808))
    ax.plot(
        sizes,
        pre,
        color="#9ecae1",
        marker="o",
        markersize=4.2,
        linewidth=1.2,
        label="Pre-yield mean",
        zorder=3,
    )
    ax.plot(
        sizes,
        post,
        color="#2171b5",
        marker="s",
        markersize=4.0,
        linewidth=1.2,
        label="Post-yield mean",
        zorder=3,
    )

    for size in sizes:
        rows = [row for row in sample_rows if row["size"] == size]
        ax.scatter(
            np.full(len(rows), size),
            [row["pre_mean"] for row in rows],
            color="#9ecae1",
            marker="o",
            s=10,
            alpha=0.45,
            linewidths=0,
            zorder=2,
        )
        ax.scatter(
            np.full(len(rows), size),
            [row["post_mean"] for row in rows],
            color="#2171b5",
            marker="s",
            s=10,
            alpha=0.45,
            linewidths=0,
            zorder=2,
        )

    ax.set_yscale("log")
    ax.set_xticks(sizes.astype(int))
    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel(r"Mean $|\Delta E_S|/V_0$")
    ax.legend(loc="best", fontsize=8.0, handlelength=1.56, borderpad=0.26)
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reconnecting", action="store_true")
    args = parser.parse_args()

    if args.reconnecting:
        runs = RECONNECTING_RUNS
        output = args.output or Path(
            "Plots/energy_error_cauchy_a_reconnecting_available_sizes_"
            "pre_post_mean.pdf"
        )
        summary_path = args.summary or Path(
            "Plots/energy_error_cauchy_a_reconnecting_available_sizes_"
            "pre_post_mean.csv"
        )
    else:
        runs = DEFAULT_RUNS
        output = args.output or Path(
            "Plots/energy_error_cauchy_a_size_scaling_pre_post_mean.pdf"
        )
        summary_path = args.summary or Path(
            "Plots/energy_error_cauchy_a_size_scaling_pre_post_mean.csv"
        )

    summaries, sample_rows = collect_statistics(
        args.data_root,
        args.workers,
        runs=runs,
        reconnecting=args.reconnecting,
    )
    save_summary(summary_path, summaries)
    make_plot(summaries, sample_rows, output)
    for row in summaries:
        print(
            f"L={row['size']}: pre={row['pre_mean']:.6e}, "
            f"post={row['post_mean']:.6e}, runs={row['runs']}"
        )
    print(f'Plot saved at: "{output}"')
    print(f'Summary saved at: "{summary_path}"')


if __name__ == "__main__":
    main()
