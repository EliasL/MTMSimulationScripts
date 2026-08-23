"""Plot post-yield size scaling from a sigma-rescue diagnostic summary.

The summary supplies the rapidGlobal xmin, KS distance, and p-value.
The observed tail is refit at that fixed xmin to obtain ``alpha`` and
``Lambda``.  The three drop measures are kept paired through the kappa split;
the standard ``Delta E_S`` population is the irreversible tail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Plotting.plotPowerLaw import dist_from_fit, make_fit
from Plotting.sigmaRescueSizeScalingPowerlawDiagnostics import _plot_fit_panel
from Plotting.standardPowerlaw import (
    EventDrops,
    kappa_detection_threshold,
    positive_es,
    split_by_kappa,
)


DEFAULT_SNAPSHOT = Path("sigma_rescue_interim/snapshots/20260819T100206Z")
DEFAULT_SUMMARY = Path(
    "Plots/powerLaw/sigma_rescue_size_scaling_individual/summary.json"
)
DEFAULT_OUTPUT_DIR = Path("Plots/powerLaw/sigma_rescue_size_scaling_rapidGlobal")
PROTOCOLS = ("delta_E_I", "delta_E_R", "delta_E_S")
PROTOCOL_LABELS = {
    "delta_E_I": r"$\Delta E_I$ (inter-strain)",
    "delta_E_R": r"$\Delta E_R$ (relaxation)",
    "delta_E_S": r"$\Delta E_S$ (stress corrected)",
}
COLORS = {"delta_E_I": "#0072B2", "delta_E_R": "#E69F00", "delta_E_S": "#009E73"}
MARKERS = {"delta_E_I": "o", "delta_E_R": "s", "delta_E_S": "^"}
COUNT_COLOR = "#7B2CBF"
FRACTION_COLOR = "#D1495B"
EXPECTED_SEEDS_PER_SIZE = 10


def _positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def _load_post_yield(snapshot_root: Path) -> pd.DataFrame:
    table = Path(snapshot_root) / "tables" / "drops_usable.csv.gz"
    if not table.is_file():
        raise FileNotFoundError(f"Missing usable drop table: {table}")
    wanted = [
        "size",
        "load_ip1",
        "delta_gamma",
        "reference_volume",
        "delta_E_I",
        "delta_E_R",
        "delta_E_S",
    ]
    frame = pd.read_csv(table, usecols=wanted)
    frame = frame[(frame["load_ip1"] > 0.7) & (frame["load_ip1"] < 1.0)].copy()
    if frame.empty:
        raise ValueError(f"No post-yield rows found in {table}")
    frame["kappa"] = frame["delta_E_R"] / (
        frame["reference_volume"] * frame["delta_gamma"] ** 2
    )
    return frame


def _load_post_yield_rescue_fractions(snapshot_root: Path) -> dict[int, float]:
    """Return valid fractions relative to the full ten-seed campaign."""
    table = Path(snapshot_root) / "tables" / "drops_all_audited.csv.gz"
    if not table.is_file():
        raise FileNotFoundError(f"Missing audited drop table: {table}")
    audited = pd.read_csv(table, usecols=["size", "seed", "load_ip1", "usable"])
    audited = audited[
        (audited["load_ip1"] > 0.7) & (audited["load_ip1"] < 1.0)
    ]
    per_seed = audited.groupby(["size", "seed"])["usable"].agg(["size", "sum"])
    fractions: dict[int, float] = {}
    for size, rows in per_seed.groupby(level="size"):
        total_counts = rows["size"].to_numpy(dtype=int)
        count_range = int(total_counts.max() - total_counts.min())
        if count_range > 1:
            raise ValueError(
                f"L={size} has inconsistent post-yield transition counts: "
                f"{sorted(set(total_counts.tolist()))}."
            )
        # A one-row endpoint difference is possible with the strict
        # ``load_ip1 < 1.0`` selection.  Use the larger complete-seed count.
        expected_total = int(total_counts.max()) * EXPECTED_SEEDS_PER_SIZE
        valid_count = int(rows["sum"].sum())
        fractions[int(size)] = valid_count / expected_total
    return fractions


def _populations(frame: pd.DataFrame, size: int) -> dict[str, np.ndarray]:
    selected = frame[frame["size"] == size]
    paired = EventDrops(
        er=selected["delta_E_R"].to_numpy(dtype=float),
        es=selected["delta_E_S"].to_numpy(dtype=float),
        kappa=selected["kappa"].to_numpy(dtype=float),
    )
    split = split_by_kappa(paired, kappa_detection_threshold())
    populations = {
        "delta_E_I": _positive(
            selected.loc[split.is_irrev, "delta_E_I"].to_numpy(dtype=float)
        ),
        "delta_E_R": _positive(paired.er[split.is_irrev]),
        "delta_E_S": positive_es(paired, split.is_irrev),
    }
    for protocol, values in populations.items():
        if values.size < 3:
            raise ValueError(f"L={size}, {protocol} has too few positive values.")
    return populations


def _paired_irreversible_event_count(frame: pd.DataFrame, size: int) -> int:
    """Count common positive irreversible events used by all three measures."""
    selected = frame[frame["size"] == size]
    paired = EventDrops(
        er=selected["delta_E_R"].to_numpy(dtype=float),
        es=selected["delta_E_S"].to_numpy(dtype=float),
        kappa=selected["kappa"].to_numpy(dtype=float),
    )
    split = split_by_kappa(paired, kappa_detection_threshold())
    values = np.column_stack(
        [
            selected["delta_E_I"].to_numpy(dtype=float),
            paired.er,
            paired.es,
        ]
    )
    common_positive_irreversible = split.is_irrev & np.all(
        np.isfinite(values) & (values > 0), axis=1
    )
    return int(np.count_nonzero(common_positive_irreversible))


def _records(frame: pd.DataFrame, summary: dict, cache_dir: Path):
    records = []
    fits_by_size = {}
    populations_by_size = {}
    for size_text in sorted(summary["fits"], key=int):
        size = int(size_text)
        populations = _populations(frame, size)
        populations_by_size[size] = populations
        fits_by_size[size] = {}
        for protocol in PROTOCOLS:
            summary_protocol = summary["fits"][size_text]["protocols"][protocol]
            xmin = float(summary_protocol["global_min_xmin"])
            data = populations[protocol]
            tail = data[data >= xmin]
            if tail.size < 3:
                raise ValueError(
                    f"L={size}, {protocol} has only {tail.size} values at xmin={xmin}."
                )
            # Keep the full population on the Fit object.  plot_fit_pdf uses
            # its below-xmin fraction to place the normalized tail model on
            # the full empirical PDF scale.
            fit = make_fit(
                data,
                xmin_range=xmin,
                cache_dir=str(cache_dir / protocol / f"L{size}"),
            )
            fit.xmin_fitting_results = {"global_min_xmin": xmin}
            fits_by_size[size][protocol] = fit
            distribution = dist_from_fit(fit)
            alpha = float(distribution.alpha)
            cutoff_rate = float(getattr(distribution, "Lambda", np.nan))
            if not np.isfinite(alpha) or not np.isfinite(cutoff_rate) or cutoff_rate <= 0:
                raise RuntimeError(f"Invalid fixed-xmin parameters for L={size}, {protocol}.")
            records.append(
                {
                    "size": size,
                    "protocol": protocol,
                    "xmin": xmin,
                    "alpha": alpha,
                    "Lambda": cutoff_rate,
                    "D": float(summary_protocol["global_min_D"]),
                    "p": float(summary_protocol["clauset_pvalue_v2"]),
                    "data_count": int(data.size),
                    "tail_count": int(tail.size),
                    "tail_fraction": float(tail.size / data.size),
                }
            )
    return records, fits_by_size, populations_by_size


def _plot_per_size_pdfs(
    fits_by_size: dict[int, dict[str, object]],
    populations_by_size: dict[int, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    """Reuse the established per-protocol PDF panel for each system size."""
    from matplotlib.lines import Line2D

    output_dir.mkdir(parents=True, exist_ok=True)
    for size in sorted(fits_by_size):
        fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(13.5, 4.1), squeeze=False)
        for axis, protocol in zip(axes[0], PROTOCOLS):
            _plot_fit_panel(
                axis,
                fits_by_size[size][protocol],
                populations_by_size[size][protocol],
                protocol,
            )
        handles = [
            Line2D([], [], color="black", marker="o", linestyle="None", label="irreversible events"),
            Line2D([], [], color="tab:red", label="truncated power-law fit"),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.945),
            fontsize="small",
        )
        fig.suptitle(
            rf"Post-yield PDFs, $L={size}$; irreversible population from "
            rf"$\kappa_{{\det}}=\mu/(2\rho)$ split; rapidGlobal xmin",
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        path = output_dir / f"L{size}_post_yield_pdf_fits.pdf"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
        plt.close(fig)


def _plot(
    records: list[dict],
    paired_event_counts: dict[int, int],
    rescue_fractions: dict[int, float],
    output: Path,
) -> None:
    """Plot scaling parameters with a temporary event-coverage panel."""
    metrics = (
        ("alpha", r"$\alpha$", False),
        ("Lambda", r"$\lambda$", True),
        ("event_count", "events used in fit", False),
        ("xmin", r"$\Delta E_{\min}$", True),
    )
    sizes = sorted({row["size"] for row in records})
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), squeeze=False)
    for ax, (metric, ylabel, log_y) in zip(axes.flat, metrics):
        if metric == "event_count":
            fraction_ax = ax.twinx()
            x = np.asarray(sizes, dtype=float)
            counts = np.asarray(
                [paired_event_counts[size] for size in sizes], dtype=float
            )
            fractions = np.asarray(
                [rescue_fractions[size] for size in sizes], dtype=float
            )
            if np.any(~np.isfinite(fractions)) or np.any(
                (fractions < 0) | (fractions > 1)
            ):
                raise ValueError("Rescue fractions must lie in [0, 1].")
            ax.plot(
                x,
                counts,
                color=COUNT_COLOR,
                marker="o",
                linewidth=1.2,
                label="paired irreversible events",
            )
            fraction_ax.plot(
                x,
                fractions,
                color=FRACTION_COLOR,
                linestyle="--",
                marker="D",
                linewidth=1.2,
                label=r"fraction of expected data with valid $\sigma_{12}$",
            )
            ax.set_xticks(sizes)
            ax.set_xlabel("System size $L$")
            ax.set_ylabel("paired irreversible events", color=COUNT_COLOR)
            ax.tick_params(axis="y", colors=COUNT_COLOR)
            ax.spines["left"].set_color(COUNT_COLOR)
            fraction_ax.set_ylabel(
                r"fraction of expected data with valid $\sigma_{12}$",
                color=FRACTION_COLOR,
            )
            fraction_ax.tick_params(axis="y", colors=FRACTION_COLOR)
            fraction_ax.spines["right"].set_color(FRACTION_COLOR)
            fraction_ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.2)
            fraction_ax.grid(False)
            ax.legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        color=COUNT_COLOR,
                        marker="o",
                        label="paired irreversible events",
                    ),
                    Line2D(
                        [],
                        [],
                        color=FRACTION_COLOR,
                        linestyle="--",
                        marker="D",
                        label=r"fraction of expected data with valid $\sigma_{12}$",
                    ),
                ],
                loc="upper left",
                fontsize="small",
                frameon=True,
            )
            continue
        for protocol in PROTOCOLS:
            rows = [
                row
                for row in records
                if row["protocol"] == protocol
            ]
            rows.sort(key=lambda row: row["size"])
            x = np.asarray([row["size"] for row in rows], dtype=float)
            y = np.asarray([row[metric] for row in rows], dtype=float)
            ax.plot(
                x,
                y,
                marker=MARKERS[protocol],
                color=COLORS[protocol],
                linewidth=1.1,
                label=PROTOCOL_LABELS[protocol],
            )
        if log_y:
            ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xlabel("System size $L$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    handles = [
        Line2D(
            [],
            [],
            color=COLORS[protocol],
            marker=MARKERS[protocol],
            label=PROTOCOL_LABELS[protocol],
        )
        for protocol in PROTOCOLS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        fontsize="small",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(
    snapshot_root: Path = DEFAULT_SNAPSHOT,
    summary_path: Path = DEFAULT_SUMMARY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    summary = json.loads(Path(summary_path).read_text())
    if summary.get("bootstrap_xmin_mode") != "rapidGlobal":
        raise ValueError(
            "This plotting wrapper requires a rapidGlobal summary; "
            f"got {summary.get('bootstrap_xmin_mode')!r}."
        )
    frame = _load_post_yield(Path(snapshot_root))
    records, fits_by_size, populations_by_size = _records(
        frame, summary, output_dir / "cache" / "fixed_xmin"
    )
    _plot_per_size_pdfs(
        fits_by_size,
        populations_by_size,
        output_dir / "per_size_pdf_fits",
    )
    output = output_dir / "post_yield_size_scaling_parameters.pdf"
    paired_event_counts = {
        int(size): _paired_irreversible_event_count(frame, int(size))
        for size in sorted(frame["size"].unique())
    }
    rescue_fractions = _load_post_yield_rescue_fractions(Path(snapshot_root))
    _plot(records, paired_event_counts, rescue_fractions, output)
    (output_dir / "size_scaling_parameters.json").write_text(
        json.dumps(
            {
                "snapshot_root": str(Path(snapshot_root).resolve()),
                "summary_path": str(Path(summary_path).resolve()),
                "records": records,
                "paired_event_counts": paired_event_counts,
                "rescue_fractions": rescue_fractions,
            },
            indent=2,
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(run(args.snapshot_root, args.summary, args.output_dir))


if __name__ == "__main__":
    main()
