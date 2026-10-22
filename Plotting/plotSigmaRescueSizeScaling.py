"""Plot post-yield size scaling from a sigma-rescue diagnostic summary.

The summary supplies the exhaustive-global xmin, KS distance, and p-value.
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
    SIGMA_RESCUE_POST_YIELD_WINDOW,
    DEFAULT_KAPPA_RHO,
    EventDrops,
    kappa_detection_threshold,
    kappa_from_relaxation_energy,
    positive_es,
    load_sigma_rescue_post_yield,
    require_gamma0_snapshot,
    split_by_kappa,
)


DEFAULT_SNAPSHOT = Path("sigma_rescue_interim/current_gamma0_piola_peak")
DEFAULT_SUMMARY = Path(
    "Plots/powerLaw/sigma_rescue_size_scaling_individual_piola_peak/summary.json"
)
DEFAULT_OUTPUT_DIR = Path("Plots/powerLaw/sigma_rescue_size_scaling_piola_peak")
PROTOCOLS = ("delta_E_I", "delta_E_R", "delta_E_S")
PROTOCOL_LABELS = {
    "delta_E_I": r"$\Delta E_I$",
    "delta_E_R": r"$\Delta E_R$",
    "delta_E_S": r"$\Delta E_S$",
}
COLORS = {"delta_E_I": "#0072B2", "delta_E_R": "#E69F00", "delta_E_S": "#009E73"}
MARKERS = {"delta_E_I": "o", "delta_E_R": "s", "delta_E_S": "^"}
COUNT_COLOR = "#7B2CBF"
FRACTION_COLOR = "#D1495B"


def _positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def _load_post_yield(snapshot_root: Path) -> pd.DataFrame:
    require_gamma0_snapshot(snapshot_root)
    table = Path(snapshot_root) / "tables" / "drops_usable.csv.gz"
    if not table.is_file():
        raise FileNotFoundError(f"Missing usable drop table: {table}")
    wanted = [
        "size",
        "seed",
        "load_ip1",
        "delta_gamma",
        "reference_volume",
        "delta_E_I",
        "delta_E_R",
        "delta_E_S",
    ]
    frame = load_sigma_rescue_post_yield(snapshot_root, wanted)
    if frame.empty:
        raise ValueError(f"No post-yield rows found in {table}")
    frame["kappa"] = kappa_from_relaxation_energy(
        frame["delta_E_R"].to_numpy(dtype=float),
        frame["delta_gamma"].to_numpy(dtype=float),
        frame["reference_volume"].to_numpy(dtype=float),
        rho=DEFAULT_KAPPA_RHO,
    )
    return frame


def _load_post_yield_rescue_fractions(snapshot_root: Path) -> dict[int, float]:
    """Return validated fractions among audited post-yield transitions."""
    table = Path(snapshot_root) / "tables" / "drops_all_audited.csv.gz"
    if not table.is_file():
        raise FileNotFoundError(f"Missing audited drop table: {table}")
    audited = load_sigma_rescue_post_yield(snapshot_root, ["size", "usable"], audited=True)
    return {int(size): float(group["usable"].mean()) for size, group in audited.groupby("size")}


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
            xmin_interval = np.asarray(summary_protocol["bootstrap_xmin_interval_95"], dtype=float)
            alpha_interval = np.asarray(summary_protocol["bootstrap_alpha_interval_95"], dtype=float)
            lambda_interval = np.asarray(summary_protocol["bootstrap_lambda_interval_95"], dtype=float)
            for name, interval in (("xmin", xmin_interval), ("alpha", alpha_interval), ("Lambda", lambda_interval)):
                if interval.shape != (2,) or not np.all(np.isfinite(interval)) or not 0 < interval[0] < interval[1]:
                    raise ValueError(f"Invalid 95% bootstrap {name} interval for L={size}, {protocol}.")
            records.append(
                {
                    "size": size,
                    "protocol": protocol,
                    "xmin": xmin,
                    "alpha": alpha,
                    "Lambda": cutoff_rate,
                    "D": float(summary_protocol["global_min_D"]),
                    "p": (float(summary_protocol["clauset_pvalue_v2"])
                          if summary_protocol["clauset_pvalue_v2"] is not None else None),
                    "xmin_interval_95": xmin_interval.tolist(),
                    "alpha_interval_95": alpha_interval.tolist(),
                    "Lambda_interval_95": lambda_interval.tolist(),
                    "data_count": int(data.size),
                    "tail_count": int(tail.size),
                    "tail_fraction": float(tail.size / data.size),
                }
            )
    return records, fits_by_size, populations_by_size


def _plot_per_size_pdfs(
    fits_by_size: dict[int, dict[str, object]],
    populations_by_size: dict[int, dict[str, np.ndarray]],
    fit_summary: dict,
    output_dir: Path,
) -> None:
    """Reuse the established per-protocol PDF panel for each system size."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for size in sorted(fits_by_size):
        for protocol in PROTOCOLS:
            fig, axis = plt.subplots(figsize=(5.5, 4.2))
            protocol_summary = fit_summary["fits"][str(size)]["protocols"][protocol]
            _plot_fit_panel(
                axis,
                fits_by_size[size][protocol],
                populations_by_size[size][protocol],
                protocol,
                bootstrap_intervals=(
                    tuple(protocol_summary["bootstrap_xmin_interval_95"]),
                    tuple(protocol_summary["bootstrap_alpha_interval_95"]),
                    tuple(protocol_summary["bootstrap_lambda_interval_95"]),
                ),
            )
            fig.tight_layout()
            path = output_dir / f"L{size}_post_yield_{protocol}_fit.pdf"
            fig.savefig(path, bbox_inches="tight")
            fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
            plt.close(fig)


def _plot(
    records: list[dict],
    paired_event_counts: dict[int, int],
    rescue_fractions: dict[int, float],
    seed_counts: dict[int, int],
    output: Path,
) -> None:
    """Plot scaling parameters and validated post-yield event coverage."""
    metrics = (
        ("alpha", r"$\hat\alpha$", False),
        ("Lambda", r"$\hat\lambda$", True),
        ("event_count", "events used in fit", False),
        ("xmin", r"$\Delta E_{S,\min}^{KS}$ (green)", True),
    )
    sizes = sorted({row["size"] for row in records})
    output.parent.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, log_y in metrics:
        fig, ax = plt.subplots(figsize=(5.76, 4.32))
        stem = {"Lambda": "lambda", "event_count": "event_coverage"}.get(metric, metric)
        path = output.parent / f"{stem}.pdf"
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
            ax.margins(y=0.16)
            for size, count in zip(sizes, counts):
                ax.annotate(f"{seed_counts[size]} seeds", (size, count),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", fontsize="x-small")
            fraction_ax.plot(
                x,
                fractions,
                color=FRACTION_COLOR,
                linestyle="--",
                marker="D",
                linewidth=1.2,
                label=r"fraction of audited post-yield data with valid $\sigma_{12}$",
            )
            ax.set_xticks(sizes)
            ax.set_xlabel("System size $L$")
            ax.set_ylabel("paired irreversible events", color=COUNT_COLOR)
            ax.tick_params(axis="y", colors=COUNT_COLOR)
            ax.spines["left"].set_color(COUNT_COLOR)
            fraction_ax.set_ylabel(
                r"fraction of audited post-yield data with valid $\sigma_{12}$",
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
                        label=r"fraction of audited post-yield data with valid $\sigma_{12}$",
                    ),
                ],
                loc="upper left",
                fontsize="small",
                frameon=True,
            )
            fig.tight_layout()
            fig.savefig(path, bbox_inches="tight")
            fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
            plt.close(fig)
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
            if (protocol == "delta_E_S" and metric in {"alpha", "xmin"}
                    and rows[0]["alpha_ci_95" if metric == "alpha" else "xmin_ci_95"] is not None):
                interval_key = "alpha_ci_95" if metric == "alpha" else "xmin_ci_95"
                intervals = np.asarray([row[interval_key] for row in rows], dtype=float)
                ax.vlines(x, intervals[:, 0], intervals[:, 1],
                          color=COLORS[protocol], alpha=0.55, linewidth=1.5)
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
    es_rows = sorted((row for row in records if row["protocol"] == "delta_E_S"),
                     key=lambda row: row["size"])
    if all(row["p"] is not None for row in es_rows):
        note = r"Green bars: 95% refit intervals for $\Delta E_S$"
        pvalues = "; ".join(f"L={row['size']}: {row['p']:.3f}" for row in es_rows)
        fig.text(0.5, 0.015, r"Selected $\Delta E_S$ fit $p$: " + pvalues,
                 ha="center", fontsize="small")
    else:
        note = "Bootstrap omitted: no p-values or uncertainty intervals"
    fig.text(0.5, 0.925, note, ha="center", fontsize="small")
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
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
    require_gamma0_snapshot(snapshot_root)
    if summary.get("delta_E_S_tangent") != "gamma0":
        raise ValueError(f"Fit summary uses an old Delta E_S convention: {summary_path}")
    if summary.get("post_yield_window") != SIGMA_RESCUE_POST_YIELD_WINDOW:
        raise ValueError(f"Fit summary uses an obsolete yield selection: {summary_path}")
    if Path(summary["snapshot_root"]).resolve() != Path(snapshot_root).resolve():
        raise ValueError("Fit summary and requested snapshot refer to different data.")
    if summary.get("bootstrap_xmin_mode") not in {"global", "skipped"}:
        raise ValueError(
            "This plotting wrapper requires a full-global or explicitly skipped bootstrap; "
            f"got {summary.get('bootstrap_xmin_mode')!r}."
        )
    if summary.get("xmin_selection") != "global minimum after exhaustive observed-candidate scan":
        raise ValueError("This plotting wrapper requires exhaustive empirical xmin selection.")
    frame = _load_post_yield(Path(snapshot_root))
    records, fits_by_size, populations_by_size = _records(
        frame, summary, output_dir / "cache" / "fixed_xmin"
    )
    _plot_per_size_pdfs(
        fits_by_size,
        populations_by_size,
        summary,
        output_dir / "per_size_pdf_fits",
    )
    output = output_dir / "post_yield_size_scaling_parameters.pdf"
    paired_event_counts = {
        int(size): _paired_irreversible_event_count(frame, int(size))
        for size in sorted(frame["size"].unique())
    }
    rescue_fractions = _load_post_yield_rescue_fractions(Path(snapshot_root))
    seed_counts = {
        int(size): int(group["seed"].nunique())
        for size, group in frame.groupby("size")
    }
    _plot(records, paired_event_counts, rescue_fractions, seed_counts, output)
    (output_dir / "size_scaling_parameters.json").write_text(
        json.dumps(
            {
                "snapshot_root": str(Path(snapshot_root).resolve()),
                "summary_path": str(Path(summary_path).resolve()),
                "records": records,
                "paired_event_counts": paired_event_counts,
                "rescue_fractions": rescue_fractions,
                "post_yield_seed_counts": seed_counts,
            },
            indent=2,
        )
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(run(args.snapshot_root, args.summary, args.output_dir))


if __name__ == "__main__":
    main()
