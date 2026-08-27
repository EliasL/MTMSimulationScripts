#!/usr/bin/env python3
"""Plot standard irreversible Delta E_S PDFs for two size-scaling job groups.

The input files are macroData CSV files only.  Event pairing and post-yield
selection are delegated to ``sizeScalingCollapse.extract_event_pairs``;
classification is then applied to the paired Delta E_R/Delta E_S arrays with
the standard kappa detector before positive Delta E_S filtering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["text.usetex"] = False
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import numpy as np

from Management.updateCSV import read_macrodata_csv
from MTMath.evaluatePowerlawFit import Fit, Truncated_Power_Law
from Plotting.findXmin import analyze_xmin
from Plotting.plotPowerLaw import (
    dist_from_fit,
    fit_equation_label,
    plot_data_pdf,
    plot_fit_pdf,
)
from Plotting.sizeScalingCollapse import extract_event_pairs
from Plotting.sizeScalingCollapse import _read_mixed_selected
from Plotting.standardPowerlaw import (
    EventDrops,
    kappa_detection_threshold,
    positive_es,
    split_by_kappa,
)


DEFAULT_SIZE = 100
POST_YIELD_OFFSET = 0.0
MAX_LOAD = 1.0
MIN_TAIL_COUNT = 100
BOOTSTRAP_CONFIDENCE = 0.05


def _post_yield_window(frame, path: Path) -> tuple[float, float]:
    """Return the post-yield start and usable end load for one run."""
    if "load" not in frame or "avg_sigma12" not in frame:
        raise KeyError(f"Missing load/avg_sigma12 columns in {path}")
    load = np.asarray(frame["load"], dtype=float)
    stress = np.asarray(frame["avg_sigma12"], dtype=float)
    finite_all = np.isfinite(load) & np.isfinite(stress)
    finite = (
        np.isfinite(load)
        & np.isfinite(stress)
        & (load <= MAX_LOAD)
    )
    if not np.any(finite):
        if not np.any(finite_all):
            raise RuntimeError(f"No finite load and avg_sigma12 values in {path}")
        finite = finite_all
        end_load = float(np.max(load[finite]))
    else:
        end_load = MAX_LOAD
    finite_indices = np.flatnonzero(finite)
    start_load = float(load[finite_indices[np.argmax(stress[finite])]])
    return start_load, end_load


def _read_stress_frame(path: Path):
    """Read only the stress columns, with compatibility for mixed exports."""
    try:
        return _read_mixed_selected(path, {"load", "avg_sigma12", "avg_P12"})
    except ValueError as error:
        if not str(error).startswith("Row length mismatch"):
            raise
        frame = read_macrodata_csv(path, update_header=False)
        return frame[["load", "avg_sigma12", "avg_P12"]]


def _preferred_data_path(path: Path) -> Path:
    """Use the newer compatibility-fixed companion when one is available."""
    fixed_path = path.with_name(f"{path.stem}_fixed{path.suffix}")
    if fixed_path.is_file() and fixed_path.stat().st_mtime >= path.stat().st_mtime:
        return fixed_path
    return path


def _collect_irreversible_delta_es(
    csv_paths: list[Path],
    *,
    size: int,
    reference_volume: float,
    cache_dir: Path,
    kappa_det: float,
) -> tuple[np.ndarray, dict]:
    if not csv_paths:
        raise ValueError("At least one macroData.csv path is required.")

    er_parts = []
    es_parts = []
    kappa_parts = []
    run_records = []
    for path in sorted(Path(value) for value in csv_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = _read_stress_frame(path)
        data_path = _preferred_data_path(path)
        post_yield_start, load_end = _post_yield_window(frame, path)
        post_yield_start += POST_YIELD_OFFSET
        if not np.isfinite(post_yield_start) or post_yield_start >= load_end:
            raise RuntimeError(
                f"No post-yield window remains below load {load_end:g} in {path}; "
                f"stress-peak load plus offset is {post_yield_start:.8g}."
            )
        extracted = extract_event_pairs(
            data_path,
            size,
            {"post": (post_yield_start, load_end)},
            cache_dir / "event_pairs",
        )
        er = np.asarray(extracted["initial_guess_energy_post"], dtype=float)
        es = np.asarray(extracted["second_order_post"], dtype=float)
        kappa = np.asarray(extracted["kappa_post"], dtype=float)
        if not (er.shape == es.shape == kappa.shape):
            raise RuntimeError(f"Unaligned paired event arrays in {path}")
        er_parts.append(er)
        es_parts.append(es)
        kappa_parts.append(kappa)
        run_records.append(
            {
                "path": str(path),
                "analysis_path": str(data_path),
                "post_yield_start": post_yield_start,
                "load_end": load_end,
                "post_yield_event_count": int(er.size),
            }
        )

    drops = EventDrops(
        er=np.concatenate(er_parts),
        es=np.concatenate(es_parts),
        kappa=np.concatenate(kappa_parts),
    )
    split = split_by_kappa(drops, kappa_det)
    es_irrev = positive_es(drops, split.is_irrev) / reference_volume
    if es_irrev.size < MIN_TAIL_COUNT:
        raise RuntimeError(
            f"Only {es_irrev.size} positive irreversible Delta E_S values; "
            f"need at least {MIN_TAIL_COUNT}."
        )
    return es_irrev, {
        "csv_count": len(csv_paths),
        "post_yield_event_count": int(drops.er.size),
        "labeled_event_count": int(np.count_nonzero(split.is_rev | split.is_irrev)),
        "reversible_event_count": int(np.count_nonzero(split.is_rev)),
        "irreversible_event_count": int(np.count_nonzero(split.is_irrev)),
        "positive_irreversible_delta_E_S_count": int(es_irrev.size),
        "runs": run_records,
    }


def _fit_population(
    data: np.ndarray, *, cache_dir: Path, description: str
) -> tuple[Fit, dict]:
    """Select the exhaustive global KS cutoff and evaluate its fixed-xmin fit."""
    xmin_analysis = analyze_xmin(
        data,
        nr_initial=100,
        min_tail_count=MIN_TAIL_COUNT,
        refine=True,
        parallel=True,
        progress=False,
        global_mode="global",
    )
    xmin = float(xmin_analysis["global_min_xmin"])
    fit = Fit(
        data=data,
        xmin=xmin,
        xmin_distribution=Truncated_Power_Law.name,
        verbose=0,
    )
    fit.evaluate_fit(
        data=data,
        confidence=BOOTSTRAP_CONFIDENCE,
        parallel=False,
        cache_dir=str(cache_dir),
        tqdmDesc=description,
    )
    distribution = dist_from_fit(fit)
    fit_summary = {
        "xmin": xmin,
        "ks_distance": float(fit.D),
        "alpha": float(distribution.alpha),
        "alpha_std": float(getattr(fit, "alpha_std", np.nan)),
        "lambda": float(distribution.Lambda),
        "lambda_std": float(getattr(fit, "Lambda_std", np.nan)),
        "p": float(getattr(fit, "p", np.nan)),
        "p_std": float(getattr(fit, "p_std", np.nan)),
        "bootstrap_replicates": max(1, int(1 / (4 * BOOTSTRAP_CONFIDENCE**2))),
        "xmin_selection": "exhaustive global KS minimum",
    }
    return fit, fit_summary


def make_plot(
    reconnecting_csv: list[Path],
    nonreconnecting_csv: list[Path],
    output_pdf: Path,
    *,
    size: int = DEFAULT_SIZE,
    cache_dir: Path,
) -> Path:
    if int(size) != size or size <= 0:
        raise ValueError(f"size must be a positive integer; got {size!r}")
    size = int(size)
    reference_volume = float(size**2)
    kappa_det = kappa_detection_threshold()
    populations = {}
    counts = {}
    fits = {}
    fit_objects = {}
    for name, paths in (
        ("reconnecting", reconnecting_csv),
        ("non-reconnecting", nonreconnecting_csv),
    ):
        data, group_counts = _collect_irreversible_delta_es(
            paths,
            size=size,
            reference_volume=reference_volume,
            cache_dir=cache_dir,
            kappa_det=kappa_det,
        )
        populations[name] = data
        counts[name] = group_counts
        fit_objects[name], fits[name] = _fit_population(
            data,
            cache_dir=cache_dir / "comparison_bootstrap_cache" / name,
            description=f"{name} L={size}",
        )

    figure, axis = plt.subplots(figsize=(3.7, 2.7))
    colors = {"reconnecting": "tab:blue", "non-reconnecting": "tab:orange"}
    for name in ("reconnecting", "non-reconnecting"):
        data = populations[name]
        color = colors[name]
        data_line_start = len(axis.lines)
        plot_data_pdf(
            axis,
            data,
            label=f"{name} data (n={data.size})",
            color=color,
            drop_label=r"E_S/V_0",
            drop_sign="positive",
            show_legend=False,
        )
        data_lines = axis.lines[data_line_start:]
        if len(data_lines) != 1:
            raise RuntimeError(
                f"Expected one data-PDF line for {name}; got {len(data_lines)}."
            )
        data_lines[0].set_markerfacecolor("none")
        data_lines[0].set_markeredgecolor(color)
        data_lines[0].set_markeredgewidth(0.7)

    for name in ("non-reconnecting", "reconnecting"):
        color = colors[name]
        fit_line_start = len(axis.lines)
        plot_fit_pdf(
            axis,
            fit_objects[name],
            color=color,
            label="_nolegend_",
            drop_label=r"E_S/V_0",
            drop_sign="positive",
            show_legend=False,
            set_title=False,
            x_grid_mode="smooth",
            xmin_only=True,
            linewidth=1.0,
        )
        fit_lines = axis.lines[fit_line_start:]
        if len(fit_lines) != 1:
            raise RuntimeError(
                f"Expected one fit-PDF line for {name}; got {len(fit_lines)}."
            )
        fit_lines[0].set_zorder(6)

    axis.grid(alpha=0.18)
    equation_label = fit_equation_label(Truncated_Power_Law.name).replace(
        "Fit:", "fit,", 1
    )
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                color=colors["reconnecting"],
                markerfacecolor="none",
                label="reconnecting data",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                color=colors["non-reconnecting"],
                markerfacecolor="none",
                label="non-reconnecting data",
            ),
            Line2D([], [], color="black", linewidth=1.0, label=equation_label),
        ],
        frameon=False,
        fontsize=5.8,
        loc="best",
    )
    fit_info = []
    for name in ("reconnecting", "non-reconnecting"):
        result = fits[name]
        fit_info.extend(
            [
                f"{name} (n={populations[name].size})",
                rf"$x_{{\min}}={result['xmin']:.2g}$, "
                rf"$\alpha={result['alpha']:.3f}\pm{result['alpha_std']:.3f}$",
                rf"$\lambda={result['lambda']:.2g}\pm{result['lambda_std']:.2g}$, "
                rf"$D={result['ks_distance']:.3f}$, "
                rf"$p={result['p']:.3f}\pm{result['p_std']:.2f}$",
                "",
            ]
        )
    axis.text(
        0.03,
        0.04,
        "\n".join(fit_info[:-1]),
        transform=axis.transAxes,
        fontsize=4.8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2},
    )
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "size": size,
        "reference_volume": reference_volume,
        "post_yield_rule": (
            "per-run peak avg_sigma12; non-reconnecting through load 1.0, "
            "reconnecting through its available run end"
        ),
        "post_yield_offset": POST_YIELD_OFFSET,
        "rho": 2.0,
        "kappa_det": float(kappa_det),
        "classification": "paired Delta E_R/Delta E_S events, then positive Delta E_S filtering",
        "populations": counts,
        "fits": fits,
        "model": "p(x) = x^(-alpha) exp(-lambda x)",
    }
    output_pdf.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconnecting-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--non-reconnecting-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    args = parser.parse_args()

    output = make_plot(
        args.reconnecting_csv,
        args.non_reconnecting_csv,
        args.output,
        size=args.size,
        cache_dir=args.cache_dir,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
