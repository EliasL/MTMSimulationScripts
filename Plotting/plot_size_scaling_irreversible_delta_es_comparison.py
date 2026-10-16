#!/usr/bin/env python3
"""Plot standard irreversible Delta E_S PDFs for two size-scaling job groups.

The input files are macroData CSV files only.  Event pairing and post-yield
selection are delegated to ``sizeScalingCollapse.extract_event_pairs``;
classification is then applied to the paired Delta E_R/Delta E_S arrays with
the standard kappa detector before positive Delta E_S filtering.

This is a two-population overlay, not the default individual-size figure.
For a single run group's established split and fit panels, use
``truncated_powerlaw_reversibility_flowchart.generate_single_size_panels``.
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
from Plotting.plotPowerLaw import (
    dist_from_fit,
    plot_data_pdf,
    plot_fit_pdf,
)
from Plotting.sizeScalingCollapse import extract_event_pairs
from Plotting.sizeScalingCollapse import _read_mixed_selected
from Plotting.standardPowerlaw import (
    POST_YIELD_RULE, yield_regimes,
    EventDrops,
    kappa_detection_threshold,
    positive_es,
    split_by_kappa,
    load_sigma_rescue_post_yield,
    kappa_from_relaxation_energy,
)
from Plotting.truncated_powerlaw_reversibility_flowchart import (
    _configure_matplotlib, _exhaustive_xmin_analysis,
)


DEFAULT_SIZE = 100
MAX_LOAD = 1.0
MIN_TAIL_COUNT = 100
BOOTSTRAP_CONFIDENCE = 0.05


def _post_yield_window(frame, path: Path) -> tuple[float, float]:
    """Return the post-yield start and usable end load for one run."""
    if "load" not in frame:
        raise KeyError(f"Missing load column in {path}")
    load = np.asarray(frame["load"], dtype=float)
    finite_all = np.isfinite(load)
    if not np.any(finite_all):
        raise RuntimeError(f"No finite load values in {path}")
    end_load = min(MAX_LOAD, float(np.max(load[finite_all])))
    return yield_regimes(frame)["post"][0], end_load


def _read_load_frame(path: Path):
    """Read loads, with compatibility for mixed exports."""
    try:
        return _read_mixed_selected(path, {"load", "avg_P12"})
    except ValueError as error:
        if not str(error).startswith("Row length mismatch"):
            raise
        frame = read_macrodata_csv(path, update_header=False)
        return frame[["load", "avg_P12"]]


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
    cache_dir: Path,
    kappa_det: float,
    snapshot_root: Path | None = None,
) -> tuple[np.ndarray, dict]:
    if not csv_paths:
        raise ValueError("At least one macroData.csv path is required.")

    er_parts = []
    es_parts = []
    kappa_parts = []
    run_records = []
    rescued = None
    if snapshot_root is not None:
        rescued = load_sigma_rescue_post_yield(
            snapshot_root, ["delta_E_R", "delta_E_S", "delta_gamma", "reference_volume"],
            audited=True,
        )
        rescued = rescued.loc[rescued["size"] == size]
    for path in sorted(Path(value) for value in csv_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = _read_load_frame(path)
        data_path = _preferred_data_path(path)
        post_yield_start, load_end = _post_yield_window(frame, path)
        if not np.isfinite(post_yield_start) or post_yield_start >= load_end:
            raise RuntimeError(
                f"No post-yield window remains below load {load_end:g} in {path}; "
                f"avg_P12 peak +0.01 is {post_yield_start:.8g}."
            )
        run_name = path.parent.name if path.name == "macroData.csv" else path.stem
        run_rescue = None if rescued is None else rescued.loc[
            rescued["run_name"] == run_name.removesuffix("_fixed")
        ]
        if run_rescue is not None and not run_rescue.empty:
            run_rescue = run_rescue.loc[
                (run_rescue["load_ip1"] > post_yield_start)
                & (run_rescue["load_ip1"] < load_end)
                & (run_rescue["delta_E_R"] > 0)
            ]
            er = run_rescue["delta_E_R"].to_numpy(dtype=float)
            es = run_rescue["delta_E_S"].to_numpy(dtype=float)
            kappa = kappa_from_relaxation_energy(
                er, run_rescue["delta_gamma"].to_numpy(dtype=float),
                run_rescue["reference_volume"].to_numpy(dtype=float),
            )
            data_path = Path(snapshot_root) / "tables" / "drops_all_audited.csv.gz"
        else:
            extracted = extract_event_pairs(
                data_path, size, {"post": (post_yield_start, load_end)},
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
                "positive_irreversible_delta_E_S_count": int(np.count_nonzero(
                    (kappa >= kappa_det) & np.isfinite(es) & (es > 0)
                )),
                "irreversible_missing_delta_E_S_count": int(np.count_nonzero(
                    (kappa >= kappa_det) & ~np.isfinite(es)
                )),
            }
        )

    drops = EventDrops(
        er=np.concatenate(er_parts),
        es=np.concatenate(es_parts),
        kappa=np.concatenate(kappa_parts),
    )
    split = split_by_kappa(drops, kappa_det)
    es_irrev = positive_es(drops, split.is_irrev)
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
    data: np.ndarray, *, cache_dir: Path, description: str, evaluate_pvalue: bool = True,
    bootstrap_confidence: float = BOOTSTRAP_CONFIDENCE, bootstrap_workers: int = 1
) -> tuple[Fit, dict]:
    """Select the exhaustive global KS cutoff and evaluate its fixed-xmin fit."""
    xmin_analysis = _exhaustive_xmin_analysis(
        data, cache_dir=cache_dir / "exhaustive_xmin", parallel=True,
    )
    xmin = float(xmin_analysis["global_min_xmin"])
    fit = Fit(
        data=data,
        xmin=xmin,
        xmin_distribution=Truncated_Power_Law.name,
        verbose=0,
    )
    fit.xmin_analysis = xmin_analysis
    fit.xmin_fitting_results = xmin_analysis
    fit.xmin_selection = "global"
    if evaluate_pvalue:
        fit.evaluate_fit(
            data=data, confidence=bootstrap_confidence, parallel=bootstrap_workers > 1,
            max_workers=bootstrap_workers,
            cache_dir=str(cache_dir), tqdmDesc=description,
        )
    distribution = dist_from_fit(fit)
    if not np.isfinite(distribution.alpha) or not np.isfinite(distribution.Lambda) or distribution.Lambda <= 0:
        raise RuntimeError(f"Invalid truncated-power-law parameters for {description}")
    fit_summary = {
        "xmin": xmin,
        "ks_distance": float(fit.D),
        "alpha": float(distribution.alpha),
        "alpha_std": float(getattr(fit, "alpha_std", np.nan)) if evaluate_pvalue else None,
        "lambda": float(distribution.Lambda),
        "lambda_std": float(getattr(fit, "Lambda_std", np.nan)) if evaluate_pvalue else None,
        "p": float(getattr(fit, "p", np.nan)) if evaluate_pvalue else None,
        "p_std": float(getattr(fit, "p_std", np.nan)) if evaluate_pvalue else None,
        "bootstrap_replicates": max(1, int(np.ceil(1 / (4 * bootstrap_confidence**2)))) if evaluate_pvalue else 0,
        "xmin_candidate_count": int(xmin_analysis["xmins"].size),
        "tail_count": int(np.count_nonzero(data >= xmin)),
        "xmin_selection": "exhaustive global KS minimum",
        "bootstrap_xmin_mode": "global" if evaluate_pvalue else None,
        "bootstrap_confidence": bootstrap_confidence if evaluate_pvalue else None,
        "alpha_mean": float(fit.alpha_mean) if evaluate_pvalue else None,
        "lambda_mean": float(fit.Lambda_mean) if evaluate_pvalue else None,
        "xmin_mean": float(fit.bootstrap_xmin_mean) if evaluate_pvalue else None,
        "xmin_std": float(fit.bootstrap_xmin_std) if evaluate_pvalue else None,
    }
    return fit, fit_summary


def make_plot(
    reconnecting_csv: list[Path],
    nonreconnecting_csv: list[Path],
    output_pdf: Path,
    *,
    size: int = DEFAULT_SIZE,
    cache_dir: Path,
    evaluate_pvalue: bool = True,
    nonreconnecting_snapshot: Path | None = None,
    bootstrap_confidence: float = BOOTSTRAP_CONFIDENCE,
    bootstrap_workers: int = 1,
    expected_populations: Path | None = None,
    check_only: bool = False,
    frozen_summary: Path | None = None,
) -> Path:
    if int(size) != size or size <= 0:
        raise ValueError(f"size must be a positive integer; got {size!r}")
    size = int(size)
    if not 0 < bootstrap_confidence <= 0.5 or bootstrap_workers < 1:
        raise ValueError("Require 0 < bootstrap_confidence <= 0.5 and positive workers")
    expected = None
    if expected_populations is not None:
        with np.load(expected_populations, allow_pickle=False) as archive:
            expected = {name: archive[name].copy() for name in archive.files}
    _configure_matplotlib()
    reference_volume = float(size**2)
    kappa_det = kappa_detection_threshold()
    populations = {}
    counts = {}
    fits = {}
    fit_objects = {}
    if frozen_summary is not None:
        saved = json.loads(Path(frozen_summary).read_text())
        if expected is None or saved["size"] != size or saved["delta_E_S_tangent"] != "gamma0" or saved["post_yield_rule"] != POST_YIELD_RULE:
            raise ValueError("Frozen plot requires matching populations, size, gamma0 and Piola yield rule")
        populations, counts = expected, saved["populations"]
        if set(populations) != {"reconnecting", "non-reconnecting"}:
            raise ValueError("Frozen plot must contain both comparison populations")
        for name, data in populations.items():
            if not np.all(np.isfinite(data) & (data > 0)) or data.size != counts[name]["positive_irreversible_delta_E_S_count"]:
                raise ValueError(f"Invalid frozen irreversible population for {name}")
    else:
        for name, paths in (
            ("reconnecting", reconnecting_csv),
            ("non-reconnecting", nonreconnecting_csv),
        ):
            data, group_counts = _collect_irreversible_delta_es(
                paths,
                size=size,
                cache_dir=cache_dir,
                kappa_det=kappa_det,
                snapshot_root=nonreconnecting_snapshot if name == "non-reconnecting" else None,
            )
            populations[name] = data
            counts[name] = group_counts
            if expected is not None and not np.array_equal(data, expected[name]):
                raise RuntimeError(f"{name} population differs from the frozen plotted data")
    if check_only:
        print("Verified both ten-seed populations against the frozen plot; no bootstrap started.")
        return Path(output_pdf)

    for name, data in populations.items():
        fit_objects[name], fits[name] = _fit_population(
            data,
            cache_dir=cache_dir / "comparison_bootstrap_cache" / name,
            description=f"{name} L={size}",
            evaluate_pvalue=evaluate_pvalue,
            bootstrap_confidence=bootstrap_confidence, bootstrap_workers=bootstrap_workers,
        )

    figure, axis = plt.subplots(figsize=(2.96, 2.16))
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
            drop_label=r"E_S",
            drop_sign="positive",
            show_legend=False,
        )
        data_lines = axis.lines[data_line_start:]
        if len(data_lines) != 1:
            raise RuntimeError(
                f"Expected one data-PDF line for {name}; got {len(data_lines)}."
            )
        data_lines[0].set_markerfacecolor("none")
        data_lines[0].set_markersize(2.6)
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
            drop_label=r"E_S",
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

    axis.grid(False, which="both")
    handles = []
    for name in ("reconnecting", "non-reconnecting"):
        result = fits[name]
        handles.append(Line2D(
            [], [], marker="o", linestyle="None", color=colors[name],
            markerfacecolor="none", markersize=3,
            label=name.capitalize(),
        ))
        alpha_label = rf"$\hat\alpha={result['alpha']:.2f}$"
        lambda_label = rf"$\hat\lambda={result['lambda']:.2f}$"
        if evaluate_pvalue:
            exponent = int(np.floor(np.log10(result["lambda"])))
            scale = 10.0 ** exponent
            alpha_label = rf"$\hat\alpha={result['alpha']:.2f}\pm{result['alpha_std']:.2f}$"
            lambda_label = (
                rf"$\hat\lambda=({result['lambda']/scale:.2f}\pm"
                rf"{result['lambda_std']/scale:.2f})\times10^{{{exponent}}}$"
            )
        label = alpha_label + ", " + lambda_label + f", tail n={result['tail_count']:,}"
        handles.append(Line2D([], [], color=colors[name], linewidth=1, label=label))
        axis.axvline(result["xmin"], color=colors[name], linestyle="--", linewidth=0.7)
    handles.append(Line2D([], [], color="black", linestyle="--", linewidth=0.7,
                          label=r"$\Delta E_{S,\min}^{KS}$"))
    axis.legend(handles=handles, fontsize=5.2, loc="lower left", framealpha=0.86)
    output_pdf = Path(output_pdf)
    axis.set_xlabel(r"$\Delta E_S$")
    axis.set_ylabel(r"$p(\Delta E_S)$")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_pdf.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)
    np.savez_compressed(output_pdf.with_suffix(".npz"), **populations)

    summary = {
        "size": size,
        "reference_volume": reference_volume,
        "energy_normalization": "none (total Delta E_S)",
        "delta_E_S_tangent": "gamma0",
        "p_value_evaluated": evaluate_pvalue,
        "nonreconnecting_snapshot": str(nonreconnecting_snapshot) if nonreconnecting_snapshot else None,
        "post_yield_rule": POST_YIELD_RULE,
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
    parser.add_argument("--skip-pvalue", action="store_true")
    parser.add_argument("--non-reconnecting-snapshot", type=Path)
    args = parser.parse_args()

    output = make_plot(
        args.reconnecting_csv,
        args.non_reconnecting_csv,
        args.output,
        size=args.size,
        cache_dir=args.cache_dir,
        evaluate_pvalue=not args.skip_pvalue,
        nonreconnecting_snapshot=args.non_reconnecting_snapshot,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
