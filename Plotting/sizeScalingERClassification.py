"""Kappa-based reversible/irreversible decomposition for size-scaling drops.

The non-reconnecting size-scaling runs are classified independently for each
system size and strain regime.  The default detector is ``kappa_det = mu / (2 rho)``
with ``rho=N/V_0=2``.  Events below that threshold are
reversible; events at or above it are irreversible.  The labels are then
applied to the paired :math:`\\Delta E_S` values from the same events.  The
standard reported fit is the irreversible-only :math:`\\Delta E_S` fit,
using an exhaustive global-minimum xmin whenever feasible.  This script also
produces :math:`\\Delta E_R` diagnostics, and ``population_mode="all"`` is an
explicit comparison mode that bypasses the standard split; neither should be
mistaken for the standard result.  With ``population_mode="all"``, no initial
E_R split is made and both fields are fitted directly on the paired positive
event populations.

This module's figures compare *multiple* sizes. For individual-size kappa,
reversible/irreversible, and fitted-PDF panels, use
``truncated_powerlaw_reversibility_flowchart.generate_single_size_panels``.

Run from the repository root with::

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python -m \
        Plotting.sizeScalingERClassification
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

from Plotting.plotPowerLaw import (
    POWERLAW_STANDARD_WORKFLOW,
    dist_from_fit,
    plot_data_pdf,
    plot_fit_pdf,
    plot_xmin_analysis,
)
from Plotting.sizeScalingCollapse import (
    YIELD_REGIME_NAMES,
    size_scaling_paths,
    pool_event_pairs,
)
from Plotting.plotPowerLaw import make_fit
from Plotting.standardPowerlaw import (
    DEFAULT_KAPPA_RHO,
    POST_YIELD_RULE,
    SIZE_SCALING_MIN_SIZE,
    EventDrops,
    kappa_detection_threshold,
    positive_es,
    split_by_kappa,
)


DEFAULT_DATA_ROOT = Path("/Volumes/data/remoteData/macro")
DEFAULT_OUTPUT_DIR = Path(
    "Plots/reversible_event_analysis/size_scaling_E_R_classification_piola_peak_post_only"
)
PROTOCOL_LABELS = {
    "initial_guess_energy": r"$\Delta E_R$",
    "second_order": r"$\Delta E_S$",
}
DROP_LABELS = {
    "initial_guess_energy": r"E_R",
    "second_order": r"E_S",
}


def _positive(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _classified_population(events, kappa_det):
    drops = EventDrops(
        er=events["initial_guess_energy"],
        es=events["second_order"],
        kappa=events["kappa"],
    )
    split = split_by_kappa(drops, kappa_det)
    valid_er = split.is_rev | split.is_irrev
    return {
        "es_all": _positive(drops.es[valid_er]),
        "es_rev": positive_es(drops, split.is_rev),
        "es_irrev": positive_es(drops, split.is_irrev),
        "er_all": drops.er[valid_er],
        "er_rev": drops.er[split.is_rev],
        "er_irrev": drops.er[split.is_irrev],
        "kappa_det": split.kappa_det,
        "reversible_count": int(np.count_nonzero(split.is_rev)),
        "irreversible_count": int(np.count_nonzero(split.is_irrev)),
        "labeled_count": int(np.count_nonzero(valid_er)),
    }


def _all_event_data(events):
    drops = EventDrops(
        er=events["initial_guess_energy"],
        es=events["second_order"],
    )
    valid_er = np.isfinite(drops.er) & (drops.er > 0)
    return {
        "es_all": _positive(drops.es[valid_er]),
        "er_all": drops.er[valid_er],
    }


def _plot_decomposition(regime, populations, output_dir):
    sizes = sorted(populations)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(sizes)))
    for name in ("es_all", "es_rev", "es_irrev"):
        fig, ax = plt.subplots(figsize=(4.4, 3.36))
        for size, color in zip(sizes, colors):
            data = populations[size][name]
            if data.size < 3:
                ax.plot([], [], color=color, marker="o", linestyle="none",
                        label=f"L={size}: too few positive drops")
                continue
            plot_data_pdf(ax, data, label=f"L={size}", color=color,
                          drop_label=DROP_LABELS["second_order"],
                          drop_sign="positive", show_legend=False)
        ax.set(xlabel=r"$\Delta E_S$", ylabel=r"$p(\Delta E_S)$")
        ax.legend(fontsize="small")
        ax.grid(alpha=0.18)
        fig.tight_layout()
        _save_figure(fig, output_dir / f"delta_E_S_{name}_{regime}_yield.pdf")


def _evaluate_fit(fit, data, confidence, parallel, cache_dir, description):
    fit.evaluate_fit(
        data=data,
        confidence=confidence,
        parallel=parallel,
        cache_dir=str(cache_dir),
        tqdmDesc=description,
    )
    distribution = dist_from_fit(fit)
    alpha = float(distribution.alpha)
    cutoff = float(getattr(distribution, "Lambda", np.nan))
    if not np.isfinite(alpha) or not np.isfinite(cutoff) or cutoff <= 0:
        raise RuntimeError(f"Invalid truncated-power-law parameters: {description}")
    return {
        "xmin": float(fit.xmin),
        "tail_count": int(np.count_nonzero(np.asarray(data) >= fit.xmin)),
        "alpha": alpha,
        "alpha_std": float(getattr(fit, "alpha_std", np.nan)),
        "Lambda": cutoff,
        "Lambda_std": float(getattr(fit, "Lambda_std", np.nan)),
        "D": float(fit.D),
        "p": float(getattr(fit, "p", np.nan)),
        "p_std": float(getattr(fit, "p_std", np.nan)),
    }


def _make_global_min_fit(
    data,
    search_cache_dir,
    fit_cache_dir,
    parallel_xmin,
    description,
    *,
    refine=True,
):
    search_fit = make_fit(
        data,
        cache_dir=str(search_cache_dir),
        parallel_xmin=parallel_xmin,
        xmin_search_kwargs={
            "progress": True,
            "progress_label": description,
            "refine": refine,
        },
    )
    analysis = getattr(search_fit, "xmin_analysis", None)
    if analysis is None or "global_min_xmin" not in analysis:
        raise RuntimeError(f"Missing global-minimum xmin analysis: {description}")
    global_xmin = float(analysis["global_min_xmin"])
    fit = make_fit(
        data,
        xmin_range=global_xmin,
        cache_dir=str(fit_cache_dir),
    )
    fit.xmin_analysis = analysis
    fit.xmin_fitting_results = analysis
    return fit


def _plot_fit_grid(
    regime,
    fit_results,
    output_dir,
    title_prefix,
    filename_prefix,
    data_label,
):
    sizes = sorted(fit_results)
    fields = ("initial_guess_energy", "second_order")
    for field in fields:
        for size in sizes:
            result = fit_results[size][field]
            fig, ax = plt.subplots(figsize=(4.4, 3.36))
            data = result["data"]
            fit = result["fit"]
            plot_data_pdf(
                ax,
                data,
                label=data_label,
                color="black",
                drop_label=DROP_LABELS[field],
                drop_sign="positive",
                show_legend=False,
            )
            plot_fit_pdf(
                ax,
                fit,
                color="C3",
                label="fit continuation below $x_{min}$",
                linestyle="--",
                drop_label=DROP_LABELS[field],
                drop_sign="positive",
                show_legend=False,
                set_title=False,
                x_grid_mode="smooth",
                xmin_only=False,
                linewidth=1.1,
            )
            plot_fit_pdf(
                ax,
                fit,
                color="C3",
                label="truncated power law",
                drop_label=DROP_LABELS[field],
                drop_sign="positive",
                show_legend=False,
                set_title=False,
                x_grid_mode="smooth",
                xmin_only=True,
                linewidth=1.3,
            )
            ax.axvline(
                fit.xmin,
                color="C0",
                linestyle="--",
                linewidth=0.9,
            )
            ax.grid(alpha=0.18)
            annotation = (
                rf"$x_{{min}}={fit.xmin:.2e}$" "\n"
                rf"$\alpha={result['alpha']:.2f}$, $\lambda={result['Lambda']:.2e}$" "\n"
                rf"$D={result['D']:.3f}$, $p={result['p']:.2f}$"
            )
            ax.text(
                0.04,
                0.04,
                annotation,
                transform=ax.transAxes,
                va="bottom",
                fontsize="x-small",
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
            ax.set_ylabel(rf"$p(\Delta {DROP_LABELS[field]})$")
            ax.set_xlabel(PROTOCOL_LABELS[field])
            ax.legend(fontsize="small")
            fig.tight_layout()
            _save_figure(fig, output_dir / f"{filename_prefix}_{field}_L{size}_{regime}_yield.pdf")


def _plot_ks_grid(
    regime,
    fit_results,
    output_dir,
    title_prefix,
    filename_prefix,
):
    sizes = sorted(fit_results)
    fields = ("initial_guess_energy", "second_order")
    for field in fields:
        for size in sizes:
            result = fit_results[size][field]
            fit = result["fit"]
            fig, ax = plt.subplots(figsize=(4.4, 3.36))
            analysis = fit.xmin_fitting_results
            if analysis is None:
                raise RuntimeError(f"Missing xmin analysis for L={size}, {field}.")
            plot_xmin_analysis(analysis, ax=ax)
            ax.set(xlabel=r"candidate $x_{min}$", ylabel=r"KS distance $D$")
            ax.set_title("")
            fig.tight_layout()
            _save_figure(fig, output_dir / f"{filename_prefix}_{field}_L{size}_{regime}_yield.pdf")


def run(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    seeds_per_size: int | None = None,
    minimum_size: int = SIZE_SCALING_MIN_SIZE,
    bins_per_decade: int = 10,
    parallel_xmin: bool = False,
    parallel_uncertainty: bool = False,
    uncertainty_accuracy: float = 0.1,
    force: bool = False,
    population_mode: str = "classified",
    yield_regime: str = "both",
    largest_first: bool = False,
    split_only: bool = False,
):
    if seeds_per_size is not None and seeds_per_size < 1:
        raise ValueError("seeds_per_size must be positive.")
    if bins_per_decade < 1:
        raise ValueError("bins_per_decade must be positive.")
    if not 0 < uncertainty_accuracy < 0.5:
        raise ValueError("uncertainty_accuracy must lie between 0 and 0.5.")
    if population_mode not in {"classified", "all"}:
        raise ValueError("population_mode must be 'classified' or 'all'.")
    if yield_regime not in {"both", "pre", "post"}:
        raise ValueError("yield_regime must be 'both', 'pre', or 'post'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths, inventory = size_scaling_paths(data_root, seeds_per_size,
                                        minimum_size=minimum_size)
    if largest_first:
        paths = dict(sorted(paths.items(), reverse=True))
    regimes = YIELD_REGIME_NAMES if yield_regime == "both" else (yield_regime,)
    events_by_regime = pool_event_pairs(
        paths,
        regimes,
        output_dir / "cache" / "event_pairs",
        force=force,
    )
    kappa_det = kappa_detection_threshold()
    classification_rows = []
    fit_rows = []
    summary = {
        "data_root": str(data_root),
        "reconnection": "none",
        "yield_split": POST_YIELD_RULE,
        "post_yield_rule": POST_YIELD_RULE,
        "yield_regimes": regimes,
        "fit_size_order": list(paths),
        "seeds_per_size": seeds_per_size,
        "minimum_size": minimum_size,
        "source_csvs": {size: [str(path) for path in files] for size, files in paths.items()},
        "sample_counts": {size: len(files) for size, files in paths.items()},
        "sample_selection": "available runs extending past their avg_P12 peak +0.01; partial runs included",
        "analysis_stage": "split_only" if split_only else "fits",
        "population_mode": population_mode,
        "selection": (
            "kappa_det = mu/(2 rho) classification with rho=N/V_0=2"
            if population_mode == "classified"
            else "all paired events; no initial E_R split"
        ),
        "kappa_detection_threshold": "kappa_det = mu/(2 rho) with rho=N/V_0=2",
        "mu": float(2.0 * DEFAULT_KAPPA_RHO * kappa_det),
        "E_S_fit_population": (
            "irreversible delta_E_S"
            if population_mode == "classified"
            else "all positive delta_E_S"
        ),
        "E_R_fit_xmin_method": "global_min",
        "E_R_fit_xmin_search": (
            "refine=True, full initial scan"
            if population_mode == "classified"
            else "refine=False, 100-point global scan"
        ),
        "E_S_fit_xmin_method": "global_min",
        "E_S_fit_xmin_search": (
            "refine=True, full initial scan"
            if population_mode == "classified"
            else "refine=False, 100-point global scan"
        ),
        "inventory": inventory,
        "classification": {} if population_mode == "classified" else None,
        "fits": {},
    }

    for regime in regimes:
        events = events_by_regime[regime]
        if population_mode == "classified":
            populations = {
                size: _classified_population(event, kappa_det)
                for size, event in events.items()
            }
            _plot_decomposition(regime, populations, output_dir)
            fit_title = r"Irreversible tails after $\kappa_{\det}$ classification"
            fit_filename_prefix = "irreversible_truncated_powerlaw_fits"
            fit_data_label = "irreversible data"
        else:
            populations = {
                size: _all_event_data(event)
                for size, event in events.items()
            }
            fit_title = "All event tails"
            fit_filename_prefix = "all_events_truncated_powerlaw_fits"
            fit_data_label = "all data"

        if population_mode == "classified":
            summary["classification"][regime] = {}
        fit_results = {}
        for size in sorted(events, reverse=largest_first):
            population = populations[size]
            if population_mode == "classified":
                size_kappa_det = population["kappa_det"]
                summary["classification"][regime][str(size)] = {
                    "kappa_det": size_kappa_det,
                    "labeled_count": population["labeled_count"],
                    "reversible_count": population["reversible_count"],
                    "irreversible_count": population["irreversible_count"],
                    "E_S_all_count": int(population["es_all"].size),
                    "E_S_reversible_count": int(population["es_rev"].size),
                    "E_S_irreversible_count": int(population["es_irrev"].size),
                }
                classification_rows.append(
                    {
                        "regime": regime,
                        "size": size,
                        "kappa_det": size_kappa_det,
                        "labeled_count": population["labeled_count"],
                        "reversible_count": population["reversible_count"],
                        "irreversible_count": population["irreversible_count"],
                        "E_S_all_count": population["es_all"].size,
                        "E_S_reversible_count": population["es_rev"].size,
                        "E_S_irreversible_count": population["es_irrev"].size,
                    }
                )
                er_data = population["er_irrev"]
                es_data = population["es_irrev"]
                classification_kappa_det = size_kappa_det
            else:
                er_data = population["er_all"]
                es_data = population["es_all"]
                classification_kappa_det = None
            if split_only:
                continue
            if er_data.size < 3 or es_data.size < 3:
                raise ValueError(
                    f"Too few fit events for L={size}, {regime}: "
                    f"E_R={er_data.size}, E_S={es_data.size}."
                )
            er_fit = _make_global_min_fit(
                er_data,
                output_dir / "cache" / "global_min" / "initial_guess_energy" / regime,
                output_dir / "cache" / "fixed_global_fits" / "initial_guess_energy" / regime,
                parallel_xmin,
                f"E_R {regime}-yield global min, L={size}",
                refine=population_mode == "classified",
            )
            es_fit = _make_global_min_fit(
                es_data,
                output_dir / "cache" / "global_min" / "second_order" / regime,
                output_dir / "cache" / "fixed_global_fits" / "second_order" / regime,
                parallel_xmin,
                f"E_S {regime}-yield global min, L={size}",
                refine=population_mode == "classified",
            )
            er_record = _evaluate_fit(
                er_fit,
                er_data,
                uncertainty_accuracy,
                parallel_uncertainty,
                output_dir / "cache" / "evaluation" / "initial_guess_energy" / regime,
                f"E_R {regime}-yield, L={size}",
            )
            es_record = _evaluate_fit(
                es_fit,
                es_data,
                uncertainty_accuracy,
                parallel_uncertainty,
                output_dir / "cache" / "evaluation" / "second_order" / regime,
                f"E_S {regime}-yield, L={size}",
            )
            if population_mode == "classified":
                es_record["es_xmin_ks"] = es_record["xmin"]
            fit_results[size] = {
                "initial_guess_energy": {"fit": er_fit, "data": er_data, **er_record},
                "second_order": {"fit": es_fit, "data": es_data, **es_record},
            }
            summary["fits"].setdefault(regime, {})[str(size)] = {}
            for field, record in fit_results[size].items():
                summary["fits"][regime][str(size)][field] = {
                    key: value
                    for key, value in record.items()
                    if key not in {"fit", "data"}
                }
                fit_rows.append(
                    {
                        "regime": regime,
                        "size": size,
                        "field": field,
                        "data_count": record["data"].size,
                        "classification_kappa_det": classification_kappa_det,
                        "population_mode": population_mode,
                        **{
                            key: value
                            for key, value in record.items()
                            if key not in {"fit", "data"}
                        },
                    }
                )
            # Per-size diagnostics belong to the reversibility flowchart;
            # keep only the cross-size grids here.
        if not split_only:
            _plot_fit_grid(
                regime, fit_results, output_dir, fit_title,
                fit_filename_prefix, fit_data_label,
            )
            _plot_ks_grid(
                regime, fit_results, output_dir, fit_title,
                f"{fit_filename_prefix}_D_simpleDrop_vs_global_min",
            )

    if classification_rows:
        pd.DataFrame(classification_rows).to_csv(
            output_dir / "classification_summary.csv", index=False
        )
    if fit_rows:
        pd.DataFrame(fit_rows).to_csv(output_dir / "fit_results.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=POWERLAW_STANDARD_WORKFLOW,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds-per-size", type=int,
                        help="Optional sample cap; default uses all available post-yield runs.")
    parser.add_argument("--minimum-size", type=int, default=SIZE_SCALING_MIN_SIZE,
                        help="Minimum system size (default: 100; use 50 for an explicitly requested comparison).")
    parser.add_argument("--split-only", action="store_true",
                        help="Regenerate raw distributions and classification counts without fits.")
    parser.add_argument("--bins-per-decade", type=int, default=10)
    parser.add_argument("--parallel-xmin", action="store_true")
    parser.add_argument("--parallel-uncertainty", action="store_true")
    parser.add_argument("--uncertainty-accuracy", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--population-mode", choices=("classified", "all"), default="classified")
    parser.add_argument("--yield-regime", choices=("both", "pre", "post"), default="both")
    parser.add_argument("--largest-first", action="store_true")
    args = parser.parse_args()
    run(
        args.data_root,
        args.output_dir,
        seeds_per_size=args.seeds_per_size,
        minimum_size=args.minimum_size,
        bins_per_decade=args.bins_per_decade,
        parallel_xmin=args.parallel_xmin,
        parallel_uncertainty=args.parallel_uncertainty,
        uncertainty_accuracy=args.uncertainty_accuracy,
        force=args.force,
        population_mode=args.population_mode,
        yield_regime=args.yield_regime,
        largest_first=args.largest_first,
        split_only=args.split_only,
    )


if __name__ == "__main__":
    main()
