"""E_R-based reversible/irreversible decomposition for size-scaling drops.

The non-reconnecting size-scaling runs are classified independently for each
system size and strain regime.  A canonical ``simpleDrop`` xmin is found from
the positive :math:`\\Delta E_R` values.  Events below that threshold are
reversible; events at or above it are irreversible.  The labels are then
applied to the aligned :math:`\\Delta E_S` values from the same events.  The
irreversible-only :math:`\\Delta E_R` and :math:`\\Delta E_S` fits use their
respective refined global-minimum xmins.  With ``population_mode="all"``, no
initial E_R split is made and both fields are fitted directly on the aligned
positive event populations.

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
    dist_from_fit,
    plot_data_pdf,
    plot_fit_pdf,
    plot_KS_fitting,
)
from Plotting.sizeScalingCollapse import (
    REGIMES,
    completed_size_scaling_paths,
    fit_xmins,
    pool_aligned_events,
)
from Plotting.plotPowerLaw import make_fit


DEFAULT_DATA_ROOT = Path("/Volumes/data/remoteData/macro")
DEFAULT_OUTPUT_DIR = Path(
    "Plots/reversible_event_analysis/size_scaling_E_R_classification"
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


def _population_data(events, xmin):
    e_r = np.asarray(events["initial_guess_energy"], dtype=float)
    e_s = np.asarray(events["second_order"], dtype=float)
    if e_r.shape != e_s.shape:
        raise RuntimeError("E_R and E_S event arrays must be aligned.")
    valid_e_r = np.isfinite(e_r) & (e_r > 0)
    reversible = valid_e_r & (e_r < xmin)
    irreversible = valid_e_r & (e_r >= xmin)
    return {
        "all": _positive(e_s[valid_e_r]),
        "reversible": _positive(e_s[reversible]),
        "irreversible": _positive(e_s[irreversible]),
        "e_r_all": e_r[valid_e_r],
        "e_r_reversible": e_r[reversible],
        "e_r_irreversible": e_r[irreversible],
        "e_s_irreversible": _positive(e_s[irreversible]),
        "reversible_count": int(np.count_nonzero(reversible)),
        "irreversible_count": int(np.count_nonzero(irreversible)),
        "labeled_count": int(np.count_nonzero(valid_e_r)),
    }


def _all_event_data(events):
    e_r = np.asarray(events["initial_guess_energy"], dtype=float)
    e_s = np.asarray(events["second_order"], dtype=float)
    if e_r.shape != e_s.shape:
        raise RuntimeError("E_R and E_S event arrays must be aligned.")
    valid_e_r = np.isfinite(e_r) & (e_r > 0)
    return {
        "all": _positive(e_s[valid_e_r]),
        "e_r_all": e_r[valid_e_r],
    }


def _plot_decomposition(regime, populations, output_dir):
    sizes = sorted(populations)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(sizes)))
    names = ("all", "reversible", "irreversible")
    titles = ("All drops", "Reversible", "Irreversible")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.9),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    for ax, name, title in zip(axes, names, titles):
        for size, color in zip(sizes, colors):
            data = populations[size][name]
            if data.size < 3:
                ax.text(
                    0.5,
                    0.5,
                    f"L={size}:\nno positive dE_S",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize="x-small",
                    color=color,
                )
                continue
            plot_data_pdf(
                ax,
                data,
                label=f"L={size}",
                color=color,
                drop_label=DROP_LABELS["second_order"],
                drop_sign="positive",
                show_legend=False,
            )
        ax.set_title(title)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel(r"$p(\Delta E_S)$")
    for ax in axes[1:]:
        ax.set_ylabel("")
    for ax in axes:
        ax.set_xlabel(r"$\Delta E_S$")
    handles = [
        Line2D([], [], color=color, marker="o", linestyle="None", label=f"L={size}")
        for size, color in zip(sizes, colors)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize="small",
    )
    fig.suptitle(
        r"$\Delta E_R$ classification applied to $\Delta E_S$ events; "
        + ("pre-yield" if regime == "pre" else "post-yield"),
        y=1.10,
        fontsize="medium",
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.77, wspace=0.08)
    _save_figure(fig, output_dir / f"delta_E_S_pdf_decomposition_{regime}_yield.pdf")


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
    fig, axes = plt.subplots(
        2,
        len(sizes),
        figsize=(2.25 * len(sizes), 5.0),
        squeeze=False,
    )
    for row, field in enumerate(fields):
        for column, size in enumerate(sizes):
            result = fit_results[size][field]
            ax = axes[row, column]
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
            ax.set_title(f"L={size}")
            ax.grid(alpha=0.18)
            if row == 0:
                ax.set_xlabel("")
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
            if column == 0:
                ax.set_ylabel(rf"$p(\Delta {DROP_LABELS[field]})$")
            else:
                ax.set_ylabel("")
            if row == 1:
                ax.set_xlabel(PROTOCOL_LABELS[field])
    handles = [
        Line2D([], [], color="black", marker="o", linestyle="None", label=data_label),
        Line2D([], [], color="C3", label="truncated power-law fit"),
        Line2D(
            [],
            [],
            color="C3",
            linestyle="--",
            label=r"fit continuation below $x_{min}$",
        ),
        Line2D([], [], color="C0", linestyle="--", label=r"$x_{min}$"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize="small",
        bbox_to_anchor=(0.5, 0.96),
    )
    fig.suptitle(
        title_prefix + "; "
        + ("pre-yield" if regime == "pre" else "post-yield"),
        y=0.995,
    )
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.13, top=0.82, wspace=0.34, hspace=0.36)
    _save_figure(fig, output_dir / f"{filename_prefix}_{regime}_yield.pdf")


def _save_ks_plot(fit, path: Path):
    result = plot_KS_fitting(fit, save=False, show=False)
    if result is None:
        raise RuntimeError(f"No valid KS xmin trace for {path}.")
    fig, (ax1, ax2) = result
    analysis = getattr(fit, "xmin_fitting_results", None)
    if analysis is None:
        raise RuntimeError(f"No xmin analysis available for {path}.")
    simple_xmin = float(analysis["simple_drop_xmin"])
    global_xmin = float(analysis["global_min_xmin"])
    simple_distance = float(analysis["simple_drop_distance"])
    global_distance = float(analysis["global_min_distance"])
    ax1.axvline(
        simple_xmin,
        color="tab:purple",
        linestyle="--",
        linewidth=1.2,
        label=f"simpleDrop xmin: {simple_xmin:.2e}",
        alpha=0.9,
    )
    ax1.scatter(
        [simple_xmin],
        [simple_distance],
        marker="D",
        color="tab:purple",
        s=34,
        zorder=6,
    )
    ax1.scatter(
        [global_xmin],
        [global_distance],
        marker="X",
        facecolor="white",
        edgecolor="0.15",
        linewidth=0.9,
        s=48,
        zorder=7,
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", ncol=2)
    _save_figure(fig, path)


def _ks_curve_data(fit):
    analysis = getattr(fit, "xmin_fitting_results", None)
    if analysis is None:
        raise RuntimeError("Missing xmin fitting results for D plot.")
    xmins = np.asarray(analysis["xmins"], dtype=float)
    distances = np.asarray(analysis["distances"], dtype=float)
    valid_fits = np.asarray(analysis.get("valid_fits", True), dtype=bool)
    if valid_fits.shape != xmins.shape:
        raise RuntimeError("xmin validity mask does not match xmin candidates.")
    mask = np.isfinite(xmins) & np.isfinite(distances) & valid_fits & (xmins > 0)
    if not np.any(mask):
        raise RuntimeError("No valid KS distances are available for D plot.")
    order = np.argsort(xmins[mask])
    return xmins[mask][order], distances[mask][order]


def _plot_ks_grid(
    regime,
    fit_results,
    output_dir,
    title_prefix,
    filename_prefix,
):
    sizes = sorted(fit_results)
    fields = ("initial_guess_energy", "second_order")
    fig, axes = plt.subplots(
        2,
        len(sizes),
        figsize=(2.25 * len(sizes), 5.0),
        sharey=True,
        squeeze=False,
    )
    for row, field in enumerate(fields):
        for column, size in enumerate(sizes):
            result = fit_results[size][field]
            fit = result["fit"]
            ax = axes[row, column]
            xmins, distances = _ks_curve_data(fit)
            ax.plot(
                xmins,
                distances,
                color="tab:blue",
                marker="o",
                markersize=2.5,
                markerfacecolor="none",
                linewidth=0.9,
            )
            analysis = fit.xmin_fitting_results
            simple_xmin = float(analysis["simple_drop_xmin"])
            global_xmin = float(analysis["global_min_xmin"])
            simple_distance = float(analysis["simple_drop_distance"])
            global_distance = float(analysis["global_min_distance"])
            ax.axvline(
                simple_xmin,
                color="tab:purple",
                linestyle="--",
                linewidth=1.0,
            )
            ax.axvline(
                global_xmin,
                color="0.15",
                linestyle=":",
                linewidth=1.2,
            )
            ax.scatter(
                [simple_xmin],
                [simple_distance],
                marker="D",
                color="tab:purple",
                s=25,
                zorder=5,
            )
            ax.scatter(
                [global_xmin],
                [global_distance],
                marker="X",
                facecolor="white",
                edgecolor="0.15",
                linewidth=0.8,
                s=36,
                zorder=6,
            )
            ax.set_xscale("log")
            ax.set_title(f"L={size}")
            ax.grid(alpha=0.18)
            ax.text(
                0.04,
                0.04,
                rf"$x_{{min}}^{{SD}}={simple_xmin:.2e}$" "\n"
                rf"$x_{{min}}^{{global}}={global_xmin:.2e}$",
                transform=ax.transAxes,
                va="bottom",
                fontsize="x-small",
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
            if column == 0:
                ax.set_ylabel(r"KS distance $D$")
            else:
                ax.set_ylabel("")
            if row == 1:
                ax.set_xlabel(r"candidate $x_{min}$")
            else:
                ax.set_xlabel("")
    handles = [
        Line2D([], [], color="tab:blue", marker="o", markerfacecolor="none", label=r"$D(x_{min})$"),
        Line2D([], [], color="tab:purple", linestyle="--", marker="D", label="simpleDrop"),
        Line2D([], [], color="0.15", linestyle=":", marker="X", label="global minimum"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize="small",
        bbox_to_anchor=(0.5, 0.96),
    )
    fig.suptitle(
        title_prefix + r"; $D(x_{min})$ comparison; "
        + ("pre-yield" if regime == "pre" else "post-yield"),
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.13,
        top=0.82,
        wspace=0.34,
        hspace=0.36,
    )
    _save_figure(fig, output_dir / f"{filename_prefix}_{regime}_yield.pdf")


def run(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    seeds_per_size: int = 6,
    bins_per_decade: int = 10,
    parallel_xmin: bool = False,
    parallel_uncertainty: bool = False,
    uncertainty_accuracy: float = 0.1,
    force: bool = False,
    population_mode: str = "classified",
):
    if seeds_per_size < 1:
        raise ValueError("seeds_per_size must be positive.")
    if bins_per_decade < 1:
        raise ValueError("bins_per_decade must be positive.")
    if not 0 < uncertainty_accuracy < 0.5:
        raise ValueError("uncertainty_accuracy must lie between 0 and 0.5.")
    if population_mode not in {"classified", "all"}:
        raise ValueError("population_mode must be 'classified' or 'all'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths, inventory = completed_size_scaling_paths(
        data_root, seeds_per_size, REGIMES["post"][1]
    )
    events_by_regime = pool_aligned_events(
        paths,
        REGIMES,
        output_dir / "cache" / "aligned_events",
        force=force,
    )
    classification_rows = []
    fit_rows = []
    summary = {
        "data_root": str(data_root),
        "reconnection": "none",
        "seeds_per_size": seeds_per_size,
        "population_mode": population_mode,
        "selection": (
            "E_R SimpleDrop classification"
            if population_mode == "classified"
            else "all aligned events; no initial E_R split"
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

    for regime in REGIMES:
        events = events_by_regime[regime]
        if population_mode == "classified":
            er_all = {
                size: _positive(event["initial_guess_energy"])
                for size, event in events.items()
            }
            er_xmin_fits = fit_xmins(
                er_all,
                parallel=parallel_xmin,
                cache_dir=output_dir / "cache" / "xmin" / "initial_guess_energy" / regime,
                description=f"E_R {regime}-yield simpleDrop",
                refine=False,
            )
            xmins = {size: float(fit.xmin) for size, fit in er_xmin_fits.items()}
            populations = {
                size: _population_data(event, xmins[size])
                for size, event in events.items()
            }
            _plot_decomposition(regime, populations, output_dir)
            fit_title = r"Irreversible tails after $\Delta E_R$ classification"
            fit_filename_prefix = "irreversible_truncated_powerlaw_fits"
            fit_data_label = "irreversible data"
        else:
            er_xmin_fits = None
            xmins = {}
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
        for size in sorted(events):
            population = populations[size]
            if population_mode == "classified":
                xmin = xmins[size]
                summary["classification"][regime][str(size)] = {
                    "xmin_delta_E_R": xmin,
                    "labeled_count": population["labeled_count"],
                    "reversible_count": population["reversible_count"],
                    "irreversible_count": population["irreversible_count"],
                    "E_S_all_count": int(population["all"].size),
                    "E_S_reversible_count": int(population["reversible"].size),
                    "E_S_irreversible_count": int(population["irreversible"].size),
                }
                classification_rows.append(
                    {
                        "regime": regime,
                        "size": size,
                        "xmin_delta_E_R": xmin,
                        "labeled_count": population["labeled_count"],
                        "reversible_count": population["reversible_count"],
                        "irreversible_count": population["irreversible_count"],
                        "E_S_all_count": population["all"].size,
                        "E_S_reversible_count": population["reversible"].size,
                        "E_S_irreversible_count": population["irreversible"].size,
                    }
                )
                er_data = population["e_r_irreversible"]
                es_data = population["e_s_irreversible"]
                classification_xmin = xmin
            else:
                er_data = population["e_r_all"]
                es_data = population["all"]
                classification_xmin = None
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
                        "classification_xmin_delta_E_R": classification_xmin,
                        "population_mode": population_mode,
                        **{
                            key: value
                            for key, value in record.items()
                            if key not in {"fit", "data"}
                        },
                    }
                )
            _save_ks_plot(
                er_fit,
                output_dir / "D_plots" / regime / f"L{size}_delta_E_R_xmin_search.pdf",
            )
            if population_mode == "classified":
                _save_ks_plot(
                    er_xmin_fits[size],
                    output_dir / "D_plots" / regime / f"L{size}_delta_E_R_classification_simpleDrop_xmin_search.pdf",
                )
            es_diagnostic_name = (
                "irreversible" if population_mode == "classified" else "all_events"
            )
            _save_ks_plot(
                es_fit,
                output_dir / "D_plots" / regime / f"L{size}_delta_E_S_{es_diagnostic_name}_xmin_search.pdf",
            )
        _plot_fit_grid(
            regime,
            fit_results,
            output_dir,
            fit_title,
            fit_filename_prefix,
            fit_data_label,
        )
        _plot_ks_grid(
            regime,
            fit_results,
            output_dir,
            fit_title,
            f"{fit_filename_prefix}_D_simpleDrop_vs_global_min",
        )

    if classification_rows:
        pd.DataFrame(classification_rows).to_csv(
            output_dir / "classification_summary.csv", index=False
        )
    pd.DataFrame(fit_rows).to_csv(output_dir / "fit_results.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds-per-size", type=int, default=6)
    parser.add_argument("--bins-per-decade", type=int, default=10)
    parser.add_argument("--parallel-xmin", action="store_true")
    parser.add_argument("--parallel-uncertainty", action="store_true")
    parser.add_argument("--uncertainty-accuracy", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--population-mode", choices=("classified", "all"), default="classified")
    args = parser.parse_args()
    run(
        args.data_root,
        args.output_dir,
        seeds_per_size=args.seeds_per_size,
        bins_per_decade=args.bins_per_decade,
        parallel_xmin=args.parallel_xmin,
        parallel_uncertainty=args.parallel_uncertainty,
        uncertainty_accuracy=args.uncertainty_accuracy,
        force=args.force,
        population_mode=args.population_mode,
    )


if __name__ == "__main__":
    main()
