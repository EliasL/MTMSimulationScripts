"""Benchmark reversible-population classifiers using forward-only quantities.

The candidate pool is defined without reversibility columns: all steps with a
positive stress-corrected Delta E_S are included. Reversibility data are used
only after the candidate Otsu cut has been found, to benchmark the cut on
matched Sylvain runs.

The size-scaling CSVs have no reference labels, so their output contains only
forward-only Otsu cuts and population sizes, not classification accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Management.updateCSV import update_df_header
from Plotting import numericalParameterJustification as npj
from Plotting import reversibleOnlyEnergyAnalysis as reversible_analysis
from Plotting.energyDropCalculations import (
    SIGMA12_RESCUE_SENTINEL,
    calculate_energy_step_data,
    calculate_stress_step_data,
    validate_sigma12_column,
)
from Plotting.sizeScalingCollapse import (
    REGIMES,
    _has_header_transition,
    _read_mixed_selected,
    completed_size_scaling_paths,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Plots/reversible_event_analysis/forward_only_deployable"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DPI = 250
MIN_CLASS_FRACTION = 0.02


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    getter: Callable[[object], np.ndarray]


def _abs(value):
    return np.abs(np.asarray(value, dtype=float))


def _dg(sample):
    return np.full_like(sample.energy_drop_density, float(sample.load_increment))


def _ratio(numerator, denominator):
    numerator = _abs(numerator)
    denominator = _abs(denominator)
    positive = denominator[denominator > 0]
    if positive.size == 0:
        raise ValueError("Cannot construct a ratio with an all-zero denominator.")
    floor = float(np.min(positive) / 2.0)
    return numerator / np.where(denominator > 0, denominator, floor)


def _sum_ratio(left, right):
    return _ratio(left, _abs(left) + _abs(right))


def _er(sample):
    return _abs(sample.relaxation_energy_density)


def _es(sample):
    return np.asarray(sample.energy_drop_density, dtype=float)


def _ei(sample):
    return _abs(sample.inter_strain_energy_density)


def _sigma_r(sample):
    return _abs(sample.relaxation_stress_drop)


def _sigma_i(sample):
    return _abs(sample.inter_strain_stress_drop)


def _sigma_s(sample):
    return _abs(sample.stress_drop)


FEATURES = (
    FeatureSpec("er", "abs(Delta E_R)/V0", _er),
    FeatureSpec("es", "Delta E_S/V0", _es),
    FeatureSpec("ei", "abs(Delta E_I)/V0", _ei),
    FeatureSpec("sigma_r", "abs(Delta sigma_R)", _sigma_r),
    FeatureSpec("sigma_i", "abs(Delta sigma_I)", _sigma_i),
    FeatureSpec("sigma_s", "abs(Delta sigma_S)", _sigma_s),
    FeatureSpec("er_over_dg", "abs(Delta E_R)/(V0 Delta gamma)",
                lambda s: _ratio(_er(s), _dg(s))),
    FeatureSpec("er_over_dg2", "abs(Delta E_R)/(V0 Delta gamma^2)",
                lambda s: _ratio(_er(s), _dg(s) ** 2)),
    FeatureSpec("es_over_dg", "Delta E_S/(V0 Delta gamma)",
                lambda s: _ratio(_es(s), _dg(s))),
    FeatureSpec("es_over_dg2", "Delta E_S/(V0 Delta gamma^2)",
                lambda s: _ratio(_es(s), _dg(s) ** 2)),
    FeatureSpec("ei_over_dg", "abs(Delta E_I)/(V0 Delta gamma)",
                lambda s: _ratio(_ei(s), _dg(s))),
    FeatureSpec("ei_over_dg2", "abs(Delta E_I)/(V0 Delta gamma^2)",
                lambda s: _ratio(_ei(s), _dg(s) ** 2)),
    FeatureSpec("sigma_r_over_dg", "abs(Delta sigma_R)/Delta gamma",
                lambda s: _ratio(_sigma_r(s), _dg(s))),
    FeatureSpec("sigma_i_over_dg", "abs(Delta sigma_I)/Delta gamma",
                lambda s: _ratio(_sigma_i(s), _dg(s))),
    FeatureSpec("sigma_s_over_dg", "abs(Delta sigma_S)/Delta gamma",
                lambda s: _ratio(_sigma_s(s), _dg(s))),
    FeatureSpec("er_over_es", "abs(Delta E_R)/Delta E_S",
                lambda s: _ratio(_er(s), _es(s))),
    FeatureSpec("ei_over_es", "abs(Delta E_I)/Delta E_S",
                lambda s: _ratio(_ei(s), _es(s))),
    FeatureSpec("sigma_r_over_sigma_s", "abs(Delta sigma_R)/abs(Delta sigma_S)",
                lambda s: _ratio(_sigma_r(s), _sigma_s(s))),
    FeatureSpec("sigma_i_over_sigma_s", "abs(Delta sigma_I)/abs(Delta sigma_S)",
                lambda s: _ratio(_sigma_i(s), _sigma_s(s))),
    FeatureSpec("sigma_r_over_sigma_i", "abs(Delta sigma_R)/abs(Delta sigma_I)",
                lambda s: _ratio(_sigma_r(s), _sigma_i(s))),
    FeatureSpec("er_over_ei", "abs(Delta E_R)/abs(Delta E_I)",
                lambda s: _ratio(_er(s), _ei(s))),
    FeatureSpec("er_fraction", "abs(Delta E_R)/(abs(Delta E_R)+abs(Delta E_I))",
                lambda s: _sum_ratio(_er(s), _ei(s))),
    FeatureSpec("sigma_r_fraction",
                "abs(Delta sigma_R)/(abs(Delta sigma_R)+abs(Delta sigma_I))",
                lambda s: _sum_ratio(_sigma_r(s), _sigma_i(s))),
    FeatureSpec("er_plus_ei", "(abs(Delta E_R)+abs(Delta E_I))/V0",
                lambda s: _er(s) + _ei(s)),
    FeatureSpec("er_minus_ei", "abs(abs(Delta E_R)-abs(Delta E_I))/V0",
                lambda s: _abs(_er(s) - _ei(s))),
    FeatureSpec("er_plus_es", "(abs(Delta E_R)+Delta E_S)/V0",
                lambda s: _er(s) + _es(s)),
    FeatureSpec("er_sigma_r", "abs(Delta E_R) abs(Delta sigma_R)/V0",
                lambda s: _er(s) * _sigma_r(s)),
    FeatureSpec("er_sigma_s", "abs(Delta E_R) abs(Delta sigma_S)/V0",
                lambda s: _er(s) * _sigma_s(s)),
    FeatureSpec("ei_sigma_i", "abs(Delta E_I) abs(Delta sigma_I)/V0",
                lambda s: _ei(s) * _sigma_i(s)),
    FeatureSpec("es_sigma_s", "Delta E_S abs(Delta sigma_S)/V0",
                lambda s: _es(s) * _sigma_s(s)),
    FeatureSpec("er_over_sigma_r2", "abs(Delta E_R)/(V0 abs(Delta sigma_R)^2)",
                lambda s: _ratio(_er(s), _sigma_r(s) ** 2)),
    FeatureSpec("es_over_sigma_s2", "Delta E_S/(V0 abs(Delta sigma_S)^2)",
                lambda s: _ratio(_es(s), _sigma_s(s) ** 2)),
    FeatureSpec("sigma_r2_over_er", "abs(Delta sigma_R)^2/(abs(Delta E_R)/V0)",
                lambda s: _ratio(_sigma_r(s) ** 2, _er(s))),
    FeatureSpec("sigma_s2_over_es", "abs(Delta sigma_S)^2/(Delta E_S/V0)",
                lambda s: _ratio(_sigma_s(s) ** 2, _es(s))),
)


def _forward_event_mask(sample):
    values = np.asarray(sample.energy_drop_density, dtype=float)
    return np.isfinite(values) & (values > 0)


def _pooled_feature(samples, feature, masks):
    values = np.concatenate([
        np.asarray(feature.getter(sample), dtype=float)[mask]
        for sample, mask in zip(samples, masks)
    ])
    if values.size == 0:
        raise RuntimeError(f"Feature {feature.name} has no forward-only events.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Feature {feature.name} contains non-finite values.")
    if np.any(values < 0):
        raise ValueError(f"Feature {feature.name} contains negative values.")
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError(f"Feature {feature.name} has no positive values.")
    return np.where(values == 0, float(np.min(positive) / 2.0), values)


def _reference_categories(setting_samples, classification):
    """Return 0=reversible, 1=irreversible, 2=discarded, 3=unlabeled."""
    categories = []
    for sample in setting_samples:
        pool = _forward_event_mask(sample)
        category = np.full(pool.shape, 3, dtype=np.int8)
        reversible = classification.final_masks[sample.path] & pool
        irreversible = classification.nonclosing_masks[sample.path] & pool
        discarded = classification.discarded_masks[sample.path] & pool
        if np.any(
            (reversible & irreversible)
            | (reversible & discarded)
            | (irreversible & discarded)
        ):
            raise RuntimeError("Reference categories overlap.")
        category[reversible] = 0
        category[irreversible] = 1
        category[discarded] = 2
        categories.append(category[pool])
    return np.concatenate(categories)


def _metrics(true_reversible, predicted_reversible):
    true_reversible = np.asarray(true_reversible, dtype=bool)
    predicted_reversible = np.asarray(predicted_reversible, dtype=bool)
    tp = int(np.count_nonzero(true_reversible & predicted_reversible))
    tn = int(np.count_nonzero(~true_reversible & ~predicted_reversible))
    fp = int(np.count_nonzero(~true_reversible & predicted_reversible))
    fn = int(np.count_nonzero(true_reversible & ~predicted_reversible))
    rev_recall = tp / (tp + fn) if tp + fn else 0.0
    irr_recall = tn / (tn + fp) if tn + fp else 0.0
    return {
        "agreement_fraction": (tp + tn) / true_reversible.size,
        "balanced_accuracy": (rev_recall + irr_recall) / 2.0,
        "reversible_recall": rev_recall,
        "irreversible_recall": irr_recall,
    }


def _evaluate_feature(feature, pool_values, categories):
    try:
        cut, details = npj.unbinned_log_otsu_cut(
            pool_values,
            min_class_fraction=MIN_CLASS_FRACTION,
        )
    except ValueError as error:
        if "No valid distinct two-population split candidates" not in str(error):
            raise
        return {
            "feature": feature.name,
            "feature_label": feature.label,
            "otsu_cut": np.nan,
            "otsu_closing_count": np.nan,
            "otsu_nonclosing_count": np.nan,
            "log10_gap_at_cut": np.nan,
            "benchmark_best_orientation": "no_valid_cut",
            "lower_balanced_accuracy": np.nan,
            "higher_balanced_accuracy": np.nan,
            "best_balanced_accuracy": np.nan,
            "lower_agreement_fraction": np.nan,
            "higher_agreement_fraction": np.nan,
            "best_agreement_fraction": np.nan,
            "status": "no_valid_cut",
        }
    scorable = categories < 2
    if not np.any(scorable) or np.all(categories[scorable] == 0):
        raise RuntimeError(f"Feature {feature.name} lacks both reference classes.")
    values = pool_values[scorable]
    true_reversible = categories[scorable] == 0
    lower = _metrics(true_reversible, values <= cut)
    higher = _metrics(true_reversible, values > cut)
    best_orientation = (
        "higher"
        if (higher["balanced_accuracy"], higher["agreement_fraction"])
        > (lower["balanced_accuracy"], lower["agreement_fraction"])
        else "lower"
    )
    best = higher if best_orientation == "higher" else lower
    return {
        "feature": feature.name,
        "feature_label": feature.label,
        "otsu_cut": cut,
        "otsu_closing_count": details["closing_count"],
        "otsu_nonclosing_count": details["nonclosing_count"],
        "log10_gap_at_cut": details["log10_gap_at_cut"],
        "benchmark_best_orientation": best_orientation,
        "lower_balanced_accuracy": lower["balanced_accuracy"],
        "higher_balanced_accuracy": higher["balanced_accuracy"],
        "best_balanced_accuracy": best["balanced_accuracy"],
        "lower_agreement_fraction": lower["agreement_fraction"],
        "higher_agreement_fraction": higher["agreement_fraction"],
        "best_agreement_fraction": best["agreement_fraction"],
        "status": "ok",
    }


def _save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png_path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(png_path, dpi=FIGURE_DPI)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"Saved {png_path}", flush=True)


def _plot_performance(results, sweep):
    summary = (
        results.groupby(["feature", "feature_label"], as_index=False)
        .agg(
            mean_best_balanced_accuracy=("best_balanced_accuracy", "mean"),
            min_best_balanced_accuracy=("best_balanced_accuracy", "min"),
            mean_lower_balanced_accuracy=("lower_balanced_accuracy", "mean"),
            mean_higher_balanced_accuracy=("higher_balanced_accuracy", "mean"),
        )
        .dropna(subset=["mean_best_balanced_accuracy"])
        .sort_values("mean_best_balanced_accuracy", ascending=True)
        .tail(20)
    )
    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    positions = np.arange(len(summary))
    ax.barh(
        positions,
        summary["mean_best_balanced_accuracy"],
        color="C0",
        alpha=0.85,
        label="mean; best orientation (benchmark)",
    )
    ax.scatter(
        summary["min_best_balanced_accuracy"],
        positions,
        color="black",
        marker="x",
        s=28,
        label="minimum across settings",
        zorder=3,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(summary["feature_label"])
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("balanced accuracy against reference labels")
    ax.set_title(
        f"Forward-only candidate benchmark; {sweep}; "
        "Otsu fit uses the full positive-Delta E_S pool"
    )
    ax.legend(loc="lower right", fontsize="small")
    _save(fig, f"candidate_performance_{sweep}")
    summary.to_csv(TABLE_DIR / f"candidate_summary_{sweep}.csv", index=False)


def _setting_label(attribute, value):
    return f"{attribute}={value:.0e}"


def _plot_distributions(
    results, feature_values, categories_by_setting, setting_attribute, sweep
):
    top_features = (
        results.groupby("feature")["best_balanced_accuracy"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    settings = sorted(results["setting"].unique())
    fig, axes = plt.subplots(
        len(settings),
        len(top_features),
        figsize=(10.0, 2.5 * len(settings)),
        squeeze=False,
    )
    colors = {0: "C0", 1: "C3", 2: "C2", 3: "0.55"}
    labels = {
        0: "reference reversible",
        1: "reference irreversible",
        2: "discarded island",
        3: "unlabeled forward row",
    }
    for row, setting in enumerate(settings):
        categories = categories_by_setting[setting]
        setting_results = results[results["setting"] == setting].set_index("feature")
        for col, feature_name in enumerate(top_features):
            ax = axes[row, col]
            values = feature_values[(setting, feature_name)]
            log_values = np.log10(values)
            low, high = float(log_values.min()), float(log_values.max())
            if np.isclose(low, high):
                low -= 0.5
                high += 0.5
            edges = np.linspace(low, high, 42)
            total = float(values.size)
            for category in range(4):
                counts, _ = np.histogram(
                    log_values[categories == category], bins=edges
                )
                if np.any(counts):
                    ax.stairs(
                        counts / total,
                        edges,
                        color=colors[category],
                        linewidth=1.2,
                        label=labels[category],
                    )
            result = setting_results.loc[feature_name]
            if np.isfinite(result["otsu_cut"]):
                ax.axvline(
                    np.log10(result["otsu_cut"]),
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    label="Otsu cut",
                )
            ax.set_title(
                f"{_setting_label(setting_attribute, setting)}; "
                f"{result['feature_label']}; "
                f"best BA={result['best_balanced_accuracy']:.3f}",
                fontsize="small",
            )
            ax.set_xlabel("log10(forward-only quantity)")
            ax.set_ylabel("P(bin)")
    axes[0, 0].legend(loc="best", fontsize="x-small")
    fig.suptitle(
        "Deployable-pool audit: Otsu is fit before applying reference labels",
        fontsize="medium",
    )
    _save(fig, f"candidate_distributions_{sweep}")


def _benchmark_sweep(batch, attribute, sweep):
    samples = npj.load_batch(batch)
    classifications = reversible_analysis.build_classifications(samples, attribute)
    groups = npj._setting_groups(samples, attribute)
    rows = []
    feature_values = {}
    categories_by_setting = {}

    for setting, setting_samples in groups.items():
        categories = _reference_categories(
            setting_samples, classifications[setting]
        )
        masks = [_forward_event_mask(sample) for sample in setting_samples]
        pool_count = int(categories.size)
        scorable_count = int(np.count_nonzero(categories < 2))
        categories_by_setting[setting] = categories
        for feature in FEATURES:
            values = _pooled_feature(setting_samples, feature, masks)
            if values.size != pool_count:
                raise RuntimeError(
                    f"Feature {feature.name} has {values.size} values, "
                    f"expected {pool_count}."
                )
            feature_values[(setting, feature.name)] = values
            row = _evaluate_feature(feature, values, categories)
            row.update(
                {
                    "setting": setting,
                    "setting_attribute": attribute,
                    "sample_count": len(setting_samples),
                    "forward_only_pool_count": pool_count,
                    "scorable_count": scorable_count,
                    "reference_reversible_count": int(np.count_nonzero(categories == 0)),
                    "reference_irreversible_count": int(np.count_nonzero(categories == 1)),
                    "discarded_island_count": int(np.count_nonzero(categories == 2)),
                    "unlabeled_forward_count": int(np.count_nonzero(categories == 3)),
                }
            )
            rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        ["setting", "best_balanced_accuracy"],
        ascending=[True, False],
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLE_DIR / f"candidate_results_{sweep}.csv", index=False)
    _plot_performance(results, sweep)
    _plot_distributions(
        results,
        feature_values,
        categories_by_setting,
        attribute,
        sweep,
    )
    return results


def _read_size_scaling_run(path, size):
    wanted = {
        "load",
        "total_energy",
        "total_energy_change",
        "total_e_change_from_init",
        "avg_sigma12",
        "avg_init_sigma12",
        "avg_sigma12_change_from_init",
    }
    if _has_header_transition(path):
        df = _read_mixed_selected(path, wanted)
    else:
        raw_wanted = wanted | {
            "avg_sigmaxy",
            "avg_init_sigmaxy",
            "avg_Pxy",
        }
        df = pd.read_csv(
            path,
            usecols=lambda column: column in raw_wanted,
            low_memory=False,
        )
        df = update_df_header(df, add_total_columns=False, L=size)
    if "avg_sigma12_change_from_init" not in df:
        required_stress = {"avg_sigma12", "avg_init_sigma12"}
        if not required_stress.issubset(df.columns):
            raise KeyError(
                f"Cannot reconstruct relaxation stress in {path}; "
                f"missing {sorted(required_stress - set(df.columns))}."
            )
        df["avg_sigma12_change_from_init"] = (
            df["avg_sigma12"] - df["avg_init_sigma12"]
        )
    required = {
        "load",
        "total_energy",
        "total_energy_change",
        "total_e_change_from_init",
        "avg_sigma12",
        "avg_init_sigma12",
        "avg_sigma12_change_from_init",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing size-scaling columns in {path}: {missing}")
    validate_sigma12_column(df, context=str(path))
    load = df["load"].to_numpy(dtype=float)
    if load.size < 2 or not np.all(np.isfinite(load)):
        raise ValueError(f"Invalid load column in {path}.")
    if not np.all(np.diff(load) > 0):
        raise ValueError(f"Load is not strictly increasing in {path}.")
    energy_steps, _ = calculate_energy_step_data(
        path,
        df=df,
        metadata={"L": size},
        average_energy=False,
    )
    stress_steps, _ = calculate_stress_step_data(path, df=df)
    volume = float(size * size)
    invalid_rows = (
        (df["avg_sigma12"].to_numpy(dtype=float) == SIGMA12_RESCUE_SENTINEL)
        | (df["avg_init_sigma12"].to_numpy(dtype=float) == SIGMA12_RESCUE_SENTINEL)
        | (
            df["avg_sigma12_change_from_init"].to_numpy(dtype=float)
            == SIGMA12_RESCUE_SENTINEL
        )
    )
    invalid_steps = invalid_rows[1:]
    relaxation_energy_density = (
        -df["total_e_change_from_init"].to_numpy(dtype=float)[1:] / volume
    )
    inter_strain_energy_density = (
        -df["total_energy_change"].to_numpy(dtype=float)[1:] / volume
    )
    relaxation_stress_drop = -df["avg_sigma12_change_from_init"].to_numpy(
        dtype=float
    )[1:]
    for values in (
        relaxation_energy_density,
        inter_strain_energy_density,
        relaxation_stress_drop,
    ):
        values[invalid_steps] = np.nan
    return SimpleNamespace(
        volume=volume,
        load_increment=float(np.median(np.diff(load))),
        gamma=load[1:],
        energy_drop_density=(
            energy_steps["stress_corrected_drop_second_order"].to_numpy(dtype=float)
            / volume
        ),
        relaxation_energy_density=relaxation_energy_density,
        inter_strain_energy_density=inter_strain_energy_density,
        relaxation_stress_drop=relaxation_stress_drop,
        inter_strain_stress_drop=stress_steps["inter_strain_drop"].to_numpy(dtype=float),
        stress_drop=stress_steps["stress_corrected_drop"].to_numpy(dtype=float),
    )


def _size_scaling_application(data_root, seeds_per_size=10):
    paths_by_size, inventory = completed_size_scaling_paths(
        data_root,
        seeds_per_size,
        REGIMES["post"][1],
    )
    rows = []
    top_features = [feature.name for feature in FEATURES[:3]]
    cuts_for_plot = []
    for size, paths in paths_by_size.items():
        runs = [_read_size_scaling_run(path, size) for path in paths]
        for regime, (low, high) in REGIMES.items():
            masks = [
                _forward_event_mask(run)
                & (run.gamma > low)
                & (run.gamma < high)
                for run in runs
            ]
            for feature in FEATURES:
                values = _pooled_feature(runs, feature, masks)
                try:
                    cut, details = npj.unbinned_log_otsu_cut(
                        values,
                        min_class_fraction=MIN_CLASS_FRACTION,
                    )
                    status = "ok"
                except ValueError as error:
                    if "No valid distinct two-population split candidates" not in str(error):
                        raise
                    cut = np.nan
                    details = {
                        "closing_count": np.nan,
                        "nonclosing_count": np.nan,
                        "log10_gap_at_cut": np.nan,
                    }
                    status = "no_valid_cut"
                row = {
                    "size": size,
                    "regime": regime,
                    "feature": feature.name,
                    "feature_label": feature.label,
                    "sample_count": len(paths),
                    "forward_only_pool_count": values.size,
                    "otsu_cut": cut,
                    "otsu_closing_count": details["closing_count"],
                    "otsu_nonclosing_count": details["nonclosing_count"],
                    "log10_gap_at_cut": details["log10_gap_at_cut"],
                    "status": status,
                }
                rows.append(row)
                if feature.name in top_features:
                    cuts_for_plot.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(TABLE_DIR / "size_scaling_candidate_cuts.csv", index=False)
    _plot_size_scaling_cuts(pd.DataFrame(cuts_for_plot))
    pd.DataFrame(
        [{"size": size, "available_runs": count}
         for size, count in inventory.items()]
    ).to_csv(TABLE_DIR / "size_scaling_inventory.csv", index=False)
    return results


def _plot_size_scaling_cuts(results):
    features = list(results["feature"].unique())
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(11.0, 4.2), squeeze=False)
    for column, regime in enumerate(REGIMES):
        ax = axes[0, column]
        subset = results[results["regime"] == regime]
        for feature in features:
            rows = subset[subset["feature"] == feature].sort_values("size")
            ax.plot(
                rows["size"],
                rows["otsu_cut"],
                marker="o",
                label=rows["feature_label"].iloc[0],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("L")
        ax.set_ylabel("forward-only Otsu cut")
        ax.set_title("Pre-yield" if regime == "pre" else "Post-yield")
        ax.legend(fontsize="small")
    fig.suptitle("Unsupervised candidate cuts on L=50--250 size-scaling CSVs")
    _save(fig, "size_scaling_candidate_cuts")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    epsilon_results = _benchmark_sweep(-2, "eps_x", "epsilon_x")
    delta_gamma_results = _benchmark_sweep(
        -1, "load_increment", "delta_gamma"
    )
    for name, results in (
        ("epsilon_x", epsilon_results),
        ("delta_gamma", delta_gamma_results),
    ):
        summary = (
            results.groupby(["feature", "feature_label"], as_index=False)
            ["best_balanced_accuracy"]
            .mean()
            .sort_values("best_balanced_accuracy", ascending=False)
            .head(15)
        )
        print()
        print(f"Benchmark summary: {name}")
        print(summary.to_string(index=False))
    size_results = _size_scaling_application(
        Path("/Volumes/data/remoteData/macro"),
        seeds_per_size=10,
    )
    print()
    print(
        f"Size-scaling application: {len(size_results)} unsupervised cut rows "
        "written; no reference accuracy is available."
    )


if __name__ == "__main__":
    main()
