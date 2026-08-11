"""Find forward-only Otsu splits that reproduce the reversibility labels.

The reference labels come from the Sylvain reversibility batches, but all
candidate features are available from a normal loading simulation.  For each
epsilon_x or Delta gamma setting, the four matching samples are pooled before
the forward-only Otsu split is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Plotting import numericalParameterJustification as npj
from Plotting import reversibleOnlyEnergyAnalysis as reversible_analysis


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT
    / "Plots/reversible_event_analysis/reversible_only/forward_only_split_all_settings"
)
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DPI = 250


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    getter: Callable[[npj.SampleData], np.ndarray]


def _save(fig: mpl.figure.Figure, name: str) -> None:
    fig.tight_layout()
    png_path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(png_path, dpi=FIGURE_DPI)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"Saved {png_path}")


def _field(sample: npj.SampleData, name: str) -> np.ndarray:
    return np.asarray(getattr(sample, name), dtype=float)


def _energy_density(sample: npj.SampleData) -> np.ndarray:
    return sample.energy_drop_density


def _stress_magnitude(sample: npj.SampleData) -> np.ndarray:
    return np.abs(sample.stress_drop)


def _relaxation_energy_magnitude(sample: npj.SampleData) -> np.ndarray:
    return np.abs(sample.relaxation_energy_density)


def _relaxation_stress_magnitude(sample: npj.SampleData) -> np.ndarray:
    return np.abs(sample.relaxation_stress_drop)


FEATURES = (
    FeatureSpec(
        "relaxation_energy_magnitude",
        r"$|\Delta E_R|/V_0$",
        _relaxation_energy_magnitude,
    ),
    FeatureSpec("energy_drop_density", r"$\Delta E_S/V_0$", _energy_density),
    FeatureSpec(
        "relaxation_stress_magnitude",
        r"$|\Delta\sigma_R|$",
        _relaxation_stress_magnitude,
    ),
)


def _selected_pool(
    samples: list[npj.SampleData],
    classifications: dict[float, reversible_analysis.SettingClassification],
) -> tuple[float, list[npj.SampleData], np.ndarray, np.ndarray]:
    groups = npj._setting_groups(samples, "load_increment")
    selected = [
        (setting, setting_samples)
        for setting, setting_samples in groups.items()
        if np.isclose(setting, npj.USED_DELTA_GAMMA)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one selected Delta gamma setting, got {selected}.")
    setting, setting_samples = selected[0]
    if not all(np.isclose(sample.eps_x, npj.USED_EPS_X) for sample in setting_samples):
        raise RuntimeError("The selected Delta gamma samples do not use epsilon_x=1e-6.")

    classification = classifications[setting]
    reversible_indices = []
    irreversible_indices = []
    for sample in setting_samples:
        reversible_indices.append(classification.final_masks[sample.path])
        irreversible_indices.append(classification.nonclosing_masks[sample.path])
    reversible_mask = np.concatenate(reversible_indices)
    irreversible_mask = np.concatenate(irreversible_indices)
    if np.any(reversible_mask & irreversible_mask):
        raise RuntimeError("Reference reversible and irreversible masks overlap.")
    event_pool = reversible_mask | irreversible_mask
    if int(event_pool.sum()) != classifications[setting].recorded_count:
        raise RuntimeError("Reference masks do not match the recorded event count.")
    return (
        setting,
        setting_samples,
        reversible_mask[event_pool],
        irreversible_mask[event_pool],
    )


def _pooled_feature(
    setting_samples: list[npj.SampleData],
    classification: reversible_analysis.SettingClassification,
    feature: FeatureSpec,
) -> np.ndarray:
    values = []
    for sample in setting_samples:
        mask = (
            classification.final_masks[sample.path]
            | classification.nonclosing_masks[sample.path]
        )
        values.append(np.asarray(feature.getter(sample), dtype=float)[mask])
    pooled = np.concatenate(values)
    if not np.all(np.isfinite(pooled)):
        raise ValueError(f"Feature {feature.name} contains non-finite values.")
    if np.any(pooled < 0):
        raise ValueError(f"Feature {feature.name} contains negative values.")
    positive = pooled[pooled > 0]
    if positive.size == 0:
        raise ValueError(f"Feature {feature.name} has no positive values.")
    zero_floor = float(np.min(positive) / 2.0)
    return np.where(pooled == 0, zero_floor, pooled)


def _reference_pool(
    setting_samples: list[npj.SampleData],
    classification: reversible_analysis.SettingClassification,
) -> tuple[np.ndarray, int]:
    """Return reference labels after excluding deliberately discarded islands."""
    labels = []
    irreversible_count = 0
    for sample in setting_samples:
        reversible_mask = classification.final_masks[sample.path]
        irreversible_mask = classification.nonclosing_masks[sample.path]
        event_mask = reversible_mask | irreversible_mask
        if np.any(reversible_mask & irreversible_mask):
            raise RuntimeError("Reference reversible and irreversible masks overlap.")
        labels.append(reversible_mask[event_mask])
        irreversible_count += int(np.count_nonzero(irreversible_mask))
    return np.concatenate(labels), irreversible_count


def _setting_feature_values(
    setting_samples: list[npj.SampleData],
    classification: reversible_analysis.SettingClassification,
    feature: FeatureSpec,
) -> tuple[np.ndarray, np.ndarray, int]:
    true_reversible, _ = _reference_pool(setting_samples, classification)
    values = _pooled_feature(setting_samples, classification, feature)
    if values.size != true_reversible.size:
        raise RuntimeError(
            f"Feature {feature.name} has {values.size} values, expected "
            f"{true_reversible.size}."
        )
    discarded_count = classification.discarded_count
    return values, true_reversible, discarded_count


def _metrics(true_reversible: np.ndarray, predicted_reversible: np.ndarray) -> dict:
    true_reversible = np.asarray(true_reversible, dtype=bool)
    predicted_reversible = np.asarray(predicted_reversible, dtype=bool)
    tp = int(np.count_nonzero(true_reversible & predicted_reversible))
    tn = int(np.count_nonzero(~true_reversible & ~predicted_reversible))
    fp = int(np.count_nonzero(~true_reversible & predicted_reversible))
    fn = int(np.count_nonzero(true_reversible & ~predicted_reversible))
    reversible_precision = tp / (tp + fp) if tp + fp else 0.0
    reversible_recall = tp / (tp + fn) if tp + fn else 0.0
    irreversible_precision = tn / (tn + fn) if tn + fn else 0.0
    irreversible_recall = tn / (tn + fp) if tn + fp else 0.0
    reversible_f1 = (
        2.0 * reversible_precision * reversible_recall
        / (reversible_precision + reversible_recall)
        if reversible_precision + reversible_recall
        else 0.0
    )
    irreversible_f1 = (
        2.0 * irreversible_precision * irreversible_recall
        / (irreversible_precision + irreversible_recall)
        if irreversible_precision + irreversible_recall
        else 0.0
    )
    return {
        "agreement_fraction": (tp + tn) / true_reversible.size,
        "balanced_accuracy": (reversible_recall + irreversible_recall) / 2.0,
        "macro_f1": (reversible_f1 + irreversible_f1) / 2.0,
        "reversible_precision": reversible_precision,
        "reversible_recall": reversible_recall,
        "irreversible_precision": irreversible_precision,
        "irreversible_recall": irreversible_recall,
        "true_reversible_count": int(np.count_nonzero(true_reversible)),
        "true_irreversible_count": int(np.count_nonzero(~true_reversible)),
        "predicted_reversible_count": int(np.count_nonzero(predicted_reversible)),
        "predicted_irreversible_count": int(np.count_nonzero(~predicted_reversible)),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _evaluate_feature(
    feature: FeatureSpec,
    values: np.ndarray,
    true_reversible: np.ndarray,
) -> tuple[dict, np.ndarray, float]:
    cut, details = npj.unbinned_log_otsu_cut(values)
    lower_reversible = _metrics(true_reversible, values <= cut)
    upper_reversible = _metrics(true_reversible, values > cut)
    if (
        upper_reversible["balanced_accuracy"],
        upper_reversible["macro_f1"],
    ) > (
        lower_reversible["balanced_accuracy"],
        lower_reversible["macro_f1"],
    ):
        selected_metrics = upper_reversible
        reversible_side = "higher"
        predicted_reversible = values > cut
    else:
        selected_metrics = lower_reversible
        reversible_side = "lower"
        predicted_reversible = values <= cut
    row = {
        "feature": feature.name,
        "feature_label": feature.label,
        "otsu_cut": cut,
        "reversible_side": reversible_side,
        "otsu_closing_count": details["closing_count"],
        "otsu_nonclosing_count": details["nonclosing_count"],
        "log10_gap_at_cut": details["log10_gap_at_cut"],
        **selected_metrics,
    }
    row["candidate_reversible_mask"] = predicted_reversible
    return row, predicted_reversible, cut


def _plot_feature_distributions(
    results: list[dict],
    feature_values: dict[str, np.ndarray],
    true_reversible: np.ndarray,
) -> None:
    n_columns = 3
    n_rows = int(np.ceil(len(results) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12.0, 3.0 * n_rows),
        squeeze=False,
    )
    for index, result in enumerate(results):
        ax = axes.flat[index]
        values = feature_values[result["feature"]]
        log_values = np.log10(values)
        bins = np.linspace(log_values.min(), log_values.max(), 55)
        ax.hist(
            log_values[true_reversible],
            bins=bins,
            density=True,
            color="C0",
            alpha=0.55,
            label="reference reversible",
        )
        ax.hist(
            log_values[~true_reversible],
            bins=bins,
            density=True,
            color="C3",
            alpha=0.55,
            label="reference irreversible",
        )
        ax.axvline(
            np.log10(result["otsu_cut"]),
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        ax.set_title(
            f"{result['feature_label']}\n"
            f"balanced accuracy={result['balanced_accuracy']:.3f}"
        )
        ax.set_xlabel(r"$\log_{10}$(candidate feature)")
        ax.set_ylabel("density")
    for ax in axes.flat[len(results) :]:
        ax.axis("off")
    axes.flat[0].legend(loc="best", fontsize="small")
    _save(fig, "forward_only_candidate_distributions")


def _plot_ranking(results: pd.DataFrame) -> None:
    ranked = results.sort_values(
        ["balanced_accuracy", "macro_f1"], ascending=[True, True]
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.barh(
        np.arange(len(ranked)),
        ranked["balanced_accuracy"],
        color="C1",
        alpha=0.8,
    )
    ax.set_yticks(np.arange(len(ranked)))
    ax.set_yticklabels(ranked["feature_label"])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel(r"balanced accuracy against $\Delta_{\mathrm{rev}}u$ labels")
    ax.set_ylabel("candidate feature")
    ax.set_title("Forward-only Otsu split ranking")
    for index, (_, row) in enumerate(ranked.iterrows()):
        ax.text(
            row["balanced_accuracy"] + 0.005,
            index,
            f"{row['balanced_accuracy']:.3f}",
            va="center",
            fontsize="small",
        )
    _save(fig, "forward_only_candidate_ranking")


def _setting_axis_label(attribute: str) -> str:
    if attribute == "eps_x":
        return r"$\epsilon_x$"
    if attribute == "load_increment":
        return r"$\Delta\gamma$"
    raise ValueError(f"Unknown setting attribute: {attribute!r}")


def _plot_all_setting_distributions(
    results: pd.DataFrame,
    feature_values: dict[tuple[float, str], np.ndarray],
    labels_by_setting: dict[float, np.ndarray],
    attribute: str,
    output_name: str | None = None,
) -> None:
    settings = sorted(results["setting"].unique())
    n_rows = len(settings)
    fig, axes = plt.subplots(
        n_rows,
        len(FEATURES),
        figsize=(10.0, 2.35 * n_rows),
        squeeze=False,
        sharey=False,
    )
    for row_index, setting in enumerate(settings):
        labels = labels_by_setting[setting]
        setting_results = results[results["setting"] == setting].set_index("feature")
        for column, feature in enumerate(FEATURES):
            ax = axes[row_index, column]
            values = feature_values[(setting, feature.name)]
            log_values = np.log10(values)
            low = float(np.min(log_values))
            high = float(np.max(log_values))
            if np.isclose(low, high):
                low -= 0.5
                high += 0.5
            bins = np.linspace(low, high, 42)
            ax.hist(
                log_values[labels],
                bins=bins,
                density=True,
                color="C0",
                alpha=0.55,
                label="reference reversible",
            )
            ax.hist(
                log_values[~labels],
                bins=bins,
                density=True,
                color="C3",
                alpha=0.55,
                label="reference irreversible",
            )
            result = setting_results.loc[feature.name]
            ax.axvline(
                np.log10(result["otsu_cut"]),
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="Otsu cut",
            )
            setting_text = f"{_setting_axis_label(attribute)}={setting:.0e}"
            ax.set_title(
                f"{setting_text}\n"
                f"{feature.label}; BA={result['balanced_accuracy']:.3f}",
                fontsize="small",
            )
            ax.set_xlabel(r"$\log_{10}$(forward-only quantity)")
            ax.set_ylabel("density")
            ax.text(
                0.03,
                0.96,
                f"n={len(labels)}",
                transform=ax.transAxes,
                va="top",
                fontsize="x-small",
            )
    axes[0, 0].legend(loc="best", fontsize="x-small")
    fig.suptitle(
        f"Forward-only Otsu candidates; pooled samples; labels from "
        r"$\Delta_{\mathrm{rev}}\mathbf{u}$",
        fontsize="medium",
    )
    output_name = attribute if output_name is None else output_name
    _save(fig, f"forward_only_candidate_distributions_{output_name}")


def _plot_accuracy_by_setting(
    results: pd.DataFrame, attribute: str, output_name: str | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {feature.name: f"C{index}" for index, feature in enumerate(FEATURES)}
    for feature in FEATURES:
        rows = results[results["feature"] == feature.name].sort_values("setting")
        ax.plot(
            rows["setting"],
            rows["balanced_accuracy"],
            marker="o",
            linewidth=1.4,
            color=colors[feature.name],
            label=feature.label,
        )
    ax.set_xscale("log")
    ax.set_ylim(0.45, 1.01)
    ax.set_xlabel(_setting_axis_label(attribute))
    ax.set_ylabel(r"balanced accuracy against $\Delta_{\mathrm{rev}}u$ labels")
    ax.set_title("Forward-only candidate performance across settings")
    ax.legend(loc="best", frameon=True)
    output_name = attribute if output_name is None else output_name
    _save(fig, f"forward_only_candidate_accuracy_{output_name}")


def run_all_settings_analysis() -> dict[str, pd.DataFrame]:
    """Evaluate the three requested forward-only quantities over both sweeps."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    analyses = {}
    for batch, attribute, name in (
        (-2, "eps_x", "epsilon_x"),
        (-1, "load_increment", "delta_gamma"),
    ):
        samples = npj.load_batch(batch)
        classifications = reversible_analysis.build_classifications(samples, attribute)
        result_rows = []
        feature_values: dict[tuple[float, str], np.ndarray] = {}
        labels_by_setting: dict[float, np.ndarray] = {}
        groups = npj._setting_groups(samples, attribute)

        for setting, setting_samples in groups.items():
            classification = classifications[setting]
            labels, irreversible_count = _reference_pool(
                setting_samples, classification
            )
            labels_by_setting[setting] = labels
            for feature in FEATURES:
                values, true_reversible, discarded_count = _setting_feature_values(
                    setting_samples, classification, feature
                )
                if not np.array_equal(labels, true_reversible):
                    raise RuntimeError("Feature and reference event ordering differ.")
                feature_values[(setting, feature.name)] = values
                row, _, _ = _evaluate_feature(feature, values, true_reversible)
                row.pop("candidate_reversible_mask")
                row.update(
                    {
                        "setting": setting,
                        "setting_attribute": attribute,
                        "sample_count": len(setting_samples),
                        "event_count": len(labels),
                        "reference_reversible_count": int(labels.sum()),
                        "reference_irreversible_count": irreversible_count,
                        "discarded_island_count": discarded_count,
                        "reference_final_cut": classification.final_cut,
                    }
                )
                result_rows.append(row)

        results = pd.DataFrame(result_rows).sort_values(
            ["setting", "balanced_accuracy", "macro_f1"],
            ascending=[True, False, False],
        )
        results.to_csv(
            TABLE_DIR / f"forward_only_candidate_results_{name}.csv", index=False
        )
        results.groupby("feature", as_index=False).agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            minimum_balanced_accuracy=("balanced_accuracy", "min"),
            maximum_balanced_accuracy=("balanced_accuracy", "max"),
        ).sort_values("mean_balanced_accuracy", ascending=False).to_csv(
            TABLE_DIR / f"forward_only_candidate_summary_{name}.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "setting": setting,
                    "setting_attribute": attribute,
                    "sample_count": len(groups[setting]),
                    "event_count": len(labels_by_setting[setting]),
                    "reference_reversible_count": int(labels_by_setting[setting].sum()),
                    "reference_irreversible_count": int(
                        results.loc[
                            results["setting"] == setting,
                            "reference_irreversible_count",
                        ].iloc[0]
                    ),
                    "discarded_island_count": int(
                        results.loc[
                            results["setting"] == setting,
                            "discarded_island_count",
                        ].iloc[0]
                    ),
                }
                for setting in sorted(groups)
            ]
        ).to_csv(TABLE_DIR / f"forward_only_reference_summary_{name}.csv", index=False)
        _plot_all_setting_distributions(
            results, feature_values, labels_by_setting, attribute, name
        )
        _plot_accuracy_by_setting(results, attribute, name)
        analyses[name] = results
    return analyses


def run_analysis() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    samples = npj.load_batch(-1)
    classifications = reversible_analysis.build_classifications(
        samples, "load_increment"
    )
    setting, setting_samples, reference_reversible, reference_irreversible = (
        _selected_pool(samples, classifications)
    )
    classification = classifications[setting]
    true_reversible = reference_reversible
    feature_values = {}
    result_rows = []
    for feature in FEATURES:
        values = _pooled_feature(setting_samples, classification, feature)
        if values.size != true_reversible.size:
            raise RuntimeError(
                f"Feature {feature.name} has {values.size} values, expected "
                f"{true_reversible.size}."
            )
        feature_values[feature.name] = values
        row, _, _ = _evaluate_feature(feature, values, true_reversible)
        row.pop("candidate_reversible_mask")
        result_rows.append(row)

    results = pd.DataFrame(result_rows)
    results = results.sort_values(
        ["balanced_accuracy", "macro_f1", "reversible_precision"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    results.insert(1, "setting", setting)
    results.insert(2, "epsilon_x", npj.USED_EPS_X)
    results.insert(3, "event_count", true_reversible.size)
    results.to_csv(TABLE_DIR / "forward_only_candidate_ranking.csv", index=False)
    _plot_ranking(results)
    _plot_feature_distributions(
        results.to_dict("records"), feature_values, true_reversible
    )
    pd.DataFrame(
        [
            {
                "delta_gamma": setting,
                "epsilon_x": npj.USED_EPS_X,
                "sample_count": len(setting_samples),
                "event_count": true_reversible.size,
                "reference_reversible_count": int(reference_reversible.sum()),
                "reference_irreversible_count": int(reference_irreversible.sum()),
                "reference_cut": classification.final_cut,
            }
        ]
    ).to_csv(TABLE_DIR / "forward_only_reference_summary.csv", index=False)
    return results


def main() -> None:
    analyses = run_all_settings_analysis()
    for name, results in analyses.items():
        summary = results.groupby("feature", as_index=False)["balanced_accuracy"].mean()
        print(f"\n{name}: mean balanced accuracy")
        print(summary.sort_values("balanced_accuracy", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
