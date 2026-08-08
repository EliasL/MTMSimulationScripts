"""Focused follow-up analysis of reversible Sylvain events.

The existing numerical-parameter loader is reused, but all outputs from this
module are written to a separate directory.  Events are classified from the
setting-wise rev_u cut and the m3-change count; ``is_reversible`` is not used.
Run without arguments::

    .venv/bin/python -m Plotting.reversibleEventAnalysis
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Plotting import numericalParameterJustification as npj


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Plots/reversible_event_analysis"
TABLE_DIR = OUTPUT_DIR / "tables"
QUANTILES = (0.5, 0.99)
FIGURE_DPI = 250
MAX_POINTS = 900
SCATTER_ALPHA = 0.2

CLASS_INFO = (
    ("reversible_elastic", r"reversible elastic", "C0", "o"),
    ("reversible_plastic", r"reversible plastic", "C2", "x"),
    ("irreversible", r"irreversible", "C3", "s"),
)


def _configure_output() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    npj.OUTPUT_DIR = OUTPUT_DIR
    npj.TABLE_DIR = TABLE_DIR
    (OUTPUT_DIR / "analysis_definition.json").write_text(
        json.dumps(
            {
                "classifier": "setting-wise unbinned log-Otsu split of recorded positive Delta E_S cycles in Delta_rev u",
                "reversible_elastic": "recorded reversible cycle and m3 change == 0",
                "reversible_plastic": "recorded reversible cycle and m3 change > 0",
                "irreversible": "event row not in the reversible population",
                "stress_drop": "second-order stress-corrected Delta sigma_S",
                "energy_drop": "second-order stress-corrected Delta E_S divided by V_0",
                "inter_strain_energy_drop": "-total_energy_change divided by V_0; positive means the measured energy decreases between consecutive relaxed states",
                "uncertainty": "standard deviation across samples",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _save(fig: mpl.figure.Figure, name: str) -> Path:
    fig.text(
        0.995,
        0.008,
        r"Classifier: setting-wise unbinned log-Otsu split of "
        r"$\Delta_{\mathrm{rev}}\mathbf{u}$; reversible plastic means "
        r"reversible with $\Delta m_3>0$",
        ha="right",
        va="bottom",
        fontsize="small",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_zorder(20)
            legend.get_frame().set_alpha(1.0)
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def _event_masks(sample: npj.SampleData) -> dict[str, np.ndarray]:
    recorded = npj.real_energy_drop_mask(sample)
    event = recorded
    closing = recorded & ~sample.avalanche
    masks = {
        "reversible_elastic": closing & (sample.m3_changes == 0),
        "reversible_plastic": closing & (sample.m3_changes > 0),
        "irreversible": event & ~closing,
    }
    total = sum(mask.astype(np.int8) for mask in masks.values())
    if np.any(total > 1) or np.any(total < 0) or not np.all(total[event] == 1):
        raise RuntimeError(f"Event classes do not partition the event rows in {sample.path}.")
    return masks


def _groups(samples: list[npj.SampleData], attribute: str):
    return npj._setting_groups(samples, attribute)


def _sampled_indices(mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size <= MAX_POINTS:
        return indices
    positions = np.linspace(0, indices.size - 1, MAX_POINTS).astype(int)
    return indices[np.unique(positions)]


def _class_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="None",
            markersize=6,
            label=label,
        )
        for _, label, color, marker in CLASS_INFO
    ]


def _setting_handles(groups, attribute: str, colors):
    return npj._setting_handles(groups, attribute, colors)


def _used_ring(ax, groups, attribute: str, settings, values) -> None:
    used = [
        i for i, setting in enumerate(settings)
        if npj._is_used_setting(attribute, setting)
    ]
    if len(used) != 1:
        raise ValueError(f"Expected one used {attribute} setting, got {used}.")
    i = used[0]
    if np.isfinite(values[i]):
        npj._add_used_marker_ring(ax, [settings[i]], [values[i]], size=68)


def _positive_values(sample: npj.SampleData, field: str, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(getattr(sample, field), dtype=float)[mask]
    if field != "rev_u":
        values = np.abs(values)
    values = values[np.isfinite(values) & (values > 0)]
    return values


def _write_cuts(samples: list[npj.SampleData], attribute: str, tag: str) -> None:
    rows = []
    for setting, group in _groups(samples, attribute).items():
        cuts = {sample.rev_u_cut for sample in group}
        if len(cuts) != 1:
            raise RuntimeError(f"Inconsistent rev_u cuts at {attribute}={setting}: {cuts}")
        cut = cuts.pop()
        recorded = np.concatenate(
            [sample.rev_u[npj.real_energy_drop_mask(sample)] for sample in group]
        )
        _, details = npj.unbinned_log_otsu_cut(recorded)
        rows.append({"setting": setting, "rev_u_cut": cut, **details})
    pd.DataFrame(rows).to_csv(TABLE_DIR / f"{tag}_rev_u_cuts.csv", index=False)


def plot_reversible_scaling(
    samples: list[npj.SampleData], attribute: str, name: str
) -> None:
    groups = _groups(samples, attribute)
    settings = np.asarray(list(groups), dtype=float)
    colors = npj._colors(len(groups))
    metrics = (
        ("rev_energy_density", r"$|\Delta_{\mathrm{rev}}E|/V_0$"),
        ("rev_sigma", r"$|\Delta_{\mathrm{rev}}\sigma_{12}|$"),
    )
    rows = []
    fit_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex="col")

    for row, post_yield in enumerate((False, True)):
        for col, (field, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for q_index, quantile in enumerate(QUANTILES):
                means = []
                stds = []
                for color, (setting, setting_samples) in zip(colors, groups.items()):
                    sample_values = []
                    for sample in setting_samples:
                        mask = npj.real_energy_drop_mask(sample) & ~sample.avalanche & (
                            sample.post_yield == post_yield
                        )
                        values = _positive_values(sample, field, mask)
                        if values.size == 0:
                            raise RuntimeError(
                                f"No reversible {field} values for {sample.path}, "
                                f"post_yield={post_yield}."
                            )
                        value = float(np.quantile(values, quantile))
                        sample_values.append(value)
                        rows.append(
                            {
                                "attribute": attribute,
                                "setting": setting,
                                "seed": sample.seed,
                                "post_yield": post_yield,
                                "field": field,
                                "quantile": quantile,
                                "value": value,
                            }
                        )
                    mean = float(np.mean(sample_values))
                    std = float(np.std(sample_values, ddof=1))
                    means.append(mean)
                    stds.append(std)
                    ax.scatter(
                        np.full(len(sample_values), setting),
                        sample_values,
                        color=color,
                        marker="o" if quantile == 0.5 else "s",
                        alpha=SCATTER_ALPHA,
                        s=16,
                    )
                means = np.asarray(means, dtype=float)
                stds = np.asarray(stds, dtype=float)
                slope, intercept = np.polyfit(np.log(settings), np.log(means), 1)
                fit_rows.append(
                    {
                        "attribute": attribute,
                        "field": field,
                        "post_yield": post_yield,
                        "quantile": quantile,
                        "slope": slope,
                        "intercept_log": intercept,
                    }
                )
                ax.errorbar(
                    settings,
                    means,
                    yerr=stds,
                    color="C0" if q_index == 0 else "C1",
                    marker="o" if q_index == 0 else "s",
                    linestyle="-",
                    linewidth=1.2,
                    capsize=3,
                    label=rf"$Q_{{{quantile:g}}}$",
                )
                _used_ring(ax, groups, attribute, settings, means)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.set_title("Post-yield" if post_yield else "Pre-yield")
            ax.legend(
                handles=_setting_handles(groups, attribute, colors)
                + [
                    Line2D([], [], color="C0", marker="o", label=r"$Q_{0.5}$"),
                    Line2D([], [], color="C1", marker="s", label=r"$Q_{0.99}$"),
                ],
                loc="upper left",
                ncol=1,
                frameon=True,
            )
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\epsilon_{\mathbf{x}}$" if attribute == "eps_x" else r"$\Delta\gamma$")
    pd.DataFrame(rows).to_csv(TABLE_DIR / f"{name}.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(TABLE_DIR / f"{name}_powerlaw_fits.csv", index=False)
    _save(fig, name)


def plot_classification_scatter(
    samples: list[npj.SampleData], attribute: str, field: str, name: str
) -> None:
    groups = _groups(samples, attribute)
    n = len(groups)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 7.0), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    ylabel = r"$|\Delta E_S|/V_0$" if field == "energy_drop_density" else r"$|\Delta\sigma_S|$"
    for col, (setting, setting_samples) in enumerate(groups.items()):
        cut = next(iter({sample.rev_u_cut for sample in setting_samples}))
        for row, post_yield in enumerate((False, True)):
            ax = axes[row, col]
            for key, label, color, marker in CLASS_INFO:
                values_x = []
                values_y = []
                for sample in setting_samples:
                    mask = _event_masks(sample)[key] & (sample.post_yield == post_yield)
                    indices = _sampled_indices(mask)
                    values_x.extend(sample.rev_u[indices])
                    values_y.extend(np.abs(getattr(sample, field)[indices]))
                x = np.asarray(values_x)
                y = np.asarray(values_y)
                valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                if np.any(valid):
                    ax.scatter(x[valid], y[valid], color=color, marker=marker, s=10, alpha=SCATTER_ALPHA, linewidths=0.45)
            ax.axvline(cut, color="black", linestyle=":", linewidth=1.0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(npj._setting_label(attribute, setting))
            if col == 0:
                ax.set_ylabel(ylabel + ("\npost-yield" if post_yield else "\npre-yield"))
            if row == 1:
                ax.set_xlabel(r"$\Delta_{\mathrm{rev}}\mathbf{u}$")
    axes[0, 0].legend(handles=_class_handles() + [Line2D([], [], color="black", linestyle=":", label=r"$\Delta_{\rm rev}\mathbf{u}$ cut")], loc="upper left", ncol=1, frameon=True)
    _save(fig, name)


def _class_fraction_rows(samples: list[npj.SampleData], attribute: str) -> pd.DataFrame:
    rows = []
    for setting, setting_samples in _groups(samples, attribute).items():
        for sample in setting_samples:
            masks = _event_masks(sample)
            for post_yield in (False, True):
                selected = sample.post_yield == post_yield
                denominator = int(sum(np.sum(mask & selected) for mask in masks.values()))
                for key, label, _color, _marker in CLASS_INFO:
                    count = int(np.sum(masks[key] & selected))
                    rows.append({"attribute": attribute, "setting": setting, "seed": sample.seed, "post_yield": post_yield, "class": key, "class_label": label, "count": count, "event_count": denominator, "fraction": count / denominator if denominator else np.nan})
    return pd.DataFrame(rows)


def plot_class_fractions(samples: list[npj.SampleData], attribute: str, name: str) -> None:
    table = _class_fraction_rows(samples, attribute)
    table.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    settings = np.asarray(sorted(table["setting"].unique()), dtype=float)
    colors = {key: color for key, _label, color, _marker in CLASS_INFO}
    markers = {key: marker for key, _label, _color, marker in CLASS_INFO}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, post_yield in zip(axes, (False, True)):
        for key, label, _color, _marker in CLASS_INFO:
            means = []
            stds = []
            for setting in settings:
                values = table.loc[(table.setting == setting) & (table.post_yield == post_yield) & (table["class"] == key), "fraction"].dropna()
                means.append(float(values.mean()) if len(values) else np.nan)
                stds.append(float(values.std(ddof=1)) if len(values) >= 2 else 0.0)
            finite = np.isfinite(means)
            if np.any(finite):
                means_array = np.asarray(means, dtype=float)
                ax.errorbar(settings[finite], means_array[finite], yerr=np.asarray(stds)[finite], color=colors[key], marker=markers[key], capsize=3, label=label)
                _used_ring(ax, _groups(samples, attribute), attribute, settings, means_array)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\epsilon_{\mathbf{x}}$" if attribute == "eps_x" else r"$\Delta\gamma$")
        ax.set_title("Post-yield" if post_yield else "Pre-yield")
    axes[0].set_ylabel("Fraction of event rows")
    axes[0].legend(loc="upper left", ncol=1, frameon=True)
    _save(fig, name)


def plot_participation_vs_rev_u(samples: list[npj.SampleData], attribute: str, name: str) -> None:
    groups = _groups(samples, attribute)
    n = len(groups)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 7.0), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for col, (setting, setting_samples) in enumerate(groups.items()):
        cut = next(iter({sample.rev_u_cut for sample in setting_samples}))
        for row, post_yield in enumerate((False, True)):
            ax = axes[row, col]
            for key, label, color, marker in CLASS_INFO:
                for sample in setting_samples:
                    mask = _event_masks(sample)[key] & (sample.post_yield == post_yield)
                    indices = _sampled_indices(mask)
                    valid = sample.rev_u[indices] > 0
                    if np.any(valid):
                        ax.scatter(sample.rev_u[indices][valid], sample.participation_fraction[indices][valid], color=color, marker=marker, s=10, alpha=SCATTER_ALPHA, linewidths=0.45)
            ax.axvline(cut, color="black", linestyle=":", linewidth=1.0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(npj._setting_label(attribute, setting))
            if row == 1:
                ax.set_xlabel(r"$\Delta_{\mathrm{rev}}\mathbf{u}$")
            if col == 0:
                ax.set_ylabel(r"Participation fraction $P$" + ("\npost-yield" if post_yield else "\npre-yield"))
    axes[0, 0].legend(handles=_class_handles() + [Line2D([], [], color="black", linestyle=":", label=r"$\Delta_{\rm rev}\mathbf{u}$ cut")], loc="upper left", ncol=1, frameon=True)
    _save(fig, name)


def plot_participation_summary(samples: list[npj.SampleData], attribute: str, name: str) -> None:
    rows = []
    for setting, setting_samples in _groups(samples, attribute).items():
        for sample in setting_samples:
            masks = _event_masks(sample)
            for post_yield in (False, True):
                for key, label, _color, _marker in CLASS_INFO:
                    values = sample.participation_fraction[masks[key] & (sample.post_yield == post_yield)]
                    rows.append({"attribute": attribute, "setting": setting, "seed": sample.seed, "post_yield": post_yield, "class": key, "class_label": label, "count": values.size, "median": float(np.median(values)) if values.size else np.nan, "q90": float(np.quantile(values, 0.9)) if values.size else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    settings = np.asarray(sorted(table.setting.unique()), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, post_yield in zip(axes, (False, True)):
        for key, label, color, marker in CLASS_INFO:
            means = []
            stds = []
            for setting in settings:
                values = table.loc[(table.setting == setting) & (table.post_yield == post_yield) & (table["class"] == key), "median"].dropna()
                means.append(float(values.mean()) if len(values) else np.nan)
                stds.append(float(values.std(ddof=1)) if len(values) >= 2 else 0.0)
            finite = np.isfinite(means)
            if np.any(finite):
                ax.errorbar(settings[finite], np.asarray(means)[finite], yerr=np.asarray(stds)[finite], color=color, marker=marker, capsize=3, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\epsilon_{\mathbf{x}}$" if attribute == "eps_x" else r"$\Delta\gamma$")
        ax.set_title("Post-yield" if post_yield else "Pre-yield")
    axes[0].set_ylabel("Mean sample-median participation fraction")
    axes[0].legend(loc="upper left", ncol=1, frameon=True)
    _save(fig, name)


def _rank_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Return the rank-based AUC, with larger scores meaning more irreversible."""
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    finite = np.isfinite(scores)
    scores = scores[finite]
    positive = positive[finite]
    n_positive = int(np.sum(positive))
    n_negative = int(np.sum(~positive))
    if not n_positive or not n_negative:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop + 1)
        start = stop
    positive_rank_sum = float(np.sum(ranks[positive]))
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )


def _population_masks(sample: npj.SampleData) -> dict[str, np.ndarray]:
    recorded = npj.real_energy_drop_mask(sample)
    positive_energy = np.isfinite(sample.energy_drop_density) & (
        sample.energy_drop_density > 0
    )
    return {
        "closing": recorded & ~sample.avalanche,
        "non_closing": recorded & sample.avalanche,
        "unrecorded": positive_energy & ~recorded,
    }


def _diagnostic_rows(
    samples: list[npj.SampleData], attribute: str
) -> pd.DataFrame:
    rows = []
    for setting, setting_samples in _groups(samples, attribute).items():
        for sample in setting_samples:
            population = _population_masks(sample)
            inter = sample.inter_strain_energy_density
            corrected = sample.energy_drop_density
            inter_drop = np.isfinite(inter) & (inter > 0)
            for post_yield in (False, True):
                selected = sample.post_yield == post_yield
                recorded = selected & npj.real_energy_drop_mask(sample)
                irreversible = selected & population["non_closing"]
                rows.append(
                    {
                        "attribute": attribute,
                        "setting": setting,
                        "seed": sample.seed,
                        "post_yield": post_yield,
                        "all_steps": int(np.sum(selected)),
                        "recorded_events": int(np.sum(recorded)),
                        "closing_events": int(np.sum(selected & population["closing"])),
                        "non_closing_events": int(np.sum(irreversible)),
                        "unrecorded_steps": int(np.sum(selected & population["unrecorded"])),
                        "positive_inter_strain_drops": int(np.sum(selected & inter_drop)),
                        "positive_inter_strain_drops_recorded": int(np.sum(recorded & inter_drop)),
                        "positive_inter_strain_drops_closing": int(np.sum(selected & population["closing"] & inter_drop)),
                        "positive_inter_strain_drops_non_closing": int(np.sum(irreversible & inter_drop)),
                        "positive_inter_strain_drops_unrecorded": int(np.sum(selected & population["unrecorded"] & inter_drop)),
                        "positive_inter_strain_drops_no_m3": int(np.sum(selected & inter_drop & (sample.m3_changes == 0))),
                        "positive_inter_strain_drops_closing_no_m3": int(np.sum(selected & population["closing"] & inter_drop & (sample.m3_changes == 0))),
                        "positive_inter_strain_drops_non_closing_no_m3": int(np.sum(irreversible & inter_drop & (sample.m3_changes == 0))),
                        "positive_inter_strain_drops_unrecorded_no_m3": int(np.sum(selected & population["unrecorded"] & inter_drop & (sample.m3_changes == 0))),
                        "positive_corrected_energy_drops_recorded": int(np.sum(recorded & np.isfinite(corrected) & (corrected > 0))),
                        "inter_strain_auc_recorded": _rank_auc(inter[recorded], population["non_closing"][recorded]),
                        "corrected_energy_auc_recorded": _rank_auc(corrected[recorded], population["non_closing"][recorded]),
                    }
                )
    table = pd.DataFrame(rows)
    return table


def _pooled_diagnostic_rows(
    samples: list[npj.SampleData], attribute: str
) -> pd.DataFrame:
    rows = []
    for setting, setting_samples in _groups(samples, attribute).items():
        for post_yield in (False, True):
            arrays = {name: [] for name in ("inter", "corrected", "recorded", "closing", "non_closing", "m3")}
            for sample in setting_samples:
                population = _population_masks(sample)
                selected = sample.post_yield == post_yield
                arrays["inter"].append(sample.inter_strain_energy_density[selected])
                arrays["corrected"].append(sample.energy_drop_density[selected])
                arrays["recorded"].append(
                    npj.real_energy_drop_mask(sample)[selected]
                )
                arrays["closing"].append(population["closing"][selected])
                arrays["non_closing"].append(population["non_closing"][selected])
                arrays["m3"].append(sample.m3_changes[selected] > 0)
            inter = np.concatenate(arrays["inter"])
            corrected = np.concatenate(arrays["corrected"])
            recorded = np.concatenate(arrays["recorded"])
            closing = np.concatenate(arrays["closing"])
            non_closing = np.concatenate(arrays["non_closing"])
            m3 = np.concatenate(arrays["m3"])
            inter_drop = np.isfinite(inter) & (inter > 0)
            rows.append(
                {
                    "attribute": attribute,
                    "setting": setting,
                    "seed": "pooled",
                    "post_yield": post_yield,
                    "all_steps": inter.size,
                    "recorded_events": int(np.sum(recorded)),
                    "closing_events": int(np.sum(closing)),
                    "non_closing_events": int(np.sum(non_closing)),
                    "unrecorded_steps": int(np.sum(~recorded)),
                    "positive_inter_strain_drops": int(np.sum(inter_drop)),
                    "positive_inter_strain_drops_recorded": int(np.sum(recorded & inter_drop)),
                    "positive_inter_strain_drops_closing": int(np.sum(closing & inter_drop)),
                    "positive_inter_strain_drops_non_closing": int(np.sum(non_closing & inter_drop)),
                    "positive_inter_strain_drops_unrecorded": int(np.sum(~recorded & inter_drop)),
                    "positive_inter_strain_drops_no_m3": int(np.sum(inter_drop & ~m3)),
                    "positive_inter_strain_drops_closing_no_m3": int(np.sum(closing & inter_drop & ~m3)),
                    "positive_inter_strain_drops_non_closing_no_m3": int(np.sum(non_closing & inter_drop & ~m3)),
                    "positive_inter_strain_drops_unrecorded_no_m3": int(np.sum(~recorded & inter_drop & ~m3)),
                    "positive_corrected_energy_drops_recorded": int(np.sum(recorded & np.isfinite(corrected) & (corrected > 0))),
                    "inter_strain_auc_recorded": _rank_auc(inter[recorded], non_closing[recorded]),
                    "corrected_energy_auc_recorded": _rank_auc(corrected[recorded], non_closing[recorded]),
                }
            )
    return pd.DataFrame(rows)


def plot_inter_strain_vs_corrected(
    samples: list[npj.SampleData], attribute: str, name: str
) -> None:
    groups = _groups(samples, attribute)
    n = len(groups)
    fig, axes = plt.subplots(2, n, figsize=(3.25 * n, 7.0), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    class_colors = {"closing": "C0", "non_closing": "C3", "unrecorded": "0.45"}
    class_labels = {"closing": "reversible", "non_closing": "irreversible", "unrecorded": "unrecorded"}
    for col, (setting, setting_samples) in enumerate(groups.items()):
        for row, post_yield in enumerate((False, True)):
            ax = axes[row, col]
            for population in ("closing", "non_closing", "unrecorded"):
                for m3_changed, marker in ((False, "o"), (True, "x")):
                    x_values = []
                    y_values = []
                    for sample in setting_samples:
                        masks = _population_masks(sample)
                        mask = (
                            masks[population]
                            & (sample.post_yield == post_yield)
                            & (sample.inter_strain_energy_density > 0)
                            & ((sample.m3_changes > 0) == m3_changed)
                        )
                        indices = _sampled_indices(mask)
                        x_values.extend(sample.inter_strain_energy_density[indices])
                        y_values.extend(sample.energy_drop_density[indices])
                    if x_values:
                        x = np.asarray(x_values)
                        y = np.asarray(y_values)
                        valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
                        if np.any(valid):
                            ax.scatter(
                                x[valid],
                                y[valid],
                                color=class_colors[population],
                                marker=marker,
                                s=12,
                                alpha=SCATTER_ALPHA,
                                linewidths=0.45,
                            )
            ax.axhline(0, color="0.5", linewidth=0.7)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-10)
            ax.set_title(npj._setting_label(attribute, setting))
            if row == 1:
                ax.set_xlabel(r"positive inter-strain drop $\Delta E_{\mathrm{inter}}/V_0$")
            if col == 0:
                ax.set_ylabel(r"$\Delta E_S/V_0$" + ("\npost-yield" if post_yield else "\npre-yield"))
    handles = [
        Line2D([], [], color=color, marker="o", linestyle="None", label=label)
        for key, label in class_labels.items()
        for color in [class_colors[key]]
    ] + [
        Line2D([], [], color="0.2", marker="o", linestyle="None", label="no $m_3$ change"),
        Line2D([], [], color="0.2", marker="x", linestyle="None", label="$m_3$ change"),
    ]
    axes[0, 0].legend(handles=handles, loc="upper left", ncol=1, frameon=True)
    _save(fig, name)


def plot_inter_strain_ccdf(
    samples: list[npj.SampleData], attribute: str, name: str
) -> None:
    groups = _groups(samples, attribute)
    n = len(groups)
    fig, axes = plt.subplots(2, n, figsize=(3.25 * n, 7.0), sharex="row", sharey="row")
    axes = np.atleast_2d(axes)
    metrics = (
        ("inter_strain_energy_density", r"$\Delta E_{\mathrm{inter}}/V_0$"),
        ("energy_drop_density", r"$\Delta E_S/V_0$"),
    )
    for row, (field, xlabel) in enumerate(metrics):
        for col, (setting, setting_samples) in enumerate(groups.items()):
            ax = axes[row, col]
            for population, color, label in (
                ("closing", "C0", "reversible"),
                ("non_closing", "C3", "irreversible"),
            ):
                values = []
                for sample in setting_samples:
                    mask = _population_masks(sample)[population] & (sample.post_yield == (row == 1))
                    values.append(np.asarray(getattr(sample, field)[mask], dtype=float))
                values = np.concatenate(values)
                values = values[np.isfinite(values) & (values > 0)]
                if values.size:
                    values.sort()
                    survival = np.arange(values.size, 0, -1) / values.size
                    ax.step(values, survival, where="post", color=color, label=label)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 1.1)
            ax.set_title(npj._setting_label(attribute, setting))
            if row == 1:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel("CCDF" + ("\npost-yield" if row == 1 else "\npre-yield"))
    axes[0, 0].legend(loc="upper right", frameon=True)
    _save(fig, name)


def run_inter_strain_diagnostic(
    samples: list[npj.SampleData], attribute: str, tag: str
) -> None:
    per_sample = _diagnostic_rows(samples, attribute)
    pooled = _pooled_diagnostic_rows(samples, attribute)
    pd.concat([per_sample, pooled], ignore_index=True).to_csv(
        TABLE_DIR / f"inter_strain_diagnostic_{tag}.csv", index=False
    )
    print(f"\nInter-strain diagnostic for {attribute}:")
    print(
        pooled[
            [
                "setting",
                "post_yield",
                "positive_inter_strain_drops",
                "positive_inter_strain_drops_closing",
                "positive_inter_strain_drops_non_closing",
                "positive_inter_strain_drops_no_m3",
                "positive_inter_strain_drops_closing_no_m3",
                "positive_inter_strain_drops_non_closing_no_m3",
                "inter_strain_auc_recorded",
                "corrected_energy_auc_recorded",
            ]
        ].to_string(index=False)
    )
    plot_inter_strain_vs_corrected(
        samples, attribute, f"inter_strain_vs_corrected_energy_{tag}"
    )
    plot_inter_strain_ccdf(samples, attribute, f"inter_strain_ccdf_{tag}")


def _run_setting_family(samples: list[npj.SampleData], attribute: str, tag: str) -> None:
    _write_cuts(samples, attribute, tag)
    npj.plot_classifier_cut_vs_setting(
        samples,
        attribute,
        f"rev_u_cut_vs_{tag}",
    )
    npj.plot_threshold_diagnostics(
        samples,
        attribute,
        f"rev_u_cut_diagnostics_{tag}",
    )
    plot_reversible_scaling(samples, attribute, f"reversible_scaling_{tag}")
    plot_classification_scatter(samples, attribute, "energy_drop_density", f"classification_energy_vs_rev_u_{tag}")
    plot_classification_scatter(samples, attribute, "stress_drop", f"classification_stress_vs_rev_u_{tag}")
    plot_class_fractions(samples, attribute, f"event_class_fractions_{tag}")
    plot_participation_vs_rev_u(samples, attribute, f"participation_vs_rev_u_{tag}")
    plot_participation_summary(samples, attribute, f"participation_summary_{tag}")
    run_inter_strain_diagnostic(samples, attribute, tag)


def main() -> None:
    _configure_output()
    epsilon_samples = npj.load_batch(-2)
    _run_setting_family(epsilon_samples, "eps_x", "epsilon_x")
    del epsilon_samples
    gc.collect()

    delta_gamma_samples = npj.load_batch(-1)
    _run_setting_family(delta_gamma_samples, "load_increment", "delta_gamma")
    del delta_gamma_samples
    gc.collect()
    print(f"Completed reversible-event analysis in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
