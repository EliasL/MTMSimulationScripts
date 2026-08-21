"""Reversible-only Sylvain energy-drop analysis.

Only recorded cycles with positive ``Delta E_S`` are analyzed.  Most settings
use one setting-wise unbinned log-Otsu split in ``Delta_rev u`` and discard
nothing.  At ``epsilon_x=1e-5`` and ``1e-4`` only, the specified left island
is removed before a final Otsu split separates reversible and irreversible events.

This is an exploratory alternative to the standard simulation workflow.  The
standard result uses a post-yield ``kappa_det = mu/2`` split, fits
only irreversible ``Delta E_S`` events, searches all observed xmin candidates,
and then performs the maximum-likelihood fit.

Run with::

    .venv/bin/python -m Plotting.reversibleOnlyEnergyAnalysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Plotting import numericalParameterJustification as npj
from Plotting.findXmin import analyze_xmin, plot_xmin_analysis
from Plotting.plotPowerLaw import (
    Truncated_Power_Law,
    dist_from_fit,
    getHist,
    make_fit,
    plot_fit_pdf,
    plot_data_pdf,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Plots/reversible_event_analysis/reversible_only"
TABLE_DIR = OUTPUT_DIR / "tables"
SCALING_DIR = OUTPUT_DIR / "scaling"
SCALING_TABLE_DIR = SCALING_DIR / "tables"
FIGURE_DPI = 250
SEARCH_MAX_XMIN = 1e-4
MIN_TAIL_COUNT = 100
NR_INITIAL = 100
SCATTER_ALPHA = 0.2
DELTA_GAMMA_SCATTER_SETTINGS = (1e-6, 1e-5, 1e-4)
IRREVERSIBLE_FIT_CONFIDENCE = 0.01
IRREVERSIBLE_FIT_PARALLEL = False


@dataclass(frozen=True)
class SettingClassification:
    attribute: str
    setting: float
    first_cut: float
    second_cut: float | None
    rule: str
    final_masks: dict[Path, np.ndarray]
    nonclosing_masks: dict[Path, np.ndarray]
    discarded_masks: dict[Path, np.ndarray]
    recorded_count: int
    final_count: int
    nonclosing_count: int
    discarded_count: int
    removed_cut: float | None = None

    @property
    def final_cut(self) -> float:
        return self.second_cut if self.second_cut is not None else self.first_cut


ENERGY_FIELDS = {
    "R": ("relaxation_energy_density", r"$\Delta E_R/V_0$"),
    "S": ("energy_drop_density", r"$\Delta E_S/V_0$"),
    "I": ("inter_strain_energy_density", r"$\Delta E_I/V_0$"),
}


def _prepare_output() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SCALING_TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _same_cut(setting_samples: list[npj.SampleData], setting: float) -> float:
    cuts = {float(sample.rev_u_cut) for sample in setting_samples}
    if len(cuts) != 1:
        raise RuntimeError(f"Inconsistent first Otsu cuts for setting {setting:g}: {cuts}")
    cut = cuts.pop()
    if not np.isfinite(cut) or cut <= 0:
        raise RuntimeError(f"Invalid first Otsu cut for setting {setting:g}: {cut}")
    return cut


def _positive_pooled(
    setting_samples: list[npj.SampleData], mask_getter,
) -> np.ndarray:
    values = np.concatenate(
        [sample.rev_u[mask_getter(sample)] for sample in setting_samples]
    )
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 50:
        raise RuntimeError(
            f"Only {values.size} positive Delta_rev u values available for a second Otsu split."
        )
    return values


def build_classifications(
    samples: list[npj.SampleData], attribute: str
) -> dict[float, SettingClassification]:
    """Classify positive energy drops, removing only the two specified islands."""
    groups = npj._setting_groups(samples, attribute)
    classifications: dict[float, SettingClassification] = {}

    for setting, setting_samples in groups.items():
        first_cut = _same_cut(setting_samples, setting)
        recorded_masks = {
            sample.path: npj.real_energy_drop_mask(sample).copy()
            for sample in setting_samples
        }

        if attribute == "eps_x" and np.isclose(setting, 1e-5):
            removed_cut = 1e-6
            rule = "discard Delta_rev u <= 1e-6, then Otsu-split the remainder"
        elif attribute == "eps_x" and np.isclose(setting, 1e-4):
            removed_cut = first_cut
            rule = "discard the first lower Otsu population, then split the remainder"
        else:
            removed_cut = None
            rule = "one Otsu split; discard nothing"

        if removed_cut is None:
            second_cut = None
        else:
            remaining_values = _positive_pooled(
                setting_samples,
                lambda sample: npj.real_energy_drop_mask(sample)
                & (sample.rev_u > removed_cut),
            )
            second_cut, _ = npj.unbinned_log_otsu_cut(remaining_values)

        final_cut = second_cut if second_cut is not None else first_cut
        discarded_masks = {}
        final_masks = {}
        nonclosing_masks = {}
        for sample in setting_samples:
            recorded = recorded_masks[sample.path]
            discarded = (
                recorded & (sample.rev_u <= removed_cut)
                if removed_cut is not None
                else np.zeros_like(recorded)
            )
            remaining = recorded & ~discarded
            discarded_masks[sample.path] = discarded
            final_masks[sample.path] = remaining & (sample.rev_u <= final_cut)
            nonclosing_masks[sample.path] = remaining & (sample.rev_u > final_cut)

        recorded_count = int(sum(mask.sum() for mask in recorded_masks.values()))
        final_count = int(sum(mask.sum() for mask in final_masks.values()))
        nonclosing_count = int(sum(mask.sum() for mask in nonclosing_masks.values()))
        discarded_count = int(sum(mask.sum() for mask in discarded_masks.values()))
        if final_count == 0:
            raise RuntimeError(f"The reversible selection is empty for setting {setting:g}.")
        if recorded_count != final_count + nonclosing_count + discarded_count:
            raise RuntimeError(f"Classification masks do not partition setting {setting:g}.")

        classifications[setting] = SettingClassification(
            attribute=attribute,
            setting=setting,
            first_cut=first_cut,
            second_cut=second_cut,
            rule=rule,
            final_masks=final_masks,
            nonclosing_masks=nonclosing_masks,
            discarded_masks=discarded_masks,
            recorded_count=recorded_count,
            final_count=final_count,
            nonclosing_count=nonclosing_count,
            discarded_count=discarded_count,
            removed_cut=removed_cut,
        )
        print(
            f"{attribute}={setting:g}: final cut={final_cut:.3e}, "
            f"reversible={final_count}, irreversible={nonclosing_count}, "
            f"discarded={discarded_count} ({rule})"
        )

    return classifications


def _selection_tag(classifications: dict[float, SettingClassification]) -> str:
    return rf"Positive $\Delta E_S$; Otsu $\Delta_{{\mathrm{{rev}}}}\mathbf{{u}}$"


def _classifier_tag(field_key: str, classifications: dict[float, SettingClassification]) -> str:
    if field_key in ENERGY_FIELDS:
        quantity = rf"$\Delta E_{field_key}$"
    elif field_key == "sigma":
        quantity = r"$|\Delta_{\mathrm{rev}}\sigma_{12}|$"
    else:
        raise ValueError(f"Unknown classifier-tag field: {field_key!r}")
    return rf"{_selection_tag(classifications)}; y: {quantity}"


def _legend_note(label: str) -> Line2D:
    return Line2D(
        [],
        [],
        linestyle="None",
        marker="None",
        color="none",
        label=label,
    )


def _save(
    fig: mpl.figure.Figure,
    name: str,
    *,
    pdf: bool = True,
    output_dir: Path = OUTPUT_DIR,
    tight_layout: bool = True,
) -> None:
    if tight_layout:
        fig.tight_layout()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_zorder(20)
            legend.get_frame().set_alpha(1.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    fig.savefig(png_path, dpi=FIGURE_DPI)
    if pdf:
        fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"Saved {png_path}")


def _values_for_setting(
    setting_samples: list[npj.SampleData],
    classification: SettingClassification,
    field: str,
) -> np.ndarray:
    values = np.concatenate(
        [
            np.asarray(getattr(sample, field), dtype=float)[
                classification.final_masks[sample.path]
            ]
            for sample in setting_samples
        ]
    )
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < MIN_TAIL_COUNT:
        raise RuntimeError(
            f"Only {values.size} positive reversible {field} values for "
            f"{classification.setting:g}."
        )
    return values


def _xmin_analysis(values: np.ndarray) -> dict:
    return analyze_xmin(
        values,
        nr_initial=NR_INITIAL,
        min_tail_count=MIN_TAIL_COUNT,
        distType=Truncated_Power_Law,
        max_xmin=SEARCH_MAX_XMIN,
        refine=False,
    )


def _plot_d_curve(
    analysis: dict,
    *,
    field_key: str,
    delta_gamma: float,
    note: str,
    name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    plot_xmin_analysis(analysis, ax=ax)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [_legend_note(note)],
        loc="best",
        frameon=True,
        handlelength=0,
        handletextpad=0.2,
        fontsize="small",
    )
    ax.set_title(rf"$\Delta\gamma={delta_gamma:.0e}$")
    ax.set_xlabel(rf"$\Delta E_{{{field_key},\min}}/V_0$")
    _save(fig, name)


def _plot_fit_panel(
    ax: mpl.axes.Axes,
    fit,
    *,
    field_key: str,
    title: str,
    show_legend: bool,
) -> None:
    field_label = rf"E_{field_key}/V_0"
    plot_data_pdf(
        ax,
        fit.data_original,
        color="C0",
        label="Empirical PDF",
        drop_label=field_label,
        drop_sign="positive",
        show_legend=False,
    )
    plot_fit_pdf(
        ax,
        fit,
        color="C1",
        label="Truncated power-law fit",
        drop_label=field_label,
        drop_sign="positive",
        show_legend=False,
        set_title=False,
        x_grid_mode="smooth",
        xmin_only=True,
        linewidth=1.8,
    )
    dist = dist_from_fit(fit)
    xmax = float(np.max(fit.data_original))
    ax.axvspan(
        fit.xmin,
        xmax,
        color="0.5",
        alpha=0.15,
        label="Fit region",
    )
    ax.axvline(
        fit.xmin,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=rf"$x_{{\min}}={fit.xmin:.2e}$",
    )
    ax.set_title(title)
    ax.text(
        0.04,
        0.05,
        rf"$\alpha={dist.alpha:.2f}$, $p={fit.p:.3f}$, $D={fit.D:.3f}$",
        transform=ax.transAxes,
        fontsize="small",
        va="bottom",
    )
    if show_legend:
        ax.legend(loc="best", fontsize="small")


def _plot_fit_figure(
    fit,
    *,
    field_key: str,
    delta_gamma: float,
    name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    _plot_fit_panel(
        ax,
        fit,
        field_key=field_key,
        title=rf"Irreversible; $\Delta\gamma={delta_gamma:.0e}$",
        show_legend=True,
    )
    _save(fig, name)


def _plot_fit_grid(
    fits: dict[float, object],
    *,
    field_key: str,
) -> None:
    groups = list(fits)
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.8), sharex=True)
    axes = np.asarray(axes).ravel()
    for ax, delta_gamma in zip(axes, groups):
        fit = fits[delta_gamma]
        _plot_fit_panel(
            ax,
            fit,
            field_key=field_key,
            title=rf"$\Delta\gamma={delta_gamma:.0e}$",
            show_legend=False,
        )
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize="small",
        frameon=True,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    _save(
        fig,
        f"powerlaw_fit_pdf_delta_gamma_E_{field_key}_irreversible_only",
        tight_layout=False,
    )


def make_delta_gamma_fits(
    samples: list[npj.SampleData], classifications: dict[float, SettingClassification]
) -> pd.DataFrame:
    groups = npj._setting_groups(samples, "load_increment")
    rows = []

    for field_key, (field, _) in ENERGY_FIELDS.items():
        note = _classifier_tag(field_key, classifications)
        for delta_gamma, setting_samples in groups.items():
            values = _values_for_setting(
                setting_samples, classifications[delta_gamma], field
            )
            analysis = _xmin_analysis(values)
            rows.append(
                {
                    "field": field_key,
                    "delta_gamma": delta_gamma,
                    "positive_reversible_steps": values.size,
                    "first_otsu_cut": classifications[delta_gamma].first_cut,
                    "final_otsu_cut": classifications[delta_gamma].final_cut,
                    "island_removal_cut": classifications[delta_gamma].removed_cut,
                    "simpleDrop_xmin": analysis["simple_drop_xmin"],
                    "simpleDrop_D": analysis["simple_drop_distance"],
                    "global_min_xmin": analysis["global_min_xmin"],
                    "global_min_D": analysis["global_min_distance"],
                    "simpleDrop_search_max": SEARCH_MAX_XMIN,
                    "xmin_refinement": analysis["refinement"],
                }
            )
            tag = f"{delta_gamma:.0e}".replace("+", "")
            _plot_d_curve(
                analysis,
                field_key=field_key,
                delta_gamma=delta_gamma,
                note=note,
                name=f"D_vs_cutoff_delta_E_{field_key}_reversible_only_delta_gamma_{tag}",
            )

    result = pd.DataFrame(rows)
    result.to_csv(
        SCALING_TABLE_DIR / "reversible_energy_xmin_vs_delta_gamma.csv", index=False
    )
    return result


def plot_delta_gamma_scaling(
    fit_rows: pd.DataFrame, classifications: dict[float, SettingClassification]
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = {field_key: f"C{index}" for index, field_key in enumerate(ENERGY_FIELDS)}
    handles = []
    exponent_rows = []
    for field_key, (_, label) in ENERGY_FIELDS.items():
        field_rows = fit_rows[fit_rows["field"] == field_key].sort_values("delta_gamma")
        color = colors[field_key]
        simple_alpha = _loglog_exponent(
            field_rows["delta_gamma"], field_rows["simpleDrop_xmin"]
        )
        global_alpha = _loglog_exponent(
            field_rows["delta_gamma"], field_rows["global_min_xmin"]
        )
        simple_label = (
            rf"{label}: simpleDrop, $f(x),\ \alpha={simple_alpha:.2f}$"
        )
        global_label = (
            rf"{label}: global min., $f(x),\ \alpha={global_alpha:.2f}$"
        )
        exponent_rows.extend(
            [
                {"field": field_key, "method": "simpleDrop", "alpha": simple_alpha},
                {"field": field_key, "method": "global_min", "alpha": global_alpha},
            ]
        )
        ax.plot(
            field_rows["delta_gamma"],
            field_rows["simpleDrop_xmin"],
            color=color,
            marker="o",
            linewidth=1.4,
            label=simple_label,
        )
        ax.plot(
            field_rows["delta_gamma"],
            field_rows["global_min_xmin"],
            color=color,
            marker="x",
            linestyle="--",
            linewidth=1.1,
            label=global_label,
        )
        handles.extend(
            [
                Line2D(
                    [],
                    [],
                    color=color,
                    marker="o",
                    linestyle="-",
                    label=simple_label,
                ),
                Line2D(
                    [],
                    [],
                    color=color,
                    marker="x",
                    linestyle="--",
                    label=global_label,
                ),
            ]
        )
    handles.append(_legend_note(r"$f(x)=A x^\alpha$"))
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize="small")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta\gamma$")
    ax.set_ylabel(r"$\Delta E_{\min}/V_0$")
    pd.DataFrame(exponent_rows).to_csv(
        SCALING_TABLE_DIR / "reversible_energy_scaling_exponents.csv", index=False
    )
    _save(fig, "reversible_energy_xmin_vs_delta_gamma", output_dir=SCALING_DIR)


def _plot_population_com_scaling(
    samples: list[npj.SampleData],
    classifications: dict[float, SettingClassification],
    *,
    attribute: str,
    field: str,
    value_column: str,
    ylabel: str,
    xlabel: str,
    output_name: str,
    table_name: str,
    exponent_table_name: str,
    extra_field: str | None = None,
    extra_column: str | None = None,
) -> None:
    """Plot a reversible-population center of mass versus a setting."""
    if (extra_field is None) != (extra_column is None):
        raise ValueError("extra_field and extra_column must be supplied together.")
    rows = []
    groups = npj._setting_groups(samples, attribute)
    for setting, setting_samples in groups.items():
        values = []
        extra_values = []
        for sample in setting_samples:
            mask = classifications[setting].final_masks[sample.path]
            primary = np.asarray(getattr(sample, field), dtype=float)[mask]
            valid = (
                np.isfinite(primary)
                & (primary > 0)
            )
            if extra_field is not None:
                extra = np.asarray(getattr(sample, extra_field), dtype=float)[mask]
                valid &= np.isfinite(extra) & (extra > 0)
                extra_values.append(extra[valid])
            values.append(primary[valid])

        values = np.concatenate(values)
        if values.size == 0:
            raise RuntimeError(
                f"No positive reversible {field} values for {attribute}={setting:g}."
            )
        row = {
            attribute: setting,
            "sample_count": len(setting_samples),
            "point_count": values.size,
            value_column: float(np.mean(values)),
        }
        if extra_column is not None:
            extra_values = np.concatenate(extra_values)
            row[extra_column] = float(np.mean(extra_values))
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(attribute)
    alpha = _loglog_exponent(result[attribute], result[value_column])
    pd.DataFrame(
        [{"quantity": value_column, "alpha": alpha}]
    ).to_csv(
        SCALING_TABLE_DIR / exponent_table_name,
        index=False,
    )
    result.to_csv(SCALING_TABLE_DIR / table_name, index=False)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    line, = ax.plot(
        result[attribute],
        result[value_column],
        color="C1",
        marker="o",
        linewidth=1.4,
        label=rf"reversible population COM, $\alpha={alpha:.2f}$",
    )
    ax.legend(
        handles=[line],
        loc="upper left",
        frameon=True,
        fontsize="small",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, output_name, output_dir=SCALING_DIR)


def plot_delta_gamma_energy_centroid_scaling(
    samples: list[npj.SampleData],
    classifications: dict[float, SettingClassification],
) -> None:
    """Plot the reversible energy-scatter centroid versus Delta gamma."""
    _plot_population_com_scaling(
        samples,
        classifications,
        attribute="load_increment",
        field="energy_drop_density",
        value_column="energy_center_of_mass",
        ylabel=r"$\langle \Delta E_S/V_0 \rangle_{\mathrm{rev}}$",
        xlabel=r"$\Delta\gamma$",
        output_name="reversible_energy_center_of_mass_vs_delta_gamma",
        table_name="reversible_energy_center_of_mass_vs_delta_gamma.csv",
        exponent_table_name="reversible_energy_center_of_mass_scaling_exponent.csv",
        extra_field="rev_u",
        extra_column="rev_u_center_of_mass",
    )


def plot_epsilon_x_rev_u_centroid_scaling(
    samples: list[npj.SampleData],
    classifications: dict[float, SettingClassification],
) -> None:
    """Plot the reversible Delta_rev u center of mass versus epsilon_x."""
    _plot_population_com_scaling(
        samples,
        classifications,
        attribute="eps_x",
        field="rev_u",
        value_column="rev_u_center_of_mass",
        ylabel=r"$\langle \Delta_{\mathrm{rev}}u \rangle_{\mathrm{rev}}$",
        xlabel=r"$\epsilon_x$",
        output_name="reversible_u_center_of_mass_vs_epsilon_x",
        table_name="reversible_u_center_of_mass_vs_epsilon_x.csv",
        exponent_table_name="reversible_u_center_of_mass_scaling_exponent.csv",
    )


def _selected_delta_gamma_irreversible_data(
    samples: list[npj.SampleData],
    classifications: dict[float, SettingClassification],
) -> tuple[float, int, dict[str, np.ndarray]]:
    groups = npj._setting_groups(samples, "load_increment")
    selected = [
        (setting, setting_samples)
        for setting, setting_samples in groups.items()
        if np.isclose(setting, npj.USED_DELTA_GAMMA)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one selected Delta gamma setting, got {selected}.")
    setting, setting_samples = selected[0]
    classification = classifications[setting]
    values = {
        "delta_energy": [],
        "delta_sigma": [],
        "rev_energy": [],
        "rev_sigma": [],
        "m3_delta_sigma2": [],
    }
    for sample in setting_samples:
        mask = classification.nonclosing_masks[sample.path]
        delta_energy = np.asarray(sample.energy_drop_density, dtype=float)[mask]
        delta_sigma = np.abs(np.asarray(sample.stress_drop, dtype=float)[mask])
        rev_energy = np.abs(np.asarray(sample.rev_energy_density, dtype=float)[mask])
        rev_sigma = np.abs(np.asarray(sample.rev_sigma, dtype=float)[mask])
        m3_density = np.asarray(sample.m3_changes, dtype=float)[mask] / sample.volume
        values["delta_energy"].append(delta_energy)
        values["delta_sigma"].append(delta_sigma)
        values["rev_energy"].append(rev_energy)
        values["rev_sigma"].append(rev_sigma)
        values["m3_delta_sigma2"].append(m3_density * delta_sigma**2)
    return setting, len(setting_samples), {
        key: np.concatenate(item_values) for key, item_values in values.items()
    }


def _plot_irreversible_relation(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    name: str,
    setting: float,
    fixed_alpha: float | None = None,
) -> dict:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if fixed_alpha is None:
        alpha, intercept, valid = _loglog_fit(x, y)
        model_label = rf"$f(x)=A x^\alpha,\ \alpha={alpha:.2f}$"
    else:
        _, _, valid = _loglog_fit(x, y)
        alpha = float(fixed_alpha)
        log_x = np.log10(x[valid])
        log_y = np.log10(y[valid])
        intercept = float(np.mean(log_y - alpha * log_x))
        model_label = rf"$f(x)\equiv A x,\ \alpha={alpha:.0f}$"
    x = x[valid]
    y = y[valid]
    fit_x = np.geomspace(x.min(), x.max(), 100)
    fit_y = 10 ** intercept * fit_x**alpha

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(
        x,
        y,
        color="C1",
        alpha=SCATTER_ALPHA,
        s=14,
        rasterized=True,
        label="irreversible population",
    )
    ax.plot(
        fit_x,
        fit_y,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=model_label,
    )
    ax.legend(loc="upper left", frameon=True, fontsize="small")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(rf"selected $\Delta\gamma={setting:.0e}$")
    _save(fig, name, output_dir=SCALING_DIR)
    return {
        "relation": name,
        "setting_parameter": "load_increment",
        "setting": setting,
        "positive_points": int(x.size),
        "alpha": alpha,
        "log10_prefactor": intercept,
    }


def plot_selected_delta_gamma_irreversible_scalings(
    samples: list[npj.SampleData],
    classifications: dict[float, SettingClassification],
) -> None:
    """Plot event-level scalings for the selected irreversible population."""
    setting, sample_count, data = _selected_delta_gamma_irreversible_data(
        samples, classifications
    )
    relations = [
        (
            data["rev_energy"],
            data["delta_energy"],
            r"$|\Delta_{\mathrm{rev}}E|/V_0$",
            r"$\Delta E_S/V_0$",
            "irreversible_deltaE_vs_delta_revE_selected_delta_gamma",
            1.0,
        ),
        (
            data["rev_sigma"],
            data["delta_sigma"],
            r"$|\Delta_{\mathrm{rev}}\sigma_{12}|$",
            r"$|\Delta\sigma_S|$",
            "irreversible_delta_sigma12_vs_delta_rev_sigma12_selected_delta_gamma",
            None,
        ),
        (
            data["m3_delta_sigma2"],
            data["delta_energy"],
            r"$(m_3/V_0)|\Delta\sigma_S|^2$",
            r"$\Delta E_S/V_0$",
            "irreversible_deltaE_vs_m3_delta_sigma12_squared_selected_delta_gamma",
            None,
        ),
    ]
    rows = []
    for x_values, y_values, x_label, y_label, name, fixed_alpha in relations:
        row = _plot_irreversible_relation(
            x_values,
            y_values,
            x_label=x_label,
            y_label=y_label,
            name=name,
            setting=setting,
            fixed_alpha=fixed_alpha,
        )
        row["sample_count"] = sample_count
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        SCALING_TABLE_DIR / "irreversible_population_scaling_exponents.csv",
        index=False,
    )


def _loglog_fit(
    x_values, y_values
) -> tuple[float, float, np.ndarray]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("At least two positive points are required for a scaling exponent.")
    alpha, intercept = np.polyfit(np.log10(x[valid]), np.log10(y[valid]), 1)
    return float(alpha), float(intercept), valid


def _loglog_exponent(x_values, y_values) -> float:
    return _loglog_fit(x_values, y_values)[0]


def write_classification_tables(
    classifications: dict[float, SettingClassification], attribute: str
) -> None:
    rows = []
    for setting, classification in classifications.items():
        rows.append(
            {
                "setting_parameter": attribute,
                "setting": setting,
                "first_otsu_cut": classification.first_cut,
                "final_otsu_cut": classification.final_cut,
                "island_removal_cut": classification.removed_cut,
                "rule": classification.rule,
                "recorded_steps": classification.recorded_count,
                "closing_steps": classification.final_count,
                "nonclosing_steps": classification.nonclosing_count,
                "discarded_steps": classification.discarded_count,
            }
        )
    pd.DataFrame(rows).to_csv(
        TABLE_DIR / f"reversible_classifier_summary_{attribute}.csv", index=False
    )


def plot_setting_scatter(
    samples: list[npj.SampleData],
    classifications: dict[float, SettingClassification],
    attribute: str,
    *,
    scatter_key: str = "S",
    settings: tuple[float, ...] | None = None,
) -> None:
    scatter_fields = {
        "S": ("energy_drop_density", r"$\Delta E_S/V_0$", False),
        "R": ("relaxation_energy_density", r"$\Delta E_R/V_0$", False),
        "sigma": ("rev_sigma", r"$|\Delta_{\mathrm{rev}}\sigma_{12}|$", True),
    }
    try:
        y_field, y_label, use_absolute = scatter_fields[scatter_key]
    except KeyError as exc:
        raise ValueError(f"Unknown setting scatter field: {scatter_key!r}") from exc
    if attribute not in {"eps_x", "load_increment"}:
        raise ValueError(f"Unknown setting attribute: {attribute!r}")
    all_groups = npj._setting_groups(samples, attribute)
    if settings is None:
        groups = all_groups
    else:
        requested = tuple(float(setting) for setting in settings)
        missing = sorted(set(requested) - set(all_groups))
        if missing:
            raise ValueError(
                f"Requested {attribute} settings are unavailable: {missing}"
            )
        groups = {setting: all_groups[setting] for setting in requested}
    colors = npj._colors(len(groups))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    draw_items = list(zip(colors, groups.items()))[::-1]

    def draw_mask(
        sample: npj.SampleData,
        mask: np.ndarray,
        marker: str,
        color: str,
        *,
        alpha: float,
        zorder: int,
    ) -> None:
        indices = npj._sampled_indices(mask)
        x_values = sample.rev_u[indices]
        y_values = getattr(sample, y_field)[indices]
        if use_absolute:
            y_values = np.abs(y_values)
        valid = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
            & (x_values > 0)
            & (y_values > 0)
        )
        ax.scatter(
            x_values[valid],
            y_values[valid],
            marker=marker,
            s=18,
            color=color,
            alpha=alpha,
            linewidths=0.8,
            rasterized=True,
            zorder=zorder,
        )

    # Draw discarded islands first, so retained populations remain legible.
    for color, (setting, setting_samples) in draw_items:
        classification = classifications[setting]
        for post_yield in (False, True):
            for sample in setting_samples:
                base = npj.real_energy_drop_mask(sample) & (
                    sample.post_yield == post_yield
                )
                draw_mask(
                    sample,
                    base & classification.discarded_masks[sample.path],
                    "x",
                    color,
                    alpha=SCATTER_ALPHA,
                    zorder=1,
                )

    # Draw retained populations above the discarded markers.
    for color, (setting, setting_samples) in draw_items:
        classification = classifications[setting]
        for post_yield in (False, True):
            for sample in setting_samples:
                base = npj.real_energy_drop_mask(sample) & (
                    sample.post_yield == post_yield
                )
                draw_mask(
                    sample,
                    base & classification.final_masks[sample.path],
                    "o",
                    color,
                    alpha=SCATTER_ALPHA,
                    zorder=2,
                )
                draw_mask(
                    sample,
                    base & classification.nonclosing_masks[sample.path],
                    "o",
                    color,
                    alpha=SCATTER_ALPHA,
                    zorder=2,
                )
        ax.axvline(
            classification.final_cut,
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=3,
        )

    handles = npj._setting_handles(groups, attribute, colors)
    if any(classifications[setting].discarded_count for setting in groups):
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                linestyle="None",
                color="black",
                label="discarded island",
            )
        )
    handles.append(
        Line2D(
            [],
            [],
            color="black",
            linestyle="--",
            label="reversible/irreversible cut",
        )
    )
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        fontsize="small",
        ncol=2,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta_{\mathrm{rev}}\mathbf{u}$")
    ax.set_ylabel(y_label)
    setting_axis_tag = {
        "eps_x": "epsilon_x",
        "load_increment": "delta_gamma",
    }[attribute]
    if len(groups) == 1:
        setting = next(iter(groups))
        ax.set_title(npj._setting_label(attribute, setting))
        setting_tag = f"{setting:.0e}".replace("+", "")
        output_prefix = {
            "S": "reversible_energy_drop_vs_rev_u",
            "R": "reversible_relaxation_energy_vs_rev_u",
            "sigma": "reversible_sigma12_vs_rev_u",
        }[scatter_key]
        output_name = f"{output_prefix}_{setting_axis_tag}_" + setting_tag
    else:
        output_prefix = {
            "S": "reversible_energy_drop_vs_rev_u",
            "R": "reversible_relaxation_energy_vs_rev_u",
            "sigma": "reversible_sigma12_vs_rev_u",
        }[scatter_key]
        output_name = f"{output_prefix}_{setting_axis_tag}"
    _save(fig, output_name)


def plot_epsilon_scatter(
    samples: list[npj.SampleData],
    classifications: dict[float, SettingClassification],
    *,
    scatter_key: str = "S",
    settings: tuple[float, ...] | None = None,
) -> None:
    """Backward-compatible wrapper for the combined epsilon_x scatter plot."""
    plot_setting_scatter(
        samples,
        classifications,
        "eps_x",
        scatter_key=scatter_key,
        settings=settings,
    )


def _delta_gamma_population_values(
    samples: list[npj.SampleData],
    classifications: dict[float, SettingClassification],
    *,
    field_key: str = "S",
) -> dict[str, dict[float, np.ndarray]]:
    if field_key not in ENERGY_FIELDS:
        raise ValueError(f"Unknown energy field: {field_key!r}")
    field = ENERGY_FIELDS[field_key][0]
    groups = npj._setting_groups(samples, "load_increment")
    populations = ("all", "reversible", "irreversible")
    population_values = {population: {} for population in populations}
    for setting, setting_samples in groups.items():
        values_by_population = {population: [] for population in populations}
        classification = classifications[setting]
        for sample in setting_samples:
            reversible = classification.final_masks[sample.path]
            irreversible = classification.nonclosing_masks[sample.path]
            classified = reversible | irreversible
            expected = npj.real_energy_drop_mask(sample)
            if not np.array_equal(classified, expected):
                raise RuntimeError(
                    f"Delta-gamma populations do not partition recorded drops in {sample.path}."
                )
            field_values = np.asarray(getattr(sample, field), dtype=float)
            values_by_population["all"].append(field_values[classified])
            values_by_population["reversible"].append(
                field_values[reversible]
            )
            values_by_population["irreversible"].append(
                field_values[irreversible]
            )
        for population in populations:
            values = np.concatenate(values_by_population[population])
            values = values[np.isfinite(values) & (values > 0)]
            if values.size == 0:
                raise RuntimeError(
                    f"No positive energy drops for {population} at Delta gamma={setting:g}."
                )
            population_values[population][setting] = values
    return population_values


def plot_irreversible_delta_gamma_energy_pdf(
    samples: list[npj.SampleData],
    classifications: dict[float, SettingClassification],
    *,
    field_key: str = "R",
) -> None:
    """Plot one positive-energy PDF containing only irreversible events."""
    if field_key not in ENERGY_FIELDS:
        raise ValueError(f"Unknown energy field: {field_key!r}")
    groups = npj._setting_groups(samples, "load_increment")
    colors = npj._colors(len(groups))
    values_by_setting = _delta_gamma_population_values(
        samples, classifications, field_key=field_key
    )["irreversible"]
    all_values = np.concatenate(list(values_by_setting.values()))
    density_values = []
    for values in values_by_setting.values():
        _, density = getHist(values)
        density_values.extend(density[density > 0])
    if not density_values:
        raise RuntimeError("No positive PDF density values were generated.")

    def log_limits(values: np.ndarray) -> tuple[float, float]:
        log_values = np.log10(values)
        return (
            10 ** (log_values.min() - 0.08),
            10 ** (log_values.max() + 0.08),
        )

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for color, setting in zip(colors, groups):
        plot_data_pdf(
            ax,
            values_by_setting[setting],
            color=color,
            show_legend=False,
            drop_label=rf"E_{field_key}/V_0",
            drop_sign="positive",
        )
    ax.set_xlim(log_limits(all_values))
    ax.set_ylim(log_limits(np.asarray(density_values)))
    ax.set_title("Irreversible")
    ax.set_xlabel(rf"$\Delta E_{field_key}/V_0$")
    ax.set_ylabel(rf"$p(\Delta E_{field_key}/V_0)$")
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="None",
            label=npj._setting_label("load_increment", setting),
        )
        for color, setting in zip(colors, groups)
    ]
    ax.legend(handles=handles, ncol=2, fontsize="small", frameon=True)
    _save(fig, f"energy_drop_pdf_delta_gamma_E_{field_key}_irreversible_only")


def plot_delta_gamma_energy_pdfs(
    samples: list[npj.SampleData],
    classifications: dict[float, SettingClassification],
    *,
    field_key: str = "S",
) -> None:
    """Plot one energy PDF and its reversible/irreversible split."""
    if field_key not in ENERGY_FIELDS:
        raise ValueError(f"Unknown energy field: {field_key!r}")
    groups = npj._setting_groups(samples, "load_increment")
    colors = npj._colors(len(groups))
    population_values = _delta_gamma_population_values(
        samples, classifications, field_key=field_key
    )

    all_values = np.concatenate(list(population_values["all"].values()))
    density_values = []
    for values_by_setting in population_values.values():
        for values in values_by_setting.values():
            _, density = getHist(values)
            density_values.extend(density[density > 0])
    if not density_values:
        raise RuntimeError("No positive PDF density values were generated.")

    def log_limits(values: np.ndarray) -> tuple[float, float]:
        log_values = np.log10(values)
        return (
            10 ** (log_values.min() - 0.08),
            10 ** (log_values.max() + 0.08),
        )

    x_limits = log_limits(all_values)
    y_limits = log_limits(np.asarray(density_values))
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="None",
            label=npj._setting_label("load_increment", setting),
        )
        for color, setting in zip(colors, groups)
    ]

    fig = plt.figure(figsize=(6.4, 4.2))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.25, 1.0),
        left=0.10,
        right=0.98,
        bottom=0.13,
        top=0.90,
        wspace=0.22,
    )
    left_grid = grid[0, 0].subgridspec(2, 1, height_ratios=(0.32, 1.0), hspace=0.04)
    right_grid = grid[0, 1].subgridspec(2, 1, hspace=0.42)
    legend_axis = fig.add_subplot(left_grid[0, 0])
    legend_axis.axis("off")
    axes = {
        "all": fig.add_subplot(left_grid[1, 0]),
        "reversible": fig.add_subplot(right_grid[0, 0]),
        "irreversible": fig.add_subplot(right_grid[1, 0]),
    }
    titles = {"all": "All drops", "reversible": "Reversible", "irreversible": "Irreversible"}
    for population, ax in axes.items():
        for color, (setting, _setting_samples) in zip(colors, groups.items()):
            plot_data_pdf(
                ax,
                population_values[population][setting],
                color=color,
                show_legend=False,
                drop_label=rf"E_{field_key}/V_0",
            )
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(labelsize="small")
        ax.set_title(titles[population], pad=4)
        ax.set_xlabel(rf"$\Delta E_{field_key}/V_0$")
        ax.set_ylabel(rf"$p(\Delta E_{field_key}/V_0)$")

    legend_axis.legend(
        handles=handles,
        loc="center",
        ncol=3,
        fontsize="x-small",
        frameon=True,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    axes["reversible"].set_ylabel("")
    axes["irreversible"].set_ylabel("")
    axes["reversible"].set_xlabel("")
    axes["reversible"].tick_params(labelbottom=False)
    output_name = (
        "energy_drop_pdf_delta_gamma_population_decomposition"
        if field_key == "S"
        else f"energy_drop_pdf_delta_gamma_E_{field_key}_population_decomposition"
    )
    _save(fig, output_name, tight_layout=False)


def make_irreversible_delta_gamma_fits(
    samples: list[npj.SampleData],
    classifications: dict[float, SettingClassification],
    *,
    field_key: str = "S",
) -> pd.DataFrame:
    """Fit truncated power laws to one irreversible energy-drop population."""
    if field_key not in ENERGY_FIELDS:
        raise ValueError(f"Unknown energy field: {field_key!r}")
    groups = npj._setting_groups(samples, "load_increment")
    values_by_setting = _delta_gamma_population_values(
        samples, classifications, field_key=field_key
    )["irreversible"]
    rows = []
    fits = {}
    for delta_gamma in groups:
        values = values_by_setting[delta_gamma]
        tag = f"{delta_gamma:.0e}".replace("+", "")
        analysis = _xmin_analysis(values)
        global_min_xmin = float(analysis["global_min_xmin"])
        if not np.isfinite(global_min_xmin) or global_min_xmin <= 0:
            raise RuntimeError(
                f"Invalid global-minimum xmin at Delta gamma={delta_gamma:g}: "
                f"{global_min_xmin}"
            )
        fit = make_fit(
            values,
            xmin_range=global_min_xmin,
            distType=Truncated_Power_Law,
            use_cache=False,
        )
        fit.xmin_analysis = analysis
        fit.xmin_fitting_results = analysis
        fit.evaluate_fit(
            data=values,
            confidence=IRREVERSIBLE_FIT_CONFIDENCE,
            parallel=IRREVERSIBLE_FIT_PARALLEL,
            max_workers=None,
            use_cache=True,
            cache_dir=str(SCALING_DIR / "irreversible_fit_cache"),
            max_synthetic_samples=5e6,
            tqdmDesc=f"irreversible Delta gamma={delta_gamma:g}",
        )
        fits[delta_gamma] = fit
        distribution = dist_from_fit(fit)
        alpha = float(distribution.alpha)
        p_value = float(fit.p)
        if not np.isfinite(alpha) or not np.isfinite(p_value):
            raise RuntimeError(
                f"Non-finite irreversible fit result at Delta gamma={delta_gamma:g}."
            )
        if not 0 <= p_value <= 1:
            raise RuntimeError(
                f"Invalid irreversible p-value at Delta gamma={delta_gamma:g}: {p_value}"
            )
        _plot_d_curve(
            analysis,
            field_key=field_key,
            delta_gamma=delta_gamma,
            note=(
                rf"irreversible population; positive $\Delta E_S$ classification; "
                rf"$\Delta E_{field_key}$ values"
            ),
            name=f"D_vs_cutoff_delta_E_{field_key}_irreversible_only_delta_gamma_{tag}",
        )
        _plot_fit_figure(
            fit,
            field_key=field_key,
            delta_gamma=delta_gamma,
            name=f"powerlaw_fit_pdf_delta_E_{field_key}_irreversible_only_delta_gamma_{tag}",
        )
        rows.append(
            {
                "population": "irreversible",
                "field": field_key,
                "delta_gamma": delta_gamma,
                "positive_drops": int(values.size),
                "xmin": float(fit.xmin),
                "tail_count": int(np.count_nonzero(values >= fit.xmin)),
                "ks_distance_D": float(fit.D),
                "exponent_alpha": alpha,
                "exponent_alpha_std": float(getattr(fit, "alpha_std", np.nan)),
                "p_value": p_value,
                "p_value_std": float(getattr(fit, "p_std", np.nan)),
                "bootstrap_confidence": IRREVERSIBLE_FIT_CONFIDENCE,
                "bootstrap_sets": int(1 / (4 * IRREVERSIBLE_FIT_CONFIDENCE**2)),
                "fit_distribution": "truncated_power_law",
                "xmin_method": "global_min",
            }
        )
    result = pd.DataFrame(rows)
    table_name = (
        "irreversible_powerlaw_fits_delta_gamma.csv"
        if field_key == "S"
        else f"irreversible_powerlaw_fits_delta_gamma_{field_key}.csv"
    )
    result.to_csv(
        SCALING_TABLE_DIR / table_name,
        index=False,
    )
    _plot_fit_grid(fits, field_key=field_key)
    print(f"\nIrreversible truncated-power-law fits for E_{field_key}:")
    print(
        result[
            [
                "delta_gamma",
                "positive_drops",
                "xmin",
                "ks_distance_D",
                "exponent_alpha",
                "p_value",
            ]
        ].to_string(index=False)
    )
    return result


def main() -> None:
    _prepare_output()

    epsilon_samples = npj.load_batch(-2)
    epsilon_classifications = build_classifications(epsilon_samples, "eps_x")
    write_classification_tables(epsilon_classifications, "eps_x")
    plot_setting_scatter(
        epsilon_samples, epsilon_classifications, "eps_x", scatter_key="S"
    )
    plot_setting_scatter(
        epsilon_samples, epsilon_classifications, "eps_x", scatter_key="sigma"
    )
    plot_epsilon_x_rev_u_centroid_scaling(
        epsilon_samples, epsilon_classifications
    )
    del epsilon_samples

    delta_gamma_samples = npj.load_batch(-1)
    delta_gamma_classifications = build_classifications(
        delta_gamma_samples, "load_increment"
    )
    write_classification_tables(delta_gamma_classifications, "load_increment")
    plot_setting_scatter(
        delta_gamma_samples,
        delta_gamma_classifications,
        "load_increment",
        scatter_key="S",
        settings=DELTA_GAMMA_SCATTER_SETTINGS,
    )
    plot_setting_scatter(
        delta_gamma_samples,
        delta_gamma_classifications,
        "load_increment",
        scatter_key="R",
        settings=DELTA_GAMMA_SCATTER_SETTINGS,
    )
    plot_setting_scatter(
        delta_gamma_samples,
        delta_gamma_classifications,
        "load_increment",
        scatter_key="sigma",
        settings=DELTA_GAMMA_SCATTER_SETTINGS,
    )
    plot_delta_gamma_energy_pdfs(
        delta_gamma_samples,
        delta_gamma_classifications,
    )
    plot_delta_gamma_energy_pdfs(
        delta_gamma_samples,
        delta_gamma_classifications,
        field_key="R",
    )
    make_irreversible_delta_gamma_fits(
        delta_gamma_samples,
        delta_gamma_classifications,
    )
    plot_irreversible_delta_gamma_energy_pdf(
        delta_gamma_samples,
        delta_gamma_classifications,
        field_key="R",
    )
    make_irreversible_delta_gamma_fits(
        delta_gamma_samples,
        delta_gamma_classifications,
        field_key="R",
    )
    fit_rows = make_delta_gamma_fits(
        delta_gamma_samples, delta_gamma_classifications
    )
    plot_delta_gamma_scaling(fit_rows, delta_gamma_classifications)
    plot_delta_gamma_energy_centroid_scaling(
        delta_gamma_samples, delta_gamma_classifications
    )
    plot_selected_delta_gamma_irreversible_scalings(
        delta_gamma_samples, delta_gamma_classifications
    )

    (OUTPUT_DIR / "analysis_definition.json").write_text(
        json.dumps(
            {
                "reversible_population": "positive Delta E_S cycles classified by setting-wise unbinned log-Otsu split of Delta_rev u",
                "secondary_epsilon_rules": {
                    "1e-5": "discard Delta_rev u <= 1e-6, then Otsu-split the remainder into reversible and irreversible",
                    "1e-4": "discard the first lower Otsu population, then split the remainder into reversible and irreversible",
                    "other settings": "one Otsu split into reversible and irreversible; discard nothing",
                },
                "energy_fields": {
                    key: field for key, (field, _) in ENERGY_FIELDS.items()
                },
                "simpleDrop_search_max_xmin": SEARCH_MAX_XMIN,
                "xmin_refinement": "coarse_scan",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Completed reversible-only analysis in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
