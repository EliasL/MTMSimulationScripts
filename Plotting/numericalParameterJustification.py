"""Numerical-parameter checks for the Sylvain reversibility batches.

Running this module without arguments regenerates the complete analysis::

    .venv/bin/python -m Plotting.numericalParameterJustification

Avalanches are defined only from the measured cycle displacement.  For every
parameter setting, an exact unbinned Otsu split in ``log10(rev_u_diff)`` is
computed from all recorded cycles pooled over seeds.  The stored
``is_reversible`` flag is never used.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Management.configGenerator import ConfigGenerator, SimulationConfig
from Management.jobs import sylvainBatches
from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import (
    calculate_energy_step_data,
    calculate_stress_step_data,
    infer_plastic_event_column,
    volume_from_metadata,
)
from Plotting.findXmin import plot_xmin_analysis
from Plotting.plotPowerLaw import (
    Truncated_Power_Law,
    dist_from_fit,
    findPrePostSplit,
    make_fit,
    plot_fit_cdf,
    pretty_variant_label,
)
from Plotting.remotePlotting import get_csv_files


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Plots/numerical_parameter_justification"
TABLE_DIR = OUTPUT_DIR / "tables"
START_LOAD = 0.14
MAX_LOAD = 1.0
LOCAL_RATE_WIDTHS = (0.02, 0.05)
QUANTILES = (0.5, 0.99)
USED_EPS_X = 1e-6
USED_DELTA_GAMMA = 1e-5
MIN_CLASS_FRACTION = 0.02
FIGURE_DPI = 250
SCATTER_ALPHA = 0.2
MAX_ECDF_POINTS = 6000
MAX_SCATTER_POINTS_PER_SAMPLE_CLASS = 750


@dataclass(frozen=True)
class SampleData:
    path: Path
    batch: int
    seed: int
    load_increment: float
    eps_x: float
    volume: float
    rev_u_cut: float
    gamma: np.ndarray
    post_yield: np.ndarray
    rev_u: np.ndarray
    rev_energy_density: np.ndarray
    rev_sigma: np.ndarray
    energy_drop_density: np.ndarray
    stress_drop: np.ndarray
    relaxation_energy: np.ndarray
    m3_changes: np.ndarray
    participation_fraction: np.ndarray
    m3_participation_fraction: np.ndarray

    @property
    def exposure(self) -> float:
        return float(self.gamma[-1] - START_LOAD)

    @property
    def avalanche(self) -> np.ndarray:
        if not np.isfinite(self.rev_u_cut) or self.rev_u_cut <= 0:
            raise ValueError(f"Invalid rev_u_cut={self.rev_u_cut} for {self.path}.")
        return self.rev_u > self.rev_u_cut

    @property
    def cycle_recorded(self) -> np.ndarray:
        return (
            (self.rev_u != 0)
            | (self.rev_energy_density != 0)
            | (self.rev_sigma != 0)
        )

    @property
    def closing_cycle(self) -> np.ndarray:
        return self.cycle_recorded & ~self.avalanche


def _job_name(path: Path) -> str:
    return path.parent.name if path.name == "macroData.csv" else path.stem


def _read_required_columns(path: Path) -> tuple[pd.DataFrame, str]:
    columns = set(pd.read_csv(path, nrows=0).columns)
    m3_column = next(
        (
            name
            for name in (
                "nr_elements_with_m3_fix_change",
                "nr_elements_with_m3_change",
            )
            if name in columns
        ),
        None,
    )
    required = {
        "load",
        "total_energy",
        "total_e_change_from_init",
        "avg_sigma12",
        "avg_sigma12_change_from_init",
        "rev_u_diff",
        "rev_energy_diff",
        "rev_sigma_12_diff",
        "participationFraction",
        "m3_participationFraction",
    }
    missing = sorted(required - columns)
    if m3_column is None:
        missing.append("nr_elements_with_m3[_fix]_change")
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")

    df = pd.read_csv(path, usecols=sorted(required | {m3_column}))
    inferred_m3_column = infer_plastic_event_column(df)
    if inferred_m3_column != m3_column:
        raise RuntimeError(
            f"Inconsistent m3 column inference for {path}: "
            f"{m3_column!r} vs {inferred_m3_column!r}."
        )
    return df, m3_column


def _load_sample(path: Path, config: SimulationConfig, batch: int) -> SampleData:
    df, m3_column = _read_required_columns(path)
    load = df["load"].to_numpy(dtype=float)
    expected_steps = round((config.maxLoad - config.startLoad) / config.loadIncrement)
    if load.size != expected_steps + 1:
        raise ValueError(
            f"Expected {expected_steps + 1} load states in {path}, found {load.size}."
        )
    if not np.isclose(load[0], START_LOAD) or not np.isclose(load[-1], MAX_LOAD):
        raise ValueError(
            f"Expected load interval [{START_LOAD}, {MAX_LOAD}] in {path}, "
            f"found [{load[0]}, {load[-1]}]."
        )
    delta_gamma = np.diff(load)
    if not np.allclose(
        delta_gamma,
        config.loadIncrement,
        # Decimal CSV serialization produces four 1e-11 deviations in the
        # 860,000-step, Delta-gamma=1e-6 files.  The endpoint and step-count
        # checks above still reject incomplete or structurally wrong runs.
        rtol=2e-5,
        atol=1e-12,
    ):
        raise ValueError(f"Non-constant or unexpected load increments in {path}.")

    numeric_columns = [
        "total_energy",
        "total_e_change_from_init",
        "avg_sigma12",
        "avg_sigma12_change_from_init",
        "rev_u_diff",
        "rev_energy_diff",
        "rev_sigma_12_diff",
        "participationFraction",
        "m3_participationFraction",
        m3_column,
    ]
    if not np.all(np.isfinite(df[numeric_columns].to_numpy(dtype=float))):
        raise ValueError(f"Non-finite required values in {path}.")

    rev_u = df["rev_u_diff"].to_numpy(dtype=float)[1:]
    if np.any(rev_u < 0):
        raise ValueError(f"Negative rev_u_diff values in {path}.")
    m3_changes = df[m3_column].to_numpy(dtype=float)[1:]
    if np.any(m3_changes < 0) or not np.allclose(m3_changes, np.rint(m3_changes)):
        raise ValueError(f"Invalid m3-change counts in {path}.")
    participation_fraction = df["participationFraction"].to_numpy(dtype=float)[1:]
    m3_participation_fraction = df["m3_participationFraction"].to_numpy(
        dtype=float
    )[1:]
    if np.any(participation_fraction <= 0) or np.any(participation_fraction > 1):
        raise ValueError(f"Participation fractions outside (0, 1] in {path}.")
    if np.any(m3_participation_fraction < 0) or np.any(
        m3_participation_fraction > 1
    ):
        raise ValueError(f"m3 participation fractions outside [0, 1] in {path}.")

    metadata = get_metadata(str(path))
    volume = volume_from_metadata(metadata)
    if volume is None or not np.isfinite(volume) or volume <= 0:
        raise ValueError(f"Could not infer a positive mesh volume for {path}.")
    if not np.isclose(volume, config.rows * config.cols):
        raise ValueError(
            f"Metadata/config volume mismatch for {path}: "
            f"{volume} vs {config.rows * config.cols}."
        )

    energy_steps, _ = calculate_energy_step_data(
        path,
        df=df,
        metadata=metadata,
        average_energy=False,
    )
    stress_steps, _ = calculate_stress_step_data(path, df=df)
    energy_drop_density = energy_steps[
        "stress_corrected_drop_second_order"
    ].to_numpy(dtype=float) / volume
    stress_drop = stress_steps["stress_corrected_drop"].to_numpy(dtype=float)
    if not np.all(np.isfinite(energy_drop_density)):
        raise ValueError(f"Non-finite second-order corrected energy drops in {path}.")
    if not np.all(np.isfinite(stress_drop)):
        raise ValueError(f"Non-finite corrected stress drops in {path}.")

    gamma = load[1:]
    yield_load = float(findPrePostSplit(df=df))
    return SampleData(
        path=path,
        batch=batch,
        seed=int(config.seed),
        load_increment=float(config.loadIncrement),
        eps_x=float(config.LBFGSEpsx),
        volume=float(volume),
        rev_u_cut=np.nan,
        gamma=gamma,
        post_yield=gamma > yield_load,
        rev_u=rev_u,
        rev_energy_density=df["rev_energy_diff"].to_numpy(dtype=float)[1:] / volume,
        rev_sigma=df["rev_sigma_12_diff"].to_numpy(dtype=float)[1:],
        energy_drop_density=energy_drop_density,
        stress_drop=stress_drop,
        relaxation_energy=-df["total_e_change_from_init"].to_numpy(dtype=float)[1:],
        m3_changes=np.rint(m3_changes).astype(np.int32),
        participation_fraction=participation_fraction,
        m3_participation_fraction=m3_participation_fraction,
    )


def unbinned_log_otsu_cut(
    values: np.ndarray,
    *,
    min_class_fraction: float = MIN_CLASS_FRACTION,
) -> tuple[float, dict]:
    """Split two populations by exact Otsu variance minimization in log space."""
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 50:
        raise ValueError(f"Need at least 50 positive cycle distances, got {values.size}.")
    if not 0 < min_class_fraction < 0.5:
        raise ValueError("min_class_fraction must lie strictly between zero and 0.5.")

    log_values = np.log10(values)
    split_counts = np.arange(1, values.size)
    lower_fraction = split_counts / values.size
    distinct = values[:-1] < values[1:]
    valid = (
        distinct
        & (lower_fraction >= min_class_fraction)
        & (lower_fraction <= 1 - min_class_fraction)
    )
    if not np.any(valid):
        raise ValueError("No valid distinct two-population split candidates.")

    cumulative = np.cumsum(log_values)
    lower_mean = cumulative[:-1] / split_counts
    upper_mean = (cumulative[-1] - cumulative[:-1]) / (values.size - split_counts)
    between_variance = (
        lower_fraction
        * (1 - lower_fraction)
        * (lower_mean - upper_mean) ** 2
    )
    valid_indices = np.flatnonzero(valid)
    split_index = int(valid_indices[np.argmax(between_variance[valid])])
    cut = float(np.sqrt(values[split_index] * values[split_index + 1]))
    details = {
        "method": "unbinned_log_otsu",
        "cut": cut,
        "recorded_cycles": int(values.size),
        "closing_count": split_index + 1,
        "nonclosing_count": int(values.size - split_index - 1),
        "closing_fraction": float((split_index + 1) / values.size),
        "log10_gap_at_cut": float(log_values[split_index + 1] - log_values[split_index]),
        "between_class_variance": float(between_variance[split_index]),
    }
    return cut, details


def load_batch(batch: int) -> list[SampleData]:
    configs, labels = sylvainBatches(batch)
    grouped_configs, grouped_labels, _ = ConfigGenerator.group_by_settings(
        configs, labels=labels
    )
    paths, _ = get_csv_files(
        grouped_configs,
        labels=grouped_labels,
        useOldFiles=False,
        forceUpdate=False,
    )
    if not paths:
        raise RuntimeError(f"No CSV files found for Sylvain batch {batch}.")

    config_by_name = {config.name: config for config in configs}
    resolved_paths = [Path(path) for group in paths for path in group]
    resolved_names = {_job_name(path) for path in resolved_paths}
    missing = sorted(set(config_by_name) - resolved_names)
    unexpected = sorted(resolved_names - set(config_by_name))
    if missing or unexpected:
        raise RuntimeError(
            f"Batch {batch} path/config mismatch. Missing={missing}, "
            f"unexpected={unexpected}."
        )

    samples = []
    for index, path in enumerate(resolved_paths, start=1):
        print(f"Batch {batch}: reading sample {index}/{len(resolved_paths)}: {path.name}")
        samples.append(_load_sample(path, config_by_name[_job_name(path)], batch))
        gc.collect()

    setting_attribute = "eps_x" if batch == -2 else "load_increment"
    classified_samples = []
    for setting, setting_samples in _setting_groups(samples, setting_attribute).items():
        recorded_distances = np.concatenate(
            [sample.rev_u[sample.cycle_recorded] for sample in setting_samples]
        )
        cut, details = unbinned_log_otsu_cut(recorded_distances)
        print(
            f"{setting_attribute}={setting:g}: rev_u cut={cut:.3e}, "
            f"closing={details['closing_count']}, "
            f"non-closing={details['nonclosing_count']}"
        )
        classified_samples.extend(
            replace(sample, rev_u_cut=cut) for sample in setting_samples
        )
    return sorted(
        classified_samples,
        key=lambda sample: (float(getattr(sample, setting_attribute)), sample.seed),
    )


def _setting_groups(samples: list[SampleData], attribute: str):
    grouped = {}
    for sample in samples:
        value = float(getattr(sample, attribute))
        grouped.setdefault(value, []).append(sample)
    groups = dict(sorted(grouped.items()))
    seed_sets = [{sample.seed for sample in group} for group in groups.values()]
    if len({frozenset(seeds) for seeds in seed_sets}) != 1:
        raise ValueError(f"Settings do not contain matching seed sets: {seed_sets}")
    if not seed_sets or len(seed_sets[0]) < 2:
        raise ValueError("At least two samples per setting are required for a sample SD.")
    return groups


def _setting_label(attribute: str, value: float) -> str:
    key = "LBFGSEpsx" if attribute == "eps_x" else "loadIncrement"
    return pretty_variant_label(f"{key}={value:g}")


def _is_used_setting(attribute: str, value: float) -> bool:
    target = USED_EPS_X if attribute == "eps_x" else USED_DELTA_GAMMA
    return bool(np.isclose(value, target, rtol=1e-12, atol=0.0))


def _setting_linestyle(attribute: str, value: float) -> str:
    return "--" if _is_used_setting(attribute, value) else "-"


def _add_used_marker_ring(ax, x, y, *, size=52, alpha=1.0):
    ax.scatter(
        x,
        y,
        marker="o",
        s=size,
        facecolors="none",
        edgecolors="black",
        linewidths=0.55,
        alpha=alpha,
        zorder=8,
    )


def _colors(n: int) -> list[str]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        raise RuntimeError("Matplotlib default color cycle is empty.")
    return [cycle[index % len(cycle)] for index in range(n)]


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return x, x
    sample_size = x.size
    if x.size > MAX_ECDF_POINTS:
        indices = np.unique(np.linspace(0, x.size - 1, MAX_ECDF_POINTS).astype(int))
        x = x[indices]
        y = (indices + 1) / sample_size
    else:
        y = np.arange(1, x.size + 1) / x.size
    return x, y


def _ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return x, x
    if x.size > MAX_ECDF_POINTS:
        indices = np.unique(np.linspace(0, x.size - 1, MAX_ECDF_POINTS).astype(int))
        y = (x.size - indices) / x.size
        x = x[indices]
    else:
        y = np.arange(x.size, 0, -1) / x.size
    return x, y


def _cut_note(fig) -> None:
    fig.text(
        0.995,
        0.008,
        r"Classifier: setting-wise unbinned log-Otsu split of $\Delta_{\mathrm{rev}}\mathbf{u}$",
        ha="right",
        va="bottom",
        fontsize="small",
    )


def _save(fig, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _cut_note(fig)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
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


def _setting_handles(groups, attribute, colors):
    return [
        Line2D(
            [],
            [],
            color=color,
            linestyle=_setting_linestyle(attribute, setting),
            marker="o" if _is_used_setting(attribute, setting) else None,
            markerfacecolor=color,
            markeredgecolor="black" if _is_used_setting(attribute, setting) else color,
            markeredgewidth=0.55,
            markersize=7.5 if _is_used_setting(attribute, setting) else 6.0,
            label=_setting_label(attribute, setting),
        )
        for color, setting in zip(colors, groups)
    ]


_PARTICIPATION_CATEGORIES = (
    (
        "no_m3",
        r"no $m_3$ change (elastic candidate)",
        "C0",
        "o",
    ),
    (
        "closing_m3",
        r"closing, $m_3$ change",
        "C2",
        "x",
    ),
    (
        "nonclosing_m3",
        r"non-closing, $m_3$ change",
        "C3",
        "s",
    ),
    (
        "unrecorded_m3",
        r"$m_3$ change, no recorded reverse cycle",
        "C1",
        "+",
    ),
)


def _participation_masks(sample: SampleData) -> dict[str, np.ndarray]:
    changed = sample.m3_changes > 0
    recorded = sample.cycle_recorded
    return {
        "no_m3": ~changed,
        "closing_m3": sample.closing_cycle & changed,
        "nonclosing_m3": sample.avalanche & changed,
        "unrecorded_m3": changed & ~recorded,
    }


def _sampled_indices(mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size <= MAX_SCATTER_POINTS_PER_SAMPLE_CLASS:
        return indices
    positions = np.linspace(
        0,
        indices.size - 1,
        MAX_SCATTER_POINTS_PER_SAMPLE_CLASS,
    ).astype(int)
    return indices[np.unique(positions)]


def _participation_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="-",
            markersize=6,
            label=label,
        )
        for _, label, color, marker in _PARTICIPATION_CATEGORIES
    ]


def _used_setting_samples(
    samples: list[SampleData], attribute: str
) -> list[SampleData]:
    matches = [
        setting_samples
        for setting, setting_samples in _setting_groups(samples, attribute).items()
        if _is_used_setting(attribute, setting)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one used {attribute} setting, got {len(matches)}.")
    return matches[0]


def plot_participation_vs_rev_u(
    samples: list[SampleData], attribute: str, name: str
):
    groups = _setting_groups(samples, attribute)
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(3.1 * len(groups), 3.8),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (setting, setting_samples) in zip(axes, groups.items()):
        cuts = {sample.rev_u_cut for sample in setting_samples}
        if len(cuts) != 1:
            raise ValueError(f"Inconsistent cuts for {attribute}={setting}: {cuts}")
        for sample in setting_samples:
            masks = _participation_masks(sample)
            for key, _, color, marker in _PARTICIPATION_CATEGORIES:
                indices = _sampled_indices(masks[key])
                valid = sample.rev_u[indices] > 0
                ax.scatter(
                    sample.rev_u[indices][valid],
                    sample.participation_fraction[indices][valid],
                    color=color,
                    marker=marker,
                    s=9,
                    linewidths=0.45 if marker in {"x", "+"} else 0,
                    alpha=SCATTER_ALPHA,
                    rasterized=True,
                    zorder=1,
                )
        ax.axvline(
            cuts.pop(),
            color="black",
            linestyle=":",
            linewidth=1.0,
            zorder=3,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(_setting_label(attribute, setting))
        ax.set_xlabel(r"$\Delta_{\mathrm{rev}}\mathbf{u}$")
    axes[0].set_ylabel(r"Participation fraction $P$")
    axes[0].legend(
        handles=_participation_handles()
        + [
            Line2D(
                [],
                [],
                color="black",
                linestyle=":",
                label=r"$\Delta_{\rm rev}\mathbf{u}$ cut",
            )
        ],
        loc="upper left",
        ncol=1,
        frameon=True,
    )
    return _save(fig, name)


def plot_participation_summary(
    samples: list[SampleData], attribute: str, name: str
):
    rows = []
    for sample in samples:
        masks = _participation_masks(sample)
        for post_yield in (False, True):
            for key, label, _, _ in _PARTICIPATION_CATEGORIES:
                values = sample.participation_fraction[
                    masks[key] & (sample.post_yield == post_yield)
                ]
                rows.append(
                    {
                        "setting": float(getattr(sample, attribute)),
                        "seed": sample.seed,
                        "post_yield": post_yield,
                        "category": key,
                        "category_label": label,
                        "count": int(values.size),
                        "q10": float(np.quantile(values, 0.1)) if values.size else np.nan,
                        "median": float(np.median(values)) if values.size else np.nan,
                        "q90": float(np.quantile(values, 0.9)) if values.size else np.nan,
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / f"{name}.csv", index=False)

    settings = np.asarray(sorted(table["setting"].unique()), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, post_yield in zip(axes, (False, True)):
        for key, label, color, marker in _PARTICIPATION_CATEGORIES:
            means = []
            stds = []
            for setting in settings:
                values = table.loc[
                    (table["setting"] == setting)
                    & (table["post_yield"] == post_yield)
                    & (table["category"] == key),
                    "median",
                ].dropna()
                means.append(float(values.mean()) if values.size else np.nan)
                stds.append(float(values.std(ddof=1)) if values.size >= 2 else 0.0)
            means = np.asarray(means)
            stds = np.asarray(stds)
            finite = np.isfinite(means)
            if not finite.any():
                print(
                    f"No {key} events for {attribute}, post_yield={post_yield}; "
                    "omitting that summary line."
                )
                continue
            ax.errorbar(
                settings[finite],
                means[finite],
                yerr=stds[finite],
                color=color,
                marker=marker,
                linewidth=1.2,
                capsize=2.5,
                label=label,
            )
            used = np.flatnonzero(
                [_is_used_setting(attribute, setting) for setting in settings]
            )
            if used.size != 1:
                raise ValueError(f"Expected one used {attribute} setting, got {used}.")
            if finite[used[0]]:
                _add_used_marker_ring(
                    ax,
                    [settings[used[0]]],
                    [means[used[0]]],
                    size=64,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(
            r"$\epsilon_{\mathbf{x}}$"
            if attribute == "eps_x"
            else r"$\Delta\gamma$"
        )
        ax.set_title("Post-yield" if post_yield else "Pre-yield")
    axes[0].set_ylabel("Mean sample median participation fraction")
    axes[0].legend(loc="upper left", ncol=1, frameon=True)
    return _save(fig, name)


def plot_participation_ecdfs(samples: list[SampleData]):
    setting_samples = _used_setting_samples(samples, "eps_x")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharey=True)
    fields = (
        ("participation_fraction", r"Participation fraction $P$"),
        (
            "m3_participation_fraction",
            r"$m_3$-element displacement participation fraction $P_{m_3}$",
        ),
    )

    for row, (field, xlabel) in enumerate(fields):
        for column, post_yield in enumerate((False, True)):
            ax = axes[row, column]
            for key, label, color, marker in _PARTICIPATION_CATEGORIES:
                if field == "m3_participation_fraction" and key not in {
                    "closing_m3",
                    "nonclosing_m3",
                    "unrecorded_m3",
                }:
                    continue
                values = np.concatenate(
                    [
                        getattr(sample, field)[
                            _participation_masks(sample)[key]
                            & (sample.post_yield == post_yield)
                        ]
                        for sample in setting_samples
                    ]
                )
                x, y = _ecdf(values)
                if x.size == 0:
                    print(
                        f"No positive {field} values for {key}, "
                        f"post_yield={post_yield}; omitting that ECDF."
                    )
                    continue
                ax.plot(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    markevery=max(1, x.size // 18),
                    markersize=3.5,
                    linewidth=1.35,
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Empirical CDF")
            if row == 0:
                ax.set_title("Post-yield" if post_yield else "Pre-yield")
    axes[0, 0].legend(loc="upper left", ncol=1, frameon=True)
    return _save(fig, "participation_fraction_ecdf_used_setting")


def plot_participation_event_metrics(samples: list[SampleData]):
    setting_samples = _used_setting_samples(samples, "eps_x")
    metrics = (
        ("energy_drop_density", r"$|\Delta E_S|/V_0$"),
        ("stress_drop", r"$|\Delta\sigma_S|$"),
        ("rev_u", r"$\Delta_{\mathrm{rev}}\mathbf{u}$"),
        ("rev_energy_density", r"$|\Delta_{\mathrm{rev}}E|/V_0$"),
        ("rev_sigma", r"$|\Delta_{\mathrm{rev}}\sigma_{12}|$"),
    )
    fig, axes = plt.subplots(2, len(metrics), figsize=(16.5, 6.8), sharey=True)
    cuts = {sample.rev_u_cut for sample in setting_samples}
    if len(cuts) != 1:
        raise ValueError(f"Inconsistent used-setting rev_u cuts: {cuts}")
    rev_u_cut = cuts.pop()

    for row, post_yield in enumerate((False, True)):
        for column, (field, xlabel) in enumerate(metrics):
            ax = axes[row, column]
            for sample in setting_samples:
                masks = _participation_masks(sample)
                for key, _, color, marker in _PARTICIPATION_CATEGORIES:
                    indices = _sampled_indices(
                        masks[key] & (sample.post_yield == post_yield)
                    )
                    x = np.abs(np.asarray(getattr(sample, field))[indices])
                    y = sample.participation_fraction[indices]
                    valid = (x > 0) & (y > 0)
                    ax.scatter(
                        x[valid],
                        y[valid],
                        color=color,
                        marker=marker,
                        s=9,
                        linewidths=0.45 if marker in {"x", "+"} else 0,
                        alpha=SCATTER_ALPHA,
                        rasterized=True,
                        zorder=1,
                    )
            if field == "rev_u":
                ax.axvline(
                    rev_u_cut,
                    color="black",
                    linestyle=":",
                    linewidth=1.0,
                    zorder=3,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
        axes[row, 0].set_ylabel(
            ("Post-yield\n" if post_yield else "Pre-yield\n")
            + "participation fraction"
        )
    axes[0, 0].legend(
        handles=_participation_handles(),
        loc="upper left",
        ncol=1,
        frameon=True,
    )
    return _save(fig, "participation_fraction_vs_event_metrics_used_setting")


def plot_m3_participation_diagnostics(samples: list[SampleData]):
    setting_samples = _used_setting_samples(samples, "eps_x")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    m3_categories = {
        "closing_m3": (r"closing, $m_3$ change", "C2"),
        "nonclosing_m3": (r"non-closing, $m_3$ change", "C3"),
        "unrecorded_m3": (r"$m_3$ change, no reverse cycle", "C1"),
    }

    for sample in setting_samples:
        masks = _participation_masks(sample)
        for key, (_, color) in m3_categories.items():
            for post_yield, marker in ((False, "^"), (True, "o")):
                indices = _sampled_indices(
                    masks[key] & (sample.post_yield == post_yield)
                )
                x_values = (
                    sample.m3_changes[indices].astype(float) / sample.volume,
                    sample.m3_participation_fraction[indices],
                )
                for ax, x in zip(axes, x_values):
                    valid = (x > 0) & (sample.participation_fraction[indices] > 0)
                    ax.scatter(
                        x[valid],
                        sample.participation_fraction[indices][valid],
                        color=color,
                        marker=marker,
                        s=12,
                        linewidths=0,
                        alpha=SCATTER_ALPHA,
                        rasterized=True,
                    )
    axes[0].plot(
        [1e-5, 1],
        [1e-5, 1],
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=r"guide $P=n_{m_3}/N$",
    )
    axes[0].set_xlabel(r"Fraction of mesh elements with an $m_3$ change $n_{m_3}/N$")
    axes[1].set_xlabel(r"$m_3$-element displacement participation $P_{m_3}$")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"Participation fraction $P$")
    axes[0].legend(
        handles=[
            Line2D([], [], color=color, marker="s", linestyle="None", label=label)
            for label, color in m3_categories.values()
        ]
        + [
            Line2D([], [], color="black", marker="^", linestyle="None", label="pre-yield"),
            Line2D([], [], color="black", marker="o", linestyle="None", label="post-yield"),
            Line2D([], [], color="black", linestyle=":", label=r"guide $P=n_{m_3}/N$"),
        ],
        loc="upper left",
        ncol=1,
        frameon=True,
    )
    return _save(fig, "m3_participation_diagnostics_used_setting")


def plot_classifier_cut_vs_setting(
    samples: list[SampleData],
    attribute: str,
    name: str,
):
    groups = _setting_groups(samples, attribute)
    settings = np.asarray(list(groups), dtype=float)
    cuts = []
    for setting, setting_samples in groups.items():
        setting_cuts = {sample.rev_u_cut for sample in setting_samples}
        if len(setting_cuts) != 1:
            raise ValueError(f"Inconsistent cuts for {attribute}={setting}: {setting_cuts}")
        cuts.append(setting_cuts.pop())
    cuts = np.asarray(cuts, dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.plot(
        settings,
        cuts,
        color="C0",
        marker="o",
        linewidth=1.3,
        label="unbinned log-Otsu cut",
    )
    used_indices = [
        index
        for index, setting in enumerate(settings)
        if _is_used_setting(attribute, setting)
    ]
    if len(used_indices) != 1:
        raise ValueError(f"Expected one used {attribute} setting, got {used_indices}.")
    used_index = used_indices[0]
    _add_used_marker_ring(
        ax,
        [settings[used_index]],
        [cuts[used_index]],
        size=68,
    )
    for setting, cut in zip(settings, cuts):
        ax.annotate(
            f"{cut:.2e}",
            (setting, cut),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize="small",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        r"$\epsilon_{\mathbf{x}}$"
        if attribute == "eps_x"
        else r"$\Delta\gamma$"
    )
    ax.set_ylabel(r"setting-specific $\Delta_{\mathrm{rev}}\mathbf{u}$ cut")
    ax.legend(loc="upper left", ncol=1, frameon=True)
    return _save(fig, name)


def _plot_powerlaw_ks_scan(fit, delta_gamma: float) -> Path:
    analysis = getattr(fit, "xmin_analysis", None)
    if analysis is None:
        raise ValueError(f"Missing xmin analysis for Delta-gamma={delta_gamma:g}.")
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    plot_xmin_analysis(analysis, ax=ax)
    ax.set_xlabel(r"$\Delta E_{\min}/V_0$")
    ax.set_ylabel(r"$D$")
    ax.set_title(_setting_label("load_increment", delta_gamma))
    setting_tag = f"{delta_gamma:.0e}".replace("+", "")
    return _save(fig, f"ks_distance_vs_delta_E_min_delta_gamma_{setting_tag}")


def plot_threshold_diagnostics(samples: list[SampleData], attribute: str, name: str):
    groups = _setting_groups(samples, attribute)
    colors = _colors(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    for color, (setting, setting_samples) in zip(colors, groups.items()):
        cuts = {sample.rev_u_cut for sample in setting_samples}
        if len(cuts) != 1:
            raise ValueError(f"Inconsistent rev_u cuts for {attribute}={setting}: {cuts}")
        cut = cuts.pop()
        recorded_u = np.concatenate(
            [sample.rev_u[sample.cycle_recorded] for sample in setting_samples]
        )
        x, y = _ecdf(recorded_u)
        axes[0].plot(
            x,
            y,
            color=color,
            linestyle=_setting_linestyle(attribute, setting),
            linewidth=1.5,
        )
        for ax in axes:
            ax.axvline(cut, color=color, linestyle=":", linewidth=1.0, alpha=0.8)

        for post_yield, marker in ((False, "^"), (True, "o")):
            for sample in setting_samples:
                selected = sample.cycle_recorded & (sample.post_yield == post_yield)
                x_values = sample.rev_u[selected]
                for ax, y_values in (
                    (axes[1], np.abs(sample.rev_energy_density[selected])),
                    (axes[2], np.abs(sample.rev_sigma[selected])),
                ):
                    valid = (x_values > 0) & (y_values > 0)
                    ax.scatter(
                        x_values[valid],
                        y_values[valid],
                        marker=marker,
                        s=18,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=0.8,
                        alpha=SCATTER_ALPHA,
                    )
    for ax in axes:
        ax.set_xscale("log")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].set_xlabel(r"$\Delta_{\mathrm{rev}}\mathbf{u}$")
    axes[0].set_ylim(0, 1)
    axes[1].set_xlabel(r"$\Delta_{\mathrm{rev}}\mathbf{u}$")
    axes[1].set_ylabel(r"$|\Delta_{\mathrm{rev}}E|/V_0$")
    axes[2].set_xlabel(r"$\Delta_{\mathrm{rev}}\mathbf{u}$")
    axes[2].set_ylabel(r"$|\Delta_{\mathrm{rev}}\sigma_{12}|$")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")

    handles = _setting_handles(groups, attribute, colors) + [
        Line2D([], [], marker="^", linestyle="None", color="black", label="pre-yield"),
        Line2D([], [], marker="o", linestyle="None", color="black", label="post-yield"),
        Line2D(
            [],
            [],
            color="black",
            linestyle=":",
            label=r"setting-specific $\Delta_{\rm rev}u$ cut",
        ),
    ]
    axes[0].legend(handles=handles, loc="upper left", ncol=1, frameon=True)
    return _save(fig, name)


_CLOSURE_METRICS = (
    ("rev_u", r"$\Delta_{\mathrm{rev}}\mathbf{u}$", 1.0),
    ("rev_energy_density", r"$|\Delta_{\mathrm{rev}}E|/V_0$", 2.0),
    ("rev_sigma", r"$|\Delta_{\mathrm{rev}}\sigma_{12}|$", 1.0),
)


def _sample_closing_values(sample: SampleData, field: str) -> np.ndarray:
    values = np.asarray(getattr(sample, field), dtype=float)[sample.closing_cycle]
    return values if field == "rev_u" else np.abs(values)


def plot_closing_quantile_scaling(samples: list[SampleData]):
    groups = _setting_groups(samples, "eps_x")
    eps_values = np.asarray(list(groups), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    rows = []

    for ax, (field, label, expected_power) in zip(axes, _CLOSURE_METRICS):
        for q_index, quantile in enumerate(QUANTILES):
            means, stds = [], []
            for eps_x, setting_samples in groups.items():
                sample_quantiles = []
                for sample in setting_samples:
                    values = _sample_closing_values(sample, field)
                    if values.size == 0:
                        raise RuntimeError(
                            f"No closing-cycle {field} values for {sample.path}."
                        )
                    value = float(np.quantile(values, quantile))
                    if not np.isfinite(value) or value <= 0:
                        raise ValueError(
                            f"Non-positive Q{quantile:g}({field}) for {sample.path}: {value}."
                        )
                    sample_quantiles.append(value)
                    rows.append(
                        {
                            "field": field,
                            "quantile": quantile,
                            "eps_x": eps_x,
                            "seed": sample.seed,
                            "value": value,
                        }
                    )
                means.append(np.mean(sample_quantiles))
                stds.append(np.std(sample_quantiles, ddof=1))
                ax.scatter(
                    np.full(len(sample_quantiles), eps_x),
                    sample_quantiles,
                    color=f"C{q_index}",
                    alpha=0.22,
                    s=14,
                )
            means = np.asarray(means)
            stds = np.asarray(stds)
            beta, intercept = np.polyfit(np.log(eps_values), np.log(means), 1)
            ax.errorbar(
                eps_values,
                means,
                yerr=stds,
                color=f"C{q_index}",
                marker="o" if quantile == 0.5 else "s",
                capsize=3,
                label=rf"$Q_{{{quantile:g}}}$, fit $\beta={beta:.2f}$",
            )
            used_indices = [
                index
                for index, eps_x in enumerate(eps_values)
                if _is_used_setting("eps_x", eps_x)
            ]
            if len(used_indices) != 1:
                raise ValueError(f"Expected one used epsilon setting, got {used_indices}.")
            used_index = used_indices[0]
            _add_used_marker_ring(
                ax,
                [eps_values[used_index]],
                [means[used_index]],
                size=58,
            )
            if quantile == 0.5:
                guide = np.exp(intercept) * eps_values**expected_power
                guide *= means[-1] / guide[-1]
                ax.plot(
                    eps_values,
                    guide,
                    color="black",
                    linestyle=":",
                    label=rf"guide $\epsilon_{{\mathbf{{x}}}}^{{{expected_power:g}}}$",
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\epsilon_{\mathbf{x}}$")
        ax.set_ylabel(label)
        ax.legend(loc="upper left", ncol=1, frameon=True)
    cuts = np.asarray(
        [next(iter({sample.rev_u_cut for sample in group})) for group in groups.values()]
    )
    axes[0].plot(
        eps_values,
        cuts,
        color="black",
        linestyle=":",
        marker="x",
        linewidth=1.0,
        label="classifier cut",
    )
    axes[0].legend(loc="upper left", ncol=1, frameon=True)
    pd.DataFrame(rows).to_csv(TABLE_DIR / "closing_quantiles_by_sample.csv", index=False)
    return _save(fig, "closing_quantile_scaling_vs_epsilon_x")


def plot_closing_collapses(samples: list[SampleData]):
    groups = _setting_groups(samples, "eps_x")
    colors = _colors(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    for color, (eps_x, setting_samples) in zip(colors, groups.items()):
        for ax, (field, label, power) in zip(axes, _CLOSURE_METRICS):
            values = np.concatenate(
                [_sample_closing_values(sample, field) for sample in setting_samples]
            ) / eps_x**power
            x, y = _ecdf(values)
            if x.size == 0:
                raise RuntimeError(f"No positive normalized {field} values at eps={eps_x}.")
            ax.plot(
                x,
                y,
                color=color,
                linestyle=_setting_linestyle("eps_x", eps_x),
                linewidth=1.5,
            )
        cut = next(iter({sample.rev_u_cut for sample in setting_samples}))
        axes[0].axvline(
            cut / eps_x,
            color=color,
            linestyle=":",
            linewidth=0.8,
            alpha=0.5,
        )

    for ax, (_, label, power) in zip(axes, _CLOSURE_METRICS):
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Empirical CDF")
        ax.set_xlabel(label + rf"$/\epsilon_{{\mathbf{{x}}}}^{{{power:g}}}$")
    axes[0].legend(
        handles=_setting_handles(groups, "eps_x", colors),
        loc="upper left",
        ncol=1,
        frameon=True,
    )
    return _save(fig, "closing_population_collapse_vs_epsilon_x")


def _sample_rate_rows(samples: list[SampleData], attribute: str) -> pd.DataFrame:
    rows = []
    for sample in samples:
        avalanche_count = int(np.sum(sample.avalanche))
        rows.append(
            {
                "setting": float(getattr(sample, attribute)),
                "seed": sample.seed,
                "steps": sample.gamma.size,
                "exposure": sample.exposure,
                "avalanche_count": avalanche_count,
                "P_av": avalanche_count / sample.gamma.size,
                "rate": avalanche_count / sample.exposure,
                "recorded_cycles": int(np.sum(sample.cycle_recorded)),
                "closing_cycles": int(np.sum(sample.closing_cycle)),
            }
        )
    return pd.DataFrame(rows)


def _rate_summary(samples: list[SampleData], attribute: str, tag: str):
    rows = _sample_rate_rows(samples, attribute)
    summaries = []
    for setting, group in rows.groupby("setting", sort=True):
        pooled_rate = group["avalanche_count"].sum() / group["exposure"].sum()
        mean_rate = group["rate"].mean()
        if np.allclose(group["exposure"], group["exposure"].iloc[0]) and not np.isclose(
            pooled_rate, mean_rate
        ):
            raise RuntimeError(
                f"Pooled and mean rates disagree at {attribute}={setting}: "
                f"{pooled_rate} vs {mean_rate}."
            )
        summaries.append(
            {
                "setting": setting,
                "samples": len(group),
                "total_avalanche_count": int(group["avalanche_count"].sum()),
                "total_exposure": group["exposure"].sum(),
                "pooled_rate": pooled_rate,
                "mean_rate": mean_rate,
                "std_rate": group["rate"].std(ddof=1),
                "mean_P_av": group["P_av"].mean(),
                "std_P_av": group["P_av"].std(ddof=1),
            }
        )
    summary = pd.DataFrame(summaries)
    rows.to_csv(TABLE_DIR / f"{tag}_rates_by_sample.csv", index=False)
    summary.to_csv(TABLE_DIR / f"{tag}_rate_summary.csv", index=False)
    return rows, summary


def plot_global_rate_vs_delta_gamma(samples: list[SampleData]):
    rows, summary = _rate_summary(samples, "load_increment", "delta_gamma")
    x = summary["setting"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    for ax, mean_column, std_column, ylabel in (
        (axes[0], "mean_rate", "std_rate", r"$r=N_{\mathrm{av}}/\gamma_T$"),
        (axes[1], "mean_P_av", "std_P_av", r"$P_{\mathrm{av}}$"),
    ):
        ax.errorbar(
            x,
            summary[mean_column],
            yerr=summary[std_column],
            marker="o",
            color="C0",
            capsize=3,
            label="sample mean $\pm$ SD",
        )
        for setting, group in rows.groupby("setting", sort=True):
            values = group["rate"] if mean_column == "mean_rate" else group["P_av"]
            ax.scatter(
                np.full(len(group), setting),
                values,
                color="C0",
                alpha=0.25,
                s=18,
            )
            if _is_used_setting("load_increment", setting):
                _add_used_marker_ring(
                    ax,
                    np.full(len(group), setting),
                    values,
                    size=36,
                    alpha=0.8,
                )
        used = summary[
            summary["setting"].map(
                lambda value: _is_used_setting("load_increment", value)
            )
        ]
        if len(used) != 1:
            raise ValueError(f"Expected one used Delta-gamma row, got {len(used)}.")
        _add_used_marker_ring(
            ax,
            used["setting"],
            used[mean_column],
            size=64,
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\Delta\gamma$")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", ncol=1, frameon=True)
    axes[1].set_yscale("log")
    guide = summary["mean_P_av"].iloc[0] * x / x[0]
    axes[1].plot(x, guide, color="black", linestyle=":", label=r"guide $\propto\Delta\gamma$")
    axes[1].legend(loc="upper left", ncol=1, frameon=True)
    return _save(fig, "global_avalanche_rate_vs_delta_gamma")


def plot_global_rate_vs_eps_x(samples: list[SampleData]):
    rows, summary = _rate_summary(samples, "eps_x", "epsilon_x")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(
        summary["setting"],
        summary["mean_rate"],
        yerr=summary["std_rate"],
        marker="o",
        color="C0",
        capsize=3,
        label="sample mean $\pm$ SD",
    )
    for setting, group in rows.groupby("setting", sort=True):
        ax.scatter(
            np.full(len(group), setting),
            group["rate"],
            color="C0",
            alpha=0.25,
            s=18,
        )
        if _is_used_setting("eps_x", setting):
            _add_used_marker_ring(
                ax,
                np.full(len(group), setting),
                group["rate"],
                size=36,
                alpha=0.8,
            )
    used = summary[
        summary["setting"].map(lambda value: _is_used_setting("eps_x", value))
    ]
    if len(used) != 1:
        raise ValueError(f"Expected one used epsilon row, got {len(used)}.")
    _add_used_marker_ring(ax, used["setting"], used["mean_rate"], size=64)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon_{\mathbf{x}}$")
    ax.set_ylabel(r"$r=N_{\mathrm{av}}/\gamma_T$")
    ax.legend(loc="upper left", ncol=1, frameon=True)
    return _save(fig, "global_avalanche_rate_vs_epsilon_x")


def _equal_width_edges(target_width: float) -> np.ndarray:
    bins = round((MAX_LOAD - START_LOAD) / target_width)
    if bins < 1:
        raise ValueError(f"Invalid local-rate width {target_width}.")
    return np.linspace(START_LOAD, MAX_LOAD, bins + 1)


def plot_local_rates(samples: list[SampleData]):
    groups = _setting_groups(samples, "load_increment")
    colors = _colors(len(groups))
    fig, axes = plt.subplots(1, len(LOCAL_RATE_WIDTHS), figsize=(13, 4.5), sharey=True)
    output_rows = []

    for ax, target_width in zip(axes, LOCAL_RATE_WIDTHS):
        edges = _equal_width_edges(target_width)
        widths = np.diff(edges)
        centers = (edges[:-1] + edges[1:]) / 2
        for color, (delta_gamma, setting_samples) in zip(colors, groups.items()):
            rates = np.asarray(
                [
                    np.histogram(sample.gamma[sample.avalanche], bins=edges)[0] / widths
                    for sample in setting_samples
                ]
            )
            mean = rates.mean(axis=0)
            std = rates.std(axis=0, ddof=1)
            linestyle = _setting_linestyle("load_increment", delta_gamma)
            for sample, sample_rates in zip(setting_samples, rates):
                ax.plot(
                    centers,
                    sample_rates,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.12,
                    linewidth=0.7,
                )
                for center, width, rate in zip(centers, widths, sample_rates):
                    output_rows.append(
                        {
                            "target_width": target_width,
                            "actual_width": width,
                            "delta_gamma": delta_gamma,
                            "seed": sample.seed,
                            "gamma_center": center,
                            "rate": rate,
                        }
                    )
            ax.plot(
                centers,
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=1.7,
            )
            ax.fill_between(
                centers,
                np.maximum(0, mean - std),
                mean + std,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        ax.set_xlabel(r"$\gamma$")
        ax.set_title(rf"Target width ${target_width:g}$")
    axes[0].set_ylabel(r"$r(\gamma)=\mathrm{d}N_{\mathrm{av}}/\mathrm{d}\gamma$")
    axes[0].legend(
        handles=_setting_handles(groups, "load_increment", colors),
        loc="upper left",
        ncol=1,
        frameon=True,
    )
    pd.DataFrame(output_rows).to_csv(TABLE_DIR / "local_rates_by_sample.csv", index=False)
    return _save(fig, "local_avalanche_rate_vs_strain")


def plot_avalanche_size_ccdf(samples: list[SampleData], attribute: str, name: str):
    groups = _setting_groups(samples, attribute)
    colors = _colors(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    count_rows = []
    fit_rows = []
    energy_handles = []
    quantities = (
        ("energy_drop_density", r"$\Delta E_S/V_0$"),
        ("stress_drop", r"$\Delta\sigma_S$"),
    )

    for color, (setting, setting_samples) in zip(colors, groups.items()):
        linestyle = _setting_linestyle(attribute, setting)
        for ax, (field, xlabel) in zip(axes, quantities):
            raw_values = np.concatenate(
                [getattr(sample, field)[sample.avalanche] for sample in setting_samples]
            )
            values = raw_values[np.isfinite(raw_values) & (raw_values > 0)]
            count_rows.append(
                {
                    "setting_parameter": attribute,
                    "setting": setting,
                    "quantity": field,
                    "classified_avalanches": raw_values.size,
                    "positive_drops": values.size,
                    "nonpositive_drops": int(np.sum(raw_values <= 0)),
                }
            )
            if values.size == 0:
                raise RuntimeError(
                    f"No positive {field} values for {attribute}={setting}."
                )
            x, y = _ccdf(values)
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.35)
            ax.set_xlabel(xlabel)

            if field == "energy_drop_density" and attribute == "load_increment":
                fit = make_fit(
                    values,
                    distType=Truncated_Power_Law,
                    cache_dir=str(OUTPUT_DIR / "fit_cache"),
                    xmin_search_kwargs={"nr_initial": 100, "min_tail_count": 100},
                )
                distribution = dist_from_fit(fit)
                tail_count = int(np.sum(values >= fit.xmin))
                fit_rows.append(
                    {
                        "delta_gamma": setting,
                        "Delta_E_min_over_volume": fit.xmin,
                        "alpha": distribution.alpha,
                        "Lambda": distribution.Lambda,
                        "KS_distance": distribution.D,
                        "positive_avalanche_drops": values.size,
                        "tail_count": tail_count,
                        "xmin_method": "simpleDrop",
                    }
                )
                plot_fit_cdf(
                    ax,
                    fit,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.2,
                    use_ccdf=True,
                    label="_nolegend_",
                    show_legend=False,
                    set_title=False,
                )
                tail_fraction = tail_count / values.size
                ax.scatter(
                    [fit.xmin],
                    [tail_fraction],
                    s=(42 if _is_used_setting(attribute, setting) else 28),
                    marker="o",
                    facecolor=color,
                    edgecolor=(
                        "black" if _is_used_setting(attribute, setting) else color
                    ),
                    linewidth=0.55,
                    zorder=8,
                )
                energy_handles.append(
                    Line2D(
                        [],
                        [],
                        color=color,
                        linestyle=linestyle,
                        marker="o",
                        markerfacecolor=color,
                        markeredgecolor=(
                            "black" if _is_used_setting(attribute, setting) else color
                        ),
                        markeredgewidth=0.55,
                        markersize=(
                            7.5 if _is_used_setting(attribute, setting) else 6.0
                        ),
                        label=(
                            f"{_setting_label(attribute, setting)}: "
                            rf"$\Delta E_{{\min}}/V_0={fit.xmin:.1e}$, "
                            rf"$\hat{{\alpha}}={distribution.alpha:.2f}$"
                        ),
                    )
                )
                _plot_powerlaw_ks_scan(fit, setting)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$P(X\geq x\mid\mathrm{avalanche})$")
    axes[0].set_xlabel(r"$\Delta E_S/V_0$")
    axes[1].set_xlabel(r"$\Delta\sigma_S$")
    axes[0].legend(
        handles=(
            energy_handles
            if energy_handles
            else _setting_handles(groups, attribute, colors)
        ),
        loc="upper left",
        ncol=1,
        frameon=True,
    )
    axes[1].legend(
        handles=_setting_handles(groups, attribute, colors),
        loc="upper left",
        ncol=1,
        frameon=True,
    )
    counts = pd.DataFrame(count_rows)
    counts.to_csv(TABLE_DIR / f"{name}_positive_drop_counts.csv", index=False)
    if fit_rows:
        pd.DataFrame(fit_rows).to_csv(
            TABLE_DIR / "avalanche_energy_powerlaw_fits_vs_delta_gamma.csv",
            index=False,
        )
    nonpositive = counts["nonpositive_drops"].sum()
    if nonpositive:
        print(
            f"{name}: excluded {nonpositive} non-positive corrected drops from log CCDFs; "
            f"counts are recorded in the output CSV."
        )
    return _save(fig, name)


def plot_elastic_energy_collapses(samples: list[SampleData]):
    groups = _setting_groups(samples, "load_increment")
    colors = _colors(len(groups))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharey=True)
    quantities = (
        ("relaxation_energy", r"$\Delta E_R/[N(\Delta\gamma)^2]$", False),
        ("energy_drop_density", r"$|\Delta E_S|/[N(\Delta\gamma)^2]$", True),
    )

    for color, (delta_gamma, setting_samples) in zip(colors, groups.items()):
        for row, (field, xlabel, use_density) in enumerate(quantities):
            for column, post_yield in enumerate((False, True)):
                selected_values = []
                for sample in setting_samples:
                    definitely_elastic = (
                        (sample.m3_changes == 0)
                        & ~sample.avalanche
                        & (sample.post_yield == post_yield)
                    )
                    values = getattr(sample, field)[definitely_elastic]
                    if use_density:
                        normalized = np.abs(values) / delta_gamma**2
                    else:
                        normalized = values / (sample.volume * delta_gamma**2)
                    selected_values.append(normalized)
                x, y = _ecdf(np.concatenate(selected_values))
                if x.size == 0:
                    raise RuntimeError(
                        f"No positive elastic {field} values for dg={delta_gamma}, "
                        f"post_yield={post_yield}."
                    )
                axes[row, column].plot(
                    x,
                    y,
                    color=color,
                    linestyle=_setting_linestyle("load_increment", delta_gamma),
                    linewidth=1.3,
                )

    for row, (_, xlabel, _) in enumerate(quantities):
        for column in range(2):
            axes[row, column].set_xscale("log")
            axes[row, column].set_ylim(0, 1)
            axes[row, column].set_xlabel(xlabel)
            axes[row, column].set_ylabel("Empirical CDF")
    axes[0, 0].set_title("Pre-yield, no m3 change")
    axes[0, 1].set_title("Post-yield, no m3 change")
    axes[0, 0].legend(
        handles=_setting_handles(groups, "load_increment", colors),
        loc="upper left",
        ncol=1,
        frameon=True,
    )
    return _save(fig, "elastic_energy_collapses_vs_delta_gamma")


def _write_classification_summary(samples: list[SampleData], tag: str):
    rows = []
    for sample in samples:
        rows.append(
            {
                "batch": sample.batch,
                "seed": sample.seed,
                "delta_gamma": sample.load_increment,
                "eps_x": sample.eps_x,
                "steps": sample.gamma.size,
                "exposure": sample.exposure,
                "recorded_cycles": int(np.sum(sample.cycle_recorded)),
                "closing_cycles": int(np.sum(sample.closing_cycle)),
                "avalanches": int(np.sum(sample.avalanche)),
                "rev_u_cut": sample.rev_u_cut,
            }
        )
    pd.DataFrame(rows).to_csv(
        TABLE_DIR / f"{tag}_classification_summary.csv", index=False
    )

    attribute = "eps_x" if tag == "epsilon_x" else "load_increment"
    threshold_rows = []
    for setting, setting_samples in _setting_groups(samples, attribute).items():
        recorded = np.concatenate(
            [sample.rev_u[sample.cycle_recorded] for sample in setting_samples]
        )
        cut, details = unbinned_log_otsu_cut(recorded)
        stored_cuts = {sample.rev_u_cut for sample in setting_samples}
        if len(stored_cuts) != 1 or not np.isclose(
            stored_cuts.pop(), cut, rtol=1e-12, atol=0.0
        ):
            raise RuntimeError(f"Stored/recomputed classifier mismatch at {setting}.")
        threshold_rows.append({"setting": setting, **details})
    pd.DataFrame(threshold_rows).to_csv(
        TABLE_DIR / f"{tag}_classification_thresholds.csv", index=False
    )


def _prepare_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for legacy_table in OUTPUT_DIR.glob("*.csv"):
        legacy_table.replace(TABLE_DIR / legacy_table.name)


def _plot_epsilon_participation(samples: list[SampleData]) -> None:
    plot_participation_vs_rev_u(
        samples,
        "eps_x",
        "participation_fraction_vs_rev_u_epsilon_x",
    )
    plot_participation_summary(
        samples,
        "eps_x",
        "participation_fraction_summary_vs_epsilon_x",
    )
    plot_participation_ecdfs(samples)
    plot_participation_event_metrics(samples)
    plot_m3_participation_diagnostics(samples)


def _plot_delta_gamma_participation(samples: list[SampleData]) -> None:
    plot_participation_vs_rev_u(
        samples,
        "load_increment",
        "participation_fraction_vs_rev_u_delta_gamma",
    )
    plot_participation_summary(
        samples,
        "load_increment",
        "participation_fraction_summary_vs_delta_gamma",
    )


def generate_participation_plots() -> None:
    _prepare_output_dirs()
    epsilon_samples = load_batch(-2)
    _plot_epsilon_participation(epsilon_samples)
    del epsilon_samples
    gc.collect()

    delta_gamma_samples = load_batch(-1)
    _plot_delta_gamma_participation(delta_gamma_samples)
    print(f"Completed participation-fraction plots in {OUTPUT_DIR.resolve()}")


def main() -> None:
    _prepare_output_dirs()

    epsilon_samples = load_batch(-2)
    _write_classification_summary(epsilon_samples, "epsilon_x")
    plot_classifier_cut_vs_setting(
        epsilon_samples,
        "eps_x",
        "classifier_cut_vs_epsilon_x",
    )
    plot_threshold_diagnostics(
        epsilon_samples,
        "eps_x",
        "threshold_diagnostics_vs_epsilon_x",
    )
    plot_closing_quantile_scaling(epsilon_samples)
    plot_closing_collapses(epsilon_samples)
    _plot_epsilon_participation(epsilon_samples)
    plot_global_rate_vs_eps_x(epsilon_samples)
    plot_avalanche_size_ccdf(
        epsilon_samples,
        "eps_x",
        "avalanche_drop_ccdf_vs_epsilon_x",
    )
    del epsilon_samples
    gc.collect()

    delta_gamma_samples = load_batch(-1)
    _write_classification_summary(delta_gamma_samples, "delta_gamma")
    plot_classifier_cut_vs_setting(
        delta_gamma_samples,
        "load_increment",
        "classifier_cut_vs_delta_gamma",
    )
    plot_threshold_diagnostics(
        delta_gamma_samples,
        "load_increment",
        "threshold_diagnostics_vs_delta_gamma",
    )
    _plot_delta_gamma_participation(delta_gamma_samples)
    plot_global_rate_vs_delta_gamma(delta_gamma_samples)
    plot_local_rates(delta_gamma_samples)
    plot_avalanche_size_ccdf(
        delta_gamma_samples,
        "load_increment",
        "avalanche_drop_ccdf_vs_delta_gamma",
    )
    plot_elastic_energy_collapses(delta_gamma_samples)
    print(f"Completed numerical-parameter analysis in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
