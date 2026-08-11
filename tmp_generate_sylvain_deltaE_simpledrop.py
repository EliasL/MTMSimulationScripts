#!/usr/bin/env python3
"""Compare simpleDrop cutoffs for the three Sylvain energy-drop definitions.

The distributions are built from the reversibility event rows, rather than all
macro-data rows.  This is important for ``DeltaE_R``: every minimization has a
relaxation-from-affine-guess value, whereas only event rows are energy drops to
compare between the three definitions.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Management.updateCSV import read_macrodata_csv
from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import (
    calculate_energy_step_data,
    volume_from_metadata,
)
from Plotting.findXmin import analyze_xmin, plot_xmin_analysis
from MTMath.evaluatePowerlawFit import Truncated_Power_Law


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/Volumes/data/remoteData/data")
OUTPUT_DIR = PROJECT_ROOT / "Plots/powerLaw/sylvain_deltaE_simpledrop"

LOAD_SETTINGS = ("0.0001", "5e-05", "1e-05", "5e-06", "1e-06")
SEEDS = range(4)
EVENT_PATTERN = re.compile(r"(?P<kind>rev|irrev)_drop_l_(?P<load>[0-9.eE+-]+)$")
DROP_LABELS = {
    "delta_E_S": r"$\Delta E_S$",
    "delta_E_I": r"$\Delta E_I$",
    "delta_E_R": r"$\Delta E_R$",
}


def _job_dir(load_setting: str, seed: int) -> Path:
    name = (
        "reversibilityProtocolTest,s100x100l0.14,"
        f"{load_setting},1.0PBCt3LBFGSEpsx1e-06"
        f"energyDropThreshold1e-05s{seed}"
    )
    path = DATA_ROOT / name
    if not path.is_dir():
        raise FileNotFoundError(f"Missing Sylvain job directory: {path}")
    return path


def _unique_load_index(
    loads: np.ndarray, target: float, load_increment: float
) -> int:
    candidates = np.flatnonzero(
        np.isclose(
            loads,
            target,
            rtol=1e-9,
            atol=max(1e-12, load_increment * 1e-6),
        )
    )
    if candidates.size != 1:
        raise RuntimeError(
            f"Could not uniquely map event load {target}; "
            f"candidates={candidates.tolist()}"
        )
    return int(candidates[0])


def _event_drops(job_dir: Path, post_yield: bool = True) -> dict[str, np.ndarray]:
    csv_path = job_dir / "macroData.csv"
    event_root = job_dir / "data" / "reversibilityData"
    df = read_macrodata_csv(csv_path)
    required = {
        "load",
        "avg_sigma12",
        "total_energy_change",
        "total_e_change_from_init",
        "is_reversible",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing columns {missing} in {csv_path}")

    metadata = get_metadata(str(csv_path))
    load_increment = float(metadata["loadIncrement"])
    if not np.isfinite(load_increment) or load_increment <= 0:
        raise ValueError(f"Invalid loadIncrement in {csv_path}")
    mesh_volume = volume_from_metadata(metadata)
    if mesh_volume is None or not np.isfinite(mesh_volume) or mesh_volume <= 0:
        raise ValueError(f"Could not infer a positive mesh volume for {csv_path}")
    mesh_volume = float(mesh_volume)

    energy_steps, _ = calculate_energy_step_data(
        str(csv_path), df=df, metadata=metadata, average_energy=False
    )
    if len(energy_steps) != len(df) - 1:
        raise RuntimeError(f"Step-data length mismatch for {csv_path}")

    loads = np.asarray(df["load"], dtype=float)
    yield_load = float(loads[int(np.nanargmax(np.asarray(df["avg_sigma12"], dtype=float)))])
    event_dirs = sorted(path for path in event_root.iterdir() if path.is_dir())
    if not event_dirs:
        raise RuntimeError(f"No event directories found in {event_root}")

    drops = {name: [] for name in DROP_LABELS}
    used_events = 0
    for event_dir in event_dirs:
        match = EVENT_PATTERN.fullmatch(event_dir.name)
        if match is None:
            raise ValueError(f"Unexpected event directory: {event_dir}")
        event_kind = match.group("kind")
        start_load = float(match.group("load"))
        event_load = start_load + load_increment
        if post_yield and not event_load > yield_load:
            continue

        row_index = _unique_load_index(loads, event_load, load_increment)
        if row_index == 0:
            raise RuntimeError(f"Event maps to the first macro row: {event_dir}")
        step_index = row_index - 1
        row = df.iloc[row_index]
        expected_reversible = event_kind == "rev"
        if bool(int(row["is_reversible"])) != expected_reversible:
            raise ValueError(f"Event/macro reversibility mismatch for {event_dir}")

        values = {
            # Same conventions as ClusterJobs/reversibility_postprocess.py.
            "delta_E_S": float(
                energy_steps["stress_corrected_drop_second_order"].iloc[step_index]
            )
            / mesh_volume,
            "delta_E_I": -float(row["total_energy_change"]) / mesh_volume,
            "delta_E_R": -float(row["total_e_change_from_init"]) / mesh_volume,
        }
        for name, value in values.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite {name}={value} at {event_dir} in {csv_path}"
                )
            # The power-law workflow models positive drops only.  A negative
            # value here means this definition did not identify a drop at this
            # event (most commonly for DeltaE_I during the elastic load work).
            if value > 0:
                drops[name].append(value)
        used_events += 1

    if used_events == 0 or any(not values for values in drops.values()):
        raise RuntimeError(f"No post-yield events found in {job_dir}")
    return {name: np.asarray(values, dtype=float) for name, values in drops.items()}


def _collect_distributions() -> dict[str, dict[str, np.ndarray]]:
    distributions = {}
    for load_setting in LOAD_SETTINGS:
        by_type = {name: [] for name in DROP_LABELS}
        for seed in SEEDS:
            job_drops = _event_drops(_job_dir(load_setting, seed))
            for name in DROP_LABELS:
                by_type[name].append(job_drops[name])
        distributions[load_setting] = {
            name: np.concatenate(values) for name, values in by_type.items()
        }
    return distributions


def _run_simple_drop(distributions):
    results = {}
    for load_setting, by_type in distributions.items():
        results[load_setting] = {}
        for name, drops in by_type.items():
            analysis = analyze_xmin(
                drops,
                nr_initial=100,
                min_tail_count=100,
                distType=Truncated_Power_Law,
                parallel=False,
                max_xmin=1e-4,
                refine=False,
            )
            results[load_setting][name] = analysis
    return results


def _save_diagnostic_plots(
    results,
    output_dir: Path = OUTPUT_DIR,
    distribution_description: str = "post-yield event distribution",
    x_label: str = r"$\Delta E_{\min}/V_0$",
    sample_counts=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    x_values = np.asarray([float(value) for value in LOAD_SETTINGS])
    handles = [
        Line2D(
            [0], [0], color="0.65", marker="o", linewidth=1,
            label=r"Displayed only: $n_{tail}<100$",
        ),
        Line2D(
            [0], [0], color="tab:red", marker="o", linewidth=0,
            label=r"Eligible raw $D(x_{\min})$",
        ),
        Line2D(
            [0], [0], color="tab:blue", marker="D", linestyle="none",
            label="simpleDrop",
        ),
        Line2D(
            [0], [0], color="0.25", marker="X", markerfacecolor="white",
            linestyle="none", label="Global minimum",
        ),
    ]
    for name, label in DROP_LABELS.items():
        fig, axes = plt.subplots(1, len(LOAD_SETTINGS), figsize=(18, 4.3), sharey=True)
        for index, (ax, load_setting) in enumerate(zip(axes, LOAD_SETTINGS)):
            plot_xmin_analysis(results[load_setting][name], ax=ax)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            analysis = results[load_setting][name]
            ax.scatter(
                [analysis["simple_drop_xmin"]],
                [analysis["simple_drop_distance"]],
                marker="D",
                s=32,
                color="tab:blue",
                edgecolor="white",
                linewidth=0.5,
                zorder=7,
            )
            ax.scatter(
                [analysis["global_min_xmin"]],
                [analysis["global_min_distance"]],
                marker="X",
                s=40,
                facecolor="white",
                edgecolor="0.25",
                linewidth=0.8,
                zorder=8,
            )
            pooled_count = int(analysis["tail_counts"][0])
            count_text = f"$N={pooled_count}$ positive drops"
            if sample_counts is not None:
                counts = sample_counts[load_setting][name]
                count_text += "\n(" + ", ".join(
                    f"s{seed}={count}" for seed, count in enumerate(counts)
                ) + ")"
            title = (
                rf"$\Delta\gamma={float(load_setting):.0e}$"
                + "\n"
                + count_text
            )
            ax.set_title(title)
            ax.set_xlabel(x_label)
            if index == 0:
                ax.set_ylabel(r"$D$")
        fig.suptitle(
            rf"{label}: KS distance versus cutoff ({distribution_description})",
            y=1.02,
        )
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 0.98))
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        for suffix, save_kwargs in (("png", {"dpi": 180}), ("pdf", {})):
            fig.savefig(output_dir / f"D_vs_cutoff_{name}_post.{suffix}",
                        bbox_inches="tight", **save_kwargs)
        plt.close(fig)


def _save_summary_plot(
    results,
    output_dir: Path = OUTPUT_DIR,
    y_label: str = r"simpleDrop $\Delta E_{\min}/V_0$",
    title: str = r"Sylvain batch -1: energy cutoff scaling",
    sample_counts=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    x_values = np.asarray([float(value) for value in LOAD_SETTINGS])
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    styles = {
        "delta_E_S": ("tab:blue", "o"),
        "delta_E_I": ("tab:orange", "s"),
        "delta_E_R": ("tab:green", "D"),
    }
    rows = []
    for name, label in DROP_LABELS.items():
        y_values = np.asarray(
            [results[setting][name]["simple_drop_xmin"] for setting in LOAD_SETTINGS]
        )
        global_values = np.asarray(
            [results[setting][name]["global_min_xmin"] for setting in LOAD_SETTINGS]
        )
        slope = float(np.polyfit(np.log10(x_values), np.log10(y_values), 1)[0])
        color, marker = styles[name]
        ax.loglog(x_values, y_values, marker=marker, color=color, linewidth=1.5,
                  label=rf"{label}, simpleDrop (slope={slope:.2f})")
        ax.loglog(x_values, global_values, marker=marker, color=color,
                  markerfacecolor="white", markeredgecolor=color,
                  linestyle="--", linewidth=1.1,
                  label=rf"{label}, global min.")
        for setting, y_value in zip(LOAD_SETTINGS, y_values):
            analysis = results[setting][name]
            rows.append({
                "delta_gamma": float(setting),
                "drop_type": name,
                "simpleDrop_delta_E_min_over_V0": float(y_value),
                "D_at_simpleDrop": float(analysis["simple_drop_distance"]),
                "global_delta_E_min_over_V0": float(analysis["global_min_xmin"]),
                "D_at_global_min": float(analysis["global_min_distance"]),
                "positive_drop_count": int(analysis["tail_counts"][0]),
                "loglog_slope": slope,
            })
            if sample_counts is not None:
                for seed, count in enumerate(sample_counts[setting][name]):
                    rows[-1][f"positive_drop_count_seed{seed}"] = int(count)
    ax.set_xlabel(r"$\Delta\gamma$")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix, save_kwargs in (("png", {"dpi": 200}), ("pdf", {})):
        fig.savefig(output_dir / f"deltaE_min_vs_deltaGamma_post.{suffix}",
                    bbox_inches="tight", **save_kwargs)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output_dir / "simpledrop_summary_post.csv", index=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    distributions = _collect_distributions()
    results = _run_simple_drop(distributions)
    _save_diagnostic_plots(results)
    _save_summary_plot(results)
    print(f"Saved plots and summary to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
