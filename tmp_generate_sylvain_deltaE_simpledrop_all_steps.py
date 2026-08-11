#!/usr/bin/env python3
"""Run the three-energy simpleDrop comparison on the original macro samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from Management.updateCSV import read_macrodata_csv
from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import (
    calculate_energy_step_data,
    volume_from_metadata,
)
from tmp_generate_sylvain_deltaE_simpledrop import (
    DROP_LABELS,
    LOAD_SETTINGS,
    PROJECT_ROOT,
    SEEDS,
    _run_simple_drop,
    _save_diagnostic_plots,
    _save_summary_plot,
)


MACRO_ROOT = Path("/Volumes/data/remoteData/macro")
OUTPUT_DIR = PROJECT_ROOT / "Plots/powerLaw/sylvain_deltaE_simpledrop_all_steps"


def _macro_path(load_setting: str, seed: int) -> Path:
    path = MACRO_ROOT / (
        "reversibilityProtocolTest,s100x100l0.14,"
        f"{load_setting},1.0PBCt3LBFGSEpsx1e-06"
        f"energyDropThreshold1e-05s{seed}.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing macro CSV: {path}")
    return path


def _positive(values: np.ndarray, name: str, csv_path: Path) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite {name} values in {csv_path}")
    values = values[values > 0]
    if values.size == 0:
        raise RuntimeError(f"No positive {name} drops in {csv_path}")
    return values


def _macro_drops(csv_path: Path) -> dict[str, np.ndarray]:
    df = read_macrodata_csv(csv_path)
    required = {
        "load",
        "avg_sigma12",
        "total_energy_change",
        "total_e_change_from_init",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing columns {missing} in {csv_path}")

    metadata = get_metadata(str(csv_path))
    volume = volume_from_metadata(metadata)
    if volume is None or not np.isfinite(volume) or volume <= 0:
        raise ValueError(f"Could not infer a positive volume in {csv_path}")

    energy_steps, _ = calculate_energy_step_data(
        str(csv_path), df=df, metadata=metadata, average_energy=False
    )
    if len(energy_steps) != len(df) - 1:
        raise RuntimeError(f"Step-data length mismatch for {csv_path}")

    loads = np.asarray(df["load"], dtype=float)
    stress = np.asarray(df["avg_sigma12"], dtype=float)
    if not np.all(np.isfinite(loads)) or not np.all(np.isfinite(stress)):
        raise ValueError(f"Non-finite load or stress values in {csv_path}")
    yield_load = float(loads[int(np.argmax(stress))])

    # Match get_energy_drops(..., strainLim="auto", postRegime=True):
    # discard the 0.01 strain buffer around the stress maximum and the final
    # endpoint, then retain positive drops for each definition independently.
    row_mask = (loads > yield_load + 1e-2) & (loads < loads.max())
    step_mask = row_mask[1:]
    return {
        "delta_E_S": _positive(
            np.asarray(
                energy_steps["stress_corrected_drop_second_order"], dtype=float
            )[step_mask],
            "delta_E_S",
            csv_path,
        ) / volume,
        "delta_E_I": _positive(
            -np.asarray(df["total_energy_change"], dtype=float)[row_mask],
            "delta_E_I",
            csv_path,
        ) / volume,
        "delta_E_R": _positive(
            -np.asarray(df["total_e_change_from_init"], dtype=float)[row_mask],
            "delta_E_R",
            csv_path,
        ) / volume,
    }


def _collect_distributions() -> dict[str, dict[str, np.ndarray]]:
    distributions = {}
    sample_counts = {}
    for load_setting in LOAD_SETTINGS:
        by_type = {name: [] for name in DROP_LABELS}
        counts = {name: [] for name in DROP_LABELS}
        for seed in SEEDS:
            drops = _macro_drops(_macro_path(load_setting, seed))
            for name in DROP_LABELS:
                by_type[name].append(drops[name])
                counts[name].append(int(drops[name].size))
        distributions[load_setting] = {
            name: np.concatenate(values) for name, values in by_type.items()
        }
        sample_counts[load_setting] = counts
    return distributions, sample_counts


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    distributions, sample_counts = _collect_distributions()
    results = _run_simple_drop(distributions)
    _save_diagnostic_plots(
        results,
        output_dir=OUTPUT_DIR,
        distribution_description="post-yield macro-step drop distribution",
        x_label=r"$\Delta E_{\min}/V_0$",
        sample_counts=sample_counts,
    )
    _save_summary_plot(
        results,
        output_dir=OUTPUT_DIR,
        y_label=r"simpleDrop $\Delta E_{\min}/V_0$",
        title=r"Sylvain batch -1: energy cutoff scaling (macro-step drops)",
        sample_counts=sample_counts,
    )
    print(f"Saved plots and summary to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
