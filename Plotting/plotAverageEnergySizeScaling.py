"""Plot mean energy with sample P12-peak markers and mean yield strains."""

from __future__ import annotations

from pathlib import Path
import csv
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Management.updateCSV import HEADER_RENAME_MAP
from Management.simulationStatus import _last_complete_csv_row
from Plotting.standardPowerlaw import SIZE_SCALING_MIN_SIZE, yield_peak_load


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Plots/average_energy_size_scaling.csv"
NORMALIZED_CSV = ROOT / "Plots/average_energy_size_scaling_volume_normalized.csv"
OUTPUT_STEM = ROOT / "Plots/average_energy_size_scaling"
SAMPLE_SOURCE_DIR = Path("/Volumes/data/remoteData/macro")
ELEMENTS_PER_REFERENCE_VOLUME = 2.0
MINIMUM_FINAL_LOAD = 0.65
SAMPLES_PER_SIZE = 8


def _sample_paths(system_size: int) -> list[Path]:
    pattern = (
        f"simpleShear,s{system_size}x{system_size}l0.15,1e-05,1.0PBCt*"
        "LBFGSEpsx1e-06s*.csv"
    )
    return sorted(
        (path for path in SAMPLE_SOURCE_DIR.glob(pattern)
         if re.search(r"s\d+\.csv$", path.name)),
        key=lambda path: int(re.search(r"s(\d+)\.csv$", path.name).group(1)),
    )


def _read_sample_curve(path: Path, system_size: int) -> pd.DataFrame:
    # Project each schema onto the required columns. Unrelated diagnostics
    # can appear or disappear when a simulation is resumed.
    columns = ["load", "avg_energy", "avg_P12"]
    records = []
    with path.open(newline="") as stream:
        rows = csv.reader(stream)
        header = next(rows)
        positions = None
        for row in rows:
            if not row:
                continue
            if row[0].startswith("#HEADER:"):
                header = [row[0].split(":", 1)[1].strip(), *row[1:]]
                positions = None
                continue
            if row[0] == header[0]:
                header = row
                positions = None
                continue
            if positions is None:
                normalized = [HEADER_RENAME_MAP.get(h.strip(), h.strip()) for h in header]
                positions = [normalized.index(column) for column in columns]
            records.append([float(row[index]) for index in positions])
    curve = pd.DataFrame(records, columns=columns)
    if curve.empty or not np.isfinite(curve.to_numpy(dtype=float)).all():
        raise ValueError(f"Expected finite energy data in {path}")
    if np.any(np.diff(curve["load"].to_numpy(dtype=float)) <= 0):
        raise ValueError(f"Load must increase strictly in {path}")
    return curve


def plot() -> tuple[Path, Path]:
    frame = pd.read_csv(SOURCE)
    required = {"system_size", "strain", "average_energy", "n_samples"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns in {SOURCE}: {sorted(missing)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.empty or not np.isfinite(frame[list(required)].to_numpy(dtype=float)).all():
        raise ValueError(f"Expected finite nonempty size-scaling data in {SOURCE}")
    if (frame["system_size"] <= 0).any() or (frame["n_samples"] <= 0).any():
        raise ValueError("System sizes and sample counts must be positive.")
    frame = frame.loc[frame["system_size"] >= SIZE_SCALING_MIN_SIZE].copy()
    if frame.empty:
        raise ValueError("No supported system sizes in the energy data.")

    sizes = sorted(frame["system_size"].astype(int).unique())
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.88, len(sizes)))
    fig, ax = plt.subplots(figsize=(7.2, 4.32), constrained_layout=True)
    averages = []
    sample_summary = []
    for size, color in zip(sizes, colors):
        curve = frame.loc[frame["system_size"] == size].sort_values("strain").copy()
        sample_paths = _sample_paths(size)
        if not sample_paths:
            raise FileNotFoundError(
                f"No individual size-scaling samples found for L={size} in "
                f"{SAMPLE_SOURCE_DIR}"
            )
        strains = curve["strain"].to_numpy(dtype=float)
        n_available = np.zeros(len(strains), dtype=int)
        samples = []
        yield_strains = []
        final_loads = {
            path: float(_last_complete_csv_row(path)["load"]) for path in sample_paths
        }
        # Prefer the furthest-loaded runs; stable sorting breaks ties by seed.
        sample_paths.sort(key=lambda path: -final_loads[path])
        for path in sample_paths:
            final_load = final_loads[path]
            included = final_load > MINIMUM_FINAL_LOAD and len(samples) < SAMPLES_PER_SIZE
            sample_summary.append({
                "system_size": size, "source_csv": str(path),
                "final_load": final_load, "included": included,
            })
            if not included:
                continue
            sample = _read_sample_curve(path, size)
            yield_strain = yield_peak_load(sample)
            yield_strains.append(yield_strain)
            x = sample["load"].to_numpy(dtype=float)
            y = ELEMENTS_PER_REFERENCE_VOLUME * sample["avg_energy"].to_numpy(
                dtype=float
            )
            available = (strains >= x[0]) & (strains <= x[-1])
            n_available += available
            yield_energy = np.interp(yield_strain, x, y)
            samples.append(
                (available, np.interp(strains, x, y), yield_strain, yield_energy)
            )
        if len(samples) != SAMPLES_PER_SIZE:
            raise ValueError(
                f"Expected {SAMPLES_PER_SIZE} samples loaded past "
                f"{MINIMUM_FINAL_LOAD} for L={size}, found {len(samples)}"
            )
        mean_energy = np.sum(
            [np.where(available, energy, 0.0) for available, energy, _, _ in samples],
            axis=0,
        )
        covered = n_available > 0
        curve = curve.loc[covered].copy()
        curve["n_samples"] = n_available[covered]
        curve["mean_energy_per_reference_volume"] = (
            mean_energy[covered] / n_available[covered]
        )
        curve["average_energy"] = (
            curve["mean_energy_per_reference_volume"] / ELEMENTS_PER_REFERENCE_VOLUME
        )
        averages.append(curve)
        for available, sample_energy, yield_strain, yield_energy in samples:
            ax.plot(
                strains[available],
                sample_energy[available],
                color=color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.1,
            )
            ax.plot(
                yield_strain, yield_energy,
                marker="x", linestyle="none", color=color, alpha=0.3,
                markersize=5, markeredgewidth=1,
            )
        # Average the individual peak strains, rather than taking a peak
        # of the ensemble-averaged stress curve.
        ax.axvline(np.mean(yield_strains), color=color, linestyle="--", linewidth=1.2)
        ax.plot(
            curve["strain"],
            curve["mean_energy_per_reference_volume"],
            color=color,
            linewidth=1.8,
            label=f"L={size}",
        )

    pd.concat(averages, ignore_index=True).to_csv(NORMALIZED_CSV, index=False)
    pd.DataFrame(sample_summary).to_csv(
        OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_samples.csv"), index=False
    )
    ax.set_xlabel(r"Shear strain $\gamma$")
    ax.set_ylabel(r"$\langle E/V_0\rangle$")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="black", linestyle="--", linewidth=1.2))
    labels.append(r"Average yield $\langle\gamma_c\rangle$")
    ax.legend(handles, labels, title="System size", loc="upper left", frameon=True)
    png_path = OUTPUT_STEM.with_suffix(".png")
    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    for path in plot():
        print(path)
