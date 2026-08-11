#!/usr/bin/env python3
"""Plot one shared-dump FIRE/CG/LBFGS minimization event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Management.updateCSV import read_macrodata_csv


COLORS = {"FIRE": "#BD9456", "CG": "#9456BD", "LBFGS": "#56BD94"}


def plot_manifest(manifest_path: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    payload = json.loads(manifest_path.read_text())
    results = {result["algorithm"]: result for result in payload["results"]}
    missing = set(COLORS) - set(results)
    if missing:
        raise ValueError(f"Manifest is missing algorithms: {sorted(missing)}")

    series = []
    for algorithm in ("FIRE", "CG", "LBFGS"):
        directories = results[algorithm]["minimization_directories"]
        if len(directories) != 1:
            raise ValueError(
                f"Expected one retained minimization directory for {algorithm}, "
                f"got {directories}"
            )
        csv_path = Path(directories[0]) / "macroData.csv"
        df = read_macrodata_csv(csv_path, L=100)
        for column in ("nr_func_evals", "total_energy"):
            if column not in df:
                raise KeyError(f"{csv_path} is missing {column}")
        series.append((algorithm, df["nr_func_evals"].to_numpy(float),
                       df["total_energy"].to_numpy(float)))

    common_min = min(float(np.min(energy)) for _, _, energy in series)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    positive = []
    for algorithm, calls, energy in series:
        excess = energy - common_min
        mask = np.isfinite(calls) & np.isfinite(excess) & (excess > 0.0)
        if not np.any(mask):
            raise ValueError(f"No positive common-minimum residual for {algorithm}")
        positive.append(excess[mask])
        ax.plot(calls[mask], excess[mask], color=COLORS[algorithm],
                linewidth=1.5, label=algorithm)
    positive_values = np.concatenate(positive)
    ax.set_yscale("log")
    ax.set_ylim(np.min(positive_values) * 0.8, np.max(positive_values) * 1.2)
    ax.set_xlim(left=0)
    ax.set_xlabel("Number of function calls")
    ax.set_ylabel(r"$E-E_{\min,\,\mathrm{common}}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.14, right=0.97)

    destination = output_dir or manifest_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    stem = "minimization_energy_vs_function_calls"
    pdf_path = destination / f"{stem}.pdf"
    png_path = destination / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    plot_manifest(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
