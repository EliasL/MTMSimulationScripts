"""Plot the normalized energy curves for the non-reconnecting L=100 samples."""

from pathlib import Path
import re

import numpy as np


def _sample_id(csv_path):
    match = re.search(r"s(\d+)$", Path(csv_path).parent.name)
    return match.group(1) if match else "unknown"


def main():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from Management.updateCSV import read_macrodata_csv
    from Plotting.dataFunctions import get_metadata

    data_root = Path("Plots/energy_prediction_normal_data")
    csv_paths = sorted(
        data_root.glob(
            "simpleShear,s100x100l0.15,1e-05,1.0PBCt3LBFGSEpsx1e-06s*/"
            "macroData.csv"
        )
    )
    if not csv_paths:
        raise FileNotFoundError("No non-reconnecting L=100 samples were found.")

    fig, ax = plt.subplots(figsize=(4.329, 2.808))
    colors = ("#6baed6", "#2171b5", "#084594")
    for color, csv_path in zip(colors, csv_paths):
        data = read_macrodata_csv(csv_path)
        metadata = get_metadata(str(csv_path))
        reference_volume = float(metadata["L"]) ** 2
        gamma = np.asarray(data["load"], dtype=float)
        energy = np.asarray(data["total_energy"], dtype=float) / reference_volume
        finite = np.isfinite(gamma) & np.isfinite(energy)
        ax.plot(
            gamma[finite],
            energy[finite],
            color=color,
            linewidth=1.0,
            label=f"Seed {_sample_id(csv_path)}",
        )

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$E/V_0$")
    ax.legend(loc="best", fontsize=8.0)
    fig.tight_layout()

    output_path = Path(
        "Plots/energy_cauchy_a_L100_no_recon_seeds0_1_2_V0norm.pdf"
    )
    fig.savefig(output_path)
    print(f'Plot saved at: "{output_path}"')


if __name__ == "__main__":
    main()
