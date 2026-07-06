from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "scenario",
    "integer_shear",
    "reconnection",
    "local_load",
    "energy",
    "tangent_condition",
}


def read_records(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header.")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        records = []
        for row in reader:
            row["integer_shear"] = int(float(row["integer_shear"]))
            row["local_load"] = float(row["local_load"])
            row["energy"] = float(row["energy"])
            row["tangent_condition"] = float(row["tangent_condition"])
            records.append(row)

    if not records:
        raise ValueError(f"{csv_path} contains no records.")
    return records


def plot_records(records: list[dict], out_path: Path) -> None:
    scenarios = ["distorted current", "distorted reference"]
    shears = [0, 2, 5, 10]
    reconnections = ["no reconnection", "edge flip"]
    colors = {0: "C0", 2: "C2", 5: "C3", 10: "C4"}
    linestyles = {"edge flip": "-", "no reconnection": "--"}
    zorders = {"edge flip": 4, "no reconnection": 3}

    fig, axes = plt.subplots(
        2, 2, figsize=(10.2, 6.0), sharex=True, constrained_layout=True
    )
    for col, scenario in enumerate(scenarios):
        for shear in shears:
            for reconnection in reconnections:
                rows = sorted(
                    (
                        record
                        for record in records
                        if record["scenario"] == scenario
                        and record["integer_shear"] == shear
                        and record["reconnection"] == reconnection
                    ),
                    key=lambda record: record["local_load"],
                )
                if not rows:
                    raise ValueError(f"Missing {scenario}, n={shear}, {reconnection}.")

                kwargs = {
                    "color": colors[shear],
                    "linestyle": linestyles[reconnection],
                    "linewidth": 1.7,
                    "marker": "o",
                    "markersize": 2.8,
                    "label": f"n={shear}, {reconnection}",
                    "zorder": zorders[reconnection],
                }
                x = [row["local_load"] for row in rows]
                axes[0, col].plot(x, [row["energy"] for row in rows], **kwargs)
                axes[1, col].plot(
                    x, [row["tangent_condition"] for row in rows], **kwargs
                )

        axes[0, col].set_title(scenario)
        axes[1, col].set_xlabel(r"$\gamma-n$")
        axes[1, col].set_yscale("log")

    for ax in axes.ravel():
        ax.grid(True, which="both", alpha=0.25)

    axes[0, 0].set_ylabel(r"$E$")
    axes[1, 0].set_ylabel(r"$\kappa_{\mathrm{tan}}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot element energy and tangent condition from comparison CSV."
    )
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=Path("Plots/no_minimization_current_vs_reference_distortion_direct_fields.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Plots/no_minimization_current_vs_reference_distortion_direct_fields.pdf"),
    )
    args = parser.parse_args()

    records = read_records(args.csv)
    plot_records(records, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
