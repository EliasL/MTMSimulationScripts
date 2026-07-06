from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Plotting.dataFunctions import VTUData, resolve_vtu_files
from Plotting.element_stiffness_spectrum import element_tangent_eigenvalues
from Plotting.stiffness1212Analysis import load_from_vtu


def variant_label(sim_dir: Path) -> str:
    if "edgeFlip" in sim_dir.name:
        return "edge flip"
    return "no reconnection"


def center_element_index(vtu_file: Path) -> int:
    data = VTUData(str(vtu_file))
    connectivity = data.get_connectivity()
    reference_nodes = data.get_reference_nodes()[:, :2]
    centroids = reference_nodes[connectivity].mean(axis=1)
    center = 0.5 * (reference_nodes.min(axis=0) + reference_nodes.max(axis=0))
    return int(np.argmin(np.sum((centroids - center) ** 2, axis=1)))


def collect_simulation(sim_dir: Path, *, loops: int) -> list[dict]:
    files = [Path(path) for path in resolve_vtu_files(sim_dir)]
    if not files:
        raise FileNotFoundError(f"No VTU files found in {sim_dir}")

    element_index = center_element_index(files[0])
    label = variant_label(sim_dir)
    records = []
    for vtu_file in files:
        data = VTUData(str(vtu_file))
        energy = data.get_energy_field()
        if element_index >= len(energy):
            raise ValueError(
                f"Element index {element_index} exceeds {len(energy)} cells in {vtu_file}."
            )

        eigenvalues = element_tangent_eigenvalues(vtu_file, loops=loops)[element_index]
        non_translation = np.sort(np.abs(eigenvalues))[2:]
        records.append(
            {
                "label": label,
                "load": load_from_vtu(str(vtu_file)),
                "element_index": element_index,
                "energy": float(energy[element_index]),
                "max_abs_eigenvalue": float(non_translation[-1]),
                "min_abs_eigenvalue": float(non_translation[0]),
                "condition": float(non_translation[-1] / non_translation[0]),
                "most_negative_eigenvalue": float(eigenvalues[0]),
                "largest_positive_eigenvalue": float(eigenvalues[-1]),
            }
        )
    return sorted(records, key=lambda record: record["load"])


def collect_records(output_root: Path, *, loops: int) -> list[dict]:
    sim_dirs = sorted(path for path in output_root.iterdir() if (path / "data").is_dir())
    if not sim_dirs:
        raise FileNotFoundError(f"No simulation output folders found in {output_root}")

    records = []
    for sim_dir in sim_dirs:
        records.extend(collect_simulation(sim_dir, loops=loops))
    return records


def write_csv(records: list[dict], out_path: Path) -> None:
    if not records:
        raise ValueError("No records to write.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def plot_records(records: list[dict], out_path: Path) -> None:
    if not records:
        raise ValueError("No records to plot.")

    styles = {
        "no reconnection": {"color": "C1", "linestyle": "-"},
        "edge flip": {"color": "C0", "linestyle": "--"},
    }
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.2), sharex=True, constrained_layout=True)

    for label in sorted({record["label"] for record in records}):
        rows = sorted(
            (record for record in records if record["label"] == label),
            key=lambda record: record["load"],
        )
        x = np.array([row["load"] for row in rows], dtype=float)
        kwargs = {
            "marker": "o",
            "markersize": 2.5,
            "linewidth": 1.4,
            "label": label,
            **styles.get(label, {}),
        }
        axes[0].plot(x, [row["energy"] for row in rows], **kwargs)
        axes[1].plot(x, [row["max_abs_eigenvalue"] for row in rows], **kwargs)
        axes[2].plot(x, [row["condition"] for row in rows], **kwargs)

    axes[0].set_ylabel("center element energy")
    axes[1].set_ylabel("center max |lambda|")
    axes[2].set_ylabel("center tangent cond.")
    axes[2].set_xlabel("double-dislocation load")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].set_title("Center element tangent spectrum, double dislocation")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one central element's energy and tangent spectrum."
    )
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--loops", type=int, default=30)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Plots/center_element_spectrum.png"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("Plots/center_element_spectrum.csv"),
    )
    args = parser.parse_args()

    records = collect_records(args.output_root, loops=args.loops)
    write_csv(records, args.csv)
    plot_records(records, args.out)
    print(args.out)
    print(args.csv)


if __name__ == "__main__":
    main()
