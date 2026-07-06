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

from MTMath.energyFunction import ContiEnergy
from Plotting.dataFunctions import VTUData, resolve_vtu_files
from Plotting.stiffness1212Analysis import (
    find_simulation_dirs,
    load_from_vtu,
    variant_label,
    variant_style,
)


def target_loads(base_load: float, max_load: float) -> list[float]:
    if base_load <= 0.0:
        raise ValueError("base_load must be positive.")
    if max_load < base_load:
        raise ValueError("max_load must be at least base_load.")
    loads = []
    period = 0
    while base_load + period <= max_load + 1e-12:
        loads.append(round(base_load + period, 12))
        period += 1
    return loads


def vtu_files_at_loads(sim_dir: Path, loads: list[float]) -> list[Path]:
    files = [Path(path) for path in resolve_vtu_files(sim_dir)]
    by_load = {round(load_from_vtu(path), 12): path for path in files}
    missing = [load for load in loads if round(load, 12) not in by_load]
    if missing:
        raise FileNotFoundError(
            f"{sim_dir} is missing VTUs for loads: "
            + ", ".join(f"{load:g}" for load in missing)
        )
    return [by_load[round(load, 12)] for load in loads]


def element_tangent_eigenvalues(vtu_file: Path, *, loops: int) -> np.ndarray:
    data = VTUData(str(vtu_file))
    dN_dX = data.get_dN_dX()
    area_ref = data.get_init_area()
    F = data.get_F()
    if len(F) != len(dN_dX):
        raise ValueError(
            f"F has {len(F)} cells but dN_dX has {len(dN_dX)} in {vtu_file}."
        )
    if len(F) != len(area_ref):
        raise ValueError(
            f"F has {len(F)} cells but initArea has {len(area_ref)} in {vtu_file}."
        )

    tangent = ContiEnergy.elasticity_tensor(F, eulerian=False, loops=loops)
    stiffness = np.einsum(
        "eipjq,eap,ebq,e->eaibj",
        tangent,
        dN_dX,
        dN_dX,
        area_ref,
    ).reshape(len(F), 6, 6)
    stiffness = 0.5 * (stiffness + np.swapaxes(stiffness, -1, -2))
    return np.linalg.eigvalsh(stiffness)


def assert_uniform_element_values(values: np.ndarray, *, name: str, vtu_file: Path) -> None:
    values = np.asarray(values, dtype=float)
    if values.shape[0] == 0:
        raise ValueError(f"{name} has no element values in {vtu_file}.")
    if not np.allclose(values, values[0], rtol=1e-8, atol=1e-10):
        max_deviation = float(np.max(np.abs(values - values[0])))
        raise ValueError(
            f"{name} is not identical for all elements in {vtu_file}; "
            f"max deviation from element 0 is {max_deviation:g}."
        )


def summarize(vtu_file: Path, *, label: str, period: int, load: float, loops: int):
    data = VTUData(str(vtu_file))
    energy = data.get_energy_field()
    assert_uniform_element_values(energy, name="energy", vtu_file=vtu_file)

    eigenvalues = element_tangent_eigenvalues(vtu_file, loops=loops)

    # Two exact translation modes are removed. The remaining four eigenvalues
    # describe the element tangent response to non-translation nodal modes.
    non_translation = np.sort(np.abs(eigenvalues), axis=1)[:, 2:]
    max_abs = non_translation[:, -1]
    min_abs = non_translation[:, 0]
    condition = max_abs / min_abs
    assert_uniform_element_values(max_abs, name="max_abs_eigenvalue", vtu_file=vtu_file)
    assert_uniform_element_values(min_abs, name="min_abs_eigenvalue", vtu_file=vtu_file)
    assert_uniform_element_values(condition, name="tangent_condition", vtu_file=vtu_file)
    assert_uniform_element_values(
        eigenvalues[:, 0], name="most_negative_eigenvalue", vtu_file=vtu_file
    )
    assert_uniform_element_values(
        eigenvalues[:, -1], name="largest_positive_eigenvalue", vtu_file=vtu_file
    )

    return {
        "label": label,
        "period": period,
        "load": load,
        "energy": float(energy[0]),
        "max_abs_eigenvalue": float(max_abs[0]),
        "min_abs_eigenvalue": float(min_abs[0]),
        "tangent_condition": float(condition[0]),
        "most_negative_eigenvalue": float(eigenvalues[0, 0]),
        "largest_positive_eigenvalue": float(eigenvalues[0, -1]),
    }


def collect_records(output_root: Path, *, base_load: float, max_load: float, loops: int):
    loads = target_loads(base_load, max_load)
    records = []
    for sim_dir in find_simulation_dirs(output_root):
        label = variant_label(sim_dir)
        files = vtu_files_at_loads(sim_dir, loads)
        for period, (load, vtu_file) in enumerate(zip(loads, files)):
            records.append(
                summarize(
                    vtu_file,
                    label=label,
                    period=period,
                    load=load,
                    loops=loops,
                )
            )
    return records


def write_csv(records: list[dict], out_path: Path):
    if not records:
        raise ValueError("No records to write.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def plot_records(records: list[dict], out_path: Path):
    if not records:
        raise ValueError("No records to plot.")

    labels = sorted({record["label"] for record in records})
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.8), sharex=True, constrained_layout=True)

    for label in labels:
        rows = sorted(
            (record for record in records if record["label"] == label),
            key=lambda record: record["load"],
        )
        x = np.array([row["load"] for row in rows], dtype=float)
        style = variant_style(label)
        style["linestyle"] = "-" if "edge flip" in label else "--"
        plot_kwargs = {
            "marker": "o",
            "markersize": 3.0,
            "linewidth": 1.6,
            "label": label,
            **style,
        }

        axes[0].plot(
            x,
            [row["energy"] for row in rows],
            **plot_kwargs,
        )
        axes[1].plot(
            x,
            [row["tangent_condition"] for row in rows],
            **plot_kwargs,
        )

    axes[0].set_ylabel("element energy")
    axes[1].set_ylabel("tangent cond.")
    axes[1].set_xlabel("simple shear load, gamma = n + 0.2")
    axes[1].set_yscale("log")
    axes[0].set_title("Element tangent spectrum at periodically identical loads")

    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample element tangent stiffness spectra at periodically identical "
            "loads such as 0.2, 1.2, 2.2, ..."
        )
    )
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--base-load", type=float, default=0.2)
    parser.add_argument("--max-load", type=float, default=11.2)
    parser.add_argument("--loops", type=int, default=30)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Plots/element_stiffness_spectrum_periodic_points.png"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("Plots/element_stiffness_spectrum_periodic_points.csv"),
    )
    args = parser.parse_args()

    records = collect_records(
        args.output_root,
        base_load=args.base_load,
        max_load=args.max_load,
        loops=args.loops,
    )
    write_csv(records, args.csv)
    plot_records(records, args.out)
    print(args.out)
    print(args.csv)


if __name__ == "__main__":
    main()
