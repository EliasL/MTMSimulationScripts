from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from MTMath.energyFunction import ContiEnergy
from MTMath.meshUtils import triangle_shape_grads_and_area
from Plotting.dataFunctions import VTUData
from Plotting.element_stiffness_spectrum import (
    assert_uniform_element_values,
    element_tangent_eigenvalues,
    vtu_files_at_loads,
)
from Plotting.stiffness1212Analysis import find_simulation_dirs


def parse_shears(text: str) -> list[int]:
    shears = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not shears:
        raise ValueError("At least one integer shear must be provided.")
    return shears


def reconnection_label(sim_dir: Path) -> str:
    return "edge flip" if "edgeFlip" in sim_dir.name else "no reconnection"


def triangle_connectivity(data: VTUData) -> np.ndarray:
    cell_blocks = data.mesh.cells
    if len(cell_blocks) != 1:
        raise ValueError(
            f"Expected a single VTU cell block in {data.vtu_file_path}, got {len(cell_blocks)}."
        )
    cell_block = cell_blocks[0]
    if cell_block.type != "triangle":
        raise ValueError(
            f"Expected triangle cells in {data.vtu_file_path}, got {cell_block.type!r}."
        )
    connectivity = np.asarray(cell_block.data, dtype=int)
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(
            f"Expected triangle connectivity with shape (elements, 3), got {connectivity.shape}."
        )
    return connectivity


def geometric_condition(data: VTUData) -> np.ndarray:
    points = np.asarray(data.mesh.points[:, :2], dtype=float)
    connectivity = triangle_connectivity(data)
    if np.any(connectivity < 0) or np.any(connectivity >= len(points)):
        raise ValueError(f"Triangle connectivity is out of bounds in {data.vtu_file_path}.")

    coords = points[connectivity]
    dN_dx, area = triangle_shape_grads_and_area(coords)
    if np.any(area <= 0.0):
        raise ValueError(f"All current triangle areas must be positive in {data.vtu_file_path}.")

    scalar_stiffness = area[:, None, None] * np.einsum("eai,ebi->eab", dN_dx, dN_dx)
    eigenvalues = np.linalg.eigvalsh(scalar_stiffness)
    if np.any(np.abs(eigenvalues[:, 0]) > 1e-10 * np.maximum(1.0, eigenvalues[:, -1])):
        raise ValueError(f"Scalar triangle stiffness should have one zero mode in {data.vtu_file_path}.")
    if np.any(eigenvalues[:, 1] <= 0.0):
        raise ValueError(f"Scalar triangle stiffness has non-positive shape mode in {data.vtu_file_path}.")
    return eigenvalues[:, 2] / eigenvalues[:, 1]


def material_condition(F: np.ndarray, *, loops: int) -> np.ndarray:
    tangent = ContiEnergy.elasticity_tensor(F, eulerian=False, loops=loops)
    tangent_matrix = tangent.reshape(len(F), 4, 4)
    singular_values = np.linalg.svd(tangent_matrix, compute_uv=False)
    if np.any(singular_values[:, -1] <= 0.0):
        raise ValueError("Material tangent has a zero singular value.")
    return singular_values[:, 0] / singular_values[:, -1]


def tangent_condition(vtu_file: Path, *, loops: int) -> np.ndarray:
    eigenvalues = element_tangent_eigenvalues(vtu_file, loops=loops)
    non_translation = np.sort(np.abs(eigenvalues), axis=1)[:, 2:]
    return non_translation[:, -1] / non_translation[:, 0]


def summarize(vtu_file: Path, *, loops: int) -> dict[str, float]:
    data = VTUData(str(vtu_file))
    kappa_geo = geometric_condition(data)
    kappa_mat = material_condition(data.get_F(), loops=loops)
    kappa_tan = tangent_condition(vtu_file, loops=loops)

    assert_uniform_element_values(kappa_geo, name="kappa_geo", vtu_file=vtu_file)
    assert_uniform_element_values(kappa_mat, name="kappa_mat", vtu_file=vtu_file)
    assert_uniform_element_values(kappa_tan, name="kappa_tan", vtu_file=vtu_file)
    return {
        "kappa_geo": float(kappa_geo[0]),
        "kappa_mat": float(kappa_mat[0]),
        "kappa_tan": float(kappa_tan[0]),
    }


def collect_records(
    output_root: Path,
    *,
    shears: list[int],
    local_loads: list[float],
    loops: int,
) -> list[dict]:
    records = []
    for sim_dir in find_simulation_dirs(output_root):
        reconnection = reconnection_label(sim_dir)
        for shear in shears:
            absolute_loads = [round(shear + local_load, 12) for local_load in local_loads]
            files = vtu_files_at_loads(sim_dir, absolute_loads)
            for local_load, absolute_load, vtu_file in zip(
                local_loads, absolute_loads, files
            ):
                record = {
                    "integer_shear": shear,
                    "reconnection": reconnection,
                    "local_load": local_load,
                    "absolute_load": absolute_load,
                }
                record.update(summarize(vtu_file, loops=loops))
                records.append(record)
    return sorted(
        records,
        key=lambda row: (row["integer_shear"], row["reconnection"], row["local_load"]),
    )


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

    quantities = [
        (
            "kappa_geo",
            r"$\kappa_{\mathrm{geo}}$",
            "current element geometry",
            (1e0, 1e5),
        ),
        (
            "kappa_mat",
            r"$\kappa_{\mathrm{mat}}$",
            "material tangent",
            (1e3, 1e8),
        ),
        ("kappa_tan", r"$\kappa_{\mathrm{tan}}$", "element tangent", (1e3, 1e8)),
    ]
    shears = sorted({int(row["integer_shear"]) for row in records})
    reconnections = ["no reconnection", "edge flip"]
    colors = {0: "C0", 2: "C2", 5: "C3", 10: "C4"}
    linestyles = {"edge flip": "-", "no reconnection": "--"}
    zorders = {"edge flip": 4, "no reconnection": 3}

    fig, axes = plt.subplots(3, 1, figsize=(5.2, 6.8), sharex=True, constrained_layout=True)
    for ax, (field, ylabel, title, ylim) in zip(axes, quantities):
        for shear in shears:
            for reconnection in reconnections:
                rows = sorted(
                    (
                        row
                        for row in records
                        if int(row["integer_shear"]) == shear
                        and row["reconnection"] == reconnection
                    ),
                    key=lambda row: row["local_load"],
                )
                if not rows:
                    raise ValueError(f"Missing n={shear}, {reconnection}.")
                ax.plot(
                    [row["local_load"] for row in rows],
                    [row[field] for row in rows],
                    color=colors.get(shear, "C0"),
                    linestyle=linestyles[reconnection],
                    linewidth=1.7,
                    marker="o",
                    markersize=2.8,
                    zorder=zorders[reconnection],
                )
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[-1].set_xlabel(r"$\gamma-n$")
    shear_handles = [
        Line2D(
            [0],
            [0],
            color=colors.get(shear, "C0"),
            linestyle="-",
            marker="o",
            markersize=2.8,
            linewidth=1.7,
            label=str(shear),
        )
        for shear in shears
    ]
    connectivity_handles = [
        Line2D([0], [0], color="0.2", linestyle="-", linewidth=1.7, label="edge flip"),
        Line2D([0], [0], color="0.2", linestyle="--", linewidth=1.7, label="fixed conn."),
    ]
    shear_legend = axes[0].legend(
        handles=shear_handles,
        title=r"$n$",
        loc="upper left",
        ncol=2,
        fontsize=7.5,
        title_fontsize=8,
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        handlelength=1.5,
        columnspacing=0.8,
    )
    axes[0].add_artist(shear_legend)
    axes[0].legend(
        handles=connectivity_handles,
        title="connectivity",
        loc="upper right",
        fontsize=7.5,
        title_fontsize=8,
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        handlelength=1.7,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose element conditioning into current element geometry, "
            "material tangent, and full element tangent diagnostics."
        )
    )
    parser.add_argument(
        "output_root",
        type=Path,
        nargs="?",
        default=Path("_no_minimization_ss_jobs/output_size3_step0p1_direct_fields"),
    )
    parser.add_argument("--shears", default="0,2,5,10")
    parser.add_argument("--loops", type=int, default=30)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Plots/no_minimization_current_condition_decomposition.pdf"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("Plots/no_minimization_current_condition_decomposition.csv"),
    )
    args = parser.parse_args()

    local_loads = [round(0.1 * i, 1) for i in range(1, 10)]
    records = collect_records(
        args.output_root,
        shears=parse_shears(args.shears),
        local_loads=local_loads,
        loops=args.loops,
    )
    write_csv(records, args.csv)
    plot_records(records, args.out)
    print(args.out)
    print(args.csv)


if __name__ == "__main__":
    main()
