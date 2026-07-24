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

DEFAULT_OUTPUT_ROOT = ROOT / "_no_minimization_ss_jobs/output_size3_step0p1_direct_fields"
DEFAULT_CURRENT_PDF = ROOT / "Plots/no_minimization_current_condition_decomposition.pdf"
DEFAULT_CURRENT_CSV = ROOT / "Plots/no_minimization_current_condition_decomposition.csv"
DEFAULT_REFERENCE_PDF = ROOT / "Plots/no_minimization_reference_condition_decomposition.pdf"
DEFAULT_REFERENCE_CSV = ROOT / "Plots/no_minimization_reference_condition_decomposition.csv"
DEFAULT_SUMMARY_PDF = ROOT / "Plots/no_minimization_conditioning_summary.pdf"
DEFAULT_SUMMARY_PNG = ROOT / "Plots/no_minimization_conditioning_summary.png"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from MTMath.energyFunction import ContiEnergy
from MTMath.reduction import lagrange_reduction
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


def reference_shear_from_name(sim_dir: Path) -> float | None:
    for part in sim_dir.name.split("GP1")[1:]:
        token = []
        for char in part:
            if char in "+-.0123456789eE":
                token.append(char)
            else:
                break
        if token:
            return float("".join(token))
    if "ReferenceTest" in sim_dir.name:
        return 0.0
    return None


def condition_number_2(matrices: np.ndarray) -> np.ndarray:
    singular_values = np.linalg.svd(matrices, compute_uv=False)
    if np.any(singular_values[:, -1] <= 0.0):
        raise ValueError("Cannot compute a condition number with a zero singular value.")
    return singular_values[:, 0] / singular_values[:, -1]


def reference_edge_matrices(data: VTUData) -> np.ndarray:
    dN_dX = data.get_dN_dX()
    if not np.allclose(dN_dX.sum(axis=1), 0.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"Shape-function gradients do not sum to zero in {data.vtu_file_path}.")

    grad_columns = np.stack([dN_dX[:, 1, :], dN_dX[:, 2, :]], axis=-1)
    DX = np.linalg.inv(grad_columns).swapaxes(-1, -2)
    det_DX = np.linalg.det(DX)
    if np.any(det_DX <= 0.0):
        raise ValueError(f"Reference element orientation must be positive in {data.vtu_file_path}.")

    expected_det = 2.0 * data.get_init_area()
    if not np.allclose(det_DX, expected_det, rtol=1e-8, atol=1e-10):
        max_error = float(np.max(np.abs(det_DX - expected_det)))
        raise ValueError(
            f"dN_dX and initArea disagree in {data.vtu_file_path}; max determinant error {max_error:g}."
        )
    return DX


def kinematic_conditions(data: VTUData) -> dict[str, np.ndarray]:
    F = data.get_F()
    DX = reference_edge_matrices(data)
    Dx = np.einsum("...ij,...jk->...ik", F, DX)
    C = data.get_C()
    C_from_F = np.einsum("...ji,...jk->...ik", F, F)
    if not np.allclose(C, C_from_F, rtol=1e-9, atol=1e-10):
        max_error = float(np.max(np.abs(C - C_from_F)))
        raise ValueError(f"C and F.T @ F disagree in {data.vtu_file_path}; max error {max_error:g}.")

    kappa_F = condition_number_2(F)
    kappa_C = condition_number_2(C)
    if not np.allclose(kappa_C, kappa_F**2, rtol=1e-8, atol=1e-8):
        max_error = float(np.max(np.abs(kappa_C - kappa_F**2)))
        raise ValueError(f"kappa_C != kappa_F^2 in {data.vtu_file_path}; max error {max_error:g}.")

    return {
        "kappa_X": condition_number_2(DX),
        "kappa_x": condition_number_2(Dx),
        "kappa_F": kappa_F,
        "kappa_C": kappa_C,
    }


def constitutive_tangent_matrix(F: np.ndarray, *, loops: int) -> np.ndarray:
    ContiEnergy._initialize_div_div_phi()
    if ContiEnergy._DIV_DIV_PHI is None:
        raise RuntimeError("ContiEnergy Hessian was not initialized.")

    C = np.einsum("...ji,...jk->...ik", F, F)
    C_R, M_R = lagrange_reduction(C, loops=loops)
    C_11, C_22, C_12 = C_R[..., 0, 0], C_R[..., 1, 1], C_R[..., 0, 1]
    H_raw = ContiEnergy._DIV_DIV_PHI(C_11, C_22, C_12, -1 / 4, 4, 1)
    H = np.asarray(H_raw, dtype=float)
    H = np.moveaxis(H, (0, 1), (-2, -1))
    if H.shape[-2:] != (3, 3):
        raise ValueError(f"Constitutive Hessian has unexpected shape {H.shape}.")

    component_basis = np.zeros((2, 2, 3), dtype=float)
    component_basis[0, 0, 0] = 1.0
    component_basis[1, 1, 1] = 1.0
    component_basis[0, 1, 2] = 0.5
    component_basis[1, 0, 2] = 0.5
    hessian_reduced = np.einsum("ija,...ab,klb->...ijkl", component_basis, H, component_basis)
    hessian = np.einsum(
        "...ir,...js,...kt,...lu,...rstu->...ijkl",
        M_R,
        M_R,
        M_R,
        M_R,
        hessian_reduced,
    )

    orthonormal_basis = np.zeros((3, 2, 2), dtype=float)
    orthonormal_basis[0, 0, 0] = 1.0
    orthonormal_basis[1, 1, 1] = 1.0
    orthonormal_basis[2, 0, 1] = 1.0 / np.sqrt(2.0)
    orthonormal_basis[2, 1, 0] = 1.0 / np.sqrt(2.0)
    c_e = 4.0 * np.einsum(
        "aij,...ijkl,bkl->...ab",
        orthonormal_basis,
        hessian,
        orthonormal_basis,
    )
    return 0.5 * (c_e + np.swapaxes(c_e, -1, -2))


def constitutive_condition(F: np.ndarray, *, loops: int) -> tuple[np.ndarray, np.ndarray]:
    c_e = constitutive_tangent_matrix(F, loops=loops)
    singular_values = np.linalg.svd(c_e, compute_uv=False)
    if np.any(singular_values[:, -1] <= 0.0):
        raise ValueError("Constitutive tangent has a zero singular value.")
    eigenvalues = np.linalg.eigvalsh(c_e)
    return singular_values[:, 0] / singular_values[:, -1], eigenvalues[:, 0]


def tangent_condition(vtu_file: Path, *, loops: int) -> np.ndarray:
    eigenvalues = element_tangent_eigenvalues(vtu_file, loops=loops)
    non_translation = np.sort(np.abs(eigenvalues), axis=1)[:, 2:]
    return non_translation[:, -1] / non_translation[:, 0]


def summarize(vtu_file: Path, *, loops: int) -> dict[str, float]:
    data = VTUData(str(vtu_file))
    conditions = kinematic_conditions(data)
    kappa_con, lambda_min_con = constitutive_condition(data.get_F(), loops=loops)
    kappa_tan = tangent_condition(vtu_file, loops=loops)

    for name, values in conditions.items():
        assert_uniform_element_values(values, name=name, vtu_file=vtu_file)
    assert_uniform_element_values(kappa_con, name="kappa_con", vtu_file=vtu_file)
    assert_uniform_element_values(lambda_min_con, name="lambda_min_con", vtu_file=vtu_file)
    assert_uniform_element_values(kappa_tan, name="kappa_tan", vtu_file=vtu_file)
    return {
        "kappa_X": float(conditions["kappa_X"][0]),
        "kappa_x": float(conditions["kappa_x"][0]),
        "kappa_F": float(conditions["kappa_F"][0]),
        "kappa_C": float(conditions["kappa_C"][0]),
        "kappa_con": float(kappa_con[0]),
        "lambda_min_con": float(lambda_min_con[0]),
        "kappa_tan": float(kappa_tan[0]),
    }


def collect_records(
    output_root: Path,
    *,
    shears: list[int],
    local_loads: list[float],
    loops: int,
    mode: str,
) -> list[dict]:
    records = []
    sim_dirs = find_simulation_dirs(output_root)
    if mode == "current":
        for sim_dir in sim_dirs:
            reconnection = reconnection_label(sim_dir)
            for shear in shears:
                absolute_loads = [
                    round(shear + local_load, 12) for local_load in local_loads
                ]
                files = vtu_files_at_loads(sim_dir, absolute_loads)
                for local_load, absolute_load, vtu_file in zip(
                    local_loads, absolute_loads, files
                ):
                    record = {
                        "integer_shear": shear,
                        "reference_shear": 0.0,
                        "reconnection": reconnection,
                        "local_load": local_load,
                        "absolute_load": absolute_load,
                    }
                    record.update(summarize(vtu_file, loops=loops))
                    records.append(record)
    elif mode == "reference":
        by_variant = {}
        for sim_dir in sim_dirs:
            reference_shear = reference_shear_from_name(sim_dir)
            if reference_shear is None:
                continue
            key = (round(reference_shear, 12), reconnection_label(sim_dir))
            if key in by_variant:
                raise ValueError(f"Duplicate simulation directories for {key}.")
            by_variant[key] = sim_dir

        for shear in shears:
            target_reference_shear = round(-float(shear), 12)
            for reconnection in ("no reconnection", "edge flip"):
                key = (target_reference_shear, reconnection)
                if key not in by_variant:
                    raise ValueError(
                        f"Missing reference shear {target_reference_shear:g}, {reconnection}."
                    )
                absolute_loads = [round(local_load, 12) for local_load in local_loads]
                files = vtu_files_at_loads(by_variant[key], absolute_loads)
                for local_load, absolute_load, vtu_file in zip(
                    local_loads, absolute_loads, files
                ):
                    record = {
                        "integer_shear": shear,
                        "reference_shear": target_reference_shear,
                        "reconnection": reconnection,
                        "local_load": local_load,
                        "absolute_load": absolute_load,
                    }
                    record.update(summarize(vtu_file, loops=loops))
                    records.append(record)
    else:
        raise ValueError(f"Unknown mode {mode!r}.")
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


def read_csv_records(csv_path: Path) -> list[dict]:
    required_columns = {
        "integer_shear",
        "reference_shear",
        "local_load",
        "absolute_load",
        "kappa_X",
        "kappa_x",
        "kappa_F",
        "kappa_C",
        "kappa_con",
        "lambda_min_con",
        "kappa_tan",
    }
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header.")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        records = []
        for row in reader:
            row["integer_shear"] = int(float(row["integer_shear"]))
            row["reference_shear"] = float(row["reference_shear"])
            row["local_load"] = float(row["local_load"])
            row["absolute_load"] = float(row["absolute_load"])
            row["kappa_X"] = float(row["kappa_X"])
            row["kappa_x"] = float(row["kappa_x"])
            row["kappa_F"] = float(row["kappa_F"])
            row["kappa_C"] = float(row["kappa_C"])
            row["kappa_con"] = float(row["kappa_con"])
            row["lambda_min_con"] = float(row["lambda_min_con"])
            row["kappa_tan"] = float(row["kappa_tan"])
            records.append(row)
    if not records:
        raise ValueError(f"{csv_path} contains no records.")
    return records


def condition_quantities(mode: str) -> list[tuple[str, str, str, tuple[float, float]]]:
    if mode == "current":
        geometry = ("kappa_x", r"$\kappa_{\mathbf{x}}$", "current geometry", (1e0, 2e2))
    elif mode == "reference":
        geometry = ("kappa_X", r"$\kappa_{\mathbf{X}}$", "reference geometry", (1e0, 2e2))
    else:
        raise ValueError(f"Unknown mode {mode!r}.")
    return [
        geometry,
        (
            "kappa_C",
            r"$\kappa_{\mathbf{C}}$",
            "right Cauchy-Green tensor",
            (1e0, 5e4),
        ),
        (
            "kappa_con",
            r"$\kappa_{\mathrm{con}}$",
            "constitutive tangent",
            (1e1, 1e10),
        ),
        ("kappa_tan", r"$\kappa_{\mathrm{tan}}$", "element tangent", (1e3, 1e8)),
    ]


def plot_records_on_axes(
    records: list[dict],
    axes,
    *,
    mode: str,
    column_title: str | None = None,
    row_titles: bool = True,
    add_legends: bool = True,
) -> None:
    if not records:
        raise ValueError("No records to plot.")

    quantities = condition_quantities(mode)
    shears = sorted({int(row["integer_shear"]) for row in records})
    reconnections = ["no reconnection", "edge flip"]
    colors = {0: "C0", 2: "C2", 5: "C3", 10: "C4"}
    linestyles = {"edge flip": "-", "no reconnection": "--"}
    zorders = {"edge flip": 4, "no reconnection": 3}

    axes = np.asarray(axes).ravel()
    if len(axes) != len(quantities):
        raise ValueError(f"Expected {len(quantities)} axes, got {len(axes)}.")

    for index, (ax, (field, ylabel, title, ylim)) in enumerate(zip(axes, quantities)):
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
        if row_titles:
            ax.set_title(title)
        elif index == 0 and column_title is not None:
            ax.set_title(column_title)
        ax.grid(True, which="both", alpha=0.25)

    axes[-1].set_xlabel(r"$\gamma-n$" if mode == "current" else r"$\gamma$")
    if not add_legends:
        return

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
        title=r"$n$" if mode == "current" else r"$-\gamma_{\mathrm{ref}}$",
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


def plot_records(records: list[dict], out_path: Path, *, mode: str) -> None:
    fig, axes = plt.subplots(
        len(condition_quantities(mode)),
        1,
        figsize=(5.2, 9.0),
        sharex=True,
        constrained_layout=True,
    )
    plot_records_on_axes(records, axes, mode=mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def records_for_mode(
    output_root: Path,
    *,
    csv_path: Path,
    shears: list[int],
    local_loads: list[float],
    loops: int,
    mode: str,
) -> list[dict]:
    if output_root.is_dir():
        records = collect_records(
            output_root,
            shears=shears,
            local_loads=local_loads,
            loops=loops,
            mode=mode,
        )
        write_csv(records, csv_path)
        return records
    if csv_path.is_file():
        return read_csv_records(csv_path)
    raise FileNotFoundError(
        f"Neither simulation output directory {output_root} nor cached records {csv_path} exist."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot T3 element condition numbers for geometry, kinematics, "
            "constitutive response, and the full element tangent."
        )
    )
    parser.add_argument(
        "output_root",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--shears", default="0,2,5,10")
    parser.add_argument("--loops", type=int, default=30)
    parser.add_argument(
        "--mode",
        choices=("current", "reference", "both"),
        default="both",
        help=(
            "current samples loads n+s from long simple-shear runs; reference "
            "samples loads s from runs whose reference shear is GP1=-n; both "
            "generates the current, reference, and combined summary figures."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PDF for a single current/reference mode (defaults by mode).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV for a single current/reference mode (defaults by mode).",
    )
    args = parser.parse_args()

    local_loads = [round(0.1 * i, 1) for i in range(1, 10)]
    shears = parse_shears(args.shears)

    if args.mode == "both":
        current_records = records_for_mode(
            args.output_root,
            csv_path=DEFAULT_CURRENT_CSV,
            shears=shears,
            local_loads=local_loads,
            loops=args.loops,
            mode="current",
        )
        reference_records = records_for_mode(
            args.output_root,
            csv_path=DEFAULT_REFERENCE_CSV,
            shears=shears,
            local_loads=local_loads,
            loops=args.loops,
            mode="reference",
        )
        plot_records(current_records, DEFAULT_CURRENT_PDF, mode="current")
        plot_records(reference_records, DEFAULT_REFERENCE_PDF, mode="reference")

        from Plotting.plot_conditioning_summary import make_figure

        make_figure(
            current_csv=DEFAULT_CURRENT_CSV,
            reference_csv=DEFAULT_REFERENCE_CSV,
            integer_shear=2,
            local_shear=0.5,
            out_pdf=DEFAULT_SUMMARY_PDF,
            out_png=DEFAULT_SUMMARY_PNG,
        )
        for path in (
            DEFAULT_CURRENT_PDF,
            DEFAULT_REFERENCE_PDF,
            DEFAULT_SUMMARY_PDF,
            DEFAULT_SUMMARY_PNG,
        ):
            print(path)
        return

    default_out = DEFAULT_CURRENT_PDF if args.mode == "current" else DEFAULT_REFERENCE_PDF
    default_csv = DEFAULT_CURRENT_CSV if args.mode == "current" else DEFAULT_REFERENCE_CSV
    records = records_for_mode(
        args.output_root,
        csv_path=args.csv or default_csv,
        shears=shears,
        local_loads=local_loads,
        loops=args.loops,
        mode=args.mode,
    )
    out_path = args.out or default_out
    csv_path = args.csv or default_csv
    plot_records(records, out_path, mode=args.mode)
    print(out_path)
    print(csv_path)


if __name__ == "__main__":
    main()
