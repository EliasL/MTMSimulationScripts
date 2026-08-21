#!/usr/bin/env python3
"""Track a fixed MTS2D element ID through reconnection.

MTS2D does not export a separate ``elementIndex`` cell field.  The cell order
written by ``writeMeshToVtu`` is the element index, and ``flipEdge`` rebuilds
the two changed elements in their original indices.  Therefore a fixed VTU
triangle slot is the element-ID history requested here, even though its
``refIndex`` node triplet can change at a reconnection.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MTMath.poincareEnergy import C2PoincareDisk
from MTMath.reduction import plastic_reduction


DEFAULT_JOB = Path(
    "/Volumes/data/MTS2D_output/"
    "reversibilityProtocolTest,s200x200l1.0,1e-05,5.1PBCedgeFlipt2epsR0.0"
    "LBFGSEpsg0.0LBFGSEpsx1e-06s0"
)
DEFAULT_OUTPUT_DIRECTORY = ROOT / "Plots/reconnecting_largest_energy_events_preview"
DEFAULT_ELEMENT_ID = 72927
LOAD_PATTERN = re.compile(r"_load=(?P<load>[0-9.eE+-]+)_")


def _load_from_name(path: Path) -> float:
    match = LOAD_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse load from {path.name}.")
    return float(match.group("load"))


def _cell_scalar(mesh, name: str, element_id: int) -> float:
    try:
        values = np.asarray(mesh.cell_data_dict[name]["triangle"], dtype=float).reshape(-1)
    except KeyError as exc:
        raise KeyError(f"Missing triangle cell field {name!r}.") from exc
    if not 0 <= element_id < values.size:
        raise IndexError(f"Element ID {element_id} is outside field {name!r} with {values.size} entries.")
    value = float(values[element_id])
    if not np.isfinite(value):
        raise ValueError(f"Element {element_id} has non-finite {name}.")
    return value


def _cell_matrix_at(mesh, prefix: str, element_id: int) -> np.ndarray:
    components = []
    for i, j in ((1, 1), (1, 2), (2, 1), (2, 2)):
        try:
            values = np.asarray(
                mesh.cell_data_dict[f"{prefix}{i}{j}"]["triangle"],
                dtype=float,
            ).reshape(-1)
        except KeyError as exc:
            raise KeyError(f"Missing triangle cell field {prefix}{i}{j!r}.") from exc
        if not 0 <= element_id < values.size:
            raise IndexError(
                f"Element ID {element_id} is outside field {prefix!r} with {values.size} entries."
            )
        components.append(float(values[element_id]))
    matrix = np.array(
        [[components[0], components[1]], [components[2], components[3]]],
        dtype=float,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Element {element_id} has non-finite {prefix}.")
    return matrix


def _poincare_coordinates(total_T: np.ndarray) -> tuple[float, float, float, float]:
    metric = total_T.T @ total_T
    determinant = float(np.linalg.det(metric))
    if not np.isfinite(determinant) or determinant <= 0:
        raise ValueError("The tracked total-T metric is not positive definite.")
    raw_x, raw_y = C2PoincareDisk(metric)
    reduced_metric, _ = plastic_reduction(metric, compute_M=True)
    reduced_x, reduced_y = C2PoincareDisk(reduced_metric)
    values = tuple(float(value) for value in (raw_x, raw_y, reduced_x, reduced_y))
    if not np.all(np.isfinite(values)):
        raise ValueError("The tracked element could not be mapped to finite disk coordinates.")
    return values


def _read_macro_energy_changes(job: Path) -> dict[float, float]:
    path = job / "macroData.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames is None or not {"load", "total_energy_change"}.issubset(rows.fieldnames):
            raise ValueError(f"macroData.csv lacks load/total_energy_change columns: {path}")
        changes = {}
        for row in rows:
            load = float(row["load"])
            change = row["total_energy_change"]
            if change not in (None, ""):
                changes[load] = float(change)
        return changes


def _read_fixed_id_row(path: Path, element_id: int) -> dict[str, object]:
    load = _load_from_name(path)
    mesh = meshio.read(path)
    triangles = np.asarray(mesh.cells_dict.get("triangle"), dtype=int)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Expected triangular cells in {path}, got {triangles.shape}.")
    if not 0 <= element_id < len(triangles):
        raise IndexError(f"Element ID {element_id} is outside {path} with {len(triangles)} cells.")
    if "refIndex" not in mesh.point_data:
        raise KeyError(f"Missing refIndex point field in {path}.")
    reference_indices = np.asarray(mesh.point_data["refIndex"], dtype=int).reshape(-1)
    triplet = tuple(sorted(int(value) for value in reference_indices[triangles[element_id]]))
    if len(set(triplet)) != 3:
        raise ValueError(f"Element ID {element_id} has a degenerate refIndex triplet in {path}.")

    T_p = _cell_matrix_at(mesh, "T", element_id)
    F_e = _cell_matrix_at(mesh, "F_E", element_id)
    total_T = F_e @ T_p
    raw_x, raw_y, reduced_x, reduced_y = _poincare_coordinates(total_T)
    return {
        "file": path.name,
        "load": load,
        "element_id": element_id,
        "refIndex_triplet": ",".join(map(str, triplet)),
        "raw_x": raw_x,
        "raw_y": raw_y,
        "reduced_x": reduced_x,
        "reduced_y": reduced_y,
        "energy": _cell_scalar(mesh, "energy_field", element_id),
        "nrm3": _cell_scalar(mesh, "nrm3", element_id),
        "delta_nrm3": _cell_scalar(mesh, "deltaNrm3", element_id),
        "T_p_norm": float(np.linalg.norm(T_p)),
        "F_e_norm": float(np.linalg.norm(F_e)),
        "total_T": total_T,
    }


def _annotate_jumps(rows: list[dict[str, object]]) -> None:
    previous = None
    for row in rows:
        for key in (
            "delta_raw_x",
            "delta_raw_y",
            "delta_reduced_x",
            "delta_reduced_y",
            "delta_T_frobenius",
            "load_gap",
        ):
            row[key] = ""
        if previous is not None:
            for coordinate in ("raw_x", "raw_y", "reduced_x", "reduced_y"):
                row[f"delta_{coordinate}"] = float(row[coordinate]) - float(previous[coordinate])
            increment = np.asarray(row["total_T"]) @ np.linalg.inv(previous["total_T"]) - np.eye(2)
            row["delta_T_frobenius"] = float(np.linalg.norm(increment))
            row["load_gap"] = float(row["load"]) - float(previous["load"])
        previous = row


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "file_index", "file", "load", "element_id", "refIndex_triplet",
        "raw_x", "raw_y", "reduced_x", "reduced_y", "energy", "nrm3",
        "delta_nrm3", "T_p_norm", "F_e_norm", "delta_raw_x", "delta_raw_y",
        "delta_reduced_x", "delta_reduced_y", "delta_T_frobenius", "load_gap",
        "macro_total_energy_change",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _render_plot(rows: list[dict[str, object]], path: Path, element_id: int) -> None:
    loads = np.asarray([float(row["load"]) for row in rows])
    raw_x = np.asarray([float(row["raw_x"]) for row in rows])
    raw_y = np.asarray([float(row["raw_y"]) for row in rows])
    reduced_x = np.asarray([float(row["reduced_x"]) for row in rows])
    reduced_y = np.asarray([float(row["reduced_y"]) for row in rows])
    dx = np.asarray([float(row.get("delta_reduced_x", "nan") or "nan") for row in rows])
    dy = np.asarray([float(row.get("delta_reduced_y", "nan") or "nan") for row in rows])
    valid = np.isfinite(dx)
    top = np.argsort(np.abs(dx[valid]), kind="stable")[-10:]
    top_rows = np.flatnonzero(valid)[top]

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    axes[0, 0].plot(raw_x, raw_y, ".-", ms=2, lw=0.6, label="raw total T")
    axes[0, 0].plot(reduced_x, reduced_y, ".-", ms=2, lw=0.6, label="reduced")
    axes[0, 0].scatter(
        reduced_x[top_rows], reduced_y[top_rows], c="red", s=20,
        label=r"largest $|\Delta x_p|$",
    )
    axes[0, 0].set_xlabel("x_p")
    axes[0, 0].set_ylabel("y_p")
    axes[0, 0].set_title("Fixed element-ID path")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(loads, reduced_x, label="x_p")
    axes[0, 1].plot(loads, reduced_y, label="y_p")
    axes[0, 1].scatter(loads[top_rows], reduced_x[top_rows], c="red", s=16)
    axes[0, 1].set_xlabel("load")
    axes[0, 1].set_ylabel("reduced Poincare coordinate")
    axes[0, 1].set_title("Reduced coordinates versus load")
    axes[0, 1].legend()

    axes[1, 0].plot(loads, dx, label=r"$\Delta x_p$")
    axes[1, 0].plot(loads, dy, label=r"$\Delta y_p$")
    axes[1, 0].axhline(0.0, color="black", lw=0.5)
    axes[1, 0].scatter(loads[top_rows], dx[top_rows], c="red", s=16)
    axes[1, 0].set_xlabel("load")
    axes[1, 0].set_ylabel("coordinate change")
    axes[1, 0].set_title("Stepwise Poincare jumps")
    axes[1, 0].legend()

    energy = np.asarray([float(row["energy"]) for row in rows])
    delta_T = np.asarray([float(row.get("delta_T_frobenius", "nan") or "nan") for row in rows])
    axes[1, 1].plot(loads, energy, label="element energy")
    axes[1, 1].set_xlabel("load")
    axes[1, 1].set_ylabel("energy")
    axes[1, 1].set_title("Energy and total-T jump")
    twin = axes[1, 1].twinx()
    twin.plot(loads, delta_T, color="tab:red", alpha=0.8, label=r"$\|\Delta T\|_F$")
    twin.set_ylabel(r"$\|\Delta T\|_F$")
    figure.suptitle(
        f"Fixed MTS2D element ID {element_id}; red points are the ten largest "
        r"$|\Delta x_p|$ steps"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=234, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def track_history(job: Path, *, element_id: int, output_directory: Path) -> tuple[Path, Path]:
    data_directory = job / "data"
    paths = sorted(data_directory.glob("*.vtu"), key=_load_from_name)
    if not paths:
        raise FileNotFoundError(f"No VTUs found in {data_directory}.")
    rows = []
    for index, path in enumerate(paths):
        row = _read_fixed_id_row(path, element_id)
        row["file_index"] = index
        rows.append(row)
        if (index + 1) % 25 == 0 or index + 1 == len(paths):
            print(f"processed {index + 1}/{len(paths)}: {path.name}", flush=True)
    _annotate_jumps(rows)
    macro_changes = _read_macro_energy_changes(job)
    for row in rows:
        row["macro_total_energy_change"] = macro_changes.get(float(row["load"]), "")
    base = output_directory / f"element_id{element_id}_history"
    csv_path = base.with_suffix(".csv")
    plot_path = base.with_suffix(".png")
    _write_csv(rows, csv_path)
    _render_plot(rows, plot_path, element_id)
    return csv_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--element-id", type=int, default=DEFAULT_ELEMENT_ID)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    if args.element_id < 0:
        raise ValueError("element-id must be nonnegative.")
    csv_path, plot_path = track_history(
        args.job, element_id=args.element_id, output_directory=args.output_directory
    )
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
