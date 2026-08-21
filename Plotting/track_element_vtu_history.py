#!/usr/bin/env python3
"""Track one physical triangle through the saved equilibrium VTUs.

Element-array slots are not assumed to be persistent through reconnection.
The target is identified by the sorted ``refIndex`` triplet belonging to a
chosen event-state slot, and every saved VTU records the matching serialized
slot (or an explicit missing/ambiguous status).
"""

from __future__ import annotations

import argparse
import csv
import gc
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
from Plotting.meshEventPlotting import MeshState, periodic_triangle_centres


DEFAULT_JOB = Path(
    "/Volumes/data/MTS2D_output/"
    "reversibilityProtocolTest,s200x200l1.0,1e-05,5.1PBCedgeFlipt2epsR0.0"
    "LBFGSEpsg0.0LBFGSEpsx1e-06s0"
)
DEFAULT_EVENT_DIRECTORY = DEFAULT_JOB / "data/reversibilityData/irrev_drop_l_1.31901"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "Plots/reconnecting_largest_energy_events_preview"
LOAD_PATTERN = re.compile(r"_load=(?P<load>[0-9.eE+-]+)_")
BOX_SIZE = 200.0
LOAD_INCREMENT = 1e-5
TARGET_SLOT = 72927


def _one_vtu(directory: Path, prefix: str) -> Path:
    paths = sorted(directory.glob(f"{prefix}.*.vtu"))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one {prefix} VTU in {directory}, found {len(paths)}."
        )
    return paths[0]


def _load_from_name(path: Path) -> float:
    match = LOAD_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse load from {path.name}.")
    return float(match.group("load"))


def _cell_field(mesh, name: str) -> np.ndarray:
    try:
        values = mesh.cell_data_dict[name]["triangle"]
    except KeyError as exc:
        raise KeyError(f"Missing triangle cell field {name!r}.") from exc
    values = np.asarray(values, dtype=float).reshape(-1)
    return values


def _cell_matrix(mesh, name: str) -> np.ndarray:
    components = [_cell_field(mesh, f"{name}{i}{j}") for i, j in ((1, 1), (1, 2), (2, 1), (2, 2))]
    if not all(component.shape == components[0].shape for component in components):
        raise ValueError(f"Matrix components for {name} have inconsistent shapes.")
    return np.stack(
        [
            np.stack([components[0], components[1]], axis=-1),
            np.stack([components[2], components[3]], axis=-1),
        ],
        axis=-2,
    )


def _target_ref_triplet(event_directory: Path, target_slot: int) -> tuple[int, int, int]:
    event_path = _one_vtu(event_directory, "state0_min_gamma")
    mesh = meshio.read(event_path)
    triangles = np.asarray(mesh.cells_dict["triangle"], dtype=int)
    reference_indices = np.asarray(mesh.point_data["refIndex"], dtype=int).reshape(-1)
    if not 0 <= target_slot < len(triangles):
        raise IndexError(f"Target slot {target_slot} is outside {event_path}.")
    triplet = tuple(sorted(int(value) for value in reference_indices[triangles[target_slot]]))
    if len(set(triplet)) != 3:
        raise ValueError(f"Target slot {target_slot} does not have three distinct refIndex values.")
    return triplet


def _poincare_coordinates(total_T: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    metric = total_T.T @ total_T
    determinant = float(np.linalg.det(metric))
    if not np.isfinite(determinant) or determinant <= 0:
        raise ValueError("The tracked total-T metric is not positive definite.")
    raw_x, raw_y = C2PoincareDisk(metric)
    reduced_metric, reduction_M = plastic_reduction(metric, compute_M=True)
    reduced_x, reduced_y = C2PoincareDisk(reduced_metric)
    return float(raw_x), float(raw_y), float(reduced_x), np.asarray(
        [float(reduced_y), *reduction_M.reshape(-1)], dtype=float
    )


def _read_row(
    path: Path,
    *,
    target_ref_triplet: tuple[int, int, int],
    target_slot: int,
    previous_total_T: np.ndarray | None,
    previous_load: float | None,
) -> tuple[dict[str, object], np.ndarray | None]:
    load = _load_from_name(path)
    mesh = meshio.read(path)
    triangles = np.asarray(mesh.cells_dict["triangle"], dtype=int)
    points = np.asarray(mesh.points[:, :2], dtype=float)
    reference_indices = np.asarray(mesh.point_data["refIndex"], dtype=int).reshape(-1)
    if triangles.shape != (80000, 3) or len(reference_indices) != len(points):
        raise ValueError(f"Unexpected mesh shape in {path}: {triangles.shape}.")
    triangle_refs = np.sort(reference_indices[triangles], axis=1)
    matches = np.flatnonzero(np.all(triangle_refs == target_ref_triplet, axis=1))
    if len(matches) > 1:
        status = "ambiguous"
        physical_slot = None
    elif len(matches) == 0:
        status = "missing"
        physical_slot = None
    else:
        status = "matched"
        physical_slot = int(matches[0])

    slot_triplet = tuple(int(value) for value in triangle_refs[target_slot])
    row: dict[str, object] = {
        "file": path.name,
        "load": load,
        "status": status,
        "physical_slot": "" if physical_slot is None else physical_slot,
        "slot_72927_matches_target": slot_triplet == target_ref_triplet,
        "slot_72927_refIndex_triplet": ",".join(map(str, slot_triplet)),
        "target_refIndex_triplet": ",".join(map(str, target_ref_triplet)),
    }
    if physical_slot is None:
        return row, None

    T_p = _cell_matrix(mesh, "T")
    F_e = _cell_matrix(mesh, "F_E")
    total_T = F_e[physical_slot] @ T_p[physical_slot]
    raw_x, raw_y, reduced_x, reduction_data = _poincare_coordinates(total_T)
    reduced_y = reduction_data[0]
    M = reduction_data[1:].reshape(2, 2)
    energy = _cell_field(mesh, "energy_field")
    nrm3 = _cell_field(mesh, "nrm3")
    delta_nrm3 = _cell_field(mesh, "deltaNrm3")
    state = MeshState(
        path=path,
        points=points,
        triangles=triangles,
        reference_indices=reference_indices,
        point_fields={},
        cell_fields={"energy_field": energy, "nrm3": nrm3, "deltaNrm3": delta_nrm3},
    )
    centroid = periodic_triangle_centres(state, load=load, box_size=BOX_SIZE)[physical_slot]
    row.update(
        {
            "raw_x": raw_x,
            "raw_y": raw_y,
            "reduced_x": reduced_x,
            "reduced_y": reduced_y,
            "energy": float(energy[physical_slot]),
            "nrm3": float(nrm3[physical_slot]),
            "delta_nrm3": float(delta_nrm3[physical_slot]),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "T_p_norm": float(np.linalg.norm(T_p[physical_slot])),
            "F_e_norm": float(np.linalg.norm(F_e[physical_slot])),
            "M11": float(M[0, 0]),
            "M12": float(M[0, 1]),
            "M21": float(M[1, 0]),
            "M22": float(M[1, 1]),
        }
    )
    if previous_total_T is not None and previous_load is not None:
        increment = total_T @ np.linalg.inv(previous_total_T) - np.eye(2)
        row["delta_T_frobenius_from_previous_match"] = float(np.linalg.norm(increment))
        row["load_gap_from_previous_match"] = load - previous_load
    return row, total_T


def _render_history_plot(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
    target_ref_triplet: tuple[int, int, int],
    target_slot: int,
) -> Path:
    def value(row: dict[str, object], key: str) -> float:
        raw = row.get(key, "")
        return float("nan") if raw in ("", None) else float(raw)

    loads = np.asarray([value(row, "load") for row in rows], dtype=float)
    matched = np.asarray([row["status"] == "matched" for row in rows])
    matched_rows = [row for row in rows if row["status"] == "matched"]
    segments: list[list[dict[str, object]]] = []
    for row in matched_rows:
        if not segments or int(row["file_index"]) != int(segments[-1][-1]["file_index"]) + 1:
            segments.append([])
        segments[-1].append(row)
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    for segment_index, segment in enumerate(segments):
        axes[0, 0].plot(
            [value(row, "raw_x") for row in segment],
            [value(row, "raw_y") for row in segment],
            ".-", ms=2, lw=0.7, color="tab:blue",
            label="raw total T" if segment_index == 0 else None,
        )
        axes[0, 0].plot(
            [value(row, "reduced_x") for row in segment],
            [value(row, "reduced_y") for row in segment],
            ".-", ms=2, lw=0.7, color="tab:orange",
            label="plastically reduced" if segment_index == 0 else None,
        )
    axes[0, 0].set_xlabel("x_p")
    axes[0, 0].set_ylabel("y_p")
    axes[0, 0].set_title("Tracked Poincare path")
    axes[0, 0].legend()
    axes[0, 0].set_aspect("equal", adjustable="box")
    for segment_index, segment in enumerate(segments):
        axes[0, 1].plot(
            [value(row, "load") for row in segment],
            [value(row, "reduced_x") for row in segment],
            color="tab:blue",
            label="x_p" if segment_index == 0 else None,
        )
        axes[0, 1].plot(
            [value(row, "load") for row in segment],
            [value(row, "reduced_y") for row in segment],
            color="tab:orange",
            label="y_p" if segment_index == 0 else None,
        )
    axes[0, 1].set_xlabel("load")
    axes[0, 1].set_title("Reduced coordinates versus load")
    axes[0, 1].legend()
    energy = np.asarray(
        [value(row, "energy") for row in rows], dtype=float
    )
    axes[1, 0].plot(loads, energy, ".-", ms=2, lw=0.7, label="E")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("load")
    axes[1, 0].set_ylabel("element energy")
    axes[1, 0].set_title("Tracked element energy")
    slots = np.asarray(
        [
            value(row, "physical_slot")
            for row in rows
        ],
        dtype=float,
    )
    slot_matches = np.asarray(
        [str(row["slot_72927_matches_target"]).lower() == "true" for row in rows]
    )
    axes[1, 1].plot(loads, slots, ".-", ms=2, lw=0.7, label="physical triangle slot")
    axes[1, 1].plot(
        loads[slot_matches],
        np.full(np.count_nonzero(slot_matches), target_slot),
        "r.", ms=2, label="slot 72927 matches",
    )
    axes[1, 1].set_xlabel("load")
    axes[1, 1].set_ylabel("serialized slot")
    axes[1, 1].set_title("Topology-aware serialization tracking")
    axes[1, 1].legend()
    figure.suptitle(f"Physical refIndex triplet {target_ref_triplet}; source slot {target_slot}")
    figure.savefig(output_path, dpi=234, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    return output_path


def track_history(
    data_directory: Path,
    event_directory: Path,
    *,
    target_slot: int,
    output_directory: Path,
    start_index: int = 0,
    max_files: int | None = None,
    make_plot: bool = True,
) -> tuple[Path, Path | None]:
    all_files = sorted(data_directory.glob("*.vtu"), key=_load_from_name)
    if not all_files:
        raise FileNotFoundError(f"No VTUs found in {data_directory}.")
    if start_index < 0 or start_index >= len(all_files):
        raise ValueError(f"start_index must lie in [0, {len(all_files) - 1}].")
    files = all_files[start_index:]
    if max_files is not None:
        if max_files <= 0:
            raise ValueError("max_files must be positive when supplied.")
        files = files[:max_files]
    target_ref_triplet = _target_ref_triplet(event_directory, target_slot)
    rows = []
    previous_total_T = None
    previous_load = None
    for file_index, path in enumerate(files):
        row, total_T = _read_row(
            path,
            target_ref_triplet=target_ref_triplet,
            target_slot=target_slot,
            previous_total_T=previous_total_T,
            previous_load=previous_load,
        )
        row["file_index"] = start_index + file_index
        rows.append(row)
        if file_index == 0 or (file_index + 1) % 25 == 0 or file_index + 1 == len(files):
            print(
                f"processed {start_index + file_index + 1}/{len(all_files)}: {path.name}",
                flush=True,
            )
        if total_T is not None:
            previous_total_T = total_T
            previous_load = float(row["load"])
        if (file_index + 1) % 25 == 0:
            gc.collect()

    output_directory.mkdir(parents=True, exist_ok=True)
    base_name = f"element_ref{target_ref_triplet[0]}_{target_ref_triplet[1]}_{target_ref_triplet[2]}_history"
    partial = start_index != 0 or start_index + len(files) != len(all_files)
    suffix = f"_part{start_index:04d}_{start_index + len(files) - 1:04d}" if partial else ""
    csv_path = output_directory / f"{base_name}{suffix}.csv"
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key != "file_index", key))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if not make_plot:
        return csv_path, None

    plot_path = _render_history_plot(
        rows,
        output_path=output_directory / f"{base_name}{suffix}.png",
        target_ref_triplet=target_ref_triplet,
        target_slot=target_slot,
    )
    return csv_path, plot_path


def merge_history_parts(
    output_directory: Path,
    *,
    target_ref_triplet: tuple[int, int, int],
    target_slot: int,
) -> tuple[Path, Path]:
    prefix = f"element_ref{target_ref_triplet[0]}_{target_ref_triplet[1]}_{target_ref_triplet[2]}_history_part"
    part_paths = sorted(output_directory.glob(f"{prefix}*.csv"))
    if not part_paths:
        raise FileNotFoundError(f"No history parts found in {output_directory}.")
    rows: list[dict[str, object]] = []
    for part_path in part_paths:
        with part_path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row["file_index"]))
    expected = list(range(int(rows[0]["file_index"]), int(rows[-1]["file_index"]) + 1))
    actual = [int(row["file_index"]) for row in rows]
    if actual != expected:
        raise ValueError("History parts do not form one contiguous file-index sequence.")
    for previous, current in zip(rows, rows[1:]):
        if int(current["file_index"]) != int(previous["file_index"]) + 1:
            current["delta_T_frobenius_from_previous_match"] = ""
            current["load_gap_from_previous_match"] = ""
    base_name = f"element_ref{target_ref_triplet[0]}_{target_ref_triplet[1]}_{target_ref_triplet[2]}_history"
    csv_path = output_directory / f"{base_name}.csv"
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key != "file_index", key))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot_path = _render_history_plot(
        rows,
        output_path=output_directory / f"{base_name}.png",
        target_ref_triplet=target_ref_triplet,
        target_slot=target_slot,
    )
    return csv_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--event-directory", type=Path, default=DEFAULT_EVENT_DIRECTORY)
    parser.add_argument("--target-slot", type=int, default=TARGET_SLOT)
    parser.add_argument("--start-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--max-files", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-plot", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--merge-parts", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    if args.merge_parts:
        outputs = merge_history_parts(
            args.output_directory,
            target_ref_triplet=_target_ref_triplet(args.event_directory, args.target_slot),
            target_slot=args.target_slot,
        )
    else:
        outputs = track_history(
            args.job / "data",
            args.event_directory,
            target_slot=args.target_slot,
            output_directory=args.output_directory,
            start_index=args.start_index,
            max_files=args.max_files,
            make_plot=not args.no_plot,
        )
    for output in outputs:
        if output is not None:
            print(output)


if __name__ == "__main__":
    main()
