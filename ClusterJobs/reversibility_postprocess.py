#!/usr/bin/env python3
"""Create compact event tables from the large reversibility VTU data.

The ``process`` command reads one simulation folder and streams its
affine/relaxed VTU pairs one event at a time. The ``manifest`` and ``merge``
commands support the Slurm array workflow in the neighboring batch scripts.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Management.updateCSV import read_macrodata_csv
from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import (
    calculate_energy_step_data,
    calculate_stress_step_data,
    volume_from_metadata,
)
from Plotting.vtuDataForSylvain import VTUData


EVENT_PATTERN = re.compile(r"(?P<kind>rev|irrev)_drop_l_(?P<load>[0-9.eE+-]+)$")

EVENT_COLUMNS = [
    "batch", "job_name", "seed", "load_increment", "LBFGSEpsx",
    "event_index", "event_kind", "is_reversible", "yield_regime",
    "event_start_load", "event_load", "mesh_volume", "bulk_modulus",
    "delta_gamma", "delta_E_S", "delta_E_I", "delta_E_R",
    "delta_sigma_S", "delta_sigma_I", "delta_sigma_R", "delta_rev_E",
    "delta_rev_sigma", "delta_rev_u", "delta_u_R", "sigma12_i", "a1212_i",
]


def _expected_job_names(batch: int) -> list[str]:
    if batch == -2:
        settings = [("1e-05", epsx) for epsx in ("0.0001", "1e-05", "1e-06", "1e-07")]
    elif batch == -1:
        settings = [(load, "1e-06") for load in ("0.0001", "5e-05", "1e-05", "5e-06", "1e-06")]
    else:
        raise ValueError("Only Sylvain batches -2 and -1 are supported.")
    return [
        f"reversibilityProtocolTest,s100x100l0.14,{load},"
        f"1.0PBCt3LBFGSEpsx{epsx}energyDropThreshold1e-05s{seed}"
        for load, epsx in settings
        for seed in range(4)
    ]


def _single_state_file(event_dir: Path, state_name: str) -> Path:
    matches = sorted(event_dir.glob(f"{state_name}.*.vtu"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {state_name} VTU in {event_dir}, found {matches}."
        )
    return matches[0]


def _ordered_xy_points(vtu_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = VTUData(vtu_path)
    points = np.asarray(data.points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(
            f"Expected 2D point coordinates in {vtu_path}, got {points.shape}."
        )
    ref_index, location, _ = data.field("refIndex")
    if location != "point":
        raise ValueError(f"Expected point-wise refIndex in {vtu_path}, got {location}.")
    ref_index = np.asarray(ref_index)
    if ref_index.ndim != 1 or ref_index.shape[0] != points.shape[0]:
        raise ValueError(f"Invalid refIndex shape {ref_index.shape} in {vtu_path}.")
    return points[:, :2], ref_index


def _delta_u_relaxation(event_dir: Path) -> float:
    affine_points, affine_refs = _ordered_xy_points(
        _single_state_file(event_dir, "state1_affine_gamma_plus")
    )
    relaxed_points, relaxed_refs = _ordered_xy_points(
        _single_state_file(event_dir, "state2_relaxed_gamma_plus")
    )
    if not np.array_equal(affine_refs, relaxed_refs):
        raise ValueError(
            f"The affine and relaxed meshes have different refIndex values in {event_dir}."
        )
    displacement = relaxed_points - affine_points
    displacement -= displacement.mean(axis=0)
    result = float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))))
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"Invalid Delta u_R={result} calculated for {event_dir}.")
    return result


def _unique_load_index(loads: np.ndarray, target: float, load_increment: float) -> int:
    candidates = np.flatnonzero(
        np.isclose(
            loads, target, rtol=1e-9,
            atol=max(1e-12, load_increment * 1e-6),
        )
    )
    if candidates.size != 1:
        raise RuntimeError(
            f"Could not uniquely map target load {target}; candidates={candidates.tolist()}."
        )
    return int(candidates[0])


def _finite_float(value, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite {name}: {value!r}")
    return result


def process_job(job_dir: Path, output_path: Path, batch: int) -> int:
    job_dir = Path(job_dir)
    output_path = Path(output_path)
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Simulation folder not found: {job_dir}")
    csv_path = job_dir / "macroData.csv"
    event_root = job_dir / "data" / "reversibilityData"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing macroData.csv: {csv_path}")
    if not event_root.is_dir():
        raise FileNotFoundError(f"Missing reversibility data directory: {event_root}")

    df = read_macrodata_csv(csv_path)
    required_columns = {
        "load", "avg_sigma12", "avg_sigma12_change_from_init",
        "total_energy_change", "total_e_change_from_init", "is_reversible",
        "rev_energy_diff", "rev_sigma_12_diff", "rev_u_diff",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise KeyError(f"Missing columns {missing} in {csv_path}.")

    metadata = get_metadata(str(csv_path))
    load_increment = _finite_float(metadata["loadIncrement"], "loadIncrement")
    if load_increment <= 0:
        raise ValueError(f"Invalid loadIncrement={load_increment} in {csv_path}.")
    mesh_volume = volume_from_metadata(metadata)
    if mesh_volume is None or not np.isfinite(mesh_volume) or mesh_volume <= 0:
        raise ValueError(f"Could not infer a positive mesh volume for {csv_path}.")
    mesh_volume = float(mesh_volume)

    loads = np.asarray(df["load"], dtype=float)
    if loads.ndim != 1 or loads.size < 2 or not np.all(np.isfinite(loads)):
        raise ValueError(f"Invalid load column in {csv_path}.")
    energy_steps, energy_info = calculate_energy_step_data(
        str(csv_path), df=df, metadata=metadata, average_energy=False
    )
    stress_steps, _ = calculate_stress_step_data(
        str(csv_path), df=df, calculate_tangent=True
    )
    if len(energy_steps) != len(df) - 1 or len(stress_steps) != len(df) - 1:
        raise RuntimeError("Step-data length does not match macroData.csv.")

    sigma = np.asarray(df["avg_sigma12"], dtype=float)
    yield_load = float(loads[int(np.argmax(sigma))])
    epsx = metadata.get("LBFGSEpsx", np.nan)
    try:
        epsx = float(epsx)
    except (TypeError, ValueError):
        epsx = np.nan

    event_dirs = sorted(path for path in event_root.iterdir() if path.is_dir())
    if not event_dirs:
        raise RuntimeError(f"No event directories found in {event_root}.")

    records = []
    for event_index, event_dir in enumerate(event_dirs):
        match = EVENT_PATTERN.fullmatch(event_dir.name)
        if match is None:
            raise ValueError(f"Unexpected reversibility event directory: {event_dir}")
        event_kind = match.group("kind")
        start_load = _finite_float(match.group("load"), "event start load")
        event_load = start_load + load_increment
        row_index = _unique_load_index(loads, event_load, load_increment)
        if row_index == 0:
            raise RuntimeError(f"Event {event_dir} maps to the first macro row.")
        step_index = row_index - 1
        row = df.iloc[row_index]

        expected_reversible = event_kind == "rev"
        actual_reversible = bool(int(row["is_reversible"]))
        if actual_reversible != expected_reversible:
            raise ValueError(
                f"Reversibility mismatch for {event_dir}: folder says "
                f"{expected_reversible}, macro row says {actual_reversible}."
            )

        records.append(
            {
                "batch": int(batch),
                "job_name": job_dir.name,
                "seed": int(metadata.get("seed", -1)),
                "load_increment": load_increment,
                "LBFGSEpsx": epsx,
                "event_index": event_index,
                "event_kind": event_kind,
                "is_reversible": expected_reversible,
                "yield_regime": "post-yield" if event_load > yield_load else "pre-yield",
                "event_start_load": start_load,
                "event_load": event_load,
                "mesh_volume": mesh_volume,
                "bulk_modulus": float(energy_info["bulk_modulus"]),
                "delta_gamma": float(energy_steps["delta_gamma"].iloc[step_index]),
                # E_S is second-order stress corrected and normalized by mesh volume.
                "delta_E_S": float(
                    energy_steps["stress_corrected_drop_second_order"].iloc[step_index]
                ) / mesh_volume,
                "delta_E_I": -_finite_float(row["total_energy_change"], "total_energy_change") / mesh_volume,
                "delta_E_R": -_finite_float(
                    row["total_e_change_from_init"], "total_e_change_from_init"
                ) / mesh_volume,
                # sigma_S is first-order elasticity corrected.
                "delta_sigma_S": float(stress_steps["stress_corrected_drop"].iloc[step_index]),
                "delta_sigma_I": float(stress_steps["inter_strain_drop"].iloc[step_index]),
                "delta_sigma_R": float(stress_steps["relaxation_drop"].iloc[step_index]),
                "delta_rev_E": _finite_float(row["rev_energy_diff"], "rev_energy_diff"),
                "delta_rev_sigma": _finite_float(row["rev_sigma_12_diff"], "rev_sigma_12_diff"),
                "delta_rev_u": _finite_float(row["rev_u_diff"], "rev_u_diff"),
                "delta_u_R": _delta_u_relaxation(event_dir),
                "sigma12_i": float(stress_steps["sigma12_i"].iloc[step_index]),
                "a1212_i": float(stress_steps["a1212_i"].iloc[step_index]),
            }
        )

    result = pd.DataFrame.from_records(records, columns=EVENT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with gzip.open(temporary_path, "wt", newline="") as stream:
        result.to_csv(stream, index=False)
    temporary_path.replace(output_path)
    print(f"Processed {len(result)} events from {job_dir} -> {output_path}")
    return len(result)


def make_manifest(
    data_root: Path,
    manifest_path: Path,
    batches: list[int],
    allow_missing: bool,
    allow_empty: bool,
) -> None:
    data_root = Path(data_root)
    lines = []
    missing = []
    for batch in batches:
        for job_name in _expected_job_names(batch):
            if (data_root / job_name).is_dir():
                lines.append(f"{batch}\t{job_name}")
            else:
                missing.append(job_name)
    if missing and not allow_missing:
        raise FileNotFoundError("Missing expected job folders:\n" + "\n".join(missing))
    if not lines and not allow_empty:
        raise RuntimeError(f"No requested job folders found under {data_root}.")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"Wrote {len(lines)} jobs to {manifest_path}; missing_expected={len(missing)}")
    if missing:
        print("Missing expected folders:", file=sys.stderr)
        for job_name in missing:
            print(job_name, file=sys.stderr)


def _read_manifest(manifest_path: Path) -> list[tuple[int, str]]:
    entries = []
    for line_number, line in enumerate(Path(manifest_path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"Invalid manifest line {line_number}: {line!r}")
        entries.append((int(parts[0]), parts[1]))
    return entries


def merge_tables(
    manifest_path: Path, output_root: Path, batch: int, output_path: Path
) -> int:
    entries = [
        (entry_batch, name)
        for entry_batch, name in _read_manifest(manifest_path)
        if entry_batch == batch
    ]
    if not entries:
        raise RuntimeError(f"No batch {batch} entries in {manifest_path}.")
    tables = []
    expected_columns = None
    for _, job_name in entries:
        table_path = Path(output_root) / f"batch_{batch}" / f"{job_name}.csv.gz"
        if not table_path.is_file():
            raise FileNotFoundError(f"Missing per-job table: {table_path}")
        table = pd.read_csv(table_path)
        if expected_columns is None:
            expected_columns = list(table.columns)
        elif list(table.columns) != expected_columns:
            raise ValueError(f"Column schema mismatch in {table_path}.")
        tables.append(table)

    merged = pd.concat(tables, ignore_index=True)
    merged = merged.sort_values(
        ["job_name", "event_load", "event_kind"], kind="stable"
    ).reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with gzip.open(temporary_path, "wt", newline="") as stream:
        merged.to_csv(stream, index=False)
    temporary_path.replace(output_path)
    print(f"Merged {len(merged)} events from {len(tables)} jobs -> {output_path}")
    return len(merged)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--data-root", type=Path, required=True)
    manifest.add_argument("--manifest", type=Path, required=True)
    manifest.add_argument("--batch", type=int, action="append", required=True)
    manifest.add_argument("--allow-missing", action="store_true")
    manifest.add_argument("--allow-empty", action="store_true")

    process = subparsers.add_parser("process")
    process.add_argument("--job-dir", type=Path, required=True)
    process.add_argument("--output", type=Path, required=True)
    process.add_argument("--batch", type=int, required=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--output-root", type=Path, required=True)
    merge.add_argument("--batch", type=int, required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "manifest":
        make_manifest(
            args.data_root,
            args.manifest,
            args.batch,
            args.allow_missing,
            args.allow_empty,
        )
        return 0
    if args.command == "process":
        process_job(args.job_dir, args.output, args.batch)
        return 0
    if args.command == "merge":
        merge_tables(args.manifest, args.output_root, args.batch, args.output)
        return 0
    raise AssertionError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
