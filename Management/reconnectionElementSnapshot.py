#!/usr/bin/env python3
"""Sample individual flips after checkpoints, without changing source runs.

This lightweight worker can run on a cluster with NumPy and the existing
vtuBeforeReconnectionExtraction module. Each checkpoint contributes the first
fully written, topology-changing reconnection cycle after the per-run stress
peak. It samples cycles, not entire avalanches or the complete loading history.
Only compact changed-element arrays are retained; full VTUs are temporary.
The command-line worker requires a single-thread Slurm allocation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

try:
    from Management.vtuBeforeReconnectionExtraction import (
        _set_config_value, reconnection_pairs, stop_owned_process,
        write_extraction_config,
    )
except ModuleNotFoundError:
    from vtuBeforeReconnectionExtraction import (
        _set_config_value, reconnection_pairs, stop_owned_process,
        write_extraction_config,
    )


def read_dump_load(path: Path) -> float:
    """Read the saved strain, without loading the large mesh XML into memory."""
    tail = b""
    with gzip.open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            block = tail + chunk
            match = re.search(rb"<load>\s*([^<]+)</load>", block)
            if match:
                value = float(match.group(1))
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite checkpoint strain: {path}")
                return value
            tail = block[-256:]
    raise ValueError(f"Missing checkpoint strain: {path}")


def read_ascii_flip_state(path: Path) -> dict[str, np.ndarray]:
    """Read the historical ASCII VTUs, preserving reference-node connectivity."""
    root = ET.parse(path).getroot()
    piece = root.find("./UnstructuredGrid/Piece")
    if piece is None:
        raise ValueError(f"Missing VTU piece: {path}")

    def array(section: str, name: str, integer: bool = False) -> np.ndarray:
        node = piece.find(f"./{section}/DataArray[@Name='{name}']")
        if node is None or node.get("format") != "ascii":
            raise ValueError(f"Missing ASCII {section}/{name}: {path}")
        values = np.fromstring(node.text or "", sep=" ", dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite {name}: {path}")
        if integer:
            if not np.all(values == np.rint(values)):
                raise ValueError(f"Noninteger {name}: {path}")
            return values.astype(np.int64)
        return values

    ref = array("PointData", "refIndex", integer=True)
    types = array("Cells", "types", integer=True)
    offsets = array("Cells", "offsets", integer=True)
    if not np.all(types == 5) or not np.array_equal(offsets, 3 * np.arange(1, len(types) + 1)):
        raise ValueError(f"Expected only triangles: {path}")
    conn = array("Cells", "connectivity", integer=True).reshape(-1, 3)
    if len(conn) != int(piece.get("NumberOfCells")):
        raise ValueError(f"Unexpected triangle count: {path}")
    result = {"connectivity": ref[conn], "energy": array("CellData", "energy_field"),
              "stored_sigma12": array("CellData", "sigma12")}
    for prefix in ("T", "F_E"):
        result[prefix] = np.stack(
            [array("CellData", f"{prefix}{i}{j}") for i, j in ((1, 1), (1, 2), (2, 1), (2, 2))],
            axis=-1,
        ).reshape(-1, 2, 2)
    if any(len(x) != len(conn) for x in result.values()):
        raise ValueError(f"Inconsistent VTU field lengths: {path}")
    return result


def compact_flip_pair(before: Path, after: Path) -> dict[str, np.ndarray]:
    b, a = read_ascii_flip_state(before), read_ascii_flip_state(after)
    if b["connectivity"].shape != a["connectivity"].shape:
        raise ValueError("The triangle count changed across a reconnection.")
    changed = np.flatnonzero(np.any(np.sort(b["connectivity"], axis=1) != np.sort(a["connectivity"], axis=1), axis=1))
    arrays = {"element_indices": changed}
    for state, data in (("before", b), ("after", a)):
        arrays.update({f"{state}_{key}": value[changed] for key, value in data.items()})
    return arrays


def _closed_vtu(path: Path) -> bool:
    with path.open("rb") as stream:
        stream.seek(max(0, os.fstat(stream.fileno()).st_size - 128))
        return stream.read().rstrip().endswith(b"</VTKFile>")


def sample_checkpoint(job: dict, dump_name: str, executable: Path, root: Path,
                      timeout: float, nr_threads: int | None = None) -> dict:
    source = Path(job["folder"])
    if root == source or source in root.parents:
        raise ValueError("The analysis output must be outside the source simulation.")
    output = root / job["simulation_name"]
    output.mkdir(parents=True, exist_ok=True)
    target = output / (dump_name + ".npz")
    metadata_path = target.with_suffix(".json")
    if target.is_file() and metadata_path.is_file():
        return json.loads(metadata_path.read_text())
    checkpoint_load = read_dump_load(source / "dumps" / dump_name)
    if checkpoint_load <= job["yield_load"] or checkpoint_load >= job["max_load"] - 1e-12:
        result = {"simulation_name": job["simulation_name"], "dump_name": dump_name,
                  "checkpoint_load": checkpoint_load, "yield_load": job["yield_load"],
                  "status": ("checkpoint_not_post_yield" if checkpoint_load <= job["yield_load"]
                             else "checkpoint_at_endpoint"),
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        metadata_path.write_text(json.dumps(result, indent=2) + "\n")
        return result
    workspace = Path(tempfile.mkdtemp(prefix="replay-", dir=output))
    config = workspace / "config.conf"
    name = write_extraction_config(source / "config.conf", config)
    text = _set_config_value(config.read_text(), "maxLoad", job["max_load"])
    text = _set_config_value(text, "writeMeshVTUs", 0)
    text = _set_config_value(text, "showProgress", 0)
    if nr_threads is not None:
        text = _set_config_value(text, "nrThreads", nr_threads)
    config.write_text(text)
    private_output = workspace / "output"
    private_output.mkdir()
    steps = private_output / name / "data/minimizationData"
    started = time.monotonic()
    checked = set()
    process = None
    success = False
    result = {"simulation_name": job["simulation_name"], "dump_name": dump_name,
              "checkpoint_load": checkpoint_load,
              "yield_load": job["yield_load"], "status": "running",
              "nr_threads_override": nr_threads,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
              "sampling": "first topology-changing reconnection cycle after checkpoint and stress peak"}
    try:
        with (workspace / "replay.log").open("w") as log:
            process = subprocess.Popen(
                [str(executable), "-d", str(source / "dumps" / dump_name),
                 "-c", str(config), "-o", str(private_output)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while True:
                folders = sorted(steps.glob("step*"), key=lambda p: int(p.name[4:]))
                for folder in folders:
                    for before, after in reconnection_pairs(folder):
                        if before.name in checked or not _closed_vtu(before) or not _closed_vtu(after):
                            continue
                        load = float(re.search(r"_load=([^_]+)_", before.name).group(1))
                        if load <= job["yield_load"] or load > job["max_load"] + 1e-9:
                            checked.add(before.name)
                            continue
                        arrays = compact_flip_pair(before, after)
                        checked.add(before.name)
                        if len(arrays["element_indices"]) == 0:
                            continue
                        stop_owned_process(process)
                        np.savez_compressed(target, **arrays)
                        result.update(status="complete", load=load,
                                      before_vtu=before.name, after_vtu=after.name,
                                      nr_elements=len(arrays["element_indices"]),
                                      seconds=time.monotonic() - started)
                        metadata_path.write_text(json.dumps(result, indent=2) + "\n")
                        success = True
                        return result
                if process.poll() is not None:
                    if process.returncode != 0:
                        raise RuntimeError(f"Replay exited {process.returncode}: {workspace / 'replay.log'}")
                    result.update(status="no_post_yield_flip_before_endpoint", seconds=time.monotonic() - started)
                    metadata_path.write_text(json.dumps(result, indent=2) + "\n")
                    success = True
                    return result
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"No complete flip pair in {timeout:g}s: {workspace}")
                time.sleep(1)
    except Exception as error:
        result.update(status="error", error=str(error), workspace=str(workspace), seconds=time.monotonic() - started)
        metadata_path.write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        if process is not None:
            stop_owned_process(process)
        if success:
            shutil.rmtree(workspace)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task-index", type=int,
                        help="Run exactly one dump from the flattened manifest.")
    parser.add_argument("--nr-threads", type=int,
                        help="Override the replay's simulation thread count.")
    args = parser.parse_args()
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Checkpoint replay analysis must run through Slurm.")
    if (os.environ.get("SLURM_CPUS_PER_TASK") != "1"
            or args.workers != 1 or args.nr_threads != 1
            or args.task_index is None):
        raise RuntimeError("Each Slurm task must run one dump with one CPU thread.")
    manifest = json.loads(args.manifest.read_text())
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.nr_threads is not None and args.nr_threads < 1:
        raise ValueError("nr-threads must be positive")
    selected = [(job, dump) for job in manifest["runs"] for dump in job["dumps"]]
    if args.task_index is not None:
        if not 0 <= args.task_index < len(selected):
            raise ValueError("task-index is outside the manifest")
        selected = [selected[args.task_index]]
    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance = {"binary_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                  "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  "started_at_unix": time.time(), "nr_threads_override": args.nr_threads,
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "task_index": args.task_index}
    provenance_path = args.output_root / "provenance.json"
    if args.task_index is not None:
        provenance_path = args.output_root / "task_provenance" / f"task_{args.task_index:05d}.json"
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        provenance["manifest"] = manifest
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    failed = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(sample_checkpoint, job, dump, args.executable, args.output_root,
                             args.timeout, args.nr_threads)
                 for job, dump in selected]
        for task in as_completed(tasks):
            result = task.result()
            print(json.dumps(result), flush=True)
            failed |= result["status"] == "error"
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
