#!/usr/bin/env python3
"""Run FIRE, CG, and LBFGS on the same restart dump until one drop occurs.

The MTS2D executable has no stop-at-first-drop option.  This wrapper therefore
uses a generous load look-ahead, watches macroData.csv for the first negative
energy change, waits for the corresponding minimization directory to finish,
and then stops that run before starting the next algorithm.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import signal
import shutil
import subprocess
import time
from pathlib import Path


ALGORITHMS = ("FIRE", "CG", "LBFGS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source-dump", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--lookahead", type=float, default=0.03)
    parser.add_argument("--energy-threshold", type=float, default=1e-10)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dump_load(dump_path: Path) -> float:
    opener = gzip.open if dump_path.name.endswith(".gz") else open
    with opener(dump_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            match = re.search(r"<load>([-+0-9.eE]+)</load>", line)
            if match:
                return float(match.group(1))
    raise RuntimeError(f"Could not find <load> in {dump_path}")


def config_text(algorithm: str, load: float, max_load: float, threads: int, name: str) -> str:
    values = {
        "FIRE": ("1e-5", "0", "1e-5", "10000"),
        "CG": ("1e-5", "0", "1e-5", "0"),
        "LBFGS": ("1e-5", "0", "1e-5", "0"),
    }
    cg_epsg, cg_epsx, fire_eps, max_it = values[algorithm]
    return f"""rows = 100
cols = 100
usingPBC = true
reconnectionMethod = none
reconnectRevert = 1
reconnectEdgeLocking = 0
experiment = simpleShear
nrThreads = {threads}
seed = 0
QDSD = 0.0
initialGuessNoise = 0.05
meshDiagonal = major
energyFunction = contiSquare
bulkModulus = 4
startLoad = {load:.17g}
loadIncrement = 1e-5
maxLoad = {max_load:.17g}
GP1 = 0.0
GP2 = 0.0
GP3 = 0.0
minimizer = {algorithm}
epsR = 1e-20
LBFGSNrCorrections = 7
LBFGSScale = 1.0
        LBFGSEpsg = 1e-5
LBFGSEpsf = 0.0
LBFGSEpsx = 0
LBFGSMaxIterations = 0
CGScale = 1.0
CGEpsg = {cg_epsg}
CGEpsf = 0.0
CGEpsx = {cg_epsx}
CGMaxIterations = 0
finc = 1.1
fdec = 0.5
alphaStart = 0.1
falpha = 0.99
dtStart = 0.01
dtMax = 0.03
dtMin = 1e-8
maxCompS = 0.01
eps = {fire_eps}
epsRel = 0.0
delta = 0.0
maxIt = {max_it}
logDuringMinimization = 1
fullMinimizationLogging = 1
writeDumps = 0
writeMeshVTUs = 0
writeDebugVTUs = 0
nrVTUFrames = 200
plasticityEventThreshold = 0.1
energyDropThreshold = 1e-10
showProgress = 1
name = {name}
"""


def read_first_drop(macro_path: Path, threshold: float) -> dict | None:
    if not macro_path.is_file() or macro_path.stat().st_size == 0:
        return None
    try:
        with macro_path.open(newline="") as stream:
            rows = csv.DictReader(stream)
            if not rows.fieldnames:
                return None
            field = next(
                (candidate for candidate in ("avg_energy_change", "total_energy_change") if candidate in rows.fieldnames),
                None,
            )
            if field is None:
                raise RuntimeError(f"No energy-change column in {macro_path}: {rows.fieldnames}")
            for row in rows:
                try:
                    value = float(row[field])
                except (TypeError, ValueError):
                    continue
                if value < -threshold:
                    row["_energy_change_field"] = field
                    row["_energy_change"] = value
                    return row
    except (OSError, csv.Error):
        # The executable can be in the middle of appending a row.
        return None
    return None


def latest_load_step(macro_path: Path) -> int | None:
    if not macro_path.is_file():
        return None
    try:
        with macro_path.open(newline="") as stream:
            rows = csv.DictReader(stream)
            last_row = None
            for row in rows:
                last_row = row
            return int(last_row["load_step"]) if last_row else None
    except (OSError, csv.Error, KeyError, TypeError, ValueError):
        return None


def first_energy_change(macro_path: Path) -> float | None:
    if not macro_path.is_file():
        return None
    try:
        with macro_path.open(newline="") as stream:
            rows = csv.DictReader(stream)
            if not rows.fieldnames:
                return None
            field = next(
                (candidate for candidate in ("avg_energy_change", "total_energy_change") if candidate in rows.fieldnames),
                None,
            )
            row = next(rows, None)
            return float(row[field]) if row is not None and field is not None else None
    except (OSError, csv.Error, KeyError, TypeError, ValueError):
        return None


def root_energy_changes(macro_path: Path) -> list[tuple[float, float]]:
    changes = []
    try:
        with macro_path.open(newline="") as stream:
            rows = csv.DictReader(stream)
            if not rows.fieldnames:
                return changes
            field = next(
                (candidate for candidate in ("avg_energy_change", "total_energy_change") if candidate in rows.fieldnames),
                None,
            )
            if field is None:
                return changes
            for row in rows:
                try:
                    changes.append((float(row["load"]), float(row[field])))
                except (KeyError, TypeError, ValueError):
                    continue
    except (OSError, csv.Error):
        return []
    return changes


def step_load(step_dir: Path) -> float | None:
    macro_path = step_dir / "macroData.csv"
    try:
        with macro_path.open(newline="") as stream:
            row = next(csv.DictReader(stream), None)
            return float(row["load"]) if row is not None else None
    except (OSError, csv.Error, KeyError, TypeError, ValueError):
        return None


def matching_root_energy(load: float, changes: list[tuple[float, float]]) -> float | None:
    matches = [value for root_load, value in changes if abs(root_load - load) < 5e-9]
    if not matches:
        return None
    negative_matches = [value for value in matches if value < 0]
    return negative_matches[-1] if negative_matches else matches[-1]


def prune_non_drop_minimizations(
    data_root: Path,
    current_step: int | None,
    threshold: float,
    changes: list[tuple[float, float]],
) -> None:
    """Bound disk use while MTS2D retains every restart minimization folder."""
    if current_step is None or not data_root.is_dir():
        return
    for step_dir in data_root.glob("step*"):
        if not step_dir.is_dir():
            continue
        match = re.fullmatch(r"step(\d+)", step_dir.name)
        if match is None or int(match.group(1)) >= current_step - 2:
            continue
        load = step_load(step_dir)
        value = matching_root_energy(load, changes) if load is not None else None
        if value is not None and value >= -threshold:
            shutil.rmtree(step_dir)


def stop_process(process: subprocess.Popen[bytes]) -> int:
    if process.poll() is not None:
        return process.returncode
    process.send_signal(signal.SIGINT)
    try:
        return process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            return process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=15)


def run_one(
    binary: Path,
    source_dump: Path,
    output_root: Path,
    algorithm: str,
    load: float,
    threads: int,
    lookahead: float,
    threshold: float,
    poll_seconds: float,
    timeout_seconds: float,
    dry_run: bool,
) -> dict:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_dump.name.replace(".xml.gz", ""))
    name = f"same_event_{stem}_{algorithm}"
    run_dir = output_root / name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing output directory: {run_dir}")
    run_dir.mkdir(parents=True)
    config_path = run_dir / "launch.conf"
    config_path.write_text(config_text(algorithm, load, load + lookahead, threads, name))
    command = [str(binary), "-d", str(source_dump), "-c", str(config_path), "-o", str(output_root)]
    print(f"\n=== {algorithm} ===")
    print(" ".join(command))
    if dry_run:
        return {"algorithm": algorithm, "command": command, "status": "dry-run"}

    log_path = run_dir / "run.log"
    with log_path.open("wb") as log_stream:
        process = subprocess.Popen(command, stdout=log_stream, stderr=subprocess.STDOUT)
        macro_path = run_dir / "macroData.csv"
        started = time.monotonic()
        first_drop = None
        while True:
            first_drop = read_first_drop(macro_path, threshold)
            data_root = run_dir / "data" / "minimizationData"
            prune_non_drop_minimizations(
                data_root,
                latest_load_step(macro_path),
                threshold,
                root_energy_changes(macro_path),
            )
            if first_drop is not None:
                print(f"  first drop at load={first_drop.get('load')} ΔE={first_drop['_energy_change']}")
                # writeToFile() has just retained the minimization folder, but give
                # collection generation a moment to finish before stopping.
                time.sleep(max(1.0, 2 * poll_seconds))
                break
            if process.poll() is not None:
                raise RuntimeError(f"{algorithm} exited with status {process.returncode} before an energy drop")
            if time.monotonic() - started > timeout_seconds:
                stop_process(process)
                raise TimeoutError(f"{algorithm} exceeded {timeout_seconds}s without an energy drop")
            time.sleep(poll_seconds)

        returncode = stop_process(process)
    changes = root_energy_changes(macro_path)
    data_dirs = sorted(
        path
        for path in (run_dir / "data" / "minimizationData").glob("step*")
        if path.is_dir()
        and (step_load_value := step_load(path)) is not None
        and (value := matching_root_energy(step_load_value, changes)) is not None
        and value < -threshold
    )
    if not data_dirs:
        raise RuntimeError(f"{algorithm} recorded a drop but retained no minimization directory in {run_dir / 'data'}")
    result = {
        "algorithm": algorithm,
        "source_dump": str(source_dump),
        "dump_load": load,
        "returncode_after_first_drop": returncode,
        "first_drop": first_drop,
        "minimization_directories": [str(path) for path in data_dirs],
        "command": command,
        "status": "completed-first-drop",
    }
    (run_dir / "first_drop.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    if not args.source_dump.is_file():
        raise FileNotFoundError(args.source_dump)
    if args.threads < 1 or args.lookahead <= 0 or args.energy_threshold < 0:
        raise ValueError("threads, lookahead, and energy threshold must be valid positive values")
    load = dump_load(args.source_dump)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"Source dump: {args.source_dump}")
    print(f"Current dump load: {load:.17g}; look-ahead max load: {load + args.lookahead:.17g}")
    for algorithm in args.algorithms:
        results.append(
            run_one(
                args.binary,
                args.source_dump,
                args.output_root,
                algorithm,
                load,
                args.threads,
                args.lookahead,
                args.energy_threshold,
                args.poll_seconds,
                args.timeout_seconds,
                args.dry_run,
            )
        )
    manifest = {
        "source_dump": str(args.source_dump),
        "dump_load": load,
        "algorithms": args.algorithms,
        "results": results,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
