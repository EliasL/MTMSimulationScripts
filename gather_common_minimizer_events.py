#!/usr/bin/env python3
"""Collect matched minimization curves from existing non-reconnecting dumps.

Each source dump is processed independently to keep memory and disk use bounded:
the modified MTS2D binary searches for one pre-event dump, then FIRE and CG are
run in parallel and LBFGS is run afterward from that exact dump.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import re
import subprocess
from pathlib import Path

from compare_minimizers_same_event import ALGORITHMS, dump_load, run_one


def parse_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def source_dumps(
    source_job: Path, explicit: list[Path] | None, min_load: float | None
) -> list[Path]:
    if explicit:
        dumps = [Path(path) for path in explicit]
    else:
        dump_dir = source_job / "dumps"
        dumps = sorted(
            [*dump_dir.glob("dump_l*.xml.gz"), *dump_dir.glob("dump_l*.xml")],
            key=dump_load,
        )
    if not dumps:
        raise FileNotFoundError("No source dumps were found.")
    for dump in dumps:
        if not dump.is_file():
            raise FileNotFoundError(dump)
    unique: list[Path] = []
    seen_loads: set[float] = set()
    for dump in dumps:
        load = dump_load(dump)
        if min_load is not None and load < min_load:
            continue
        if load not in seen_loads:
            unique.append(dump)
            seen_loads.add(load)
    return unique


def config_value(config: dict[str, str], key: str, default: str) -> str:
    return config.get(key, default)


def collector_config(
    source_config: dict[str, str], source_load: float, max_load: float, name: str,
    threads: int, threshold: float,
) -> str:
    rows = config_value(source_config, "rows", "100")
    cols = config_value(source_config, "cols", "100")
    if rows != "100" or cols != "100":
        raise ValueError(f"This collector is for 100x100 systems, got {rows}x{cols}.")
    if config_value(source_config, "usingPBC", "true").lower() != "true":
        raise ValueError("The common-event collector requires periodic boundaries.")
    if config_value(source_config, "reconnectionMethod", "").lower() != "none":
        raise ValueError("Refusing a source job that allows reconnection.")
    return f"""rows = 100
cols = 100
usingPBC = true
reconnectionMethod = none
experiment = reversibilityProtocolTest
nrThreads = {threads}
seed = {config_value(source_config, 'seed', '0')}
QDSD = {config_value(source_config, 'QDSD', '0.0')}
initialGuessNoise = 0.0
meshDiagonal = {config_value(source_config, 'meshDiagonal', 'major')}
energyFunction = {config_value(source_config, 'energyFunction', 'contiSquare')}
bulkModulus = {config_value(source_config, 'bulkModulus', '4.0')}
startLoad = {config_value(source_config, 'startLoad', '0.0')}
loadIncrement = {config_value(source_config, 'loadIncrement', '1e-5')}
maxLoad = {max_load:.17g}
GP1 = 0.0
GP2 = 0.0
GP3 = 0.0
minimizer = LBFGS
epsR = {config_value(source_config, 'epsR', '0.0')}
LBFGSNrCorrections = {config_value(source_config, 'LBFGSNrCorrections', '3')}
LBFGSScale = {config_value(source_config, 'LBFGSScale', '1.0')}
LBFGSEpsg = 1e-5
LBFGSEpsf = {config_value(source_config, 'LBFGSEpsf', '0.0')}
LBFGSEpsx = 0.0
LBFGSMaxIterations = {config_value(source_config, 'LBFGSMaxIterations', '0')}
CGEpsg = 1e-5
CGEpsf = 0.0
CGEpsx = 0.0
CGMaxIterations = 0
finc = 1.1
fdec = 0.5
alphaStart = 0.1
falpha = 0.99
dtStart = 0.01
dtMax = 0.03
dtMin = 1e-12
maxCompS = 0.01
eps = 1e-5
epsRel = 0.0
delta = 0.0
maxIt = 200000
logDuringMinimization = false
fullMinimizationLogging = false
writeDebugVTUs = false
writeDumps = false
writeMeshVTUs = false
dumpPreEventAfterReversibility = true
saveElasticReversibilityStates = false
maximumSavedElasticReversibilityStates = 0
saveFinalReversibilityState = false
nrVTUFrames = 1
plasticityEventThreshold = {config_value(source_config, 'plasticityEventThreshold', '0.05')}
energyDropThreshold = {threshold:.17g}
showProgress = -1
name = {name}
"""


def existing_event_loads(output_root: Path) -> set[float]:
    loads: set[float] = set()
    for manifest in output_root.glob("event_*/event_manifest.json"):
        payload = json.loads(manifest.read_text())
        loads.add(float(payload["pre_event_load"]))
    return loads


def record_failure(event_root: Path, stage: str, error: Exception) -> None:
    (event_root / "failure.json").write_text(
        json.dumps(
            {"stage": stage, "error": repr(error), "status": "failed"},
            indent=2,
            sort_keys=True,
        )
    )


def collect_one(
    source_dump: Path,
    source_config: dict[str, str],
    binary: Path,
    output_root: Path,
    event_index: int,
    args: argparse.Namespace,
) -> dict | None:
    source_load = dump_load(source_dump)
    if any(abs(source_load - old) < args.load_tolerance for old in existing_event_loads(output_root)):
        print(f"Skipping {source_dump}: an event at this load is already recorded.")
        return None

    tag = f"{event_index:04d}_source_{source_load:.8f}"
    event_root = output_root / f"event_{tag}"
    if event_root.exists():
        if (event_root / "failure.json").is_file():
            print(f"Skipping previously failed event: {event_root}")
            return None
        raise FileExistsError(f"Refusing to reuse event directory: {event_root}")
    event_root.mkdir(parents=True)
    collector_root = event_root / "collector_output"
    collector_root.mkdir()
    collector_name = f"collector_{tag}"
    collector_conf = event_root / "collector.conf"
    collector_conf.write_text(
        collector_config(
            source_config,
            source_load,
            source_load + args.collector_lookahead,
            collector_name,
            args.threads,
            args.collector_threshold,
        )
    )
    collector_command = [
        str(binary), "-c", str(collector_conf), "-d", str(source_dump),
        "-o", str(collector_root), "-r",
    ]
    print(f"\n=== Collecting from {source_dump} ===")
    print(" ".join(collector_command))
    try:
        with (event_root / "collector.log").open("w") as log:
            completed = subprocess.run(
                collector_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.collector_timeout,
                check=False,
            )
    except Exception as error:
        record_failure(event_root, "collector", error)
        print(f"Collector failed for {source_dump}: {error}")
        return None
    if completed.returncode != 0:
        error = RuntimeError(
            f"Collector failed with status {completed.returncode}: {event_root}"
        )
        record_failure(event_root, "collector", error)
        print(error)
        return None

    pre_event_dumps = sorted(
        collector_root.rglob("pre_event_l_*.xml.gz"), key=dump_load
    )
    if len(pre_event_dumps) == 0:
        print(f"No qualifying event found from {source_dump}; removing empty event data.")
        shutil.rmtree(event_root)
        return None
    if len(pre_event_dumps) != 1:
        raise RuntimeError(f"Expected one pre-event dump, found {pre_event_dumps}")
    pre_event_dump = pre_event_dumps[0]
    pre_event_load = dump_load(pre_event_dump)
    if any(abs(pre_event_load - old) < args.load_tolerance for old in existing_event_loads(output_root)):
        print(f"Skipping duplicate pre-event load {pre_event_load:.17g}.")
        shutil.rmtree(event_root)
        return None

    matched_root = event_root / "matched"
    matched_root.mkdir()
    common = dict(
        binary=binary,
        source_dump=pre_event_dump,
        output_root=matched_root,
        load=pre_event_load,
        threads=args.threads,
        lookahead=args.minimizer_lookahead,
        threshold=args.minimizer_threshold,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.minimizer_timeout,
        dry_run=args.dry_run,
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                algorithm: pool.submit(run_one, algorithm=algorithm, **common)
                for algorithm in ("FIRE", "CG")
            }
            parallel_results = [
                futures[algorithm].result() for algorithm in ("FIRE", "CG")
            ]
        lbfgs_result = run_one(algorithm="LBFGS", **common)
    except Exception as error:
        record_failure(event_root, "minimizer", error)
        print(f"Skipping incomplete event {event_root}: {error}")
        return None
    payload = {
        "source_dump": str(source_dump),
        "source_load": source_load,
        "pre_event_dump": str(pre_event_dump),
        "pre_event_load": pre_event_load,
        "algorithms": ALGORITHMS,
        "results": parallel_results + [lbfgs_result],
        "status": "completed" if not args.dry_run else "dry-run",
    }
    manifest_path = event_root / "event_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if not args.dry_run:
        from plot_common_event_minimization import plot_manifest

        plot_manifest(manifest_path, event_root)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-job", type=Path)
    group.add_argument("--source-dump", type=Path, action="append")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-source-dumps", type=int, default=3)
    parser.add_argument(
        "--min-source-load",
        type=float,
        help="Only use source dumps with physical load strictly above this value.",
    )
    parser.add_argument("--collector-lookahead", type=float, default=0.01)
    parser.add_argument("--collector-threshold", type=float, default=1e-4)
    parser.add_argument("--minimizer-lookahead", type=float, default=2e-4)
    parser.add_argument("--minimizer-threshold", type=float, default=1e-10)
    parser.add_argument("--load-tolerance", type=float, default=5e-9)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--collector-timeout", type=float, default=7200.0)
    parser.add_argument("--minimizer-timeout", type=float, default=7200.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    if args.threads < 1 or args.max_source_dumps < 0:
        raise ValueError("threads must be positive and max-source-dumps nonnegative")
    source_job = args.source_job
    source_config: dict[str, str] = {}
    if source_job is not None:
        source_job = source_job.resolve()
        source_config_path = source_job / "config.conf"
        if not source_config_path.is_file():
            raise FileNotFoundError(source_config_path)
        source_config = parse_config(source_config_path)
    dumps = source_dumps(source_job or Path("."), args.source_dump, args.min_source_load)
    if args.max_source_dumps:
        dumps = dumps[:args.max_source_dumps]
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, source_dump in enumerate(dumps, start=1):
        result = collect_one(
            source_dump.resolve(), source_config, args.binary.resolve(),
            args.output_root.resolve(), index, args,
        )
        if result is not None:
            results.append(result)
    (args.output_root / "batch_manifest.json").write_text(
        json.dumps({"events": results, "count": len(results)}, indent=2, sort_keys=True)
    )
    print(f"Completed {len(results)} matched events.")


if __name__ == "__main__":
    main()
