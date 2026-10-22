#!/usr/bin/env python3
"""Portable, sequential force/minimization benchmark-suite runner for MTS2D.

The native benchmark owns the timed region. This runner owns experiment
generation, independent-process repetitions, OpenMP affinity, time/memory
budgets, metadata, statistics, and plots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


BENCHMARK_ROOT = Path(__file__).resolve().parent
SCRIPT_ROOT = BENCHMARK_ROOT.parent
REPO_ROOT = SCRIPT_ROOT.parent / "MTS2D"
DEFAULT_BUILD_DIR = SCRIPT_ROOT / "build-benchmark"
DEFAULT_OUTPUT_DIR = SCRIPT_ROOT / "output" / "benchmarks"
DEFAULT_INITIAL_CONDITIONS_DIR = BENCHMARK_ROOT / "initialConditions"


PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "experiments": ["system", "strong", "weak", "workload"],
        "modes": ["force-kernel", "force-evaluation"],
        "sizes": [8, 16],
        "workload_sizes": [8, 16],
        "workload_loads": [0.15, 0.7],
        "strong_size": 16,
        "weak_base_nodes": 64,
        "repetitions": 2,
        "sample_seconds": 0.05,
        "max_calls": 20,
        "min_calls": 2,
        "warmup_calls": 2,
        "case_budget_minutes": 0.5,
        "total_budget_hours": 5.0 / 60.0,
        "thread_cap": 2,
    },
    "quick": {
        "experiments": ["system", "strong", "weak", "workload"],
        "modes": ["force-kernel", "force-evaluation"],
        "sizes": [32, 64, 128, 256],
        "workload_sizes": [50, 100],
        "workload_loads": [0.15, 0.7],
        "strong_size": 256,
        "weak_base_nodes": 4096,
        "repetitions": 5,
        "sample_seconds": 2.0,
        "max_calls": 10_000,
        "min_calls": 5,
        "warmup_calls": 5,
        "case_budget_minutes": 20.0,
        "total_budget_hours": 1.0,
        "thread_cap": None,
    },
    "shortBenchmark": {
        "experiments": ["system", "strong", "weak", "workload"],
        "modes": ["force-kernel", "force-evaluation"],
        "sizes": [32, 64, 128, 256, 512, 1024],
        "workload_sizes": [50, 100],
        "workload_loads": [0.15, 0.7],
        "strong_size": 512,
        "weak_base_nodes": 4096,
        "repetitions": 3,
        "sample_seconds": 2.0,
        "max_calls": 10_000,
        "min_calls": 5,
        "warmup_calls": 5,
        "case_budget_minutes": 15.0,
        # The ceiling is deliberately generous; the suite finishes as soon as
        # the planned samples are complete.
        "total_budget_hours": 0.9,
        "thread_cap": None,
        "workload_threads": "screen",
    },
    "full": {
        "experiments": ["system", "strong", "weak", "workload"],
        "modes": ["force-kernel", "force-evaluation"],
        "sizes": [32, 64, 128, 256, 512, 1024],
        "workload_sizes": [50, 100, 150, 200, 250, 500],
        "workload_loads": [0.15, 0.7],
        "strong_size": 512,
        "weak_base_nodes": 4096,
        "repetitions": 10,
        "sample_seconds": 10.0,
        "max_calls": 10_000,
        "min_calls": 5,
        "warmup_calls": 10,
        "case_budget_minutes": 20.0,
        "total_budget_hours": 12.0,
        "thread_cap": None,
        "large_workload_size_threshold": 500,
        "large_workload_repetitions": 3,
        "large_workload_threads": "sparse",
        "large_initial_condition_budget_minutes": 40.0,
    },
    "custom": {
        "experiments": ["system", "strong", "weak", "workload"],
        "modes": ["force-evaluation"],
        "sizes": [32, 64],
        "workload_sizes": [16, 32],
        "workload_loads": [0.15, 0.7],
        "strong_size": 64,
        "weak_base_nodes": 1024,
        "repetitions": 5,
        "sample_seconds": 2.0,
        "max_calls": 10_000,
        "min_calls": 5,
        "warmup_calls": 5,
        "case_budget_minutes": 20.0,
        "total_budget_hours": 1.0,
        "thread_cap": None,
    },
}


@dataclass(frozen=True)
class Case:
    case_id: str
    experiment: str
    mode: str
    rows: int
    cols: int
    threads: int
    target_nodes: int
    nodes_per_thread: float
    scenario: str
    reconnection: str
    history: str
    fixture_load: float
    affinity_policy: str
    repetition_limit: int | None = None


class BenchmarkError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_text(command: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT if cwd is None else cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid range: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("Expected positive comma-separated integers.")
    return sorted(set(values))


def parse_float_list(text: str) -> list[float]:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise argparse.ArgumentTypeError(
            "Expected non-negative, finite comma-separated loads."
        )
    return sorted(set(values))


def load_label(load: float) -> str:
    return format(load, ".12g")


def initial_condition_path(
    root: Path, history: str, load: float, rows: int, cols: int, seed: int
) -> Path:
    return (
        root
        / history
        / f"load_{load_label(load)}"
        / f"size_{rows}x{cols}"
        / f"seed_{seed}.xml.gz"
    )


def fixture_seeds(base: int, stride: int, repetitions: int) -> list[int]:
    return sorted({base + stride * repetition for repetition in range(repetitions)})


def available_initial_condition_seeds(
    root: Path, history: str, load: float, rows: int, cols: int
) -> list[int]:
    directory = (
        root
        / history
        / f"load_{load_label(load)}"
        / f"size_{rows}x{cols}"
    )
    seeds: list[int] = []
    for path in directory.glob("seed_*.xml.gz"):
        match = re.fullmatch(r"seed_(\d+)\.xml\.gz", path.name)
        if match is not None:
            seeds.append(int(match.group(1)))
    return sorted(set(seeds))


def initial_condition_plan(
    cases: list[Case],
    root: Path,
    seeds: list[int],
    generate_load_015: bool,
    allow_available_seed_fallback: bool = False,
) -> list[dict[str, Any]]:
    specifications = {
        (case.history, case.reconnection, case.fixture_load, case.rows, case.cols)
        for case in cases
        if case.experiment == "workload"
    }
    plan: list[dict[str, Any]] = []
    for history, reconnection, load, rows, cols in sorted(specifications):
        selected_seeds = list(seeds)
        requested_seed: int | None = None
        if allow_available_seed_fallback and len(seeds) == 1:
            requested_seed = seeds[0]
            requested_path = initial_condition_path(
                root, history, load, rows, cols, requested_seed
            )
            is_generated_load = math.isclose(
                load, 0.15, rel_tol=0.0, abs_tol=1e-12
            )
            if not requested_path.is_file() and not (
                is_generated_load and generate_load_015
            ):
                available = available_initial_condition_seeds(
                    root, history, load, rows, cols
                )
                if available:
                    selected_seeds = [available[0]]
        for seed in selected_seeds:
            path = initial_condition_path(root, history, load, rows, cols, seed)
            if path.is_file():
                status = "existing"
            elif math.isclose(load, 0.15, rel_tol=0.0, abs_tol=1e-12) and generate_load_015:
                status = "generate"
            else:
                status = "missing_skip"
            plan.append(
                {
                    "history": history,
                    "reconnection": reconnection,
                    "load": load,
                    "rows": rows,
                    "cols": cols,
                    "seed": seed,
                    "requested_seed": requested_seed if requested_seed is not None else seed,
                    "used_available_seed_fallback": (
                        requested_seed is not None and seed != requested_seed
                    ),
                    "path": str(path),
                    "status": status,
                }
            )
    return plan


def sample_fixture_seed(
    case: Case,
    plan: list[dict[str, Any]],
    base_seed: int,
    seed_stride: int,
    repetition: int,
) -> int:
    requested = base_seed + seed_stride * (repetition - 1)
    if case.experiment != "workload":
        return requested
    matches = [
        row
        for row in plan
        if row["history"] == case.history
        and math.isclose(
            float(row["load"]), case.fixture_load, rel_tol=0.0, abs_tol=1e-12
        )
        and int(row["rows"]) == case.rows
        and int(row["cols"]) == case.cols
        and row["status"] == "existing"
    ]
    exact = [row for row in matches if int(row["seed"]) == requested]
    selected = exact or matches
    if len(selected) != 1:
        raise BenchmarkError(
            f"{case.case_id}: expected one selected initial-condition seed, "
            f"found {len(selected)}."
        )
    return int(selected[0]["seed"])


def filter_cases_with_initial_conditions(
    cases: list[Case], root: Path, seeds: list[int]
) -> tuple[list[Case], list[dict[str, Any]]]:
    runnable: list[Case] = []
    skipped: list[dict[str, Any]] = []
    for case in cases:
        if case.experiment != "workload":
            runnable.append(case)
            continue
        missing = [
            initial_condition_path(
                root,
                case.history,
                case.fixture_load,
                case.rows,
                case.cols,
                seed,
            )
            for seed in seeds
        ]
        missing = [path for path in missing if not path.is_file()]
        if missing:
            skipped.append(
                {
                    "case_id": case.case_id,
                    "reason": "required initial condition is missing",
                    "history": case.history,
                    "load": case.fixture_load,
                    "missing_paths": [str(path) for path in missing],
                }
            )
        else:
            runnable.append(case)
    return runnable, skipped


def filter_cases_by_plan(
    cases: list[Case], plan: list[dict[str, Any]], seeds: list[int]
) -> tuple[list[Case], list[dict[str, Any]]]:
    unavailable = {
        (
            row["history"],
            row["load"],
            row["rows"],
            row["cols"],
            row["seed"],
        ): row
        for row in plan
        if row["status"] == "missing_skip"
    }
    runnable: list[Case] = []
    skipped: list[dict[str, Any]] = []
    for case in cases:
        missing = [
            unavailable.get(
                (case.history, case.fixture_load, case.rows, case.cols, seed)
            )
            for seed in seeds
        ]
        missing = [row for row in missing if row is not None]
        if case.experiment == "workload" and missing:
            skipped.append(
                {
                    "case_id": case.case_id,
                    "reason": "required initial condition is missing and will not be generated",
                    "history": case.history,
                    "load": case.fixture_load,
                    "missing_paths": [row["path"] for row in missing],
                }
            )
        else:
            runnable.append(case)
    return runnable, skipped


def parse_names(text: str, allowed: set[str], label: str) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    invalid = sorted(set(values) - allowed)
    if not values or invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid {label}: {', '.join(invalid) if invalid else text}"
        )
    return values


def parse_cpu_list(text: str) -> list[int]:
    values: list[int] = []
    for token in text.strip().split(","):
        if not token:
            continue
        if "-" in token:
            start, end = (int(value) for value in token.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    return sorted(set(values))


def allowed_cpus() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        try:
            return sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    return list(range(os.cpu_count() or 1))


def read_int(path: Path, default: int = -1) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return default


def linux_topology(cpus: list[int]) -> dict[str, Any]:
    if platform.system() != "Linux":
        return {}

    numa_by_cpu: dict[int, int] = {}
    for node_path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        node = int(node_path.name[4:])
        try:
            node_cpus = parse_cpu_list(
                (node_path / "cpulist").read_text(encoding="utf-8")
            )
        except OSError:
            continue
        for cpu in node_cpus:
            numa_by_cpu[cpu] = node

    cpu_records: list[dict[str, int]] = []
    for cpu in cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        record = {
            "cpu": cpu,
            "package": read_int(topology / "physical_package_id"),
            "die": read_int(topology / "die_id"),
            "core": read_int(topology / "core_id", cpu),
            "numa_node": numa_by_cpu.get(cpu, -1),
        }
        cpu_records.append(record)

    core_keys = {
        (record["package"], record["die"], record["core"])
        for record in cpu_records
    }
    socket_counts: dict[int, set[tuple[int, int]]] = {}
    for record in cpu_records:
        socket_counts.setdefault(record["package"], set()).add(
            (record["die"], record["core"])
        )
    return {
        "cpus": cpu_records,
        "physical_cores_available": len(core_keys),
        "logical_cpus_available": len(cpu_records),
        "sockets": len(socket_counts),
        "physical_cores_per_socket": {
            str(package): len(cores) for package, cores in sorted(socket_counts.items())
        },
        "numa_nodes": sorted(set(numa_by_cpu.values())),
    }


def ordered_physical_place_cpus(topology: dict[str, Any]) -> list[int]:
    """Choose one logical CPU per physical core, grouped by NUMA locality."""
    core_records: dict[tuple[int, int, int], list[dict[str, int]]] = {}
    for record in topology.get("cpus", []):
        key = (int(record["package"]), int(record["die"]), int(record["core"]))
        core_records.setdefault(key, []).append(record)

    representatives: list[dict[str, int]] = []
    for records in core_records.values():
        representatives.append(min(records, key=lambda item: int(item["cpu"])))

    # CPU numbering can alternate sockets (as on Cauchy), so numeric CPU order
    # is not a locality order. Grouping by NUMA/package makes OpenMP close use
    # nearby cores first; spread then samples evenly across that same list.
    representatives.sort(
        key=lambda item: (
            int(item["numa_node"]) if int(item["numa_node"]) >= 0 else 1 << 30,
            int(item["package"]),
            int(item["die"]),
            int(item["core"]),
            int(item["cpu"]),
        )
    )
    return [int(record["cpu"]) for record in representatives]


def available_memory_bytes() -> int | None:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size) * int(available_pages)
    except (ValueError, OSError, AttributeError):
        return None


def standard_thread_counts(max_threads: int, topology: dict[str, Any]) -> list[int]:
    del topology
    values = {1, max_threads}
    power = 2
    while power < max_threads:
        values.add(power)
        power *= 2
    return sorted(values)


def screening_thread_counts(max_threads: int, topology: dict[str, Any]) -> list[int]:
    """Return a sparse power-of-two grid for expensive minimizations."""
    del topology
    values = {1, max_threads}
    for count in (2, 4, 8, 16, 32, 64):
        if count <= max_threads:
            values.add(count)
    if max_threads < 8:
        power = 2
        while power < max_threads:
            values.add(power)
            power *= 2
    return sorted(values)


def sparse_large_thread_counts(max_threads: int) -> list[int]:
    """Use a small but informative grid for the most expensive workloads."""
    values = {1, max_threads}
    for count in (8, 32, 64):
        if count <= max_threads:
            values.add(count)
    return sorted(values)


def near_square_dimensions(target_nodes: int) -> tuple[int, int]:
    root = max(2, math.isqrt(target_nodes))
    for rows in range(root, 1, -1):
        if target_nodes % rows == 0:
            return rows, target_nodes // rows
    rows = root
    cols = math.ceil(target_nodes / rows)
    return rows, cols


def build_cases(
    experiments: list[str],
    modes: list[str],
    sizes: list[int],
    workload_sizes: list[int],
    workload_loads: list[float],
    strong_size: int,
    weak_base_nodes: int,
    threads: list[int],
    workload_threads: list[int],
    large_workload_threads: list[int],
    large_workload_size_threshold: int | None,
    large_workload_repetitions: int | None,
    fixed_threads: int,
    affinity_policies: list[str],
) -> list[Case]:
    cases: list[Case] = []
    force_affinity = affinity_policies[0]
    if "system" in experiments:
        for mode in modes:
            for size in sizes:
                nodes = size * size
                cases.append(
                    Case(
                        case_id=(
                            f"system_{mode}_s{size}_t{fixed_threads}_a{force_affinity}"
                        ),
                        experiment="system",
                        mode=mode,
                        rows=size,
                        cols=size,
                        threads=fixed_threads,
                        target_nodes=nodes,
                        nodes_per_thread=nodes / fixed_threads,
                        scenario="controlled",
                        reconnection="none",
                        history="controlled",
                        fixture_load=0.0,
                        affinity_policy=force_affinity,
                    )
                )
    if "strong" in experiments:
        for mode in modes:
            for thread_count in threads:
                nodes = strong_size * strong_size
                cases.append(
                    Case(
                        case_id=(
                            f"strong_{mode}_s{strong_size}_t{thread_count}_"
                            f"a{force_affinity}"
                        ),
                        experiment="strong",
                        mode=mode,
                        rows=strong_size,
                        cols=strong_size,
                        threads=thread_count,
                        target_nodes=nodes,
                        nodes_per_thread=nodes / thread_count,
                        scenario="controlled",
                        reconnection="none",
                        history="controlled",
                        fixture_load=0.0,
                        affinity_policy=force_affinity,
                    )
                )
    if "weak" in experiments:
        for mode in modes:
            for thread_count in threads:
                target = weak_base_nodes * thread_count
                rows, cols = near_square_dimensions(target)
                nodes = rows * cols
                cases.append(
                    Case(
                        case_id=(
                            f"weak_{mode}_r{rows}c{cols}_t{thread_count}_"
                            f"a{force_affinity}"
                        ),
                        experiment="weak",
                        mode=mode,
                        rows=rows,
                        cols=cols,
                        threads=thread_count,
                        target_nodes=target,
                        nodes_per_thread=nodes / thread_count,
                        scenario="controlled",
                        reconnection="none",
                        history="controlled",
                        fixture_load=0.0,
                        affinity_policy=force_affinity,
                    )
                )
    if "workload" in experiments:
        histories = (
            ("noReconnecting", "none"),
            ("edgeFlipping", "edgeFlip"),
        )
        for size in workload_sizes:
            nodes = size * size
            is_large = (
                large_workload_size_threshold is not None
                and size >= large_workload_size_threshold
            )
            size_threads = large_workload_threads if is_large else workload_threads
            repetition_limit = large_workload_repetitions if is_large else None
            for fixture_load in workload_loads:
                for history, reconnection in histories:
                    for thread_count in size_threads:
                        policies = (
                            affinity_policies[:1]
                            if thread_count == 1
                            else affinity_policies
                        )
                        for affinity in policies:
                            cases.append(
                                Case(
                                    case_id=(
                                        f"workload_replay_{history}_l"
                                        f"{load_label(fixture_load).replace('.', 'p')}_"
                                        f"s{size}_t{thread_count}_a{affinity}"
                                    ),
                                    experiment="workload",
                                    mode="minimization",
                                    rows=size,
                                    cols=size,
                                    threads=thread_count,
                                    target_nodes=nodes,
                                    nodes_per_thread=nodes / thread_count,
                                    scenario="replay",
                                    reconnection=reconnection,
                                    history=history,
                                    fixture_load=fixture_load,
                                    affinity_policy=affinity,
                                    repetition_limit=repetition_limit,
                                )
                            )
    return cases


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cmake_build_metadata(build_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    cache_path = build_dir / "CMakeCache.txt"
    selected_keys = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CXX_FLAGS_RELEASE",
        "ENABLE_FAST_MATH",
        "ENABLE_NOINLINE",
        "IDE_LIGHTWEIGHT",
        "MTS2D_SOURCE_DIR",
        "OpenMP_CXX_FLAGS",
        "OpenMP_omp_LIBRARY",
        "OpenMP_libomp_LIBRARY",
    }
    try:
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "//")) or "=" not in line:
                continue
            key_and_type, value = line.split("=", 1)
            key = key_and_type.split(":", 1)[0]
            if key in selected_keys:
                metadata[key] = value
    except OSError:
        pass
    flags_path = build_dir / "mts2d" / "CMakeFiles" / "shared_objs.dir" / "flags.make"
    try:
        effective_flags = [
            line.strip()
            for line in flags_path.read_text(encoding="utf-8").splitlines()
            if line.startswith(("CXX_DEFINES =", "CXX_FLAGS ="))
        ]
        metadata["effective_shared_object_flags"] = effective_flags
    except OSError:
        pass
    return metadata


def build_benchmark(args: argparse.Namespace) -> Path:
    requested_executable = args.exe.resolve() if args.exe is not None else None
    build_dir = args.build_dir.resolve()
    if args.no_build:
        executable = requested_executable or (build_dir / "benchmark_MTS2D")
        if not executable.exists():
            raise FileNotFoundError(executable)
        return executable

    configure = [
        "cmake",
        "-S",
        str(BENCHMARK_ROOT),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DIDE_LIGHTWEIGHT=ON",
        f"-DMTS2D_SOURCE_DIR={REPO_ROOT}",
        f"-DENABLE_NOINLINE={'ON' if args.keep_noinline else 'OFF'}",
    ]
    print("Configuring the benchmark build...", flush=True)
    subprocess.run(configure, cwd=REPO_ROOT, check=True)
    print("Building benchmark_MTS2D...", flush=True)
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "benchmark_MTS2D",
            "--parallel",
            str(args.build_jobs),
            "--clean-first",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    executable = requested_executable or (build_dir / "benchmark_MTS2D")
    if not executable.exists():
        raise FileNotFoundError(executable)
    return executable.resolve()


def affinity_environment(
    policy: str, threads: int, physical_place_cpus: list[int]
) -> dict[str, str]:
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["OMP_MAX_ACTIVE_LEVELS"] = "1"
    # Vendor-specific variables can override or conflict with the portable
    # OpenMP settings. Keep the reportable policy unambiguous.
    environment.pop("KMP_AFFINITY", None)
    environment.pop("GOMP_CPU_AFFINITY", None)
    places = (
        ",".join(f"{{{cpu}}}" for cpu in physical_place_cpus)
        if physical_place_cpus
        else "cores"
    )
    if policy in {"auto", "close"}:
        environment["OMP_PLACES"] = places
        environment["OMP_PROC_BIND"] = "close"
    elif policy == "spread":
        environment["OMP_PLACES"] = places
        environment["OMP_PROC_BIND"] = "spread"
    else:
        environment.pop("OMP_PLACES", None)
        environment["OMP_PROC_BIND"] = "false"
    return environment


def parse_native_json(stdout: str) -> dict[str, Any]:
    last_error = ""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            last_error = str(error)
            continue
        if isinstance(value, dict):
            return value
    tail = stdout.strip()[-1000:]
    detail = f" ({last_error})" if last_error else ""
    raise BenchmarkError(
        f"Native benchmark did not emit valid JSON{detail}; stdout tail: {tail!r}"
    )


def run_native(
    executable: Path,
    work_dir: Path,
    case: Case,
    calls: int,
    warmup_calls: int,
    timeout_seconds: float,
    force_shear: float,
    force_perturbation: float,
    workload_disorder: float,
    initial_noise: float,
    replay_load_tolerance: float,
    generated_load_increment: float,
    fixture_seed: int,
    physical_place_cpus: list[int],
    initial_condition: Path | None = None,
    save_initial_condition: Path | None = None,
    preconditioner: str = "exactTangent",
) -> dict[str, Any]:
    disorder = workload_disorder if case.mode == "minimization" else 0.0
    command = [
        str(executable),
        "--rows",
        str(case.rows),
        "--cols",
        str(case.cols),
        "--threads",
        str(case.threads),
        "--seed",
        str(fixture_seed),
        "--calls",
        str(calls),
        "--warmup-calls",
        str(warmup_calls),
        "--mode",
        case.mode,
        "--scenario",
        case.scenario,
        "--reconnection",
        case.reconnection,
        "--preconditioner",
        preconditioner,
        "--shear",
        repr(force_shear),
        "--perturbation",
        repr(force_perturbation),
        "--disorder",
        repr(disorder),
        "--initial-load",
        repr(case.fixture_load),
        "--initial-load-tolerance",
        repr(replay_load_tolerance),
        "--load-increment",
        repr(generated_load_increment),
        "--initial-noise",
        repr(initial_noise),
    ]
    if initial_condition is not None:
        command.extend(["--initial-condition", str(initial_condition)])
    if save_initial_condition is not None:
        command.extend(
            ["--save-initial-condition", str(save_initial_condition)]
        )
    try:
        result = subprocess.run(
            command,
            cwd=work_dir,
            env=affinity_environment(
                case.affinity_policy, case.threads, physical_place_cpus
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(0.1, timeout_seconds),
        )
    except subprocess.TimeoutExpired as error:
        raise BenchmarkError(
            f"Timed out after {timeout_seconds:.1f}s: {case.case_id}"
        ) from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise BenchmarkError(
            f"Native benchmark failed ({result.returncode}) for {case.case_id}: {detail}"
        )
    native = parse_native_json(result.stdout)
    if native.get("threads_observed") != case.threads:
        raise BenchmarkError(
            f"{case.case_id}: requested {case.threads} threads but observed "
            f"{native.get('threads_observed')}"
        )
    return native


def choose_calls(
    seconds_per_call: float,
    requested_calls: int | None,
    sample_seconds: float,
    min_calls: int,
    max_calls: int,
    repetitions: int,
    case_budget_seconds: float,
) -> int:
    if requested_calls is not None:
        desired = requested_calls
    elif seconds_per_call > 0.0:
        desired = max(1, int(round(sample_seconds / seconds_per_call)))
    else:
        desired = max_calls
    if seconds_per_call > 0.0:
        # Leave 20% for process initialization, warm-up, and timing variance.
        budget_cap = int(
            (0.8 * case_budget_seconds)
            / max(1, repetitions)
            / seconds_per_call
        )
        desired = min(desired, max(1, budget_cap))
    return max(1, min(max_calls, max(min_calls, desired)))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")


def write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for key in fieldnames or []:
        if key not in seen:
            keys.append(key)
            seen.add(key)
    if not rows:
        if keys:
            with path.open("w", newline="", encoding="utf-8") as output:
                csv.DictWriter(output, fieldnames=keys).writeheader()
        else:
            path.write_text("", encoding="utf-8")
        return
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            flat = {
                key: json.dumps(value, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(flat)


def summarize(
    cases: list[Case], samples: list[dict[str, Any]], checksum_tolerance: float
) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_case.setdefault(str(sample["case_id"]), []).append(sample)

    summaries: list[dict[str, Any]] = []
    for case in cases:
        values = by_case.get(case.case_id, [])
        if not values:
            continue
        seconds_per_call = [float(value["seconds_per_call"]) for value in values]
        calls_per_second = [float(value["calls_per_second"]) for value in values]
        checksums = [
            float(value["checksum"]) / max(1, int(value["calls"]))
            for value in values
        ]
        mean_time = statistics.fmean(seconds_per_call)
        stdev_time = (
            statistics.stdev(seconds_per_call) if len(seconds_per_call) > 1 else 0.0
        )
        mean_checksum = statistics.fmean(checksums)
        checksum_spread = max(checksums) - min(checksums)
        checksum_scale = max(1.0, abs(mean_checksum))
        checksum_comparison_applicable = case.mode != "minimization"
        function_evaluations = [
            int(value.get("function_evaluations", 0)) for value in values
        ]
        minimizer_iterations = [
            int(value.get("minimizer_iterations", 0)) for value in values
        ]
        edge_flips = [int(value.get("edge_flips", 0)) for value in values]
        reconnect_cycles = [
            int(value.get("reconnect_cycles", 0)) for value in values
        ]
        initialization_seconds = [
            float(value.get("initialization_seconds", 0.0)) for value in values
        ]
        summaries.append(
            {
                **asdict(case),
                "nodes": int(values[0]["nodes"]),
                "elements": int(values[0]["elements"]),
                "samples": len(values),
                "calls_per_sample": int(values[0]["calls"]),
                "mean_seconds_per_call": mean_time,
                "stdev_seconds_per_call": stdev_time,
                "median_seconds_per_call": statistics.median(seconds_per_call),
                "min_seconds_per_call": min(seconds_per_call),
                "max_seconds_per_call": max(seconds_per_call),
                "coefficient_of_variation_percent": (
                    100.0 * stdev_time / mean_time if mean_time > 0.0 else 0.0
                ),
                "standard_error_seconds_per_call": (
                    stdev_time / math.sqrt(len(seconds_per_call))
                ),
                "mean_calls_per_second": statistics.fmean(calls_per_second),
                "stdev_calls_per_second": (
                    statistics.stdev(calls_per_second)
                    if len(calls_per_second) > 1
                    else 0.0
                ),
                "normalized_checksum_mean": mean_checksum,
                "normalized_checksum_spread": checksum_spread,
                "checksum_comparison_applicable": checksum_comparison_applicable,
                "repeat_checksum_ok": (
                    checksum_spread <= checksum_tolerance * checksum_scale
                    if checksum_comparison_applicable
                    else None
                ),
                "fixture_seeds": sorted(
                    {int(value.get("fixture_seed", 0)) for value in values}
                ),
                "actual_initial_loads": sorted(
                    {
                        float(value.get("initial_load", case.fixture_load))
                        for value in values
                    }
                ),
                "force_scratch_bytes_estimate": int(
                    values[0]["force_scratch_bytes_estimate"]
                ),
                "mean_initialization_seconds": statistics.fmean(
                    initialization_seconds
                ),
                "mean_function_evaluations": statistics.fmean(
                    function_evaluations
                ),
                "stdev_function_evaluations": (
                    statistics.stdev(function_evaluations)
                    if len(function_evaluations) > 1
                    else 0.0
                ),
                "mean_minimizer_iterations": statistics.fmean(
                    minimizer_iterations
                ),
                "mean_edge_flips": statistics.fmean(edge_flips),
                "stdev_edge_flips": (
                    statistics.stdev(edge_flips) if len(edge_flips) > 1 else 0.0
                ),
                "edge_flip_trigger_rate": sum(value > 0 for value in edge_flips)
                / len(edge_flips),
                "mean_reconnect_cycles": statistics.fmean(reconnect_cycles),
                "termination_types": sorted(
                    {int(value.get("termination_type", 0)) for value in values}
                ),
            }
        )

    for experiment in ("strong", "weak"):
        modes = {row["mode"] for row in summaries if row["experiment"] == experiment}
        for mode in modes:
            group = [
                row
                for row in summaries
                if row["experiment"] == experiment and row["mode"] == mode
            ]
            baseline = next((row for row in group if row["threads"] == 1), None)
            if baseline is None:
                continue
            baseline_time = float(baseline["mean_seconds_per_call"])
            baseline_checksum = float(baseline["normalized_checksum_mean"])
            for row in group:
                current_time = float(row["mean_seconds_per_call"])
                if experiment == "strong":
                    speedup = baseline_time / current_time
                    row["speedup"] = speedup
                    row["parallel_efficiency"] = speedup / int(row["threads"])
                    checksum_scale = max(1.0, abs(baseline_checksum))
                    relative_difference = abs(
                        float(row["normalized_checksum_mean"]) - baseline_checksum
                    ) / checksum_scale
                    row["thread_checksum_relative_difference"] = relative_difference
                    row["thread_checksum_ok"] = relative_difference <= checksum_tolerance
                else:
                    row["weak_scaling_efficiency"] = baseline_time / current_time

    workload_groups = {
        (
            row["history"],
            row["fixture_load"],
            row["reconnection"],
            row["rows"],
            row["cols"],
        )
        for row in summaries
        if row["experiment"] == "workload"
    }
    for history, fixture_load, reconnection, rows, cols in workload_groups:
        group = [
            row
            for row in summaries
            if row["experiment"] == "workload"
            and row["history"] == history
            and row["fixture_load"] == fixture_load
            and row["reconnection"] == reconnection
            and row["rows"] == rows
            and row["cols"] == cols
        ]
        baseline = next((row for row in group if row["threads"] == 1), None)
        if baseline is None:
            continue
        baseline_time = float(baseline["mean_seconds_per_call"])
        for row in group:
            speedup = baseline_time / float(row["mean_seconds_per_call"])
            row["speedup"] = speedup
            row["parallel_efficiency"] = speedup / int(row["threads"])
    return summaries


def recommendation_confidence(
    row: dict[str, Any], affinity_verified: bool
) -> str:
    samples = int(row["samples"])
    cv = float(row["coefficient_of_variation_percent"])
    if not affinity_verified:
        return "exploratory"
    if samples >= 5 and cv <= 5.0:
        return "high"
    if samples >= 3 and cv <= 10.0:
        return "medium"
    return "exploratory"


def choose_recommendation(
    candidates: list[dict[str, Any]], affinity_verified: bool
) -> dict[str, Any]:
    fastest = min(candidates, key=lambda row: float(row["mean_seconds_per_call"]))
    fastest_mean = float(fastest["mean_seconds_per_call"])
    fastest_se = float(fastest["standard_error_seconds_per_call"])
    statistically_tied: list[dict[str, Any]] = []
    for row in candidates:
        mean = float(row["mean_seconds_per_call"])
        combined_se = math.hypot(
            float(row["standard_error_seconds_per_call"]), fastest_se
        )
        if mean <= 1.10 * fastest_mean and mean - fastest_mean <= 1.96 * combined_se:
            statistically_tied.append(row)
    chosen = min(
        statistically_tied or [fastest],
        key=lambda row: (
            int(row["threads"]),
            0 if row["affinity_policy"] == "close" else 1,
            float(row["mean_seconds_per_call"]),
        ),
    )
    return {
        "scenario": chosen["scenario"],
        "history": chosen["history"],
        "fixture_load": chosen["fixture_load"],
        "actual_initial_loads": chosen["actual_initial_loads"],
        "reconnection": chosen["reconnection"],
        "rows": int(chosen["rows"]),
        "cols": int(chosen["cols"]),
        "threads": int(chosen["threads"]),
        "affinity_policy": chosen["affinity_policy"],
        "mean_seconds_per_minimization": float(chosen["mean_seconds_per_call"]),
        "stdev_seconds_per_minimization": float(chosen["stdev_seconds_per_call"]),
        "mean_function_evaluations": float(chosen["mean_function_evaluations"]),
        "mean_edge_flips": float(chosen["mean_edge_flips"]),
        "edge_flip_trigger_rate": float(chosen["edge_flip_trigger_rate"]),
        "samples": int(chosen["samples"]),
        "fixture_seeds": chosen["fixture_seeds"],
        "confidence": recommendation_confidence(chosen, affinity_verified),
        "fastest_measured_threads": int(fastest["threads"]),
        "fastest_measured_affinity": fastest["affinity_policy"],
        "fastest_mean_seconds": fastest_mean,
        "selection_rule": (
            "Prefer fewer threads among candidates within 10% of the fastest "
            "whose difference is no larger than 1.96 combined standard errors."
        ),
    }


def create_recommendations(
    output_dir: Path, manifest: dict[str, Any], summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    workloads = [row for row in summaries if row["experiment"] == "workload"]
    groups = sorted(
        {
            (
                row["history"],
                row["fixture_load"],
                row["reconnection"],
                row["rows"],
                row["cols"],
            )
            for row in workloads
        }
    )
    recommendations: list[dict[str, Any]] = []
    invalid_groups: list[dict[str, Any]] = []
    for history, fixture_load, reconnection, rows, cols in groups:
        candidates = [
            row
            for row in workloads
            if row["history"] == history
            and row["fixture_load"] == fixture_load
            and row["reconnection"] == reconnection
            and row["rows"] == rows
            and row["cols"] == cols
            and row["repeat_checksum_ok"] is not False
        ]
        if candidates:
            recommendations.append(
                choose_recommendation(
                    candidates, bool(manifest.get("affinity_verified", False))
                )
            )

    affinity_verified = bool(manifest.get("affinity_verified", False))
    recommendation_dir = output_dir / "recommended_configs"
    recommendation_dir.mkdir(exist_ok=True)
    physical_place_cpus = [
        int(cpu)
        for cpu in manifest.get("system", {}).get("physical_place_cpu_ids", [])
    ]
    omp_places = (
        ",".join(f"{{{cpu}}}" for cpu in physical_place_cpus)
        if physical_place_cpus
        else "cores"
    )
    profile_files: list[dict[str, str]] = []
    profile_keys = sorted(
        {
            (row["history"], row["fixture_load"], row["reconnection"])
            for row in recommendations
        }
    )
    profile_selections: list[tuple[str, dict[str, Any]]] = []
    for history, fixture_load, reconnection in profile_keys:
        profile_rows = [
            row
            for row in recommendations
            if row["history"] == history
            and row["fixture_load"] == fixture_load
            and row["reconnection"] == reconnection
        ]
        selected = max(profile_rows, key=lambda row: int(row["rows"]) * int(row["cols"]))
        profile_selections.append(
            (
                f"{history}_load_{load_label(float(fixture_load)).replace('.', 'p')}",
                selected,
            )
        )

    for stem, selected in profile_selections:
        env_path = recommendation_dir / f"{stem}.env"
        config_path = recommendation_dir / f"{stem}.conf.fragment"
        env_path.write_text(
            "# Hardware-specific candidate generated by the MTS2D benchmark.\n"
            f"# Confidence: {selected['confidence']}; "
            f"affinity_verified={affinity_verified}\n"
            f"export OMP_NUM_THREADS={selected['threads']}\n"
            f"export OMP_PLACES='{omp_places}'\n"
            f"export OMP_PROC_BIND={selected['affinity_policy']}\n"
            "export OMP_DYNAMIC=FALSE\n"
            "export OMP_MAX_ACTIVE_LEVELS=1\n",
            encoding="utf-8",
        )
        config_path.write_text(
            "# Merge this fragment into the simulation config; it is not standalone.\n"
            f"# Confidence: {selected['confidence']}; "
            f"affinity_verified={affinity_verified}\n"
            f"# Profile: {stem}, reconnection={selected['reconnection']}, "
            f"validated at {selected['rows']}x{selected['cols']}\n"
            f"nrThreads = {selected['threads']}\n",
            encoding="utf-8",
        )
        profile_files.append(
            {
                "profile": stem,
                "environment": str(env_path.relative_to(output_dir)),
                "simulation_config_fragment": str(config_path.relative_to(output_dir)),
            }
        )

    result = {
        "schema_version": 1,
        "host": manifest["system"]["hostname"],
        "generated_from_executable_sha256": manifest["executable_sha256"],
        "fixture_kind": "dump_replay",
        "affinity_verified": bool(manifest.get("affinity_verified", False)),
        "openmp_physical_place_cpu_ids": physical_place_cpus,
        "scope": (
            "Candidates are specific to the measured sizes, loads, and simulation-"
            "history dumps. They are not universal settings for every regime."
        ),
        "design": (
            "Each timed minimization starts from a relaxed native dump, applies one "
            "serialized load increment, and preserves whether that state evolved with "
            "no reconnection or edge flipping."
        ),
        "recommendations_by_workload_and_size": recommendations,
        "invalid_groups": invalid_groups,
        "profile_files": profile_files,
    }
    write_json(output_dir / "recommendations.json", result)
    return result


def plot_groups(
    summaries: list[dict[str, Any]],
) -> list[tuple[str, str, str, list[dict[str, Any]]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in summaries:
        if row["experiment"] == "workload":
            key = (
                "workload",
                row["history"],
                row["fixture_load"],
                row["reconnection"],
                row["affinity_policy"],
                row["rows"],
                row["cols"],
            )
        else:
            key = (row["experiment"], row["mode"])
        grouped.setdefault(key, []).append(row)

    result: list[tuple[str, str, str, list[dict[str, Any]]]] = []
    for key, rows in sorted(grouped.items()):
        experiment = str(key[0])
        if experiment == "workload":
            _, history, fixture_load, reconnection, affinity, mesh_rows, mesh_cols = key
            slug = (
                f"workload_{history}_load_"
                f"{load_label(float(fixture_load)).replace('.', 'p')}_"
                f"{reconnection}_s{mesh_rows}x{mesh_cols}_"
                f"a{affinity}"
            )
            label = (
                f"workload — {history}, load {fixture_load:g}, {reconnection}, "
                f"{mesh_rows}×{mesh_cols}, {affinity}"
            )
        else:
            slug = f"{key[0]}_{key[1]}"
            label = f"{key[0]} scaling — {key[1]}"
        sort_key = "elements" if experiment == "system" else "threads"
        result.append(
            (slug, label, experiment, sorted(rows, key=lambda row: row[sort_key]))
        )
    return result


def make_plots(
    output_dir: Path, summaries: list[dict[str, Any]], title_suffix: str
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return make_svg_plots(output_dir, summaries, title_suffix)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    created: list[str] = []
    for slug, label, experiment, group in plot_groups(summaries):
        if not group:
            continue
        x_key = "elements" if experiment == "system" else "threads"
        x = [float(row[x_key]) for row in group]
        y = [float(row["mean_seconds_per_call"]) for row in group]
        yerr = [float(row["stdev_seconds_per_call"]) for row in group]
        figure, axis = plt.subplots(figsize=(7.2, 4.5))
        axis.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=1.5)
        axis.set_xlabel("Elements" if experiment == "system" else "OpenMP threads")
        sample_label = "minimization" if experiment == "workload" else "force evaluation"
        axis.set_ylabel(f"Seconds per {sample_label} (mean ± SD)")
        axis.set_title(f"MTS2D {label}\n{title_suffix}")
        axis.grid(True, alpha=0.25)
        if experiment == "system":
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
        elif all(value > 0 for value in x):
            axis.set_xscale("log", base=2)
        figure.tight_layout()
        path = plots_dir / f"{slug}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        created.append(str(path.relative_to(output_dir)))

        metric = (
            "parallel_efficiency"
            if experiment in {"strong", "workload"}
            else "weak_scaling_efficiency"
            if experiment == "weak"
            else None
        )
        if metric is not None and all(metric in row for row in group):
            figure, axis = plt.subplots(figsize=(7.2, 4.5))
            axis.plot(x, [float(row[metric]) for row in group], marker="o")
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
            axis.set_xlabel("OpenMP threads")
            axis.set_ylabel(metric.replace("_", " ").title())
            axis.set_title(f"MTS2D efficiency — {label}\n{title_suffix}")
            axis.set_xscale("log", base=2)
            axis.grid(True, alpha=0.25)
            figure.tight_layout()
            path = plots_dir / f"{slug}_efficiency.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            created.append(str(path.relative_to(output_dir)))
    return created


def write_svg_plot(
    path: Path,
    x: list[float],
    y: list[float],
    yerr: list[float],
    x_label: str,
    y_label: str,
    title: str,
    log_x: bool,
    log_y: bool,
    ideal_y: float | None = None,
) -> None:
    """Dependency-free SVG fallback for compute nodes without matplotlib."""

    width, height = 820, 500
    left, right, top, bottom = 100, 35, 60, 75
    plot_width = width - left - right
    plot_height = height - top - bottom

    def transform_x(value: float) -> float:
        return math.log2(value) if log_x else value

    def transform_y(value: float) -> float:
        return math.log10(max(value, 1e-300)) if log_y else value

    tx = [transform_x(value) for value in x]
    lower_y = [max(1e-300, value - error) for value, error in zip(y, yerr)]
    upper_y = [value + error for value, error in zip(y, yerr)]
    ty_all = [transform_y(value) for value in lower_y + upper_y]
    if ideal_y is not None:
        ty_all.append(transform_y(ideal_y))
    x_min, x_max = min(tx), max(tx)
    y_min, y_max = min(ty_all), max(ty_all)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        padding = max(0.5, abs(y_min) * 0.1)
        y_min -= padding
        y_max += padding
    y_padding = 0.08 * (y_max - y_min)
    y_min -= y_padding
    y_max += y_padding

    def sx(value: float) -> float:
        return left + (transform_x(value) - x_min) * plot_width / (x_max - x_min)

    def sy(value: float) -> float:
        return top + (y_max - transform_y(value)) * plot_height / (y_max - y_min)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}'
        '.axis{stroke:#333;stroke-width:1.5}.series{stroke:#1769aa;stroke-width:2;fill:none}'
        '.error{stroke:#1769aa;stroke-width:1.4}.point{fill:#1769aa}</style>',
        f'<text x="{width / 2}" y="27" text-anchor="middle" font-size="17">'
        f'{html.escape(title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x_position = left + fraction * plot_width
        y_position = top + fraction * plot_height
        x_transformed = x_min + fraction * (x_max - x_min)
        y_transformed = y_max - fraction * (y_max - y_min)
        x_value = 2**x_transformed if log_x else x_transformed
        y_value = 10**y_transformed if log_y else y_transformed
        svg.extend(
            [
                f'<line class="grid" x1="{x_position:.2f}" y1="{top}" '
                f'x2="{x_position:.2f}" y2="{top + plot_height}"/>',
                f'<text x="{x_position:.2f}" y="{top + plot_height + 24}" '
                f'text-anchor="middle" font-size="11">{x_value:.3g}</text>',
                f'<line class="grid" x1="{left}" y1="{y_position:.2f}" '
                f'x2="{left + plot_width}" y2="{y_position:.2f}"/>',
                f'<text x="{left - 10}" y="{y_position + 4:.2f}" '
                f'text-anchor="end" font-size="11">{y_value:.3g}</text>',
            ]
        )
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + plot_height}"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 20}" '
            f'text-anchor="middle" font-size="13">{html.escape(x_label)}</text>',
            f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" '
            f'font-size="13" transform="rotate(-90 22 {top + plot_height / 2})">'
            f'{html.escape(y_label)}</text>',
        ]
    )
    if ideal_y is not None:
        ideal_position = sy(ideal_y)
        svg.append(
            f'<line x1="{left}" y1="{ideal_position:.2f}" x2="{left + plot_width}" '
            f'y2="{ideal_position:.2f}" stroke="#555" stroke-dasharray="6 5"/>'
        )
    points = " ".join(f"{sx(xv):.2f},{sy(yv):.2f}" for xv, yv in zip(x, y))
    svg.append(f'<polyline class="series" points="{points}"/>')
    for xv, yv, error in zip(x, y, yerr):
        x_position = sx(xv)
        low = sy(max(1e-300, yv - error))
        high = sy(yv + error)
        svg.extend(
            [
                f'<line class="error" x1="{x_position:.2f}" y1="{low:.2f}" '
                f'x2="{x_position:.2f}" y2="{high:.2f}"/>',
                f'<line class="error" x1="{x_position - 5:.2f}" y1="{low:.2f}" '
                f'x2="{x_position + 5:.2f}" y2="{low:.2f}"/>',
                f'<line class="error" x1="{x_position - 5:.2f}" y1="{high:.2f}" '
                f'x2="{x_position + 5:.2f}" y2="{high:.2f}"/>',
                f'<circle class="point" cx="{x_position:.2f}" cy="{sy(yv):.2f}" r="4"/>',
            ]
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def make_svg_plots(
    output_dir: Path, summaries: list[dict[str, Any]], title_suffix: str
) -> list[str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    created: list[str] = []
    for slug, label, experiment, group in plot_groups(summaries):
        if not group:
            continue
        x_key = "elements" if experiment == "system" else "threads"
        x = [float(row[x_key]) for row in group]
        y = [float(row["mean_seconds_per_call"]) for row in group]
        yerr = [float(row["stdev_seconds_per_call"]) for row in group]
        path = plots_dir / f"{slug}.svg"
        sample_label = "minimization" if experiment == "workload" else "force evaluation"
        write_svg_plot(
            path,
            x,
            y,
            yerr,
            "Elements" if experiment == "system" else "OpenMP threads",
            f"Seconds per {sample_label} (mean ± SD)",
            f"MTS2D {label} — {title_suffix}",
            log_x=True,
            log_y=experiment == "system",
        )
        created.append(str(path.relative_to(output_dir)))

        metric = (
            "parallel_efficiency"
            if experiment in {"strong", "workload"}
            else "weak_scaling_efficiency"
            if experiment == "weak"
            else None
        )
        if metric is not None and all(metric in row for row in group):
            path = plots_dir / f"{slug}_efficiency.svg"
            write_svg_plot(
                path,
                x,
                [float(row[metric]) for row in group],
                [0.0] * len(group),
                "OpenMP threads",
                metric.replace("_", " ").title(),
                f"MTS2D efficiency — {label} — {title_suffix}",
                log_x=True,
                log_y=False,
                ideal_y=1.0,
            )
            created.append(str(path.relative_to(output_dir)))
    return created


def scheduler_environment() -> dict[str, str]:
    prefixes = ("SLURM_", "PBS_", "LSB_", "COBALT_")
    names = {
        "OMP_PLACES",
        "OMP_PROC_BIND",
        "OMP_WAIT_POLICY",
        "KMP_AFFINITY",
        "GOMP_CPU_AFFINITY",
    }
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(prefixes) or key in names
    }


def environment_warnings(
    affinity: str,
    cpus: list[int],
    topology: dict[str, Any],
    max_threads: int,
    physical_place_cpus: list[int],
) -> list[str]:
    warnings: list[str] = []
    if affinity == "none" and max_threads > 1:
        warnings.append("Thread binding is disabled; scaling variance may be higher.")
    if platform.system() == "Linux" and max_threads >= 32:
        if not any(
            name in os.environ
            for name in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "COBALT_JOBID")
        ):
            warnings.append(
                "No batch-scheduler allocation was detected on a large Linux node. "
                "For reportable results, request exclusive cores and launch the runner "
                "inside that allocation."
            )
    if "SLURM_JOB_ID" in os.environ and "SLURM_CPUS_PER_TASK" not in os.environ:
        warnings.append(
            "Slurm is active but SLURM_CPUS_PER_TASK is absent; verify that one task "
            "owns all CPUs used by the benchmark."
        )
    if topology.get("sockets", 1) > 1:
        warnings.append(
            "Multiple CPU sockets are available. The explicit physical-core place "
            "list groups close teams by NUMA locality and spread teams across the "
            "machine, but memory placement remains launcher- and first-touch-specific."
        )
    if platform.system() == "Linux" and len(physical_place_cpus) < max_threads:
        warnings.append(
            "The topology-aware OpenMP place list contains fewer physical cores than "
            "the largest requested team."
        )
    if len(cpus) < max_threads:
        warnings.append("Requested threads exceed the process CPU affinity mask.")
    return warnings


def create_report(
    output_dir: Path,
    manifest: dict[str, Any],
    summaries: list[dict[str, Any]],
    plots: list[str],
    recommendations: dict[str, Any],
) -> None:
    lines = [
        "# MTS2D benchmark report",
        "",
        f"- Status: {manifest['status']}",
        f"- Host: {manifest['system']['hostname']}",
        f"- Preset: {manifest['configuration']['preset']}",
        "- Affinity policies: "
        + ", ".join(manifest["configuration"]["affinity_policies"]),
        f"- Affinity verified: {manifest.get('affinity_verified', False)}",
        f"- Successful samples: {manifest['successful_samples']}",
        f"- Elapsed benchmark time: {manifest['elapsed_seconds']:.3f} s",
        "",
    ]
    if manifest["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in manifest["warnings"])
        lines.append("")
    configuration = manifest["configuration"]
    if "workload" in configuration["experiments"]:
        lines.extend(
            [
                "## Initial-condition replay workloads",
                "",
                "- Catalog: `" + configuration["initial_conditions_dir"] + "`",
                "- Requested loads: "
                + ", ".join(str(value) for value in configuration["workload_loads"]),
                "- Workload thread grid: "
                + ", ".join(
                    str(value) for value in configuration["workload_threads"]
                ),
                "- Load folders are nominal bins; stored loads must be within "
                f"{configuration['replay_load_tolerance']:g} of the folder value.",
                "- Histories: `noReconnecting` and `edgeFlipping`",
                "- Each sample loads a relaxed dump, applies that dump's serialized "
                "load increment, and times the following minimization.",
                "- Missing load-0.15 states may be generated using initial noise "
                f"{configuration['initial_noise']:g}; existing files are never replaced.",
                "- Generated load-0.15 states use quenched-disorder SD "
                f"{configuration['workload_disorder']:g} and load increment "
                f"{configuration['generated_load_increment']:g}.",
                "- Missing states at other loads are skipped; no affine fallback is used.",
                "- Initial-condition seed: "
                f"{configuration['fixture_seed']} (repetition stride "
                f"{configuration['fixture_seed_stride']})",
                "- Available-seed fallback: "
                + (
                    "enabled (the lowest valid catalog seed is reused when the "
                    "requested fixed seed is absent)"
                    if configuration["allow_available_fixture_fallback"]
                    else "disabled"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Result files",
            "",
            "- `samples.jsonl`: append-only raw native results",
            "- `samples.csv`: raw results in tabular form",
            "- `summary.csv` and `summary.json`: means, sample standard deviations, "
            "medians, efficiency, and correctness checks",
            "- `manifest.json`: build, machine, scheduler, topology, and budget metadata",
            "- `initial_condition_plan.json`: catalog paths that existed or were skipped",
            "- `initial_condition_preparation.json`: separately timed load-0.15 generation",
            "- `first_minimization_benchmarks.json`: first-noisy timings across thread counts",
            "- `recommendations.json`: conservative candidates by workload and size",
            "- `recommended_configs/`: OpenMP environment and simulation-config fragments",
        ]
    )
    if plots:
        lines.append("- `plots/`: mean timings with standard-deviation error bars")
    recommended_rows = recommendations["recommendations_by_workload_and_size"]
    if recommended_rows:
        lines.extend(
            [
                "",
                "## Hardware-specific configuration candidates",
                "",
                "These are separate profiles, not one universal setting. The files use "
                "the largest tested size in each workload profile; consult "
                "`recommendations.json` for every measured size.",
                "",
                "| History | Load | Reconnection | Size | Threads | Binding | Mean ± SD | Confidence |",
                "|---|---:|---|---:|---:|---|---:|---|",
            ]
        )
        for row in recommended_rows:
            lines.append(
                "| {history} | {fixture_load:g} | {reconnection} | {rows}×{cols} | {threads} | "
                "{affinity_policy} | {mean:.6g} ± {stdev:.3g} s | {confidence} |".format(
                    mean=row["mean_seconds_per_minimization"],
                    stdev=row["stdev_seconds_per_minimization"],
                    **row,
                )
            )
    lines.extend(["", "## Summary", ""])
    lines.append(
        "| Experiment | Mode | History | Load | Reconnect | Binding | Rows×cols | "
        "Threads | Samples | Mean s/sample | SD | CV % |"
    )
    lines.append("|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        lines.append(
            "| {experiment} | {mode} | {history} | {fixture_load:g} | {reconnection} | "
            "{affinity_policy} | "
            "{rows}×{cols} | {threads} | {samples} | "
            "{mean:.6g} | {stdev:.3g} | {cv:.2f} |".format(
                mean=row["mean_seconds_per_call"],
                stdev=row["stdev_seconds_per_call"],
                cv=row["coefficient_of_variation_percent"],
                **row,
            )
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and run portable MTS2D force/minimization benchmarks "
            "sequentially, with affinity, budgets, repeated samples, "
            "statistics, recommendations, and plots."
        )
    )
    parser.add_argument("--mts2d-source", type=Path, default=REPO_ROOT)
    parser.add_argument("--preset", choices=PRESETS, default="quick")
    parser.add_argument(
        "--experiments", default=None, help="system,strong,weak,workload"
    )
    parser.add_argument(
        "--modes", default=None, help="force-kernel,force-evaluation"
    )
    parser.add_argument("--sizes", default=None, help="Square system sizes, e.g. 32,64,128")
    parser.add_argument(
        "--workload-sizes",
        default=None,
        help="Square sizes for dump-replay minimization workloads",
    )
    parser.add_argument(
        "--workload-loads",
        default=None,
        help="Initial-condition loads to replay, e.g. 0.15,0.7",
    )
    parser.add_argument("--strong-size", type=int, default=None)
    parser.add_argument("--weak-base-nodes", type=int, default=None)
    parser.add_argument(
        "--threads",
        default="auto",
        help="auto, a comma list, or an inclusive range such as 1-8",
    )
    parser.add_argument(
        "--workload-threads",
        default=None,
        help=(
            "Thread counts for expensive minimization replays: inherit the force "
            "grid by default, use 'auto', 'screen', a comma list, or a range"
        ),
    )
    parser.add_argument(
        "--large-workload-threads",
        default=None,
        help=(
            "Thread grid at or above --large-workload-size-threshold; use "
            "'sparse', 'auto', a comma list, or a range"
        ),
    )
    parser.add_argument(
        "--large-workload-size-threshold",
        type=int,
        default=None,
        help="Apply the reduced large-workload plan from this L onward",
    )
    parser.add_argument(
        "--large-workload-repetitions",
        type=int,
        default=None,
        help="Maximum repetitions for each large minimization-replay case",
    )
    parser.add_argument("--fixed-threads", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument(
        "--calls",
        default="auto",
        help=(
            "auto or a requested calls-per-sample target; safety budgets may "
            "reduce the requested value"
        ),
    )
    parser.add_argument("--min-calls", type=int, default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--warmup-calls", type=int, default=None)
    parser.add_argument("--sample-seconds", type=float, default=None)
    parser.add_argument("--case-budget-minutes", type=float, default=None)
    parser.add_argument("--total-budget-hours", type=float, default=None)
    parser.add_argument(
        "--affinity",
        choices=("auto", "close", "spread", "none"),
        default="close",
        help=(
            "close is the default; auto enables the special close-versus-spread "
            "comparison, while another explicit value uses only that policy"
        ),
    )
    parser.add_argument("--shear", type=float, default=0.15)
    parser.add_argument("--perturbation", type=float, default=1e-3)
    parser.add_argument("--workload-disorder", type=float, default=0.0)
    parser.add_argument("--initial-noise", type=float, default=0.05)
    parser.add_argument(
        "--preconditioner", choices=("exactTangent", "diagonal", "none"),
        default="exactTangent", help="Curvature preconditioning for minimizations",
    )
    parser.add_argument(
        "--replay-load-tolerance",
        type=float,
        default=0.005,
        help="Allowed difference between a nominal load folder and stored dump load",
    )
    parser.add_argument("--generated-load-increment", type=float, default=1e-5)
    parser.add_argument(
        "--initial-conditions-dir",
        type=Path,
        default=DEFAULT_INITIAL_CONDITIONS_DIR,
    )
    parser.add_argument(
        "--initial-condition-budget-minutes",
        type=float,
        default=20.0,
        help="Maximum preparation time for each missing load-0.15 dump",
    )
    parser.add_argument(
        "--large-initial-condition-budget-minutes",
        type=float,
        default=None,
        help=(
            "Preparation and first-minimization timeout at or above the large "
            "workload threshold"
        ),
    )
    parser.add_argument(
        "--no-first-minimization-thread-sweep",
        action="store_true",
        help="Prepare dumps but omit the separate first-noisy thread sweep",
    )
    parser.add_argument(
        "--first-minimization-only",
        action="store_true",
        help="Run the first-noisy minimization sweep, then skip ordinary samples",
    )
    parser.add_argument(
        "--first-minimization-histories",
        default=None,
        help="Restrict the first-noisy sweep to noReconnecting,edgeFlipping",
    )
    parser.add_argument(
        "--no-generate-load-015",
        action="store_true",
        help="Skip missing load-0.15 dumps instead of preparing them",
    )
    parser.add_argument(
        "--fixture-seed",
        type=int,
        default=0,
        help="Deterministic seed for workload fixtures",
    )
    parser.add_argument(
        "--fixture-seed-stride",
        type=int,
        default=0,
        help=(
            "Seed increment between repetitions (default 0 reuses one controlled "
            "fixture; use 1 only after screening the fixture seeds)"
        ),
    )
    parser.add_argument(
        "--allow-available-fixture-fallback",
        action="store_true",
        help=(
            "With a fixed fixture seed, reuse the lowest valid catalog seed for "
            "a workload when the requested seed is absent"
        ),
    )
    parser.add_argument("--checksum-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-memory-fraction", type=float, default=0.5)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument(
        "--keep-noinline",
        action="store_true",
        help="Keep profiler-friendly MTS_NOINLINE annotations in the benchmark build.",
    )
    parser.add_argument(
        "--build-jobs", type=int, default=max(1, min(16, len(allowed_cpus())))
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configured_value(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    return PRESETS[args.preset][name] if value is None else value


def main() -> int:
    global REPO_ROOT
    args = parse_args()
    REPO_ROOT = args.mts2d_source.resolve()
    if not (REPO_ROOT / "src" / "Simulation" / "simulation.h").is_file():
        raise FileNotFoundError(f"Not an MTS2D source directory: {REPO_ROOT}")
    preset = PRESETS[args.preset]
    experiments = (
        preset["experiments"]
        if args.experiments is None
        else parse_names(
            args.experiments,
            {"system", "strong", "weak", "workload"},
            "experiments",
        )
    )
    modes = (
        preset["modes"]
        if args.modes is None
        else parse_names(
            args.modes,
            {"force-kernel", "force-evaluation"},
            "benchmark modes",
        )
    )
    sizes = preset["sizes"] if args.sizes is None else parse_int_list(args.sizes)
    workload_sizes = (
        preset["workload_sizes"]
        if args.workload_sizes is None
        else parse_int_list(args.workload_sizes)
    )
    workload_loads = (
        preset["workload_loads"]
        if args.workload_loads is None
        else parse_float_list(args.workload_loads)
    )
    strong_size = configured_value(args, "strong_size")
    weak_base_nodes = configured_value(args, "weak_base_nodes")
    repetitions = configured_value(args, "repetitions")
    sample_seconds = configured_value(args, "sample_seconds")
    max_calls = configured_value(args, "max_calls")
    min_calls = configured_value(args, "min_calls")
    warmup_calls = configured_value(args, "warmup_calls")
    case_budget_seconds = 60.0 * configured_value(args, "case_budget_minutes")
    total_budget_seconds = 3600.0 * configured_value(args, "total_budget_hours")
    large_initial_condition_budget_minutes = (
        preset.get(
            "large_initial_condition_budget_minutes",
            args.initial_condition_budget_minutes,
        )
        if args.large_initial_condition_budget_minutes is None
        else args.large_initial_condition_budget_minutes
    )
    requested_calls = None if args.calls == "auto" else int(args.calls)

    if repetitions < 1 or min_calls < 1 or max_calls < min_calls:
        raise SystemExit("Invalid repetition or force-call limits.")
    if args.fixture_seed_stride < 0:
        raise SystemExit("--fixture-seed-stride must be non-negative.")
    if args.allow_available_fixture_fallback and args.fixture_seed_stride != 0:
        raise SystemExit(
            "--allow-available-fixture-fallback requires --fixture-seed-stride 0."
        )
    if requested_calls is not None and requested_calls < 1:
        raise SystemExit("--calls must be positive or 'auto'.")
    if (
        any(size < 2 for size in sizes + workload_sizes)
        or strong_size < 2
        or weak_base_nodes < 4
        or args.fixed_threads < 1
    ):
        raise SystemExit("Invalid mesh size or fixed thread count.")
    if sample_seconds <= 0.0 or case_budget_seconds <= 0.0 or total_budget_seconds <= 0.0:
        raise SystemExit("Timing targets and budgets must be positive.")
    if not 0.0 < args.max_memory_fraction <= 1.0:
        raise SystemExit("--max-memory-fraction must be in (0, 1].")
    if min(
        args.workload_disorder,
        args.initial_noise,
        args.replay_load_tolerance,
    ) < 0.0:
        raise SystemExit(
            "Workload disorder, initial noise, and replay load tolerance must "
            "be non-negative."
        )
    if args.generated_load_increment <= 0.0:
        raise SystemExit("--generated-load-increment must be positive.")
    if args.initial_condition_budget_minutes <= 0.0:
        raise SystemExit("--initial-condition-budget-minutes must be positive.")
    if large_initial_condition_budget_minutes <= 0.0:
        raise SystemExit(
            "--large-initial-condition-budget-minutes must be positive."
        )
    if args.first_minimization_only and args.no_first_minimization_thread_sweep:
        raise SystemExit(
            "--first-minimization-only requires the first-minimization thread sweep."
        )
    first_minimization_histories = (
        ["noReconnecting", "edgeFlipping"]
        if args.first_minimization_histories is None
        else parse_names(
            args.first_minimization_histories,
            {"noReconnecting", "edgeFlipping"},
            "first-minimization histories",
        )
    )

    cpus = allowed_cpus()
    topology = linux_topology(cpus)
    physical_place_cpus = ordered_physical_place_cpus(topology)
    available_physical_threads = int(
        topology.get("physical_cores_available", len(cpus))
    )
    if args.threads == "auto":
        auto_max_threads = available_physical_threads
        if preset.get("thread_cap") is not None:
            auto_max_threads = min(auto_max_threads, int(preset["thread_cap"]))
        threads = standard_thread_counts(auto_max_threads, topology)
    else:
        threads = parse_int_list(args.threads)
        if max(threads) > available_physical_threads:
            raise SystemExit(
                f"Requested {max(threads)} threads, but only "
                f"{available_physical_threads} physical "
                "cores are available in this process CPU set."
            )
    workload_thread_spec = (
        preset.get("workload_threads")
        if args.workload_threads is None
        else args.workload_threads
    )
    if workload_thread_spec is None:
        workload_threads = threads
    elif workload_thread_spec == "auto":
        workload_threads = standard_thread_counts(
            available_physical_threads, topology
        )
    elif workload_thread_spec == "screen":
        workload_threads = screening_thread_counts(
            available_physical_threads, topology
        )
    else:
        workload_threads = parse_int_list(workload_thread_spec)
    if max(workload_threads) > available_physical_threads:
        raise SystemExit(
            f"Requested {max(workload_threads)} workload threads, but only "
            f"{available_physical_threads} physical cores are available in this "
            "process CPU set."
        )
    large_workload_size_threshold = (
        preset.get("large_workload_size_threshold")
        if args.large_workload_size_threshold is None
        else args.large_workload_size_threshold
    )
    large_workload_repetitions = (
        preset.get("large_workload_repetitions")
        if args.large_workload_repetitions is None
        else args.large_workload_repetitions
    )
    large_thread_spec = (
        preset.get("large_workload_threads")
        if args.large_workload_threads is None
        else args.large_workload_threads
    )
    if large_workload_size_threshold is None or large_thread_spec is None:
        large_workload_threads = workload_threads
    elif large_thread_spec == "sparse":
        large_workload_threads = sparse_large_thread_counts(
            available_physical_threads
        )
    elif large_thread_spec == "auto":
        large_workload_threads = standard_thread_counts(
            available_physical_threads, topology
        )
    else:
        large_workload_threads = parse_int_list(large_thread_spec)
    if max(large_workload_threads) > available_physical_threads:
        raise SystemExit(
            f"Requested {max(large_workload_threads)} large-workload threads, "
            f"but only {available_physical_threads} physical cores are available."
        )
    if large_workload_size_threshold is not None and large_workload_size_threshold < 2:
        raise SystemExit("--large-workload-size-threshold must be at least 2.")
    if large_workload_repetitions is not None and large_workload_repetitions < 1:
        raise SystemExit("--large-workload-repetitions must be positive.")
    if args.fixed_threads > available_physical_threads:
        raise SystemExit("--fixed-threads exceeds the available physical cores.")
    max_case_threads = max(
        max(threads),
        max(workload_threads),
        max(large_workload_threads),
        args.fixed_threads,
    )
    if args.affinity == "auto":
        affinity_policies = (
            ["close", "spread"] if "workload" in experiments else ["close"]
        )
    else:
        affinity_policies = [args.affinity]

    cases = build_cases(
        experiments,
        modes,
        sizes,
        workload_sizes,
        workload_loads,
        strong_size,
        weak_base_nodes,
        threads,
        workload_threads,
        large_workload_threads,
        large_workload_size_threshold,
        large_workload_repetitions,
        args.fixed_threads,
        affinity_policies,
    )
    memory_available = available_memory_bytes()
    memory_limit = (
        int(memory_available * args.max_memory_fraction)
        if memory_available is not None
        else None
    )
    skipped: list[dict[str, Any]] = []
    runnable: list[Case] = []
    for case in cases:
        scratch = case.threads * case.rows * case.cols * 16
        if memory_limit is not None and scratch > memory_limit:
            skipped.append(
                {
                    "case_id": case.case_id,
                    "reason": "estimated force scratch exceeds memory safety limit",
                    "estimated_force_scratch_bytes": scratch,
                    "memory_safety_limit_bytes": memory_limit,
                }
            )
        else:
            runnable.append(case)
    cases = runnable

    initial_conditions_root = args.initial_conditions_dir.resolve()
    required_seeds = fixture_seeds(
        args.fixture_seed, args.fixture_seed_stride, repetitions
    )
    catalog_plan = initial_condition_plan(
        cases,
        initial_conditions_root,
        required_seeds,
        not args.no_generate_load_015,
        args.allow_available_fixture_fallback,
    )

    if args.dry_run:
        first_minimization_plan = [
            planned
            for planned in catalog_plan
            if math.isclose(
                float(planned["load"]), 0.15, rel_tol=0.0, abs_tol=1e-12
            )
            and int(planned["seed"]) == args.fixture_seed
            and planned["history"] in first_minimization_histories
        ]
        if args.first_minimization_only:
            cases = []
        else:
            cases, catalog_skips = filter_cases_by_plan(
                cases, catalog_plan, required_seeds
            )
            skipped.extend(catalog_skips)
        print(
            json.dumps(
                {
                    "preset": args.preset,
                    "threads": threads,
                    "workload_threads": workload_threads,
                    "large_workload_threads": large_workload_threads,
                    "large_workload_size_threshold": large_workload_size_threshold,
                    "large_workload_repetitions": large_workload_repetitions,
                    "first_minimization_only": args.first_minimization_only,
                    "first_minimization_histories": first_minimization_histories,
                    "first_minimization_plan": first_minimization_plan,
                    "physical_cores_available": available_physical_threads,
                    "physical_place_cpu_ids": physical_place_cpus,
                    "affinity_policies": affinity_policies,
                    "initial_conditions_dir": str(initial_conditions_root),
                    "initial_condition_plan": catalog_plan,
                    "cases": [asdict(case) for case in cases],
                    "skipped": skipped,
                },
                indent=2,
            )
        )
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir / f"{args.preset}_{stamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    events_path = output_dir / "events.jsonl"
    samples_jsonl = output_dir / "samples.jsonl"

    build_start = time.monotonic()
    executable = build_benchmark(args)
    build_seconds = time.monotonic() - build_start
    suite_started_at = utc_now()
    suite_start = time.monotonic()
    total_deadline = suite_start + total_budget_seconds
    affinity = affinity_policies[0]
    warnings = environment_warnings(
        affinity, cpus, topology, max_case_threads, physical_place_cpus
    )
    preparation_records: list[dict[str, Any]] = []
    first_minimization_records: list[dict[str, Any]] = []

    def initial_minimization_budget_seconds(size: int) -> float:
        is_large = (
            large_workload_size_threshold is not None
            and size >= large_workload_size_threshold
        )
        minutes = (
            large_initial_condition_budget_minutes
            if is_large
            else args.initial_condition_budget_minutes
        )
        return 60.0 * minutes

    write_json(output_dir / "initial_condition_plan.json", catalog_plan)
    for planned in catalog_plan:
        if planned["status"] != "generate":
            continue
        target = Path(planned["path"])
        timeout = min(
            initial_minimization_budget_seconds(int(planned["rows"])),
            total_deadline - time.monotonic(),
        )
        record: dict[str, Any] = {
            **planned,
            "started_at": utc_now(),
            "preparation_threads": 1,
        }
        if timeout <= 0.0:
            record.update(status="failed", error="total benchmark budget exhausted")
            preparation_records.append(record)
            write_json(
                output_dir / "initial_condition_preparation.json",
                preparation_records,
            )
            continue
        preparation_case = Case(
            case_id=(
                f"prepare_{planned['history']}_load_"
                f"{load_label(float(planned['load'])).replace('.', 'p')}_"
                f"s{planned['rows']}x{planned['cols']}_seed_{planned['seed']}"
            ),
            experiment="initial_condition_preparation",
            mode="minimization",
            rows=int(planned["rows"]),
            cols=int(planned["cols"]),
            threads=1,
            target_nodes=int(planned["rows"]) * int(planned["cols"]),
            nodes_per_thread=int(planned["rows"]) * int(planned["cols"]),
            scenario="first-noisy",
            reconnection=str(planned["reconnection"]),
            history=str(planned["history"]),
            fixture_load=float(planned["load"]),
            affinity_policy=affinity_policies[0],
        )
        print(f"Preparing missing initial condition: {target}", flush=True)
        started = time.monotonic()
        try:
            native = run_native(
                executable,
                output_dir,
                preparation_case,
                1,
                0,
                timeout,
                args.shear,
                args.perturbation,
                args.workload_disorder,
                args.initial_noise,
                args.replay_load_tolerance,
                args.generated_load_increment,
                int(planned["seed"]),
                physical_place_cpus,
                save_initial_condition=target,
                preconditioner=args.preconditioner,
            )
            if not target.is_file():
                raise BenchmarkError(
                    "native preparation completed without publishing the dump"
                )
            record.update(
                status="generated" if native["initial_condition_saved"] else "existing",
                finished_at=utc_now(),
                elapsed_wall_seconds=time.monotonic() - started,
                dump_sha256=sha256_file(target),
                dump_bytes=target.stat().st_size,
                native=native,
            )
            if native["initial_condition_saved"]:
                metadata_path = target.with_name(
                    f"seed_{planned['seed']}.metadata.json"
                )
                metadata = {
                    "schema_version": 1,
                    "kind": "generated_first_noisy_minimization",
                    "created_at": utc_now(),
                    "dump": str(target.relative_to(initial_conditions_root)),
                    "dump_sha256": record["dump_sha256"],
                    "history": planned["history"],
                    "reconnection": planned["reconnection"],
                    "load": planned["load"],
                    "rows": planned["rows"],
                    "cols": planned["cols"],
                    "seed": planned["seed"],
                    "initial_noise": args.initial_noise,
                    "preconditioner": args.preconditioner,
                    "load_increment": args.generated_load_increment,
                    "workload_disorder": args.workload_disorder,
                }
                try:
                    with metadata_path.open("x", encoding="utf-8") as output:
                        output.write(
                            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
                        )
                except FileExistsError:
                    pass
        except KeyboardInterrupt:
            record.update(
                status="interrupted",
                finished_at=utc_now(),
                elapsed_wall_seconds=time.monotonic() - started,
            )
            preparation_records.append(record)
            write_json(
                output_dir / "initial_condition_preparation.json",
                preparation_records,
            )
            raise
        except (BenchmarkError, OSError) as error:
            record.update(
                status="failed",
                finished_at=utc_now(),
                elapsed_wall_seconds=time.monotonic() - started,
                error=str(error),
            )
            warnings.append(
                f"Initial-condition preparation failed for {target}: {error}"
            )
        preparation_records.append(record)
        write_json(
            output_dir / "initial_condition_preparation.json",
            preparation_records,
        )

    write_json(
        output_dir / "first_minimization_benchmarks.json",
        first_minimization_records,
    )
    if not args.no_first_minimization_thread_sweep:
        first_minimization_plan = [
            planned
            for planned in catalog_plan
            if math.isclose(
                float(planned["load"]), 0.15, rel_tol=0.0, abs_tol=1e-12
            )
            and int(planned["seed"]) == args.fixture_seed
            and planned["history"] in first_minimization_histories
        ]
        for planned in first_minimization_plan:
            size = int(planned["rows"])
            is_large = (
                large_workload_size_threshold is not None
                and size >= large_workload_size_threshold
            )
            size_threads = (
                large_workload_threads if is_large else workload_threads
            )
            for thread_count in size_threads:
                reused = next(
                    (
                        record
                        for record in preparation_records
                        if record.get("status") in {"generated", "existing"}
                        and record["history"] == planned["history"]
                        and int(record["rows"]) == size
                        and int(record["cols"]) == int(planned["cols"])
                        and int(record["seed"]) == int(planned["seed"])
                        and thread_count == 1
                    ),
                    None,
                )
                record: dict[str, Any] = {
                    "history": planned["history"],
                    "reconnection": planned["reconnection"],
                    "load": planned["load"],
                    "rows": planned["rows"],
                    "cols": planned["cols"],
                    "seed": planned["seed"],
                    "preparation_threads": thread_count,
                    "affinity_policy": affinity_policies[0],
                    "started_at": utc_now(),
                }
                if reused is not None:
                    record.update(
                        status="measured",
                        source="initial_condition_preparation",
                        finished_at=reused.get("finished_at"),
                        elapsed_wall_seconds=reused.get("elapsed_wall_seconds"),
                        native=reused["native"],
                    )
                    first_minimization_records.append(record)
                    write_json(
                        output_dir / "first_minimization_benchmarks.json",
                        first_minimization_records,
                    )
                    continue
                timeout = min(
                    initial_minimization_budget_seconds(size),
                    total_deadline - time.monotonic(),
                )
                if timeout <= 0.0:
                    record.update(
                        status="failed",
                        error="total benchmark budget exhausted",
                    )
                    first_minimization_records.append(record)
                    write_json(
                        output_dir / "first_minimization_benchmarks.json",
                        first_minimization_records,
                    )
                    break
                first_case = Case(
                    case_id=(
                        f"first_{planned['history']}_s{size}_t{thread_count}_"
                        f"seed_{planned['seed']}"
                    ),
                    experiment="first_minimization",
                    mode="minimization",
                    rows=size,
                    cols=int(planned["cols"]),
                    threads=thread_count,
                    target_nodes=size * int(planned["cols"]),
                    nodes_per_thread=(
                        size * int(planned["cols"]) / thread_count
                    ),
                    scenario="first-noisy",
                    reconnection=str(planned["reconnection"]),
                    history=str(planned["history"]),
                    fixture_load=float(planned["load"]),
                    affinity_policy=affinity_policies[0],
                )
                print(
                    "Benchmarking first minimization: "
                    f"{planned['history']} L={size}, threads={thread_count}",
                    flush=True,
                )
                started = time.monotonic()
                try:
                    native = run_native(
                        executable,
                        output_dir,
                        first_case,
                        1,
                        0,
                        timeout,
                        args.shear,
                        args.perturbation,
                        args.workload_disorder,
                        args.initial_noise,
                        args.replay_load_tolerance,
                        args.generated_load_increment,
                        int(planned["seed"]),
                        physical_place_cpus,
                        preconditioner=args.preconditioner,
                    )
                    record.update(
                        status="measured",
                        finished_at=utc_now(),
                        elapsed_wall_seconds=time.monotonic() - started,
                        native=native,
                    )
                except (BenchmarkError, OSError) as error:
                    record.update(
                        status="failed",
                        finished_at=utc_now(),
                        elapsed_wall_seconds=time.monotonic() - started,
                        error=str(error),
                    )
                    warnings.append(
                        "First-minimization benchmark failed for "
                        f"{planned['history']} L={size}, threads={thread_count}: "
                        f"{error}"
                    )
                first_minimization_records.append(record)
                write_json(
                    output_dir / "first_minimization_benchmarks.json",
                    first_minimization_records,
                )

    if args.first_minimization_only:
        cases = []
    else:
        catalog_plan = initial_condition_plan(
            cases,
            initial_conditions_root,
            required_seeds,
            generate_load_015=False,
            allow_available_seed_fallback=args.allow_available_fixture_fallback,
        )
        cases, catalog_skips = filter_cases_by_plan(cases, catalog_plan, required_seeds)
        skipped.extend(catalog_skips)
    write_json(output_dir / "initial_condition_plan.json", catalog_plan)
    write_json(
        output_dir / "initial_condition_preparation.json", preparation_records
    )
    original_affinity_variables = {
        name: os.environ.get(name, "")
        for name in (
            "OMP_PLACES",
            "OMP_PROC_BIND",
            "KMP_AFFINITY",
            "GOMP_CPU_AFFINITY",
        )
    }
    git_commit = run_text(["git", "rev-parse", "HEAD"])
    git_status = run_text(["git", "status", "--porcelain"])
    configuration = {
        "preset": args.preset,
        "experiments": experiments,
        "modes": modes,
        "sizes": sizes,
        "workload_sizes": workload_sizes,
        "workload_loads": workload_loads,
        "strong_size": strong_size,
        "weak_base_nodes": weak_base_nodes,
        "threads": threads,
        "workload_threads": workload_threads,
        "large_workload_threads": large_workload_threads,
        "large_workload_size_threshold": large_workload_size_threshold,
        "large_workload_repetitions": large_workload_repetitions,
        "fixed_threads": args.fixed_threads,
        "repetitions": repetitions,
        "calls": args.calls,
        "min_calls": min_calls,
        "max_calls": max_calls,
        "warmup_calls": warmup_calls,
        "sample_seconds": sample_seconds,
        "case_budget_seconds": case_budget_seconds,
        "total_budget_seconds": total_budget_seconds,
        "affinity": args.affinity,
        "affinity_policies": affinity_policies,
        "shear": args.shear,
        "perturbation": args.perturbation,
        "workload_disorder": args.workload_disorder,
        "initial_noise": args.initial_noise,
        "preconditioner": args.preconditioner,
        "replay_load_tolerance": args.replay_load_tolerance,
        "generated_load_increment": args.generated_load_increment,
        "initial_conditions_dir": str(initial_conditions_root),
        "generate_missing_load_015": not args.no_generate_load_015,
        "initial_condition_budget_seconds": (
            60.0 * args.initial_condition_budget_minutes
        ),
        "large_initial_condition_budget_seconds": (
            60.0 * large_initial_condition_budget_minutes
        ),
        "first_minimization_thread_sweep": (
            not args.no_first_minimization_thread_sweep
        ),
        "first_minimization_only": args.first_minimization_only,
        "first_minimization_histories": first_minimization_histories,
        "fixture_seed": args.fixture_seed,
        "fixture_seed_stride": args.fixture_seed_stride,
        "allow_available_fixture_fallback": (
            args.allow_available_fixture_fallback
        ),
        "checksum_tolerance": args.checksum_tolerance,
        "max_memory_fraction": args.max_memory_fraction,
        "shuffle_seed": args.shuffle_seed,
        "keep_noinline": args.keep_noinline,
    }
    system = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpu_count_os": os.cpu_count(),
        "allowed_cpu_ids": cpus,
        "available_memory_bytes_at_start": memory_available,
        "linux_topology": topology,
        "physical_place_cpu_ids": physical_place_cpus,
        "scheduler_environment": scheduler_environment(),
        "original_affinity_environment": original_affinity_variables,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": suite_started_at,
        "repo": str(REPO_ROOT),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "executable": str(executable),
        "executable_sha256": sha256_file(executable),
        "build_seconds": build_seconds,
        "build_metadata": cmake_build_metadata(args.build_dir.resolve()),
        "configuration": configuration,
        "system": system,
        "warnings": warnings,
        "cases": [asdict(case) for case in cases],
        "initial_condition_plan": catalog_plan,
        "initial_condition_preparations": preparation_records,
        "first_minimization_benchmarks": first_minimization_records,
        "skipped_cases": skipped,
        "failed_runs": [],
        "successful_samples": 0,
        "elapsed_seconds": 0.0,
    }
    write_json(output_dir / "manifest.json", manifest)

    print(f"Results: {output_dir}", flush=True)
    planned_samples = sum(
        min(repetitions, case.repetition_limit or repetitions) for case in cases
    )
    print(
        f"Running {len(cases)} cases and {planned_samples} planned samples "
        f"sequentially with affinity policies={','.join(affinity_policies)}.",
        flush=True,
    )

    benchmark_start = suite_start
    rng = random.Random(args.shuffle_seed)
    calibration_order = list(cases)
    rng.shuffle(calibration_order)
    calls_by_case: dict[str, int] = {}
    case_spent: dict[str, float] = {case.case_id: 0.0 for case in cases}
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def remaining_timeout(case: Case) -> float:
        total_remaining = total_deadline - time.monotonic()
        case_remaining = case_budget_seconds - case_spent[case.case_id]
        return min(total_remaining, case_remaining)

    try:
        for index, case in enumerate(calibration_order, start=1):
            if case.mode == "minimization":
                calls_by_case[case.case_id] = 1
                append_jsonl(
                    events_path,
                    {
                        "time": utc_now(),
                        "event": "calibration_not_required",
                        "case_id": case.case_id,
                        "selected_calls": 1,
                        "reason": "one independent minimization per process",
                    },
                )
                continue
            timeout = remaining_timeout(case)
            if timeout <= 0.0:
                if time.monotonic() >= total_deadline:
                    warnings.append("The total benchmark budget was exhausted during calibration.")
                    raise TimeoutError("total benchmark budget exhausted")
                failures.append(
                    {
                        "case_id": case.case_id,
                        "stage": "calibration",
                        "error": "per-case budget exhausted",
                    }
                )
                continue
            print(f"Calibrating {index}/{len(cases)}: {case.case_id}", flush=True)
            probe_calls = min(3, max_calls)
            started = time.monotonic()
            try:
                probe = run_native(
                    executable,
                    output_dir,
                    case,
                    probe_calls,
                    min(warmup_calls, 3),
                    timeout,
                    args.shear,
                    args.perturbation,
                    args.workload_disorder,
                    args.initial_noise,
                    args.replay_load_tolerance,
                    args.generated_load_increment,
                    args.fixture_seed,
                    physical_place_cpus,
                    preconditioner=args.preconditioner,
                )
            except BenchmarkError as error:
                case_spent[case.case_id] += time.monotonic() - started
                total_expired = time.monotonic() >= total_deadline
                failures.append(
                    {
                        "case_id": case.case_id,
                        "stage": "calibration",
                        "error": (
                            "total benchmark budget exhausted during calibration"
                            if total_expired
                            else str(error)
                        ),
                    }
                )
                append_jsonl(events_path, {"time": utc_now(), **failures[-1]})
                if total_expired:
                    warnings.append("The total benchmark budget was exhausted during calibration.")
                    raise TimeoutError("total benchmark budget exhausted")
                continue
            case_spent[case.case_id] += time.monotonic() - started
            calls = choose_calls(
                float(probe["seconds_per_call"]),
                requested_calls,
                sample_seconds,
                min_calls,
                max_calls,
                min(repetitions, case.repetition_limit or repetitions),
                case_budget_seconds - case_spent[case.case_id],
            )
            calls_by_case[case.case_id] = calls
            append_jsonl(
                events_path,
                {
                    "time": utc_now(),
                    "event": "calibration",
                    "case_id": case.case_id,
                    "probe_seconds_per_call": probe["seconds_per_call"],
                    "selected_calls": calls,
                },
            )

        sample_order = 0
        for repetition in range(1, repetitions + 1):
            round_cases = [
                case
                for case in cases
                if case.case_id in calls_by_case
                and repetition <= (case.repetition_limit or repetitions)
            ]
            rng.shuffle(round_cases)
            for case in round_cases:
                sample_order += 1
                timeout = remaining_timeout(case)
                if timeout <= 0.0:
                    if time.monotonic() >= total_deadline:
                        warnings.append("The total benchmark budget was exhausted during sampling.")
                        raise TimeoutError("total benchmark budget exhausted")
                    failure = {
                        "case_id": case.case_id,
                        "stage": "sample",
                        "repetition": repetition,
                        "error": "per-case budget exhausted",
                    }
                    failures.append(failure)
                    append_jsonl(events_path, {"time": utc_now(), **failure})
                    continue
                print(
                    f"Sample {sample_order}: {case.case_id}, repetition "
                    f"{repetition}/{min(repetitions, case.repetition_limit or repetitions)}, "
                    f"calls={calls_by_case[case.case_id]}",
                    flush=True,
                )
                started = time.monotonic()
                try:
                    sample_seed = sample_fixture_seed(
                        case,
                        catalog_plan,
                        args.fixture_seed,
                        args.fixture_seed_stride,
                        repetition,
                    )
                    sample_initial_condition = (
                        initial_condition_path(
                            initial_conditions_root,
                            case.history,
                            case.fixture_load,
                            case.rows,
                            case.cols,
                            sample_seed,
                        )
                        if case.experiment == "workload"
                        else None
                    )
                    native = run_native(
                        executable,
                        output_dir,
                        case,
                        calls_by_case[case.case_id],
                        0 if case.mode == "minimization" else warmup_calls,
                        timeout,
                        args.shear,
                        args.perturbation,
                        args.workload_disorder,
                        args.initial_noise,
                        args.replay_load_tolerance,
                        args.generated_load_increment,
                        sample_seed,
                        physical_place_cpus,
                        initial_condition=sample_initial_condition,
                        preconditioner=args.preconditioner,
                    )
                except BenchmarkError as error:
                    case_spent[case.case_id] += time.monotonic() - started
                    total_expired = time.monotonic() >= total_deadline
                    failure = {
                        "case_id": case.case_id,
                        "stage": "sample",
                        "repetition": repetition,
                        "error": (
                            "total benchmark budget exhausted during sample"
                            if total_expired
                            else str(error)
                        ),
                    }
                    failures.append(failure)
                    append_jsonl(events_path, {"time": utc_now(), **failure})
                    if total_expired:
                        warnings.append("The total benchmark budget was exhausted during sampling.")
                        raise TimeoutError("total benchmark budget exhausted")
                    continue
                case_spent[case.case_id] += time.monotonic() - started
                sample = {
                    **asdict(case),
                    "repetition": repetition,
                    "sample_order": sample_order,
                    "recorded_at": utc_now(),
                    "affinity_policy": case.affinity_policy,
                    **native,
                }
                samples.append(sample)
                append_jsonl(samples_jsonl, sample)
                if time.monotonic() >= total_deadline:
                    warnings.append("The total benchmark budget was exhausted.")
                    raise TimeoutError("total benchmark budget exhausted")
    except KeyboardInterrupt:
        warnings.append("The benchmark was interrupted; partial results were preserved.")
        manifest["status"] = "interrupted"
    except TimeoutError:
        manifest["status"] = "budget_exhausted"
    else:
        manifest["status"] = "complete" if not failures else "complete_with_failures"

    bound_samples = [
        sample for sample in samples if sample["affinity_policy"] != "none"
    ]
    cpu_to_physical_core = {
        int(record["cpu"]): (
            int(record["package"]),
            int(record["die"]),
            int(record["core"]),
        )
        for record in topology.get("cpus", [])
    }

    def distinct_physical_cores(sample: dict[str, Any]) -> int:
        cpu_ids = [
            int(item.get("cpu", -1)) for item in sample["thread_placement"]
        ]
        if any(cpu < 0 or cpu not in cpu_to_physical_core for cpu in cpu_ids):
            return -1
        return len({cpu_to_physical_core[cpu] for cpu in cpu_ids})

    affinity_verified = not bound_samples
    if bound_samples:
        if platform.system() == "Linux":
            affinity_verified = all(
                distinct_physical_cores(sample) == int(sample["threads_observed"])
                for sample in bound_samples
            )
        else:
            affinity_verified = all(
                int(sample.get("openmp_num_places", 0)) > 0
                for sample in bound_samples
            )
    manifest["affinity_verified"] = affinity_verified

    if bound_samples and any(
        int(sample.get("openmp_num_places", 0)) == 0 for sample in bound_samples
    ):
        warnings.append(
            "The OpenMP runtime accepted the binding policy but exposed no OpenMP "
            "places. Treat affinity as unverified on this runtime; Linux CPU IDs, "
            "when available, are the stronger check."
        )
    if platform.system() == "Linux" and samples:
        missing_cpu_ids = any(
            any(int(item.get("cpu", -1)) < 0 for item in sample["thread_placement"])
            for sample in samples
        )
        if missing_cpu_ids:
            warnings.append(
                "At least one Linux sample could not report its per-thread CPU ID."
            )
        duplicate_cpu_ids = any(
            len(
                {
                    int(item["cpu"])
                    for item in sample["thread_placement"]
                    if int(item.get("cpu", -1)) >= 0
                }
            )
            < int(sample["threads_observed"])
            for sample in samples
        )
        if duplicate_cpu_ids:
            warnings.append(
                "At least one sample placed multiple OpenMP threads on the same "
                "logical CPU; verify scheduler and affinity settings."
            )
        duplicate_physical_cores = any(
            distinct_physical_cores(sample) < int(sample["threads_observed"])
            for sample in bound_samples
        )
        if duplicate_physical_cores:
            warnings.append(
                "At least one sample placed multiple OpenMP threads on sibling "
                "hardware threads of the same physical core."
            )

    summaries = summarize(cases, samples, args.checksum_tolerance)
    write_csv(output_dir / "samples.csv", samples)
    write_csv(
        output_dir / "summary.csv",
        summaries,
        fieldnames=[
            *asdict(
                Case(
                    case_id="",
                    experiment="",
                    mode="",
                    rows=0,
                    cols=0,
                    threads=0,
                    target_nodes=0,
                    nodes_per_thread=0.0,
                    scenario="",
                    reconnection="",
                    history="",
                    fixture_load=0.0,
                    affinity_policy="",
                )
            ).keys(),
            "nodes",
            "elements",
            "samples",
            "calls_per_sample",
            "mean_seconds_per_call",
            "stdev_seconds_per_call",
            "median_seconds_per_call",
            "min_seconds_per_call",
            "max_seconds_per_call",
            "coefficient_of_variation_percent",
            "standard_error_seconds_per_call",
            "mean_calls_per_second",
            "stdev_calls_per_second",
            "normalized_checksum_mean",
            "normalized_checksum_spread",
            "checksum_comparison_applicable",
            "repeat_checksum_ok",
            "fixture_seeds",
            "actual_initial_loads",
            "force_scratch_bytes_estimate",
            "mean_initialization_seconds",
            "mean_function_evaluations",
            "stdev_function_evaluations",
            "mean_minimizer_iterations",
            "mean_edge_flips",
            "stdev_edge_flips",
            "edge_flip_trigger_rate",
            "mean_reconnect_cycles",
            "termination_types",
        ],
    )
    write_json(output_dir / "summary.json", summaries)
    recommendations = create_recommendations(output_dir, manifest, summaries)
    if recommendations["invalid_groups"]:
        warnings.append(
            "At least one edgeFlip workload did not trigger an actual edge flip; "
            "no recommendation was made for that group."
        )
    plot_title = (
        f"{platform.node()} · affinity={','.join(affinity_policies)}"
    )
    plots = make_plots(output_dir, summaries, plot_title)
    if not plots:
        warnings.append("Matplotlib was unavailable, so no plots were generated.")

    manifest["finished_at"] = utc_now()
    manifest["warnings"] = warnings
    manifest["failed_runs"] = failures
    manifest["successful_samples"] = len(samples)
    manifest["elapsed_seconds"] = time.monotonic() - benchmark_start
    manifest["selected_calls_by_case"] = calls_by_case
    manifest["case_elapsed_seconds"] = case_spent
    manifest["plots"] = plots
    write_json(output_dir / "manifest.json", manifest)
    create_report(output_dir, manifest, summaries, plots, recommendations)

    print(
        f"Finished with status={manifest['status']}; {len(samples)} samples, "
        f"{len(failures)} failed runs.",
        flush=True,
    )
    print(f"Report: {output_dir / 'report.md'}", flush=True)
    return 0 if manifest["status"] in {"complete", "complete_with_failures"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
