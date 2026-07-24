#!/usr/bin/env python3
"""Reproduce confirmed LBFGS crashes and test a different correction count.

Each case is kept in a separate output tree.  The original simulation output
is never used as a destination: only its config and nearest normal dump are
read.  For each case we run, in order:

1. M=7 from the nearest normal dump, to create a crash_ dump;
2. M=7 from that crash_ dump, to verify reproducibility;
3. M=6 from that same crash_ dump.

The script detects crashes from the existing macro data by looking for the
large-energy/large-force signature produced by the reduction guard.  It
intentionally refuses to guess a crash load for an incomplete run without
that signature.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_ROOT = Path("/Volumes/data/MTS2D_output")
DEFAULT_OUTPUT_ROOT = Path("/Volumes/data/MTS2D_debug_m6_batch")
DEFAULT_EXECUTABLE = Path(
    "/Users/eliaslundheim/work/PhD/MTS2D/build-release/MTS2D"
)

# These thresholds are deliberately well above the normal energies/forces in
# this 100x100 batch, but low enough to include reductions that abort before
# the values reach the much larger overflow seen in the first crash cases.
CRASH_ENERGY_THRESHOLD = 1e4
CRASH_FORCE_THRESHOLD = 1e6
CRASH_SIGNATURE = re.compile(
    r"Reduction exploded .*?eIndex=(\d+), m3Nr=(\d+), "
    r"load=([0-9.eE+-]+), loadSteps=(\d+)"
)
DUMP_NAME = re.compile(r"^dump_l(.+)\.xml\.gz$")


@dataclass(frozen=True)
class CrashCase:
    simulation_path: Path
    config_path: Path
    seed: int
    load_increment: float
    crash_load: float
    nearest_dump: Path

    @property
    def label(self) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.simulation_path.name)
        return safe.strip("_")


def parse_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def minimization_csvs(simulation_path: Path) -> Iterable[Path]:
    candidates = [simulation_path / "macroData.csv"]
    candidates.extend(
        simulation_path.glob("data/minimizationData/*/macroData.csv")
    )
    seen: set[Path] = set()
    for path in candidates:
        if path.is_file() and path not in seen:
            seen.add(path)
            yield path


def find_crash_load(simulation_path: Path) -> float | None:
    loads: list[float] = []
    for csv_path in minimization_csvs(simulation_path):
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    energy = abs(float(row["total_energy"]))
                    max_force = abs(float(row["max_force"]))
                    load = float(row["load"])
                except (KeyError, TypeError, ValueError):
                    continue
                if energy > CRASH_ENERGY_THRESHOLD or max_force > CRASH_FORCE_THRESHOLD:
                    loads.append(load)
    return max(loads) if loads else None


def dump_load(path: Path) -> float | None:
    match = DUMP_NAME.match(path.name)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def nearest_normal_dump(simulation_path: Path, crash_load: float) -> Path:
    dumps: list[tuple[float, Path]] = []
    for path in (simulation_path / "dumps").glob("dump_*.xml.gz"):
        if path.name.startswith("crash_"):
            continue
        load = dump_load(path)
        if load is not None and load <= crash_load + 1e-12:
            dumps.append((load, path))
    if not dumps:
        raise FileNotFoundError(
            f"No normal dump at or before crash load {crash_load} in "
            f"{simulation_path / 'dumps'}"
        )
    return max(dumps, key=lambda item: item[0])[1]


def discover_cases(input_root: Path) -> tuple[list[CrashCase], list[str]]:
    cases: list[CrashCase] = []
    skipped: list[str] = []
    for simulation_path in sorted(input_root.iterdir()):
        if not simulation_path.is_dir():
            continue
        config_path = simulation_path / "config.conf"
        if not config_path.is_file():
            continue
        config = parse_config(config_path)
        if (
            config.get("experiment") != "reversibilityProtocolTest"
            or config.get("reconnectionMethod") != "edgeFlip"
            or config.get("minimizer") != "LBFGS"
            or config.get("rows") != "100"
            or config.get("cols") != "100"
            or float(config.get("maxLoad", "nan")) != 1.0
        ):
            continue

        crash_load = find_crash_load(simulation_path)
        if crash_load is None:
            skipped.append(f"{simulation_path}: no confirmed crash signature")
            continue
        try:
            increment = float(config["loadIncrement"])
            seed = int(config["seed"])
            dump = nearest_normal_dump(simulation_path, crash_load)
        except (KeyError, ValueError, FileNotFoundError) as error:
            raise RuntimeError(f"Cannot prepare {simulation_path}: {error}") from error
        cases.append(
            CrashCase(
                simulation_path=simulation_path,
                config_path=config_path,
                seed=seed,
                load_increment=increment,
                crash_load=crash_load,
                nearest_dump=dump,
            )
        )
    return cases, skipped


def write_config(
    source: Path, destination: Path, *, max_load: float, corrections: int
) -> None:
    replacements = {
        "maxLoad": f"{max_load:.17g}",
        "LBFGSNrCorrections": str(corrections),
    }
    output: list[str] = []
    seen: set[str] = set()
    for raw_line in source.read_text().splitlines(keepends=True):
        line = raw_line.split("#", 1)[0]
        if "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in replacements:
                newline = "\n" if raw_line.endswith("\n") else ""
                output.append(f"{key} = {replacements[key]}{newline}")
                seen.add(key)
                continue
        output.append(raw_line)
    missing = set(replacements) - seen
    if missing:
        raise RuntimeError(f"Config {source} is missing keys {sorted(missing)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(output))


def failure_signatures(log_path: Path) -> list[tuple[int, int, float, int]]:
    text = log_path.read_text(errors="replace")
    signatures = []
    for match in CRASH_SIGNATURE.finditer(text):
        signatures.append(
            (
                int(match.group(1)),
                int(match.group(2)),
                float(match.group(3)),
                int(match.group(4)),
            )
        )
    return signatures


def run_phase(
    executable: Path,
    dump: Path,
    config: Path,
    output: Path,
    log: Path,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        "-d",
        str(dump),
        "-c",
        str(config),
        "-o",
        str(output) + "/",
    ]
    with log.open("w") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.run(
            command,
            cwd=executable.parent,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode


def crash_dump_in(output: Path) -> Path:
    dumps = sorted(output.glob("**/dumps/crash_*.xml.gz"), key=lambda p: p.stat().st_mtime)
    if not dumps:
        raise FileNotFoundError(f"No crash dump was generated below {output}")
    return dumps[-1]


def run_case(case: CrashCase, executable: Path, output_root: Path) -> dict:
    case_root = output_root / case.label
    case_root.mkdir(parents=True, exist_ok=True)
    cap = case.crash_load + 5 * case.load_increment
    result = {
        "simulation": str(case.simulation_path),
        "seed": case.seed,
        "crash_load": case.crash_load,
        "nearest_dump": str(case.nearest_dump),
        "max_load_cap": cap,
        "status": "started",
    }
    try:
        config_m7 = case_root / "configs" / "m7.conf"
        config_m6 = case_root / "configs" / "m6.conf"
        write_config(case.config_path, config_m7, max_load=cap, corrections=7)
        write_config(case.config_path, config_m6, max_load=cap, corrections=6)

        original_output = case_root / "original_m7"
        original_code = run_phase(
            executable,
            case.nearest_dump,
            config_m7,
            original_output,
            case_root / "original_m7.log",
        )
        original_dump = crash_dump_in(original_output)
        original_signatures = failure_signatures(case_root / "original_m7.log")

        verify_output = case_root / "verify_m7"
        verify_code = run_phase(
            executable,
            original_dump,
            config_m7,
            verify_output,
            case_root / "verify_m7.log",
        )
        verify_signatures = failure_signatures(case_root / "verify_m7.log")

        m6_output = case_root / "m6"
        m6_code = run_phase(
            executable,
            original_dump,
            config_m6,
            m6_output,
            case_root / "m6.log",
        )
        m6_signatures = failure_signatures(case_root / "m6.log")

        result.update(
            {
                "original_exit_code": original_code,
                "crash_dump": str(original_dump),
                "original_signatures": original_signatures,
                "verify_exit_code": verify_code,
                "verify_signatures": verify_signatures,
                "reproduced": bool(original_signatures)
                and bool(verify_signatures)
                and original_signatures[-1] == verify_signatures[-1],
                "m6_exit_code": m6_code,
                "m6_signatures": m6_signatures,
                "m6_avoided_crash": m6_code == 0 and not m6_signatures,
                "status": "complete",
            }
        )
    except Exception as error:
        result.update({"status": "error", "error": repr(error)})
    (case_root / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.executable.is_file():
        raise FileNotFoundError(args.executable)
    cases, skipped = discover_cases(args.input_root)
    print(f"Discovered {len(cases)} confirmed crash cases.", flush=True)
    for case in cases:
        print(
            f"- {case.label}: crash={case.crash_load:.17g}, "
            f"nearest={case.nearest_dump}, cap="
            f"{case.crash_load + 5 * case.load_increment:.17g}",
            flush=True,
        )
    for message in skipped:
        print(f"Skipped: {message}", flush=True)
    if args.dry_run:
        return 0
    if not cases:
        raise RuntimeError("No confirmed crash cases found")

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_case, case, args.executable, args.output_root): case
            for case in cases
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"Finished {result['simulation']}: {result['status']}", flush=True
            )
    summary = args.output_root / "summary.json"
    summary.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"Wrote {summary}", flush=True)
    return 0 if all(item["status"] == "complete" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
