#!/usr/bin/env python3
"""Resume one minimizer-comparison segment without duplicating its trajectory."""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import shutil
import subprocess
from pathlib import Path


LOAD_PATTERN = re.compile(r"<load>([-+0-9.eE]+)</load>")
STEP_PATTERN = re.compile(r"<loadSteps>(\d+)</loadSteps>")
VTU_STEP_PATTERN = re.compile(r"\.(\d+)\.vtu$")
PVD_STEP_PATTERN = re.compile(r'timestep="(\d+)"')


def dump_state(path: Path) -> tuple[float, int]:
    opener = gzip.open if path.name.endswith(".gz") else open
    load = None
    load_step = None
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if load is None:
                match = LOAD_PATTERN.search(line)
                if match:
                    load = float(match.group(1))
            if load_step is None:
                match = STEP_PATTERN.search(line)
                if match:
                    load_step = int(match.group(1))
            if load is not None and load_step is not None:
                return load, load_step
    raise RuntimeError(f"Missing load metadata in {path}")


def last_macro_load(path: Path) -> float | None:
    if not path.is_file():
        return None
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        last = None
        for row in rows:
            last = float(row["load"])
    return last


def truncate_to_restart(output_dir: Path, restart_step: int) -> None:
    macro_path = output_dir / "macroData.csv"
    if macro_path.is_file():
        with macro_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or "load_step" not in reader.fieldnames:
                raise RuntimeError(f"Invalid macro CSV: {macro_path}")
            rows = [row for row in reader if int(row["load_step"]) <= restart_step]
            fieldnames = reader.fieldnames
        temporary = macro_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(macro_path)

    data_dir = output_dir / "data"
    if data_dir.is_dir():
        for path in data_dir.rglob("*.vtu"):
            match = VTU_STEP_PATTERN.search(path.name)
            if match and int(match.group(1)) > restart_step:
                path.unlink()

    collection_path = output_dir / "collection.pvd"
    if collection_path.is_file():
        lines = collection_path.read_text().splitlines(keepends=True)
        retained = []
        for line in lines:
            match = PVD_STEP_PATTERN.search(line)
            if match is None or int(match.group(1)) <= restart_step:
                retained.append(line)
        collection_path.write_text("".join(retained))


def choose_restart(output_dir: Path, source_dump: Path) -> tuple[Path, float, int]:
    dump_dir = output_dir / "dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    initial_dump = dump_dir / "initial_dump.xml.gz"
    if not initial_dump.exists():
        shutil.copy2(source_dump, initial_dump)

    candidates = []
    paths = [initial_dump, *dump_dir.glob("dump_l*.xml.gz")]
    for path in paths:
        load, load_step = dump_state(path)
        candidates.append((load, load_step, path))
    if not candidates:
        raise RuntimeError(f"No restart dump in {dump_dir}")
    load, load_step, path = max(candidates)
    return path, load, load_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dump", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target-load", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.name
    macro_path = output_dir / "macroData.csv"
    completed_load = last_macro_load(macro_path)
    if completed_load is not None and completed_load >= args.target_load - 5e-6:
        print(f"Already complete at load {completed_load:.17g}")
        return

    restart, restart_load, restart_step = choose_restart(
        output_dir, args.source_dump
    )
    if restart_load >= args.target_load - 5e-6:
        print(f"Already complete at restart load {restart_load:.17g}")
        return
    truncate_to_restart(output_dir, restart_step)
    command = [
        str(args.binary),
        "-d",
        str(restart),
        "-c",
        str(args.config),
        "-o",
        str(args.output_root),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
