#!/usr/bin/env python3
"""Rerun comparison FIRE events that stopped at FIRE's iteration cap."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from compare_minimizers_same_event import dump_load, run_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fire-max-it", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--lookahead", type=float, default=2e-4)
    parser.add_argument("--energy-threshold", type=float, default=1e-10)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=float, default=43_200.0)
    return parser.parse_args()


def read_csv(stream: io.BufferedIOBase) -> list[dict[str, str]]:
    return list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8", newline="")))


def archive_member(path: str, seed: str) -> str:
    normalized = path.replace("\\", "/")
    marker = f"{seed}/"
    index = normalized.find(marker)
    if index < 0:
        raise RuntimeError(f"Cannot map {path} into {seed}.zip")
    return normalized[index:]


def capped_events(input_root: Path) -> list[tuple[str, str, dict, str | None]]:
    events: list[tuple[str, str, dict, str | None]] = []
    for seed_index in range(10):
        seed = f"seed_{seed_index}"
        seed_dir = input_root / seed
        if seed_dir.is_dir():
            manifests = ((str(path), path.read_text(), None) for path in seed_dir.rglob("event_manifest.json"))
        else:
            archive_path = input_root / f"{seed}.zip"
            if not archive_path.is_file():
                raise FileNotFoundError(f"Neither {seed_dir} nor {archive_path} exists")
            with zipfile.ZipFile(archive_path) as archive:
                manifests = [
                    (member, archive.read(member).decode("utf-8"), str(archive_path))
                    for member in archive.namelist()
                    if member.endswith("/event_manifest.json")
                ]
        for manifest_path, text, archive_path in manifests:
            manifest = json.loads(text)
            fire = next((result for result in manifest["results"] if result["algorithm"] == "FIRE"), None)
            if fire is None or fire["status"] != "completed-first-drop":
                continue
            macro_path = str(Path(fire["minimization_directories"][0]).parents[2] / "macroData.csv")
            if archive_path is None:
                with open(macro_path, "rb") as stream:
                    rows = read_csv(stream)
            else:
                with zipfile.ZipFile(archive_path) as archive, archive.open(archive_member(macro_path, seed)) as stream:
                    rows = read_csv(stream)
            first_drop_step = int(fire["first_drop"]["load_step"])
            if any(
                int(row["load_step"]) <= first_drop_step and row["FIRE_Term_reason"] == "4"
                for row in rows
            ):
                events.append((seed, manifest_path, fire, archive_path))
    return events


def rerun_one(
    event: tuple[str, str, dict, str | None], args: argparse.Namespace
) -> str:
    seed, manifest_path, original_fire, archive_path = event
    event_name = Path(manifest_path).parent.name
    event_root = args.output_root / seed / event_name
    result_path = event_root / "rerun_manifest.json"
    if result_path.is_file():
        return f"Skipping completed rerun: {event_root}"
    if event_root.exists():
        raise FileExistsError(f"Refusing to reuse incomplete rerun directory: {event_root}")
    event_root.mkdir(parents=True)

    original_dump = original_fire["source_dump"]
    with tempfile.TemporaryDirectory(prefix=f"{seed}_{event_name}_") as temporary:
        if archive_path is None:
            source_dump = Path(original_dump)
        else:
            source_dump = Path(temporary) / Path(original_dump).name
            with zipfile.ZipFile(archive_path) as archive, archive.open(
                archive_member(original_dump, seed)
            ) as source, source_dump.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        if not source_dump.is_file():
            raise FileNotFoundError(source_dump)
        rerun = run_one(
            binary=args.binary,
            source_dump=source_dump,
            output_root=event_root,
            algorithm="FIRE",
            load=dump_load(source_dump),
            threads=args.threads,
            lookahead=args.lookahead,
            threshold=args.energy_threshold,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
            dry_run=False,
            fire_max_it=args.fire_max_it,
        )
    rerun["source_dump"] = original_dump
    result_path.write_text(
        json.dumps(
            {
                "original_event_manifest": manifest_path,
                "original_fire_result": original_fire,
                "fire_max_it": args.fire_max_it,
                "rerun_fire_result": rerun,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return f"Completed rerun: {event_root}"


def main() -> None:
    args = parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    if args.fire_max_it < 1 or args.threads < 1 or args.jobs < 1:
        raise ValueError("fire-max-it, threads, and jobs must be positive")
    events = capped_events(args.input_root)
    if not events:
        raise RuntimeError("No capped FIRE events were found")
    print(f"Rerunning {len(events)} FIRE-capped events with maxIt={args.fire_max_it}")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for message in pool.map(lambda event: rerun_one(event, args), events):
            print(message, flush=True)


if __name__ == "__main__":
    main()
