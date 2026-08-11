#!/usr/bin/env python3
"""Extract one or two verified no-forward-m3 events after each dump."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

from replay_selected_real_space_event import (
    ReplayArguments,
    prepare_replay_configuration,
    read_macro_data,
    run_replay,
    validate_replay,
)


DUMP_PATTERN = re.compile(r"dump_l(?P<load>[0-9.eE+-]+)(?:\.xml)?\.gz$")


def dump_load(path: Path) -> float:
    match = DUMP_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected dump filename: {path.name}")
    return float(match.group("load"))


def select_targets(macro: pd.DataFrame, start_load: float, increment: float, limit: int) -> list[float]:
    m3_column = next(
        (column for column in ("nr_elements_with_m3_change", "nr_elements_with_m3_fix_change")
         if column in macro.columns),
        None,
    )
    if "load" not in macro.columns or m3_column is None:
        raise KeyError("Macro data must contain load and a forward m3-change column")
    candidates = macro[
        (macro["load"] > start_load + max(1e-10, increment * 1e-5))
        & (macro[m3_column] == 0)
    ]
    return [float(value) for value in candidates["load"].head(limit)]


def extract_from_dumps(
    source_job: Path,
    output_root: Path,
    mts2d_binary: Path,
    maximum_events_per_dump: int,
    selected_dumps: list[Path] | None = None,
) -> Path:
    source_job = Path(source_job)
    macro = read_macro_data(source_job / "macroData.csv")
    config_values = {}
    for line in (source_job / "config.conf").read_text().splitlines():
        content = line.split("#", 1)[0].strip()
        if "=" in content:
            key, value = content.split("=", 1)
            config_values[key.strip()] = value.strip()
    increment = float(config_values["loadIncrement"])
    dumps = sorted(selected_dumps or (source_job / "dumps").glob("*.xml.gz"), key=dump_load)
    if not dumps:
        raise FileNotFoundError(f"No dump files found in {source_job / 'dumps'}")
    records = []
    for dump in dumps:
        start_load = dump_load(dump)
        targets = select_targets(macro, start_load, increment, maximum_events_per_dump)
        if not targets:
            continue
        event_output = output_root / source_job.name / dump.stem
        replay_args = ReplayArguments(
            source_job=source_job,
            dump=dump,
            target_load=targets[-1],
            output_directory=event_output,
            mts2d_binary=mts2d_binary,
            maximum_elastic_events=maximum_events_per_dump,
        )
        config = prepare_replay_configuration(replay_args)
        run_replay(replay_args, config)
        event_directories = sorted(event_output.rglob("elastic_replay_l_*"))
        if len(event_directories) != len(targets):
            raise RuntimeError(
                f"Expected {len(targets)} elastic events after {dump}, "
                f"found {event_directories}."
            )
        for event_directory in event_directories:
            target_load = float(event_directory.name.removeprefix("elastic_replay_l_")) + increment
            event_args = ReplayArguments(
                source_job=source_job,
                dump=dump,
                target_load=target_load,
                output_directory=event_output,
                mts2d_binary=mts2d_binary,
                maximum_elastic_events=maximum_events_per_dump,
            )
            validate_replay(event_args, event_directory)
            records.append({
                "source_job": str(source_job),
                "dump": str(dump),
                "dump_load": start_load,
                "target_load": target_load,
                "event_directory": str(event_directory),
                "verified_forward_m3_changes": 0,
            })
    manifest = output_root / source_job.name / "elastic_replay_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys() if records else ["source_job"])
        writer.writeheader()
        writer.writerows(records)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mts2d-binary", type=Path, required=True)
    parser.add_argument("--maximum-events-per-dump", type=int, default=2)
    parser.add_argument("--dump", type=Path, action="append", dest="selected_dumps")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.maximum_events_per_dump <= 0:
        raise ValueError("maximum-events-per-dump must be positive")
    manifest = extract_from_dumps(
        args.source_job,
        args.output_root,
        args.mts2d_binary,
        args.maximum_events_per_dump,
        args.selected_dumps,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
