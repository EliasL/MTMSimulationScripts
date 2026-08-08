#!/usr/bin/env python3
"""Replay one selected event with a forced backward test.

The implementation should restart from one explicitly supplied dump, advance
to one target load, save the five real-space states, and stop.  It must never
modify the source simulation directory or replay a list of events implicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import StringIO
import re
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from Plotting.real_space_events.acquisition import state_paths_from_directory


EVENT_DIRECTORY_PATTERN = re.compile(
    r"(?:elastic_replay|rev_drop|irrev_drop)_l_(?P<load>[0-9.eE+-]+)"
)


def event_start_load(path: Path) -> float:
    match = EVENT_DIRECTORY_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected replay directory name: {path.name}")
    return float(match.group("load"))


def replay_macro_path(event_directory: Path) -> Path:
    matches = [
        parent / "macroData.csv"
        for parent in event_directory.parents
        if (parent / "macroData.csv").is_file()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one replay macroData.csv above {event_directory}, found {matches}."
        )
    return matches[0]


def read_macro_data(path: Path) -> pd.DataFrame:
    """Read resumed macro data using its latest declared column layout."""

    lines = Path(path).read_text().splitlines()
    declared_headers = [
        line.removeprefix("#HEADER:")
        for line in lines
        if line.startswith("#HEADER:")
    ]
    ordinary_headers = [
        line for line in lines if line.startswith("load_step,")
    ]
    headers = declared_headers or ordinary_headers
    if not headers:
        raise ValueError(f"No macro-data header found in {path}.")
    data_lines = [
        line
        for line in lines
        if line and not line.startswith("#") and not line.startswith("load_step,")
    ]
    if not data_lines:
        return pd.DataFrame(columns=headers[-1].split(","))
    return pd.read_csv(StringIO("\n".join([headers[-1], *data_lines])))


def read_config_values(path: Path) -> dict[str, str]:
    values = {}
    for line in Path(path).read_text().splitlines():
        content = line.split("#", 1)[0].strip()
        if "=" in content:
            key, value = content.split("=", 1)
            values[key.strip()] = value.strip()
    return values


@dataclass(frozen=True)
class ReplayArguments:
    source_job: Path
    dump: Path
    target_load: float
    output_directory: Path
    mts2d_binary: Path
    expected_event_kind: str = "elastic"
    maximum_elastic_events: int = 1
    allow_stress_diagnostic_mismatch: bool = False

    def validate(self) -> None:
        if self.expected_event_kind not in {"elastic", "plastic"}:
            raise ValueError(
                "expected_event_kind must be either 'elastic' or 'plastic', got "
                f"{self.expected_event_kind!r}"
            )
        if self.maximum_elastic_events < 0:
            raise ValueError("maximum_elastic_events cannot be negative")


def prepare_replay_configuration(args: ReplayArguments) -> Path:
    """Write a private config for exactly one selected target load.

    The configuration must preserve the original numerical settings, disable
    unrelated bulk output, use a private run name, and enable a dedicated
    selected-event protocol that does not return early for zero m3 change.
    """

    args.validate()
    source_config = args.source_job / "config.conf"
    if not source_config.is_file():
        raise FileNotFoundError(f"Missing source config: {source_config}")
    if not args.dump.is_file():
        raise FileNotFoundError(f"Missing dump: {args.dump}")
    if not args.target_load > 0:
        raise ValueError(f"target_load must be positive, got {args.target_load}")

    lines = source_config.read_text().splitlines()
    overrides = {
        "experiment": "reversibilityProtocolTest",
        "maxLoad": f"{args.target_load:.17g}",
        "writeDumps": "false",
        "writeMeshVTUs": "false",
        "saveElasticReversibilityStates": str(
            args.maximum_elastic_events > 0
        ).lower(),
        "maximumSavedElasticReversibilityStates": str(
            args.maximum_elastic_events
        ),
        "saveFinalReversibilityState": str(
            args.expected_event_kind == "plastic"
        ).lower(),
        "showProgress": "-1",
        "forceReRun": "true",
    }
    found = set()
    output = []
    for line in lines:
        content = line.split("#", 1)[0]
        key = content.split("=", 1)[0].strip()
        if key in overrides:
            output.append(f"{key}={overrides[key]}")
            found.add(key)
        else:
            output.append(line)
    output.extend(f"{key}={value}" for key, value in overrides.items() if key not in found)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    source_macro = args.source_job / "macroData.csv"
    if not source_macro.is_file():
        raise FileNotFoundError(f"Missing source macro data: {source_macro}")
    source_header = next(
        line for line in source_macro.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    replay_job_output = args.output_directory / args.source_job.name
    replay_job_output.mkdir(parents=True, exist_ok=True)
    replay_macro = replay_job_output / "macroData.csv"
    if replay_macro.exists() and replay_macro.stat().st_size > 0:
        raise FileExistsError(f"Replay output already exists: {replay_macro}")
    # Supplying the original header lets loadSimulation recover the
    # reversibility columns before it initializes the private CSV file.
    replay_macro.write_text(source_header + "\n")
    config_path = args.output_directory / "replay_config.conf"
    config_path.write_text("\n".join(output) + "\n")
    return config_path


def run_replay(args: ReplayArguments, config_path: Path) -> Path:
    """Run MTS2D in a private temporary working directory and return its event dir."""

    if not args.mts2d_binary.is_file():
        raise FileNotFoundError(f"MTS2D binary not found: {args.mts2d_binary}")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(args.mts2d_binary),
            "-d", str(args.dump),
            "-c", str(config_path),
            "-o", str(args.output_directory),
            "-r",
        ],
        check=True,
    )
    args.validate()
    elastic_directories = sorted(
        path for path in args.output_directory.rglob("elastic_replay_l_*")
        if path.is_dir()
    )
    if len(elastic_directories) > args.maximum_elastic_events:
        raise RuntimeError(
            f"Replay saved {len(elastic_directories)} elastic events, exceeding "
            f"the configured cap of {args.maximum_elastic_events}."
        )
    config_values = read_config_values(args.source_job / "config.conf")
    increment = float(config_values["loadIncrement"])
    expected_start_load = args.target_load - increment
    pattern = (
        "elastic_replay_l_*"
        if args.expected_event_kind == "elastic"
        else "*_drop_l_*"
    )
    event_directories = sorted(
        path for path in args.output_directory.rglob(pattern) if path.is_dir()
    )
    matching = [
        path for path in event_directories
        if abs(event_start_load(path) - expected_start_load)
        <= max(1e-10, increment * 1e-5)
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"Expected one replayed event at start load {expected_start_load} below "
            f"{args.output_directory}, found {matching}; all events={event_directories}."
        )
    return matching[0]


def validate_replay(args: ReplayArguments, event_directory: Path) -> None:
    """Compare forward energy, stress and m3 values with the original macro row."""

    state_paths_from_directory(event_directory)
    source_config = args.source_job / "config.conf"
    config_values = read_config_values(source_config)
    load_increment = float(config_values["loadIncrement"])
    expected_start_load = args.target_load - load_increment
    actual_start_load = event_start_load(event_directory)
    if abs(actual_start_load - expected_start_load) > max(1e-10, load_increment * 1e-5):
        raise ValueError(
            f"Replay start load {actual_start_load} does not match "
            f"target-load increment {expected_start_load}."
        )
    macro_path = args.source_job / "macroData.csv"
    if not macro_path.is_file():
        raise FileNotFoundError(f"Missing source macro data: {macro_path}")
    macro = read_macro_data(macro_path)
    rows = macro[macro["load"].sub(args.target_load).abs() <= max(1e-10, load_increment * 1e-5)]
    if len(rows) != 1:
        raise RuntimeError(f"Could not uniquely validate target load {args.target_load}.")
    row = rows.iloc[0]
    m3_column = next(
        (column for column in ("nr_elements_with_m3_change", "nr_elements_with_m3_fix_change")
         if column in row.index),
        None,
    )
    if m3_column is None:
        raise KeyError("Source macro data has no forward m3-change column.")
    source_m3 = int(row[m3_column])
    if args.expected_event_kind == "elastic" and source_m3 != 0:
        raise ValueError("Selected elastic target has a forward m3 change.")
    if args.expected_event_kind == "plastic" and source_m3 <= 0:
        raise ValueError("Selected plastic target has no forward m3 change.")

    replay_macro = read_macro_data(replay_macro_path(event_directory))
    replay_rows = replay_macro[
        replay_macro["load"].sub(args.target_load).abs()
        <= max(1e-10, load_increment * 1e-5)
    ]
    if len(replay_rows) != 1:
        raise RuntimeError(
            f"Could not uniquely validate replay target load {args.target_load}."
        )
    replay_row = replay_rows.iloc[0]
    replay_m3_column = next(
        (column for column in ("nr_elements_with_m3_change", "nr_elements_with_m3_fix_change")
         if column in replay_row.index),
        None,
    )
    if replay_m3_column is None:
        raise KeyError("Replay macro data has no forward m3-change column.")
    replay_m3 = int(replay_row[replay_m3_column])
    if args.expected_event_kind == "elastic" and replay_m3 != 0:
        raise ValueError("Replayed elastic target has a forward m3 change.")
    if args.expected_event_kind == "plastic" and replay_m3 <= 0:
        raise ValueError("Replayed plastic target has no forward m3 change.")
    for column, relative_tolerance, absolute_tolerance in (
        ("total_energy", 1e-8, 1e-8),
        ("total_e_change_from_init", 5e-2, 1e-9),
    ):
        if column not in row.index or column not in replay_row.index:
            raise KeyError(f"Cannot validate replay without {column!r}.")
        if not np.isclose(
            float(row[column]),
            float(replay_row[column]),
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ):
            raise ValueError(
                f"Replay target {column}={float(replay_row[column]):.12g} does "
                f"not reproduce source value {float(row[column]):.12g}."
            )
    stress_matches = np.isclose(
        float(row["avg_sigma12"]),
        float(replay_row["avg_sigma12"]),
        rtol=1e-8,
        atol=1e-10,
    )
    if not stress_matches:
        message = (
            f"Replay target avg_sigma12={float(replay_row['avg_sigma12']):.12g} "
            f"does not reproduce source value {float(row['avg_sigma12']):.12g}."
        )
        if not args.allow_stress_diagnostic_mismatch:
            raise ValueError(message)
        warnings.warn(
            message + " Accepting only because the explicit diagnostic-mismatch "
            "override was supplied; corrected-drop metadata must come from the "
            "original macro row.",
            stacklevel=2,
        )
    if "is_reversible" in replay_row.index and int(replay_row["is_reversible"]) != 1:
        raise ValueError("Replay target failed the reversibility test.")


def validate_saved_events(args: ReplayArguments, target_directory: Path) -> Path:
    """Validate the target and every capped elastic save, then write a manifest."""

    validate_replay(args, target_directory)
    increment = float(read_config_values(args.source_job / "config.conf")["loadIncrement"])
    records = [{
        "event_kind": args.expected_event_kind,
        "target_load": args.target_load,
        "event_directory": str(target_directory),
    }]
    elastic_directories = sorted(
        path for path in args.output_directory.rglob("elastic_replay_l_*")
        if path.is_dir()
    )
    for event_directory in elastic_directories:
        elastic_args = ReplayArguments(
            source_job=args.source_job,
            dump=args.dump,
            target_load=event_start_load(event_directory) + increment,
            output_directory=args.output_directory,
            mts2d_binary=args.mts2d_binary,
            expected_event_kind="elastic",
            maximum_elastic_events=args.maximum_elastic_events,
            allow_stress_diagnostic_mismatch=args.allow_stress_diagnostic_mismatch,
        )
        validate_replay(elastic_args, event_directory)
        records.append({
            "event_kind": "elastic",
            "target_load": elastic_args.target_load,
            "event_directory": str(event_directory),
        })
    manifest = args.output_directory / "replay_manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--target-load", type=float, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--mts2d-binary", type=Path, required=True)
    parser.add_argument(
        "--expected-event-kind", choices=("elastic", "plastic"), default="elastic"
    )
    parser.add_argument("--maximum-elastic-events", type=int, default=1)
    parser.add_argument("--allow-stress-diagnostic-mismatch", action="store_true")
    return parser


def main() -> int:
    ns = build_parser().parse_args()
    args = ReplayArguments(
        source_job=ns.source_job,
        dump=ns.dump,
        target_load=ns.target_load,
        output_directory=ns.output_directory,
        mts2d_binary=ns.mts2d_binary,
        expected_event_kind=ns.expected_event_kind,
        maximum_elastic_events=ns.maximum_elastic_events,
        allow_stress_diagnostic_mismatch=ns.allow_stress_diagnostic_mismatch,
    )
    config = prepare_replay_configuration(args)
    event_directory = run_replay(args, config)
    manifest = validate_saved_events(args, event_directory)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
