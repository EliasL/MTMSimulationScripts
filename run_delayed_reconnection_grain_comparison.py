#!/usr/bin/env python3
"""Branch a non-reconnecting state and enable edge flips for one load step.

The source run has no exact restart dump near the grain comparison at
gamma=0.69859.  This script first replays the non-reconnecting run to
gamma=0.69858, then advances two branches to gamma=0.69859: one with the
original topology and one with the standard edge-flip/revert loop enabled.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
from pathlib import Path

from Management.configGenerator import SimulationConfig


ROOT = Path(__file__).resolve().parent
MTS2D_ROOT = ROOT.parent / "MTS2D"
SOURCE_RUN = Path(
    "/Volumes/data/MTS2D_output/"
    "reversibilityProtocolTest,s200x200l0.15,1e-05,2.0PBCt3epsR0.0"
    "LBFGSEpsg0.0LBFGSEpsx1e-06s0"
)
SOURCE_CONFIG = SOURCE_RUN / "config.conf"
SOURCE_DUMP = SOURCE_RUN / "dumps/dump_l0.6.xml.gz"
REFERENCE_VTU = SOURCE_RUN / "data/" / (
    "reversibilityProtocolTest,s200x200l0.15,1e-05,2.0PBCt3epsR0.0"
    "LBFGSEpsg0.0LBFGSEpsx1e-06s0_load=0.69859_nrM=43_.54860.vtu"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/data/Lundheim_MTM_PhD_backup_2023_2026/2026/"
    "delayed_reconnection_gamma_0p69859/final_comparison"
)
CONFIG_DIRECTORY = ROOT / "output/delayed_reconnection_gamma_0p69859/configs"
LOAD_PATTERN = re.compile(r"_load=([-+0-9.eE]+)_")
XML_LOAD_PATTERN = re.compile(r"<load>([-+0-9.eE]+)</load>")


def serialized_load(path: Path) -> float:
    """Read the first serialized mesh load without parsing the full dump."""
    with gzip.open(path, "rt") as stream:
        for line in stream:
            match = XML_LOAD_PATTERN.search(line)
            if match is not None:
                return float(match.group(1))
    raise ValueError(f"No serialized <load> value found in {path}")


def make_config(
    source: Path,
    *,
    name: str,
    max_load: float,
    reconnection_method: str,
    experiment: str | None = None,
    lbfgs_nr_corrections: int = 8,
) -> tuple[SimulationConfig, Path]:
    config = SimulationConfig(source)
    config.name = name
    config.maxLoad = max_load
    config.reconnectionMethod = reconnection_method
    if experiment is not None:
        config.experiment = experiment
    config.reconnectRevert = 1
    config.reconnectEdgeLocking = 0
    config.minimum_transport_distance = 1
    config.LBFGSNrCorrections = lbfgs_nr_corrections
    config.nrVTUFrames = 1
    config.writeDumps = 1
    config.writeMeshVTUs = 1
    CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = Path(config.write_to_file(CONFIG_DIRECTORY, f"{name}.conf"))
    return config, path


def run_stage(
    binary: Path,
    dump: Path,
    config_path: Path,
    output_root: Path,
    *,
    make_dump_at: float | None = None,
) -> None:
    command = [
        str(binary),
        "-d",
        str(dump),
        "-c",
        str(config_path),
        "-o",
        str(output_root),
    ]
    if make_dump_at is not None:
        command.extend(("--makeDumpAt", f"{make_dump_at:.17g}"))
    subprocess.run(command, cwd=MTS2D_ROOT, check=True)


def find_dump_at(directory: Path, target: float, tolerance: float) -> Path:
    dumps = sorted(directory.glob("*.xml.gz"))
    matches = [path for path in dumps if abs(serialized_load(path) - target) <= tolerance]
    if len(matches) != 1:
        loads = {str(path): serialized_load(path) for path in dumps}
        raise RuntimeError(
            f"Expected one dump at gamma={target:.8f}, found {len(matches)}: {loads}"
        )
    return matches[0]


def find_vtu_at(directory: Path, target: float, tolerance: float) -> Path:
    matches = []
    for path in directory.glob("*.vtu"):
        match = LOAD_PATTERN.search(path.name)
        if match is not None and abs(float(match.group(1)) - target) <= tolerance:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one main VTU at gamma={target:.8f} below {directory}, "
            f"found {len(matches)}."
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--source-dump", type=Path, default=SOURCE_DUMP)
    parser.add_argument(
        "--branch-dump",
        type=Path,
        help="Use an already prepared dump at branch-load and skip reconstruction.",
    )
    parser.add_argument(
        "--binary", type=Path, default=MTS2D_ROOT / "build-release/MTS2D"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--branch-load", type=float, default=0.69858)
    parser.add_argument("--target-load", type=float, default=0.69859)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    required_paths = [args.source_config, args.source_dump, args.binary]
    if args.branch_dump is not None:
        required_paths.append(args.branch_dump)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.target_load <= args.branch_load:
        raise ValueError("target-load must be greater than branch-load.")

    source_config = SimulationConfig(args.source_config)
    tolerance = 0.5 * abs(source_config.loadIncrement)
    if abs(args.target_load - args.branch_load - source_config.loadIncrement) > 1e-12:
        raise ValueError("The branch and target loads must be one load increment apart.")
    source_load = serialized_load(args.branch_dump or args.source_dump)
    if args.branch_dump is None and source_load >= args.branch_load - tolerance:
        raise ValueError(
            f"Source dump load {source_load:.17g} is not before the branch load."
        )
    if args.branch_dump is not None and abs(source_load - args.branch_load) > tolerance:
        raise ValueError(
            f"Prepared branch dump is at {source_load:.17g}, expected {args.branch_load:.17g}."
        )
    if args.output_root.exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to reuse existing output root: {args.output_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=args.resume)

    reconnecting, reconnecting_config = make_config(
        args.source_config,
        name="delayedReconnectionEdgeFlip",
        max_load=args.target_load,
        reconnection_method="edgeFlip",
        experiment="simpleShear",
    )

    if args.branch_dump is None:
        reconstruct, reconstruct_config = make_config(
            args.source_config,
            name="delayedReconnectionReconstruct",
            max_load=args.branch_load,
            reconnection_method="none",
            experiment="simpleShear",
        )
        run_stage(
            args.binary,
            args.source_dump,
            reconstruct_config,
            args.output_root,
            make_dump_at=args.branch_load,
        )
        branch_dump = find_dump_at(
            args.output_root / reconstruct.name / "dumps", args.branch_load, tolerance
        )
    else:
        branch_dump = args.branch_dump
    reconnecting_directory = args.output_root / reconnecting.name
    if not reconnecting_directory.exists():
        run_stage(args.binary, branch_dump, reconnecting_config, args.output_root)
    elif not args.resume:
        raise FileExistsError(reconnecting_directory)

    retry_dumps = list((reconnecting_directory / "dumps").glob("crash_dump_*.xml.gz"))
    effective_corrections = 9 if retry_dumps else 8
    baseline, baseline_config = make_config(
        args.source_config,
        name=f"delayedReconnectionBaselineM{effective_corrections}",
        max_load=args.target_load,
        reconnection_method="none",
        experiment="simpleShear",
        lbfgs_nr_corrections=effective_corrections,
    )
    baseline_directory = args.output_root / baseline.name
    if not baseline_directory.exists():
        run_stage(args.binary, branch_dump, baseline_config, args.output_root)
    elif not args.resume:
        raise FileExistsError(baseline_directory)

    baseline_vtu = find_vtu_at(
        baseline_directory / "data", args.target_load, tolerance
    )
    reconnecting_vtu = find_vtu_at(
        reconnecting_directory / "data", args.target_load, tolerance
    )
    manifest = {
        "source_dump": str(args.source_dump),
        "source_dump_load": serialized_load(args.source_dump),
        "branch_dump": str(branch_dump),
        "branch_load": serialized_load(branch_dump),
        "target_load": args.target_load,
        "reference_vtu": str(REFERENCE_VTU),
        "baseline_vtu": str(baseline_vtu),
        "reconnecting_vtu": str(reconnecting_vtu),
        "settings": {
            "reconnectionMethod": "edgeFlip",
            "reconnectRevert": 1,
            "reconnectEdgeLocking": 0,
            "minimum_transport_distance": 1,
            "reconnectingConfiguredLBFGSNrCorrections": 8,
            "effectiveLBFGSNrCorrections": effective_corrections,
        },
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
