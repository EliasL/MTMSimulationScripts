#!/usr/bin/env python3
"""Prepare the 40-interval FIRE/CG/LBFGS comparison campaign on Magi."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shlex
from pathlib import Path

from Management.configGenerator import SimulationConfig
from Management.queueLocalJobs import queue_local_jobs


ALGORITHMS = ("LBFGS", "CG", "FIRE")
LOAD_INCREMENT = 1e-5
EPSX = 1e-6
FIRE_EPS = 2.09264333e-5
THREADS = 2
JOB_COPIES = 20
TIME_LIMIT = "3-00:00:00"
MEMORY = "4G"
LOAD_PATTERN = re.compile(r"<load>([-+0-9.eE]+)</load>")


def dump_load(path: Path) -> float:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            match = LOAD_PATTERN.search(line)
            if match:
                return float(match.group(1))
    raise RuntimeError(f"Could not find load in {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_boundaries(dump_dir: Path, load_manifest: Path | None = None) -> list[dict]:
    cached_loads = None
    if load_manifest is not None:
        cached_loads = json.loads(load_manifest.read_text())
        if set(cached_loads) != {f"{index / 10:.1f}" for index in range(10, 51)}:
            raise RuntimeError(f"Unexpected load manifest contents: {load_manifest}")
    boundaries = []
    for index in range(10, 51):
        nominal = index / 10
        path = dump_dir / f"dump_l{nominal:.1f}.xml.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        boundaries.append(
            {
                "nominal_load": nominal,
                "actual_load": (
                    float(cached_loads[f"{nominal:.1f}"])
                    if cached_loads is not None
                    else dump_load(path)
                ),
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if any(
        right["actual_load"] <= left["actual_load"]
        for left, right in zip(boundaries, boundaries[1:])
    ):
        raise RuntimeError("Dump loads are not strictly increasing")
    return boundaries


def segment_config(start: float, stop: float, algorithm: str, name: str) -> SimulationConfig:
    if algorithm not in ALGORITHMS:
        raise ValueError(algorithm)
    config = SimulationConfig(
        rows=200,
        cols=200,
        usingPBC="true",
        reconnectionMethod="edgeFlip",
        reconnectRevert=1,
        reconnectEdgeLocking=0,
        experiment="simpleShear",
        nrThreads=THREADS,
        seed=0,
        startLoad=start,
        loadIncrement=LOAD_INCREMENT,
        maxLoad=stop,
        minimizer=algorithm,
        epsR=1e-20,
        LBFGSEpsg=1e-15,
        LBFGSEpsf=0.0,
        LBFGSEpsx=EPSX if algorithm == "LBFGS" else 0.0,
        CGEpsg=1e-15,
        CGEpsf=0.0,
        CGEpsx=EPSX if algorithm == "CG" else 0.0,
        eps=FIRE_EPS if algorithm == "FIRE" else 1e-15,
        FIREEpsx=0.0,
        maxIt=1_000_000,
        logDuringMinimization=0,
        fullMinimizationLogging=0,
        writeDumps=1,
        writeMeshVTUs=1,
        writeDebugVTUs=0,
        nrVTUFrames=5,
    )
    config.name = name
    config.validate_threshold()
    return config


def prepare(
    dump_dir: Path,
    campaign_root: Path,
    output_root: Path,
    binary: Path,
    submit: bool,
    load_manifest: Path | None = None,
    algorithms: tuple[str, ...] = ALGORITHMS,
) -> dict:
    algorithms = tuple(algorithms)
    if not algorithms or len(set(algorithms)) != len(algorithms):
        raise ValueError("Algorithms must be a non-empty unique sequence")
    if any(algorithm not in ALGORITHMS for algorithm in algorithms):
        raise ValueError(f"Unsupported algorithms: {algorithms}")
    boundaries = source_boundaries(dump_dir, load_manifest)
    config_dir = campaign_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    commands = []
    job_names = []
    runner = Path.home() / "simulation/SimulationScripts/ClusterJobs/run_minimizer_segment.py"

    for segment_index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        for algorithm in algorithms:
            name = f"mincmp-s{segment_index:02d}-{algorithm}"
            config = segment_config(
                left["actual_load"], right["actual_load"], algorithm, name
            )
            config_path = Path(
                config.write_to_file(config_dir, filename=f"{name}.conf")
            ).resolve()
            command = " ".join(
                shlex.quote(str(value))
                for value in (
                    "python3",
                    runner,
                    "--binary",
                    binary,
                    "--config",
                    config_path,
                    "--source-dump",
                    left["path"],
                    "--output-root",
                    output_root,
                    "--name",
                    name,
                    "--target-load",
                    f"{right['actual_load']:.17g}",
                )
            )
            jobs.append(
                {
                    "segment": segment_index,
                    "algorithm": algorithm,
                    "name": name,
                    "source_dump": left["path"],
                    "start_load": left["actual_load"],
                    "target_load": right["actual_load"],
                    "config": str(config_path),
                    "output": str((output_root / name).resolve()),
                }
            )
            commands.append(command)
            job_names.append(name)

    manifest = {
        "boundaries": boundaries,
        "jobs": jobs,
        "settings": {
            "loadIncrement": LOAD_INCREMENT,
            "epsx": EPSX,
            "fireEps": FIRE_EPS,
            "threads": THREADS,
            "jobCopies": JOB_COPIES,
            "timeLimit": TIME_LIMIT,
            "memory": MEMORY,
            "algorithms": algorithms,
            "reconnectionMethod": "edgeFlip",
            "reconnectRevert": 1,
            "reconnectEdgeLocking": 0,
            "nrVTUFramesPerSegment": 5,
        },
        "mts2dVersion": SimulationConfig.get_git_tag_of_MTS2D(),
        "mts2dBinarySha256": sha256(binary),
        "sourceLoadManifest": str(load_manifest.resolve()) if load_manifest else None,
        "submitted": False,
        "submissionJobIds": [],
    }
    manifest_path = campaign_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    expected_jobs = 40 * len(algorithms)
    if len(jobs) != expected_jobs or len({job["name"] for job in jobs}) != expected_jobs:
        raise RuntimeError(
            f"Campaign must contain exactly {expected_jobs} unique jobs"
        )
    submitted_job_ids = queue_local_jobs(
        commands,
        job_names,
        jobCopies=JOB_COPIES,
        time_limit=TIME_LIMIT,
        nrThreads=THREADS,
        memory=MEMORY,
        dryRun=not submit,
    )
    if submit:
        if len(submitted_job_ids) != len(jobs) * JOB_COPIES:
            raise RuntimeError(
                f"Expected {len(jobs) * JOB_COPIES} submitted copies, "
                f"got {len(submitted_job_ids)}"
            )
        manifest["submitted"] = True
        manifest["submissionJobIds"] = submitted_job_ids
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path.home() / "simulation/minimizer_segment_comparison",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "simulation/MTS2D_output",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path.home() / "simulation/MTS2D/build-release/MTS2D",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=list(ALGORITHMS),
    )
    parser.add_argument(
        "--load-manifest",
        type=Path,
        default=Path(__file__).with_name("minimizer_segment_loads.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = prepare(
        arguments.dump_dir.resolve(),
        arguments.campaign_root.resolve(),
        arguments.output_root.resolve(),
        arguments.binary.resolve(),
        arguments.submit,
        arguments.load_manifest.resolve(),
        tuple(arguments.algorithms),
    )
    print(
        f"{'SUBMITTED' if arguments.submit else 'PREPARED'} "
        f"{len(result['jobs'])} jobs with {JOB_COPIES} copies each"
    )
