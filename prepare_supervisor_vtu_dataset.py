#!/usr/bin/env python3
"""Collect size-scaling dumps or evenly spaced VTUs for a supervisor dataset.

The script assumes that the simulations have already been run.  It selects one
matching MTS2D job for every requested ``(size, seed)`` pair and classifies its
files using the maximum of ``avg_sigma12`` in ``macroData.csv`` (or the legacy
``avg_sigmaxy`` column when that is the schema in the source file).  Dump mode
copies every valid dump; VTU mode copies a fixed number of meshes from each
side of that split.

By default, the collector targets the non-reconnecting ``simpleShear``
size-scaling campaign.  Other campaigns can be selected with the command-line
options.

Cluster transfer is opt-in with ``--download``.  No cluster command is run by
importing this module.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SIZES = (50, 100, 150, 200)
SIZE_SCALING_THREADS = {50: 2, 100: 3, 150: 4, 200: 8}
DEFAULT_REMOTE_SOURCE_MAP = {
    "elundheim@schwartz.pmmh-cluster.espci.fr:/data2/elundheim/MTS2D_output":
        (50, 100, 150),
    "elundheim@condorcet.pmmh-cluster.espci.fr:/data/elundheim/MTS2D_output":
        (50,),
    "elundheim@poincare.pmmh-cluster.espci.fr:/data/elundheim/MTS2D_output":
        (150, 200),
    "elundheim@pascal.pmmh-cluster.espci.fr:/data/elundheim/MTS2D_output":
        (200,),
}
LOAD_PATTERN = re.compile(
    r"(?:^|_)load=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
DUMP_LOAD_PATTERN = re.compile(
    r"(?:^|_)l(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE
)
DUMP_SUFFIXES = (".xml.gz",)


@dataclass(frozen=True)
class Job:
    size: int
    seed: int
    folder: Path
    config: Path
    macro_data: Path


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be a non-empty list of positive integers")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("sizes must not contain duplicates")
    return sizes


def read_config(path: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if key in settings:
            raise ValueError(f"Duplicate setting {key!r} in {path}:{line_number}")
        settings[key] = value.strip()
    return settings


def required_setting(settings: dict[str, str], key: str, path: Path) -> str:
    try:
        return settings[key]
    except KeyError as error:
        raise KeyError(f"Missing {key!r} in {path}") from error


def setting_int(settings: dict[str, str], key: str, path: Path) -> int:
    try:
        return int(required_setting(settings, key, path))
    except ValueError as error:
        raise ValueError(f"Invalid integer setting {key!r} in {path}") from error


def setting_float(settings: dict[str, str], key: str, path: Path) -> float:
    try:
        value = float(required_setting(settings, key, path))
    except ValueError as error:
        raise ValueError(f"Invalid floating-point setting {key!r} in {path}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite setting {key!r} in {path}")
    return value


def close_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def remote_job_glob(args: argparse.Namespace, size: int, remote_source: str) -> str:
    """Return a narrow glob for the standard non-reconnecting size campaign."""
    if (
        args.experiment == "simpleShear"
        and args.reconnection == "none"
        and close_float(args.start_load, 0.15)
        and close_float(args.max_load, 1.0)
        and close_float(args.load_increment, 1e-5)
        and size in SIZE_SCALING_THREADS
    ):
        threads = SIZE_SCALING_THREADS[size]
        return (
            f"{remote_source.rstrip('/')}/"
            f"simpleShear,s{size}x{size}l0.15,1e-05,1.0"
            f"PBCt{threads}LBFGSEpsx1e-06s*"
        )
    return (
        f"{remote_source.rstrip('/')}/"
        f"{args.experiment},s{size}x{size}*"
    )


def remote_sources_for_size(args: argparse.Namespace, size: int) -> list[str]:
    if args.remote_sources is not None:
        return args.remote_sources
    return [
        source
        for source, sizes in DEFAULT_REMOTE_SOURCE_MAP.items()
        if size in sizes
    ]


def discover_jobs(
    source_root: Path,
    *,
    sizes: Iterable[int],
    seed_start: int,
    seed_count: int,
    experiment: str,
    reconnection: str,
    start_load: float,
    max_load: float,
    load_increment: float,
) -> list[Job]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing source root: {source_root}")

    jobs: list[Job] = []
    for size in sizes:
        for seed in range(seed_start, seed_start + seed_count):
            matches: list[Job] = []
            for folder in sorted(source_root.iterdir()):
                if not folder.is_dir():
                    continue
                if not folder.name.startswith(f"{experiment},s{size}x{size}"):
                    continue
                config = folder / "config.conf"
                if not config.is_file():
                    raise FileNotFoundError(f"Missing config.conf: {config}")
                settings = read_config(config)
                # MTS2D historically called this setting ``scenario``.  The
                # executable still accepts that legacy spelling.
                actual_experiment = settings.get("experiment", settings.get("scenario"))
                if actual_experiment is None:
                    raise KeyError(f"Missing 'experiment' (or legacy 'scenario') in {config}")
                if actual_experiment != experiment:
                    continue
                if setting_int(settings, "rows", config) != size:
                    continue
                if setting_int(settings, "cols", config) != size:
                    continue
                if setting_int(settings, "seed", config) != seed:
                    continue
                if required_setting(settings, "reconnectionMethod", config) != reconnection:
                    continue
                if not close_float(setting_float(settings, "startLoad", config), start_load):
                    continue
                if not close_float(setting_float(settings, "maxLoad", config), max_load):
                    continue
                if not close_float(
                    setting_float(settings, "loadIncrement", config), load_increment
                ):
                    continue

                macro_data = folder / "macroData.csv"
                if not macro_data.is_file():
                    raise FileNotFoundError(f"Missing macroData.csv: {macro_data}")
                matches.append(Job(size, seed, folder, config, macro_data))

            if not matches:
                raise FileNotFoundError(
                    f"No matching {experiment} job for size={size}, seed={seed}, "
                    f"reconnection={reconnection}, startLoad={start_load:g}, "
                    f"loadIncrement={load_increment:g}, maxLoad={max_load:g}"
                )
            if len(matches) > 1:
                names = ", ".join(str(job.folder) for job in matches)
                raise RuntimeError(
                    f"More than one matching job for size={size}, seed={seed}: {names}. "
                    "Tighten the campaign settings before collecting data."
                )
            jobs.append(matches[0])
    return jobs


def read_yield_load(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames is None or "load" not in rows.fieldnames:
            raise KeyError(f"{path} has no 'load' column")
        stress_column = next(
            (
                column
                for column in ("avg_sigma12", "avg_sigmaxy")
                if column in rows.fieldnames
            ),
            None,
        )
        if stress_column is None:
            raise KeyError(f"{path} has neither 'avg_sigma12' nor legacy 'avg_sigmaxy'")
        if stress_column != "avg_sigma12":
            print(f"Warning: using legacy {stress_column} as yield proxy for {path}")

        best_stress = -math.inf
        best_load: float | None = None
        for row_number, row in enumerate(rows, 2):
            if row.get("load") == "load":
                # Some long/resumed macro files append their #HEADER line
                # before continuing the data rows.
                continue
            try:
                load = float(row["load"])
                stress = float(row[stress_column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid load/stress in {path}:{row_number}") from error
            if not math.isfinite(load) or not math.isfinite(stress):
                raise ValueError(f"Non-finite load/stress in {path}:{row_number}")
            if stress > best_stress:
                best_stress = stress
                best_load = load

    if best_load is None:
        raise ValueError(f"{path} contains no data rows")
    return best_load


def vtu_load(path: Path) -> float:
    match = LOAD_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not read load from VTU filename: {path}")
    load = float(match.group(1))
    if not math.isfinite(load):
        raise ValueError(f"Non-finite load in VTU filename: {path}")
    return load


def load_vtus(job: Job, start_load: float, max_load: float) -> list[tuple[float, Path]]:
    records = []
    for path in sorted((job.folder / "data").glob("*.vtu")):
        load = vtu_load(path)
        if start_load - 1e-12 <= load <= max_load + 1e-12:
            records.append((load, path))
    records.sort(key=lambda item: (item[0], item[1].name))
    if not records:
        raise ValueError(f"No VTUs in the requested strain range for {job.folder}")
    return records


def is_dump_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.is_file()
        and not name.startswith((".", "broken_"))
        and not name.endswith(".tmp.xml")
        and name.endswith(DUMP_SUFFIXES)
    )


def dump_load(path: Path) -> float:
    match = DUMP_LOAD_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not read load from dump filename: {path}")
    load = float(match.group(1))
    if not math.isfinite(load):
        raise ValueError(f"Non-finite load in dump filename: {path}")
    return load


def load_dumps(job: Job) -> list[tuple[float, Path]]:
    dump_folder = job.folder / "dumps"
    if not dump_folder.is_dir():
        raise FileNotFoundError(f"Missing dump folder: {dump_folder}")
    paths = [path for path in dump_folder.iterdir() if is_dump_file(path)]
    if not paths:
        raise ValueError(f"No valid dump files found in {dump_folder}")
    records = [(dump_load(path), path) for path in paths]
    return sorted(records, key=lambda item: (item[0], item[1].name))


def select_evenly(records: list[tuple[float, Path]], count: int) -> list[tuple[float, Path]]:
    if count <= 0:
        raise ValueError("samples-per-regime must be positive")
    if len(records) < count:
        raise ValueError(
            f"Requested {count} meshes but only {len(records)} are available "
            f"between loads {records[0][0]:g} and {records[-1][0]:g}"
        )
    if count == 1:
        target_loads = [(records[0][0] + records[-1][0]) / 2]
    else:
        first_load, last_load = records[0][0], records[-1][0]
        target_loads = [
            first_load + i * (last_load - first_load) / (count - 1)
            for i in range(count)
        ]

    loads = [record[0] for record in records]
    selected: list[tuple[float, Path]] = []
    lower_index = 0
    for output_index, target in enumerate(target_loads):
        remaining = count - output_index - 1
        upper_index = len(records) - remaining - 1
        insertion = bisect.bisect_left(loads, target, lower_index, upper_index + 1)
        candidates = {
            max(lower_index, min(upper_index, insertion)),
            max(lower_index, min(upper_index, insertion - 1)),
            max(lower_index, min(upper_index, insertion + 1)),
        }
        chosen_index = min(candidates, key=lambda index: (abs(loads[index] - target), index))
        selected.append(records[chosen_index])
        lower_index = chosen_index + 1
    return selected


def format_load(load: float) -> str:
    return f"{load:.12g}"


def run_rsync(command: list[str], args: argparse.Namespace) -> None:
    if args.dry_run:
        command.append("--dry-run")
    print("$ " + shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


def download_candidates(args: argparse.Namespace) -> None:
    """Download the VTU inputs used by the original collector."""
    if not args.dry_run:
        args.source_root.parent.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        for remote_source in remote_sources_for_size(args, size):
            remote_glob = remote_job_glob(args, size, remote_source)
            command = [
                "rsync",
                "-a",
                "--progress",
                "--prune-empty-dirs",
                "--exclude=*sigmaRescue*",
                "--exclude=*/sigmaRescue*",
                "--exclude=backups/",
                "--exclude=*/backups/",
                "--include=*/",
                "--include=config.conf",
                "--include=*/config.conf",
                "--include=macroData.csv",
                "--include=*/macroData.csv",
                "--include=data/",
                "--include=data/*.vtu",
                "--include=*/data/*.vtu",
                "--exclude=*",
            ]
            command.extend([remote_glob, str(args.source_root) + "/"])
            run_rsync(command, args)


def download_dump_candidates(args: argparse.Namespace) -> None:
    """Download configs, macro data, and every dump file for each size."""
    if not args.dry_run:
        args.source_root.parent.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        for remote_source in remote_sources_for_size(args, size):
            remote_glob = remote_job_glob(args, size, remote_source)
            command = [
                "rsync",
                "-a",
                "--progress",
                "--prune-empty-dirs",
                "--exclude=broken_*",
                "--exclude=*/broken_*",
                "--exclude=*sigmaRescue*",
                "--exclude=*/sigmaRescue*",
                "--exclude=backups/",
                "--exclude=*/backups/",
                "--include=*/",
                "--include=config.conf",
                "--include=*/config.conf",
                "--include=macroData.csv",
                "--include=*/macroData.csv",
                "--include=dumps/",
                "--include=*/dumps/",
                "--include=dumps/*.xml.gz",
                "--include=*/dumps/*.xml.gz",
                "--exclude=*",
            ]
            command.extend([remote_glob, str(args.source_root) + "/"])
            run_rsync(command, args)


def write_manifest(root: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    fields = [
        "size",
        "seed",
        "regime",
        "load",
        "yield_load",
        "source_job",
        "source_vtu",
        "output_vtu",
    ]
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "artifact": "vtu",
        "sizes": list(args.sizes),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "samples_per_regime": args.samples_per_regime,
        "reconnection": args.reconnection,
        "experiment": args.experiment,
        "start_load": args.start_load,
        "max_load": args.max_load,
        "load_increment": args.load_increment,
        "mesh_count": len(rows),
    }
    (root / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def write_dump_manifest(
    root: Path, rows: list[dict[str, object]], args: argparse.Namespace
) -> None:
    fields = [
        "size",
        "seed",
        "regime",
        "load",
        "yield_load",
        "source_job",
        "source_dump",
        "output_dump",
    ]
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "artifact": "dump",
        "sizes": list(args.sizes),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "reconnection": args.reconnection,
        "experiment": args.experiment,
        "start_load": args.start_load,
        "max_load": args.max_load,
        "load_increment": args.load_increment,
        "dump_count": len(rows),
    }
    (root / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def prepare_dataset(args: argparse.Namespace) -> None:
    jobs = discover_jobs(
        args.source_root,
        sizes=args.sizes,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        experiment=args.experiment,
        reconnection=args.reconnection,
        start_load=args.start_load,
        max_load=args.max_load,
        load_increment=args.load_increment,
    )
    print(f"Found {len(jobs)} source jobs; planning {len(jobs) * args.samples_per_regime * 2} VTUs.")

    if args.dry_run:
        for job in jobs:
            print(f"  {job.size=} {job.seed=}: {job.folder}")
        return

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output folder: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    manifest_rows: list[dict[str, object]] = []
    try:
        for job in jobs:
            yield_load = (
                args.yield_load if args.yield_load is not None else read_yield_load(job.macro_data)
            )
            records = load_vtus(job, args.start_load, args.max_load)
            regimes = {
                "pre_yield": [record for record in records if record[0] <= yield_load],
                "post_yield": [record for record in records if record[0] > yield_load],
            }
            seed_root = staging_root / f"L{job.size}x{job.size}" / f"seed{job.seed:03d}"
            seed_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(job.config, seed_root / "source_config.conf")

            for regime, regime_records in regimes.items():
                selected = select_evenly(regime_records, args.samples_per_regime)
                regime_root = seed_root / regime
                regime_root.mkdir()
                for index, (load, source_vtu) in enumerate(selected):
                    destination = regime_root / (
                        f"mesh_{index:03d}_load_{format_load(load)}.vtu"
                    )
                    shutil.copy2(source_vtu, destination)
                    manifest_rows.append(
                        {
                            "size": job.size,
                            "seed": job.seed,
                            "regime": regime,
                            "load": load,
                            "yield_load": yield_load,
                            "source_job": str(job.folder),
                            "source_vtu": str(source_vtu),
                            "output_vtu": str(destination.relative_to(staging_root)),
                        }
                    )

        write_manifest(staging_root, manifest_rows, args)
        staging_root.replace(output_root)
    except Exception:
        shutil.rmtree(staging_root)
        raise
    print(f"Wrote {len(manifest_rows)} VTUs to {output_root}")


def prepare_dump_dataset(args: argparse.Namespace) -> None:
    jobs = discover_jobs(
        args.source_root,
        sizes=args.sizes,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        experiment=args.experiment,
        reconnection=args.reconnection,
        start_load=args.start_load,
        max_load=args.max_load,
        load_increment=args.load_increment,
    )
    print(f"Found {len(jobs)} source jobs; collecting every valid dump.")

    if args.dry_run:
        for job in jobs:
            print(f"  {job.size=} {job.seed=}: {job.folder}")
        return

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output folder: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    manifest_rows: list[dict[str, object]] = []
    try:
        for job in jobs:
            yield_load = (
                args.yield_load
                if args.yield_load is not None
                else read_yield_load(job.macro_data)
            )
            records = load_dumps(job)
            regimes = {
                "pre_yield": [record for record in records if record[0] <= yield_load],
                "post_yield": [record for record in records if record[0] > yield_load],
            }
            missing_regimes = [regime for regime, items in regimes.items() if not items]
            if missing_regimes:
                print(
                    f"Warning: no dump files in {', '.join(missing_regimes)} for "
                    f"{job.folder}; yield load is {yield_load:g}"
                )

            seed_root = staging_root / f"L{job.size}x{job.size}" / f"seed{job.seed:03d}"
            seed_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(job.macro_data, seed_root / "macroData.csv")
            shutil.copy2(job.config, seed_root / "source_config.conf")

            for regime, regime_records in regimes.items():
                regime_root = seed_root / regime
                dump_root = regime_root / "dumps"
                dump_root.mkdir(parents=True, exist_ok=True)
                # MTS2D searches for config.conf in the parent of dumps/ when
                # the -c option is omitted.
                shutil.copy2(job.config, regime_root / "config.conf")
                for load, source_dump in regime_records:
                    destination = dump_root / source_dump.name
                    shutil.copy2(source_dump, destination)
                    manifest_rows.append(
                        {
                            "size": job.size,
                            "seed": job.seed,
                            "regime": regime,
                            "load": load,
                            "yield_load": yield_load,
                            "source_job": str(job.folder),
                            "source_dump": str(source_dump),
                            "output_dump": str(destination.relative_to(staging_root)),
                        }
                    )

        write_dump_manifest(staging_root, manifest_rows, args)
        staging_root.replace(output_root)
    except Exception:
        shutil.rmtree(staging_root)
        raise
    print(f"Wrote {len(manifest_rows)} dumps to {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        choices=("dumps", "vtu"),
        default="dumps",
        help="collect all dumps (default) or use the original VTU collector",
    )
    parser.add_argument("--source-root", type=Path, default=Path("/Volumes/data/MTS2D_output"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--sizes", type=parse_sizes, default=DEFAULT_SIZES)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument(
        "--samples-per-regime",
        type=int,
        default=100,
        help="VTU meshes per regime; ignored for dump collection",
    )
    parser.add_argument("--reconnection", choices=("none", "edgeFlip", "delaunay"), default="none")
    parser.add_argument("--experiment", default="simpleShear")
    parser.add_argument("--start-load", type=float, default=0.15)
    parser.add_argument("--max-load", type=float, default=1.0)
    parser.add_argument("--load-increment", type=float, default=1e-5)
    parser.add_argument("--yield-load", type=float, help="Override the per-job stress-maximum split")
    parser.add_argument("--download", action="store_true", help="Download the selected artifact with rsync first")
    parser.add_argument(
        "--remote-source",
        dest="remote_sources",
        action="append",
        help="Override the default server roots; repeat for multiple roots",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without downloading or writing files")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_root is None:
        args.output_root = Path(
            "supervisor_dump_dataset"
            if args.artifact == "dumps"
            else "supervisor_vtu_dataset"
        )
    if args.seed_start < 0 or args.seed_count <= 0:
        raise ValueError("seed-start must be non-negative and seed-count must be positive")
    if args.start_load >= args.max_load:
        raise ValueError("start-load must be smaller than max-load")
    if args.load_increment <= 0:
        raise ValueError("load-increment must be positive")
    if args.yield_load is not None and not math.isfinite(args.yield_load):
        raise ValueError("yield-load must be finite")
    if args.download:
        if args.artifact == "dumps":
            download_dump_candidates(args)
        else:
            download_candidates(args)
        if args.dry_run:
            return
    if args.artifact == "dumps":
        prepare_dump_dataset(args)
    else:
        prepare_dataset(args)


if __name__ == "__main__":
    main()
