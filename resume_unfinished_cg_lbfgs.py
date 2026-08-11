#!/usr/bin/env python3
"""Download and resume unfinished 200x200 simulations.

Jobs are downloaded sequentially within each batch, then run concurrently.
The next batch is not started until all processes in the current batch have
exited. Re-running the script keeps local progress and skips completed jobs.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from typing import BinaryIO


DEFAULT_HOST = "schwartz.pmmh-cluster.espci.fr"
DEFAULT_REMOTE_ROOT = "/data2/elundheim/MTS2D_output"
DEFAULT_LOCAL_ROOT = Path("/Volumes/data/MTS2D_local_resume")
DEFAULT_BINARY = Path("/Users/eliaslundheim/work/PhD/MTS2D/build-release/MTS2D")

JOB_TEMPLATES = {
    "CG": (
        "simpleShear,s200x200l0.15,1e-05,1.0PBCt3minimizerCG"
        "LBFGSEpsg1e-05CGEpsg1e-05eps1e-05s{seed}"
    ),
    "LBFGS": (
        "simpleShear,s200x200l0.15,1e-05,1.0PBCt3"
        "LBFGSEpsg1e-05CGEpsg1e-05eps1e-05s{seed}"
    ),
}


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("Empty entry in --seeds")
        if "-" in part:
            first_text, last_text = part.split("-", 1)
            first = int(first_text)
            last = int(last_text)
            if last < first:
                raise ValueError(f"Invalid descending seed range: {part}")
            seeds.extend(range(first, last + 1))
        else:
            seeds.append(int(part))
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seed in --seeds")
    return seeds


def parse_algorithms(value: str) -> list[str]:
    algorithms = [part.strip().upper() for part in value.split(",")]
    if any(not algorithm for algorithm in algorithms):
        raise ValueError("Empty entry in --algorithms")
    if any(algorithm not in JOB_TEMPLATES for algorithm in algorithms):
        raise ValueError(
            f"Algorithms must be selected from {sorted(JOB_TEMPLATES)}, got {value!r}"
        )
    if len(algorithms) != len(set(algorithms)):
        raise ValueError("Duplicate algorithm in --algorithms")
    return algorithms


def job_name(algorithm: str, seed: int) -> str:
    try:
        return JOB_TEMPLATES[algorithm].format(seed=seed)
    except KeyError as exc:
        raise ValueError(f"Unknown algorithm: {algorithm}") from exc


def read_config_float(config_path: Path, key: str) -> float:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^#\s]+)", re.MULTILINE)
    text = config_path.read_text()
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {key!r} setting in {config_path}, found {len(matches)}"
        )
    return float(matches[0])


def set_config_threads(config_path: Path, threads: int) -> None:
    if threads <= 0:
        raise ValueError("Thread count must be positive")
    backup_path = config_path.with_name("config.cluster.conf")
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)

    text = config_path.read_text()
    updated, count = re.subn(
        r"^\s*nrThreads\s*=\s*[^#\s]+",
        f"nrThreads = {threads}",
        text,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one nrThreads setting in {config_path}, found {count}"
        )
    config_path.write_text(updated)


def last_csv_load(csv_path: Path) -> float:
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"Empty CSV file: {csv_path}") from exc

        normalized = [column.strip().lower() for column in header]
        if "load" not in normalized:
            raise RuntimeError(f"No load column in {csv_path}")
        load_index = normalized.index("load")
        last_value: float | None = None
        for row in reader:
            if not row:
                continue
            if len(row) <= load_index:
                raise RuntimeError(f"Malformed row in {csv_path}: {row!r}")
            last_value = float(row[load_index])

    if last_value is None:
        raise RuntimeError(f"CSV contains a header but no data: {csv_path}")
    return last_value


def is_complete(job_dir: Path) -> tuple[bool, float, float]:
    config_path = job_dir / "config.conf"
    csv_path = job_dir / "macroData.csv"
    if not config_path.is_file() or not csv_path.is_file():
        return False, float("nan"), float("nan")
    max_load = read_config_float(config_path, "maxLoad")
    load_increment = read_config_float(config_path, "loadIncrement")
    current_load = last_csv_load(csv_path)
    return current_load + load_increment / 2 >= max_load, current_load, max_load


def latest_dump(job_dir: Path) -> Path:
    dump_dir = job_dir / "dumps"
    if not dump_dir.is_dir():
        raise FileNotFoundError(f"Missing dump directory: {dump_dir}")
    candidates = [
        path
        for path in dump_dir.iterdir()
        if path.is_file()
        and (path.name.endswith(".xml.gz") or path.name.endswith(".mtsb"))
        and not path.name.startswith("crash_")
    ]
    if not candidates:
        raise FileNotFoundError(f"No restart dump found in {dump_dir}")

    load_pattern = re.compile(
        r"^dump_l([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\.(?:xml\.gz|mtsb)$"
    )
    dumps_by_load: list[tuple[float, Path]] = []
    for path in candidates:
        match = load_pattern.match(path.name)
        if match is None:
            raise RuntimeError(f"Unexpected restart dump filename: {path}")
        dumps_by_load.append((float(match.group(1)), path))
    return max(dumps_by_load, key=lambda item: item[0])[1]


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def rsync_job(args: argparse.Namespace, name: str) -> Path:
    local_dir = args.local_root / name
    marker = local_dir / ".cluster_download_complete"
    if args.skip_download:
        if not local_dir.is_dir():
            raise FileNotFoundError(
                f"--skip-download was given, but {local_dir} does not exist"
            )
        return local_dir
    if marker.is_file():
        print(f"  download already complete: {name}")
        return local_dir

    source = f"{args.user}@{args.host}:{args.remote_root}/{name}"
    command = [
        "rsync",
        "-azP",
        "--partial",
        "-e",
        "ssh -T",
        source,
        str(args.local_root),
    ]
    print(f"  download: {format_command(command)}")
    subprocess.run(command, check=True)
    if not local_dir.is_dir():
        raise RuntimeError(f"rsync succeeded but did not create {local_dir}")
    marker.touch()
    return local_dir


def select_binary(args: argparse.Namespace, dump_path: Path) -> Path:
    if dump_path.name.endswith(".mtsb"):
        if args.mtsb_binary is None:
            raise RuntimeError(
                f"{dump_path} is a binary .mtsb dump. Pass --mtsb-binary with a "
                "compatible historical executable; the current executable reads XML dumps."
            )
        return args.mtsb_binary
    return args.binary


def prepare_job(
    args: argparse.Namespace, algorithm: str, seed: int
) -> tuple[str, Path, Path, list[str]] | None:
    name = job_name(algorithm, seed)
    job_dir = rsync_job(args, name)
    complete, current_load, max_load = is_complete(job_dir)
    if complete:
        print(f"  {algorithm} seed {seed}: complete ({current_load:g}/{max_load:g})")
        (job_dir / ".local_resume_complete").touch()
        return None

    set_config_threads(job_dir / "config.conf", args.threads)
    dump_path = latest_dump(job_dir)
    binary = select_binary(args, dump_path)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"Executable not found or not executable: {binary}")

    command = [
        str(binary),
        "-d",
        str(dump_path),
        "-c",
        str(job_dir / "config.conf"),
        "-o",
        str(args.local_root),
    ]
    print(
        f"  {algorithm} seed {seed}: resume from {dump_path.name}; "
        f"CSV load={current_load:g}, maxLoad={max_load:g}"
    )
    return algorithm, job_dir, dump_path, command


def start_job(
    args: argparse.Namespace,
    prepared: tuple[str, Path, Path, list[str]],
) -> tuple[str, Path, subprocess.Popen[bytes], BinaryIO]:
    algorithm, job_dir, _, command = prepared
    log_path = job_dir / "local_resume.log"
    log_handle = log_path.open("ab", buffering=0)
    heading = (
        f"\n[{datetime.now().isoformat(timespec='seconds')}] "
        f"{format_command(command)}\n"
    )
    log_handle.write(heading.encode())
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(args.threads)
    process = subprocess.Popen(
        command,
        cwd=args.local_root,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    (job_dir / ".local_resume.pid").write_text(f"{process.pid}\n")
    print(f"  started {algorithm}: pid={process.pid}, log={log_path}")
    return algorithm, job_dir, process, log_handle


def wait_for_jobs(
    running: list[tuple[str, Path, subprocess.Popen[bytes], BinaryIO]],
) -> None:
    failures: list[str] = []
    try:
        for algorithm, job_dir, process, log_handle in running:
            return_code = process.wait()
            log_handle.close()
            (job_dir / ".local_resume.status").write_text(f"{return_code}\n")
            if return_code != 0:
                failures.append(f"{algorithm} exited with status {return_code}")
                continue
            complete, current_load, max_load = is_complete(job_dir)
            if not complete:
                failures.append(
                    f"{algorithm} exited successfully but stopped at "
                    f"{current_load:g}/{max_load:g}"
                )
                continue
            (job_dir / ".local_resume_complete").touch()
            print(f"  finished {algorithm}: {current_load:g}/{max_load:g}")
    except KeyboardInterrupt:
        print("\nStopping the active batch...", file=sys.stderr)
        for _, _, process, _ in running:
            if process.poll() is None:
                process.terminate()
        for _, _, process, log_handle in running:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log_handle.close()
        raise

    if failures:
        raise RuntimeError("; ".join(failures))


def print_dry_run(
    args: argparse.Namespace, algorithms: list[str], seeds: list[int]
) -> None:
    print("Dry run: no files will be downloaded or changed.\n")
    for batch_start in range(0, len(seeds), args.batch_size):
        batch = seeds[batch_start : batch_start + args.batch_size]
        print(f"Batch seeds {batch}:")
        for seed in batch:
            print(f"  Seed {seed}:")
            for algorithm in algorithms:
                name = job_name(algorithm, seed)
                source = f"{args.user}@{args.host}:{args.remote_root}/{name}"
                print(f"    {algorithm}: rsync {source} -> {args.local_root}")
        labels = ", ".join(algorithms)
        print(f"  then run {labels} concurrently; wait before the next batch")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,2,5-7")
    parser.add_argument(
        "--algorithms",
        default="CG,LBFGS",
        help="comma-separated algorithms, e.g. CG or CG,LBFGS",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="number of seeds to run concurrently",
    )
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--user", default="elundheim")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument(
        "--mtsb-binary",
        type=Path,
        help="historical executable to use only when the latest dump is .mtsb",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    algorithms = parse_algorithms(args.algorithms)
    seeds = parse_seeds(args.seeds)
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.dry_run:
        print_dry_run(args, algorithms, seeds)
        return 0

    args.local_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(args.local_root).free / 1024**3
    print(f"Local root: {args.local_root} ({free_gib:.1f} GiB free)")
    if free_gib < args.minimum_free_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB free; require at least "
            f"{args.minimum_free_gib:.1f} GiB"
        )

    lock_path = args.local_root / ".resume_unfinished_cg_lbfgs.lock"
    with lock_path.open("w") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another batch driver holds {lock_path}") from exc
        lock_handle.write(f"{os.getpid()}\n")
        lock_handle.flush()

        for batch_start in range(0, len(seeds), args.batch_size):
            batch = seeds[batch_start : batch_start + args.batch_size]
            print(f"\n=== Batch seeds {batch} ===")
            prepared = []
            for seed in batch:
                for algorithm in algorithms:
                    job = prepare_job(args, algorithm, seed)
                    if job is not None:
                        prepared.append(job)
            if not prepared:
                print("  all simulations in this batch are already complete")
                continue
            running = [start_job(args, job) for job in prepared]
            wait_for_jobs(running)

    print("\nAll requested simulations are complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
