"""Hold queued user jobs while leaving sigma-rescue jobs untouched.

The default action is a dry run.  Only pending jobs are selected; running jobs
are deliberately left alone so that they can finish and release their cores.
Jobs whose name starts with ``sigR_`` are excluded from selection and holding.

Examples
--------
Inspect the Pascal queue:

    python -m Management.holdNonRescueJobs --server pascal

Apply the holds after reviewing the dry-run output:

    python -m Management.holdNonRescueJobs --server pascal --apply --yes

Release is intentionally a separate command so that resuming the jobs cannot
happen accidentally:

    python -m Management.holdNonRescueJobs --server pascal --release --yes
"""

from __future__ import annotations

import argparse
from collections import Counter
import re
import shlex
from dataclasses import dataclass

from .connectToCluster import Servers, connectToCluster


RESCUE_PREFIX = "sigR_"
JOB_ID_PATTERN = re.compile(r"^[0-9]+(?:_[0-9]+)?(?:\+[0-9]+)?$")
CHUNK_SIZE = 100
SERVER_NAMES = {
    name: getattr(Servers, name)
    for name in (
        "galois",
        "pascal",
        "schwartz",
        "lagrange",
        "condorcet",
        "dalembert",
        "poincare",
        "fourier",
        "descartes",
        "legendre",
        "duchemin",
        "cauchy",
    )
}


@dataclass(frozen=True)
class Job:
    job_id: str
    name: str
    state: str
    cpus: str
    reason: str
    elapsed: str
    time_left: str


def _remote_output(ssh, command: str) -> str:
    _, stdout, stderr = ssh.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()
    output = stdout.read().decode().strip()
    error = stderr.read().decode().strip()
    if exit_code != 0:
        raise RuntimeError(f"Remote command failed: {command}\n{error}")
    return output


def _list_jobs(ssh, username: str) -> list[Job]:
    command = (
        "squeue -h -r -u "
        + shlex.quote(username)
        + " -o '%i|%j|%T|%C|%R|%M|%L'"
    )
    jobs: list[Job] = []
    for line in _remote_output(ssh, command).splitlines():
        fields = line.split("|", maxsplit=6)
        if len(fields) != 7:
            raise ValueError(f"Unexpected squeue row: {line!r}")
        jobs.append(Job(*fields))
    return jobs


def _is_rescue(job: Job) -> bool:
    return job.name.startswith(RESCUE_PREFIX)


def _is_user_held(job: Job) -> bool:
    # This is the reason Slurm reports after ``scontrol hold``.  Restricting
    # release to this reason avoids releasing jobs that are merely waiting on
    # dependencies or resources.
    reason = job.reason.removeprefix("(").removesuffix(")")
    return job.state == "PENDING" and reason == "JobHeldUser"


def _target_jobs(jobs: list[Job], *, release: bool) -> tuple[list[Job], int]:
    rescue_count = sum(_is_rescue(job) for job in jobs)
    if release:
        selected = [job for job in jobs if not _is_rescue(job) and _is_user_held(job)]
    else:
        selected = [
            job
            for job in jobs
            if job.state == "PENDING" and not _is_rescue(job)
        ]
    return selected, rescue_count


def _validate_job_ids(jobs: list[Job]) -> None:
    invalid = [job.job_id for job in jobs if not JOB_ID_PATTERN.fullmatch(job.job_id)]
    if invalid:
        raise ValueError(f"Unexpected Slurm job IDs: {invalid[:5]}")


def _change_holds(ssh, jobs: list[Job], *, release: bool) -> None:
    _validate_job_ids(jobs)
    action = "release" if release else "hold"
    for start in range(0, len(jobs), CHUNK_SIZE):
        chunk = jobs[start : start + CHUNK_SIZE]
        ids = " ".join(job.job_id for job in chunk)
        _remote_output(ssh, f"scontrol {action} {ids}")


def _print_report(
    jobs: list[Job], selected: list[Job], rescue_count: int, *, release: bool
) -> None:
    print(f"All visible jobs: {len(jobs)}")
    print(f"Rescue jobs excluded: {rescue_count}")
    if release:
        print(f"User-held non-rescue jobs selected: {len(selected)}")
    else:
        print(f"Pending non-rescue jobs selected: {len(selected)}")
    if not selected:
        print("No pending non-rescue jobs matched.")
        return
    groups = Counter(job.name for job in selected)
    print("Selected job groups:")
    for name, count in groups.most_common():
        print(f"  {count:4d}  {name}")
    print("First selected job IDs:", " ".join(job.job_id for job in selected[:10]))
    print("Last selected job IDs:", " ".join(job.job_id for job in selected[-10:]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        choices=sorted(SERVER_NAMES),
        default="pascal",
        help="Cluster server to inspect (default: pascal).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Apply holds after confirmation.")
    mode.add_argument("--release", action="store_true", help="Release pending user holds.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt; required for non-dry-run actions in scripts.",
    )
    args = parser.parse_args()
    if (args.apply or args.release) and not args.yes:
        parser.error("--apply/--release requires --yes to prevent accidental queue changes.")

    server = SERVER_NAMES[args.server]
    ssh = connectToCluster(server, verbose=False)
    if ssh is None:
        raise RuntimeError(f"Could not connect to {server}.")
    try:
        username_output = _remote_output(ssh, "whoami")
        if not username_output or "\n" in username_output:
            raise ValueError(f"Unexpected remote username: {username_output!r}")
        jobs = _list_jobs(ssh, username_output)
        selected, rescue_count = _target_jobs(jobs, release=args.release)
        _print_report(jobs, selected, rescue_count, release=args.release)
        if not selected:
            return 0
        if not args.apply and not args.release:
            print("Dry run: no queue changes made.")
            return 0
        action = "release" if args.release else "hold"
        print(f"Applying {action} to {len(selected)} jobs on {args.server}.")
        _change_holds(ssh, selected, release=args.release)
        print(f"Successfully applied {action} to {len(selected)} jobs.")
        return 0
    finally:
        ssh.close()


if __name__ == "__main__":
    raise SystemExit(main())
