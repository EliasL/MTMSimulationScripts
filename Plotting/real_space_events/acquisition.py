"""Plan targeted VTU downloads and one-off forced backward replays.

This module must never download a complete simulation folder.  It should first
identify individual event directories remotely, write a human-readable
manifest, and fetch only the five VTUs plus small provenance files selected by
the user.
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess

import numpy as np
import pandas as pd

from Plotting.remoteDataPaths import REAL_SPACE_EVENT_PATH

from .models import DownloadRequest, EventStatePaths, RemoteSource, ReplayRequest


LOCAL_EVENT_ROOT = Path(REAL_SPACE_EVENT_PATH)
STATE_PREFIXES = (
    "state0_min_gamma",
    "state1_affine_gamma_plus",
    "state2_relaxed_gamma_plus",
    "state3_affine_gamma_minus",
    "state4_relaxed_gamma",
)


def management_sources_for_job(
    job_name: str, *, index_path: Path = Path("Management/data.json")
) -> tuple[RemoteSource, ...]:
    """Resolve cluster locations from the existing management index."""

    data = json.loads(Path(index_path).read_text())
    sources = []
    for host, entries in data.items():
        paths = entries[0] if entries and isinstance(entries[0], list) else []
        for path_string in paths:
            path = Path(path_string)
            if path.name == job_name:
                sources.append(RemoteSource(host=host, data_root=path.parent))
    unique = {(source.host, source.data_root): source for source in sources}
    return tuple(unique.values())


def _ssh_target(source: RemoteSource) -> str:
    return source.host if "@" in source.host else f"elundheim@{source.host}"


def _remote_file_names(source: RemoteSource, directory: Path) -> list[str]:
    """List one exact remote directory through one read-only SSH query."""

    if source.host in {"local", "localhost"}:
        return sorted(path.name for path in directory.iterdir() if path.is_file())
    result = subprocess.run(
        [
            "ssh", "-T", _ssh_target(source),
            "find", str(directory), "-maxdepth", "1", "-type", "f",
            "-print",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(
        Path(line).name for line in result.stdout.splitlines() if line
    )


def locate_remote_event_directory(
    catalog_row: pd.Series, sources: tuple[RemoteSource, ...]
) -> tuple[RemoteSource, Path] | None:
    """Find one exact saved event directory using read-only remote queries."""

    job_name = str(catalog_row["job_name"])
    load = float(catalog_row["load"])
    delta_gamma = float(catalog_row.get("delta_gamma", np.nan))
    start_load = float(catalog_row.get("event_start_load", load - delta_gamma))
    if not np.isfinite(start_load):
        raise ValueError("Cannot identify the event start load from the catalogue row.")
    expected_names = (f"rev_drop_l_{start_load:.5f}", f"irrev_drop_l_{start_load:.5f}")
    for source in sources:
        candidate_root = source.data_root / job_name / "data" / "reversibilityData"
        for name in expected_names:
            candidate = candidate_root / name
            if source.host in {"local", "localhost"}:
                exists = candidate.is_dir()
            else:
                exists = subprocess.run(
                    ["ssh", "-T", _ssh_target(source), "test", "-d", str(candidate)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode == 0
            if exists:
                return source, candidate
    return None


def plan_downloads(
    selected_events: pd.DataFrame,
    sources: tuple[RemoteSource, ...],
    *,
    local_root: Path = LOCAL_EVENT_ROOT,
) -> list[DownloadRequest]:
    """Return requests only for events with existing complete five-state data."""

    requests = []
    for _, row in selected_events.iterrows():
        found = locate_remote_event_directory(row, sources)
        if found is None:
            continue
        source, remote_directory = found
        event_id = str(row["event_id"])
        local_directory = Path(local_root) / event_id.replace("/", "_")
        requests.append(
            DownloadRequest(
                event_id=event_id,
                source=source,
                remote_event_directory=remote_directory,
                local_event_directory=local_directory,
            )
        )
    return requests


def download_event(request: DownloadRequest) -> EventStatePaths:
    """Download exactly one file for each state prefix.

    Refuse ambiguous glob matches, partial existing directories, or a request
    whose destination escapes the configured event root.  Use rsync only after
    the complete remote file list is resolved and printed.
    """

    request.local_event_directory.mkdir(parents=True, exist_ok=True)
    remote_names = _remote_file_names(request.source, request.remote_event_directory)
    local_files = {}
    for prefix in STATE_PREFIXES:
        matches = [name for name in remote_names if name.startswith(f"{prefix}.") and name.endswith(".vtu")]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one remote {prefix} file in {request.remote_event_directory}; "
                f"found {matches}. Resolve this before downloading."
            )
        filename = matches[0]
        destination = request.local_event_directory / filename
        if request.source.host in {"local", "localhost"}:
            shutil.copy2(request.remote_event_directory / filename, destination)
        else:
            subprocess.run(
                [
                    "rsync", "-a", "-e", "ssh -T",
                    f"{_ssh_target(request.source)}:{request.remote_event_directory / filename}",
                    str(destination),
                ],
                check=True,
            )
        local_files[prefix] = destination
    return EventStatePaths(**local_files)


def state_paths_from_directory(event_directory: Path) -> EventStatePaths:
    """Resolve one complete local event directory without downloading anything."""

    resolved = {}
    for prefix in STATE_PREFIXES:
        matches = sorted(Path(event_directory).glob(f"{prefix}.*.vtu"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one {prefix} VTU in {event_directory}; found {matches}."
            )
        resolved[prefix] = matches[0]
    return EventStatePaths(**resolved)


def plan_replays(
    selected_events: pd.DataFrame,
    sources: tuple[RemoteSource, ...],
    *,
    maximum_events: int = 2,
) -> list[ReplayRequest]:
    """Plan at most one or two forced backward tests for unmeasured events."""

    unmeasured = selected_events[
        selected_events["event_class"].eq("reversibility_unmeasured")
    ].head(maximum_events)
    requests = []
    for _, row in unmeasured.iterrows():
        found = locate_remote_event_directory(row, sources)
        if found is None:
            continue
        source, remote_directory = found
        job_directory = source.data_root / str(row["job_name"])
        requests.append(
            ReplayRequest(
                event_id=str(row["event_id"]),
                source=source,
                job_directory=job_directory,
                dump_path=job_directory / "dumps" / "SELECT_NEAREST_DUMP",
                target_load=float(row["load"]),
                output_directory=LOCAL_EVENT_ROOT / str(row["event_id"]).replace("/", "_"),
            )
        )
    return requests


def write_acquisition_manifest(
    downloads: list[DownloadRequest], replays: list[ReplayRequest], path: Path
) -> Path:
    """Write JSON provenance before any network or cluster mutation occurs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "downloads": [request.__dict__ for request in downloads],
        "replays": [request.__dict__ for request in replays],
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path
