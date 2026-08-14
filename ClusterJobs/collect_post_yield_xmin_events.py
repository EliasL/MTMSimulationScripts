#!/usr/bin/env python3
"""Stage a post-yield irreversible-event collection around the fitted xmin.

The workflow is deliberately split into small, reviewable steps:

1. ``select``: fit xmin and freeze a morphology-blind target/backup manifest.
2. ``inventory``: list available remote checkpoints.
3. ``plan``: match selected events to preceding checkpoints and make 2-job waves.
4. ``fetch``: download only selected inputs.
5. ``replay``: run at most two two-thread target replays.
6. ``validate``: accept only strictly reproduced five-state events.
7. ``render``: build the marked PDF and 3x2 event atlas.

The ``run`` step performs the stages in order and waits for new checkpoints
when the source runs have not reached all selected events.  All writes and
replay processes are protected by a 100 GB free-space guard on the data volume.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from ClusterJobs.replay_selected_real_space_event import (
    ReplayArguments,
    validate_replay,
)
from Plotting.real_space_events.acquisition import (
    _ssh_target,
    state_paths_from_directory,
)
from Plotting.real_space_events.catalog import build_standard_scatter_catalog
from Plotting.real_space_events.render import render_event_pdf
from Plotting.real_space_events.models import RenderOptions
from Plotting.real_space_events.models import RemoteSource
from Plotting.real_space_events.xmin_atlas import (
    _combine_pdfs,
    _fit_default_irreversible,
    _render_fit_pdf,
    _standard_sources,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/Volumes/data/MTS2D_xmin_collection/post_yield_irreversible")
DEFAULT_DATA_VOLUME = Path("/Volumes/data")
DEFAULT_SEEDS = (0, 1, 2)  # Seed 3's checkpoint-producing rerun is much slower.
STEPS = (
    "select",
    "inventory",
    "plan",
    "fetch",
    "replay",
    "validate",
    "render",
    "run",
)
DUMP_PATTERN = re.compile(r"dump_l(?P<load>[0-9.eE+-]+)\.xml\.gz$")
REPLAY_SCRIPT = ROOT / "ClusterJobs/replay_selected_real_space_event.py"
DEFAULT_MTS2D_BINARY = ROOT.parent / "MTS2D/build-release/MTS2D"


@dataclass(frozen=True)
class SpaceGuard:
    """Abort the campaign if the data volume approaches the hard free-space floor."""

    volume: Path = DEFAULT_DATA_VOLUME
    minimum_free_gb: float = 100.0

    def __post_init__(self) -> None:
        if self.minimum_free_gb < 100.0:
            raise ValueError("minimum_free_gb cannot be lower than the required 100 GB.")

    @property
    def minimum_free_bytes(self) -> int:
        return int(self.minimum_free_gb * 1_000_000_000)

    def check(self) -> None:
        if not self.volume.is_dir():
            raise FileNotFoundError(f"Data volume is not mounted: {self.volume}")
        free = shutil.disk_usage(self.volume).free
        if free < self.minimum_free_bytes:
            raise RuntimeError(
                f"Aborting: {self.volume} has {free / 1e9:.2f} GB free, below "
                f"the hard floor of {self.minimum_free_gb:.2f} GB."
            )


def _guard_or_default(guard: SpaceGuard | None) -> SpaceGuard:
    result = guard or SpaceGuard()
    result.check()
    return result


def _write_text_guarded(path: Path, text: str, guard: SpaceGuard) -> None:
    guard.check()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    guard.check()


def _write_csv_guarded(frame: pd.DataFrame, path: Path, guard: SpaceGuard) -> None:
    guard.check()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    guard.check()


def _run_checked_with_space_guard(
    command: list[str], guard: SpaceGuard, *, cwd: Path | None = None
) -> None:
    """Run a transfer while checking free space between process polls."""

    guard.check()
    process = subprocess.Popen(command, cwd=cwd)
    try:
        while process.poll() is None:
            guard.check()
            time.sleep(5)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    guard.check()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def _sleep_with_space_guard(seconds: float, guard: SpaceGuard) -> None:
    end = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < end:
        guard.check()
        time.sleep(min(30.0, end - time.monotonic()))


@dataclass(frozen=True)
class CollectionSettings:
    """Frozen scientific and resource choices for one collection campaign."""

    batch: int = -2
    setting: float = 1e-6
    target_count_per_side: int = 10
    backup_count_per_side: int = 5
    below_min_ratio: float = 0.5
    above_max_ratio: float = 2.0
    preferred_seeds: tuple[int, ...] = DEFAULT_SEEDS
    dump_spacing: float = 0.01
    load_increment: float = 1e-5
    maximum_concurrent_replays: int = 2
    replay_threads: int = 2

    def validate(self) -> None:
        if self.target_count_per_side <= 0 or self.backup_count_per_side < 0:
            raise ValueError("Target count must be positive and backup count nonnegative.")
        if not 0 < self.below_min_ratio < 1:
            raise ValueError("below_min_ratio must lie strictly between zero and one.")
        if self.above_max_ratio <= 1:
            raise ValueError("above_max_ratio must exceed one.")
        if not self.preferred_seeds or len(set(self.preferred_seeds)) != len(
            self.preferred_seeds
        ):
            raise ValueError("preferred_seeds must be a nonempty sequence of unique seeds.")
        if self.dump_spacing <= 0 or self.load_increment <= 0:
            raise ValueError("Dump spacing and load increment must be positive.")
        if self.maximum_concurrent_replays != 2 or self.replay_threads != 2:
            raise ValueError("This campaign is intentionally fixed at two 2-thread replays.")


def _candidate_pool(
    irreversible: pd.DataFrame, xmin: float, settings: CollectionSettings
) -> pd.DataFrame:
    """Build the energy-only candidate pool without inspecting morphology."""

    required = {
        "event_id",
        "job_name",
        "seed",
        "load",
        "event_start_load",
        "yield_regime",
        "population",
        "delta_E_S_over_V0",
    }
    missing = required.difference(irreversible.columns)
    if missing:
        raise KeyError(f"Irreversible catalogue is missing columns: {sorted(missing)}")
    if not np.isfinite(xmin) or xmin <= 0:
        raise ValueError(f"xmin must be finite and positive, got {xmin}.")
    if not irreversible["yield_regime"].eq("post").all():
        raise ValueError("Candidate input contains non-post-yield events.")
    if not irreversible["population"].eq("nonclosing").all():
        raise ValueError("Candidate input contains events outside the irreversible population.")

    pool = irreversible[irreversible["seed"].isin(settings.preferred_seeds)].copy()
    pool["xmin_ratio"] = pool["delta_E_S_over_V0"] / xmin
    below = pool[
        (pool["xmin_ratio"] >= settings.below_min_ratio)
        & (pool["xmin_ratio"] < 1.0)
    ].copy()
    below["xmin_side"] = "below"
    above = pool[
        (pool["xmin_ratio"] >= 1.0)
        & (pool["xmin_ratio"] <= settings.above_max_ratio)
    ].copy()
    above["xmin_side"] = "above"
    pool = pd.concat([below, above], ignore_index=True)
    pool["required_dump_load"] = (
        np.floor((pool["event_start_load"] + 1e-12) / settings.dump_spacing)
        * settings.dump_spacing
    )
    if pool.empty:
        raise RuntimeError("No events satisfy the configured xmin-ratio windows.")
    return pool.sort_values(["xmin_side", "xmin_ratio", "seed", "load"])


def _select_balanced_log_spread(
    pool: pd.DataFrame,
    *,
    side: str,
    lower_ratio: float,
    upper_ratio: float,
    settings: CollectionSettings,
) -> pd.DataFrame:
    """Choose targets and backups deterministically across ratio and seed.

    This is the critical anti-cherry-picking rule.  Selection uses only the
    energy ratio and a fixed seed cycle; participation, m3 count and spatial
    morphology are deliberately absent from the ranking.
    """

    side_pool = pool[pool["xmin_side"].eq(side)].copy()
    requested = settings.target_count_per_side + settings.backup_count_per_side
    if len(side_pool) < requested:
        raise RuntimeError(
            f"Only {len(side_pool)} {side} candidates satisfy the frozen window; "
            f"need at least {requested}."
        )
    remaining = side_pool.copy()
    selected = []
    target_ratios = np.geomspace(
        lower_ratio, upper_ratio, settings.target_count_per_side + 2
    )[1:-1]
    backup_ratios = np.geomspace(
        lower_ratio, upper_ratio, settings.backup_count_per_side + 2
    )[1:-1]
    choices = [
        ("target", rank, ratio)
        for rank, ratio in enumerate(target_ratios, start=1)
    ] + [
        ("backup", rank, ratio)
        for rank, ratio in enumerate(backup_ratios, start=1)
    ]
    for index, (role, rank, desired_ratio) in enumerate(choices):
        seed = settings.preferred_seeds[index % len(settings.preferred_seeds)]
        candidates = remaining[remaining["seed"].eq(seed)].copy()
        if candidates.empty:
            raise RuntimeError(
                f"Seed {seed} has too few unused {side} candidates for the fixed "
                "balanced selection rule. Change the seed set explicitly."
            )
        candidates["selection_error"] = np.abs(
            np.log(candidates["xmin_ratio"] / desired_ratio)
        )
        chosen = candidates.sort_values(
            ["selection_error", "required_dump_load", "event_id"]
        ).iloc[0].copy()
        chosen["selection_role"] = role
        chosen["selection_rank"] = rank
        chosen["desired_xmin_ratio"] = desired_ratio
        selected.append(chosen)
        remaining = remaining[~remaining["event_id"].eq(chosen["event_id"])]
    return pd.DataFrame(selected)


def select_candidates(
    output_root: Path, settings: CollectionSettings, guard: SpaceGuard | None = None
) -> Path:
    """Step 1: fit post-yield xmin and freeze target plus backup candidates."""

    settings.validate()
    guard = _guard_or_default(guard)
    catalog = build_standard_scatter_catalog(
        batch=settings.batch, setting=settings.setting
    )
    irreversible, values, analysis, _ = _fit_default_irreversible(catalog, "post")
    xmin = float(analysis["global_min_xmin"])
    pool = _candidate_pool(irreversible, xmin, settings)
    below = _select_balanced_log_spread(
        pool,
        side="below",
        lower_ratio=settings.below_min_ratio,
        upper_ratio=1.0,
        settings=settings,
    )
    above = _select_balanced_log_spread(
        pool,
        side="above",
        lower_ratio=1.0,
        upper_ratio=settings.above_max_ratio,
        settings=settings,
    )
    selected = pd.concat([below, above], ignore_index=True)
    if selected["event_id"].duplicated().any():
        raise RuntimeError("The frozen candidate manifest contains duplicate events.")

    guard.check()
    output_root.mkdir(parents=True, exist_ok=True)
    pool_path = output_root / "candidate_pool.csv"
    manifest_path = output_root / "candidate_manifest.csv"
    _write_csv_guarded(pool, pool_path, guard)
    _write_csv_guarded(selected, manifest_path, guard)
    summary = {
        "batch": settings.batch,
        "setting": settings.setting,
        "population": "post-yield irreversible",
        "catalogue_count": int(len(irreversible)),
        "global_min_xmin": xmin,
        "global_min_distance": float(analysis["global_min_distance"]),
        "tail_count": int(np.count_nonzero(values >= xmin)),
        "target_count_per_side": settings.target_count_per_side,
        "backup_count_per_side": settings.backup_count_per_side,
        "below_ratio_window": [settings.below_min_ratio, 1.0],
        "above_ratio_window": [1.0, settings.above_max_ratio],
        "preferred_seeds": list(settings.preferred_seeds),
        "selection_variables": ["delta_E_S_over_V0 / xmin", "seed"],
    }
    _write_text_guarded(
        output_root / "selection_summary.json",
        json.dumps(summary, indent=2) + "\n",
        guard,
    )
    return manifest_path


def _dump_load(path: Path) -> float:
    match = DUMP_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected dump filename: {path.name}")
    return float(match.group("load"))


def _list_remote_dumps(
    job_name: str, host: str, remote_job_directory: Path, guard: SpaceGuard
) -> list[Path]:
    """List dump files from one exact remote job directory, read-only."""

    guard.check()
    dump_directory = remote_job_directory / "dumps"
    if host in {"local", "localhost"}:
        paths = sorted(dump_directory.glob("dump_l*.xml.gz"), key=_dump_load)
    else:
        result = subprocess.run(
            [
                "ssh",
                "-T",
                "-o",
                "ConnectTimeout=5",
                _ssh_target(RemoteSource(host=host, data_root=Path("."))),
                "find",
                str(dump_directory),
                "-maxdepth",
                "1",
                "-type",
                "f",
                "-name",
                "dump_l*.xml.gz",
                "-print",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not list dumps for {job_name!r} on {host}: "
                f"{result.stderr.strip()}"
            )
        paths = sorted(
            (Path(line.strip()) for line in result.stdout.splitlines() if line.strip()),
            key=_dump_load,
        )
    if not paths:
        raise RuntimeError(f"No restart dumps found for {job_name!r} at {dump_directory}.")
    guard.check()
    return paths


def inventory_checkpoints(
    output_root: Path, settings: CollectionSettings, guard: SpaceGuard | None = None
) -> Path:
    """Step 2: inventory remote dumps without downloading or changing them."""

    settings.validate()
    guard = _guard_or_default(guard)
    manifest_path = output_root / "candidate_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run the select step first: missing {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    records = []
    for job_name in sorted(manifest["job_name"].unique()):
        sources = _standard_sources(str(job_name))
        if len(sources) != 1:
            raise RuntimeError(
                f"Expected one source for {job_name!r}, found {sources}. "
                "Resolve this ambiguity before downloading."
            )
        source = sources[0]
        remote_job_directory = source.data_root / str(job_name)
        paths = _list_remote_dumps(
            str(job_name), source.host, remote_job_directory, guard
        )
        for dump_path in paths:
            records.append(
                {
                    "job_name": str(job_name),
                    "dump_load": _dump_load(dump_path),
                    "dump_path": str(dump_path),
                    "remote_host": source.host,
                    "remote_job_directory": str(remote_job_directory),
                    "remote_config_path": str(remote_job_directory / "config.conf"),
                    "remote_macro_path": str(remote_job_directory / "macroData.csv"),
                }
            )
    inventory = pd.DataFrame(records).sort_values(["job_name", "dump_load"])
    if inventory.duplicated(["job_name", "dump_load"]).any():
        raise RuntimeError("Remote inventory contains duplicate job/load checkpoints.")
    inventory_path = output_root / "checkpoint_inventory.csv"
    _write_csv_guarded(inventory, inventory_path, guard)
    return inventory_path


def attach_checkpoints(
    candidates: pd.DataFrame,
    inventory: pd.DataFrame,
    settings: CollectionSettings,
) -> pd.DataFrame:
    """Critical pure planner: attach each event's nearest preceding dump."""

    required_inventory = {"job_name", "dump_load", "dump_path"}
    missing = required_inventory.difference(inventory.columns)
    if missing:
        raise KeyError(f"Checkpoint inventory is missing columns: {sorted(missing)}")
    if inventory.duplicated(["job_name", "dump_load"]).any():
        raise ValueError("Checkpoint inventory contains duplicate job/load rows.")
    inventory = inventory.copy()
    inventory["dump_load"] = pd.to_numeric(inventory["dump_load"], errors="raise")

    planned = []
    for _, event in candidates.iterrows():
        preceding = inventory[
            inventory["job_name"].eq(event["job_name"])
            & (inventory["dump_load"] <= float(event["event_start_load"]) + 1e-12)
        ].sort_values("dump_load")
        row = event.copy()
        if preceding.empty:
            row["checkpoint_status"] = "waiting_for_preceding_dump"
            row["dump_load"] = np.nan
            row["dump_path"] = ""
            row["replay_gap"] = np.nan
            row["replay_steps"] = np.nan
        else:
            dump = preceding.iloc[-1]
            gap = float(event["event_start_load"]) - float(dump["dump_load"])
            if gap < -1e-10:
                raise RuntimeError("Internal error: selected dump follows its event.")
            row["dump_load"] = float(dump["dump_load"])
            row["dump_path"] = str(dump["dump_path"])
            for column in inventory.columns:
                if column not in {"job_name", "dump_load", "dump_path"}:
                    row[column] = dump[column]
            row["replay_gap"] = gap
            row["replay_steps"] = int(np.ceil(gap / settings.load_increment - 1e-8))
            row["checkpoint_status"] = (
                "ready"
                if gap <= settings.dump_spacing + settings.load_increment / 2
                else "waiting_for_nearer_dump"
            )
        planned.append(row)
    return pd.DataFrame(planned)


def plan_replays(
    output_root: Path,
    inventory_path: Path,
    settings: CollectionSettings,
    guard: SpaceGuard | None = None,
) -> Path:
    """Step 3: make auditable waves without downloading or starting anything."""

    settings.validate()
    guard = _guard_or_default(guard)
    manifest_path = output_root / "candidate_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run the select step first: missing {manifest_path}")
    if not inventory_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint inventory is missing: {inventory_path}. Complete step 2 first."
        )
    planned = attach_checkpoints(
        pd.read_csv(manifest_path), pd.read_csv(inventory_path), settings
    )
    planned["nr_threads"] = settings.replay_threads
    planned["wave"] = pd.Series(pd.NA, index=planned.index, dtype="Int64")
    ready_indices = planned.index[planned["checkpoint_status"].eq("ready")]
    for offset, index in enumerate(ready_indices):
        planned.loc[index, "wave"] = (
            offset // settings.maximum_concurrent_replays + 1
        )
    planned["replay_output_directory"] = planned.apply(
        lambda row: str(
            output_root
            / "replays"
            / f"{row['xmin_side']}_{row['selection_role']}_{int(row['selection_rank']):02d}"
        ),
        axis=1,
    )
    plan_path = output_root / "replay_plan.csv"
    _write_csv_guarded(planned, plan_path, guard)
    return plan_path


def _copy_remote_file(
    host: str, remote_path: Path, local_path: Path, guard: SpaceGuard
) -> None:
    """Copy one source file, resuming an existing destination with rsync."""

    guard.check()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if host in {"local", "localhost"}:
        source = Path(remote_path)
        if not source.is_file():
            raise FileNotFoundError(f"Missing local source file: {source}")
        shutil.copy2(source, local_path)
    else:
        command = [
            "rsync",
            "-a",
            "-e",
            "ssh -T",
            f"{_ssh_target(RemoteSource(host=host, data_root=Path('.')))}:{remote_path}",
            str(local_path),
        ]
        _run_checked_with_space_guard(command, guard)
    if not local_path.is_file() or local_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is missing or empty: {local_path}")
    guard.check()


def fetch_inputs(
    output_root: Path, settings: CollectionSettings, guard: SpaceGuard | None = None
) -> Path:
    """Step 4: fetch only configs, macro files, and planned restart dumps."""

    settings.validate()
    guard = _guard_or_default(guard)
    plan_path = output_root / "replay_plan.csv"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Run the plan step first: missing {plan_path}")
    plan = pd.read_csv(plan_path)
    ready = plan[plan["checkpoint_status"].eq("ready")].copy()
    if ready.empty:
        fetch_manifest = output_root / "fetch_manifest.csv"
        _write_csv_guarded(
            pd.DataFrame(
                columns=[
                    "event_id",
                    "job_name",
                    "dump_load",
                    "remote_dump_path",
                    "local_dump_path",
                ]
            ),
            fetch_manifest,
            guard,
        )
        return fetch_manifest
    required = {
        "remote_host",
        "remote_config_path",
        "remote_macro_path",
        "dump_path",
    }
    missing = required.difference(ready.columns)
    if missing:
        raise KeyError(f"Replay plan is missing remote input columns: {sorted(missing)}")

    inputs_root = output_root / "inputs"
    local_jobs: dict[str, Path] = {}
    downloaded = []
    for job_name, job_rows in ready.groupby("job_name", sort=True):
        row = job_rows.iloc[0]
        local_job = inputs_root / str(job_name)
        local_job.mkdir(parents=True, exist_ok=True)
        _copy_remote_file(
            str(row["remote_host"]),
            Path(str(row["remote_config_path"])),
            local_job / "config.conf",
            guard,
        )
        _copy_remote_file(
            str(row["remote_host"]),
            Path(str(row["remote_macro_path"])),
            local_job / "macroData.csv",
            guard,
        )
        local_jobs[str(job_name)] = local_job

    updated = plan.copy()
    for job_name, local_job in local_jobs.items():
        updated.loc[updated["job_name"].eq(job_name), "local_source_job"] = str(local_job)
    for index, row in ready.iterrows():
        local_job = local_jobs[str(row["job_name"])]
        local_dump = local_job / Path(str(row["dump_path"])).name
        _copy_remote_file(
            str(row["remote_host"]),
            Path(str(row["dump_path"])),
            local_dump,
            guard,
        )
        updated.loc[index, "local_dump_path"] = str(local_dump)
        downloaded.append(
            {
                "event_id": row["event_id"],
                "job_name": row["job_name"],
                "dump_load": row["dump_load"],
                "remote_dump_path": row["dump_path"],
                "local_dump_path": str(local_dump),
            }
        )
    _write_csv_guarded(updated, plan_path, guard)
    fetch_manifest = output_root / "fetch_manifest.csv"
    _write_csv_guarded(pd.DataFrame(downloaded), fetch_manifest, guard)
    return fetch_manifest


def _replay_command(
    row: pd.Series, mts2d_binary: Path, settings: CollectionSettings
) -> list[str]:
    required = {"local_source_job", "local_dump_path", "replay_output_directory"}
    missing = [column for column in required if not str(row.get(column, ""))]
    if missing:
        raise RuntimeError(f"Replay row {row['event_id']} is missing {missing}.")
    source_job = Path(str(row["local_source_job"]))
    dump = Path(str(row["local_dump_path"]))
    if not source_job.is_dir() or not dump.is_file():
        raise FileNotFoundError(
            f"Replay inputs are incomplete for {row['event_id']}: {source_job}, {dump}"
        )
    return [
        sys.executable,
        "-u",
        str(REPLAY_SCRIPT),
        "--source-job",
        str(source_job),
        "--dump",
        str(dump),
        "--target-load",
        f"{float(row['load']):.17g}",
        "--output-directory",
        str(row["replay_output_directory"]),
        "--mts2d-binary",
        str(mts2d_binary),
        "--nr-threads",
        str(settings.replay_threads),
        "--expected-event-kind",
        "plastic",
        "--maximum-elastic-events",
        "0",
    ]


def _terminate_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _quarantine_incomplete_replays(rows: list[pd.Series]) -> None:
    """Move interrupted replay directories aside so a later resume can retry."""

    for row in rows:
        output_directory = Path(str(row["replay_output_directory"]))
        if not output_directory.is_dir() or not any(output_directory.iterdir()):
            continue
        quarantine = output_directory.with_name(output_directory.name + ".aborted")
        if quarantine.exists():
            raise FileExistsError(
                f"Cannot quarantine interrupted replay; destination exists: {quarantine}"
            )
        output_directory.rename(quarantine)


def replay_events(
    output_root: Path,
    settings: CollectionSettings,
    guard: SpaceGuard | None = None,
    *,
    mts2d_binary: Path = DEFAULT_MTS2D_BINARY,
    roles: tuple[str, ...] = ("target",),
) -> Path:
    """Step 5: execute ready events in waves of at most two 2-thread jobs."""

    settings.validate()
    guard = _guard_or_default(guard)
    if not mts2d_binary.is_file():
        raise FileNotFoundError(f"MTS2D binary not found: {mts2d_binary}")
    plan_path = output_root / "replay_plan.csv"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Run the plan step first: missing {plan_path}")
    plan = pd.read_csv(plan_path)
    candidates = plan[
        plan["checkpoint_status"].eq("ready") & plan["selection_role"].isin(roles)
    ].copy()
    previous_path = output_root / "replay_results.csv"
    previous = pd.read_csv(previous_path) if previous_path.is_file() else pd.DataFrame()
    completed_ids = set()
    failed_ids = set()
    if not previous.empty:
        completed_ids = set(
            previous.loc[previous["status"].eq("completed"), "event_id"].astype(str)
        )
        failed_ids = set(
            previous.loc[previous["status"].eq("failed"), "event_id"].astype(str)
        )
    pending = []
    for _, row in candidates.iterrows():
        event_id = str(row["event_id"])
        output_directory = Path(str(row["replay_output_directory"]))
        if event_id in completed_ids or event_id in failed_ids:
            continue
        if (output_directory / "replay_manifest.csv").is_file():
            completed_ids.add(event_id)
            continue
        if output_directory.exists() and any(output_directory.iterdir()):
            raise RuntimeError(
                f"Refusing to overwrite incomplete replay output: {output_directory}"
            )
        pending.append(row)

    results = []
    for start in range(0, len(pending), settings.maximum_concurrent_replays):
        wave = pending[start : start + settings.maximum_concurrent_replays]
        processes = []
        log_handles = []
        try:
            for row in wave:
                guard.check()
                output_directory = Path(str(row["replay_output_directory"]))
                output_directory.parent.mkdir(parents=True, exist_ok=True)
                log_path = output_directory.parent / f"{output_directory.name}.log"
                log_handle = log_path.open("a")
                log_handles.append(log_handle)
                env = dict(os.environ)
                env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
                env.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
                process = subprocess.Popen(
                    _replay_command(row, mts2d_binary, settings),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                processes.append(process)
            while any(process.poll() is None for process in processes):
                guard.check()
                time.sleep(5)
            for row, process in zip(wave, processes):
                manifest_exists = Path(
                    str(row["replay_output_directory"])
                ).joinpath("replay_manifest.csv").is_file()
                status = (
                    "completed"
                    if process.returncode == 0 and manifest_exists
                    else "failed"
                )
                results.append(
                    {
                        "event_id": row["event_id"],
                        "xmin_side": row["xmin_side"],
                        "selection_role": row["selection_role"],
                        "returncode": process.returncode,
                        "status": status,
                        "output_directory": row["replay_output_directory"],
                    }
                )
        except BaseException as error:
            _terminate_processes(processes)
            if isinstance(error, RuntimeError) and str(error).startswith("Aborting:"):
                _quarantine_incomplete_replays(wave)
            raise
        finally:
            for handle in log_handles:
                handle.close()
    if results:
        new_results = pd.DataFrame(results)
        combined = pd.concat([previous, new_results], ignore_index=True)
        combined = combined.drop_duplicates("event_id", keep="last")
    else:
        combined = previous
    if combined.empty:
        combined = pd.DataFrame(
            columns=[
                "event_id",
                "xmin_side",
                "selection_role",
                "returncode",
                "status",
                "output_directory",
            ]
        )
    replay_results_path = output_root / "replay_results.csv"
    _write_csv_guarded(combined, replay_results_path, guard)
    return replay_results_path


def validate_events(
    output_root: Path,
    settings: CollectionSettings,
    guard: SpaceGuard | None = None,
) -> Path:
    """Step 6: strictly validate every completed replay and its five states."""

    settings.validate()
    guard = _guard_or_default(guard)
    plan_path = output_root / "replay_plan.csv"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Run the plan step first: missing {plan_path}")
    plan = pd.read_csv(plan_path)
    records = []
    for _, row in plan.iterrows():
        record = {
            "event_id": row["event_id"],
            "xmin_side": row["xmin_side"],
            "selection_role": row["selection_role"],
            "selection_rank": row["selection_rank"],
            "status": "not_attempted",
            "reason": "",
            "local_event_directory": "",
        }
        output_directory = Path(str(row.get("replay_output_directory", "")))
        replay_manifest = output_directory / "replay_manifest.csv"
        if not replay_manifest.is_file():
            records.append(record)
            continue
        try:
            manifest = pd.read_csv(replay_manifest)
            plastic = manifest[manifest["event_kind"].eq("plastic")]
            if len(plastic) != 1:
                raise RuntimeError("Replay manifest must contain exactly one plastic event.")
            event_directory = Path(str(plastic.iloc[0]["event_directory"]))
            state_paths_from_directory(event_directory)
            replay_args = ReplayArguments(
                source_job=Path(str(row["local_source_job"])),
                dump=Path(str(row["local_dump_path"])),
                target_load=float(row["load"]),
                output_directory=output_directory,
                mts2d_binary=DEFAULT_MTS2D_BINARY,
                nr_threads=settings.replay_threads,
                expected_event_kind="plastic",
                maximum_elastic_events=0,
            )
            validate_replay(replay_args, event_directory)
            record["status"] = "validated"
            record["local_event_directory"] = str(event_directory)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            record["status"] = "invalid"
            record["reason"] = f"{type(error).__name__}: {error}"
        records.append(record)
    validation = pd.DataFrame(records)
    validation_path = output_root / "validation_results.csv"
    _write_csv_guarded(validation, validation_path, guard)
    return validation_path


def _validated_selection(
    output_root: Path, settings: CollectionSettings
) -> pd.DataFrame:
    manifest = pd.read_csv(output_root / "candidate_manifest.csv")
    validation = pd.read_csv(output_root / "validation_results.csv")
    valid = validation[validation["status"].eq("validated")].copy()
    merged = manifest.merge(
        valid[["event_id", "local_event_directory"]], on="event_id", how="inner"
    )
    selected = []
    for side in ("below", "above"):
        side_rows = merged[merged["xmin_side"].eq(side)].copy()
        side_rows["role_order"] = side_rows["selection_role"].map(
            {"target": 0, "backup": 1}
        )
        side_rows = side_rows.sort_values(["role_order", "selection_rank"])
        if len(side_rows) < settings.target_count_per_side:
            raise RuntimeError(
                f"Only {len(side_rows)} validated {side} events; "
                f"need {settings.target_count_per_side}."
            )
        selected.append(side_rows.head(settings.target_count_per_side))
    return pd.concat(selected, ignore_index=True)


def render_atlas(
    output_root: Path,
    settings: CollectionSettings,
    guard: SpaceGuard | None = None,
) -> Path:
    """Step 7: render exactly the validated target/backup replacement cohort."""

    settings.validate()
    guard = _guard_or_default(guard)
    selected = _validated_selection(output_root, settings)
    catalog = build_standard_scatter_catalog(
        batch=settings.batch, setting=settings.setting
    )
    irreversible, values, analysis, fit = _fit_default_irreversible(catalog, "post")
    xmin = float(analysis["global_min_xmin"])
    selection_summary = json.loads(
        (output_root / "selection_summary.json").read_text()
    )
    if not np.isclose(
        xmin,
        float(selection_summary["global_min_xmin"]),
        rtol=1e-12,
        atol=0,
    ):
        raise RuntimeError("The fitted xmin changed since the candidate manifest was frozen.")

    pdf_root = output_root / "pdf"
    pdf_root.mkdir(parents=True, exist_ok=True)
    selected = selected.copy()
    selected["atlas_rank"] = selected.groupby("xmin_side").cumcount() + 1
    selected["atlas_label"] = selected.apply(
        lambda row: f"{row['xmin_side']}_xmin_{int(row['atlas_rank']):02d}", axis=1
    )
    _write_csv_guarded(selected, output_root / "validated_selected_events.csv", guard)
    summary_path = pdf_root / "post_yield_irreversible_pdf_with_selected_events.pdf"
    guard.check()
    _render_fit_pdf(
        summary_path,
        values,
        selected,
        analysis,
        fit,
        "post",
        selected_kind="validated event",
    )
    guard.check()
    page_paths = [summary_path]
    for _, row in selected.iterrows():
        guard.check()
        event = row.copy()
        event["reconnection_mode"] = "none"
        event["reversibility_measured"] = True
        event["saved_event_directory"] = row["local_event_directory"]
        page_path = pdf_root / f"{row['atlas_label']}.pdf"
        render_event_pdf(
            event,
            state_paths_from_directory(Path(row["local_event_directory"])),
            page_path,
            RenderOptions(output_root=pdf_root, output_format="pdf"),
            setting_catalog=catalog,
        )
        guard.check()
        page_paths.append(page_path)
    final_path = pdf_root / "post_yield_irreversible_xmin_event_atlas.pdf"
    _combine_pdfs(page_paths, final_path)
    guard.check()
    return final_path


def _validated_counts(validation_path: Path) -> dict[str, int]:
    validation = pd.read_csv(validation_path)
    valid = validation[validation["status"].eq("validated")]
    return {
        side: int(valid[valid["xmin_side"].eq(side)].shape[0])
        for side in ("below", "above")
    }


def _target_events_waiting(output_root: Path, settings: CollectionSettings) -> bool:
    plan = pd.read_csv(output_root / "replay_plan.csv")
    validation = pd.read_csv(output_root / "validation_results.csv")
    validated_ids = set(validation.loc[validation["status"].eq("validated"), "event_id"])
    target = plan[plan["selection_role"].eq("target")]
    unresolved = target[~target["event_id"].isin(validated_ids)]
    return bool(unresolved["checkpoint_status"].ne("ready").any())


def run_collection(
    output_root: Path,
    settings: CollectionSettings,
    guard: SpaceGuard | None = None,
    *,
    mts2d_binary: Path = DEFAULT_MTS2D_BINARY,
    poll_seconds: float = 1800.0,
    resume: bool = False,
) -> Path:
    """Run the campaign, waiting for source runs to produce later checkpoints."""

    settings.validate()
    guard = _guard_or_default(guard)
    output_root = Path(output_root)
    candidate_manifest = output_root / "candidate_manifest.csv"
    if candidate_manifest.is_file() and not resume:
        raise FileExistsError(
            f"Existing campaign found at {output_root}. Use --resume to continue it."
        )
    if not candidate_manifest.is_file():
        select_candidates(output_root, settings, guard)

    while True:
        guard.check()
        inventory_checkpoints(output_root, settings, guard)
        plan_replays(
            output_root,
            output_root / "checkpoint_inventory.csv",
            settings,
            guard,
        )
        fetch_inputs(output_root, settings, guard)
        replay_events(
            output_root,
            settings,
            guard,
            mts2d_binary=mts2d_binary,
            roles=("target",),
        )
        validation_path = validate_events(output_root, settings, guard)
        counts = _validated_counts(validation_path)
        print(f"Validated targets: {counts}", flush=True)
        if all(counts[side] >= settings.target_count_per_side for side in counts):
            return render_atlas(output_root, settings, guard)

        # Backups are used only after every still-unvalidated target is either
        # replayed or waiting for a checkpoint.  This avoids spending replay
        # time on backups while a target candidate is merely delayed.
        if not _target_events_waiting(output_root, settings):
            replay_events(
                output_root,
                settings,
                guard,
                mts2d_binary=mts2d_binary,
                roles=("backup",),
            )
            validation_path = validate_events(output_root, settings, guard)
            counts = _validated_counts(validation_path)
            print(f"Validated targets plus backups: {counts}", flush=True)
            if all(counts[side] >= settings.target_count_per_side for side in counts):
                return render_atlas(output_root, settings, guard)

        print(
            f"Waiting {poll_seconds:g} seconds for new checkpoints; "
            f"validated counts remain {counts}.",
            flush=True,
        )
        _sleep_with_space_guard(poll_seconds, guard)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-inventory", type=Path, default=None)
    parser.add_argument("--data-volume", type=Path, default=DEFAULT_DATA_VOLUME)
    parser.add_argument("--minimum-free-gb", type=float, default=100.0)
    parser.add_argument("--mts2d-binary", type=Path, default=DEFAULT_MTS2D_BINARY)
    parser.add_argument("--poll-seconds", type=float, default=1800.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-backups", action="store_true")
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--backup-count", type=int, default=5)
    parser.add_argument("--below-min-ratio", type=float, default=0.5)
    parser.add_argument("--above-max-ratio", type=float, default=2.0)
    parser.add_argument("--preferred-seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    guard = SpaceGuard(args.data_volume, args.minimum_free_gb)
    guard.check()
    settings = CollectionSettings(
        target_count_per_side=args.target_count,
        backup_count_per_side=args.backup_count,
        below_min_ratio=args.below_min_ratio,
        above_max_ratio=args.above_max_ratio,
        preferred_seeds=tuple(args.preferred_seeds),
    )
    if args.step == "select":
        print(select_candidates(args.output_root, settings, guard))
    elif args.step == "inventory":
        print(inventory_checkpoints(args.output_root, settings, guard))
    elif args.step == "plan":
        inventory = args.checkpoint_inventory or (
            args.output_root / "checkpoint_inventory.csv"
        )
        print(plan_replays(args.output_root, inventory, settings, guard))
    elif args.step == "fetch":
        print(fetch_inputs(args.output_root, settings, guard))
    elif args.step == "replay":
        roles = ("target", "backup") if args.include_backups else ("target",)
        print(
            replay_events(
                args.output_root,
                settings,
                guard,
                mts2d_binary=args.mts2d_binary,
                roles=roles,
            )
        )
    elif args.step == "validate":
        print(validate_events(args.output_root, settings, guard))
    elif args.step == "render":
        print(render_atlas(args.output_root, settings, guard))
    elif args.step == "run":
        print(
            run_collection(
                args.output_root,
                settings,
                guard,
                mts2d_binary=args.mts2d_binary,
                poll_seconds=args.poll_seconds,
                resume=args.resume,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
