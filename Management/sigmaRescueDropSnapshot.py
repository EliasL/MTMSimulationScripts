"""Snapshot completed sigma-rescue inputs for later interim analysis.

This module is intentionally a small layer around the existing rescue and
energy-drop code.  It never edits source simulations or cluster rescue output.
The intended workflow is:

1. Freeze a timestamped inventory of accepted ``result.json`` files.
2. Download one source ``macroData.csv`` and ``config.conf`` per completed run,
   plus only ``validated_sigma.csv`` for each accepted rescue task.
3. Merge corrected stress by the exact ``(load_step, load)`` key.
4. Calculate all three energy drops once, at event resolution.
5. Retain a transition only when one validated stress provider covers both
   endpoint rows.  This deliberately discards replay-boundary transitions.

The remote inventory and atomic downloader are deliberately usable without
constructing any drop files.  Drop-table construction remains a separate,
explicit later step.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path, PurePosixPath
import re
import shlex
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from Management.connectToCluster import (
    Servers,
    connectToCluster,
    getServerUserName,
)
from Management.sigmaRescue import (
    INVALID_SIGMA_SENTINEL,
    RESCUED_COLUMNS,
    _config_value,
    _read_macro_rows,
    inspect_schema_intervals,
)
from Plotting.energyDropCalculations import calculate_energy_step_data


ROW_KEY_COLUMNS = ("load_step", "load")
ENERGY_COLUMNS = (
    "total_energy",
    "total_energy_change",
    "total_e_change_from_init",
)
DROP_COLUMNS = ("delta_E_I", "delta_E_R", "delta_E_S")
INVALID_SIGMA_VALUE = float(INVALID_SIGMA_SENTINEL)
ACCEPTED_RESCUE_STATUSES = {"validated", "validated_with_sentinels"}
CAMPAIGN_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


_REMOTE_DISCOVERY = r'''
import csv
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
hash_files = sys.argv[3] == "1"
accepted = {"validated", "validated_with_sentinels"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def metadata(path):
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "sha256": digest(path) if hash_files else "",
    }


def config_values(path):
    values = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            values[key] = value
    return values


def last_data_load(path):
    last = None
    with path.open(newline="") as stream:
        for raw_row in csv.reader(stream):
            if not raw_row or raw_row[0].strip().lower().startswith("#header:"):
                continue
            try:
                last = float(raw_row[1])
            except (IndexError, TypeError, ValueError):
                continue
    if last is None:
        raise ValueError(f"No data load found in {path}")
    return last


def artifact(server, remote_path, relative_path):
    path = Path(remote_path)
    result = {"server": server, "remote_path": str(path), "relative_path": relative_path}
    result.update(metadata(path))
    return result


manifest_root = root / "stage" / "manifests"
if not manifest_root.is_dir():
    raise FileNotFoundError(manifest_root)

server = sys.argv[2]
source_cache = {}
records = []
for manifest_path in sorted(manifest_root.glob("*.json")):
    manifest = json.loads(manifest_path.read_text())
    required_manifest = {"size", "seed", "run_name", "output_directory"}
    if not required_manifest.issubset(manifest):
        continue
    size = int(manifest["size"])
    seed = int(manifest["seed"])
    run_name = str(manifest["run_name"])
    source_config_value = manifest.get("source_config")
    if source_config_value is None:
        source_config_value = root.parent.parent / "MTS2D_output" / run_name / "config.conf"
    source_config = Path(source_config_value).resolve()
    source_folder = source_config.parent
    source_macro = source_folder / "macroData.csv"
    if not source_config.is_file() or not source_macro.is_file():
        continue

    source_key = (size, seed, run_name)
    if source_key not in source_cache:
        config = config_values(source_config)
        if config.get("reconnectionMethod", "none") != "none":
            source_cache[source_key] = None
        else:
            max_load = float(config.get("maxLoad", "nan"))
            completed = (
                max_load == max_load
                and last_data_load(source_macro) >= max_load - 1e-10
            )
            source_cache[source_key] = (
                artifact(server, source_macro, ""),
                artifact(server, source_config, ""),
            ) if completed else None
    source_artifacts = source_cache[source_key]
    if source_artifacts is None:
        continue

    output_directory = Path(manifest["output_directory"]).resolve()
    result_paths = sorted(output_directory.rglob("result.json"))
    if len(result_paths) != 1:
        continue
    result_path = result_paths[0]
    result = json.loads(result_path.read_text())
    if result.get("status") not in accepted:
        continue
    validated_path = Path(result["validated_sigma"])
    if not validated_path.is_absolute():
        validated_path = (result_path.parent / validated_path).resolve()
    if not validated_path.is_file():
        continue

    task_id = str(
        result.get("segment_id")
        or result.get("replay_id")
        or manifest.get("prefix_id")
        or manifest.get("segment_id")
    )
    safe_run = "".join(char if char.isalnum() or char in "._-" else "_" for char in run_name)
    safe_task = "".join(char if char.isalnum() or char in "._-" else "_" for char in task_id)
    base = f"raw/L{size:03d}/seed_{seed:03d}/{safe_run}"
    records.append(
        {
            "source_key": [size, seed, run_name],
            "task_id": task_id,
            "status": result["status"],
            "source_macro": artifact(server, source_macro, f"{base}/macroData.csv"),
            "source_config": artifact(server, source_config, f"{base}/config.conf"),
            "result": artifact(server, result_path, f"{base}/rescue/{safe_task}/result.json"),
            "validated_sigma": artifact(server, validated_path, f"{base}/rescue/{safe_task}/validated_sigma.csv"),
        }
    )

print(json.dumps({"campaign_root": str(root), "records": records}))
'''


@dataclass(frozen=True)
class RemoteArtifact:
    """One immutable file in a frozen download plan."""

    server: str
    remote_path: str
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ValidatedSigmaFragment:
    """A downloaded, accepted sigma-rescue contribution."""

    task_id: str
    path: Path


@dataclass(frozen=True)
class SnapshotRun:
    """One completed source run with all locally downloaded rescue fragments."""

    size: int
    seed: int
    run_name: str
    run_directory_name: str
    source_macro: Path
    fragments: tuple[ValidatedSigmaFragment, ...]


@dataclass(frozen=True)
class SnapshotTableBuild:
    """Summary of one completed local snapshot-table build."""

    table_root: Path
    run_count: int
    audited_transition_count: int
    usable_transition_count: int


def discover_remote_campaign(
    *,
    servers: Sequence[str] | None = None,
    campaign_name: str = "size_scaling_sigma12_v1",
    hash_files: bool = False,
) -> list[RemoteArtifact]:
    """Inventory accepted rescue CSVs for completed non-reconnecting runs.

    Each server is checked for both its ``/data`` and ``/data2`` campaign
    roots.  The remote helper verifies source completion from the source
    ``config.conf`` and final macro-data load.  By default it records remote
    file sizes and leaves SHA256 calculation to the local downloader; this
    avoids repeatedly reading large rescue CSVs over the cluster filesystem.
    Set ``hash_files=True`` for a strict remote-hash inventory.  Duplicate
    source identities across servers are rejected instead of being silently
    combined.
    """

    if not CAMPAIGN_NAME_RE.fullmatch(campaign_name):
        raise ValueError(f"Unsafe campaign name: {campaign_name!r}")
    if servers is None:
        servers = Servers.search_servers

    artifacts: list[RemoteArtifact] = []
    source_servers: dict[tuple[int, int, str], str] = {}
    seen_tasks: set[tuple[int, int, str, str]] = set()

    for server in servers:
        user = getServerUserName(server)
        for base in ("/data", "/data2"):
            root = f"{base}/{user}/MTS2D_sigma_rescue/{campaign_name}"
            ssh = connectToCluster(server, False)
            if ssh is None:
                continue
            try:
                check = ssh.exec_command(f"test -d {shlex.quote(root)}")[1]
                if check.channel.recv_exit_status() != 0:
                    continue
                command = (
                    "python3 -c "
                    + shlex.quote(_REMOTE_DISCOVERY)
                    + " "
                    + shlex.quote(root)
                    + " "
                    + shlex.quote(server)
                    + " "
                    + ("1" if hash_files else "0")
                )
                _, stdout, stderr = ssh.exec_command(command)
                payload = stdout.read().decode("utf-8")
                error_output = stderr.read().decode("utf-8").strip()
                if not payload.strip():
                    raise RuntimeError(
                        f"Empty remote inventory from {server}:{root}: {error_output}"
                    )
                data = json.loads(payload)
            finally:
                ssh.close()

            for record in data["records"]:
                source_key = tuple(record["source_key"])
                task_key = (*source_key, record["task_id"])
                previous_server = source_servers.setdefault(source_key, server)
                if previous_server != server:
                    raise ValueError(
                        f"Completed source run {source_key} appears on both "
                        f"{previous_server} and {server}."
                    )
                if task_key in seen_tasks:
                    raise ValueError(f"Duplicate rescue task discovered: {task_key}")
                seen_tasks.add(task_key)
                for key in ("source_macro", "source_config", "result", "validated_sigma"):
                    artifacts.append(RemoteArtifact(**record[key]))

    # The same source files appear once for every accepted task.  Deduplicate
    # them only when the complete artifact identity agrees.
    unique: dict[tuple[str, str], RemoteArtifact] = {}
    for artifact in artifacts:
        key = (artifact.server, artifact.remote_path)
        previous = unique.get(key)
        if previous is not None and previous != artifact:
            raise ValueError(f"Conflicting artifact metadata for {key}")
        unique[key] = artifact
    return list(unique.values())


def write_artifact_inventory(artifacts: Sequence[RemoteArtifact], path: Path) -> Path:
    """Write a frozen JSON inventory, refusing to overwrite an existing one."""

    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite inventory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"artifacts": [artifact.__dict__ for artifact in artifacts]},
            indent=2,
        )
        + "\n"
    )
    return path


def load_artifact_inventory(path: Path) -> list[RemoteArtifact]:
    """Load and validate a previously frozen artifact inventory."""

    path = Path(path).resolve()
    data = json.loads(path.read_text())
    records = data.get("artifacts")
    if not isinstance(records, list):
        raise ValueError(f"Invalid artifact inventory: {path}")
    artifacts = [RemoteArtifact(**record) for record in records]
    if len({artifact.relative_path for artifact in artifacts}) != len(artifacts):
        raise ValueError(f"Duplicate relative artifact paths in {path}")
    return artifacts


def sha256_file(path: Path) -> str:
    """Return the SHA256 of one local file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_destination(snapshot_root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe snapshot relative path: {relative_path!r}")
    root = Path(snapshot_root).expanduser().resolve()
    destination = root.joinpath(*relative.parts).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Snapshot path escapes {root}: {destination}")
    return destination


def _verify_artifact(path: Path, artifact: RemoteArtifact) -> None:
    if path.stat().st_size != artifact.size_bytes:
        raise ValueError(
            f"Size mismatch for {path}: expected {artifact.size_bytes}, "
            f"got {path.stat().st_size}."
        )
    digest = sha256_file(path)
    if artifact.sha256 and digest != artifact.sha256:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {artifact.sha256}, got {digest}."
        )


def download_artifacts(
    artifacts: Sequence[RemoteArtifact],
    snapshot_root: Path,
    *,
    dry_run: bool = True,
) -> list[Path]:
    """Download a frozen artifact list atomically over one SFTP session/server.

    Existing verified files are reused.  Existing mismatched files, partial
    files, unsafe destinations and hash mismatches are fatal.  Inventories
    made in fast mode contain an empty remote SHA256 and are still fully
    verified by hashing each local file after transfer.  ``dry_run`` is true
    by default so the first invocation only prints the exact transfer set.
    """

    destinations = []
    pending_by_server: dict[str, list[tuple[RemoteArtifact, Path]]] = defaultdict(list)
    for artifact in artifacts:
        destination = _artifact_destination(snapshot_root, artifact.relative_path)
        destinations.append(destination)
        if destination.exists():
            _verify_artifact(destination, artifact)
            continue
        if dry_run:
            print(f"{artifact.server}:{artifact.remote_path} -> {destination}")
            continue
        pending_by_server[artifact.server].append((artifact, destination))

    if dry_run:
        return destinations

    for server, pending in pending_by_server.items():
        ssh = connectToCluster(server, False)
        if ssh is None:
            raise RuntimeError(f"Unable to connect to {server} for artifact download.")
        sftp = ssh.open_sftp()
        try:
            total = len(pending)
            for index, (artifact, destination) in enumerate(pending, start=1):
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_name(f".{destination.name}.partial")
                if partial.exists():
                    raise FileExistsError(f"Refusing stale partial download: {partial}")
                if index == 1 or index == total or index % 25 == 0:
                    print(f"{server}: downloading {index}/{total} ({destination.name})")
                sftp.get(artifact.remote_path, str(partial))
                _verify_artifact(partial, artifact)
                os.replace(partial, destination)
        finally:
            sftp.close()
            ssh.close()
    return destinations


def write_download_inventory(
    artifacts: Sequence[RemoteArtifact],
    destinations: Sequence[Path],
    path: Path,
) -> Path:
    """Record local hashes after a frozen artifact list has been downloaded."""

    if len(artifacts) != len(destinations):
        raise ValueError("Artifact and destination lists must have equal length.")
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite download inventory: {path}")
    records = []
    for artifact, destination in zip(artifacts, destinations):
        destination = Path(destination).resolve()
        _verify_artifact(destination, artifact)
        records.append(
            {
                **artifact.__dict__,
                "local_path": str(destination),
                "local_sha256": sha256_file(destination),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"artifacts": records}, indent=2) + "\n")
    return path


def _downloaded_snapshot_records(snapshot_root: Path) -> dict[str, dict]:
    """Return downloaded-artifact records keyed by safe snapshot path.

    The downloader already verified every local file before writing its
    inventory.  This function validates the snapshot layout and recorded
    hashes without needlessly hashing all of the data a second time.
    """

    snapshot_root = Path(snapshot_root).resolve()
    inventory_path = snapshot_root / "downloaded_inventory.json"
    if not inventory_path.is_file():
        raise FileNotFoundError(
            f"A completed download inventory is required: {inventory_path}"
        )
    payload = json.loads(inventory_path.read_text())
    raw_records = payload.get("artifacts")
    if not isinstance(raw_records, list):
        raise ValueError(f"Invalid downloaded inventory: {inventory_path}")

    records: dict[str, dict] = {}
    for record in raw_records:
        required = {
            "relative_path",
            "local_path",
            "local_sha256",
            "size_bytes",
            "sha256",
            "remote_path",
        }
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError(f"Invalid downloaded-artifact record in {inventory_path}")
        relative_path = str(record["relative_path"])
        if relative_path in records:
            raise ValueError(f"Duplicate downloaded artifact path: {relative_path}")
        expected_path = _artifact_destination(snapshot_root, relative_path)
        local_path = Path(record["local_path"]).resolve()
        if local_path != expected_path:
            raise ValueError(
                f"Downloaded artifact is outside its expected snapshot location: "
                f"{local_path} != {expected_path}"
            )
        if not local_path.is_file():
            raise FileNotFoundError(f"Downloaded artifact is missing: {local_path}")
        if local_path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Downloaded artifact size changed: {local_path}")
        local_hash = str(record["local_sha256"])
        remote_hash = str(record["sha256"])
        if len(local_hash) != 64 or any(char not in "0123456789abcdef" for char in local_hash):
            raise ValueError(f"Invalid recorded local SHA256 for {local_path}")
        if remote_hash and local_hash != remote_hash:
            raise ValueError(f"Recorded local/remote SHA256 disagreement for {local_path}")
        records[relative_path] = record
    return records


def discover_snapshot_runs(snapshot_root: Path) -> list[SnapshotRun]:
    """Recover strict run/fragment groups from a completed local snapshot.

    This intentionally consumes ``downloaded_inventory.json`` rather than
    globbing the directory tree.  Every source and rescue file must therefore
    have been included in the immutable acquisition record.
    """

    snapshot_root = Path(snapshot_root).resolve()
    records = _downloaded_snapshot_records(snapshot_root)
    source_records: dict[str, dict] = {}
    for relative_path, record in records.items():
        path = PurePosixPath(relative_path)
        if (
            len(path.parts) == 5
            and path.parts[0] == "raw"
            and path.parts[-1] == "macroData.csv"
        ):
            prefix = str(PurePosixPath(*path.parts[:-1]))
            if prefix in source_records:
                raise ValueError(f"Duplicate source macroData record for {prefix}")
            source_records[prefix] = record
    if not source_records:
        raise ValueError(f"No source macroData.csv files found in {snapshot_root}")

    runs: list[SnapshotRun] = []
    claimed_rescue_paths: set[str] = set()
    for prefix, source_record in source_records.items():
        prefix_path = PurePosixPath(prefix)
        _, size_directory, seed_directory, run_directory_name = prefix_path.parts
        size_match = re.fullmatch(r"L(\d+)", size_directory)
        seed_match = re.fullmatch(r"seed_(\d+)", seed_directory)
        if size_match is None or seed_match is None:
            raise ValueError(f"Invalid source snapshot path: {prefix}")
        size = int(size_match.group(1))
        seed = int(seed_match.group(1))
        if size <= 0:
            raise ValueError(f"Invalid system size in {prefix}")

        config_path = f"{prefix}/config.conf"
        if config_path not in records:
            raise FileNotFoundError(f"Missing source config in snapshot: {config_path}")
        source_macro = _artifact_destination(
            snapshot_root, str(source_record["relative_path"])
        )
        remote_source_path = PurePosixPath(str(source_record["remote_path"]))
        run_name = remote_source_path.parent.name
        if not run_name:
            raise ValueError(f"Cannot determine run name from {remote_source_path}")

        rescue_prefix = f"{prefix}/rescue/"
        fragments: list[ValidatedSigmaFragment] = []
        task_ids: set[str] = set()
        for relative_path, record in records.items():
            if not relative_path.startswith(rescue_prefix):
                continue
            path = PurePosixPath(relative_path)
            if len(path.parts) != 7 or path.parts[-1] != "validated_sigma.csv":
                continue
            task_directory = path.parts[-2]
            result_path = str(path.parent / "result.json")
            result_record = records.get(result_path)
            if result_record is None:
                raise FileNotFoundError(
                    f"Missing result.json beside rescued sigma: {relative_path}"
                )
            result_file = _artifact_destination(snapshot_root, result_path)
            result = json.loads(result_file.read_text())
            if result.get("status") not in ACCEPTED_RESCUE_STATUSES:
                raise ValueError(f"Unaccepted rescue result in snapshot: {result_file}")
            task_id = str(result.get("segment_id") or result.get("replay_id") or task_directory)
            if task_id in task_ids:
                raise ValueError(f"Duplicate rescue task ID for {prefix}: {task_id}")
            task_ids.add(task_id)
            fragments.append(
                ValidatedSigmaFragment(
                    task_id=task_id,
                    path=_artifact_destination(snapshot_root, relative_path),
                )
            )
            claimed_rescue_paths.add(relative_path)

        if not fragments:
            raise ValueError(f"No accepted rescue fragments for {prefix}")
        runs.append(
            SnapshotRun(
                size=size,
                seed=seed,
                run_name=run_name,
                run_directory_name=run_directory_name,
                source_macro=source_macro,
                fragments=tuple(sorted(fragments, key=lambda fragment: fragment.task_id)),
            )
        )

    all_rescue_paths = {
        relative_path
        for relative_path in records
        if relative_path.endswith("/validated_sigma.csv")
    }
    orphaned = all_rescue_paths - claimed_rescue_paths
    if orphaned:
        raise ValueError(f"Orphaned rescued sigma file: {sorted(orphaned)[0]}")
    identities = [(run.size, run.seed, run.run_name) for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate source-run identities in the local snapshot.")
    return sorted(runs, key=lambda run: (run.size, run.seed, run.run_name))


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    """Write one compressed CSV without exposing a partially written final file."""

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite table: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    if partial.exists():
        raise FileExistsError(f"Refusing stale partial table: {partial}")
    frame.to_csv(partial, index=False, compression="gzip")
    os.replace(partial, destination)


def build_snapshot_drop_tables(snapshot_root: Path) -> SnapshotTableBuild:
    """Create audited local drop tables, without fitting or plotting anything.

    The final ``tables`` directory is created only after every source run has
    been merged successfully.  A failure leaves ``tables.partial`` intact for
    inspection and never exposes incomplete tables as final data.
    """

    snapshot_root = Path(snapshot_root).resolve()
    runs = discover_snapshot_runs(snapshot_root)
    table_root = snapshot_root / "tables"
    work_root = snapshot_root / "tables.partial"
    if table_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing tables: {table_root}")
    if work_root.exists():
        raise FileExistsError(f"Refusing stale partial table directory: {work_root}")
    work_root.mkdir()

    tables: list[pd.DataFrame] = []
    run_summaries = []
    for index, run in enumerate(runs, start=1):
        print(
            f"Combining {index}/{len(runs)}: L={run.size}, seed={run.seed}, "
            f"{len(run.fragments)} accepted rescue fragments"
        )
        merged = merge_available_sigma(
            run.source_macro,
            run.fragments,
            size=run.size,
        )
        table = build_interim_drop_table(
            merged,
            source_macro=run.source_macro,
            run_name=run.run_name,
            size=run.size,
            seed=run.seed,
        )
        run_directory = (
            work_root
            / "by_run"
            / f"L{run.size:03d}"
            / f"seed_{run.seed:03d}"
            / run.run_directory_name
        )
        _write_csv_atomic(table, run_directory / "drops_audited.csv.gz")
        _write_csv_atomic(
            table.loc[table["usable"]].copy(), run_directory / "drops_usable.csv.gz"
        )
        tables.append(table)
        run_summaries.append(
            {
                "size": run.size,
                "seed": run.seed,
                "run_name": run.run_name,
                "accepted_fragment_count": len(run.fragments),
                "audited_transition_count": len(table),
                "usable_transition_count": int(table["usable"].sum()),
            }
        )

    combined = combine_drop_tables(tables)
    _write_csv_atomic(combined, work_root / "drops_all_audited.csv.gz")
    _write_csv_atomic(
        combined.loc[combined["usable"]].copy(), work_root / "drops_usable.csv.gz"
    )
    (work_root / "build_manifest.json").write_text(
        json.dumps(
            {
                "run_count": len(runs),
                "audited_transition_count": len(combined),
                "usable_transition_count": int(combined["usable"].sum()),
                "runs": run_summaries,
            },
            indent=2,
        )
        + "\n"
    )
    os.replace(work_root, table_root)
    return SnapshotTableBuild(
        table_root=table_root,
        run_count=len(runs),
        audited_transition_count=len(combined),
        usable_transition_count=int(combined["usable"].sum()),
    )


def _numeric_frame(source_macro: Path, size: int) -> pd.DataFrame:
    _, rows = _read_macro_rows(source_macro)
    frame = pd.DataFrame(rows)
    required = set(ROW_KEY_COLUMNS) | set(ENERGY_COLUMNS) | set(RESCUED_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing source columns in {source_macro}: {missing}")
    for column in required | {"avg_P12"} & set(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["load_step"] = frame["load_step"].astype(np.int64)
    if frame.duplicated(list(ROW_KEY_COLUMNS)).any():
        duplicate = frame.loc[
            frame.duplicated(list(ROW_KEY_COLUMNS), keep=False),
            list(ROW_KEY_COLUMNS),
        ].iloc[0]
        raise ValueError(
            f"Duplicate source row key in {source_macro}: "
            f"({int(duplicate['load_step'])}, {float(duplicate['load'])})."
        )
    if not frame["load_step"].is_monotonic_increasing:
        raise ValueError(f"Source load_step is not increasing: {source_macro}")
    if not frame["load"].is_monotonic_increasing:
        raise ValueError(f"Source load is not increasing: {source_macro}")
    if int(size) <= 0:
        raise ValueError("size must be positive.")
    return frame.reset_index(drop=True)


def merge_available_sigma(
    source_macro: Path,
    fragments: Sequence[ValidatedSigmaFragment],
    *,
    size: int,
) -> pd.DataFrame:
    """Merge available corrected stress into an immutable source trajectory.

    Energy and event columns always come from the original completed source
    CSV.  Old-schema source stress is ignored.  Correct-new source rows retain
    native stress, while bad-old rows receive stress only from accepted rescue
    fragments.  Overlap conflicts are fatal.

    The returned ``sigma_providers`` column contains a tuple of providers for
    each row.  A later transition is usable only if its endpoint tuples have a
    non-empty intersection.
    """

    source_macro = Path(source_macro).resolve()
    frame = _numeric_frame(source_macro, size)
    intervals = inspect_schema_intervals(source_macro)
    status_by_key = {
        key: interval.sigma_status
        for interval in intervals
        for key in interval.row_keys
    }
    source_keys = list(zip(frame["load_step"].astype(int), frame["load"].astype(float)))
    if set(source_keys) != set(status_by_key):
        raise ValueError(
            f"Schema intervals do not cover the source rows exactly: {source_macro}"
        )

    frame["source_sigma_schema"] = [status_by_key[key] for key in source_keys]
    providers: list[set[str]] = [set() for _ in source_keys]
    key_to_index = {key: index for index, key in enumerate(source_keys)}

    for index, status in enumerate(frame["source_sigma_schema"]):
        if status == "correct-new":
            values = frame.loc[index, list(RESCUED_COLUMNS)].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values == INVALID_SIGMA_VALUE):
                raise ValueError(
                    f"Invalid native new-schema stress at source row {source_keys[index]}."
                )
            providers[index].add("source-native")
        elif status == "bad-old":
            frame.loc[index, list(RESCUED_COLUMNS)] = INVALID_SIGMA_VALUE
        else:
            raise ValueError(f"Unexpected sigma schema status: {status!r}")

    for fragment in fragments:
        fragment_path = Path(fragment.path).resolve()
        validated = pd.read_csv(fragment_path)
        required = set(ROW_KEY_COLUMNS) | set(RESCUED_COLUMNS)
        missing = sorted(required - set(validated.columns))
        if missing:
            raise KeyError(f"Missing columns in {fragment_path}: {missing}")
        for column in required:
            validated[column] = pd.to_numeric(validated[column], errors="raise")
        validated["load_step"] = validated["load_step"].astype(np.int64)
        if validated.duplicated(list(ROW_KEY_COLUMNS)).any():
            raise ValueError(f"Duplicate row key inside {fragment_path}.")

        for row in validated.itertuples(index=False):
            key = (int(row.load_step), float(row.load))
            if key not in key_to_index:
                raise ValueError(f"Rescue row {key} is absent from {source_macro}.")
            index = key_to_index[key]
            if frame.at[index, "source_sigma_schema"] != "bad-old":
                raise ValueError(
                    f"Rescue fragment unexpectedly covers correct-new row {key}."
                )
            incoming = np.asarray(
                [getattr(row, column) for column in RESCUED_COLUMNS], dtype=float
            )
            if not np.all(np.isfinite(incoming)):
                raise ValueError(f"Non-finite rescued stress in {fragment_path} at {key}.")
            if providers[index]:
                existing = frame.loc[index, list(RESCUED_COLUMNS)].to_numpy(dtype=float)
                if not np.allclose(existing, incoming, rtol=1e-8, atol=1e-10):
                    raise ValueError(
                        f"Conflicting rescue overlap at {key}: "
                        f"{existing.tolist()} versus {incoming.tolist()}."
                    )
            else:
                frame.loc[index, list(RESCUED_COLUMNS)] = incoming
            providers[index].add(fragment.task_id)

    frame["sigma_providers"] = [tuple(sorted(value)) for value in providers]
    frame["sigma_available"] = [bool(value) for value in providers]
    frame["sigma_valid"] = (
        frame["sigma_available"]
        & np.isfinite(frame["avg_sigma12"])
        & (frame["avg_sigma12"] != INVALID_SIGMA_VALUE)
    )
    return frame


def _transition_reasons(
    consecutive_step: np.ndarray,
    consecutive_load: np.ndarray,
    valid_sigma: np.ndarray,
    shared_provider: np.ndarray,
    finite_drops: np.ndarray,
) -> list[str]:
    reasons = []
    for values in zip(
        consecutive_step,
        consecutive_load,
        valid_sigma,
        shared_provider,
        finite_drops,
    ):
        names = []
        if not values[0]:
            names.append("nonconsecutive-load-step")
        if not values[1]:
            names.append("unexpected-load-increment")
        if not values[2]:
            names.append("missing-or-invalid-sigma")
        if not values[3]:
            names.append("cross-provider-boundary")
        if not values[4]:
            names.append("nonfinite-drop")
        reasons.append(";".join(names))
    return reasons


def build_interim_drop_table(
    frame: pd.DataFrame,
    *,
    source_macro: Path,
    run_name: str,
    size: int,
    seed: int,
) -> pd.DataFrame:
    """Create one audited, paired event table for a partially rescued run.

    ``Delta E_I`` and ``Delta E_R`` are read from the authoritative source
    trajectory.  ``Delta E_S`` is calculated by the existing second-order
    energy helper.  The three quantities stay aligned on one source event key.
    No positivity or strain-regime filtering is performed here.
    """

    source_macro = Path(source_macro).resolve()
    config_path = source_macro.parent / "config.conf"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"The snapshot must place config.conf beside macroData.csv: {config_path}"
        )
    load_increment = float(_config_value(config_path, "loadIncrement"))
    if load_increment <= 0:
        raise ValueError(f"Invalid loadIncrement in {config_path}: {load_increment}")

    required = (
        set(ROW_KEY_COLUMNS)
        | set(ENERGY_COLUMNS)
        | {"avg_sigma12", "sigma_providers", "sigma_valid"}
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing merged columns: {missing}")
    if len(frame) < 2:
        raise ValueError("At least two source rows are required for a drop table.")

    steps, _ = calculate_energy_step_data(
        source_macro,
        df=frame,
        metadata={"L": int(size)},
        average_energy=False,
    )
    delta_e_i = -frame["total_energy_change"].to_numpy(dtype=float)[1:]
    delta_e_r = -frame["total_e_change_from_init"].to_numpy(dtype=float)[1:]
    delta_e_s = steps["stress_corrected_drop_second_order"].to_numpy(dtype=float)
    drops = np.column_stack((delta_e_i, delta_e_r, delta_e_s))

    load_step = frame["load_step"].to_numpy(dtype=np.int64)
    load = frame["load"].to_numpy(dtype=float)
    consecutive_step = np.diff(load_step) == 1
    consecutive_load = np.isclose(
        np.diff(load), load_increment, rtol=1e-9, atol=1e-12
    )
    valid_sigma = (
        frame["sigma_valid"].to_numpy(dtype=bool)[:-1]
        & frame["sigma_valid"].to_numpy(dtype=bool)[1:]
    )
    provider_intersections = [
        set(left).intersection(right)
        for left, right in zip(
            frame["sigma_providers"].iloc[:-1],
            frame["sigma_providers"].iloc[1:],
        )
    ]
    shared_provider = np.asarray(
        [bool(value) for value in provider_intersections], dtype=bool
    )
    finite_drops = np.all(np.isfinite(drops), axis=1)
    usable = (
        consecutive_step
        & consecutive_load
        & valid_sigma
        & shared_provider
        & finite_drops
    )

    reference_volume = float(int(size) ** 2)
    table = pd.DataFrame(
        {
            "event_id": [f"{run_name}:{step}" for step in load_step[1:]],
            "run_name": run_name,
            "size": int(size),
            "seed": int(seed),
            "load_step_i": load_step[:-1],
            "load_step_ip1": load_step[1:],
            "load_i": load[:-1],
            "load_ip1": load[1:],
            "delta_gamma": np.diff(load),
            "total_energy_i": frame["total_energy"].to_numpy(dtype=float)[:-1],
            "total_energy_ip1": frame["total_energy"].to_numpy(dtype=float)[1:],
            "total_energy_change_ip1": frame["total_energy_change"].to_numpy(dtype=float)[1:],
            "total_e_change_from_init_ip1": frame[
                "total_e_change_from_init"
            ].to_numpy(dtype=float)[1:],
            "avg_sigma12_i": frame["avg_sigma12"].to_numpy(dtype=float)[:-1],
            "avg_sigma12_ip1": frame["avg_sigma12"].to_numpy(dtype=float)[1:],
            "reference_volume": reference_volume,
            "delta_E_I": delta_e_i,
            "delta_E_R": delta_e_r,
            "delta_E_S": delta_e_s,
            "common_sigma_provider": [
                "|".join(sorted(value)) for value in provider_intersections
            ],
            "usable": usable,
            "exclusion_reason": _transition_reasons(
                consecutive_step,
                consecutive_load,
                valid_sigma,
                shared_provider,
                finite_drops,
            ),
        }
    )
    for column in DROP_COLUMNS:
        table[f"{column}_over_V0"] = table[column] / reference_volume
    if table["event_id"].duplicated().any():
        raise ValueError(f"Duplicate event IDs generated for {run_name}.")
    return table


def combine_drop_tables(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-run tables while preserving seed identity and uniqueness."""

    tables = list(tables)
    if not tables:
        raise ValueError("No drop tables were provided.")
    combined = pd.concat(tables, ignore_index=True)
    if combined["event_id"].duplicated().any():
        duplicate = combined.loc[combined["event_id"].duplicated(), "event_id"].iloc[0]
        raise ValueError(f"Duplicate event across run tables: {duplicate}")
    return combined.sort_values(
        ["size", "seed", "load_step_ip1"], ignore_index=True
    )
