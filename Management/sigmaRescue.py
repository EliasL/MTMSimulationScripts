#!/usr/bin/env python3
"""Plan and coordinate recovery of incorrect historical ``avg_sigma12`` data.

STATUS
======
The original simulations, dumps, configs and macro CSVs are immutable inputs.
Every replay writes to a private rescue tree.  Submission remains explicit;
the command-line runner below is deliberately small so it can be copied to a
cluster together with one frozen segment manifest.

Problem
-------
Some size-scaling simulations have correct energies and evolution but an old
stress implementation.  Later rows may switch to a new ``#HEADER`` and contain
correct ``avg_sigma12`` values.  Only rows written with the affected old schema
must be rescued; correct new-schema rows must be copied byte-for-byte or value-
for-value and never recomputed unnecessarily.

Recovery strategy
-----------------
1. Discover the expected size-scaling runs with ``size_scaling_job`` and locate
   their exact source folders with ``DataManager`` when working across servers.
2. Parse each macro CSV as a sequence of schema eras.  Record exactly which
   ``(load_step, load)`` rows require a corrected stress value.
3. Discover and load-sort dumps with ``find_dumps``.  If affected rows precede
   the first usable dump, add a from-scratch prefix replay through that dump.
   Then build independent replay segments from each dump to the following
   dump/load boundary.  Reject plans with uncovered bad rows, duplicate dump
   loads or ambiguous boundaries.
4. Replay the prefix and every segment in its own directory with the current verified MTS2D
   binary and a private config.  Preserve all numerical settings; change only
   output controls, the private name and the stopping load.
5. Compare replayed evolution against the original macro rows before accepting
   stress values.  Energy and load must agree within explicit tolerances.  The
   legacy P12 column is not an evolution invariant across code eras and is
   never used as a sigma substitute.  A trajectory disagreement marks the
   affected rows with an explicit ``-1`` sigma sentinel; structural failures
   still abort the task.
6. Stitch only validated stress columns into a new canonical CSV.  Never edit
   ``macroData.csv`` in place.  Preserve already-correct new-schema stress rows.
7. Publish an audit manifest containing source fingerprints, binary identity,
   segment status, validation statistics and the origin of every rescued row.

Important boundary question
---------------------------
The first smoke test must establish whether resuming dump load ``d`` writes a
row at ``d`` or at ``d + loadIncrement``, and whether ``maxLoad`` is inclusive.
Do not encode an assumption.  Adjacent segments should initially overlap by one
row; validation and stitching must require identical overlap values and then
deduplicate by ``(load_step, load)``.

Dump-format compatibility
-------------------------
Before batch work, test at least one April dump and one June dump with the exact
binary intended for rescue.  Record the binary checksum/commit and loader exit
status in a compatibility manifest.  Test copies or read-only source paths in a
private workspace.  Do not use ``SimulationManager.resumeSimulation`` here: its
normal recovery behavior may rename an unreadable source dump to ``broken_*``.
MTS2D's dump-load error exit code is currently 2, but this must be confirmed for
the chosen binary.  If one era is incompatible, stop and build/test a separate
loader or conversion path; never silently skip that era.

Cluster layout
--------------
For the full rescue, run segments on the server that already holds their dumps
and transfer only the small validated segment CSV/manifest outputs.  Pin every
source to a server, absolute path and fingerprint before submission.  Reuse
``queueLocalJobs.get_batch_script`` for individual Slurm jobs during testing.
For the large campaign, add a Slurm-array wrapper whose task index selects one
immutable record from ``segments.jsonl``; avoid thousands of SSH submissions.

Suggested rescue tree
---------------------
``rescue_root/``
    ``plan.json``                         frozen campaign metadata
    ``prefix.json``                        from-scratch first-dump task, if needed
    ``segments.jsonl``                    one immutable record per task
    ``compatibility.json``                April/June dump-load matrix
    ``segments/<run>/<segment_id>/``      private config/output/log/result
    ``stitched/<run>/macroData.sigmaRescued.csv``
    ``stitched/<run>/audit.csv``          row-to-segment provenance

The coordinator is the only process allowed to stitch.  Segment tasks never
share an output directory.  A segment becomes usable only after validation has
written an atomic ``result.json`` with status ``validated`` or
``validated_with_sentinels``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

# Allow the copied script to import sibling Management modules when invoked by
# absolute path from a Slurm working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Management.jobs import size_scaling_job
try:
    from Management.updateCSV import HEADER_RENAME_MAP
except ModuleNotFoundError as error:
    # Cluster runner environments intentionally need only the CSV aliases and
    # standard-library replay code; pandas is not required to run a segment.
    if error.name != "pandas":
        raise
    HEADER_RENAME_MAP = {
        "Load": "load",
        "Avg energy": "avg_energy",
        "Max energy": "max_energy",
        "Avg RSS": "avg_P12",
        "Nr plastic deformations": "nr_plastic_deformations",
        "Nr FIRE iterations": "nr_iterations",
        "Nr LBFGS iterations": "nr_iterations",
        "Nr CG iterations": "nr_iterations",
        "Nr FIRE func evals": "nr_func_evals",
        "Nr LBFGS func evals": "nr_func_evals",
        "Nr CG iterations.1": "nr_func_evals",
        "FIRE Term reason": "FIRE_Term_reason",
        "LBFGS Term reason": "LBFGS_Term_reason",
        "CG Term reason": "CG_Term_reason",
        "Run time": "run_time",
        "Est time remaining": "est_time_remaining",
        "avg_init_energy_change": "avg_e_change_from_init",
        "avg_RSS": "avg_P12",
        "max_plastic_deformation": "max_m3_nr",
        "Alpha": "load",
        "PreEnergy": "init_energy",
        "PostEnergy": "energy",
        "PreStress": "avg_init_sigma12",
        "PostStress": "avg_sigma12",
        "EnergyChange": "total_e_change_from_init",
        "StressChange": "avg_sigma_change_from_init",
        "avg_sigmaxy": "avg_sigma12",
        "avg_Pxy": "avg_P12",
        "avg_init_sigmaxy": "avg_init_sigma12",
        "avg_sigmaxy_change_from_init": "avg_sigma12_change_from_init",
        "rev_d": "rev_u_diff",
        "nr_plastic_deformations": "nr_elements_with_m3_fix_change",
    }
try:
    from Management.vtuBeforeReconnectionExtraction import find_dumps
except ModuleNotFoundError as error:
    if error.name != "Management.vtuBeforeReconnectionExtraction":
        raise

    def find_dumps(folder: Path) -> list[Path]:
        """Small standard-library fallback for cluster runner environments."""

        folder = Path(folder)
        accepted = (".xml", ".xml.gz", ".mtsb")
        paths = [
            path for path in folder.iterdir()
            if path.is_file() and path.name.endswith(accepted) and path.stat().st_size > 0
        ]
        return sorted(paths, key=dump_load)


OLD_SIGMA_COLUMN = "avg_sigmaxy"
CORRECT_SIGMA_COLUMN = "avg_sigma12"
ROW_KEY_COLUMNS = ("load_step", "load")
RESCUED_COLUMNS = (
    "avg_sigma12",
    "avg_init_sigma12",
    "avg_sigma12_change_from_init",
)
INVALID_SIGMA_SENTINEL = "-1"
EVOLUTION_VALIDATION_COLUMNS = (
    "total_energy",
    "avg_energy",
)
_DUMP_LOAD_PATTERN = re.compile(
    r"_l(?P<load>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)

_REMOTE_INVENTORY_SCRIPT = r'''
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

folder = Path(sys.argv[1]).resolve()
fingerprint_dumps = sys.argv[3] == "1"
read_dump_states = sys.argv[4] == "1"
if not folder.is_dir():
    raise FileNotFoundError(folder)
load_pattern = re.compile(r"_l(?P<load>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")

def nominal_load(path):
    match = load_pattern.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract dump load from {path}")
    return float(match.group("load"))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def state(path):
    if path.suffix == ".mtsb":
        return None, None
    opener = gzip.open if path.name.endswith(".gz") else open
    load = step = None
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if load is None:
                match = re.search(r"<load>([^<]+)</load>", line)
                if match:
                    load = float(match.group(1))
            if step is None:
                match = re.search(r"<loadSteps>([^<]+)</loadSteps>", line)
                if match:
                    step = int(match.group(1))
            if load is not None and step is not None:
                return load, step
    raise ValueError(f"Dump has no load/loadSteps metadata: {path}")

dumps = sorted(
    (path for path in (folder / "dumps").iterdir()
     if path.is_file() and path.name.endswith((".xml", ".xml.gz", ".mtsb"))
     and path.stat().st_size > 0),
    key=nominal_load,
)
records = []
for path in dumps:
    actual_load = actual_step = None
    if read_dump_states:
        actual_load, actual_step = state(path)
    records.append({
        "path": str(path),
        "load": nominal_load(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path) if fingerprint_dumps else "not-computed",
        "format_era": sys.argv[2],
        "state_load": actual_load,
        "state_step": actual_step,
    })
print(json.dumps({
    "folder": str(folder),
    "config_sha256": sha256(folder / "config.conf"),
    "macro_sha256": sha256(folder / "macroData.csv"),
    "dumps": records,
}))
'''


TEST_SEQUENCE = """
L=50, seed=0 test gates
-----------------------
1. Inventory only: identify source folder, schema eras, dump loads, fingerprints
   and bad-row coverage.  Require a prefix task when bad rows precede the first
   usable dump, and require a plan with no uncovered rows.
2. Loader smoke test: resume one copied/read-only dump into an empty private
   directory and stop after the minimum possible number of steps.
3. Boundary test: replay the from-scratch prefix and one complete dump-to-dump
   segment, determining exact start/end row semantics for both.
4. Evolution test: require every replayed energy/load value to match the
   original segment.  Inspect the corrected sigma curve manually; do not use
   legacy P12 values as a substitute.
5. Parallel test: run two adjacent segments concurrently in distinct folders;
   require their overlap rows to agree.
6. Single-seed test: replay every required L=50 seed=0 segment, stitch into a
   new CSV, and prove all non-rescued values are unchanged.
7. Analysis test: regenerate only the L=50 seed=0 stress-derived quantities and
   compare old versus rescued results.
8. Compatibility gate: repeat loader/boundary tests on representative April and
   June dumps before enabling additional seeds or sizes.
9. Scale one dimension at a time: all L=50 seeds, then one seed of each larger
   size, then the remaining campaign.
""".strip()


@dataclass(frozen=True)
class SourceRun:
    """One immutable source simulation selected for rescue."""

    name: str
    size: int
    seed: int
    server: str
    folder: str
    config_path: str
    macro_path: str
    config_sha256: str
    macro_sha256: str


@dataclass(frozen=True)
class SchemaInterval:
    """A contiguous macro-data interval interpreted under one declared header."""

    first_step: int
    last_step: int
    first_load: float
    last_load: float
    header: tuple[str, ...]
    sigma_status: str  # ``bad-old`` or ``correct-new``; no other value is valid.
    row_keys: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class DumpRecord:
    """A dump pinned to a source and fingerprint before any job is submitted."""

    path: str
    load: float
    size_bytes: int
    sha256: str
    format_era: str  # e.g. ``april`` or ``june`` after explicit classification.
    state_load: float | None = None
    state_step: int | None = None


@dataclass(frozen=True)
class SegmentPlan:
    """One independent replay task, serialized as one JSONL record."""

    segment_id: str
    run_name: str
    size: int
    seed: int
    server: str
    start_dump: DumpRecord
    stop_load: float
    expected_first_step: int
    expected_last_step: int
    expected_first_load: float
    expected_last_load: float
    output_directory: str


@dataclass(frozen=True)
class PrefixPlan:
    """The required from-scratch replay through the first usable dump."""

    prefix_id: str
    run_name: str
    size: int
    seed: int
    server: str
    source_config: str
    boundary_dump: DumpRecord
    stop_load: float
    expected_first_step: int
    expected_last_step: int
    expected_first_load: float
    expected_last_load: float
    output_directory: str


@dataclass(frozen=True)
class ValidationTolerance:
    """All tolerances are campaign metadata, not hidden implementation defaults."""

    load_atol: float = 1e-12
    energy_rtol: float = 2e-8
    # Restarting from a serialized dump changes the last printed digits of
    # energy differences, even when the relaxed trajectory is unchanged.
    energy_atol: float = 1e-8


def expected_run_names(reconnection: str = "none") -> dict[str, tuple[int, int]]:
    """Reuse the canonical job generator instead of duplicating run names."""

    groups, _ = size_scaling_job(reconnection=reconnection)
    return {
        config.name: (int(config.rows), int(config.seed))
        for group in groups
        for config in group
    }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a full SHA256 fingerprint for one immutable input file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_load(path: Path) -> float:
    """Extract the load encoded in a supported dump filename."""
    match = _DUMP_LOAD_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"Cannot extract load from dump filename: {path}")
    return float(match.group("load"))


def prefix_dump_load(path: Path) -> float:
    """Return the exact load of a prefix dump when its format supports it."""

    path = Path(path)
    if path.name.endswith((".xml", ".xml.gz")):
        return dump_state(path)[0]
    return dump_load(path)


def select_prefix_dump(
    dumps: Sequence[Path],
    *,
    stop_load: float,
    load_increment: float,
) -> tuple[Path, list[tuple[Path, float]]]:
    """Select the checkpoint nearest the requested prefix stopping load.

    A short run can write more than one checkpoint before ``maxLoad``.  Keep
    the complete checkpoint inventory, but select the final checkpoint by its
    exact serialized load when possible rather than assuming there is exactly
    one dump or trusting the rounded filename load.
    """

    if not dumps:
        raise RuntimeError("Prefix run produced no dump files.")
    if stop_load <= 0:
        raise ValueError(f"Prefix stop load must be positive, got {stop_load}")
    if load_increment <= 0:
        raise ValueError(f"Load increment must be positive, got {load_increment}")

    inventory = sorted(
        ((Path(path), prefix_dump_load(Path(path))) for path in dumps),
        key=lambda item: item[1],
    )
    selected, selected_load = min(
        inventory,
        key=lambda item: abs(item[1] - stop_load),
    )
    tolerance = max(2.0 * load_increment, 1e-10)
    if abs(selected_load - stop_load) > tolerance:
        raise RuntimeError(
            "Prefix dumps do not reach the requested stopping load: "
            f"nearest dump={selected_load:.17g}, stop_load={stop_load:.17g}, "
            f"tolerance={tolerance:.17g}."
        )
    return selected, inventory


def dump_state(path: Path) -> tuple[float, int]:
    """Read the exact relaxed load and step stored inside a native XML dump."""

    load = None
    step = None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if load is None:
                match = re.search(r"<load>([^<]+)</load>", line)
                if match:
                    load = float(match.group(1))
            if step is None:
                match = re.search(r"<loadSteps>([^<]+)</loadSteps>", line)
                if match:
                    step = int(match.group(1))
            if load is not None and step is not None:
                return load, step
    raise ValueError(f"Dump has no load/loadSteps metadata: {path}")


def _set_config_value(text: str, key: str, value: object) -> str:
    """Replace one config setting or append it when the source omits it."""
    pattern = re.compile(
        rf"^(?P<prefix>\s*{re.escape(key)}\s*=\s*)"
        r"(?P<value>[^#\r\n]*?)(?P<suffix>\s*(?:#.*)?)$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) > 1:
        raise ValueError(f"Config contains more than one {key!r} setting.")
    if not matches:
        separator = "" if text.endswith("\n") else "\n"
        return f"{text}{separator}{key} = {value}\n"
    return pattern.sub(
        lambda match: f"{match.group('prefix')}{value}{match.group('suffix')}",
        text,
        count=1,
    )


def _config_value(path: Path, key: str) -> str:
    """Read one unique scalar config setting without guessing a default."""

    matches = []
    for line in Path(path).read_text().splitlines():
        content = line.split("#", 1)[0].strip()
        if "=" not in content:
            continue
        name, value = content.split("=", 1)
        if name.strip() == key:
            matches.append(value.strip())
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {key!r} in {path}, found {len(matches)}.")
    return matches[0]


def _source_folder_from_dump(dump: Path) -> Path:
    """Resolve the source job from the standard ``job/dumps/dump_*.xml`` layout."""

    dump = Path(dump).resolve()
    if dump.parent.name != "dumps":
        raise ValueError(f"Dump is not inside a source dumps directory: {dump}")
    folder = dump.parent.parent
    if not (folder / "config.conf").is_file():
        raise FileNotFoundError(f"Dump source has no config.conf: {folder}")
    return folder


def _private_run_name(segment: SegmentPlan) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9.-]", "", segment.segment_id)
    return f"{segment.run_name}sigmaRescue{safe_id}"


def _canonical_header(header: Sequence[str], path: Path) -> tuple[str, ...]:
    raw = tuple(column.strip() for column in header)
    canonical = tuple(HEADER_RENAME_MAP.get(column, column) for column in raw)
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"Duplicate canonical columns in {path}: {canonical}")
    if "load_step" not in canonical or "load" not in canonical:
        raise ValueError(f"Missing row-key columns in {path}: {canonical}")
    return canonical


def _read_macro_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read mixed-header macro data while retaining canonical column names."""

    path = Path(path)
    active_header: tuple[str, ...] | None = None
    output_header: list[str] = []
    rows: list[dict[str, str]] = []
    with path.open(newline="") as stream:
        for line_number, raw_row in enumerate(csv.reader(stream), start=1):
            if not raw_row:
                continue
            token = raw_row[0].strip()
            if token.lower().startswith("#header:"):
                active_header = _canonical_header(
                    [token.split(":", 1)[1], *raw_row[1:]], path
                )
                for column in active_header:
                    if column not in output_header:
                        output_header.append(column)
                continue
            if active_header is None:
                active_header = _canonical_header(raw_row, path)
                for column in active_header:
                    if column not in output_header:
                        output_header.append(column)
                continue
            if len(raw_row) != len(active_header):
                raise ValueError(
                    f"Row length mismatch in {path} at line {line_number}: "
                    f"expected {len(active_header)}, got {len(raw_row)}."
                )
            rows.append(dict(zip(active_header, raw_row)))
    if not rows:
        raise ValueError(f"No data rows found in {path}.")
    return tuple(output_header), rows


def _row_key(row: dict[str, str]) -> tuple[int, float]:
    try:
        return int(row["load_step"]), float(row["load"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid macro row key: {row}") from error


def _numeric(row: dict[str, str], column: str, path: Path) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing/non-numeric {column!r} in {path}: {row}") from error
    if not (value == value and abs(value) != float("inf")):
        raise ValueError(f"Non-finite {column!r} in {path}: {value}")
    return value


def remote_inventory(
    server: str,
    source_folder: str,
    *,
    format_era: str,
    fingerprint_dumps: bool = False,
    read_dump_states: bool = False,
) -> dict[str, object]:
    """Collect remote config/macro/dump metadata without copying large files."""

    try:
        completed = subprocess.run(
            [
                "ssh", "-T", server, "python3", "-", source_folder, format_era,
                "1" if fingerprint_dumps else "0",
                "1" if read_dump_states else "0",
            ],
            input=_REMOTE_INVENTORY_SCRIPT,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"Remote inventory exceeded 60 seconds on {server} for {source_folder}."
        ) from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"Remote inventory failed on {server} for {source_folder}: "
            f"{completed.stderr.strip()}"
        )
    try:
        inventory = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Remote inventory returned non-JSON output on {server}: "
            f"{completed.stdout[-1000:]}"
        ) from error
    if inventory.get("folder") != str(Path(source_folder).resolve()):
        raise ValueError(
            f"Remote inventory path mismatch: requested {source_folder}, "
            f"returned {inventory.get('folder')}"
        )
    if not inventory.get("dumps"):
        raise FileNotFoundError(f"No dumps found in remote source {source_folder}")
    return inventory


def _source_run_from_remote_inventory(
    *,
    server: str,
    source_folder: str,
    macro_path: Path,
    inventory: dict[str, object],
    reconnection: str,
) -> SourceRun:
    name = Path(source_folder).name
    expected = expected_run_names(reconnection=reconnection)
    if name not in expected:
        raise ValueError(f"Source folder is not a canonical size-scaling run: {name}")
    size, seed = expected[name]
    remote_macro = Path(source_folder) / "macroData.csv"
    return SourceRun(
        name=name,
        size=size,
        seed=seed,
        server=server,
        folder=source_folder,
        config_path=str(Path(source_folder) / "config.conf"),
        macro_path=str(remote_macro),
        config_sha256=str(inventory["config_sha256"]),
        macro_sha256=str(inventory["macro_sha256"]),
    )


def write_plan(
    source: SourceRun,
    segments: Sequence[SegmentPlan],
    destination: Path,
    *,
    prefix: PrefixPlan | None = None,
) -> Path:
    """Write one local frozen plan and one manifest per replay task."""

    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite plan directory: {destination}")
    segment_directory = destination / "segments"
    segment_directory.mkdir(parents=True)
    prefix_path = None
    if prefix is not None:
        prefix_path = destination / "prefix.json"
        prefix_to_json(prefix, prefix_path)
    segment_paths = []
    for segment in segments:
        safe_id = re.sub(r"[^A-Za-z0-9.-]", "", segment.segment_id)
        segment_path = segment_directory / f"{safe_id}.json"
        segment_to_json(segment, segment_path)
        segment_paths.append(str(segment_path))
    plan = {
        "status": "frozen",
        "source": asdict(source),
        "prefix": asdict(prefix) if prefix is not None else None,
        "prefix_manifest": str(prefix_path) if prefix_path is not None else None,
        "segments": [asdict(segment) for segment in segments],
        "segment_manifests": segment_paths,
    }
    (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return destination / "plan.json"


def prepare_remote_plan(
    *,
    server: str,
    source_folder: str,
    macro_path: Path,
    rescue_root: str,
    destination: Path,
    format_era: str,
    reconnection: str = "none",
    fingerprint_dumps: bool = False,
    read_dump_states: bool = False,
) -> Path:
    """Build a local plan from a remote source and a local macro CSV."""

    macro_path = Path(macro_path).expanduser().resolve()
    if not macro_path.is_file():
        raise FileNotFoundError(macro_path)
    inventory = remote_inventory(
        server,
        source_folder,
        format_era=format_era,
        fingerprint_dumps=fingerprint_dumps,
        read_dump_states=read_dump_states,
    )
    if not read_dump_states:
        raise ValueError(
            "Refusing to build a rescue plan without exact dump state metadata; "
            "rerun with read_dump_states=True."
        )
    source = _source_run_from_remote_inventory(
        server=server,
        source_folder=source_folder,
        macro_path=macro_path,
        inventory=inventory,
        reconnection=reconnection,
    )
    intervals = inspect_schema_intervals(macro_path)
    dumps = [DumpRecord(**record) for record in inventory["dumps"]]
    prefix = build_prefix_plan(source, intervals, dumps, Path(rescue_root))
    segments = build_segment_plan(
        source, intervals, dumps, Path(rescue_root), prefix=prefix
    )
    return write_plan(source, segments, destination, prefix=prefix)


def discover_source_runs(
    data_root: Path,
    *,
    server: str,
    size: int | None = None,
    seed: int | None = None,
    reconnection: str = "none",
) -> list[SourceRun]:
    """Discover exact source folders and fingerprint immutable inputs.

    TODO:
    - For the local L=50 test, match folders under ``data_root`` against
      :func:`expected_run_names`.
    - For production, use ``Management.dataManager.DataManager`` to locate the
      folder that owns each dump and reject ambiguous replicas unless the plan
      explicitly selects one fingerprint.
    - Require config, macroData.csv and dumps/.  Never infer settings from the
      generated defaults when an original config exists.
    """

    data_root = Path(data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing source data root: {data_root}")
    expected = expected_run_names(reconnection=reconnection)
    runs = []
    for name, (job_size, job_seed) in sorted(expected.items()):
        if size is not None and job_size != size:
            continue
        if seed is not None and job_seed != seed:
            continue
        folder = data_root / name
        if not folder.exists():
            continue
        if not folder.is_dir():
            raise ValueError(f"Expected simulation folder, found non-directory: {folder}")
        config_path = folder / "config.conf"
        macro_path = folder / "macroData.csv"
        dump_path = folder / "dumps"
        missing = [
            path for path in (config_path, macro_path, dump_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete source simulation {folder}; missing {missing}."
            )
        if not dump_path.is_dir():
            raise NotADirectoryError(dump_path)
        runs.append(
            SourceRun(
                name=name,
                size=job_size,
                seed=job_seed,
                server=server,
                folder=str(folder),
                config_path=str(config_path),
                macro_path=str(macro_path),
                config_sha256=sha256_file(config_path),
                macro_sha256=sha256_file(macro_path),
            )
        )
    return runs


def inspect_schema_intervals(macro_path: Path) -> list[SchemaInterval]:
    """Parse ordinary and ``#HEADER`` declarations without losing row identity.

    Use ``HEADER_RENAME_MAP`` only for canonical names.  The presence of
    ``avg_sigmaxy`` identifies an affected historical interval; renaming it to
    ``avg_sigma12`` does *not* make its values valid.  A declared native
    ``avg_sigma12`` interval is the already-correct era and must be preserved.

    TODO: reject duplicate headers, malformed row lengths, non-monotonic steps,
    non-monotonic loads, unknown stress layouts and multiple values for one row
    key.  Keep the original header text in campaign provenance.
    """

    macro_path = Path(macro_path)
    if not macro_path.is_file():
        raise FileNotFoundError(macro_path)

    def canonical_header(header: list[str]) -> tuple[str, ...]:
        raw = tuple(column.strip() for column in header)
        result = tuple(HEADER_RENAME_MAP.get(column, column) for column in raw)
        if len(set(result)) != len(result):
            raise ValueError(f"Duplicate canonical columns in {macro_path}: {result}")
        if "load_step" not in result or "load" not in result:
            raise ValueError(f"Missing row-key columns in {macro_path}: {result}")
        has_old = OLD_SIGMA_COLUMN in raw
        has_new = CORRECT_SIGMA_COLUMN in raw
        if has_old == has_new:
            raise ValueError(
                f"Expected exactly one sigma schema in {macro_path}; "
                f"old={has_old}, new={has_new}."
            )
        return raw

    intervals: list[SchemaInterval] = []
    active_header: tuple[str, ...] | None = None
    active_status: str | None = None
    first_step = last_step = None
    first_load = last_load = None
    active_row_keys: list[tuple[int, float]] = []
    previous_step = None
    previous_load = None

    def finish_interval() -> None:
        nonlocal first_step, last_step, first_load, last_load, active_row_keys
        if first_step is None:
            return
        assert active_header is not None
        assert active_status is not None
        intervals.append(
            SchemaInterval(
                first_step=first_step,
                last_step=last_step,
                first_load=first_load,
                last_load=last_load,
                header=active_header,
                sigma_status=active_status,
                row_keys=tuple(active_row_keys),
            )
        )
        first_step = last_step = first_load = last_load = None
        active_row_keys = []

    with macro_path.open(newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if not row:
                continue
            token = row[0].strip()
            if token.lower().startswith("#header:"):
                finish_interval()
                declared = [token.split(":", 1)[1], *row[1:]]
                active_header = canonical_header(declared)
                active_status = (
                    "bad-old" if OLD_SIGMA_COLUMN in active_header else "correct-new"
                )
                continue
            if active_header is None:
                active_header = canonical_header(row)
                active_status = (
                    "bad-old" if OLD_SIGMA_COLUMN in active_header else "correct-new"
                )
                continue
            if len(row) != len(active_header):
                raise ValueError(
                    f"Row length mismatch in {macro_path} at line {line_number}: "
                    f"expected {len(active_header)}, got {len(row)}."
                )
            try:
                step = int(row[active_header.index("load_step")])
                load = float(row[active_header.index("load")])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid row key in {macro_path} at line {line_number}."
                ) from error
            if previous_step is not None and step <= previous_step:
                raise ValueError(f"Non-increasing load_step in {macro_path} at line {line_number}.")
            if previous_load is not None and load <= previous_load:
                raise ValueError(f"Non-increasing load in {macro_path} at line {line_number}.")
            if first_step is None:
                first_step, first_load = step, load
            last_step, last_load = step, load
            active_row_keys.append((step, load))
            previous_step, previous_load = step, load

    finish_interval()
    if not intervals:
        raise ValueError(f"No data rows found in {macro_path}.")
    return intervals


def inventory_dumps(dump_folder: Path, *, format_era: str) -> list[DumpRecord]:
    """Discover, sort and fingerprint dumps without modifying them.

    Reuse :func:`find_dumps` for accepted suffixes, empty-file rejection and
    load sorting.  Add strict duplicate-load detection and full SHA256 hashing.
    Full hashing is expensive but justified once per immutable rescue campaign.
    """

    dump_folder = Path(dump_folder)
    if not dump_folder.is_dir():
        raise FileNotFoundError(dump_folder)
    if not format_era.strip():
        raise ValueError("format_era must identify the dump-format era.")
    dumps = find_dumps(dump_folder)
    records = []
    previous_load = None
    for path in dumps:
        load = dump_load(path)
        if previous_load is not None and load <= previous_load:
            raise ValueError(f"Dump loads are not strictly increasing in {dump_folder}.")
        state_load = state_step = None
        if path.suffix != ".mtsb":
            state_load, state_step = dump_state(path)
        records.append(
            DumpRecord(
                path=str(path.resolve()),
                load=load,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                format_era=format_era,
                state_load=state_load,
                state_step=state_step,
            )
        )
        previous_load = load
    if not records:
        raise FileNotFoundError(f"No dumps found in {dump_folder}.")
    return records


def build_segment_plan(
    source: SourceRun,
    intervals: Sequence[SchemaInterval],
    dumps: Sequence[DumpRecord],
    rescue_root: Path,
    *,
    prefix: PrefixPlan | None = None,
) -> list[SegmentPlan]:
    """Cover post-prefix bad-old rows with dump-based segments and no ambiguity.

    TODO:
    - Intersect dump intervals with only ``bad-old`` schema rows.
    - Include a one-row overlap until dump/maxLoad boundary semantics are proven.
    - Detect bad rows before the earliest usable dump and after the final
      recoverable boundary; fail instead of producing a partial plan.
    - Give every task a deterministic ID containing run, start and stop loads.
    - Refuse any output directory inside the source simulation folder.
    """

    bad_intervals = [interval for interval in intervals if interval.sigma_status == "bad-old"]
    all_bad_rows = sorted(
        (key for interval in bad_intervals for key in interval.row_keys),
        key=lambda key: key[0],
    )
    if not all_bad_rows:
        return []
    if not dumps:
        raise ValueError(f"No dumps available for {source.name}.")
    def state_load(dump: DumpRecord) -> float:
        return dump.state_load if dump.state_load is not None else dump.load

    ordered_dumps = sorted(dumps, key=state_load)
    if any(
        state_load(left) >= state_load(right)
        for left, right in zip(ordered_dumps, ordered_dumps[1:])
    ):
        raise ValueError(f"Dump loads are not strictly increasing for {source.name}.")

    tolerance = 1e-10
    first_dump = ordered_dumps[0]
    first_dump_load = state_load(first_dump)
    if prefix is not None:
        if prefix.boundary_dump.path != first_dump.path:
            raise ValueError(
                f"Prefix boundary dump is not the first usable dump for {source.name}: "
                f"{prefix.boundary_dump.path} versus {first_dump.path}"
            )
        if first_dump.state_step is not None:
            bad_rows = [
                key for key in all_bad_rows if key[0] > first_dump.state_step
            ]
        else:
            bad_rows = [
                key for key in all_bad_rows if key[1] > first_dump_load + tolerance
            ]
    else:
        bad_rows = all_bad_rows

    if not bad_rows:
        return []

    first_bad_load = bad_rows[0][1]
    starting_dumps = [
        dump for dump in ordered_dumps if state_load(dump) <= first_bad_load + tolerance
    ]
    if not starting_dumps:
        raise ValueError(
            f"Bad rows begin at load {first_bad_load:g}, before the first usable dump "
            f"for {source.name}; build and include a prefix plan first."
        )

    segments: list[SegmentPlan] = []
    row_index = 0
    current_dump_index = max(
        index
        for index, dump in enumerate(ordered_dumps)
        if state_load(dump) <= first_bad_load + tolerance
    )
    while row_index < len(bad_rows):
        current_dump = ordered_dumps[current_dump_index]
        next_dump = (
            ordered_dumps[current_dump_index + 1]
            if current_dump_index + 1 < len(ordered_dumps)
            else None
        )
        boundary = state_load(next_dump) if next_dump is not None else float("inf")
        chunk_start = row_index
        while row_index < len(bad_rows) and bad_rows[row_index][1] <= boundary + tolerance:
            row_index += 1
        if row_index == chunk_start:
            if next_dump is None:
                raise RuntimeError(f"Could not advance rescue plan for {source.name}.")
            current_dump_index += 1
            continue

        chunk = bad_rows[chunk_start:row_index]
        stop_load = min(boundary, chunk[-1][1])
        segment_id = (
            f"L{source.size}_s{source.seed}_from_{state_load(current_dump):.10g}"
            f"_to_{stop_load:.10g}"
        )
        safe_id = segment_id.replace(".", "p").replace("-", "m")
        output_directory = rescue_root / "segments" / source.name / safe_id
        source_folder = Path(source.folder).resolve()
        if output_directory.resolve().is_relative_to(source_folder):
            raise ValueError(f"Rescue output cannot be inside source folder: {output_directory}")
        segments.append(
            SegmentPlan(
                segment_id=segment_id,
                run_name=source.name,
                size=source.size,
                seed=source.seed,
                server=source.server,
                start_dump=current_dump,
                stop_load=stop_load,
                expected_first_step=chunk[0][0],
                expected_last_step=chunk[-1][0],
                expected_first_load=chunk[0][1],
                expected_last_load=chunk[-1][1],
                output_directory=str(output_directory),
            )
        )
        if next_dump is None or stop_load >= bad_rows[-1][1] - tolerance:
            break
        current_dump_index += 1
    if row_index < len(bad_rows):
        raise RuntimeError(f"Rescue plan did not cover all bad rows for {source.name}.")
    return segments


def build_prefix_plan(
    source: SourceRun,
    intervals: Sequence[SchemaInterval],
    dumps: Sequence[DumpRecord],
    rescue_root: Path,
) -> PrefixPlan | None:
    """Plan the from-scratch replay needed before the first usable dump.

    The prefix is deliberately a separate task from dump replays.  It repairs
    affected rows before the first checkpoint without making the later jobs
    depend on an output path that does not exist until the prefix finishes.
    The dump's XML ``state_load``/``state_step`` is used whenever available;
    the filename and nominal load are only provenance labels.
    """

    bad_rows = sorted(
        (
            key
            for interval in intervals
            if interval.sigma_status == "bad-old"
            for key in interval.row_keys
        ),
        key=lambda key: key[0],
    )
    if not bad_rows:
        return None
    if not dumps:
        raise ValueError(f"No dumps available for {source.name}.")

    def state_load(dump: DumpRecord) -> float:
        return dump.state_load if dump.state_load is not None else dump.load

    ordered_dumps = sorted(dumps, key=state_load)
    if any(
        state_load(left) >= state_load(right)
        for left, right in zip(ordered_dumps, ordered_dumps[1:])
    ):
        raise ValueError(f"Dump loads are not strictly increasing for {source.name}.")

    boundary_dump = ordered_dumps[0]
    boundary_load = state_load(boundary_dump)
    if boundary_dump.state_step is not None:
        prefix_rows = [key for key in bad_rows if key[0] <= boundary_dump.state_step]
    else:
        prefix_rows = [key for key in bad_rows if key[1] <= boundary_load + 1e-10]
    if not prefix_rows:
        return None

    prefix_id = (
        f"L{source.size}_s{source.seed}_prefix_to_{boundary_load:.10g}"
    )
    safe_id = prefix_id.replace(".", "p").replace("-", "m")
    output_directory = Path(rescue_root) / "prefix" / source.name / safe_id
    source_folder = Path(source.folder).resolve()
    if output_directory.resolve().is_relative_to(source_folder):
        raise ValueError(f"Rescue output cannot be inside source folder: {output_directory}")
    return PrefixPlan(
        prefix_id=prefix_id,
        run_name=source.name,
        size=source.size,
        seed=source.seed,
        server=source.server,
        source_config=source.config_path,
        boundary_dump=boundary_dump,
        stop_load=boundary_load,
        expected_first_step=prefix_rows[0][0],
        expected_last_step=prefix_rows[-1][0],
        expected_first_load=prefix_rows[0][1],
        expected_last_load=prefix_rows[-1][1],
        output_directory=str(output_directory),
    )


def build_bounded_test_segment(
    source: SourceRun,
    intervals: Sequence[SchemaInterval],
    start_dump: DumpRecord,
    stop_dump: DumpRecord,
    rescue_root: Path,
) -> SegmentPlan:
    """Build one explicitly bounded test segment from two adjacent dumps.

    This is useful when the first available dump starts after the simulation's
    initial load.  It refuses to claim coverage outside the two dump states,
    and selects only bad-old rows strictly after the starting state and up to
    the stopping state.
    """

    if start_dump.state_load is None or start_dump.state_step is None:
        raise ValueError(f"Start dump lacks exact state metadata: {start_dump.path}")
    if stop_dump.state_load is None or stop_dump.state_step is None:
        raise ValueError(f"Stop dump lacks exact state metadata: {stop_dump.path}")
    if stop_dump.state_load <= start_dump.state_load:
        raise ValueError("Test segment dumps are not in increasing state-load order.")
    bad_rows = [
        key
        for interval in intervals
        if interval.sigma_status == "bad-old"
        for key in interval.row_keys
        if key[0] > start_dump.state_step and key[0] <= stop_dump.state_step
    ]
    if not bad_rows:
        raise ValueError(
            f"No bad-old rows between {start_dump.state_load:g} and "
            f"{stop_dump.state_load:g} for {source.name}."
        )
    segment_id = (
        f"L{source.size}_s{source.seed}_from_{start_dump.state_load:.10g}"
        f"_to_{stop_dump.state_load:.10g}"
    )
    safe_id = segment_id.replace(".", "p").replace("-", "m")
    output_directory = Path(rescue_root) / safe_id
    if output_directory.resolve().is_relative_to(Path(source.folder).resolve()):
        raise ValueError(f"Rescue output cannot be inside source folder: {output_directory}")
    return SegmentPlan(
        segment_id=segment_id,
        run_name=source.name,
        size=source.size,
        seed=source.seed,
        server=source.server,
        start_dump=start_dump,
        stop_load=stop_dump.state_load,
        expected_first_step=bad_rows[0][0],
        expected_last_step=bad_rows[-1][0],
        expected_first_load=bad_rows[0][1],
        expected_last_load=bad_rows[-1][1],
        output_directory=str(output_directory),
    )


def write_private_config(
    source_config: Path,
    destination: Path,
    segment: SegmentPlan,
) -> None:
    """Preserve numerical settings and write segment-only output overrides.

    Reuse/promote the config editing helpers currently in
    ``vtuBeforeReconnectionExtraction`` rather than adding another parser.
    Expected overrides are a unique private name, ``maxLoad``, disabled dumps
    and bulky VTU/debug output, and deterministic progress settings.  Do not
    change experiment, minimizer, tolerances, load increment, seed, topology or
    thread count.  Record a source-versus-private config diff in the manifest and
    reject any override outside an explicit allow-list.
    """

    source_config = Path(source_config).resolve()
    destination = Path(destination).resolve()
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    if destination == source_config:
        raise ValueError("Private rescue config cannot overwrite the source config.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite rescue config: {destination}")

    private_name = _private_run_name(segment)
    overrides = {
        "name": private_name,
        "maxLoad": f"{segment.stop_load:.17g}",
        "writeDumps": 0,
        "writeDebugVTUs": 0,
        "writeMeshVTUs": 0,
        # -1 makes the MTS2D loader non-interactive when the dump's historical
        # output path differs from the private rescue path.
        "showProgress": -1,
    }
    text = source_config.read_text()
    for key, value in overrides.items():
        text = _set_config_value(text, key, value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)


def write_prefix_config(
    source_config: Path,
    destination: Path,
    *,
    run_name: str,
    stop_load: float,
) -> None:
    """Create an isolated from-scratch config that writes the first dump."""

    source_config = Path(source_config).resolve()
    destination = Path(destination).resolve()
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    if destination == source_config or destination.exists():
        raise FileExistsError(f"Refusing to overwrite prefix config: {destination}")
    if stop_load <= 0:
        raise ValueError(f"Prefix stop load must be positive, got {stop_load}")
    text = source_config.read_text()
    for key, value in {
        "name": run_name,
        "maxLoad": f"{stop_load:.17g}",
        "writeDumps": 1,
        "writeDebugVTUs": 0,
        "writeMeshVTUs": 0,
        "showProgress": -1,
    }.items():
        text = _set_config_value(text, key, value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)


def run_prefix(
    source_config: Path,
    *,
    run_name: str,
    stop_load: float,
    output_directory: Path,
    executable: Path,
) -> Path:
    """Run from the configured start load through one exact first-dump load."""

    executable = Path(executable).resolve()
    source_config = Path(source_config).resolve()
    final_directory = Path(output_directory).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    if final_directory.exists():
        raise FileExistsError(f"Refusing to overwrite prefix output: {final_directory}")
    if final_directory.is_relative_to(source_config.parent):
        raise ValueError(f"Prefix output cannot be inside source folder: {final_directory}")
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    partial_directory = Path(
        tempfile.mkdtemp(prefix=f".{final_directory.name}.partial-", dir=final_directory.parent)
    )
    config_path = partial_directory / "prefix_config.conf"
    write_prefix_config(
        source_config,
        config_path,
        run_name=run_name,
        stop_load=stop_load,
    )
    command = [
        str(executable),
        "-c",
        str(config_path),
        "-o",
        str(partial_directory),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (partial_directory / "stdout.log").write_text(completed.stdout)
    (partial_directory / "stderr.log").write_text(completed.stderr)
    (partial_directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Prefix run {run_name} failed with exit code {completed.returncode}; "
            f"preserved {partial_directory}"
        )
    macros = sorted(partial_directory.rglob("macroData.csv"))
    if len(macros) != 1:
        raise RuntimeError(f"Expected one prefix macroData.csv, found {macros}")
    relative_macro = macros[0].relative_to(partial_directory)
    dumps = sorted(
        path for path in partial_directory.rglob("*")
        if path.is_file() and path.name.startswith("dump_") and path.stat().st_size > 0
    )
    load_increment = float(_config_value(source_config, "loadIncrement"))
    selected_dump, dump_inventory = select_prefix_dump(
        dumps,
        stop_load=stop_load,
        load_increment=load_increment,
    )
    manifest = {
        "status": "ran",
        "run_name": run_name,
        "stop_load": stop_load,
        "binary": str(executable),
        "binary_sha256": sha256_file(executable),
        "returncode": completed.returncode,
        "macro": str(final_directory / relative_macro),
        "dump": str(final_directory / selected_dump.relative_to(partial_directory)),
        "dump_load": prefix_dump_load(selected_dump),
        "dumps": [
            {
                "path": str(final_directory / path.relative_to(partial_directory)),
                "load": load,
            }
            for path, load in dump_inventory
        ],
    }
    (partial_directory / "prefix_result.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    partial_directory.rename(final_directory)
    return final_directory / relative_macro


def run_prefix_plan(prefix: PrefixPlan, *, executable: Path) -> Path:
    """Execute one frozen prefix plan without changing the source tree."""

    return run_prefix(
        Path(prefix.source_config),
        run_name=prefix.prefix_id,
        stop_load=prefix.stop_load,
        output_directory=Path(prefix.output_directory),
        executable=executable,
    )


def loader_smoke_test(
    segment: SegmentPlan,
    *,
    executable: Path,
    work_directory: Path,
) -> dict[str, object]:
    """Load one dump privately and record compatibility without touching source.

    Invoke MTS2D with an argument list (never ``shell=True``).  The source dump
    and config are read-only inputs; output is below ``work_directory``.  Capture
    stdout, stderr, exit status, binary checksum/commit and produced row keys.
    Confirm the meaning of dump-load exit code 2 for the selected binary.
    """

    executable = Path(executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    work_directory = Path(work_directory).resolve()
    if work_directory.exists():
        raise FileExistsError(f"Smoke-test directory already exists: {work_directory}")
    source_folder = _source_folder_from_dump(Path(segment.start_dump.path))
    increment = float(_config_value(source_folder / "config.conf", "loadIncrement"))
    dump_state_load = segment.start_dump.state_load or segment.start_dump.load
    smoke_segment = replace(
        segment,
        stop_load=dump_state_load + increment,
        output_directory=str(work_directory / "output"),
    )
    work_directory.mkdir(parents=True)
    config_path = work_directory / "smoke_config.conf"
    write_private_config(source_folder / "config.conf", config_path, smoke_segment)
    output_directory = Path(smoke_segment.output_directory)
    output_directory.mkdir(parents=True)
    command = [
        str(executable),
        "-d",
        smoke_segment.start_dump.path,
        "-c",
        str(config_path),
        "-o",
        str(output_directory),
        "-r",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (work_directory / "stdout.log").write_text(completed.stdout)
    (work_directory / "stderr.log").write_text(completed.stderr)
    (work_directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Loader smoke test failed with exit code {completed.returncode}; "
            f"see {work_directory / 'stderr.log'}"
        )
    macros = sorted(output_directory.rglob("macroData.csv"))
    if len(macros) != 1:
        raise RuntimeError(f"Expected one smoke-test macroData.csv, found {macros}")
    header, rows = _read_macro_rows(macros[0])
    result = {
        "status": "loaded",
        "dump": segment.start_dump.path,
        "dump_load": segment.start_dump.load,
        "binary": str(executable),
        "binary_sha256": sha256_file(executable),
        "exit_code": completed.returncode,
        "macro": str(macros[0]),
        "header": header,
        "first_key": _row_key(rows[0]),
        "last_key": _row_key(rows[-1]),
        "rows": len(rows),
    }
    (work_directory / "smoke_result.json").write_text(
        json.dumps(result, indent=2, default=list) + "\n"
    )
    return result


def run_segment(segment: SegmentPlan, *, executable: Path) -> Path:
    """Replay exactly one segment into its private output directory.

    TODO: write to a temporary sibling directory; preserve failures for
    inspection; atomically rename successful raw output; never overwrite a
    previous attempt.  This function runs no stitching and writes no shared
    files.  Follow the private subprocess pattern in
    ``vtuBeforeReconnectionExtraction.extract_dump``.
    """

    executable = Path(executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    dump = Path(segment.start_dump.path).resolve()
    if not dump.is_file():
        raise FileNotFoundError(dump)
    source_folder = _source_folder_from_dump(dump)
    final_directory = Path(segment.output_directory).resolve()
    if final_directory.is_relative_to(source_folder):
        raise ValueError(f"Rescue output cannot be inside source folder: {final_directory}")
    if final_directory.exists():
        raise FileExistsError(f"Refusing to overwrite rescue output: {final_directory}")
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    partial_directory = Path(
        tempfile.mkdtemp(prefix=f".{final_directory.name}.partial-", dir=final_directory.parent)
    )
    config_path = partial_directory / "rescue_config.conf"
    write_private_config(source_folder / "config.conf", config_path, segment)
    command = [
        str(executable),
        "-d",
        str(dump),
        "-c",
        str(config_path),
        "-o",
        str(partial_directory),
        "-r",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (partial_directory / "stdout.log").write_text(completed.stdout)
    (partial_directory / "stderr.log").write_text(completed.stderr)
    (partial_directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Segment {segment.segment_id} failed with exit code "
            f"{completed.returncode}; preserved {partial_directory}"
        )
    macros = sorted(partial_directory.rglob("macroData.csv"))
    if len(macros) != 1:
        raise RuntimeError(
            f"Segment {segment.segment_id} produced {len(macros)} macroData.csv files; "
            f"preserved {partial_directory}"
        )
    relative_macro = macros[0].relative_to(partial_directory)
    manifest = {
        "status": "ran",
        "segment_id": segment.segment_id,
        "dump": str(dump),
        "binary": str(executable),
        "binary_sha256": sha256_file(executable),
        "returncode": completed.returncode,
        "macro": str(final_directory / relative_macro),
    }
    (partial_directory / "run_result.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    partial_directory.rename(final_directory)
    return final_directory / relative_macro


def _validate_replay(
    replay_id: str,
    expected_first_step: int,
    expected_last_step: int,
    replay_macro: Path,
    source_macro: Path,
    tolerances: ValidationTolerance,
) -> Path:
    """Validate one replay before extracting corrected stress columns.

    Require unique agreement on ``ROW_KEY_COLUMNS`` and compare stable energy
    trajectory columns.  Restart-relative initial fields and stress fields are
    intentionally excluded: the former are re-established from the dump, and
    the latter are the schema outputs being repaired.  Require native
    new-schema ``avg_sigma12`` in replay output, finite values, and complete
    expected coverage.

    Write a compact validated CSV plus an atomic result manifest.  Energy
    mismatches are retained as explicit ``-1`` sigma sentinels so the campaign
    can continue; structural errors still raise and leave the replay unusable.
    """

    replay_macro = Path(replay_macro).resolve()
    source_macro = Path(source_macro).resolve()
    _, source_rows = _read_macro_rows(source_macro)
    replay_header, replay_rows = _read_macro_rows(replay_macro)
    if CORRECT_SIGMA_COLUMN not in replay_header:
        raise KeyError(
            f"Replay {replay_macro} has no native {CORRECT_SIGMA_COLUMN}; "
            "refusing to use a P12 substitute."
        )
    source_by_step: dict[int, dict[str, str]] = {}
    for row in source_rows:
        step, _ = _row_key(row)
        if step in source_by_step:
            raise ValueError(f"Duplicate source load_step {step} in {source_macro}")
        source_by_step[step] = row
    replay_by_step: dict[int, dict[str, str]] = {}
    for row in replay_rows:
        step, _ = _row_key(row)
        if step in replay_by_step:
            raise ValueError(f"Duplicate replay load_step {step} in {replay_macro}")
        replay_by_step[step] = row

    expected_steps = range(expected_first_step, expected_last_step + 1)
    expected_rows = [source_by_step[step] for step in expected_steps if step in source_by_step]
    if not expected_rows:
        raise ValueError(f"No source rows for {replay_id} in {source_macro}")
    validated_rows: list[dict[str, str]] = []
    invalid_steps: list[int] = []
    mismatch_columns: dict[str, int] = {}
    for source_row in expected_rows:
        step, source_load = _row_key(source_row)
        replay_row = replay_by_step.get(step)
        if replay_row is None:
            raise ValueError(f"Replay {replay_macro} is missing load_step {step}")
        replay_load = _numeric(replay_row, "load", replay_macro)
        if abs(replay_load - source_load) > tolerances.load_atol:
            raise ValueError(
                f"Replay load mismatch at step {step}: {replay_load:.17g} "
                f"versus {source_load:.17g}"
            )
        row_mismatches: list[str] = []
        for column in EVOLUTION_VALIDATION_COLUMNS:
            source_value = _numeric(source_row, column, source_macro)
            replay_value = _numeric(replay_row, column, replay_macro)
            rtol, atol = tolerances.energy_rtol, tolerances.energy_atol
            if abs(replay_value - source_value) > atol + rtol * abs(source_value):
                row_mismatches.append(column)
                mismatch_columns[column] = mismatch_columns.get(column, 0) + 1
        for column in RESCUED_COLUMNS:
            _numeric(replay_row, column, replay_macro)
        if row_mismatches:
            invalid_steps.append(step)
            validated_rows.append(
                {
                    "load_step": replay_row["load_step"],
                    "load": replay_row["load"],
                    **{column: INVALID_SIGMA_SENTINEL for column in RESCUED_COLUMNS},
                }
            )
        else:
            validated_rows.append(
                {
                    column: replay_row[column]
                    for column in ("load_step", "load", *RESCUED_COLUMNS)
                }
            )

    output_path = replay_macro.parent / "validated_sigma.csv"
    if output_path.exists():
        raise FileExistsError(output_path)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("load_step", "load", *RESCUED_COLUMNS)
        )
        writer.writeheader()
        writer.writerows(validated_rows)
    result_path = replay_macro.parent / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    result = {
        "status": "validated_with_sentinels" if invalid_steps else "validated",
        "segment_id": replay_id,
        "replay_id": replay_id,
        "source_macro": str(source_macro),
        "replay_macro": str(replay_macro),
        "validated_sigma": str(output_path),
        "first_step": validated_rows[0]["load_step"],
        "last_step": validated_rows[-1]["load_step"],
        "rows": len(validated_rows),
        "invalid_rows": len(invalid_steps),
        "first_invalid_step": invalid_steps[0] if invalid_steps else None,
        "last_invalid_step": invalid_steps[-1] if invalid_steps else None,
        "mismatch_columns": mismatch_columns,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return output_path


def validate_segment(
    segment: SegmentPlan,
    replay_macro: Path,
    source_macro: Path,
    tolerances: ValidationTolerance,
) -> Path:
    """Validate one dump-to-dump replay."""

    return _validate_replay(
        segment.segment_id,
        segment.expected_first_step,
        segment.expected_last_step,
        replay_macro,
        source_macro,
        tolerances,
    )


def validate_prefix(
    prefix: PrefixPlan,
    replay_macro: Path,
    source_macro: Path,
    tolerances: ValidationTolerance,
) -> Path:
    """Validate the from-scratch replay through the first usable dump."""

    return _validate_replay(
        prefix.prefix_id,
        prefix.expected_first_step,
        prefix.expected_last_step,
        replay_macro,
        source_macro,
        tolerances,
    )


def stitch_rescued_sigma(
    source: SourceRun,
    intervals: Sequence[SchemaInterval],
    validated_segments: Sequence[Path],
    destination: Path,
) -> Path:
    """Create a new canonical CSV while preserving all non-target source values.

    Start from a canonicalized copy using the parsing ideas in
    ``Management.updateCSV.fix_mixed_macrodata_csv``.  Replace
    ``RESCUED_COLUMNS`` only for ``bad-old`` rows.  Preserve correct-new stress
    rows exactly.  Require exactly one validated source for every rescued row,
    identical overlap values, no missing/extra rows and unchanged values for all
    columns outside the explicit replacement set.  Write via a temporary file,
    then atomically rename to ``macroData.sigmaRescued.csv``.  Never replace the
    original macro CSV.
    """

    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite stitched output: {destination}")
    source_header, source_rows = _read_macro_rows(Path(source.macro_path))
    bad_keys = {
        key for interval in intervals if interval.sigma_status == "bad-old"
        for key in interval.row_keys
    }
    source_by_key = {_row_key(row): row for row in source_rows}
    if len(source_by_key) != len(source_rows):
        raise ValueError(f"Duplicate source row key in {source.macro_path}")

    rescued_by_key: dict[tuple[int, float], dict[str, str]] = {}
    provenance: dict[tuple[int, float], str] = {}
    rescue_status: dict[tuple[int, float], str] = {}
    for validated_path in validated_segments:
        validated_path = Path(validated_path)
        _, rows = _read_macro_rows(validated_path)
        segment_id = validated_path.parent.name
        for row in rows:
            key = _row_key(row)
            if key not in bad_keys:
                continue
            if key in rescued_by_key:
                previous = rescued_by_key[key]
                for column in RESCUED_COLUMNS:
                    old_value = _numeric(previous, column, validated_path)
                    new_value = _numeric(row, column, validated_path)
                    if abs(old_value - new_value) > 1e-10 + 1e-8 * abs(old_value):
                        raise ValueError(
                            f"Conflicting rescue values for {key}, {column}: "
                            f"{old_value} versus {new_value}"
                        )
                continue
            rescued_by_key[key] = {column: row[column] for column in RESCUED_COLUMNS}
            provenance[key] = segment_id
            rescue_status[key] = (
                "invalid_energy_sentinel"
                if any(row[column] == INVALID_SIGMA_SENTINEL for column in RESCUED_COLUMNS)
                else "validated"
            )
    missing = sorted(bad_keys - rescued_by_key.keys())
    if missing:
        raise ValueError(f"No validated rescue covers {len(missing)} rows; first={missing[0]}")

    output_header = list(source_header)
    for column in RESCUED_COLUMNS:
        if column not in output_header:
            output_header.append(column)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite partial stitched output: {temporary}")
    audit_path = destination.with_name("audit.csv")
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {audit_path}")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_header, extrasaction="ignore")
        writer.writeheader()
        for row in source_rows:
            key = _row_key(row)
            output_row = {column: row.get(column, "") for column in output_header}
            if key in bad_keys:
                output_row.update(rescued_by_key[key])
            writer.writerow(output_row)
    os.replace(temporary, destination)

    with audit_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("load_step", "load", "source_schema", "segment_id", "rescue_status")
        )
        for row in source_rows:
            key = _row_key(row)
            writer.writerow(
                (
                    *key,
                    "bad-old" if key in bad_keys else "correct-new",
                    provenance.get(key, ""),
                    rescue_status.get(key, "not_rescued"),
                )
            )
    return destination


def submit_segments(
    segments_manifest: Path,
    *,
    server: str,
    dry_run: bool = True,
) -> None:
    """Prepare Slurm work; submission must remain dry-run by default.

    During L=50 testing, reuse ``queueLocalJobs.get_batch_script`` for a few
    explicit jobs.  Before full scale, implement a Slurm array where each task
    reads exactly one immutable JSONL record and invokes ``run-segment``.  Reuse
    ``runOnCluster.run_remote_command`` for the final remote submission.  Print
    resolved source/output paths, task count, CPUs, memory and time limits in a
    dry run, and require an explicit ``--submit`` switch for external changes.
    """

    if dry_run:
        raise NotImplementedError("Dry-run rendering has not been implemented yet.")
    raise RuntimeError("Submission is disabled in the sigma-rescue skeleton.")


def segment_to_json(segment: SegmentPlan, path: Path) -> Path:
    """Write one frozen segment record without overwriting an existing plan."""

    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(segment), indent=2) + "\n")
    return path


def prefix_to_json(prefix: PrefixPlan, path: Path) -> Path:
    """Write one frozen prefix record without overwriting an existing file."""

    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(prefix), indent=2) + "\n")
    return path


def segment_from_json(path: Path) -> SegmentPlan:
    """Load one immutable segment record and reject unknown fields."""

    data = json.loads(Path(path).read_text())
    expected = set(SegmentPlan.__dataclass_fields__)
    if set(data) != expected:
        raise ValueError(
            f"Unexpected segment manifest fields in {path}: "
            f"expected={sorted(expected)}, got={sorted(data)}"
        )
    data["start_dump"] = DumpRecord(**data["start_dump"])
    return SegmentPlan(**data)


def prefix_from_json(path: Path) -> PrefixPlan:
    """Load one immutable prefix record and reject unknown fields."""

    data = json.loads(Path(path).read_text())
    expected = set(PrefixPlan.__dataclass_fields__)
    if set(data) != expected:
        raise ValueError(
            f"Unexpected prefix manifest fields in {path}: "
            f"expected={sorted(expected)}, got={sorted(data)}"
        )
    data["boundary_dump"] = DumpRecord(**data["boundary_dump"])
    return PrefixPlan(**data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-test-sequence",
        action="store_true",
        help="Print the staged L=50, seed=0 implementation/test sequence.",
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser(
        "run-segment", help="Replay one frozen segment into its private output tree."
    )
    run_parser.add_argument("--segment-json", type=Path, required=True)
    run_parser.add_argument("--executable", type=Path, required=True)
    prefix_parser = subparsers.add_parser(
        "run-prefix", help="Run from the start through one first-dump load."
    )
    prefix_parser.add_argument("--source-config", type=Path, required=True)
    prefix_parser.add_argument("--run-name", required=True)
    prefix_parser.add_argument("--stop-load", type=float, required=True)
    prefix_parser.add_argument("--output-directory", type=Path, required=True)
    prefix_parser.add_argument("--executable", type=Path, required=True)
    prefix_plan_parser = subparsers.add_parser(
        "run-prefix-plan", help="Replay one frozen from-scratch prefix plan."
    )
    prefix_plan_parser.add_argument("--prefix-json", type=Path, required=True)
    prefix_plan_parser.add_argument("--executable", type=Path, required=True)
    smoke_parser = subparsers.add_parser(
        "loader-smoke", help="Load one frozen dump into a private smoke-test tree."
    )
    smoke_parser.add_argument("--segment-json", type=Path, required=True)
    smoke_parser.add_argument("--executable", type=Path, required=True)
    smoke_parser.add_argument("--work-directory", type=Path, required=True)
    validate_parser = subparsers.add_parser(
        "validate-segment", help="Validate replay evolution and extract sigma values."
    )
    validate_parser.add_argument("--segment-json", type=Path, required=True)
    validate_parser.add_argument("--source-macro", type=Path, required=True)
    validate_parser.add_argument("--replay-macro", type=Path, required=True)
    validate_prefix_parser = subparsers.add_parser(
        "validate-prefix", help="Validate a from-scratch prefix replay."
    )
    validate_prefix_parser.add_argument("--prefix-json", type=Path, required=True)
    validate_prefix_parser.add_argument("--source-macro", type=Path, required=True)
    validate_prefix_parser.add_argument("--replay-macro", type=Path, required=True)
    prepare_parser = subparsers.add_parser(
        "prepare-plan",
        help="Inspect one remote source and write local frozen segment manifests.",
    )
    prepare_parser.add_argument("--server", required=True)
    prepare_parser.add_argument("--source-folder", required=True)
    prepare_parser.add_argument("--macro-path", type=Path, required=True)
    prepare_parser.add_argument("--rescue-root", required=True)
    prepare_parser.add_argument("--destination", type=Path, required=True)
    prepare_parser.add_argument("--format-era", required=True)
    prepare_parser.add_argument("--reconnection", default="none")
    prepare_parser.add_argument(
        "--fingerprint-dumps",
        action="store_true",
        help="Hash every remote dump; omit for a lightweight metadata-only plan.",
    )
    prepare_parser.add_argument(
        "--read-dump-states",
        action="store_true",
        help="Read exact load/step values from compressed dumps before planning.",
    )
    inventory_parser = subparsers.add_parser(
        "inventory-remote",
        help="Collect lightweight remote file metadata without planning jobs.",
    )
    inventory_parser.add_argument("--server", required=True)
    inventory_parser.add_argument("--source-folder", required=True)
    inventory_parser.add_argument("--format-era", required=True)
    inventory_parser.add_argument("--fingerprint-dumps", action="store_true")
    inventory_parser.add_argument("--read-dump-states", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.show_test_sequence:
        print(TEST_SEQUENCE)
        return 0
    if args.command == "run-segment":
        segment = segment_from_json(args.segment_json)
        replay_macro = run_segment(segment, executable=args.executable)
        print(replay_macro)
        return 0
    if args.command == "run-prefix":
        prefix_macro = run_prefix(
            args.source_config,
            run_name=args.run_name,
            stop_load=args.stop_load,
            output_directory=args.output_directory,
            executable=args.executable,
        )
        print(prefix_macro)
        return 0
    if args.command == "run-prefix-plan":
        prefix_macro = run_prefix_plan(
            prefix_from_json(args.prefix_json), executable=args.executable
        )
        print(prefix_macro)
        return 0
    if args.command == "loader-smoke":
        segment = segment_from_json(args.segment_json)
        result = loader_smoke_test(
            segment, executable=args.executable, work_directory=args.work_directory
        )
        print(json.dumps(result, indent=2, default=list))
        return 0
    if args.command == "validate-segment":
        segment = segment_from_json(args.segment_json)
        validated = validate_segment(
            segment,
            args.replay_macro,
            args.source_macro,
            ValidationTolerance(),
        )
        print(validated)
        return 0
    if args.command == "validate-prefix":
        prefix = prefix_from_json(args.prefix_json)
        validated = validate_prefix(
            prefix,
            args.replay_macro,
            args.source_macro,
            ValidationTolerance(),
        )
        print(validated)
        return 0
    if args.command == "prepare-plan":
        plan = prepare_remote_plan(
            server=args.server,
            source_folder=args.source_folder,
            macro_path=args.macro_path,
            rescue_root=args.rescue_root,
            destination=args.destination,
            format_era=args.format_era,
            reconnection=args.reconnection,
            fingerprint_dumps=args.fingerprint_dumps,
            read_dump_states=args.read_dump_states,
        )
        print(plan)
        return 0
    if args.command == "inventory-remote":
        inventory = remote_inventory(
            args.server,
            args.source_folder,
            format_era=args.format_era,
            fingerprint_dumps=args.fingerprint_dumps,
            read_dump_states=args.read_dump_states,
        )
        print(
            json.dumps(
                {
                    "folder": inventory["folder"],
                    "dumps": len(inventory["dumps"]),
                    "first_dump": inventory["dumps"][0],
                    "last_dump": inventory["dumps"][-1],
                },
                indent=2,
            )
        )
        return 0
    build_parser().print_help()
    print("\nJob submission remains intentionally disabled in this development step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
