"""Selection of simulation families used by reconnection extraction/plots."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional

from Management.jobs import size_scaling_job


@dataclass(frozen=True)
class SimulationJob:
    folder: Path
    job_type: str
    size: Optional[int]
    seed: Optional[int]


def _expected_jobs(job_type: str) -> Dict[str, tuple[int, int]]:
    if job_type != "size-scaling":
        raise ValueError(
            f"Unknown job type {job_type!r}; available job types: 'size-scaling'."
        )
    groups, _ = size_scaling_job(reconnection="edgeFlip")
    return {
        config.name: (int(config.rows), int(config.seed))
        for group in groups
        for config in group
    }


def discover_simulation_jobs(
    data_root: Path,
    job_type: str = "size-scaling",
    size: Optional[int] = None,
    require_dumps: bool = False,
    require_extraction: bool = False,
) -> List[SimulationJob]:
    """Find completed local folders belonging to a known job family.

    Missing configurations and folders without requested data are skipped so a
    partially completed batch remains usable.  A folder that exists but lacks
    its config is treated as an error rather than silently reclassifying it.
    """
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing data root: {data_root}")
    expected = _expected_jobs(job_type)
    jobs = []
    for name, (job_size, seed) in expected.items():
        if size is not None and job_size != size:
            continue
        folder = data_root / name
        if not folder.is_dir():
            continue
        config = folder / "config.conf"
        if not config.is_file():
            raise FileNotFoundError(f"Expected config file is missing: {config}")
        if require_dumps and not (folder / "dumps").is_dir():
            continue
        if require_extraction:
            extraction_folder = folder / "beforeReconnectionVtuData"
            if not extraction_folder.is_dir():
                continue
            # A running extractor creates this directory before its first
            # result.  Do not hand such an empty/in-progress job to plotting.
            if not any(
                path.is_dir()
                and not path.name.startswith(".")
                and re.match(r".+_step\d+$", path.name)
                for path in extraction_folder.iterdir()
            ):
                continue
        jobs.append(SimulationJob(folder, job_type, job_size, seed))
    return sorted(jobs, key=lambda job: (job.size, job.seed, job.folder.name))
