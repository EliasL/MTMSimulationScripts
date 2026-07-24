#!/usr/bin/env python3
"""Extract the first completed reconnection-minimization folder per dump.

Each dump is resumed with minimization logging in a private working directory.
The source simulation is never used as an MTS2D output directory.  Once a
completed minimization step contains a matched ``_pre``/``_post`` VTU pair,
only the process started for that dump is stopped and the whole step folder is
copied to ``beforeReconnectionVtuData`` in the source simulation folder.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


OUTPUT_FOLDER_NAME = "beforeReconnectionVtuData"
MINIMIZATION_FOLDER = Path("data/minimizationData")
STEP_PATTERN = re.compile(r"step(\d+)$")
LOAD_PATTERN = re.compile(r"_l(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _config_value(text: str, key: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^#\r\n]+)", re.MULTILINE)
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one '{key}' setting in config, found {len(matches)}."
        )
    return matches[0].strip()


def _set_config_value(text: str, key: str, value: object) -> str:
    pattern = re.compile(
        rf"^(?P<prefix>\s*{re.escape(key)}\s*=\s*)"
        r"(?P<value>[^#\r\n]*?)(?P<suffix>\s*(?:#.*)?)$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) > 1:
        raise ValueError(f"Config contains more than one '{key}' setting.")
    if not matches:
        separator = "" if text.endswith("\n") else "\n"
        return f"{text}{separator}{key} = {value}\n"
    return pattern.sub(
        lambda match: f"{match.group('prefix')}{value}{match.group('suffix')}",
        text,
        count=1,
    )


def _extraction_run_name(source_name: str) -> str:
    existing = re.search(r"logDuringMinimization[01]", source_name)
    if existing:
        return (
            source_name[: existing.start()]
            + "logDuringMinimization1"
            + source_name[existing.end() :]
        )

    seed_suffix = re.search(r"s-?\d+$", source_name)
    if seed_suffix:
        return (
            source_name[: seed_suffix.start()]
            + "logDuringMinimization1"
            + source_name[seed_suffix.start() :]
        )
    return source_name + "logDuringMinimization1"


def write_extraction_config(source: Path, destination: Path) -> str:
    """Write a logging-only variant while preserving every unknown setting."""
    text = source.read_text()
    run_name = _extraction_run_name(_config_value(text, "name"))
    settings = {
        "name": run_name,
        # Do not let an endpoint dump stop before a later reconnection event.
        "maxLoad": "1e100",
        "logDuringMinimization": 1,
        "fullMinimizationLogging": 0,
        "writeDumps": 0,
        "writeDebugVTUs": 0,
        # MTS2D otherwise deletes small plastic-event minimization folders.
        "plasticityEventThreshold": 0,
    }
    for key, value in settings.items():
        text = _set_config_value(text, key, value)
    destination.write_text(text)
    return run_name


def _is_dump(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.is_file()
        and not name.startswith((".", "broken_"))
        and not name.endswith(".tmp.xml")
        and name.endswith((".xml", ".xml.gz", ".mtsb"))
    )


def find_dumps(folder: Path) -> List[Path]:
    dumps = [path for path in folder.iterdir() if _is_dump(path)]
    empty = [path for path in dumps if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"Empty dump file: {empty[0]}")

    def sort_key(path: Path) -> Tuple[int, float, str]:
        match = LOAD_PATTERN.search(path.name)
        return (0, float(match.group(1)), path.name) if match else (1, 0.0, path.name)

    return sorted(dumps, key=sort_key)


def reconnection_pairs(step_folder: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for before in step_folder.glob("*_pre.*.vtu"):
        after = before.with_name(before.name.replace("_pre.", "_post.", 1))
        if after.is_file():
            pairs.append((before, after))
    return sorted(pairs)


def find_completed_event_step(
    minimization_folder: Path, process_finished: bool = False
) -> Optional[Path]:
    if not minimization_folder.is_dir():
        return None

    steps = []
    for path in minimization_folder.iterdir():
        match = STEP_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            steps.append((int(match.group(1)), path))
    steps.sort()

    for index, (_, path) in enumerate(steps):
        has_successor = index < len(steps) - 1
        is_complete = process_finished or has_successor or (path / "collection.pvd").is_file()
        if is_complete and reconnection_pairs(path):
            return path
    return None


def wait_for_event_step(
    process: subprocess.Popen,
    minimization_folder: Path,
    poll_interval: float,
    timeout: Optional[float],
) -> Path:
    start = time.monotonic()
    while True:
        return_code = process.poll()
        event = find_completed_event_step(
            minimization_folder, process_finished=return_code is not None
        )
        if event is not None:
            return event
        if return_code is not None:
            raise RuntimeError(
                "MTS2D stopped before producing a completed reconnection event "
                f"(exit code {return_code})."
            )
        if timeout is not None and time.monotonic() - start >= timeout:
            raise TimeoutError(
                f"No reconnection event was completed within {timeout:g} seconds."
            )
        time.sleep(poll_interval)


def stop_owned_process(process: subprocess.Popen, grace_period: float = 10.0) -> None:
    """Stop only this exact Popen child; never search for MTS2D processes."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_period)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _existing_result(output_folder: Path, dump_name: str) -> Optional[Path]:
    pattern = re.compile(rf"{re.escape(dump_name)}_step\d+$")
    matches = [
        path
        for path in output_folder.iterdir()
        if path.is_dir() and pattern.fullmatch(path.name)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple extracted results found for {dump_name}.")
    if matches and not reconnection_pairs(matches[0]):
        raise RuntimeError(f"Existing result has no matched pre/post pair: {matches[0]}")
    return matches[0] if matches else None


def _copy_step(step_folder: Path, output_folder: Path, dump_name: str) -> Path:
    destination = output_folder / f"{dump_name}_{step_folder.name}"
    if destination.exists():
        raise FileExistsError(f"Will not overwrite existing result: {destination}")

    temporary = output_folder / f".{destination.name}.copying-{uuid.uuid4().hex}"
    try:
        shutil.copytree(step_folder, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def extract_dump(
    dump: Path,
    source_config: Path,
    executable: Path,
    output_folder: Path,
    work_parent: Path,
    poll_interval: float,
    timeout: Optional[float],
) -> Path:
    existing = _existing_result(output_folder, dump.name)
    if existing is not None:
        print(
            f"Skipping {dump.name}; result already exists: {existing.name}",
            flush=True,
        )
        return existing

    workspace = Path(tempfile.mkdtemp(prefix=f"{dump.name}-", dir=work_parent))
    private_output = workspace / "output"
    private_output.mkdir()
    extraction_config = workspace / "config.conf"
    run_name = write_extraction_config(source_config, extraction_config)
    minimization_folder = private_output / run_name / MINIMIZATION_FOLDER
    command = [
        str(executable),
        "-d",
        str(dump),
        "-c",
        str(extraction_config),
        "-o",
        str(private_output),
    ]

    process = None
    copied = None
    try:
        print(f"Running from {dump.name}", flush=True)
        process = subprocess.Popen(command, start_new_session=True)
        event_step = wait_for_event_step(
            process, minimization_folder, poll_interval, timeout
        )
        stop_owned_process(process)
        copied = _copy_step(event_step, output_folder, dump.name)
        print(
            f"Saved {copied.name} "
            f"({len(reconnection_pairs(copied))} reconnection pair(s))",
            flush=True,
        )
        return copied
    finally:
        if process is not None:
            stop_owned_process(process)
        if copied is not None:
            shutil.rmtree(workspace)
        else:
            print(f"Preserving failed run for inspection: {workspace}")


def extract_simulation(
    simulation_folder: Path,
    executable: Path,
    poll_interval: float = 2.0,
    timeout: Optional[float] = None,
) -> List[Path]:
    simulation_folder = simulation_folder.expanduser().resolve()
    executable = executable.expanduser().resolve()
    source_config = simulation_folder / "config.conf"
    dump_folder = simulation_folder / "dumps"
    if not source_config.is_file():
        raise FileNotFoundError(f"Missing simulation config: {source_config}")
    if not dump_folder.is_dir():
        raise FileNotFoundError(f"Missing dumps folder: {dump_folder}")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"MTS2D executable is missing or not executable: {executable}")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive.")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive.")

    dumps = find_dumps(dump_folder)
    if not dumps:
        raise FileNotFoundError(f"No dump files found in {dump_folder}")

    output_folder = simulation_folder / OUTPUT_FOLDER_NAME
    output_folder.mkdir(exist_ok=True)
    work_parent = output_folder / ".work"
    work_parent.mkdir(exist_ok=True)

    results = []
    for dump in dumps:
        results.append(
            extract_dump(
                dump,
                source_config,
                executable,
                output_folder,
                work_parent,
                poll_interval,
                timeout,
            )
        )
    try:
        work_parent.rmdir()
    except OSError:
        pass
    return results


def parse_args() -> argparse.Namespace:
    default_executable = (
        Path(__file__).resolve().parents[2] / "MTS2D/build-release/MTS2D"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "simulation_folder",
        type=Path,
        help="Folder containing config.conf and dumps/.",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=default_executable,
        help=f"MTS2D executable (default: {default_executable}).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between minimization-folder checks (default: 2).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Optional maximum seconds to run each dump; the default has no timeout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = extract_simulation(
        args.simulation_folder,
        args.executable,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )
    print(f"Finished: {len(results)} dump(s) have extracted reconnection data.")


if __name__ == "__main__":
    main()
