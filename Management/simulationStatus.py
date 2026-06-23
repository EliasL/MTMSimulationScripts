from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

from tabulate import tabulate
from tqdm import tqdm

from Management.connectToCluster import Servers, connectToCluster
from Management.configGenerator import SimulationConfig
from Management.jobManager import JobManager
from Plotting.remotePlotting import (
    MACRO_PATH,
    REMOTE_FOLDER_NAME,
    _discover_remote_data_paths,
    _list_remote_output_folders,
    _normalize_remote_path,
)


@dataclass(frozen=True)
class CsvSource:
    path: str
    source: str


@dataclass(frozen=True)
class SimulationStatus:
    name: str
    label: str | None
    progress: float | None
    current_load: float | None
    final_load: float | None
    state: str
    running: bool | None
    server: str
    csv_path: str


def flatten_configs(configs) -> list[SimulationConfig]:
    if isinstance(configs, SimulationConfig):
        return [configs]

    flat = []
    for item in configs:
        if isinstance(item, SimulationConfig):
            flat.append(item)
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes, os.PathLike)):
            flat.extend(flatten_configs(item))
        else:
            raise TypeError(f"Expected SimulationConfig, got {type(item).__name__}.")
    return flat


def flatten_labels(labels) -> list[str]:
    if isinstance(labels, str):
        return [labels]

    flat = []
    for item in labels:
        if isinstance(item, str):
            flat.append(item)
        elif item is None:
            flat.append("")
        elif isinstance(item, Iterable) and not isinstance(item, (bytes, os.PathLike)):
            flat.extend(flatten_labels(item))
        else:
            flat.append(str(item))
    return flat


def _split_configs_and_labels(configs, labels=None):
    if labels is not None:
        return configs, flatten_labels(labels)

    if (
        isinstance(configs, tuple)
        and len(configs) == 2
        and isinstance(configs[1], Iterable)
        and not isinstance(configs[1], (str, bytes, os.PathLike))
    ):
        return configs[0], flatten_labels(configs[1])

    return configs, None


def config_name(config: SimulationConfig) -> str:
    name = getattr(config, "name", None)
    return name if name is not None else config.generate_name(withExtension=False)


def _path_config_name(path: str | os.PathLike) -> str:
    path = Path(path)
    return path.parent.name if path.name == "macroData.csv" else path.stem


def _short_source_name(source: str) -> str:
    if not source:
        return ""
    if source.startswith("/"):
        return "local"
    return source.split(":", 1)[0].split(".", 1)[0]


def _local_csv_sources(configs: list[SimulationConfig]) -> dict[str, CsvSource]:
    paths = {}
    local_base = Path(Servers.local_path_mac) / "MTS2D_output"
    has_local_base = local_base.is_dir()
    for config in configs:
        name = config_name(config)
        for folder, source in [(Path("/tmp/MTS2D"), "tmp"), (Path(MACRO_PATH), "macro")]:
            candidate = folder / f"{name}.csv"
            if candidate.is_file():
                paths[name] = CsvSource(str(candidate), source)
                break

        if name in paths:
            continue
        if not has_local_base:
            continue
        local_candidate = local_base / name / "macroData.csv"
        if local_candidate.is_file():
            paths[name] = CsvSource(str(local_candidate), "local")
    return paths


def _remote_csv_sources(configs: list[SimulationConfig]) -> dict[str, CsvSource]:
    if not configs or not Servers.search_servers:
        return {}

    sources = {}
    server_names = ", ".join(_short_source_name(server) for server in Servers.search_servers)
    print(
        f"Checking {len(Servers.search_servers)} server(s) for {len(configs)} config(s): {server_names}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=len(Servers.search_servers)) as executor:
        future_to_server = {
            executor.submit(_remote_csv_sources_from_server, server, configs): server
            for server in Servers.search_servers
        }
        with tqdm(total=len(future_to_server), desc="Checking servers", unit="server") as progress:
            for future in as_completed(future_to_server):
                server = future_to_server[future]
                server_name = _short_source_name(server)
                progress.set_postfix_str(server_name)
                try:
                    server_sources = future.result()
                except Exception as exc:
                    tqdm.write(f"{server_name} CSV search failed: {exc}")
                    progress.update(1)
                    continue
                if server_sources:
                    tqdm.write(f"{server_name}: found {len(server_sources)} CSV file(s).")
                for name, source in server_sources.items():
                    sources.setdefault(name, source)
                progress.update(1)
    return sources


def _remote_csv_sources_from_server(
    server: str,
    configs: list[SimulationConfig],
) -> dict[str, CsvSource]:
    if server.startswith("/"):
        return {}

    ssh = connectToCluster(server, False)
    if ssh is None:
        return {}

    result = {}
    server_name = _short_source_name(server)
    names = {config_name(config) for config in configs}
    os.makedirs(MACRO_PATH, exist_ok=True)

    try:
        data_paths = _discover_remote_data_paths(ssh, server)
        if not data_paths:
            return {}

        sftp = ssh.open_sftp()
        try:
            for data_path in data_paths:
                folders = set(_list_remote_output_folders(ssh, data_path))
                for name in names & folders:
                    if name in result:
                        continue
                    remote_path = _normalize_remote_path(
                        data_path,
                        REMOTE_FOLDER_NAME,
                        name,
                        "macroData.csv",
                    )
                    local_path = Path(MACRO_PATH) / f"{name}.csv"
                    sftp.get(remote_path, str(local_path))
                    result[name] = CsvSource(str(local_path), server_name)
        finally:
            sftp.close()
    finally:
        ssh.close()

    return result


def _csv_sources(
    configs: list[SimulationConfig],
    force_update: bool,
    search_remote: bool | str,
) -> dict[str, CsvSource]:
    if force_update:
        print("Skipping local/cache CSV check because force_update=True.", flush=True)
        sources = {}
    else:
        print(f"Checking local/cache CSV files for {len(configs)} config(s)...", flush=True)
        sources = _local_csv_sources(configs)
        print(f"Found {len(sources)} local/cache CSV file(s).", flush=True)

    if search_remote is False:
        print("Server status check disabled.", flush=True)
        return sources

    if search_remote == "conditional":
        remote_configs = []
        for config in configs:
            source = sources.get(config_name(config))
            if source is None:
                remote_configs.append(config)
                continue
            _, _, state, _ = _csv_status(config, source.path)
            if state != "Done":
                remote_configs.append(config)
        done_locally = len(configs) - len(remote_configs)
        print(
            f"Conditional server status check: {done_locally} done locally, "
            f"{len(remote_configs)} missing or unfinished.",
            flush=True,
        )
    elif search_remote is True:
        remote_configs = configs
        print(f"Server status check requested for all {len(remote_configs)} config(s).", flush=True)
    else:
        raise ValueError("search_remote must be False, True, or 'conditional'.")

    if not remote_configs:
        print("Skipping server status check.", flush=True)
        return sources

    for name, source in _remote_csv_sources(remote_configs).items():
        sources[name] = source
    return sources


def _headers_from_line(line: str, path: str | os.PathLike) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("#HEADER:"):
        stripped = stripped.split(":", 1)[1].strip()
    headers = stripped.split(",")
    if not headers or headers == [""]:
        raise RuntimeError(f"{path} has no CSV header.")
    return headers


def _last_complete_csv_row(path: str | os.PathLike) -> dict[str, str] | None:
    with open(path, encoding="utf-8", errors="replace") as file:
        header_line = file.readline()
        if not header_line:
            raise RuntimeError(f"{path} has no CSV header.")
        headers = _headers_from_line(header_line, path)
        row = None
        saw_data = False

        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#HEADER:"):
                headers = _headers_from_line(stripped, path)
                continue
            if stripped.startswith("#"):
                continue
            values = stripped.split(",")
            if values == headers:
                continue
            saw_data = True
            if len(values) == len(headers):
                row = dict(zip(headers, values))
            elif len(values) == len(headers) - 1:
                row = dict(zip(headers[1:], values))

    if row is not None:
        return row
    if not saw_data:
        return None
    raise RuntimeError(f"{path} has data rows, but none match the active CSV header.")


def _float_value(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        if key in row and row[key] != "":
            return float(row[key])
    return None


def _progress_from_load(current_load: float | None, final_load: float | None) -> float | None:
    if current_load is None or final_load is None:
        return None
    if final_load == 0.0:
        raise ValueError("Cannot compute load progress because final load is zero.")
    return max(0.0, min(100.0, 100.0 * current_load / final_load))


def _csv_status(config: SimulationConfig, csv_path: str | None) -> tuple[float | None, float | None, str, str]:
    if csv_path is None:
        return None, None, "Missing CSV", ""

    row = _last_complete_csv_row(csv_path)
    if row is None:
        return 0.0, None, "No data", str(csv_path)

    current_load = _float_value(row, "load", "Load")
    progress = _progress_from_load(current_load, float(config.maxLoad))
    if progress is None:
        state = "Unknown"
    elif progress >= 100.0:
        state = "Done"
    else:
        state = "Not running"
    return progress, current_load, state, str(csv_path)


def _running_processes_by_name() -> dict[str, object]:
    manager = JobManager()
    manager.findProcesses()
    return {
        process.name: process
        for process in manager.processes
        if hasattr(process, "name") and process.name
    }


def collect_status(
    configs,
    force_update: bool = False,
    check_running: bool = False,
    search_remote: bool | str = "conditional",
    labels=None,
) -> list[SimulationStatus]:
    configs, labels = _split_configs_and_labels(configs, labels)
    configs = flatten_configs(configs)
    if not configs:
        return []
    if labels is not None and len(labels) != len(configs):
        raise ValueError(f"Expected {len(configs)} labels, got {len(labels)}.")

    csv_by_name = _csv_sources(configs, force_update, search_remote)

    rows = []
    incomplete = False
    for index, config in enumerate(configs):
        name = config_name(config)
        csv_source = csv_by_name.get(name)
        csv_path = None if csv_source is None else csv_source.path
        progress, current_load, state, csv_path = _csv_status(config, csv_path)
        if state != "Done":
            incomplete = True
        label = None if labels is None else labels[index]
        source_name = "" if csv_source is None else csv_source.source
        rows.append([config, label, progress, current_load, state, csv_path, source_name])

    running_by_name = _running_processes_by_name() if incomplete and check_running else {}

    statuses = []
    for config, label, progress, current_load, state, csv_path, source_name in rows:
        name = config_name(config)
        process = running_by_name.get(name)
        running = process is not None and state != "Done"
        if not check_running and state != "Done":
            running = None
            if state == "Not running":
                state = "Unfinished"
        server = _short_source_name(getattr(process, "server", "")) if running else source_name
        if running:
            state = "Running"
            process_load = getattr(process, "current_load", None)
            if process_load is not None:
                current_load = float(process_load)
                progress = _progress_from_load(current_load, float(config.maxLoad))
        statuses.append(
            SimulationStatus(
                name=name,
                label=label,
                progress=progress,
                current_load=current_load,
                final_load=float(config.maxLoad),
                state=state,
                running=running,
                server=server,
                csv_path=csv_path,
            )
        )
    return statuses


def _status_path(status: SimulationStatus) -> Path | None:
    if not status.csv_path:
        return None
    path = Path(status.csv_path)
    return path.parent if path.name != "macroData.csv" else path.parent.parent


def _print_status_header(statuses: list[SimulationStatus]) -> None:
    paths = [path for path in (_status_path(status) for status in statuses) if path]
    if paths:
        try:
            common_path = os.path.commonpath([str(path) for path in paths])
        except ValueError:
            common_path = ""
        if common_path:
            print(f"Path: {common_path}")

    print(f"Configs ({len(statuses)}):")
    for status in statuses:
        print(f"  {status.name}.csv")


def print_status_table(
    configs,
    force_update: bool = False,
    check_running: bool = False,
    search_remote: bool | str = "conditional",
    labels=None,
) -> None:
    statuses = collect_status(
        configs,
        force_update=force_update,
        check_running=check_running,
        search_remote=search_remote,
        labels=labels,
    )
    if statuses:
        _print_status_header(statuses)

    table = []
    for status in statuses:
        progress = "N/A" if status.progress is None else f"{status.progress:.1f}%"
        load = (
            "N/A"
            if status.current_load is None or status.final_load is None
            else f"{status.current_load:g}/{status.final_load:g}"
        )
        table.append(
            [
                status.label if status.label else status.name,
                progress,
                load,
                "not checked" if status.running is None else ("yes" if status.running else "no"),
                status.state,
                status.server,
            ]
        )

    print(
        tabulate(
            table,
            headers=["Config", "Progress", "Load", "Running", "State", "Server"],
            tablefmt="grid",
        )
    )
