#!/usr/bin/env python3
"""Aggregate matched-event minimization trajectories from a seed or archive set."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from plot_common_event_minimization import COLORS


ALGORITHMS = ("FIRE", "CG", "LBFGS")
PLOT_ORDER = ("CG", "FIRE", "LBFGS")
END_MARKERS = {"LBFGS": "o", "FIRE": "^", "CG": "x"}
LINE_STYLES = {"CG": ":", "FIRE": "--", "LBFGS": "-"}
PLOT_ZORDERS = {"CG": 1.0, "FIRE": 2.0, "LBFGS": 3.0}
TRACE_ALPHA = 0.2016
END_MARKER_ALPHA = 0.165
TRACE_ALPHAS = {"CG": TRACE_ALPHA, "FIRE": 0.3024, "LBFGS": TRACE_ALPHA}
MARKER_ALPHAS = {"CG": END_MARKER_ALPHA, "FIRE": 0.2475, "LBFGS": END_MARKER_ALPHA}
UNCERTAINTY_FILL_ALPHAS = {"CG": 0.10, "FIRE": 0.15, "LBFGS": 0.15}
UNCERTAINTY_OUTLINE_ALPHA = 0.45
UNCERTAINTY_OUTLINE_WIDTH = 0.35


def read_trajectory_csv(stream, label: str) -> tuple[np.ndarray, np.ndarray]:
    reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8", newline=""))
    if reader.fieldnames is None:
        raise ValueError(f"Missing CSV header in {label}")
    required = ("nr_func_evals", "total_energy")
    missing = set(required) - set(reader.fieldnames)
    if missing:
        raise KeyError(f"{label} is missing columns: {sorted(missing)}")

    calls = []
    energy = []
    for row_number, row in enumerate(reader, start=2):
        try:
            calls.append(float(row["nr_func_evals"]))
            energy.append(float(row["total_energy"]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid trajectory value in {label}, row {row_number}") from error
    return np.asarray(calls), np.asarray(energy)


def prepare_trajectory(
    calls: np.ndarray,
    energy: np.ndarray,
    label: str,
    use_running_minimum: bool,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(calls) & np.isfinite(energy)
    if not np.all(finite):
        raise ValueError(f"Non-finite trajectory values in {label}")
    if len(calls) < 2 or np.any(calls < 0):
        raise ValueError(f"Invalid minimization trajectory in {label}")

    # LBFGS can restart its local function-evaluation counter after a failed trial.
    cumulative_calls = np.empty_like(calls)
    offset = 0.0
    cumulative_calls[0] = calls[0]
    for index in range(1, len(calls)):
        if calls[index] < calls[index - 1]:
            offset += calls[index - 1]
        cumulative_calls[index] = calls[index] + offset
    if np.any(np.diff(cumulative_calls) < 0):
        raise ValueError(f"Could not make function calls monotone in {label}")
    cumulative_calls -= cumulative_calls[0]

    values = np.minimum.accumulate(energy) if use_running_minimum else energy
    order = np.argsort(cumulative_calls, kind="stable")
    cumulative_calls, values = cumulative_calls[order], values[order]
    unique_calls, first = np.unique(cumulative_calls, return_index=True)
    values = values[first]
    if unique_calls[-1] <= 0.0:
        raise ValueError(f"No positive function-call range in {label}")
    return unique_calls, values


def load_trajectory_from_path(
    result: dict, use_running_minimum: bool
) -> tuple[np.ndarray, np.ndarray]:
    algorithm = result["algorithm"]
    directories = result["minimization_directories"]
    if len(directories) != 1:
        raise ValueError(f"Expected one minimization directory for {algorithm}: {directories}")

    csv_path = Path(directories[0]) / "macroData.csv"
    with csv_path.open("rb") as stream:
        calls, energy = read_trajectory_csv(stream, str(csv_path))
    return prepare_trajectory(calls, energy, str(csv_path), use_running_minimum)


def archive_member_for_directory(directory: str, seed_name: str) -> str:
    normalized = directory.replace("\\", "/")
    marker = f"{seed_name}/"
    marker_start = normalized.find(marker)
    if marker_start < 0:
        raise ValueError(f"Could not map manifest path into {seed_name}: {directory}")
    return normalized[marker_start:] + "/macroData.csv"


def load_trajectory_from_archive(
    result: dict,
    archive: zipfile.ZipFile,
    seed_name: str,
    use_running_minimum: bool,
) -> tuple[np.ndarray, np.ndarray]:
    algorithm = result["algorithm"]
    directories = result["minimization_directories"]
    if len(directories) != 1:
        raise ValueError(f"Expected one minimization directory for {algorithm}: {directories}")
    member = archive_member_for_directory(directories[0], seed_name)
    label = f"{archive.filename}:{member}"
    try:
        with archive.open(member) as stream:
            calls, energy = read_trajectory_csv(stream, label)
    except KeyError as error:
        raise FileNotFoundError(f"Missing minimization trajectory {label}") from error
    return prepare_trajectory(calls, energy, label, use_running_minimum)


def load_trajectory_from_archive_or_path(
    result: dict,
    archive: zipfile.ZipFile,
    seed_name: str,
    use_running_minimum: bool,
) -> tuple[np.ndarray, np.ndarray]:
    csv_path = Path(result["minimization_directories"][0]) / "macroData.csv"
    if csv_path.is_file():
        return load_trajectory_from_path(result, use_running_minimum)
    return load_trajectory_from_archive(result, archive, seed_name, use_running_minimum)


def normalized_event(
    results: dict,
    trajectory_loader,
    label: str,
    use_running_minimum: bool,
) -> dict:
    trajectories = {
        algorithm: trajectory_loader(results[algorithm], use_running_minimum)
        for algorithm in ALGORITHMS
    }
    start_energies = np.asarray(
        [trajectories[algorithm][1][0] for algorithm in ALGORITHMS]
    )
    completed_drops = np.asarray(
        [
            start_energies[index] - np.min(trajectories[algorithm][1])
            for index, algorithm in enumerate(ALGORITHMS)
        ]
    )
    event_drop = float(np.max(completed_drops))
    if not np.isfinite(event_drop) or event_drop <= 0.0:
        raise ValueError(f"Invalid common energy drop in {label}: {event_drop}")

    lbfgs_calls = trajectories["LBFGS"][0][-1]
    normalized = {}
    for index, algorithm in enumerate(ALGORITHMS):
        calls, energy = trajectories[algorithm]
        # Align each trajectory at its own recorded start, then normalize all
        # three algorithms by the largest completed drop for this event.
        residual = 1.0 - (start_energies[index] - energy) / event_drop
        residual[0] = 1.0
        normalized[algorithm] = (
            calls / lbfgs_calls,
            np.maximum(residual, 1e-12),
        )
    return normalized


def completed_results(payload: dict, label: str) -> dict | None:
    if not all(item.get("status") == "completed-first-drop" for item in payload["results"]):
        return None
    results = {item["algorithm"]: item for item in payload["results"]}
    missing = set(ALGORITHMS) - set(results)
    if missing:
        raise ValueError(f"{label} is missing algorithms: {sorted(missing)}")
    return results


def has_same_first_drop_load(results: dict) -> bool:
    loads = []
    for algorithm in ALGORITHMS:
        try:
            loads.append(str(results[algorithm]["first_drop"]["load"]))
        except KeyError as error:
            raise KeyError(f"Missing first-drop load for {algorithm}") from error
    return len(set(loads)) == 1


def load_fire_reruns(rerun_root: Path | None) -> dict[str, dict]:
    if rerun_root is None:
        return {}
    if not rerun_root.is_dir():
        raise FileNotFoundError(rerun_root)
    reruns = {}
    for manifest_path in rerun_root.rglob("rerun_manifest.json"):
        event_id = str(manifest_path.parent.relative_to(rerun_root))
        payload = json.loads(manifest_path.read_text())
        result = payload.get("rerun_fire_result")
        if not isinstance(result, dict) or result.get("algorithm") != "FIRE":
            raise ValueError(f"Invalid FIRE rerun manifest: {manifest_path}")
        if event_id in reruns:
            raise ValueError(f"Duplicate FIRE rerun for {event_id}")
        reruns[event_id] = result
    if not reruns:
        raise ValueError(f"No FIRE rerun manifests found in {rerun_root}")
    return reruns


def replace_fire_result(results: dict, event_id: str, reruns: dict[str, dict]) -> dict:
    rerun = reruns.get(event_id)
    if rerun is None:
        return results
    original = results["FIRE"]
    original_step = int(original["first_drop"]["load_step"])
    rerun_step = int(rerun["first_drop"]["load_step"])
    original_load = float(original["first_drop"]["load"])
    rerun_load = float(rerun["first_drop"]["load"])
    if original_step != rerun_step or original_load != rerun_load:
        raise ValueError(
            f"FIRE rerun changes the detected drop for {event_id}: "
            f"step/load {original_step}/{original_load} -> {rerun_step}/{rerun_load}"
        )
    return {**results, "FIRE": rerun}


def events_from_directory(
    seed_root: Path,
    use_running_minimum: bool,
    same_first_drop_load: bool,
    excluded_events: set[str],
    fire_reruns: dict[str, dict],
) -> list[dict]:
    events = []
    for manifest_path in sorted(seed_root.glob("event_*/event_manifest.json")):
        event_id = f"{seed_root.name}/{manifest_path.parent.name}"
        if event_id in excluded_events:
            continue
        payload = json.loads(manifest_path.read_text())
        results = completed_results(payload, str(manifest_path))
        if results is None:
            continue
        results = replace_fire_result(results, event_id, fire_reruns)
        if same_first_drop_load and not has_same_first_drop_load(results):
            continue
        events.append(
            normalized_event(results, load_trajectory_from_path, str(manifest_path), use_running_minimum)
        )
    return events


def events_from_archive(
    archive_path: Path,
    use_running_minimum: bool,
    same_first_drop_load: bool,
    excluded_events: set[str],
    fire_reruns: dict[str, dict],
) -> list[dict]:
    seed_name = archive_path.stem
    events = []
    with zipfile.ZipFile(archive_path) as archive:
        manifest_members = sorted(
            name
            for name in archive.namelist()
            if name.startswith(f"{seed_name}/event_")
            and name.endswith("/event_manifest.json")
        )
        for member in manifest_members:
            event_id = f"{seed_name}/{Path(member).parent.name}"
            if event_id in excluded_events:
                continue
            payload = json.loads(archive.read(member))
            results = completed_results(payload, f"{archive_path}:{member}")
            if results is None:
                continue
            results = replace_fire_result(results, event_id, fire_reruns)
            if same_first_drop_load and not has_same_first_drop_load(results):
                continue
            loader = lambda result, running: load_trajectory_from_archive_or_path(
                result, archive, seed_name, running
            )
            events.append(
                normalized_event(results, loader, f"{archive_path}:{member}", use_running_minimum)
            )
    return events


def collect_events(
    data_root: Path,
    use_running_minimum: bool,
    same_first_drop_load: bool = False,
    excluded_events: set[str] | None = None,
    fire_rerun_root: Path | None = None,
) -> list[dict]:
    return collect_events_with_cache(
        data_root,
        use_running_minimum,
        None,
        same_first_drop_load,
        excluded_events,
        fire_rerun_root,
    )


def cache_source_signature(
    data_root: Path,
    use_running_minimum: bool,
    same_first_drop_load: bool,
    excluded_events: set[str],
    fire_rerun_root: Path | None,
) -> str:
    data_root = data_root.resolve()
    if data_root.name.startswith("seed_") and data_root.is_dir():
        source_paths = [path for path in data_root.rglob("*") if path.is_file()]
    elif (data_root / "seed_0").is_dir():
        source_paths = [path for path in (data_root / "seed_0").rglob("*") if path.is_file()]
        source_paths.extend(data_root.glob("seed_*.zip"))
    else:
        raise ValueError(f"Expected a seed directory or comparison-data root: {data_root}")
    if fire_rerun_root is not None:
        if not fire_rerun_root.is_dir():
            raise FileNotFoundError(fire_rerun_root)
        source_paths.extend(path for path in fire_rerun_root.rglob("*") if path.is_file())

    files = []
    for path in sorted(source_paths, key=str):
        stat = path.stat()
        files.append((str(path), stat.st_size, stat.st_mtime_ns))
    return json.dumps(
        {
            "data_root": str(data_root),
            "files": files,
            "use_running_minimum": use_running_minimum,
            "same_first_drop_load": same_first_drop_load,
            "excluded_events": sorted(excluded_events),
            "fire_rerun_root": None if fire_rerun_root is None else str(fire_rerun_root.resolve()),
            "version": 4,
        },
        sort_keys=True,
    )


def load_event_cache(cache_path: Path, source_signature: str) -> list[dict] | None:
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as cached:
        if int(cached["cache_version"].item()) != 2:
            return None
        if str(cached["source_signature"].item()) != source_signature:
            return None
        event_count = int(cached["event_count"].item())
        events = [{} for _ in range(event_count)]
        for algorithm in ALGORITHMS:
            calls = cached[f"{algorithm}_calls"]
            residual = cached[f"{algorithm}_residual"]
            offsets = cached[f"{algorithm}_offsets"]
            if offsets.shape != (event_count + 1,):
                raise ValueError(f"Invalid {algorithm} offsets in cache {cache_path}")
            if offsets[0] != 0 or offsets[-1] != len(calls) or len(calls) != len(residual):
                raise ValueError(f"Invalid {algorithm} arrays in cache {cache_path}")
            for index in range(event_count):
                start, end = offsets[index], offsets[index + 1]
                events[index][algorithm] = (calls[start:end], residual[start:end])
        return events


def save_event_cache(
    events: list[dict], cache_path: Path, source_signature: str
) -> None:
    arrays = {
        "cache_version": np.asarray(2),
        "source_signature": np.asarray(source_signature),
        "event_count": np.asarray(len(events)),
    }
    for algorithm in ALGORITHMS:
        calls = [event[algorithm][0] for event in events]
        residual = [event[algorithm][1] for event in events]
        offsets = np.concatenate(([0], np.cumsum([len(values) for values in calls])))
        arrays[f"{algorithm}_calls"] = np.concatenate(calls)
        arrays[f"{algorithm}_residual"] = np.concatenate(residual)
        arrays[f"{algorithm}_offsets"] = offsets

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f".{cache_path.name}.tmp")
    with temporary_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary_path.replace(cache_path)


def collect_events_with_cache(
    data_root: Path,
    use_running_minimum: bool,
    cache_path: Path | None,
    same_first_drop_load: bool = False,
    excluded_events: set[str] | None = None,
    fire_rerun_root: Path | None = None,
) -> list[dict]:
    excluded_events = set() if excluded_events is None else set(excluded_events)
    fire_reruns = load_fire_reruns(fire_rerun_root)
    source_signature = (
        cache_source_signature(
            data_root,
            use_running_minimum,
            same_first_drop_load,
            excluded_events,
            fire_rerun_root,
        )
        if cache_path is not None
        else None
    )
    if cache_path is not None:
        cached_events = load_event_cache(cache_path, source_signature)
        if cached_events is not None:
            print(f"Loaded {len(cached_events)} events from cache {cache_path}")
            return cached_events

    if data_root.name.startswith("seed_") and data_root.is_dir():
        events = events_from_directory(
            data_root,
            use_running_minimum,
            same_first_drop_load,
            excluded_events,
            fire_reruns,
        )
    elif (data_root / "seed_0").is_dir():
        events = events_from_directory(
            data_root / "seed_0",
            use_running_minimum,
            same_first_drop_load,
            excluded_events,
            fire_reruns,
        )
        for archive_path in sorted(data_root.glob("seed_*.zip")):
            events.extend(
                events_from_archive(
                    archive_path,
                    use_running_minimum,
                    same_first_drop_load,
                    excluded_events,
                    fire_reruns,
                )
            )
    else:
        raise ValueError(f"Expected a seed directory or comparison-data root: {data_root}")
    if not events:
        raise ValueError(f"No completed matched events found in {data_root}")
    if cache_path is not None:
        save_event_cache(events, cache_path, source_signature)
        print(f"Saved event cache {cache_path}")
    return events


def plot_events(events: list[dict], output: Path) -> None:
    progress_grid = np.linspace(0.0, 1.0, 500)
    fig, ax = plt.subplots(figsize=(4.92, 3.48))

    for algorithm in PLOT_ORDER:
        interpolated_x = []
        interpolated_log_residuals = []
        for event in events:
            normalized_calls, residual = event[algorithm]
            progress = normalized_calls / normalized_calls[-1]
            ax.plot(
                normalized_calls,
                residual,
                color=COLORS[algorithm],
                alpha=TRACE_ALPHAS[algorithm],
                linewidth=0.65,
                linestyle=LINE_STYLES[algorithm],
                zorder=PLOT_ZORDERS[algorithm],
                rasterized=True,
            )
            ax.plot(
                normalized_calls[-1],
                residual[-1],
                linestyle="None",
                marker=END_MARKERS[algorithm],
                markersize=3.0,
                markerfacecolor="none",
                markeredgecolor=COLORS[algorithm],
                markeredgewidth=0.7,
                alpha=MARKER_ALPHAS[algorithm],
                zorder=PLOT_ZORDERS[algorithm] + 0.1,
                rasterized=True,
            )
            interpolated_x.append(np.interp(progress_grid, progress, normalized_calls))
            interpolated_log_residuals.append(
                np.interp(progress_grid, progress, np.log10(residual))
            )

        interpolated_x = np.asarray(interpolated_x)
        if np.any(interpolated_x[:, 1:] <= 0.0):
            raise ValueError(f"{algorithm} has non-positive function calls after the common start")
        log_x = np.log(interpolated_x[:, 1:])
        mean_log_x = np.mean(log_x, axis=0)
        std_log_x = np.std(log_x, axis=0, ddof=1)
        geometric_mean_x = np.empty(progress_grid.shape)
        geometric_mean_x[0] = 0.0
        geometric_mean_x[1:] = np.exp(mean_log_x)
        lower_x = np.zeros_like(geometric_mean_x)
        upper_x = np.zeros_like(geometric_mean_x)
        lower_x[1:] = np.exp(mean_log_x - std_log_x)
        upper_x[1:] = np.exp(mean_log_x + std_log_x)
        log_residuals = np.asarray(interpolated_log_residuals)
        mean_log_residual = np.mean(log_residuals, axis=0)
        geometric_mean_residual = 10.0**mean_log_residual
        ax.fill_betweenx(
            geometric_mean_residual,
            lower_x,
            upper_x,
            color=COLORS[algorithm],
            alpha=UNCERTAINTY_FILL_ALPHAS[algorithm],
            linewidth=0.0,
            zorder=PLOT_ZORDERS[algorithm] + 0.2,
        )
        for boundary in (lower_x, upper_x):
            ax.plot(
                boundary,
                geometric_mean_residual,
                color="black",
                alpha=UNCERTAINTY_OUTLINE_ALPHA,
                linewidth=UNCERTAINTY_OUTLINE_WIDTH,
                linestyle="-",
                label="_nolegend_",
                zorder=(
                    10.0
                    if algorithm == "LBFGS"
                    else PLOT_ZORDERS[algorithm] + 0.25
                ),
            )
        ax.plot(
            geometric_mean_x,
            geometric_mean_residual,
            color=COLORS[algorithm],
            linewidth=2.4,
            linestyle=LINE_STYLES[algorithm],
            label=rf"{algorithm} ({geometric_mean_x[-1]:.2f}$\times$)",
            zorder=PLOT_ZORDERS[algorithm] + 0.3,
        )

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(1e-12, 1.4)
    ax.set_xlabel(
        "Function calls / LBFGS function calls for the same event"
    )
    ax.set_ylabel(
        r"Remaining drop, $1-\Delta E/\Delta E_{\max}$"
    )
    ax.legend(loc="upper right", ncol=1)
    ax.grid(True, which="both", alpha=0.18)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Events: {len(events)}")
    for algorithm in ALGORITHMS:
        ratios = [event[algorithm][0][-1] for event in events]
        geometric_mean_ratio = float(np.exp(np.mean(np.log(ratios))))
        print(
            f"{algorithm}: mean_call_ratio={np.mean(ratios):.6f}, "
            f"geometric_mean_call_ratio={geometric_mean_ratio:.6f}, "
            f"median={np.median(ratios):.6f}, min={np.min(ratios):.6f}, "
            f"max={np.max(ratios):.6f}"
        )
    print(f"Saved {output}")


def plot_seed(
    data_root: Path,
    output: Path,
    use_running_minimum: bool = True,
    cache_path: Path | None = None,
    same_first_drop_load: bool = False,
    excluded_events: set[str] | None = None,
    fire_rerun_root: Path | None = None,
) -> None:
    plot_events(
        collect_events_with_cache(
            data_root,
            use_running_minimum,
            cache_path,
            same_first_drop_load,
            excluded_events,
            fire_rerun_root,
        ),
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--raw-energy",
        action="store_true",
        help="Plot raw energy rather than the monotone best-so-far envelope.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Path for the extracted trajectory cache (default: next to the output).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write an extracted trajectory cache.",
    )
    parser.add_argument(
        "--same-first-drop-load",
        action="store_true",
        help="Keep only events where all three algorithms first drop at the same recorded load.",
    )
    parser.add_argument(
        "--exclude-event",
        action="append",
        default=[],
        metavar="SEED/EVENT",
        help="Exclude a specific event identifier; may be repeated.",
    )
    parser.add_argument(
        "--fire-rerun-root",
        type=Path,
        help="Replace capped FIRE trajectories with reruns from this directory.",
    )
    args = parser.parse_args()
    if args.no_cache and args.cache is not None:
        parser.error("--cache and --no-cache cannot be used together")
    cache_path = None if args.no_cache else args.cache
    if cache_path is None and not args.no_cache:
        cache_path = args.output.with_suffix(".cache.npz")
    plot_seed(
        args.seed_root,
        args.output,
        use_running_minimum=not args.raw_energy,
        cache_path=cache_path,
        same_first_drop_load=args.same_first_drop_load,
        excluded_events=set(args.exclude_event),
        fire_rerun_root=args.fire_rerun_root,
    )


if __name__ == "__main__":
    main()
