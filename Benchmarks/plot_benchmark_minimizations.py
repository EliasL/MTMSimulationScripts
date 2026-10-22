#!/usr/bin/env python3
"""Create comparative minimization plots as QA PNGs and vector PDFs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class Point:
    x: float
    mean: float
    stdev: float
    samples: int


@dataclass(frozen=True)
class PowerLawFit:
    prefactor: float
    exponent: float
    r_squared: float


@dataclass(frozen=True)
class Preparation:
    history: str
    size: int
    threads: int
    elapsed_seconds: float
    function_evaluations: float
    edge_flips: float


LABELS = {
    "early": "Early event",
    "late": "Late event",
    "first_none": "First minimization",
    "first_edge": "First minimization - edge flipping",
}

EVENT_MARKERS = {
    "early": "o",
    "late": "^",
    "first_none": "s",
    "first_edge": "D",
}

REPLAY_EVENTS = ("early", "late")
ERRORBAR_LOWER_FLOOR_FRACTION = 0.5


def configure_matplotlib() -> bool:
    """Use native Matplotlib styling with LaTeX/Computer Modern typography."""
    plt.rcdefaults()
    latex_available = all(shutil.which(program) for program in ("latex", "dvipng"))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "cmr10",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "text.usetex": latex_available,
        }
    )
    return latex_available


def read_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def event_key(row: dict) -> str | None:
    load = float(row["fixture_load"])
    history = str(row["history"])
    if math.isclose(load, 0.15) and history == "noReconnecting":
        return "early"
    if math.isclose(load, 0.7) and history == "noReconnecting":
        return "late"
    return None


def load_preparations(
    paths: Iterable[Path],
) -> dict[tuple[str, int, int], Preparation]:
    preparations: dict[tuple[str, int, int], Preparation] = {}
    for path in paths:
        for record in read_json(path):
            if record.get("status") not in {"generated", "existing", "measured"}:
                continue
            native = record.get("native") or {}
            elapsed = native.get("seconds_per_call", native.get("elapsed_seconds"))
            if elapsed is None:
                continue
            preparation = Preparation(
                history=str(record["history"]),
                size=int(record["rows"]),
                threads=int(record.get("preparation_threads", 1)),
                elapsed_seconds=float(elapsed),
                function_evaluations=float(native.get("function_evaluations", 0.0)),
                edge_flips=float(native.get("edge_flips", 0.0)),
            )
            preparations[
                (preparation.history, preparation.size, preparation.threads)
            ] = preparation
    return preparations


def errorbar(
    axis,
    points: list[Point],
    key: str,
    *,
    color: str | None = None,
    label: bool = True,
    label_text: str | None = None,
) -> str | None:
    if not points:
        return None
    lower_errors = [
        max(0.0, min(point.stdev, point.mean * (1.0 - ERRORBAR_LOWER_FLOOR_FRACTION)))
        for point in points
    ]
    upper_errors = [max(0.0, point.stdev) for point in points]
    container = axis.errorbar(
        [point.x for point in points],
        [point.mean for point in points],
        yerr=[lower_errors, upper_errors],
        fmt=f"{EVENT_MARKERS[key]}-",
        color=color,
        capsize=2.5,
        elinewidth=0.8,
        markerfacecolor="none",
        label=(label_text or LABELS[key]) if label else None,
    )
    return container.lines[0].get_color()


def hollow_plot(
    axis,
    x,
    y,
    key: str,
    *,
    color: str | None = None,
    label: bool = True,
    linestyle: str = "-",
    label_text: str | None = None,
) -> str:
    (line,) = axis.plot(
        x,
        y,
        label=(label_text or LABELS[key]) if label else None,
        marker=EVENT_MARKERS[key],
        markerfacecolor="none",
        color=color,
        linestyle=linestyle,
    )
    return line.get_color()


def fit_power_law(x: Iterable[float], y: Iterable[float]) -> PowerLawFit | None:
    """Fit y = C L^alpha by ordinary least squares in log space."""
    pairs = [
        (math.log(float(x_value)), math.log(float(y_value)))
        for x_value, y_value in zip(x, y)
        if float(x_value) > 0.0 and float(y_value) > 0.0
    ]
    if len(pairs) < 2:
        return None
    log_x = [pair[0] for pair in pairs]
    log_y = [pair[1] for pair in pairs]
    mean_x = statistics.fmean(log_x)
    mean_y = statistics.fmean(log_y)
    denominator = sum((value - mean_x) ** 2 for value in log_x)
    if denominator == 0.0:
        return None
    exponent = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in pairs
    ) / denominator
    intercept = mean_y - exponent * mean_x
    residual = sum(
        (y_value - (intercept + exponent * x_value)) ** 2
        for x_value, y_value in pairs
    )
    total = sum((value - mean_y) ** 2 for value in log_y)
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return PowerLawFit(math.exp(intercept), exponent, r_squared)


def power_law_label(key: str, fit: PowerLawFit | None) -> str:
    if fit is None:
        return LABELS[key]
    return rf"{LABELS[key]} ($\alpha={fit.exponent:.2f}$)"


def add_power_law_line(
    axis,
    x: Iterable[float],
    fit: PowerLawFit | None,
    color: str | None,
) -> None:
    values = [float(value) for value in x if float(value) > 0.0]
    if fit is None or color is None or len(values) < 2:
        return
    lower = min(values)
    upper = max(values)
    fit_x = [
        math.exp(
            math.log(lower)
            + index * (math.log(upper) - math.log(lower)) / 199.0
        )
        for index in range(200)
    ]
    axis.plot(
        fit_x,
        [fit.prefactor * value**fit.exponent for value in fit_x],
        color=color,
        linestyle="--",
        label="_nolegend_",
    )


def errorbar_with_power_law(axis, points: list[Point], key: str) -> PowerLawFit | None:
    fit = fit_power_law(
        [point.x for point in points],
        [point.mean for point in points],
    )
    color = errorbar(axis, points, key, label_text=power_law_label(key, fit))
    add_power_law_line(axis, [point.x for point in points], fit, color)
    return fit


def hollow_with_power_law(
    axis, x: list[float], y: list[float], key: str
) -> PowerLawFit | None:
    fit = fit_power_law(x, y)
    color = hollow_plot(axis, x, y, key, label_text=power_law_label(key, fit))
    add_power_law_line(axis, x, fit, color)
    return fit


def finish_axes(axis, threads: list[int], *, log_y: bool = True) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(threads)
    axis.set_xticklabels([str(thread) for thread in threads])
    if log_y:
        axis.set_yscale("log")
    axis.minorticks_off()


def legend_note(label: str) -> Line2D:
    return Line2D([], [], linestyle="none", marker="none", label=label)


def power_law_note(*, standard_deviation: bool = False) -> Line2D:
    label = r"Dashed: $C L^\alpha$ fit"
    if standard_deviation:
        label += r"; bars: $+1$ SD, lower clipped"
    return legend_note(label)


def add_axis_legend(
    axis,
    *,
    standard_deviation: bool = False,
    extra_handles: Iterable[Line2D] = (),
    **kwargs,
) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if standard_deviation:
        handles.append(legend_note(r"Error bars: $+1$ SD, lower clipped"))
        labels.append(r"Error bars: $+1$ SD, lower clipped")
    for handle in extra_handles:
        handles.append(handle)
        labels.append(handle.get_label())
    axis.legend(handles, labels, **kwargs)


def size_colors(sizes: list[int]) -> dict[int, str]:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {size: colors[index % len(colors)] for index, size in enumerate(sizes)}


def encoded_legend_handles(
    sizes: list[int],
    colors: dict[int, str],
    event_keys: Iterable[str],
    extra_handles: Iterable[Line2D] = (),
) -> list[Line2D]:
    size_handles = [
        Line2D([], [], color=colors[size], label=rf"$L={size}$")
        for size in sizes
    ]
    event_handles = [
        Line2D(
            [],
            [],
            color="black",
            marker=EVENT_MARKERS[key],
            markerfacecolor="none",
            linestyle="none",
            label=LABELS[key],
        )
        for key in event_keys
    ]
    return size_handles + event_handles + list(extra_handles)


def add_encoded_legend(
    figure,
    sizes: list[int],
    colors: dict[int, str],
    event_keys: Iterable[str],
    *,
    columns: int = 5,
    y: float = 0.94,
    extra_handles: Iterable[Line2D] = (),
) -> None:
    figure.legend(
        handles=encoded_legend_handles(
            sizes, colors, event_keys, extra_handles
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=columns,
    )


def save_figure(figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=220)
    figure.savefig(output_dir / f"{stem}.pdf")
    plt.close(figure)


def workload_points(rows: list[dict], size: int, key: str) -> list[Point]:
    return [
        Point(
            x=float(row["threads"]),
            mean=float(row["mean_seconds_per_call"]),
            stdev=float(row["stdev_seconds_per_call"]),
            samples=int(row["samples"]),
        )
        for row in sorted(rows, key=lambda item: item["threads"])
        if int(row["rows"]) == size and event_key(row) == key
    ]


def thread_scaling_plot(
    output_dir: Path,
    workload: list[dict],
    preparations: dict[tuple[str, int, int], Preparation],
    sizes: list[int],
    threads: list[int],
) -> None:
    figure, axis = plt.subplots(figsize=(7.6, 5.2))
    colors = size_colors(sizes)
    for size in sizes:
        for key in REPLAY_EVENTS:
            points = [
                point
                for point in workload_points(workload, size, key)
                if int(point.x) in threads
            ]
            errorbar(axis, points, key, color=colors[size], label=False)
        for history, key in (
            ("noReconnecting", "first_none"),
            ("edgeFlipping", "first_edge"),
        ):
            selected = sorted(
                (
                    preparation
                    for preparation in preparations.values()
                    if preparation.history == history
                    and preparation.size == size
                    and preparation.threads in threads
                ),
                key=lambda preparation: preparation.threads,
            )
            if selected:
                hollow_plot(
                    axis,
                    [preparation.threads for preparation in selected],
                    [preparation.elapsed_seconds for preparation in selected],
                    key,
                    color=colors[size],
                    label=False,
                )
    finish_axes(axis, threads)
    axis.set_xlabel("OpenMP threads")
    axis.set_ylabel("Seconds per minimization")
    axis.legend(
        handles=encoded_legend_handles(
            sizes,
            colors,
            (*REPLAY_EVENTS, "first_none", "first_edge"),
            (legend_note(r"Error bars: $+1$ SD, lower clipped"),),
        ),
        loc="upper right",
        ncol=2,
    )
    figure.tight_layout()
    save_figure(figure, output_dir, "minimization-thread-scaling")


def linear_size_scaling_plot(
    output_dir: Path,
    workload: list[dict],
    preparations: dict[tuple[str, int, int], Preparation],
    sizes: list[int],
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for key in REPLAY_EVENTS:
        points = [
            Point(
                x=float(row["rows"]),
                mean=float(row["mean_seconds_per_call"]),
                stdev=float(row["stdev_seconds_per_call"]),
                samples=int(row["samples"]),
            )
            for row in workload
            if int(row["threads"]) == 1 and event_key(row) == key
        ]
        errorbar_with_power_law(
            axis, sorted(points, key=lambda point: point.x), key
        )
    for history, key in (
        ("noReconnecting", "first_none"),
        ("edgeFlipping", "first_edge"),
    ):
        selected = sorted(
            (
                preparation
                for preparation in preparations.values()
                if preparation.history == history and preparation.threads == 1
            ),
            key=lambda item: item.size,
        )
        hollow_with_power_law(
            axis,
            [preparation.size for preparation in selected],
            [preparation.elapsed_seconds for preparation in selected],
            key,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([rf"${size}$" for size in sizes])
    axis.set_xlabel(r"Linear system size $L$")
    axis.set_ylabel("Seconds per minimization")
    axis.minorticks_off()
    add_axis_legend(
        axis,
        extra_handles=(power_law_note(standard_deviation=True),),
        ncol=2,
        loc="upper left",
    )
    figure.tight_layout()
    save_figure(figure, output_dir, "minimization-linear-size-scaling")


def speedup_plot(
    output_dir: Path, workload: list[dict], sizes: list[int], threads: list[int]
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    colors = size_colors(sizes)
    maximum = 1.0
    for size in sizes:
        for key in REPLAY_EVENTS:
            raw = [
                point
                for point in workload_points(workload, size, key)
                if int(point.x) in threads
            ]
            if not raw:
                continue
            base = next((point for point in raw if point.x == 1), None)
            if base is None:
                continue
            points: list[Point] = []
            for point in raw:
                speedup = base.mean / point.mean
                uncertainty = 0.0 if point.x == 1 else speedup * math.sqrt(
                    (base.stdev / base.mean) ** 2 + (point.stdev / point.mean) ** 2
                )
                maximum = max(maximum, speedup + uncertainty)
                points.append(Point(point.x, speedup, uncertainty, point.samples))
            errorbar(axis, points, key, color=colors[size], label=False)
    finish_axes(axis, threads, log_y=False)
    ideal_x = [
        2.0
        ** (
            math.log2(min(threads))
            + index
            * (math.log2(max(threads)) - math.log2(min(threads)))
            / 199.0
        )
        for index in range(200)
    ]
    axis.plot(ideal_x, ideal_x, color="black", linestyle="--")
    axis.set_ylim(0.8, max(3.6, maximum * 1.12))
    axis.set_xlabel("OpenMP threads")
    axis.set_ylabel("Speedup relative to one thread")
    add_encoded_legend(
        figure,
        sizes,
        colors,
        REPLAY_EVENTS,
        columns=4,
        y=0.92,
        extra_handles=(
            Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                label="Ideal linear speedup",
            ),
            legend_note(r"Error bars: propagated SD, lower clipped"),
        ),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.80))
    save_figure(figure, output_dir, "minimization-speedup")


def raw_group_points(
    samples: list[dict], size: int, threads: list[int], key: str, metric: str
) -> list[Point]:
    result: list[Point] = []
    for thread in threads:
        values = [
            float(row[metric])
            for row in samples
            if row.get("experiment") == "workload"
            and row.get("affinity_policy") == "close"
            and int(row["rows"]) == size
            and int(row["threads"]) == thread
            and event_key(row) == key
        ]
        if not values:
            continue
        result.append(
            Point(
                x=float(thread),
                mean=statistics.fmean(values),
                stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
                samples=len(values),
            )
        )
    return result


def metric_plot(
    output_dir: Path,
    samples: list[dict],
    sizes: list[int],
    threads: list[int],
    *,
    metric: str,
    ylabel: str,
    stem: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    colors = size_colors(sizes)
    for size in sizes:
        for key in REPLAY_EVENTS:
            points = raw_group_points(samples, size, threads, key, metric)
            errorbar(axis, points, key, color=colors[size], label=False)
    finish_axes(axis, threads)
    axis.set_xlabel("OpenMP threads")
    axis.set_ylabel(ylabel)
    add_encoded_legend(
        figure,
        sizes,
        colors,
        REPLAY_EVENTS,
        columns=4,
        y=0.92,
        extra_handles=(legend_note(r"Error bars: $+1$ SD, lower clipped"),),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.80))
    save_figure(figure, output_dir, stem)


def minimizer_function_evaluations_plot(
    output_dir: Path, samples: list[dict], sizes: list[int]
) -> None:
    """Plot minimizer path length against L, pooling over thread counts."""
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for key in REPLAY_EVENTS:
        points = raw_size_points(
            samples,
            sizes,
            key,
            "function_evaluations",
            thread=None,
        )
        errorbar_with_power_law(axis, points, key)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([rf"${size}$" for size in sizes])
    axis.set_xlabel(r"Linear system size $L$")
    axis.set_ylabel("Function evaluations per minimization")
    axis.minorticks_off()
    add_axis_legend(
        axis,
        extra_handles=(power_law_note(standard_deviation=True),),
    )
    figure.tight_layout()
    save_figure(figure, output_dir, "minimizer-function-evaluations")


def knee_thread_count_plot(
    output_dir: Path,
    workload: list[dict],
    preparations: dict[tuple[str, int, int], Preparation],
    sizes: list[int],
    threads: list[int],
) -> None:
    """Plot the measured speedup-efficiency knee for every workload."""
    figure, axis = plt.subplots(figsize=(7.6, 5.1))
    selected_threads: list[int] = []
    for key in (*REPLAY_EVENTS, "first_none", "first_edge"):
        knee_points: list[tuple[int, int]] = []
        for size in sizes:
            if key in REPLAY_EVENTS:
                timings = [
                    (int(point.x), point.mean)
                    for point in workload_points(workload, size, key)
                    if int(point.x) in threads
                ]
            else:
                history = (
                    "noReconnecting" if key == "first_none" else "edgeFlipping"
                )
                timings = [
                    (preparation.threads, preparation.elapsed_seconds)
                    for preparation in preparations.values()
                    if preparation.history == history
                    and preparation.size == size
                    and preparation.threads in threads
                ]
            timings = sorted(set(timings))
            base = next((elapsed for thread, elapsed in timings if thread == 1), None)
            if base is None or len(timings) < 2:
                continue
            # For one workload, E(p) / T(p) is proportional to S(p)^2 / p.
            knee_thread, _ = max(
                (
                    (thread, (base / elapsed) ** 2 / thread)
                    for thread, elapsed in timings
                ),
                key=lambda item: (item[1], -item[0]),
            )
            knee_points.append((size, knee_thread))
            selected_threads.append(knee_thread)
        if knee_points:
            hollow_plot(
                axis,
                [size for size, _ in knee_points],
                [thread for _, thread in knee_points],
                key,
            )
    if not selected_threads:
        plt.close(figure)
        return
    axis.set_xscale("log")
    axis.set_yscale("log", base=2)
    axis.set_xticks(sizes)
    axis.set_xticklabels([rf"${size}$" for size in sizes])
    displayed_threads = sorted(
        {
            int(thread)
            for line in axis.lines
            for thread in line.get_ydata()
        }
    )
    axis.set_yticks(displayed_threads)
    axis.set_yticklabels([str(thread) for thread in displayed_threads])
    axis.set_ylim(
        min(displayed_threads) / 1.1,
        max(displayed_threads) * 1.1,
    )
    axis.set_xlabel(r"Linear system size $L$")
    axis.set_ylabel("Recommended OpenMP threads")
    axis.minorticks_off()
    handles, labels = axis.get_legend_handles_labels()
    handles.append(legend_note(r"Knee: maximize $E(p)/T(p)$"))
    labels.append(r"Knee: maximize $E(p)/T(p)$")
    axis.legend(handles, labels, ncol=2)
    figure.tight_layout()
    save_figure(figure, output_dir, "recommended-thread-count-vs-L")


def first_minimization_plot(
    output_dir: Path,
    preparations: dict[tuple[str, int, int], Preparation],
    sizes: list[int],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    histories = (
        ("noReconnecting", "first_none"),
        ("edgeFlipping", "first_edge"),
    )
    metrics = (
        ("elapsed_seconds", "Seconds"),
        ("function_evaluations", "Function evaluations"),
    )
    for axis, (metric, ylabel) in zip(
        axes,
        metrics,
    ):
        for history, key in histories:
            selected = sorted(
                (
                    preparation
                    for preparation in preparations.values()
                    if preparation.history == history
                    and preparation.threads == 1
                ),
                key=lambda preparation: preparation.size,
            )
            x = [float(preparation.size) for preparation in selected]
            y = [float(getattr(preparation, metric)) for preparation in selected]
            hollow_with_power_law(axis, x, y, key)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(sizes)
        axis.set_xticklabels([rf"${size}$" for size in sizes])
        axis.set_xlabel(r"Linear system size $L$")
        axis.set_ylabel(ylabel)
        axis.minorticks_off()
        add_axis_legend(
            axis,
            extra_handles=(power_law_note(),),
            loc="best",
        )
    figure.tight_layout()
    save_figure(figure, output_dir, "first-minimization-cost-vs-L")


def force_linear_size_plot(output_dir: Path, summaries: list[dict]) -> None:
    rows = [
        row
        for row in summaries
        if row["experiment"] == "system"
        and int(row["threads"]) == 1
        and row["affinity_policy"] == "close"
    ]
    if not rows:
        return
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    labels = {
        "force-kernel": "Force kernel",
        "force-evaluation": "Full force evaluation",
    }
    markers = {"force-kernel": "o", "force-evaluation": "s"}
    for mode in ("force-kernel", "force-evaluation"):
        selected = sorted(
            (row for row in rows if row["mode"] == mode),
            key=lambda row: int(row["rows"]),
        )
        x = [float(row["rows"]) for row in selected]
        y = [float(row["mean_seconds_per_call"]) for row in selected]
        fit = fit_power_law(x, y)
        label = labels[mode]
        if fit is not None:
            label += rf" ($\alpha={fit.exponent:.2f}$)"
        container = axis.errorbar(
            x,
            y,
            yerr=[float(row["stdev_seconds_per_call"]) for row in selected],
            fmt=f"{markers[mode]}-",
            markerfacecolor="none",
            label=label,
        )
        add_power_law_line(axis, x, fit, container.lines[0].get_color())
    sizes = sorted({int(row["rows"]) for row in rows})
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([rf"${size}$" for size in sizes])
    axis.set_xlabel(r"Linear system size $L$")
    axis.set_ylabel("Seconds per force call")
    axis.minorticks_off()
    add_axis_legend(
        axis,
        extra_handles=(power_law_note(standard_deviation=True),),
    )
    figure.tight_layout()
    save_figure(figure, output_dir, "force-size-scaling-vs-L")


def raw_size_points(
    samples: list[dict],
    sizes: list[int],
    key: str,
    metric: str,
    *,
    thread: int | None = 1,
) -> list[Point]:
    result: list[Point] = []
    for size in sizes:
        values = [
            float(row[metric])
            for row in samples
            if row.get("experiment") == "workload"
            and row.get("affinity_policy") == "close"
            and int(row["rows"]) == size
            and int(row["threads"]) != 60
            and (thread is None or int(row["threads"]) == thread)
            and event_key(row) == key
        ]
        if not values:
            continue
        result.append(
            Point(
                x=float(size),
                mean=statistics.fmean(values),
                stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
                samples=len(values),
            )
        )
    return result


def minimizer_work_vs_linear_size_plot(
    output_dir: Path, samples: list[dict], sizes: list[int]
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    metrics = (
        ("function_evaluations", "Function evaluations per minimization"),
        ("seconds_per_function_evaluation", "Seconds per function evaluation"),
    )
    for axis, (metric, ylabel) in zip(axes, metrics):
        for key in REPLAY_EVENTS:
            points = raw_size_points(
                samples,
                sizes,
                key,
                metric,
                thread=None if metric == "function_evaluations" else 1,
            )
            errorbar_with_power_law(axis, points, key)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(sizes)
        axis.set_xticklabels([rf"${size}$" for size in sizes])
        axis.set_xlabel(r"Linear system size $L$")
        axis.set_ylabel(ylabel)
        axis.minorticks_off()
        add_axis_legend(
            axis,
            extra_handles=(power_law_note(standard_deviation=True),),
            loc="best",
        )
    figure.tight_layout()
    save_figure(figure, output_dir, "minimizer-work-vs-L")


def write_plot_data(
    path: Path,
    summaries: list[dict],
    preparations: dict[tuple[str, int, int], Preparation],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(
            [
                "event",
                "load",
                "size",
                "threads",
                "mean_seconds",
                "stdev_seconds",
                "samples",
                "mean_function_evaluations",
                "stdev_function_evaluations",
            ]
        )
        for row in summaries:
            key = event_key(row)
            if (
                row["experiment"] != "workload"
                or key is None
                or int(row["threads"]) == 60
                or row["affinity_policy"] != "close"
            ):
                continue
            writer.writerow(
                [
                    LABELS[key],
                    row["fixture_load"],
                    f"{row['rows']}x{row['cols']}",
                    row["threads"],
                    row["mean_seconds_per_call"],
                    row["stdev_seconds_per_call"],
                    row["samples"],
                    row["mean_function_evaluations"],
                    row["stdev_function_evaluations"],
                ]
            )
        for preparation in sorted(preparations.values(), key=lambda item: (item.history, item.size)):
            key = "first_edge" if preparation.history == "edgeFlipping" else "first_none"
            writer.writerow(
                [
                    LABELS[key],
                    0.15,
                    f"{preparation.size}x{preparation.size}",
                    preparation.threads,
                    preparation.elapsed_seconds,
                    "",
                    1,
                    preparation.function_evaluations,
                    "",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--first-minimization",
        type=Path,
        action="append",
        default=[],
        help=(
            "initial_condition_preparation.json or first_minimization_benchmarks.json; "
            "may be passed more than once"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    latex_enabled = configure_matplotlib()
    result_dir = args.result_dir.resolve()
    output_dir = (args.output_dir or result_dir / "analysis_plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = read_json(result_dir / "summary.json")
    samples = read_jsonl(result_dir / "samples.jsonl")
    workload = [
        row
        for row in summaries
        if row["experiment"] == "workload"
        and row["affinity_policy"] == "close"
        and event_key(row) is not None
    ]
    preparation_paths = [path.resolve() for path in args.first_minimization]
    for filename in (
        "initial_condition_preparation.json",
        "first_minimization_benchmarks.json",
    ):
        candidate = result_dir / filename
        if candidate.is_file() and candidate not in preparation_paths:
            preparation_paths.append(candidate)
    preparations = load_preparations(preparation_paths)
    sizes = sorted({int(row["rows"]) for row in workload})
    threads = sorted(
        {int(row["threads"]) for row in workload if int(row["threads"]) != 60}
    )
    if not sizes or not threads:
        raise SystemExit("No minimization workload summaries were found.")

    thread_scaling_plot(output_dir, workload, preparations, sizes, threads)
    linear_size_scaling_plot(output_dir, workload, preparations, sizes)
    speedup_plot(output_dir, workload, sizes, threads)
    minimizer_function_evaluations_plot(output_dir, samples, sizes)
    metric_plot(
        output_dir,
        samples,
        sizes,
        threads,
        metric="seconds_per_function_evaluation",
        ylabel="Seconds per function evaluation",
        stem="time-per-function-evaluation",
    )
    first_minimization_plot(output_dir, preparations, sizes)
    force_linear_size_plot(output_dir, summaries)
    minimizer_work_vs_linear_size_plot(output_dir, samples, sizes)
    knee_thread_count_plot(output_dir, workload, preparations, sizes, threads)
    write_plot_data(output_dir / "plot-data.csv", summaries, preparations)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "latex_enabled": latex_enabled,
                "png_plots": sorted(path.name for path in output_dir.glob("*.png")),
                "pdf_plots": sorted(path.name for path in output_dir.glob("*.pdf")),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
