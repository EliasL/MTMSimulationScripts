from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from MTMath.energyFunction import ContiEnergy
from Plotting.dataFunctions import VTUData, infer_strain_from_vtu, resolve_vtu_files


def load_window_from_name(name: str) -> tuple[float, float] | None:
    match = re.search(r"l([-+0-9.eE]+),[-+0-9.eE]+,([-+0-9.eE]+)", name)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def reference_shear_from_text(text: str) -> float | None:
    match = re.search(r"(?:distorted reference|GP1)\s*([-+0-9.eE]+)", text)
    if match is None:
        return None
    return float(match.group(1))


def current_shear_from_text(text: str) -> float | None:
    match = re.search(r"current shear\s*([-+0-9.eE]+)\s+to\s+[-+0-9.eE]+", text)
    if match is None:
        return None
    return float(match.group(1))


def parse_windows(text: str) -> list[tuple[float, float]]:
    windows = []
    for part in text.split(","):
        bounds = part.split(":")
        if len(bounds) != 2:
            raise ValueError(f"Window must be written as start:stop, got {part!r}")
        start, stop = (float(value) for value in bounds)
        if stop <= start:
            raise ValueError(f"Window stop must be larger than start, got {part!r}")
        windows.append((start, stop))
    if not windows:
        raise ValueError("At least one window must be provided.")
    return windows


def variant_label(sim_dir: Path) -> str:
    name = sim_dir.name
    if name.startswith("simpleShearReferenceTest"):
        shear_value = reference_shear_from_text(name)
        shear = f" {shear_value:g}" if shear_value is not None else ""
        reconnection = "edge flip" if "edgeFlip" in name else "no reconnection"
        return f"distorted reference{shear}, {reconnection}"
    window = load_window_from_name(name)
    if window is not None:
        start, stop = window
        if start != 0.0:
            reconnection = "edge flip" if "edgeFlip" in name else "no reconnection"
            return f"current shear {start:g} to {stop:g}, {reconnection}"
    if "edgeFlip" in name:
        return "simple shear, edge flip"
    return "simple shear, no reconnection"


def variant_style(label: str) -> dict:
    linestyle = "-" if "edge flip" in label else "--"
    if "current shear" in label:
        shear = current_shear_from_text(label)
        current_colors = {2.0: "C2", 5.0: "C3", 10.0: "C4"}
        color = current_colors.get(shear, "C2")
        return {"color": color, "linestyle": linestyle, "zorder": 3}
    if "distorted reference" in label:
        shear = reference_shear_from_text(label)
        reference_colors = {2.0: "C2", 5.0: "C3", 10.0: "C4"}
        return {
            "color": reference_colors.get(shear, "C2"),
            "linestyle": linestyle,
            "zorder": 3,
        }
    if "edge flip" in label:
        return {"color": "C0", "linestyle": linestyle, "zorder": 4}
    return {"color": "C1", "linestyle": linestyle, "zorder": 2}


def load_from_vtu(vtu_file: str) -> float:
    load = infer_strain_from_vtu(vtu_file)
    if load is None or not np.isfinite(load):
        load = getattr(VTUData(vtu_file), "load", np.nan)
    if not np.isfinite(load):
        raise ValueError(f"Could not infer load from {vtu_file}")
    return float(load)


def stiffness_1212(vtu_file: str, *, eulerian: bool, loops: int) -> np.ndarray:
    F = VTUData(vtu_file).get_F()
    hessian = ContiEnergy.elasticity_tensor(F, eulerian=eulerian, loops=loops)
    return hessian[..., 0, 1, 0, 1]


def collect_series(
    sim_dir: Path,
    *,
    eulerian: bool,
    loops: int,
    local_load: bool,
    load_window: tuple[float, float] | None = None,
):
    load_offset = 0.0
    if local_load:
        window = load_window if load_window is not None else load_window_from_name(sim_dir.name)
        if window is None:
            raise ValueError(f"Could not infer load window from {sim_dir.name}")
        load_offset = window[0]

    rows = []
    for vtu_file in resolve_vtu_files(sim_dir):
        absolute_load = load_from_vtu(vtu_file)
        if load_window is not None:
            start, stop = load_window
            if absolute_load < start - 1e-12 or absolute_load > stop + 1e-12:
                continue

        values = stiffness_1212(vtu_file, eulerian=eulerian, loops=loops)
        rows.append(
            (
                absolute_load - load_offset,
                float(np.mean(values)),
                float(np.quantile(values, 0.10)),
                float(np.quantile(values, 0.90)),
            )
        )

    if not rows:
        if load_window is None:
            raise FileNotFoundError(f"No VTU files found for {sim_dir}")
        raise FileNotFoundError(f"No VTU files found for {sim_dir} in {load_window}")
    return np.array(sorted(rows), dtype=float)


def find_simulation_dirs(output_root: Path) -> list[Path]:
    sim_dirs = [path for path in output_root.iterdir() if (path / "data").is_dir()]
    if not sim_dirs:
        raise FileNotFoundError(f"No simulation output folders found in {output_root}")
    return sorted(sim_dirs, key=variant_sort_key)


def collect_continuous_window_series(
    output_root: Path,
    windows: list[tuple[float, float]],
    *,
    eulerian: bool,
    loops: int,
):
    all_series = {}
    expected_base_labels = {"simple shear, no reconnection", "simple shear, edge flip"}
    sim_dirs = find_simulation_dirs(output_root)
    seen_base_labels = {variant_label(sim_dir) for sim_dir in sim_dirs}
    missing = expected_base_labels - seen_base_labels
    if missing:
        raise ValueError(
            "Continuous window comparison needs completed simple-shear runs for: "
            + ", ".join(sorted(missing))
        )

    for sim_dir in sim_dirs:
        base_label = variant_label(sim_dir)
        if base_label not in expected_base_labels:
            continue
        reconnection = "edge flip" if "edge flip" in base_label else "no reconnection"
        for start, stop in windows:
            if start == 0.0:
                label = base_label
            else:
                label = f"current shear {start:g} to {stop:g}, {reconnection}"
            all_series[label] = collect_series(
                sim_dir,
                eulerian=eulerian,
                loops=loops,
                local_load=True,
                load_window=(start, stop),
            )

    return all_series


def variant_sort_key(sim_dir: Path):
    label = variant_label(sim_dir)
    if label == "simple shear, no reconnection":
        return (0, 0.0, 0)
    if label == "simple shear, edge flip":
        return (1, 0.0, 0)
    shear = current_shear_from_text(label)
    if shear is not None:
        return (2, shear, 1 if "edge flip" in label else 0)
    shear = reference_shear_from_text(label)
    if shear is not None:
        return (3, shear, 1 if "edge flip" in label else 0)
    return (9, label, 0)


def plot_stiffness(
    output_root: Path,
    out_path: Path,
    *,
    eulerian: bool,
    loops: int,
    local_load: bool,
):
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    component = "a1212" if eulerian else "A1212"

    for sim_dir in find_simulation_dirs(output_root):
        series = collect_series(
            sim_dir,
            eulerian=eulerian,
            loops=loops,
            local_load=local_load,
        )
        load, mean, q10, q90 = series.T
        label = variant_label(sim_dir)
        style = variant_style(label)
        (line,) = ax.plot(
            load,
            mean,
            marker="o",
            markersize=2.5,
            linewidth=1.5,
            label=label,
            **style,
        )
        ax.fill_between(
            load,
            q10,
            q90,
            color=line.get_color(),
            alpha=0.16,
            linewidth=0,
        )

    ax.set_xlabel("local simple shear load" if local_load else "simple shear load")
    ax.set_ylabel(f"mean {component} over elements")
    ax.set_title(f"{component} stiffness component")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_difference_from_reference_on_axis(
    output_root: Path,
    ax,
    *,
    eulerian: bool,
    loops: int,
    local_load: bool,
    title: str | None = None,
):
    all_series = {
        variant_label(sim_dir): collect_series(
            sim_dir,
            eulerian=eulerian,
            loops=loops,
            local_load=local_load,
        )
        for sim_dir in find_simulation_dirs(output_root)
    }
    plot_difference_series_on_axis(
        all_series,
        ax,
        eulerian=eulerian,
        local_load=local_load,
        title=title,
    )


def plot_difference_series_on_axis(
    all_series: dict[str, np.ndarray],
    ax,
    *,
    eulerian: bool,
    local_load: bool,
    title: str | None = None,
):
    component = "a1212" if eulerian else "A1212"
    reference_label = "simple shear, no reconnection"
    if reference_label not in all_series:
        raise ValueError(f"Missing reference series: {reference_label}")

    reference = all_series[reference_label]
    ref_load = reference[:, 0]
    ref_mean = reference[:, 1]
    zero_labels = []

    for label, series in all_series.items():
        if label == reference_label:
            continue

        series_by_load = {
            round(float(load), 12): float(mean) for load, mean in series[:, :2]
        }
        common = [
            (load, mean, series_by_load[round(float(load), 12)])
            for load, mean in zip(ref_load, ref_mean)
            if round(float(load), 12) in series_by_load
        ]
        if not common:
            raise ValueError(f"No common load values between reference and {label}")

        load = np.array([row[0] for row in common])
        ref_mean_common = np.array([row[1] for row in common])
        series_mean_common = np.array([row[2] for row in common])
        signed_difference = ref_mean_common - series_mean_common
        magnitude = np.abs(signed_difference)
        positive = magnitude > 0
        if not np.any(positive):
            zero_labels.append(label)
            continue

        style = variant_style(label)
        style["zorder"] = style.get("zorder", 2) + 1
        ax.plot(
            load[positive],
            magnitude[positive],
            marker="o",
            markersize=2.5,
            linewidth=1.5,
            label=label,
            **style,
        )

    if zero_labels:
        ax.text(
            0.02,
            0.04,
            "exactly zero on this grid:\n" + "\n".join(zero_labels),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            zorder=10,
            bbox={"facecolor": "white", "alpha": 0.95, "edgecolor": "none"},
        )

    ax.set_yscale("log")
    ax.set_xlabel("local simple shear load" if local_load else "simple shear load")
    ax.set_ylabel(f"|{component}(no reconnection) - {component}(variant)|")
    ax.set_title(title or f"{component} difference from no-reconnection reference")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)


def plot_difference_from_reference(
    output_root: Path,
    out_path: Path,
    *,
    eulerian: bool,
    loops: int,
    local_load: bool,
):
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    plot_difference_from_reference_on_axis(
        output_root,
        ax,
        eulerian=eulerian,
        loops=loops,
        local_load=local_load,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_difference_column(
    reference_output_root: Path,
    current_output_root: Path,
    out_path: Path,
    *,
    eulerian: bool,
    loops: int,
    current_windows: list[tuple[float, float]] | None = None,
):
    component = "a1212" if eulerian else "A1212"
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 8.6), constrained_layout=True)
    plot_difference_from_reference_on_axis(
        reference_output_root,
        axes[0],
        eulerian=eulerian,
        loops=loops,
        local_load=False,
        title=f"{component}: distorted reference state",
    )
    if current_windows is None:
        plot_difference_from_reference_on_axis(
            current_output_root,
            axes[1],
            eulerian=eulerian,
            loops=loops,
            local_load=True,
            title=f"{component}: distorted current state",
        )
    else:
        all_series = collect_continuous_window_series(
            current_output_root,
            current_windows,
            eulerian=eulerian,
            loops=loops,
        )
        plot_difference_series_on_axis(
            all_series,
            axes[1],
            eulerian=eulerian,
            local_load=True,
            title=f"{component}: distorted current state",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the 1212 component of the element stiffness/Hessian from "
            "MTS2D VTUs."
        )
    )
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("Plots/stiffness_1212.png"))
    parser.add_argument(
        "--eulerian",
        action="store_true",
        help="Plot pushed-forward tangent a1212 instead of Hessian A1212.",
    )
    parser.add_argument("--loops", type=int, default=1000)
    parser.add_argument(
        "--local-load",
        action="store_true",
        help="Subtract each simulation's startLoad before comparing or plotting.",
    )
    parser.add_argument(
        "--difference-from-reference",
        action="store_true",
        help=(
            "Plot absolute difference from the simple-shear no-reconnection "
            "reference on a log y-axis."
        ),
    )
    parser.add_argument(
        "--column-with",
        type=Path,
        help=(
            "Create a two-row difference plot with this output root as the "
            "current-state distortion panel."
        ),
    )
    parser.add_argument(
        "--continuous-windows",
        help=(
            "Slice continuous current-state runs into load windows written as "
            "start:stop,start:stop. For example: 0:1,2:3,5:6,10:11."
        ),
    )
    args = parser.parse_args()

    continuous_windows = (
        None if args.continuous_windows is None else parse_windows(args.continuous_windows)
    )

    if args.column_with is not None:
        out_path = plot_difference_column(
            args.output_root,
            args.column_with,
            args.out,
            eulerian=args.eulerian,
            loops=args.loops,
            current_windows=continuous_windows,
        )
    elif continuous_windows is not None:
        fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
        all_series = collect_continuous_window_series(
            args.output_root,
            continuous_windows,
            eulerian=args.eulerian,
            loops=args.loops,
        )
        plot_difference_series_on_axis(
            all_series,
            ax,
            eulerian=args.eulerian,
            local_load=True,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=220)
        plt.close(fig)
        out_path = args.out
    elif args.difference_from_reference:
        out_path = plot_difference_from_reference(
            args.output_root,
            args.out,
            eulerian=args.eulerian,
            loops=args.loops,
            local_load=args.local_load,
        )
    else:
        out_path = plot_stiffness(
            args.output_root,
            args.out,
            eulerian=args.eulerian,
            loops=args.loops,
            local_load=args.local_load,
        )
    print(out_path)


if __name__ == "__main__":
    main()
