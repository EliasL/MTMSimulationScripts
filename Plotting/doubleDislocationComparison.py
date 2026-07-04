import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotAll import plotAll
from Plotting.makeAnimations import framesToMp4, select_vtu_files
from Plotting.dataFunctions import get_data_from_name, resolve_vtu_files


DOUBLE_DISLOCATION_ROOT = Path("/Volumes/data/MTS2D_output/DoubleDislocationTest")


@dataclass(frozen=True)
class RunSpec:
    slug: str
    label: str
    method: str
    edge_flip: bool
    minor: bool
    include_difference: bool


RUN_SPECS = (
    RunSpec(
        "major_reference",
        "less mesh locking",
        "singleTriangle",
        edge_flip=False,
        minor=False,
        include_difference=False,
    ),
    RunSpec(
        "minor_reference",
        "mesh locking",
        "singleTriangle",
        edge_flip=False,
        minor=True,
        include_difference=True,
    ),
    RunSpec(
        "lev_edge_flip_minor",
        "double triangle edge flip",
        "LevDoubleTriangle",
        edge_flip=True,
        minor=True,
        include_difference=True,
    ),
    RunSpec(
        "single_edge_flip_minor",
        "single triangle edge flip",
        "singleTriangle",
        edge_flip=True,
        minor=True,
        include_difference=True,
    ),
)


def _coerce_tlim(t_lim, x_lim, gamma_lim):
    limits = [limit for limit in (t_lim, x_lim, gamma_lim) if limit is not None]
    if len(limits) > 1:
        raise ValueError("Use only one of t_lim, x_lim, or gamma_lim.")
    return limits[0] if limits else None


def _time_tag(t_max):
    return f"t{str(t_max).replace('.', 'p')}"


def _legacy_gamma_tag(t_max):
    return f"gamma{str(t_max).replace('.', 'p')}"


def _double_dislocation_folder(root, method, *, edge_flip, minor):
    folders = [
        path
        for path in (root / method).iterdir()
        if path.is_dir()
        and ("edgeFlip" in path.name) == edge_flip
        and ("meshDiagonalminor" in path.name) == minor
    ]
    if len(folders) != 1:
        raise ValueError(
            f"Expected one folder for {method=}, {edge_flip=}, {minor=}; found {len(folders)}"
        )
    return folders[0]


def _macro_data(path, x_col, energy_col):
    csv_path = path / "macroData.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    for col in (x_col, energy_col):
        if col not in df.columns:
            raise ValueError(f"{col} not found in {csv_path}")
    return df[[x_col, energy_col]].copy()


def _require_same_data(left, right, description, x_col, energy_col):
    if (
        len(left) != len(right)
        or not np.array_equal(left[x_col].to_numpy(), right[x_col].to_numpy())
        or not np.array_equal(left[energy_col].to_numpy(), right[energy_col].to_numpy())
    ):
        raise ValueError(f"{description} differs between the two method folders")


def _load_runs(root=DOUBLE_DISLOCATION_ROOT, x_col="load", energy_col="total_energy"):
    root = Path(root)
    runs = []
    for spec in RUN_SPECS:
        path = _double_dislocation_folder(
            root, spec.method, edge_flip=spec.edge_flip, minor=spec.minor
        )
        runs.append(
            {
                "spec": spec,
                "path": path,
                "data": _macro_data(path, x_col, energy_col),
            }
        )

    major_from_lev = _macro_data(
        _double_dislocation_folder(
            root, "LevDoubleTriangle", edge_flip=False, minor=False
        ),
        x_col,
        energy_col,
    )
    minor_from_lev = _macro_data(
        _double_dislocation_folder(root, "LevDoubleTriangle", edge_flip=False, minor=True),
        x_col,
        energy_col,
    )
    _require_same_data(
        runs[0]["data"],
        major_from_lev,
        "Non-flipping major reference",
        x_col,
        energy_col,
    )
    _require_same_data(
        runs[1]["data"],
        minor_from_lev,
        "Non-flipping minor reference",
        x_col,
        energy_col,
    )
    return runs


def _mask_t(t, t_lim):
    mask = np.ones(len(t), dtype=bool)
    if t_lim is not None:
        tmin, tmax = t_lim
        if tmin is not None:
            mask &= t >= tmin
        if tmax is not None:
            mask &= t <= tmax
    if not np.any(mask):
        raise ValueError(f"No data points found inside {t_lim=}")
    return mask


def _reference_difference(run, reference, x_col, energy_col, t_lim, absolute=True):
    df = run["data"]
    ref = reference["data"]
    if len(df) != len(ref) or not np.allclose(df[x_col], ref[x_col]):
        raise ValueError(f"{run['spec'].label} loads do not match the reference loads")
    x = df[x_col].to_numpy(dtype=float)
    y = df[energy_col].to_numpy(dtype=float) - ref[energy_col].to_numpy(dtype=float)
    if absolute:
        y = np.abs(y)
    mask = _mask_t(x, t_lim)
    return x[mask], y[mask]


def _positive_for_log(y, label):
    y = np.asarray(y, dtype=float).copy()
    y[y <= 0] = np.nan
    if np.all(np.isnan(y)):
        raise ValueError(f"No positive values to show on a log axis for {label}")
    return y


def _difference_ylim(runs, reference, x_col, energy_col, t_lim, include_slugs):
    values = []
    for run in runs:
        if run["spec"].slug not in include_slugs:
            continue
        _, y = _reference_difference(run, reference, x_col, energy_col, t_lim)
        values.append(_positive_for_log(y, run["spec"].label))
    y = np.concatenate(values)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        raise ValueError("No positive edge-flip energy differences found")
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if ymin == ymax:
        return ymin / 2, ymax * 2
    margin = 10 ** (0.08 * (np.log10(ymax) - np.log10(ymin)))
    return ymin / margin, ymax * margin


def _plot_energy_axis(energy_ax, runs, colors, *, t_lim, x_col, energy_col):
    for run in runs:
        x = run["data"][x_col].to_numpy(dtype=float)
        y = run["data"][energy_col].to_numpy(dtype=float)
        mask = _mask_t(x, t_lim)
        energy_ax.plot(
            x[mask],
            y[mask],
            label=run["spec"].label,
            color=colors[run["spec"].slug],
            linewidth=1.5,
        )
    energy_ax.set_xlabel(r"$t$")
    energy_ax.set_ylabel(r"$E$")
    if t_lim is not None:
        energy_ax.set_xlim(t_lim)
    energy_ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    energy_ax.tick_params(labelsize=8)
    energy_ax.legend(loc="best", fontsize=7, framealpha=0.9)


def _plot_difference_axis(
    difference_ax, runs, colors, *, t_lim, x_col, energy_col, legend_loc
):
    reference = runs[0]
    difference_runs = [run for run in runs if run["spec"].include_difference]
    difference_runs.sort(key=lambda run: run["spec"].slug == "minor_reference")
    for run in difference_runs:
        x, y = _reference_difference(run, reference, x_col, energy_col, t_lim)
        linestyle = "--" if run["spec"].slug == "minor_reference" else "-"
        zorder = 4 if linestyle == "--" else 2
        difference_ax.plot(
            x,
            _positive_for_log(y, run["spec"].label),
            linestyle=linestyle,
            color=colors[run["spec"].slug],
            label=run["spec"].label,
            linewidth=1.5,
            zorder=zorder,
        )
    difference_ax.set_ylim(
        _difference_ylim(
            runs,
            reference,
            x_col,
            energy_col,
            t_lim,
            {"lev_edge_flip_minor", "single_edge_flip_minor"},
        )
    )
    difference_ax.set_yscale("log")
    difference_ax.set_xlabel(r"$t$")
    difference_ax.set_ylabel(r"$|\Delta E|$")
    if t_lim is not None:
        difference_ax.set_xlim(t_lim)
    difference_ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    difference_ax.tick_params(labelsize=8)
    difference_ax.legend(loc=legend_loc, fontsize=7, framealpha=0.9)


def _plot_energy_axes(
    energy_ax,
    difference_ax,
    *,
    t_lim,
    energy_col,
    legend_loc,
):
    x_col = "load"
    runs = _load_runs(x_col=x_col, energy_col=energy_col)
    colors = {
        "major_reference": "tab:blue",
        "minor_reference": "tab:orange",
        "lev_edge_flip_minor": "tab:green",
        "single_edge_flip_minor": "tab:purple",
    }
    _plot_energy_axis(energy_ax, runs, colors, t_lim=t_lim, x_col=x_col, energy_col=energy_col)
    _plot_difference_axis(
        difference_ax,
        runs,
        colors,
        t_lim=t_lim,
        x_col=x_col,
        energy_col=energy_col,
        legend_loc=legend_loc,
    )
    return runs


def doubleDislocationComparison(
    show=True,
    t_lim=None,
    x_lim=None,
    gamma_lim=None,
    energy_col="total_energy",
    legend_loc="lower right",
    output_path=None,
    save=True,
):
    t_lim = _coerce_tlim(t_lim, x_lim, gamma_lim)
    if t_lim is None:
        t_lim = (0, 2)
    x_col = "load"
    runs = _load_runs(x_col=x_col, energy_col=energy_col)
    colors = {
        "major_reference": "tab:blue",
        "minor_reference": "tab:orange",
        "lev_edge_flip_minor": "tab:green",
        "single_edge_flip_minor": "tab:purple",
    }
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _plot_difference_axis(
        ax,
        runs,
        colors,
        t_lim=t_lim,
        x_col=x_col,
        energy_col=energy_col,
        legend_loc=legend_loc,
    )
    fig.tight_layout()

    if save:
        Path("Plots").mkdir(exist_ok=True)
        out_path = output_path or Path("Plots") / "double_dislocation_comparison.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    if show:
        plt.show()
    return fig, ax


def doubleDislocationEnergy(
    show=True,
    t_lim=None,
    x_lim=None,
    gamma_lim=None,
    energy_col="total_energy",
    output_path=None,
    save=True,
):
    t_lim = _coerce_tlim(t_lim, x_lim, gamma_lim)
    if t_lim is None:
        t_lim = (0, 2)
    x_col = "load"
    runs = _load_runs(x_col=x_col, energy_col=energy_col)
    colors = {
        "major_reference": "tab:blue",
        "minor_reference": "tab:orange",
        "lev_edge_flip_minor": "tab:green",
        "single_edge_flip_minor": "tab:purple",
    }
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _plot_energy_axis(
        ax,
        runs,
        colors,
        t_lim=t_lim,
        x_col=x_col,
        energy_col=energy_col,
    )
    fig.tight_layout()

    if save:
        Path("Plots").mkdir(exist_ok=True)
        out_path = output_path or Path("Plots") / "double_dislocation_energy.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    if show:
        plt.show()
    return fig, ax


def _prepare_video_input(source_path, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    for filename in ("collection.pvd", "macroData.csv"):
        source_file = source_path / filename
        if not source_file.exists():
            raise FileNotFoundError(source_file)
        shutil.copy2(source_file, output_path / filename)

    data_link = output_path / "data"
    data_target = source_path / "data"
    if not data_target.is_dir():
        raise FileNotFoundError(data_target)
    if data_link.exists() or data_link.is_symlink():
        if not data_link.is_symlink() or data_link.resolve() != data_target.resolve():
            raise FileExistsError(f"{data_link} exists and is not the expected data symlink")
    else:
        data_link.symlink_to(data_target, target_is_directory=True)
    return output_path / "collection.pvd"


def _selected_movie_loads(
    pvd_file, t_lim, fps, seconds_per_unit_shear, all_images, min_time
):
    vtu_files = resolve_vtu_files(pvd_file)
    if t_lim is not None:
        tmin, tmax = t_lim
        vtu_files = [
            vtu_file
            for vtu_file in vtu_files
            if (tmin is None or float(get_data_from_name(vtu_file)["load"]) >= tmin)
            and (tmax is None or float(get_data_from_name(vtu_file)["load"]) <= tmax)
        ]
    if not vtu_files:
        raise ValueError(f"No VTU files found inside {t_lim=}")

    first = get_data_from_name(vtu_files[0])
    last = get_data_from_name(vtu_files[-1])
    x_change = float(last["load"]) - float(first["load"])
    nr_steps = seconds_per_unit_shear * x_change * fps
    selected = select_vtu_files(vtu_files, nr_steps, all_images)
    if len(selected) < nr_steps:
        fps = len(selected) / min_time
    loads = [float(get_data_from_name(vtu_file)["load"]) for vtu_file in selected]
    return loads, fps


def _mesh_frame_sets(local_dirs, frame_count=None):
    frame_sets = [
        sorted((local_dir / "frames" / "mesh").glob("mesh_frame_*.png"))
        for local_dir in local_dirs
    ]
    if frame_count is not None:
        too_short = [
            len(frame_set) for frame_set in frame_sets if len(frame_set) < frame_count
        ]
        if too_short:
            raise ValueError(
                f"Expected at least {frame_count} mesh frames, found {sorted(too_short)}"
            )
        frame_sets = [frame_set[:frame_count] for frame_set in frame_sets]
    lengths = {len(frame_set) for frame_set in frame_sets}
    if 0 in lengths:
        raise FileNotFoundError("Missing mesh frames for combined mesh video")
    if len(lengths) != 1:
        min_length = min(lengths)
        max_length = max(lengths)
        if max_length - min_length > 1:
            raise ValueError(f"Mesh frame counts differ: {sorted(lengths)}")
        frame_sets = [frame_set[:min_length] for frame_set in frame_sets]
    return frame_sets


def _crop_box(image, threshold=0.985):
    rgb = image[..., :3]
    mask = np.any(rgb < threshold, axis=2)
    if not np.any(mask):
        return 0, image.shape[0], 0, image.shape[1]
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _crop_boxes(frame_sets, sample_count=7, margin=10):
    boxes = []
    for frame_set in frame_sets:
        sample_indices = np.linspace(0, len(frame_set) - 1, sample_count, dtype=int)
        run_boxes = [None, None, None, None]
        shape = None
        for index in sample_indices:
            image = plt.imread(frame_set[index])
            shape = image.shape
            y0, y1, x0, x1 = _crop_box(image)
            if run_boxes[0] is None:
                run_boxes = [y0, y1, x0, x1]
            else:
                run_boxes = [
                    min(run_boxes[0], y0),
                    max(run_boxes[1], y1),
                    min(run_boxes[2], x0),
                    max(run_boxes[3], x1),
                ]
        if shape is None:
            raise ValueError("Cannot crop an empty frame set")
        boxes.append(
            (
                max(0, run_boxes[0] - margin),
                min(shape[0], run_boxes[1] + margin),
                max(0, run_boxes[2] - margin),
                min(shape[1], run_boxes[3] + margin),
            )
        )
    return boxes


def _resize_and_pad(image, size):
    width, height = size
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    fill = 1.0 if np.issubdtype(resized.dtype, np.floating) else 255
    canvas = np.full((height, width, resized.shape[2]), fill, dtype=resized.dtype)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _read_mesh_panel(frame_path, crop_box, panel_size):
    image = plt.imread(frame_path)
    y0, y1, x0, x1 = crop_box
    return _resize_and_pad(image[y0:y1, x0:x1], panel_size)


def _add_mesh_legend(ax, label):
    handle = mlines.Line2D([], [], linestyle="none", label=label)
    ax.legend(
        handles=[handle],
        loc="lower left",
        bbox_to_anchor=(0.05, 0.08),
        frameon=True,
        framealpha=0.9,
        fontsize=10,
        handlelength=0,
        handletextpad=0,
        borderpad=0.35,
    )


def _mesh_axes(fig, with_energy_plots):
    if with_energy_plots:
        positions = (
            [0.000, 0.500, 1 / 3, 0.500],
            [1 / 3, 0.500, 1 / 3, 0.500],
            [0.000, 0.000, 1 / 3, 0.500],
            [1 / 3, 0.000, 1 / 3, 0.500],
        )
    else:
        positions = (
            [0.000, 0.500, 0.500, 0.500],
            [0.500, 0.500, 0.500, 0.500],
            [0.000, 0.000, 0.500, 0.500],
            [0.500, 0.000, 0.500, 0.500],
        )
    axes = [fig.add_axes(position) for position in positions]
    for ax in axes:
        ax.set_axis_off()
    return axes


def _plot_axes(fig):
    return (
        fig.add_axes([0.715, 0.575, 0.265, 0.355]),
        fig.add_axes([0.715, 0.105, 0.265, 0.355]),
    )


def _make_composed_video(
    frame_sets,
    labels,
    loads,
    t_lim,
    output_path,
    fps,
    *,
    with_energy_plots,
    energy_col,
    reuse_images,
):
    frame_count = len(frame_sets[0])
    if len(loads) < frame_count:
        raise ValueError(f"Expected {frame_count} loads, found {len(loads)}")
    loads = loads[:frame_count]

    output_path = Path(output_path)
    frame_dir = output_path.parent / f"{output_path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    output_frames = [
        frame_dir / f"{output_path.stem}_frame_{i:04d}.png"
        for i in range(frame_count)
    ]

    if not reuse_images or not all(path.exists() for path in output_frames):
        panel_size = (640, 544)
        crop_boxes = _crop_boxes(frame_sets)
        figure_size = (19.2, 10.88) if with_energy_plots else (12.8, 10.88)
        fig = plt.figure(figsize=figure_size, dpi=100)
        mesh_axes = _mesh_axes(fig, with_energy_plots)
        mesh_images = []
        for ax, frame_set, crop_box, label in zip(
            mesh_axes, frame_sets, crop_boxes, labels
        ):
            image = _read_mesh_panel(frame_set[0], crop_box, panel_size)
            mesh_images.append(ax.imshow(image, aspect="auto"))
            _add_mesh_legend(ax, label)

        current_lines = []
        if with_energy_plots:
            energy_ax, difference_ax = _plot_axes(fig)
            _plot_energy_axes(
                energy_ax,
                difference_ax,
                t_lim=t_lim,
                energy_col=energy_col,
                legend_loc="lower right",
            )
            current_lines = [
                energy_ax.axvline(loads[0], color="red", linewidth=1.5),
                difference_ax.axvline(loads[0], color="red", linewidth=1.5),
            ]

        for index, frame_path in enumerate(output_frames):
            for image_artist, frame_set, crop_box in zip(
                mesh_images, frame_sets, crop_boxes
            ):
                image_artist.set_data(
                    _read_mesh_panel(frame_set[index], crop_box, panel_size)
                )
            for line in current_lines:
                line.set_xdata([loads[index], loads[index]])
            fig.savefig(frame_path, dpi=100)
        plt.close(fig)

    framesToMp4([str(path) for path in output_frames], str(output_path), fps)


def _ensure_mesh_inputs(
    source_runs,
    work_dir,
    t_lim,
    fps,
    seconds_per_unit_shear,
    min_time,
    reuse_images,
    stage_inputs,
):
    local_dirs = []
    local_pvds = []
    for run in source_runs:
        if stage_inputs:
            local_dir = work_dir / f"{run['spec'].slug}__{run['path'].name}"
            local_pvd = _prepare_video_input(run["path"], local_dir)
        else:
            local_dir = run["path"]
            local_pvd = local_dir / "collection.pvd"
            if not local_pvd.exists():
                raise FileNotFoundError(local_pvd)
        local_dirs.append(local_dir)
        local_pvds.append(local_pvd)
        frames = sorted((local_dir / "frames" / "mesh").glob("mesh_frame_*.png"))
        if stage_inputs and frames:
            continue
        plotAll(
            str(local_pvd),
            plots=False,
            videoes=True,
            videoNames=["mesh"],
            combineVideos=False,
            xlim=t_lim,
            reuseImages=reuse_images,
            allImages=True,
            fps=fps,
            seconds_per_unit_shear=seconds_per_unit_shear,
            minTime=min_time,
            num_processes=1,
            useTqdm=False,
        )
    return local_dirs, local_pvds


def doubleDislocationMeshMovies(
    t_max_values=(2, 4),
    x_max_values=None,
    gamma_max_values=None,
    output_root=Path("Plots") / "double_dislocation_movies",
    fps=30,
    seconds_per_unit_shear=15,
    min_time=7,
    reuse_images=True,
    energy_col="total_energy",
    stage_inputs=False,
):
    if x_max_values is not None:
        t_max_values = x_max_values
    if gamma_max_values is not None:
        t_max_values = gamma_max_values
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_runs = _load_runs(energy_col=energy_col)
    labels = [run["spec"].label for run in source_runs]

    outputs = {}
    for t_max in t_max_values:
        t_lim = (0, t_max)
        tag = _time_tag(t_max)
        legacy_dir = output_root / _legacy_gamma_tag(t_max)
        work_dir = legacy_dir if legacy_dir.exists() else output_root / tag
        work_dir.mkdir(parents=True, exist_ok=True)
        local_dirs, local_pvds = _ensure_mesh_inputs(
            source_runs,
            work_dir,
            t_lim,
            fps,
            seconds_per_unit_shear,
            min_time,
            reuse_images,
            stage_inputs,
        )
        loads, movie_fps = _selected_movie_loads(
            local_pvds[0], t_lim, fps, seconds_per_unit_shear, True, min_time
        )
        frame_sets = _mesh_frame_sets(local_dirs, frame_count=len(loads))

        mesh_output = output_root / f"double_dislocation_meshes_{tag}.mp4"
        _make_composed_video(
            frame_sets,
            labels,
            loads,
            t_lim,
            mesh_output,
            movie_fps,
            with_energy_plots=False,
            energy_col=energy_col,
            reuse_images=reuse_images,
        )

        combined_output = (
            output_root / f"double_dislocation_meshes_with_energy_plots_{tag}.mp4"
        )
        _make_composed_video(
            frame_sets,
            labels,
            loads,
            t_lim,
            combined_output,
            movie_fps,
            with_energy_plots=True,
            energy_col=energy_col,
            reuse_images=reuse_images,
        )
        outputs[tag] = {
            "mesh": mesh_output,
            "mesh_with_energy_plots": combined_output,
        }
    return outputs
