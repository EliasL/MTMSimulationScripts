from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("/private/tmp/mts2d_matplotlib_cache").resolve())
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imageio_ffmpeg import get_ffmpeg_exe

from Plotting.element_tracking import ElementMatrixHistory
from Plotting.element_tracking_animation import (
    build_gamma_timeline,
    compose_picture_in_picture,
    load_element_matrix_histories,
    load_mesh_neighborhoods,
    render_mesh_animation,
    render_periodic_mesh_animation,
    render_poincare_animation,
)


TARGET_CELL = 389
BOX_SIZE = 40.0
JUMP_LOAD = 1.55835
MAIN_DATA_DIR = Path(
    "/Volumes/data/MTS2D_output/"
    "simpleShear,s40x40l0.15,1e-05,5.1PBCedgeFlipt5epsR1e-05"
    "LBFGSEpsg1e-08LBFGSEpsx1e-06s0/data"
)
DETAIL_DATA_DIR = Path(
    "/Volumes/data/MTS2D_output/T21_jump_resume_tests/"
    "reproduce_s40_T21_fulllog_from_l1.55834/data/minimizationData/step140836"
)
DEFAULT_OUTPUT_DIR = ROOT / "output" / "animations"

LOAD_RE = re.compile(r"_load=([-+0-9.eE]+)_")
MIN_STEP_RE = re.compile(r"minStep=(\d+)\.(\d+)")
NRM_RE = re.compile(r"_nrM=(\d+)_")


def parse_load(path: Path) -> float:
    match = LOAD_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse load from {path.name}")
    return float(match.group(1))


def detail_sort_key(path: Path) -> tuple[int, int, int, str]:
    min_step = MIN_STEP_RE.search(path.name)
    nrm = NRM_RE.search(path.name)
    if min_step is None or nrm is None:
        raise ValueError(f"Could not parse minimization order from {path.name}")
    role = 0 if "_pre." in path.name else 1 if "_post." in path.name else 2
    return int(min_step.group(2)), role, int(nrm.group(1)), path.name


def physical_vtu_paths(data_dir: Path | None = None) -> tuple[Path, ...]:
    if data_dir is not None:
        if (data_dir / "data").is_dir():
            data_dir = data_dir / "data"
        if not data_dir.is_dir():
            raise FileNotFoundError(data_dir)
        paths = sorted(
            data_dir.glob("*.vtu"), key=lambda path: (parse_load(path), path.name)
        )
        if not paths:
            raise RuntimeError(f"No physical VTU states found in {data_dir}")
        return tuple(paths)

    if not MAIN_DATA_DIR.is_dir() or not DETAIL_DATA_DIR.is_dir():
        raise FileNotFoundError("The main or detailed VTU directory is missing.")
    main = sorted(
        MAIN_DATA_DIR.glob("*.vtu"), key=lambda path: (parse_load(path), path.name)
    )
    detail = sorted(
        (
            path
            for path in DETAIL_DATA_DIR.glob("*.vtu")
            if "_reference." not in path.name
        ),
        key=detail_sort_key,
    )
    if not main or not detail:
        raise RuntimeError("Expected both main and detailed physical VTU states.")
    return tuple(
        [path for path in main if parse_load(path) < JUMP_LOAD - 1e-12]
        + detail
        + [path for path in main if parse_load(path) > JUMP_LOAD + 1e-12]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate one element in the Poincare disk and physical mesh."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Simulation root or data directory; omit for the detailed cell-389 run.",
    )
    parser.add_argument("--element", type=int, default=TARGET_CELL)
    parser.add_argument("--box-size", type=float, default=BOX_SIZE)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--matrix", choices=("T", "T_p"), default="T",
                        help="Track total deformation (default) or legacy plastic history.")
    parser.add_argument("--output-prefix")
    parser.add_argument("--gamma-per-frame", type=float, default=0.002)
    parser.add_argument("--camera-smoothing-gamma", type=float, default=0.04)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--periodic-dpi", type=int, default=48)
    parser.add_argument("--skip-poincare", action="store_true")
    parser.add_argument("--skip-mesh", action="store_true")
    parser.add_argument("--skip-periodic-mesh", action="store_true")
    parser.add_argument("--skip-composite", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    return parser.parse_args()


def cached_matrix_histories(
    paths: tuple[Path, ...],
    element_index: int,
    cache_path: Path,
) -> tuple[ElementMatrixHistory, ElementMatrixHistory]:
    """Cache the small tracked histories, with strict source-file validation."""
    source_paths = np.asarray([str(path) for path in paths])
    signatures = np.asarray(
        [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths],
        dtype=np.int64,
    )
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            required = {"source_paths", "signatures", "element", "T", "F_E"}
            if set(cache.files) != required:
                raise ValueError(f"Unexpected matrix-cache contents in {cache_path}")
            matches = (
                int(cache["element"].item()) == element_index
                and np.array_equal(cache["source_paths"], source_paths)
                and np.array_equal(cache["signatures"], signatures)
            )
            if matches:
                print(f"loading tracked matrices from {cache_path}", flush=True)
                return (
                    ElementMatrixHistory(paths, cache["T"], element_index, "T"),
                    ElementMatrixHistory(paths, cache["F_E"], element_index, "F_E"),
                )
        print(f"matrix cache is stale; rebuilding {cache_path}", flush=True)

    histories = load_element_matrix_histories(paths, element_index, ("T", "F_E"))
    history, elastic_history = histories
    np.savez_compressed(
        cache_path,
        source_paths=source_paths,
        signatures=signatures,
        element=np.asarray(element_index),
        T=history.matrices,
        F_E=elastic_history.matrices,
    )
    return history, elastic_history


def main() -> None:
    args = parse_args()
    if args.element < 0 or args.box_size <= 0 or args.periodic_dpi <= 0:
        raise ValueError(
            "element must be non-negative; box-size and periodic-dpi must be positive."
        )
    output_dir = args.output_directory.resolve()
    paths = physical_vtu_paths(args.data_dir)
    loads = [parse_load(path) for path in paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    default_prefix = f"cell{args.element}"
    if args.data_dir is not None:
        default_prefix += f"_s{args.box_size:g}x{args.box_size:g}"
    prefix = args.output_prefix or default_prefix
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        raise ValueError("output-prefix may contain only letters, digits, dot, _ and -.")

    print(f"loading T and F_E from {len(paths)} physical VTU states", flush=True)
    history, elastic_history = cached_matrix_histories(
        paths,
        args.element,
        output_dir / f"{default_prefix}_matrix_histories.npz",
    )
    reconstructed_history = ElementMatrixHistory(
        paths,
        elastic_history.matrices @ history.matrices,
        args.element,
        "F_E*T",
    )
    if np.any(np.linalg.det(reconstructed_history.matrices) <= 0):
        raise ValueError("F_E*T must remain orientation preserving.")
    if args.matrix == "T":
        history = reconstructed_history
    timeline = build_gamma_timeline(
        history,
        loads,
        reconstructed_history,
        gamma_per_frame=args.gamma_per_frame,
        camera_smoothing_gamma=args.camera_smoothing_gamma,
    )
    print(
        f"timeline: {len(timeline.node_history_indices)} nodes, "
        f"{timeline.frame_count} frames, "
        f"delta gamma={timeline.frame_loads[1] - timeline.frame_loads[0]:.7g}",
        flush=True,
    )

    ffmpeg = get_ffmpeg_exe()
    poincare_path = output_dir / f"{prefix}_poincare_recentered.mp4"
    mesh_path = output_dir / f"{prefix}_mesh_neighborhood.mov"
    periodic_mesh_path = output_dir / f"{prefix}_periodic_mesh.mov"
    composite_path = output_dir / f"{prefix}_poincare_with_mesh.mp4"

    if not args.skip_poincare:
        render_poincare_animation(
            history,
            timeline,
            poincare_path,
            ffmpeg_executable=ffmpeg,
            reconstructed_history=reconstructed_history if args.matrix == "T_p" else None,
            grid_mode="fixed" if args.matrix == "T" else "integer",
            fps=args.fps,
            dpi=args.dpi,
        )
    elif not args.skip_composite and not poincare_path.is_file():
        raise FileNotFoundError(poincare_path)

    if not args.skip_mesh:
        print("loading target-centred mesh neighbourhoods", flush=True)
        snapshots = load_mesh_neighborhoods(paths, args.element)
        render_mesh_animation(
            history,
            timeline,
            snapshots,
            mesh_path,
            ffmpeg_executable=ffmpeg,
            fps=args.fps,
            dpi=args.dpi,
        )
    elif not args.skip_composite and not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    if not args.skip_periodic_mesh:
        print("streaming full periodic meshes", flush=True)
        render_periodic_mesh_animation(
            timeline,
            paths,
            loads,
            args.element,
            periodic_mesh_path,
            box_size=args.box_size,
            ffmpeg_executable=ffmpeg,
            fps=args.fps,
            dpi=args.periodic_dpi,
        )
    elif not args.skip_composite and not periodic_mesh_path.is_file():
        raise FileNotFoundError(periodic_mesh_path)

    if not args.skip_composite:
        compose_picture_in_picture(
            poincare_path,
            local_neighborhood_video=mesh_path,
            periodic_overview_video=periodic_mesh_path,
            output_path=composite_path,
            ffmpeg_executable=ffmpeg,
        )
        if not args.keep_intermediates:
            mesh_path.unlink()
            periodic_mesh_path.unlink()

    print(poincare_path, flush=True)
    if mesh_path.is_file():
        print(mesh_path, flush=True)
    if periodic_mesh_path.is_file():
        print(periodic_mesh_path, flush=True)
    print(composite_path, flush=True)


if __name__ == "__main__":
    main()
