#!/usr/bin/env python3
"""Render a coarse protocol animation for the rank-006 tracked element.

The full minimization replay is described in the accompanying notes.  This
wrapper exercises the existing single-element animator on the five saved
reversibility states while a logged minimization VTU sequence is unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Plotting.element_tracking_animation import (
    build_gamma_timeline,
    load_element_matrix_histories,
    load_mesh_neighborhoods,
    render_mesh_animation,
    render_poincare_animation,
)


DEFAULT_EVENT = Path(
    "/Volumes/data/MTS2D_output/"
    "reversibilityProtocolTest,s200x200l1.0,1e-05,5.1PBCedgeFlipt2epsR0.0"
    "LBFGSEpsg0.0LBFGSEpsx1e-06s0/data/reversibilityData/irrev_drop_l_1.43444"
)


def protocol_paths(event_directory: Path) -> tuple[Path, ...]:
    patterns = (
        "state0_min_gamma.*.vtu",
        "state1_affine_gamma_plus.*.vtu",
        "state2_relaxed_gamma_plus.*.vtu",
        "state3_affine_gamma_minus.*.vtu",
        "state4_relaxed_gamma.*.vtu",
    )
    paths = []
    for pattern in patterns:
        matches = sorted(event_directory.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one VTU matching {pattern} in {event_directory}, "
                f"found {len(matches)}."
            )
        paths.append(matches[0])
    return tuple(paths)


def render_protocol_animation(
    event_directory: Path,
    *,
    element_index: int,
    output_directory: Path,
    ffmpeg_executable: Path,
) -> tuple[Path, Path]:
    paths = protocol_paths(event_directory)
    history = load_element_matrix_histories(paths, element_index, ("T",))[0]
    timeline = build_gamma_timeline(
        history,
        np.arange(len(paths), dtype=float),
        history,
        gamma_per_frame=0.05,
        camera_smoothing_gamma=0.0,
    )
    snapshots = load_mesh_neighborhoods(paths, element_index, progress_every=0)
    output_directory.mkdir(parents=True, exist_ok=True)
    poincare = render_poincare_animation(
        history,
        timeline,
        output_directory / "rank006_slot6198_protocol_poincare.mp4",
        ffmpeg_executable=ffmpeg_executable,
        fps=24,
        dpi=120,
    )
    mesh = render_mesh_animation(
        history,
        timeline,
        snapshots,
        output_directory / "rank006_slot6198_protocol_local_mesh.mov",
        ffmpeg_executable=ffmpeg_executable,
        fps=24,
        dpi=120,
    )
    return poincare, mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-directory", type=Path, default=DEFAULT_EVENT)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("Plots/reconnecting_largest_energy_events_preview"),
    )
    parser.add_argument("--element-index", type=int, default=6198)
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path(imageio_ffmpeg.get_ffmpeg_exe()),
    )
    args = parser.parse_args()
    if not args.ffmpeg.is_file():
        raise FileNotFoundError("An ffmpeg executable is required.")
    for path in render_protocol_animation(
        args.event_directory,
        element_index=args.element_index,
        output_directory=args.output_directory,
        ffmpeg_executable=args.ffmpeg,
    ):
        print(path)


if __name__ == "__main__":
    main()
