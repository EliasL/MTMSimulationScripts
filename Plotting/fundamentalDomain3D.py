"""Draw the square-symmetry fundamental EPN as a volume in G-metric space.

The Poincare disk is the ``det(C) = 1`` slice of this space.  The full
fundamental region instead consists of all symmetric metrics satisfying
``0 <= G12 <= G11 / 2`` and ``G11 <= G22``.  It is unbounded, so this figure
shows its intersection with a finite, explicitly labelled viewing window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


DEFAULT_C11_BOUNDS = (0.0, 4.0)
DEFAULT_C22_MAX = 6.0
DEFAULT_VIEW_PADDING = 0.0
SQUARE_EPN_COLORS = ("#4C78A8", "#E45756", "#54A24B", "#B279A2")
SQUARE_EPN_ALPHA = 0.60


def fundamental_domain_vertices(
    c11_bounds: tuple[float, float] = DEFAULT_C11_BOUNDS,
    c22_max: float = DEFAULT_C22_MAX,
) -> np.ndarray:
    """Return vertices of the displayed truncated fundamental-domain wedge.

    Vertex coordinates use the plotting order ``(C11, C22, C12)``.  The
    ``G11 = 0`` is included as the rank-deficient limiting face, so the plot
    reaches the origin.  The physical EPN itself has ``G11 > 0``.  The
    truncation is solely for visualization; the physical fundamental domain
    remains unbounded towards large ``G22`` and under uniform scaling.
    """
    c11_min, c11_max = map(float, c11_bounds)
    if not (0.0 <= c11_min < c11_max <= c22_max):
        raise ValueError("Require 0 <= G11_min < G11_max <= G22_max.")

    return np.array(
        [
            (c11_min, c11_min, 0.0),
            (c11_max, c11_max, 0.0),
            (c11_max, c22_max, 0.0),
            (c11_min, c22_max, 0.0),
            (c11_min, c11_min, c11_min / 2.0),
            (c11_max, c11_max, c11_max / 2.0),
            (c11_max, c22_max, c11_max / 2.0),
            (c11_min, c22_max, c11_min / 2.0),
        ],
        dtype=float,
    )


def square_epn_vertices(
    c11_bounds: tuple[float, float] = DEFAULT_C11_BOUNDS,
    c22_max: float = DEFAULT_C22_MAX,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Return the four square-symmetry EPN wedges in plotting coordinates.

    The four wedges are the sign and coordinate-swap images of the fundamental
    domain.  Together they form the elastic square-symmetry neighbourhood.
    """
    base_vertices = fundamental_domain_vertices(c11_bounds, c22_max)
    base_metrics = np.zeros((len(base_vertices), 2, 2), dtype=float)
    base_metrics[:, 0, 0] = base_vertices[:, 0]
    base_metrics[:, 1, 1] = base_vertices[:, 1]
    base_metrics[:, 0, 1] = base_metrics[:, 1, 0] = base_vertices[:, 2]

    sign_flip = np.array([[1.0, 0.0], [0.0, -1.0]])
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    transforms = (
        ("$G_{12} \\geq 0,\\ G_{11} \\leq G_{22}$", np.eye(2)),
        ("$G_{12} \\leq 0,\\ G_{11} \\leq G_{22}$", sign_flip),
        ("$G_{12} \\geq 0,\\ G_{22} \\leq G_{11}$", swap),
        ("$G_{12} \\leq 0,\\ G_{22} \\leq G_{11}$", sign_flip @ swap),
    )
    regions = []
    for label, transform in transforms:
        metric = transform.T @ base_metrics @ transform
        regions.append(
            (
                label,
                np.stack((metric[:, 0, 0], metric[:, 1, 1], metric[:, 0, 1]), axis=1),
            )
        )
    return tuple(regions)


_FACES = (
    (0, 1, 2, 3),  # G12 = 0
    (4, 7, 6, 5),  # G12 = G11 / 2
    (0, 4, 5, 1),  # G22 = G11
    (3, 2, 6, 7),  # artificial G22 display cap
    (0, 3, 7, 4),  # artificial G11 lower cap
    (1, 5, 6, 2),  # artificial G11 upper cap
)


def _add_volume(axis: plt.Axes, vertices: np.ndarray, color: str, alpha: float) -> None:
    """Add one translucent clipped wedge and its physical boundary edges."""
    axis.add_collection3d(
        Poly3DCollection(
            [vertices[list(face)] for face in _FACES],
            facecolors=color,
            edgecolors="#263238",
            linewidths=0.85,
            alpha=alpha,
        )
    )
    for face in _FACES[:3]:
        closed = np.r_[vertices[list(face)], vertices[face[0]][None, :]]
        axis.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="#263238", linewidth=1.35)


def draw_fundamental_domain_3d(
    *,
    c11_bounds: tuple[float, float] = DEFAULT_C11_BOUNDS,
    c22_max: float = DEFAULT_C22_MAX,
    elevation: float = 23.0,
    azimuth: float = -56.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw the finite viewing-window intersection of the 3D EPN."""
    vertices = fundamental_domain_vertices(c11_bounds, c22_max)
    figure = plt.figure(figsize=(8.0, 6.6), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    _add_volume(axis, vertices, SQUARE_EPN_COLORS[0], alpha=0.28)

    c11_min, c11_max = c11_bounds
    axis.set_xlim(c11_min, c11_max)
    axis.set_ylim(c11_min, c22_max)
    axis.set_zlim(0.0, c11_max / 2.0)
    axis.set_box_aspect((c11_max - c11_min, c22_max - c11_min, c11_max / 2.0))
    axis.set_xlabel(r"$G_{11}$", labelpad=9)
    axis.set_ylabel(r"$G_{22}$", labelpad=9)
    axis.set_zlabel(r"$G_{12}$", labelpad=9)
    axis.set_title("Square-symmetry fundamental EPN in G-metric space", pad=18)
    axis.text2D(
        0.02,
        0.96,
        r"$0\leq G_{12}\leq G_{11}/2,\quad G_{11}\leq G_{22}$",
        transform=axis.transAxes,
        va="top",
    )
    axis.text2D(
        0.02,
        0.02,
        r"Grey faces are finite viewing-window caps; the EPN is unbounded.",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
    )
    axis.view_init(elev=elevation, azim=azimuth)
    return figure, axis


def draw_square_epns_3d(
    *,
    c11_bounds: tuple[float, float] = DEFAULT_C11_BOUNDS,
    c22_max: float = DEFAULT_C22_MAX,
    elevation: float = 23.0,
    azimuth: float = -56.0,
    view_padding: float = DEFAULT_VIEW_PADDING,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw the four coloured square-symmetry EPN wedges."""
    figure = plt.figure(figsize=(8.0, 6.6), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    regions = square_epn_vertices(c11_bounds, c22_max)
    for (_, vertices), color in zip(regions, SQUARE_EPN_COLORS, strict=True):
        _add_volume(axis, vertices, color, alpha=SQUARE_EPN_ALPHA)

    if view_padding < 0.0:
        raise ValueError("view_padding must be non-negative.")
    _, c11_max = c11_bounds
    axis_limit = c22_max * (1.0 + view_padding)
    c12_limit = c11_max * (1.0 + view_padding) / 2.0
    axis.set_xlim(0.0, axis_limit)
    axis.set_ylim(0.0, axis_limit)
    axis.set_zlim(-c12_limit, c12_limit)
    axis.set_box_aspect((axis_limit, axis_limit, 2.0 * c12_limit))
    axis.set_xlabel(r"$G_{11}$", labelpad=9)
    axis.set_ylabel(r"$G_{22}$", labelpad=9)
    axis.set_zlabel(r"$G_{12}$", labelpad=9)
    axis.set_title("Four square-symmetry EPNs in G-metric space", pad=18)
    axis.text2D(
        0.02,
        0.96,
        r"Four sign/swap images of $0\leq G_{12}\leq G_{11}/2,\ G_{11}\leq G_{22}$",
        transform=axis.transAxes,
        va="top",
    )
    axis.text2D(
        0.02,
        0.02,
        r"Finite caps only set the viewing window; each EPN is unbounded.",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
    )
    axis.view_init(elev=elevation, azim=azimuth)
    return figure, axis


def save_figure(output: str | Path, **kwargs) -> Path:
    """Render the 3D domain and save it without opening an interactive window."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = draw_fundamental_domain_3d(**kwargs)
    figure.savefig(output_path, dpi=240)
    plt.close(figure)
    return output_path


def save_square_epns_figure(output: str | Path, **kwargs) -> Path:
    """Render the four EPN wedges as a still figure."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = draw_square_epns_3d(**kwargs)
    figure.savefig(output_path, dpi=240)
    plt.close(figure)
    return output_path


def save_square_epns_animation(
    output: str | Path,
    *,
    duration: float = 6.0,
    fps: int = 12,
    elevation: float = 23.0,
    initial_azimuth: float = -56.0,
    **kwargs,
) -> Path:
    """Save one complete, constant-elevation camera orbit of the four EPNs."""
    if duration <= 0.0 or fps <= 0:
        raise ValueError("duration and fps must be positive.")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = draw_square_epns_3d(
        elevation=elevation,
        azimuth=initial_azimuth,
        **kwargs,
    )
    frame_count = round(duration * fps)

    def update(frame: int):
        axis.view_init(elev=elevation, azim=initial_azimuth + 360.0 * frame / frame_count)
        return (axis,)

    movie = animation.FuncAnimation(figure, update, frames=frame_count, blit=False)
    import imageio_ffmpeg

    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=("-pix_fmt", "yuv420p"),
    )
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    movie.save(output_path, writer=writer, dpi=160)
    plt.close(figure)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Plots/fundamental_domain_3d.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--c11-max", type=float, default=DEFAULT_C11_BOUNDS[1])
    parser.add_argument("--c22-max", type=float, default=DEFAULT_C22_MAX)
    parser.add_argument("--four-regions", action="store_true", help="Draw all four EPN wedges.")
    parser.add_argument("--animation", action="store_true", help="Save a rotating MP4 instead of a PNG.")
    parser.add_argument("--duration", type=float, default=6.0, help="Animation duration in seconds.")
    parser.add_argument("--fps", type=int, default=12, help="Animation frames per second.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    parameters = dict(
        c11_bounds=(DEFAULT_C11_BOUNDS[0], arguments.c11_max),
        c22_max=arguments.c22_max,
    )
    if arguments.animation:
        if not arguments.four_regions:
            raise ValueError("--animation currently requires --four-regions.")
        saved = save_square_epns_animation(
            arguments.output,
            duration=arguments.duration,
            fps=arguments.fps,
            **parameters,
        )
    elif arguments.four_regions:
        saved = save_square_epns_figure(arguments.output, **parameters)
    else:
        saved = save_figure(arguments.output, **parameters)
    print(f"Saved {saved}")
