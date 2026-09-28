"""Manim orbit of the four square-symmetry EPNs in G-metric space.

This scene intentionally uses Manim's documented ``Surface`` primitive,
rather than ``Polygon``.  Each geometric boundary is emitted exactly once:
neighbouring EPNs share two physical walls, so drawing a copy for each EPN
creates coincident triangles and OpenGL depth fighting.

Render from the repository root with::

    PATH=$PWD/Plotting/manim-bin:$PATH MPLCONFIGDIR=.matplotlib-cache \
    PYTHONPATH=. ./.venv/bin/manim --renderer=opengl -qm \
    --media_dir Plots/manim Plotting/fundamentalDomain3DManim.py FourSquareEPNs
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from manim import Arrow3D, BLACK, DEGREES, GREY, PI, UP, WHITE, Line3D, MathTex, Surface, ThreeDAxes, ThreeDScene, config

from Plotting.fundamentalDomain3D import (
    DEFAULT_C11_BOUNDS,
    DEFAULT_C22_MAX,
    SQUARE_EPN_COLORS,
    _FACES,
    square_epn_vertices,
)


config.background_color = WHITE
PHYSICAL_SHEET_OPACITY = 0.28
CLIPPING_CAP_OPACITY = 0.045


def _canonical_face(vertices: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    """Key a face independently of its winding and its owning EPN."""
    return tuple(sorted(tuple(point) for point in np.round(vertices, decimals=12)))


def _canonical_edge(start: np.ndarray, end: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    """Key an edge independently of orientation."""
    return tuple(sorted((tuple(np.round(start, 12)), tuple(np.round(end, 12)))))


def _unique_boundary_faces() -> Iterator[tuple[np.ndarray, str, bool, np.ndarray]]:
    """Yield once each nondegenerate boundary, with a consistent outward normal."""
    seen: set[tuple[tuple[float, float, float], ...]] = set()
    for (_, vertices), color in zip(
        square_epn_vertices(DEFAULT_C11_BOUNDS, DEFAULT_C22_MAX),
        SQUARE_EPN_COLORS,
        strict=True,
    ):
        region_centre = vertices.mean(axis=0)
        for face in _FACES:
            raw_corners = vertices[list(face)]
            key = _canonical_face(raw_corners)
            if key not in seen:
                seen.add(key)
                corners = np.array(
                    [
                        point
                        for index, point in enumerate(raw_corners)
                        if index == 0 or not np.allclose(point, raw_corners[index - 1])
                    ]
                )
                if len(corners) > 1 and np.allclose(corners[0], corners[-1]):
                    corners = corners[:-1]
                if len(corners) < 3:
                    continue
                normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
                if np.linalg.norm(normal) == 0.0:
                    continue
                if np.dot(normal, corners.mean(axis=0) - region_centre) < 0.0:
                    corners = corners[::-1]
                yield corners, color, face in _FACES[:3], region_centre


def _unique_boundary_edges() -> Iterator[tuple[np.ndarray, np.ndarray, bool]]:
    """Yield the unique outlines after shared EPN walls have been merged."""
    edges: dict[tuple[tuple[float, float, float], ...], tuple[np.ndarray, np.ndarray, bool]] = {}
    for corners, _, physical, _ in _unique_boundary_faces():
        for start, end in zip(corners, np.roll(corners, -1, axis=0), strict=True):
            key = _canonical_edge(start, end)
            previous = edges.get(key)
            edges[key] = (start, end, physical if previous is None else physical or previous[2])
    yield from edges.values()


def _surface_from_polygon(axes: ThreeDAxes, corners: np.ndarray, color: str, physical: bool) -> Surface:
    """Make an outward-oriented triangular or quadrilateral Manim surface."""
    if len(corners) not in (3, 4):
        raise ValueError("A displayed EPN face must be a triangle or quadrilateral.")

    if len(corners) == 3:
        first, second, third = corners

        def surface_point(u: float, v: float) -> np.ndarray:
            return (1.0 - u) * first + u * ((1.0 - v) * second + v * third)

    else:
        first, second, third, fourth = corners

        def surface_point(u: float, v: float) -> np.ndarray:
            return (
                (1.0 - u) * (1.0 - v) * first
                + u * (1.0 - v) * second
                + u * v * third
                + (1.0 - u) * v * fourth
            )

    return Surface(
        lambda u, v: axes.c2p(*surface_point(u, v)),
        u_range=(0.0, 1.0),
        v_range=(0.0, 1.0),
        resolution=(2, 2),
        fill_color=color if physical else GREY,
        fill_opacity=PHYSICAL_SHEET_OPACITY if physical else CLIPPING_CAP_OPACITY,
        checkerboard_colors=False,
        # Alpha-blended OpenGL surfaces must not write to the depth buffer:
        # otherwise a nearer translucent wall incorrectly hides the remaining
        # EPN walls.  There are no coincident faces in this scene, so this does
        # not reintroduce depth fighting as the camera moves.
        depth_test=False,
    )


class FourSquareEPNs(ThreeDScene):
    """One complete camera orbit around four translucent G-metric EPN wedges."""

    def construct(self) -> None:
        axes = ThreeDAxes(
            x_range=(0.0, DEFAULT_C22_MAX, 1.0),
            y_range=(0.0, DEFAULT_C22_MAX, 1.0),
            z_range=(-DEFAULT_C11_BOUNDS[1] / 2.0, DEFAULT_C11_BOUNDS[1] / 2.0, 1.0),
            x_length=7.2,
            y_length=7.2,
            z_length=4.8,
            axis_config={"color": BLACK, "stroke_width": 2.0},
        )
        coordinate_key = MathTex(
            r"\displaystyle (x,y,z)=(G_{11},G_{22},G_{12})",
            color=BLACK,
        ).scale(0.82).to_edge(UP)
        continuation_key = MathTex(
            r"\text{Coloured rays continue beyond the pale clipping caps}",
            color=BLACK,
        ).scale(0.48).next_to(coordinate_key, direction=-UP, buff=0.08)
        self.add(axes)
        self.add_fixed_in_frame_mobjects(coordinate_key, continuation_key)

        self.add(
            *(
                _surface_from_polygon(axes, corners, color, physical)
                for corners, color, physical, _ in _unique_boundary_faces()
            )
        )

        # True EPN boundaries remain dark; finite viewing-window caps are only
        # faint guides.  This avoids representing the displayed truncation as
        # the end of a region.
        self.add(
            *(
                Line3D(
                    axes.c2p(*start),
                    axes.c2p(*end),
                    color=BLACK if physical else GREY,
                    thickness=0.018 if physical else 0.006,
                    depth_test=False,
                )
                for start, end, physical in _unique_boundary_edges()
            )
        )

        # One interior ray per EPN makes the cone's unbounded scaling direction
        # visible without inventing a closed terminal face.
        directions = (
            (3.0, 4.6, 0.8),
            (3.0, 4.6, -0.8),
            (4.6, 3.0, 0.8),
            (4.6, 3.0, -0.8),
        )
        self.add(
            *(
                Arrow3D(
                    axes.c2p(*(0.05 * np.asarray(direction))),
                    axes.c2p(*(1.42 * np.asarray(direction))),
                    color=color,
                    thickness=0.012,
                    height=0.15,
                    base_radius=0.045,
                    resolution=12,
                    depth_test=False,
                )
                for direction, color in zip(directions, SQUARE_EPN_COLORS, strict=True)
            )
        )

        self.set_camera_orientation(phi=67 * DEGREES, theta=-52 * DEGREES)
        self.begin_ambient_camera_rotation(rate=2 * PI / 6.0)
        self.wait(6)
        self.stop_ambient_camera_rotation()
