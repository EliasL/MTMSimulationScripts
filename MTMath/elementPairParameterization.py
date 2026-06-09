"""Two-element edge-flip parameterization plots.

The T23 parameterization follows Sylvain's notation:

    x2 = (0, 0), x3 = (L, 0),
    x1 = (s L, -1 / L), x4 = (t L, 1 / L).

For this first plotting pass we use the symmetric closure v = s - t = 0, so
s = t = (1 + u) / 2.

By default main() runs two flip modes on the same physical T23-parameterized
configuration. In firstFlip mode, T23 is current and has det(F)=1. In
secondFlip mode, T14 is treated as current after the first flip; its elements
do not generally have det(F)=1, and that is intentional.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CACHE_ROOT = Path(__file__).resolve().parents[1] / ".cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm, Normalize, PowerNorm, to_rgba
from matplotlib.patches import Patch, Polygon

from MTMath.energyFunction import ContiEnergy

T23_TRIANGLES = ((1, 2, 3), (2, 3, 4))
T14_TRIANGLES = ((1, 2, 4), (1, 4, 3))
DIAGONAL_23 = (2, 3)
DIAGONAL_14 = (1, 4)
REFERENCE_L = float(np.sqrt(2.0))
REFERENCE_U = 0.0
ROOT_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT_DIR / "Plots"
HEATMAP_COMBINED = "combined"
HEATMAP_ENERGY_ONLY = "energy_only"
HEATMAP_REGIONS_ONLY = "regions_only"
HEATMAP_CONTENTS = (HEATMAP_COMBINED, HEATMAP_ENERGY_ONLY, HEATMAP_REGIONS_ONLY)

# =============================================================================
# User-editable plotting defaults
# =============================================================================
# Edit these values directly when running the script from an IDE or with:
#     python MTMath/elementPairParameterization.py


@dataclass(frozen=True)
class MaterialConfig:
    beta: float = -0.25
    K: float = 4.0
    noise: float = 1.0


@dataclass(frozen=True)
class FlipMode:
    name: str
    current_triangles: tuple[tuple[int, int, int], tuple[int, int, int]]
    flipped_triangles: tuple[tuple[int, int, int], tuple[int, int, int]]
    current_diagonal: tuple[int, int]
    flipped_diagonal: tuple[int, int]


FIRST_FLIP_T23_TO_T14 = FlipMode(
    name="firstFlip_T23_to_T14",
    current_triangles=T23_TRIANGLES,
    flipped_triangles=T14_TRIANGLES,
    current_diagonal=DIAGONAL_23,
    flipped_diagonal=DIAGONAL_14,
)
SECOND_FLIP_T14_TO_T23 = FlipMode(
    name="secondFlip_T14_to_T23",
    current_triangles=T14_TRIANGLES,
    flipped_triangles=T23_TRIANGLES,
    current_diagonal=DIAGONAL_14,
    flipped_diagonal=DIAGONAL_23,
)
DEFAULT_FLIP_MODES = (FIRST_FLIP_T23_TO_T14, SECOND_FLIP_T14_TO_T23)


@dataclass(frozen=True)
class PairGridConfig:
    L_values: tuple[float, ...]
    u_values: tuple[float, ...]
    output_path: Path
    flipped_output_path: Path


@dataclass(frozen=True)
class ReconnectionContourConfig:
    draw_current: bool = True
    draw_flipped: bool = False
    draw_failure_reasons: bool = True
    show_empty_failure_reasons: bool = True
    debug_only: bool = False
    level: float = 0.5
    current_color: str = "cyan"
    flipped_color: str = "lime"
    g12_negative_color: str = "tab:red"
    g12_too_large_color: str = "tab:purple"
    shared_edge_not_longest_color: str = "deepskyblue"
    linewidth: float = 1.4
    fill_alpha: float = 0.12
    failure_fill_alpha: float = 0.16


@dataclass(frozen=True)
class HeatmapElementPairGridConfig:
    draw: bool = True
    size: int = 3
    padding_fraction: float = 0.10
    scale_fraction: float = 0.085
    alpha: float = 0.55
    linewidth: float = 0.8


@dataclass(frozen=True)
class HeatmapConfig:
    output_path: Path
    resolution: int
    L_range: tuple[float, float]
    u_range: tuple[float, float]
    cmap: str = "coolwarm"
    color_scale: str = "linear"  # "linear" or "power"
    power_gamma: float = 0.5
    reconnection_contours: ReconnectionContourConfig = ReconnectionContourConfig()
    element_pair_grid: HeatmapElementPairGridConfig = HeatmapElementPairGridConfig()
    contents: tuple[str, ...] = HEATMAP_CONTENTS


@dataclass(frozen=True)
class MatrixFieldPlotConfig:
    output_path: Path
    resolution: int
    L_range: tuple[float, float]
    u_range: tuple[float, float]
    title: str
    colorbar_label: str
    component_symbol: str = ""
    cmap: str = "coolwarm"
    color_scale: str = "linear"  # "linear" or "power"
    power_gamma: float = 0.5
    centered_colorbar: bool = True
    element_pair_grid: HeatmapElementPairGridConfig = HeatmapElementPairGridConfig()


@dataclass(frozen=True)
class PlotConfig:
    flip_mode: FlipMode
    material: MaterialConfig
    pair_grid: PairGridConfig
    heatmap: HeatmapConfig
    focused_heatmap: HeatmapConfig
    cauchy_stress_difference: MatrixFieldPlotConfig
    first_element_G: MatrixFieldPlotConfig
    flip_modes: tuple[FlipMode, ...] = DEFAULT_FLIP_MODES
    plot_first_element_G: bool = False
    show: bool = False


CONFIG = PlotConfig(
    flip_mode=FIRST_FLIP_T23_TO_T14,
    material=MaterialConfig(),
    pair_grid=PairGridConfig(
        L_values=(1, REFERENCE_L, REFERENCE_L * 2),
        u_values=(0.8, 0.0, -0.8),
        output_path=PLOTS_DIR / "two_element_parameterization_grid.png",
        flipped_output_path=PLOTS_DIR / "two_element_parameterization_flipped_grid.png",
    ),
    heatmap=HeatmapConfig(
        output_path=PLOTS_DIR / "two_element_parameterization_flip_energy_heatmap.png",
        resolution=100,
        L_range=(0.75, REFERENCE_L * 2),
        u_range=(-0.9, 0.9),
    ),
    focused_heatmap=HeatmapConfig(
        output_path=PLOTS_DIR / "two_element_parameterization_flip_energy_heatmap_focused.png",
        resolution=100,
        L_range=(REFERENCE_L, 2),
        u_range=(-0.3, 0.3),
    ),
    cauchy_stress_difference=MatrixFieldPlotConfig(
        output_path=PLOTS_DIR
        / "two_element_parameterization_cauchy_stress_difference.png",
        resolution=100,
        L_range=(REFERENCE_L, 2),
        u_range=(-0.3, 0.3),
        title=r"$\sigma_{flipped} - \sigma_{current}$",
        colorbar_label=r"$\Delta\sigma$",
        component_symbol=r"\sigma",
    ),
    first_element_G=MatrixFieldPlotConfig(
        output_path=PLOTS_DIR / "two_element_parameterization_first_element_G.png",
        resolution=100,
        L_range=(0.75, REFERENCE_L * 3),
        u_range=(-0.9, 0.9),
        title="First current element G",
        colorbar_label="G",
        component_symbol="G",
        cmap="viridis",
        centered_colorbar=False,
    ),
    flip_modes=DEFAULT_FLIP_MODES,
    plot_first_element_G=False,
    show=False,
)


def validate_flip_mode(flip_mode: FlipMode) -> None:
    if flip_mode not in DEFAULT_FLIP_MODES:
        raise ValueError(f"Unsupported flip mode: {flip_mode.name}")


def t23_parameterized_vertices(
    L: float,
    u: float,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> dict[int, np.ndarray]:
    """Return the v=0 vertices from Sylvain's T23-controlled parameterization."""
    if L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")
    if abs(u) >= 1.0:
        raise ValueError(
            f"u must be in (-1, 1) so the flipped triangles stay positive, got {u}."
        )
    validate_flip_mode(flip_mode)

    middle_x = 0.5 * (1.0 + u) * L
    return {
        1: np.array([middle_x, -1.0 / L]),
        2: np.array([0.0, 0.0]),
        3: np.array([L, 0.0]),
        4: np.array([middle_x, 1.0 / L]),
    }


def square_reference_vertices(
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> dict[int, np.ndarray]:
    return t23_parameterized_vertices(REFERENCE_L, REFERENCE_U, flip_mode=flip_mode)


def vertices_to_array(vertices: dict[int, np.ndarray]) -> np.ndarray:
    return np.array([vertices[index] for index in (1, 2, 3, 4)], dtype=float)



def getG(
    vertices: dict[int, np.ndarray],
    triangles: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    """Return one Gram matrix per triangle using its two shortest edge vectors.

    For each triangle, the two shortest edges are oriented away from their shared
    vertex before computing G_ij = v_i dot v_j. The result has shape
    (len(triangles), 2, 2).
    """
    G_values = []
    for triangle in triangles:
        a, b, c = triangle
        edges = [
            (a, b, vertices[b] - vertices[a]),
            (a, c, vertices[c] - vertices[a]),
            (b, c, vertices[c] - vertices[b]),
        ]
        shortest_edges = sorted(edges, key=lambda edge: float(np.dot(edge[2], edge[2])))[:2]

        first_start, first_end, _ = shortest_edges[0]
        second_start, second_end, _ = shortest_edges[1]
        shared_vertices = {first_start, first_end} & {second_start, second_end}
        if len(shared_vertices) != 1:
            raise RuntimeError(f"Expected the two shortest edges to share one vertex in {triangle}.")
        shared_vertex = shared_vertices.pop()

        vectors = []
        for start, end, _ in shortest_edges:
            other_vertex = end if start == shared_vertex else start
            vectors.append(vertices[other_vertex] - vertices[shared_vertex])

        vector_matrix = np.column_stack(vectors)
        G_values.append(vector_matrix.T @ vector_matrix)

    return np.array(G_values)


# Helper: check if a shared edge is a longest edge of a triangle
def shared_edge_is_longest(
    vertices: dict[int, np.ndarray],
    triangle: tuple[int, int, int],
    shared_edge: tuple[int, int],
) -> bool:
    """Return True if the specified shared edge is a longest edge of triangle."""
    shared_edge = tuple(sorted(shared_edge))
    a, b, c = triangle
    triangle_edges = [tuple(sorted(edge)) for edge in ((a, b), (a, c), (b, c))]
    if shared_edge not in triangle_edges:
        raise ValueError(f"Shared edge {shared_edge} is not part of triangle {triangle}.")

    edge_lengths = {}
    for edge in triangle_edges:
        vector = vertices[edge[1]] - vertices[edge[0]]
        edge_lengths[edge] = float(np.dot(vector, vector))

    return edge_lengths[shared_edge] >= max(edge_lengths.values())


def reconnection_condition_masks(
    L_values: tuple[float, ...] | list[float] | np.ndarray,
    u_values: tuple[float, ...] | list[float] | np.ndarray,
    triangles: tuple[tuple[int, int, int], ...] | None = None,
    shared_edge: tuple[int, int] | None = None,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> dict[str, np.ndarray]:
    """Return C++ inRegion and failure-reason masks on an (u, L) grid.

    The returned array has shape (len(u_values), len(L_values)), matching the
    heatmap orientation used by build_flip_energy_heatmap. The inside mask is
    True when every triangle satisfies 0 <= G_12 <= min(G_11, G_22), and when
    the specified shared edge is a longest edge of every triangle. G is computed
    from the two shortest edge vectors of each triangle.
    """
    L_values = _as_1d_float_array(L_values, "L_values")
    u_values = _as_1d_float_array(u_values, "u_values")
    if triangles is None:
        triangles = flip_mode.current_triangles
    if shared_edge is None:
        shared_edge = flip_mode.current_diagonal
    shape = (len(u_values), len(L_values))
    inside_mask = np.zeros(shape, dtype=bool)
    g12_negative_mask = np.zeros(shape, dtype=bool)
    g12_too_large_mask = np.zeros(shape, dtype=bool)
    shared_edge_not_longest_mask = np.zeros(shape, dtype=bool)

    for row, u in enumerate(u_values):
        for col, L in enumerate(L_values):
            vertices = t23_parameterized_vertices(float(L), float(u), flip_mode=flip_mode)
            G_values = getG(vertices, triangles)
            G11 = G_values[:, 0, 0]
            G12 = G_values[:, 0, 1]
            G22 = G_values[:, 1, 1]
            g12_negative_per_triangle = G12 < 0.0
            g12_too_large_per_triangle = G12 > np.minimum(G11, G22)
            shared_edge_longest_per_triangle = np.array(
                [
                    shared_edge_is_longest(vertices, triangle, shared_edge)
                    for triangle in triangles
                ],
                dtype=bool,
            )
            inside_per_triangle = (
                ~g12_negative_per_triangle
                & ~g12_too_large_per_triangle
                & shared_edge_longest_per_triangle
            )
            inside_mask[row, col] = bool(np.all(inside_per_triangle))
            g12_negative_mask[row, col] = bool(np.any(g12_negative_per_triangle))
            g12_too_large_mask[row, col] = bool(np.any(g12_too_large_per_triangle))
            shared_edge_not_longest_mask[row, col] = bool(
                np.any(~shared_edge_longest_per_triangle)
            )

    return {
        "inside": inside_mask,
        "g12_negative": g12_negative_mask,
        "g12_too_large": g12_too_large_mask,
        "shared_edge_not_longest": shared_edge_not_longest_mask,
    }


def insideReconnectionZone(
    L_values: tuple[float, ...] | list[float] | np.ndarray,
    u_values: tuple[float, ...] | list[float] | np.ndarray,
    triangles: tuple[tuple[int, int, int], ...] | None = None,
    shared_edge: tuple[int, int] | None = None,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    """Return a Boolean mask for the C++ inRegion condition on an (u, L) grid."""
    return reconnection_condition_masks(
        L_values,
        u_values,
        triangles=triangles,
        shared_edge=shared_edge,
        flip_mode=flip_mode,
    )["inside"]


def t23_parameterized_vertices_array(
    L: np.ndarray,
    u: np.ndarray,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    L = np.asarray(L, dtype=float)
    u = np.asarray(u, dtype=float)
    if np.any(L <= 0.0):
        raise ValueError("All L values must be positive.")
    if np.any(np.abs(u) >= 1.0):
        raise ValueError("All u values must be in (-1, 1).")
    if L.shape != u.shape:
        raise ValueError(f"L and u must have the same shape, got {L.shape} and {u.shape}.")
    validate_flip_mode(flip_mode)

    vertices = np.empty(L.shape + (4, 2), dtype=float)
    middle_x = 0.5 * (1.0 + u) * L
    vertices[..., 0, 0] = middle_x
    vertices[..., 0, 1] = -1.0 / L
    vertices[..., 1, :] = 0.0
    vertices[..., 2, 0] = L
    vertices[..., 2, 1] = 0.0
    vertices[..., 3, 0] = middle_x
    vertices[..., 3, 1] = 1.0 / L
    return vertices


def triangle_deformation_gradient(
    reference_vertices: dict[int, np.ndarray],
    current_vertices: dict[int, np.ndarray],
    triangle: tuple[int, int, int],
) -> np.ndarray:
    anchor, node_a, node_b = triangle
    Dm = np.column_stack(
        [
            reference_vertices[node_a] - reference_vertices[anchor],
            reference_vertices[node_b] - reference_vertices[anchor],
        ]
    )
    Ds = np.column_stack(
        [
            current_vertices[node_a] - current_vertices[anchor],
            current_vertices[node_b] - current_vertices[anchor],
        ]
    )
    return Ds @ np.linalg.inv(Dm)


def deformation_gradients_from_vertex_array(
    reference_vertices: np.ndarray,
    current_vertices: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> np.ndarray:
    if reference_vertices.shape != (4, 2):
        raise ValueError(
            f"reference_vertices must have shape (4, 2), got {reference_vertices.shape}."
        )
    if current_vertices.shape[-2:] != (4, 2):
        raise ValueError(
            "current_vertices must have shape (..., 4, 2), "
            f"got {current_vertices.shape}."
        )

    F_values = []
    for triangle in triangles:
        anchor, node_a, node_b = [index - 1 for index in triangle]
        Dm = np.column_stack(
            [
                reference_vertices[node_a] - reference_vertices[anchor],
                reference_vertices[node_b] - reference_vertices[anchor],
            ]
        )
        Ds = np.stack(
            [
                current_vertices[..., node_a, :] - current_vertices[..., anchor, :],
                current_vertices[..., node_b, :] - current_vertices[..., anchor, :],
            ],
            axis=-1,
        )
        F_values.append(Ds @ np.linalg.inv(Dm))
    return np.stack(F_values, axis=-3)


def triangle_reference_area(
    reference_vertices: dict[int, np.ndarray],
    triangle: tuple[int, int, int],
) -> float:
    anchor, node_a, node_b = triangle
    Dm = np.column_stack(
        [
            reference_vertices[node_a] - reference_vertices[anchor],
            reference_vertices[node_b] - reference_vertices[anchor],
        ]
    )
    return float(0.5 * abs(np.linalg.det(Dm)))


def verify_t23_triangles_are_area_preserving(
    current_vertices: dict[int, np.ndarray],
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> None:
    validate_flip_mode(flip_mode)
    reference_vertices = square_reference_vertices(flip_mode=flip_mode)
    determinants = [
        np.linalg.det(
            triangle_deformation_gradient(reference_vertices, current_vertices, triangle)
        )
        for triangle in T23_TRIANGLES
    ]
    if not np.allclose(determinants, 1.0, rtol=1e-12, atol=1e-12):
        raise RuntimeError(
            f"Expected T23 parameterization det(F)=1, got {determinants}."
        )


def _as_1d_float_array(values: tuple[float, ...] | list[float] | np.ndarray, name: str):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence, got shape {array.shape}.")
    return array


def pair_energy(
    current_vertices: dict[int, np.ndarray],
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    material: MaterialConfig = CONFIG.material,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> float:
    """Return reference-area weighted energy for a two-triangle topology."""
    reference_vertices = square_reference_vertices(flip_mode=flip_mode)
    F = np.array(
        [
            triangle_deformation_gradient(reference_vertices, current_vertices, triangle)
            for triangle in triangles
        ]
    )
    reference_areas = np.array(
        [triangle_reference_area(reference_vertices, triangle) for triangle in triangles]
    )
    energy_density = ContiEnergy.energy_from_F(
        F,
        beta=material.beta,
        K=material.K,
        noise=material.noise,
        zeroReference=True,
    )
    return float(np.sum(reference_areas * energy_density))


def current_pair_energy(
    current_vertices: dict[int, np.ndarray],
    material: MaterialConfig = CONFIG.material,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> float:
    return pair_energy(
        current_vertices,
        flip_mode.current_triangles,
        material=material,
        flip_mode=flip_mode,
    )


def flipped_pair_energy(
    current_vertices: dict[int, np.ndarray],
    material: MaterialConfig = CONFIG.material,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> float:
    return pair_energy(
        current_vertices,
        flip_mode.flipped_triangles,
        material=material,
        flip_mode=flip_mode,
    )


def pair_energy_grid(
    L_values: np.ndarray,
    u_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    material: MaterialConfig = CONFIG.material,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    L_grid, u_grid = np.meshgrid(L_values, u_values, indexing="xy")
    current_vertices = t23_parameterized_vertices_array(L_grid, u_grid, flip_mode=flip_mode)
    reference_vertices = square_reference_vertices(flip_mode=flip_mode)
    reference_array = vertices_to_array(reference_vertices)
    F = deformation_gradients_from_vertex_array(reference_array, current_vertices, triangles)
    reference_areas = np.array(
        [triangle_reference_area(reference_vertices, triangle) for triangle in triangles]
    )
    energy_density = ContiEnergy.energy_from_F(
        F,
        beta=material.beta,
        K=material.K,
        noise=material.noise,
        zeroReference=True,
    )
    return np.sum(reference_areas * energy_density, axis=-1)


def pair_deformation_gradient_grid(
    L_values: np.ndarray,
    u_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> tuple[np.ndarray, np.ndarray]:
    L_grid, u_grid = np.meshgrid(L_values, u_values, indexing="xy")
    current_vertices = t23_parameterized_vertices_array(L_grid, u_grid, flip_mode=flip_mode)
    reference_vertices = square_reference_vertices(flip_mode=flip_mode)
    reference_array = vertices_to_array(reference_vertices)
    F = deformation_gradients_from_vertex_array(reference_array, current_vertices, triangles)
    reference_areas = np.array(
        [triangle_reference_area(reference_vertices, triangle) for triangle in triangles]
    )
    return F, reference_areas


def pair_cauchy_stress_grid(
    L_values: np.ndarray,
    u_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    material: MaterialConfig = CONFIG.material,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    F, reference_areas = pair_deformation_gradient_grid(
        L_values,
        u_values,
        triangles,
        flip_mode=flip_mode,
    )
    sigma = ContiEnergy.cauchy_from_F(
        F,
        beta=material.beta,
        K=material.K,
        noise=material.noise,
    )
    J = np.linalg.det(F)
    if np.any(J <= 0.0):
        raise RuntimeError(
            f"Expected positive element Jacobians for Cauchy stress, got min J={J.min()}."
        )
    current_areas = reference_areas * J
    total_current_area = np.sum(current_areas, axis=-1)
    return (
        np.sum(current_areas[..., None, None] * sigma, axis=-3)
        / total_current_area[..., None, None]
    )


def cauchy_stress_difference_grid(
    L_values: np.ndarray,
    u_values: np.ndarray,
    material: MaterialConfig = CONFIG.material,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    current_stress = pair_cauchy_stress_grid(
        L_values,
        u_values,
        flip_mode.current_triangles,
        material=material,
        flip_mode=flip_mode,
    )
    flipped_stress = pair_cauchy_stress_grid(
        L_values,
        u_values,
        flip_mode.flipped_triangles,
        material=material,
        flip_mode=flip_mode,
    )
    return flipped_stress - current_stress


def first_element_G_grid(
    L_values: np.ndarray,
    u_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    values = np.empty((len(u_values), len(L_values), 2, 2), dtype=float)
    if triangles is None:
        triangles = flip_mode.current_triangles
    first_triangle = (triangles[0],)
    for row, u in enumerate(u_values):
        for col, L in enumerate(L_values):
            vertices = t23_parameterized_vertices(float(L), float(u), flip_mode=flip_mode)
            values[row, col] = getG(vertices, first_triangle)[0]
    return values


def edge_flip_energy_difference_grid(
    L_values: np.ndarray,
    u_values: np.ndarray,
    material: MaterialConfig = CONFIG.material,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_energy = pair_energy_grid(
        L_values,
        u_values,
        flip_mode.current_triangles,
        material=material,
        flip_mode=flip_mode,
    )
    flipped_energy = pair_energy_grid(
        L_values,
        u_values,
        flip_mode.flipped_triangles,
        material=material,
        flip_mode=flip_mode,
    )
    return flipped_energy - current_energy, current_energy, flipped_energy


def _all_vertices(
    L_values: np.ndarray,
    u_values: np.ndarray,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> list[dict[int, np.ndarray]]:
    vertex_sets = []
    for u in u_values:
        for L in L_values:
            current_vertices = t23_parameterized_vertices(
                float(L),
                float(u),
                flip_mode=flip_mode,
            )
            verify_t23_triangles_are_area_preserving(
                current_vertices,
                flip_mode=flip_mode,
            )
            vertex_sets.append(current_vertices)
    return vertex_sets


def _plot_edge(
    ax: plt.Axes,
    vertices: dict[int, np.ndarray],
    edge: tuple[int, int],
    **kwargs,
) -> None:
    points = np.array([vertices[edge[0]], vertices[edge[1]]])
    ax.plot(points[:, 0], points[:, 1], **kwargs)


def _triangle_edges(
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> set[tuple[int, int]]:
    edges = set()
    for a, b, c in triangles:
        edges |= {tuple(sorted(edge)) for edge in ((a, b), (b, c), (a, c))}
    return edges


def plot_element_pair(
    ax: plt.Axes,
    vertices: dict[int, np.ndarray],
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    active_diagonal: tuple[int, int] | None = None,
    inactive_diagonal: tuple[int, int] | None = None,
    reference_vertices: dict[int, np.ndarray] | None = None,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> None:
    if triangles is None:
        triangles = flip_mode.current_triangles
    if active_diagonal is None:
        active_diagonal = flip_mode.current_diagonal
    if reference_vertices is not None:
        for triangle in triangles:
            points = np.array([reference_vertices[index] for index in triangle])
            ax.add_patch(
                Polygon(
                    points,
                    closed=True,
                    facecolor="0.4",
                    edgecolor="0.2",
                    alpha=0.1,
                    linewidth=1.2,
                    zorder=0,
                )
            )

    colors = ("tab:blue", "tab:orange")
    for triangle, color in zip(triangles, colors):
        points = np.array([vertices[index] for index in triangle])
        ax.add_patch(
            Polygon(
                points,
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=0.16,
                linewidth=1.6,
                zorder=1,
            )
        )

    active_diagonal = tuple(sorted(active_diagonal))
    for edge in sorted(_triangle_edges(triangles) - {active_diagonal}):
        _plot_edge(ax, vertices, edge, color="0.25", linewidth=1.1)
    _plot_edge(ax, vertices, active_diagonal, color="black", linewidth=2.0)
    if inactive_diagonal is not None:
        _plot_edge(
            ax,
            vertices,
            inactive_diagonal,
            color="0.45",
            linewidth=1.4,
            linestyle="--",
        )

    points = np.array([vertices[index] for index in sorted(vertices)])
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=30,
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
        zorder=3,
    )

    label_offsets = {
        1: np.array([0.03, -0.08]),
        2: np.array([-0.13, -0.06]),
        3: np.array([0.03, -0.06]),
        4: np.array([0.03, 0.05]),
    }

    for index, point in vertices.items():
        ax.text(
            point[0] + label_offsets[index][0],
            point[1] + label_offsets[index][1],
            f"x{index}",
            fontsize=9,
            color="black",
        )


# === Helper: Draw a grid of element-pair glyphs over a heatmap axis ===
def plot_heatmap_element_pair_grid(
    ax: plt.Axes,
    L_range: tuple[float, float],
    u_range: tuple[float, float],
    config: HeatmapElementPairGridConfig = CONFIG.heatmap.element_pair_grid,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    active_diagonal: tuple[int, int] | None = None,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> None:
    """Draw a small grid of element-pair glyphs over a heatmap axis."""
    if config.size < 1:
        raise ValueError(f"grid_size must be at least 1, got {config.size}.")
    if not 0.0 <= config.padding_fraction < 0.5:
        raise ValueError(
            "padding_fraction must be in [0, 0.5), "
            f"got {config.padding_fraction}."
        )
    if triangles is None:
        triangles = flip_mode.current_triangles
    if active_diagonal is None:
        active_diagonal = flip_mode.current_diagonal

    L_min, L_max = L_range
    u_min, u_max = u_range
    L_span = L_max - L_min
    u_span = u_max - u_min
    L_centers = np.linspace(
        L_min + config.padding_fraction * L_span,
        L_max - config.padding_fraction * L_span,
        config.size,
    )
    u_centers = np.linspace(
        u_min + config.padding_fraction * u_span,
        u_max - config.padding_fraction * u_span,
        config.size,
    )
    target_size = config.scale_fraction * min(L_span, u_span)
    active_diagonal = tuple(sorted(active_diagonal))
    edges = sorted(_triangle_edges(triangles))
    sampled_geometries = []
    max_radius = 0.0

    for u_center in u_centers:
        for L_center in L_centers:
            vertices = t23_parameterized_vertices(
                float(L_center),
                float(u_center),
                flip_mode=flip_mode,
            )
            points = np.array(list(vertices.values()))
            centroid = points.mean(axis=0)
            radius = float(np.max(np.linalg.norm(points - centroid, axis=1)))
            sampled_geometries.append((L_center, u_center, vertices, centroid))
            max_radius = max(max_radius, radius)

    if max_radius <= 0.0:
        raise RuntimeError("Expected non-degenerate element pairs in heatmap overlay.")
    scale = target_size / max_radius

    for L_center, u_center, vertices, centroid in sampled_geometries:
        transformed_vertices = {
            index: np.array([L_center, u_center]) + scale * (point - centroid)
            for index, point in vertices.items()
        }

        for edge in edges:
            edge_points = np.array(
                [transformed_vertices[edge[0]], transformed_vertices[edge[1]]]
            )
            is_active_diagonal = tuple(sorted(edge)) == active_diagonal
            ax.plot(
                edge_points[:, 0],
                edge_points[:, 1],
                color="black",
                linewidth=config.linewidth * (1.7 if is_active_diagonal else 1.0),
                alpha=config.alpha,
                zorder=8,
            )


def _shared_limits(vertex_sets: list[dict[int, np.ndarray]]) -> tuple[float, float, float, float]:
    points = np.vstack([np.array(list(vertices.values())) for vertices in vertex_sets])
    xy_min = points.min(axis=0)
    xy_max = points.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1.0)
    margin = 0.18 * span.max()
    return (
        xy_min[0] - margin,
        xy_max[0] + margin,
        xy_min[1] - margin,
        xy_max[1] + margin,
    )


def build_symmetric_parameterization_grid(
    config: PairGridConfig = CONFIG.pair_grid,
    material: MaterialConfig = CONFIG.material,
    flip_mode: FlipMode = CONFIG.flip_mode,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    active_diagonal: tuple[int, int] | None = None,
    inactive_diagonal: tuple[int, int] | None = None,
    title: str = "Symmetric two-element parameterization (v=0)",
) -> plt.Figure:
    """Build a grid of two-triangle element pairs for the v=0 closure."""
    L_values = _as_1d_float_array(config.L_values, "L_values")
    u_values = _as_1d_float_array(config.u_values, "u_values")
    if triangles is None:
        triangles = flip_mode.current_triangles
    if active_diagonal is None:
        active_diagonal = flip_mode.current_diagonal
    vertex_sets = _all_vertices(L_values, u_values, flip_mode=flip_mode)
    reference_vertices = square_reference_vertices(flip_mode=flip_mode)
    limits = _shared_limits(vertex_sets + [reference_vertices])

    fig, axes = plt.subplots(
        len(u_values),
        len(L_values),
        figsize=(3.4 * len(L_values), 3.4 * len(u_values)),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for row, u in enumerate(u_values):
        for col, L in enumerate(L_values):
            ax = axes[row, col]
            vertices = vertex_sets[row * len(L_values) + col]
            energy = pair_energy(
                vertices,
                triangles,
                material=material,
                flip_mode=flip_mode,
            )
            plot_element_pair(
                ax,
                vertices,
                triangles=triangles,
                active_diagonal=active_diagonal,
                inactive_diagonal=inactive_diagonal,
                reference_vertices=reference_vertices,
                flip_mode=flip_mode,
            )
            L_label = r"$\sqrt{2}$" if np.isclose(L, REFERENCE_L) else f"{L:g}"
            ax.set_title(f"L={L_label}, u={u:g}\nE={energy:.3e}", fontsize=11)
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.22, linewidth=0.6)
            if row == len(u_values) - 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("y")

    fig.suptitle(f"{title} ({flip_mode.name})", fontsize=14)
    return fig


def build_flipped_parameterization_grid(
    config: PairGridConfig = CONFIG.pair_grid,
    material: MaterialConfig = CONFIG.material,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> plt.Figure:
    return build_symmetric_parameterization_grid(
        config=config,
        material=material,
        flip_mode=flip_mode,
        triangles=flip_mode.flipped_triangles,
        active_diagonal=flip_mode.flipped_diagonal,
        title="Symmetric two-element parameterization after flip (v=0)",
    )


def heatmap_color_norm(
    values: np.ndarray,
    config: HeatmapConfig = CONFIG.heatmap,
):
    if config.color_scale == "linear":
        return CenteredNorm(vcenter=0.0)
    if config.color_scale == "power":
        if config.power_gamma <= 0.0:
            raise ValueError(f"power_gamma must be positive, got {config.power_gamma}.")
        max_abs = float(np.nanmax(np.abs(values)))
        if max_abs <= 0.0:
            max_abs = 1.0
        return PowerNorm(gamma=config.power_gamma, vmin=-max_abs, vmax=max_abs)
    raise ValueError(
        f"Unsupported color_scale {config.color_scale!r}. Use 'linear' or 'power'."
    )


def validate_heatmap_content(content: str) -> None:
    if content not in HEATMAP_CONTENTS:
        raise ValueError(
            f"Unsupported heatmap content {content!r}. Use one of {HEATMAP_CONTENTS}."
        )


def has_region_boundary(mask: np.ndarray) -> bool:
    return bool(np.any(mask) and not np.all(mask))


def region_legend_patch(color: str, fill_alpha: float, label: str) -> Patch:
    return Patch(
        facecolor=to_rgba(color, fill_alpha),
        edgecolor=color,
        label=label,
    )


def sampled_parameter_values(
    resolution: int,
    L_range: tuple[float, float],
    u_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if resolution < 2:
        raise ValueError(f"resolution must be at least 2, got {resolution}.")
    if L_range[0] <= 0.0 or L_range[1] <= 0.0:
        raise ValueError(f"L_range values must be positive, got {L_range}.")
    if L_range[0] >= L_range[1]:
        raise ValueError(f"L_range must be increasing, got {L_range}.")
    if u_range[0] <= -1.0 or u_range[1] >= 1.0:
        raise ValueError(f"u_range must stay inside (-1, 1), got {u_range}.")
    if u_range[0] >= u_range[1]:
        raise ValueError(f"u_range must be increasing, got {u_range}.")
    return (
        np.linspace(L_range[0], L_range[1], resolution),
        np.linspace(u_range[0], u_range[1], resolution),
    )


def matrix_field_color_norm(
    values: np.ndarray,
    config: MatrixFieldPlotConfig,
):
    if config.color_scale == "linear":
        if config.centered_colorbar:
            return CenteredNorm(vcenter=0.0)
        return Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    if config.color_scale == "power":
        if config.power_gamma <= 0.0:
            raise ValueError(f"power_gamma must be positive, got {config.power_gamma}.")
        if config.centered_colorbar:
            max_abs = float(np.nanmax(np.abs(values)))
            if max_abs <= 0.0:
                max_abs = 1.0
            return PowerNorm(gamma=config.power_gamma, vmin=-max_abs, vmax=max_abs)
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        return PowerNorm(gamma=config.power_gamma, vmin=vmin, vmax=vmax)
    raise ValueError(
        f"Unsupported color_scale {config.color_scale!r}. Use 'linear' or 'power'."
    )


def build_matrix_field_heatmaps(
    matrix_values: np.ndarray,
    L_values: np.ndarray,
    u_values: np.ndarray,
    config: MatrixFieldPlotConfig,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> plt.Figure:
    if matrix_values.shape != (len(u_values), len(L_values), 2, 2):
        raise ValueError(
            "matrix_values must have shape "
            f"({len(u_values)}, {len(L_values)}, 2, 2), got {matrix_values.shape}."
        )
    norm = matrix_field_color_norm(matrix_values, config)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.0, 7.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            image = ax.imshow(
                matrix_values[..., i, j],
                origin="lower",
                extent=(L_values[0], L_values[-1], u_values[0], u_values[-1]),
                aspect="auto",
                cmap=config.cmap,
                norm=norm,
                interpolation="nearest",
            )
            ax.axvline(REFERENCE_L, color="0.2", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axhline(REFERENCE_U, color="0.2", linestyle=":", linewidth=1.0, alpha=0.7)
            if config.element_pair_grid.draw:
                plot_heatmap_element_pair_grid(
                    ax,
                    L_range=(L_values[0], L_values[-1]),
                    u_range=(u_values[0], u_values[-1]),
                    config=config.element_pair_grid,
                    triangles=flip_mode.current_triangles,
                    active_diagonal=flip_mode.current_diagonal,
                    flip_mode=flip_mode,
                )
            title = (
                rf"${config.component_symbol}_{{{i + 1},{j + 1}}}$"
                if config.component_symbol
                else f"[{i}, {j}]"
            )
            ax.set_title(title)
            if i == 1:
                ax.set_xlabel("L")
            if j == 0:
                ax.set_ylabel("u")

    if image is None:
        raise RuntimeError("Expected at least one matrix component to plot.")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist())
    colorbar.set_label(config.colorbar_label)
    fig.suptitle(
        f"{config.title} ({config.resolution}x{config.resolution}, "
        f"{config.color_scale})"
    )
    return fig


def build_cauchy_stress_difference_heatmaps(
    config: MatrixFieldPlotConfig = CONFIG.cauchy_stress_difference,
    material: MaterialConfig = CONFIG.material,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    L_values, u_values = sampled_parameter_values(
        config.resolution,
        config.L_range,
        config.u_range,
    )
    matrix_values = cauchy_stress_difference_grid(
        L_values,
        u_values,
        material=material,
        flip_mode=flip_mode,
    )
    fig = build_matrix_field_heatmaps(
        matrix_values,
        L_values,
        u_values,
        config,
        flip_mode=flip_mode,
    )
    return fig, {
        "L_values": L_values,
        "u_values": u_values,
        "matrix_values": matrix_values,
    }


def build_first_element_G_heatmaps(
    config: MatrixFieldPlotConfig = CONFIG.first_element_G,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    L_values, u_values = sampled_parameter_values(
        config.resolution,
        config.L_range,
        config.u_range,
    )
    matrix_values = first_element_G_grid(L_values, u_values, flip_mode=flip_mode)
    fig = build_matrix_field_heatmaps(
        matrix_values,
        L_values,
        u_values,
        config,
        flip_mode=flip_mode,
    )
    return fig, {
        "L_values": L_values,
        "u_values": u_values,
        "matrix_values": matrix_values,
    }


def build_flip_energy_heatmap(
    config: HeatmapConfig = CONFIG.heatmap,
    material: MaterialConfig = CONFIG.material,
    flip_mode: FlipMode = CONFIG.flip_mode,
    content: str = HEATMAP_COMBINED,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    validate_heatmap_content(content)
    L_values, u_values = sampled_parameter_values(
        config.resolution,
        config.L_range,
        config.u_range,
    )
    delta_energy, current_energy, flipped_energy = edge_flip_energy_difference_grid(
        L_values,
        u_values,
        material=material,
        flip_mode=flip_mode,
    )
    abs_delta_energy = np.abs(delta_energy)
    show_energy = content in (HEATMAP_COMBINED, HEATMAP_ENERGY_ONLY)
    show_regions = content in (HEATMAP_COMBINED, HEATMAP_REGIONS_ONLY)
    norm = heatmap_color_norm(delta_energy, config=config) if show_energy else None
    current_reconnection_masks = reconnection_condition_masks(
        L_values,
        u_values,
        triangles=flip_mode.current_triangles,
        shared_edge=flip_mode.current_diagonal,
        flip_mode=flip_mode,
    )
    current_reconnection_zone_mask = current_reconnection_masks["inside"]
    flipped_reconnection_zone_mask = insideReconnectionZone(
        L_values,
        u_values,
        triangles=flip_mode.flipped_triangles,
        shared_edge=flip_mode.flipped_diagonal,
        flip_mode=flip_mode,
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    contours = config.reconnection_contours
    overlay = config.element_pair_grid
    image = None

    if contours.debug_only:
        debug_mask = np.zeros_like(current_reconnection_zone_mask, dtype=float)
        debug_mask[current_reconnection_zone_mask] += 1.0
        debug_mask[flipped_reconnection_zone_mask] += 2.0
        image = ax.imshow(
            debug_mask,
            origin="lower",
            extent=(L_values[0], L_values[-1], u_values[0], u_values[-1]),
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=3.0,
            interpolation="nearest",
        )
    elif show_energy:
        image = ax.imshow(
            delta_energy,
            origin="lower",
            extent=(L_values[0], L_values[-1], u_values[0], u_values[-1]),
            aspect="auto",
            cmap=config.cmap,
            norm=norm,
            interpolation="nearest",
        )
    else:
        ax.set_xlim(L_values[0], L_values[-1])
        ax.set_ylim(u_values[0], u_values[-1])
        ax.set_facecolor("white")

    if show_regions and not contours.debug_only:
        if contours.draw_current and np.any(current_reconnection_zone_mask):
            ax.contourf(
                L_values,
                u_values,
                current_reconnection_zone_mask.astype(float),
                levels=[contours.level, 1.5],
                colors=[contours.current_color],
                alpha=contours.fill_alpha,
                zorder=6,
            )
        if contours.draw_failure_reasons:
            failure_reason_styles = (
                (
                    "g12_negative",
                    contours.g12_negative_color,
                    7,
                ),
                (
                    "g12_too_large",
                    contours.g12_too_large_color,
                    8,
                ),
                (
                    "shared_edge_not_longest",
                    contours.shared_edge_not_longest_color,
                    9,
                ),
            )
            for reason_key, color, zorder in failure_reason_styles:
                reason_mask = current_reconnection_masks[reason_key]
                if np.any(reason_mask):
                    ax.contourf(
                        L_values,
                        u_values,
                        reason_mask.astype(float),
                        levels=[contours.level, 1.5],
                        colors=[color],
                        alpha=contours.failure_fill_alpha,
                        zorder=zorder,
                    )
        if contours.draw_flipped and np.any(flipped_reconnection_zone_mask):
            ax.contourf(
                L_values,
                u_values,
                flipped_reconnection_zone_mask.astype(float),
                levels=[contours.level, 1.5],
                colors=[contours.flipped_color],
                alpha=contours.fill_alpha,
                zorder=7,
            )

    if overlay.draw:
        plot_heatmap_element_pair_grid(
            ax,
            L_range=(L_values[0], L_values[-1]),
            u_range=(u_values[0], u_values[-1]),
            config=overlay,
            triangles=flip_mode.current_triangles,
            active_diagonal=flip_mode.current_diagonal,
            flip_mode=flip_mode,
        )

    legend_handles = []

    has_current_reconnection_boundary = has_region_boundary(
        current_reconnection_zone_mask
    )
    if show_regions and contours.draw_current and has_current_reconnection_boundary:
        ax.contour(
            L_values,
            u_values,
            current_reconnection_zone_mask.astype(float),
            levels=[contours.level],
            colors=contours.current_color,
            linewidths=contours.linewidth,
            alpha=1.0,
            zorder=14,
        )
        legend_handles.append(
            region_legend_patch(
                contours.current_color,
                contours.fill_alpha,
                "current no-flip region: both triangles inside",
            )
        )

    if show_regions and contours.draw_failure_reasons:
        failure_reason_legend = (
            (
                "g12_negative",
                contours.g12_negative_color,
                r"flip reason: $G_{12}<0$",
            ),
            (
                "g12_too_large",
                contours.g12_too_large_color,
                r"flip reason: $G_{12}>\min(G_{11}, G_{22})$",
            ),
            (
                "shared_edge_not_longest",
                contours.shared_edge_not_longest_color,
                "no flip: shared edge is not longest",
            ),
        )
        for reason_key, color, label in failure_reason_legend:
            reason_mask = current_reconnection_masks[reason_key]
            reason_is_present = np.any(reason_mask)
            if reason_is_present and has_region_boundary(reason_mask):
                ax.contour(
                    L_values,
                    u_values,
                    reason_mask.astype(float),
                    levels=[contours.level],
                    colors=color,
                    linewidths=contours.linewidth,
                    alpha=1.0,
                    zorder=12,
                )
            if reason_is_present or contours.show_empty_failure_reasons:
                shown_label = label if reason_is_present else f"{label} (not present)"
                legend_handles.append(
                    region_legend_patch(
                        color,
                        contours.failure_fill_alpha,
                        shown_label,
                    )
                )

    has_flipped_reconnection_boundary = has_region_boundary(
        flipped_reconnection_zone_mask
    )
    if show_regions and contours.draw_flipped and has_flipped_reconnection_boundary:
        ax.contour(
            L_values,
            u_values,
            flipped_reconnection_zone_mask.astype(float),
            levels=[contours.level],
            colors=contours.flipped_color,
            linewidths=contours.linewidth,
            linestyles="--",
            alpha=1.0,
            zorder=11,
        )
        legend_handles.append(
            region_legend_patch(
                contours.flipped_color,
                contours.fill_alpha,
                "flipped no-flip region: both triangles inside",
            )
        )

    if legend_handles:
        legend = ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
        legend.set_zorder(100)

    ax.axvline(REFERENCE_L, color="0.2", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(REFERENCE_U, color="0.2", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("L")
    ax.set_ylabel("u")
    if contours.debug_only:
        ax.set_title(rf"Reconnection-zone mask ({config.resolution}x{config.resolution})")
    elif content == HEATMAP_REGIONS_ONLY:
        ax.set_title(rf"Reconnection regions ({config.resolution}x{config.resolution})")
    else:
        ax.set_title(
            rf"$E_{{\mathrm{{flipped}}}} - E_{{\mathrm{{current}}}}$ "
            rf"({config.resolution}x{config.resolution}, {config.color_scale})"
        )
    if image is not None:
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(
            "reconnection zone: 1=current, 2=flipped, 3=both"
            if contours.debug_only
            else r"$\Delta E = E_{\mathrm{flipped}} - E_{\mathrm{current}}$"
        )

    data = {
        "L_values": L_values,
        "u_values": u_values,
        "delta_energy": delta_energy,
        "abs_delta_energy": abs_delta_energy,
        "current_energy": current_energy,
        "flipped_energy": flipped_energy,
        "inside_current_reconnection_zone": current_reconnection_zone_mask,
        "inside_flipped_reconnection_zone": flipped_reconnection_zone_mask,
        "inside_reconnection_zone": current_reconnection_zone_mask,
        "current_g12_negative": current_reconnection_masks["g12_negative"],
        "current_g12_too_large": current_reconnection_masks["g12_too_large"],
        "current_shared_edge_not_longest": current_reconnection_masks[
            "shared_edge_not_longest"
        ],
    }
    return fig, data


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")


def flip_mode_output_path(path: Path, flip_mode: FlipMode) -> Path:
    return path.with_name(f"{path.stem}_{flip_mode.name}{path.suffix}")


def heatmap_content_output_path(path: Path, content: str) -> Path:
    validate_heatmap_content(content)
    if content == HEATMAP_COMBINED:
        return path
    return path.with_name(f"{path.stem}_{content}{path.suffix}")


def build_heatmap_variants(
    config: HeatmapConfig,
    material: MaterialConfig,
    flip_mode: FlipMode,
) -> tuple[list[tuple[str, plt.Figure]], dict[str, dict[str, np.ndarray]]]:
    if len(config.contents) == 0:
        raise ValueError("HeatmapConfig.contents must contain at least one variant.")

    figures = []
    data_by_content = {}
    for content in config.contents:
        fig, data = build_flip_energy_heatmap(
            config=config,
            material=material,
            flip_mode=flip_mode,
            content=content,
        )
        figures.append((content, fig))
        data_by_content[content] = data
    return figures, data_by_content


def representative_heatmap_data(
    data_by_content: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not data_by_content:
        raise ValueError("Expected at least one heatmap data set.")
    if HEATMAP_COMBINED in data_by_content:
        return data_by_content[HEATMAP_COMBINED]
    return next(iter(data_by_content.values()))


def build_and_save_for_flip_mode(
    config: PlotConfig,
    flip_mode: FlipMode,
) -> list[plt.Figure]:
    print(f"\nFlip mode: {flip_mode.name}")
    fig = build_symmetric_parameterization_grid(
        config=config.pair_grid,
        material=config.material,
        flip_mode=flip_mode,
    )
    flipped_fig = build_flipped_parameterization_grid(
        config=config.pair_grid,
        material=config.material,
        flip_mode=flip_mode,
    )
    heatmap_figures, heatmap_data_by_content = build_heatmap_variants(
        config=config.heatmap,
        material=config.material,
        flip_mode=flip_mode,
    )
    focused_heatmap_figures, focused_heatmap_data_by_content = build_heatmap_variants(
        config=config.focused_heatmap,
        material=config.material,
        flip_mode=flip_mode,
    )
    cauchy_fig, cauchy_data = build_cauchy_stress_difference_heatmaps(
        config=config.cauchy_stress_difference,
        material=config.material,
        flip_mode=flip_mode,
    )
    if config.plot_first_element_G:
        first_element_G_fig, first_element_G_data = build_first_element_G_heatmaps(
            config=config.first_element_G,
            flip_mode=flip_mode,
        )
    else:
        first_element_G_fig = None
        first_element_G_data = None

    save_figure(fig, flip_mode_output_path(config.pair_grid.output_path, flip_mode))
    save_figure(
        flipped_fig,
        flip_mode_output_path(config.pair_grid.flipped_output_path, flip_mode),
    )
    for content, heatmap_fig in heatmap_figures:
        save_figure(
            heatmap_fig,
            flip_mode_output_path(
                heatmap_content_output_path(config.heatmap.output_path, content),
                flip_mode,
            ),
        )
    for content, focused_heatmap_fig in focused_heatmap_figures:
        save_figure(
            focused_heatmap_fig,
            flip_mode_output_path(
                heatmap_content_output_path(
                    config.focused_heatmap.output_path,
                    content,
                ),
                flip_mode,
            ),
        )
    save_figure(
        cauchy_fig,
        flip_mode_output_path(config.cauchy_stress_difference.output_path, flip_mode),
    )
    if first_element_G_fig is not None:
        save_figure(
            first_element_G_fig,
            flip_mode_output_path(config.first_element_G.output_path, flip_mode),
        )

    heatmap_data = representative_heatmap_data(heatmap_data_by_content)
    focused_heatmap_data = representative_heatmap_data(focused_heatmap_data_by_content)
    delta_energy = heatmap_data["delta_energy"]
    focused_delta_energy = focused_heatmap_data["delta_energy"]
    print(
        "Delta E range: "
        f"{float(np.nanmin(delta_energy)):.6e} to {float(np.nanmax(delta_energy)):.6e}"
    )
    print(
        "Focused Delta E range: "
        f"{float(np.nanmin(focused_delta_energy)):.6e} "
        f"to {float(np.nanmax(focused_delta_energy)):.6e}"
    )
    cauchy_values = cauchy_data["matrix_values"]
    print(
        "Cauchy stress difference range: "
        f"{float(np.nanmin(cauchy_values)):.6e} "
        f"to {float(np.nanmax(cauchy_values)):.6e}"
    )
    if first_element_G_data is not None:
        first_element_G_values = first_element_G_data["matrix_values"]
        print(
            "First element G range: "
            f"{float(np.nanmin(first_element_G_values)):.6e} "
            f"to {float(np.nanmax(first_element_G_values)):.6e}"
        )
    current_reconnection_zone_mask = heatmap_data["inside_current_reconnection_zone"]
    flipped_reconnection_zone_mask = heatmap_data["inside_flipped_reconnection_zone"]
    focused_current_reconnection_zone_mask = focused_heatmap_data[
        "inside_current_reconnection_zone"
    ]
    focused_flipped_reconnection_zone_mask = focused_heatmap_data[
        "inside_flipped_reconnection_zone"
    ]
    print(
        "Current reconnection-zone occupancy: "
        f"{int(np.count_nonzero(current_reconnection_zone_mask))} / "
        f"{current_reconnection_zone_mask.size}"
    )
    print(
        "Flipped reconnection-zone occupancy: "
        f"{int(np.count_nonzero(flipped_reconnection_zone_mask))} / "
        f"{flipped_reconnection_zone_mask.size}"
    )
    print(
        "Focused current reconnection-zone occupancy: "
        f"{int(np.count_nonzero(focused_current_reconnection_zone_mask))} / "
        f"{focused_current_reconnection_zone_mask.size}"
    )
    print(
        "Focused flipped reconnection-zone occupancy: "
        f"{int(np.count_nonzero(focused_flipped_reconnection_zone_mask))} / "
        f"{focused_flipped_reconnection_zone_mask.size}"
    )

    figures = [
        fig,
        flipped_fig,
        *[heatmap_fig for _, heatmap_fig in heatmap_figures],
        *[
            focused_heatmap_fig
            for _, focused_heatmap_fig in focused_heatmap_figures
        ],
        cauchy_fig,
    ]
    if first_element_G_fig is not None:
        figures.append(first_element_G_fig)
    if not config.show:
        for figure in figures:
            plt.close(figure)
        return []
    return figures


def main(config: PlotConfig = CONFIG) -> None:
    if len(config.flip_modes) == 0:
        raise ValueError("config.flip_modes must contain at least one flip mode.")

    figures = []
    for flip_mode in config.flip_modes:
        figures.extend(build_and_save_for_flip_mode(config, flip_mode))

    if config.show:
        plt.show()
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
