"""Two-element edge-flip parameterization plots.

The T23 parameterization follows Sylvain's notation:

    x2 = (0, 0), x3 = (L, 0),
    x1 = (s L, -1 / L), x4 = (t L, 1 / L).

The full T23-controlled geometry is described by (L, u, v), where
u = s + t - 1 and v = s - t. The plotting config chooses a two-parameter
slice: either the symmetric slice v=0, or the antisymmetric slice u=0.

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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Polygon
from tqdm import tqdm

from MTMath.energyFunction import ContiEnergy

T23_TRIANGLES = ((1, 2, 3), (2, 3, 4))
T14_TRIANGLES = ((1, 2, 4), (1, 4, 3))
DIAGONAL_23 = (2, 3)
DIAGONAL_14 = (1, 4)
REFERENCE_L = float(np.sqrt(2.0))
REFERENCE_U = 0.0
REFERENCE_V = 0.0
REFERENCE_PARAMETER = 0.0
ROOT_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT_DIR / "Plots"
HEATMAP_COMBINED = "combined"
HEATMAP_ENERGY_ONLY = "energy_only"
HEATMAP_REGIONS_ONLY = "regions_only"
HEATMAP_CONTENTS = (HEATMAP_COMBINED, HEATMAP_ENERGY_ONLY, HEATMAP_REGIONS_ONLY)
PARAMETERIZATION_SYMMETRIC = "symmetric"
PARAMETERIZATION_ANTISYMMETRIC = "antisymmetric"
PARAMETERIZATION_MODES = (
    PARAMETERIZATION_SYMMETRIC,
    PARAMETERIZATION_ANTISYMMETRIC,
)
NODE_BOTTOM = 1
NODE_LEFT = 2
NODE_RIGHT = 3
NODE_TOP = 4
G_VECTOR_CHOICE_SHORTEST = "shortest_edges"
G_VECTOR_CHOICE_OPTION_1 = "local_corner_1"
G_VECTOR_CHOICE_OPTION_2 = "local_corner_2"
G_VECTOR_CHOICE_OPTION_3 = "local_corner_3"
FIXED_G_VECTOR_CHOICES = (
    G_VECTOR_CHOICE_OPTION_1,
    G_VECTOR_CHOICE_OPTION_2,
    G_VECTOR_CHOICE_OPTION_3,
)
G_VECTOR_CHOICES = (G_VECTOR_CHOICE_SHORTEST, *FIXED_G_VECTOR_CHOICES)
PROGRESS_GRID_SIZE_THRESHOLD = 400

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
class ParameterizationConfig:
    mode: str = PARAMETERIZATION_SYMMETRIC


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
    parameter_values: tuple[float, ...]
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
    g12_too_large_hatch: str = "///"
    shared_edge_not_longest_color: str = "deepskyblue"
    linewidth: float = 1.4
    fill_alpha: float = 0.12
    failure_fill_alpha: float = 0.16


@dataclass(frozen=True)
class HeatmapElementPairGridConfig:
    draw: bool = True
    size: int = 4
    padding_fraction: float = 0.10
    scale_fraction: float = 0.085
    alpha: float = 0.55
    linewidth: float = 0.8
    g_vector_color: str = "gold"
    g_vector_alpha: float = 0.9
    g_vector_linewidth: float = 1.8
    g_vector_arrow_scale: float = 7.0


@dataclass(frozen=True)
class ReferenceEnergyContourConfig:
    draw: bool = True
    shear: float = 0.5
    color: str = "black"
    linestyle: str = "--"
    linewidth: float = 1.4
    zorder: int = 16
    use_absolute_delta_energy: bool = True
    label: str | None = None


@dataclass(frozen=True)
class HeatmapConfig:
    output_path: Path
    resolution: int
    L_range: tuple[float, float]
    parameter_range: tuple[float, float]
    cmap: str = "coolwarm"
    color_scale: str = "linear"  # "linear" or "power"
    power_gamma: float = 0.5
    reconnection_contours: ReconnectionContourConfig = ReconnectionContourConfig()
    element_pair_grid: HeatmapElementPairGridConfig = HeatmapElementPairGridConfig()
    contents: tuple[str, ...] = HEATMAP_CONTENTS
    extra_region_g_vector_choices: tuple[str, ...] = FIXED_G_VECTOR_CHOICES


@dataclass(frozen=True)
class MatrixFieldPlotConfig:
    output_path: Path
    resolution: int
    L_range: tuple[float, float]
    parameter_range: tuple[float, float]
    title: str
    colorbar_label: str
    component_symbol: str = ""
    cmap: str = "coolwarm"
    color_scale: str = "linear"  # "linear" or "power"
    power_gamma: float = 0.5
    centered_colorbar: bool = True
    components: tuple[tuple[int, int], ...] = (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )
    columns: int | None = None
    element_pair_grid: HeatmapElementPairGridConfig = HeatmapElementPairGridConfig()


@dataclass(frozen=True)
class PlotConfig:
    flip_mode: FlipMode
    parameterization: ParameterizationConfig
    material: MaterialConfig
    reference_energy_contour: ReferenceEnergyContourConfig
    pair_grid: PairGridConfig
    heatmap: HeatmapConfig
    focused_heatmap: HeatmapConfig
    cauchy_stress_difference: MatrixFieldPlotConfig
    first_element_G: MatrixFieldPlotConfig
    flip_modes: tuple[FlipMode, ...] = DEFAULT_FLIP_MODES
    remove_figure_titles: bool = True
    plot_focused_heatmap: bool = False
    plot_first_element_G: bool = False
    show: bool = False


CONFIG = PlotConfig(
    flip_mode=FIRST_FLIP_T23_TO_T14,
    # Default to the symmetric slice v=0 from PDF Section 6.1. Switching to
    # PARAMETERIZATION_ANTISYMMETRIC gives the affine slice u=0; PDF Section 5
    # predicts Delta E_flip(L, 0, v)=0 there.
    parameterization=ParameterizationConfig(),
    material=MaterialConfig(),
    reference_energy_contour=ReferenceEnergyContourConfig(),
    pair_grid=PairGridConfig(
        L_values=(1, REFERENCE_L, REFERENCE_L * 2),
        parameter_values=(0.8, 0.0, -0.8),
        output_path=PLOTS_DIR / "two_element_parameterization_grid.pdf",
        flipped_output_path=PLOTS_DIR / "two_element_parameterization_flipped_grid.pdf",
    ),
    heatmap=HeatmapConfig(
        output_path=PLOTS_DIR / "two_element_parameterization_flip_energy_heatmap.pdf",
        resolution=500,
        L_range=(0.75, REFERENCE_L * 2),
        parameter_range=(-0.9, 0.9),
    ),
    focused_heatmap=HeatmapConfig(
        output_path=PLOTS_DIR
        / "two_element_parameterization_flip_energy_heatmap_focused.pdf",
        resolution=500,
        L_range=(REFERENCE_L, 2),
        parameter_range=(-0.3, 0.3),
    ),
    cauchy_stress_difference=MatrixFieldPlotConfig(
        output_path=PLOTS_DIR
        / "two_element_parameterization_cauchy_stress_difference.pdf",
        resolution=500,
        L_range=(0.75, REFERENCE_L * 2),
        parameter_range=(-0.9, 0.9),
        title=r"$\sigma_{flipped} - \sigma_{current}$",
        colorbar_label=r"$\Delta\sigma$",
        component_symbol=r"\Delta\sigma",
        components=((0, 0), (0, 1), (1, 1)),
        columns=3,
    ),
    first_element_G=MatrixFieldPlotConfig(
        output_path=PLOTS_DIR / "two_element_parameterization_first_element_G.pdf",
        resolution=500,
        L_range=(0.75, REFERENCE_L * 3),
        parameter_range=(-0.9, 0.9),
        title="First current element G",
        colorbar_label="G",
        component_symbol="G",
        cmap="viridis",
        centered_colorbar=False,
    ),
    flip_modes=DEFAULT_FLIP_MODES,
    remove_figure_titles=True,
    plot_first_element_G=False,
    show=False,
)


def validate_flip_mode(flip_mode: FlipMode) -> None:
    if flip_mode not in DEFAULT_FLIP_MODES:
        raise ValueError(f"Unsupported flip mode: {flip_mode.name}")


def validate_parameterization(parameterization: ParameterizationConfig) -> None:
    if parameterization.mode not in PARAMETERIZATION_MODES:
        raise ValueError(
            f"Unsupported parameterization mode {parameterization.mode!r}. "
            f"Use one of {PARAMETERIZATION_MODES}."
        )


def parameter_axis_label(parameterization: ParameterizationConfig) -> str:
    validate_parameterization(parameterization)
    return r"$u$" if parameterization.mode == PARAMETERIZATION_SYMMETRIC else r"$v$"


def parameterization_title(parameterization: ParameterizationConfig) -> str:
    validate_parameterization(parameterization)
    if parameterization.mode == PARAMETERIZATION_SYMMETRIC:
        return "Symmetric two-element parameterization (v=0)"
    return "Antisymmetric two-element parameterization (u=0)"


def parameter_value_to_uv(
    parameter_value: float,
    parameterization: ParameterizationConfig,
) -> tuple[float, float]:
    """Map the plotted second-axis value to (u, v).

    PDF Sections 4 and 6 define u=s+t-1 and v=s-t. The symmetric slice fixes
    v=0, while the antisymmetric slice fixes u=0.
    """
    validate_parameterization(parameterization)
    if parameterization.mode == PARAMETERIZATION_SYMMETRIC:
        return float(parameter_value), REFERENCE_V
    return REFERENCE_U, float(parameter_value)


def parameter_values_to_uv_arrays(
    parameter_values: np.ndarray,
    parameterization: ParameterizationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    validate_parameterization(parameterization)
    parameter_values = np.asarray(parameter_values, dtype=float)
    if parameterization.mode == PARAMETERIZATION_SYMMETRIC:
        return parameter_values, np.zeros_like(parameter_values)
    return np.zeros_like(parameter_values), parameter_values


def uv_to_st(u: np.ndarray | float, v: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return (s, t) from PDF Section 4, equation (24)."""
    return 0.5 * (1.0 + u + v), 0.5 * (1.0 + u - v)


def t23_parameterized_vertices(
    L: float,
    u: float = REFERENCE_U,
    *,
    v: float = REFERENCE_V,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> dict[int, np.ndarray]:
    """Return vertices from PDF Section 4, equation (27)."""
    if L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")
    if abs(u) >= 1.0:
        raise ValueError(
            f"u must be in (-1, 1) so the flipped triangles stay positive, got {u}."
        )
    validate_flip_mode(flip_mode)

    s, t = uv_to_st(u, v)
    return {
        1: np.array([s * L, -1.0 / L]),
        2: np.array([0.0, 0.0]),
        3: np.array([L, 0.0]),
        4: np.array([t * L, 1.0 / L]),
    }


def t23_parameterized_vertices_from_parameter(
    L: float,
    parameter_value: float,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> dict[int, np.ndarray]:
    u, v = parameter_value_to_uv(parameter_value, parameterization)
    return t23_parameterized_vertices(L, u=u, v=v, flip_mode=flip_mode)


def square_reference_vertices(
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> dict[int, np.ndarray]:
    return t23_parameterized_vertices(
        REFERENCE_L,
        u=REFERENCE_U,
        v=REFERENCE_V,
        flip_mode=flip_mode,
    )


def vertices_to_array(vertices: dict[int, np.ndarray]) -> np.ndarray:
    return np.array([vertices[index] for index in (1, 2, 3, 4)], dtype=float)


def validate_g_vector_choice(g_vector_choice: str) -> None:
    if g_vector_choice not in G_VECTOR_CHOICES:
        raise ValueError(
            f"Unsupported G-vector choice {g_vector_choice!r}. "
            f"Use one of {G_VECTOR_CHOICES}."
        )


def shared_edge_from_triangles(
    triangles: tuple[tuple[int, int, int], ...],
) -> frozenset[int]:
    if len(triangles) != 2:
        raise ValueError(f"Expected exactly two triangles, got {len(triangles)}.")
    shared_vertices = set(triangles[0]) & set(triangles[1])
    if len(shared_vertices) != 2:
        raise ValueError(
            f"Expected the two triangles to share one edge, got {triangles}."
        )
    return frozenset(shared_vertices)


def fixed_g_corner_vertex(
    triangle: tuple[int, int, int],
    triangles: tuple[tuple[int, int, int], ...],
    g_vector_choice: str,
) -> int:
    shared_edge = shared_edge_from_triangles(triangles)
    if g_vector_choice == G_VECTOR_CHOICE_OPTION_2:
        candidates = [vertex for vertex in triangle if vertex not in shared_edge]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one non-shared corner in triangle {triangle}."
            )
        return candidates[0]

    if shared_edge == frozenset((NODE_LEFT, NODE_RIGHT)):
        corner = (
            NODE_LEFT
            if g_vector_choice == G_VECTOR_CHOICE_OPTION_1
            else NODE_RIGHT
        )
    elif shared_edge == frozenset((NODE_BOTTOM, NODE_TOP)):
        corner = (
            NODE_TOP
            if g_vector_choice == G_VECTOR_CHOICE_OPTION_1
            else NODE_BOTTOM
        )
    else:
        raise ValueError(f"Unsupported shared edge {tuple(sorted(shared_edge))}.")

    if corner not in triangle:
        raise RuntimeError(f"Selected corner {corner} is not in triangle {triangle}.")
    return corner


def fixed_g_vector_choice_label(
    triangles: tuple[tuple[int, int, int], ...],
    g_vector_choice: str,
) -> str:
    option_number = FIXED_G_VECTOR_CHOICES.index(g_vector_choice) + 1
    shared_edge = shared_edge_from_triangles(triangles)
    if shared_edge == frozenset((NODE_LEFT, NODE_RIGHT)):
        labels = ("left corner for both", "bottom/top corners", "right corner for both")
    elif shared_edge == frozenset((NODE_BOTTOM, NODE_TOP)):
        labels = ("top corner for both", "left/right corners", "bottom corner for both")
    else:
        raise ValueError(f"Unsupported shared edge {tuple(sorted(shared_edge))}.")
    return f"option {option_number}: {labels[option_number - 1]}"


def g_vector_choice_label(
    g_vector_choice: str,
    triangles: tuple[tuple[int, int, int], ...] | None = None,
) -> str:
    validate_g_vector_choice(g_vector_choice)
    if g_vector_choice == G_VECTOR_CHOICE_SHORTEST:
        return "two shortest edges"
    if triangles is None:
        option_number = FIXED_G_VECTOR_CHOICES.index(g_vector_choice) + 1
        return f"option {option_number}"
    return fixed_g_vector_choice_label(triangles, g_vector_choice)


def triangle_g_vectors_and_edges(
    vertices: dict[int, np.ndarray],
    triangle: tuple[int, int, int],
    g_vector_choice: str,
    triangles: tuple[tuple[int, int, int], ...] | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    validate_g_vector_choice(g_vector_choice)
    if g_vector_choice == G_VECTOR_CHOICE_SHORTEST:
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
            raise RuntimeError(
                f"Expected the two selected edges to share one vertex in {triangle}."
            )
        shared_vertex = shared_vertices.pop()

        vectors = []
        selected_edges = []
        for start, end, _ in shortest_edges:
            other_vertex = end if start == shared_vertex else start
            vectors.append(vertices[other_vertex] - vertices[shared_vertex])
            selected_edges.append((shared_vertex, other_vertex))
        return vectors, selected_edges

    if triangles is None:
        raise ValueError("Fixed G-vector choices require the full triangle pair.")
    shared_vertex = fixed_g_corner_vertex(triangle, triangles, g_vector_choice)
    other_vertices = [vertex for vertex in triangle if vertex != shared_vertex]
    vectors = [vertices[vertex] - vertices[shared_vertex] for vertex in other_vertices]
    selected_edges = [(shared_vertex, vertex) for vertex in other_vertices]
    return vectors, selected_edges


def selected_g_vector_edges(
    vertices: dict[int, np.ndarray],
    triangles: tuple[tuple[int, int, int], ...],
    g_vector_choice: str,
) -> list[tuple[int, int]]:
    edges = []
    for triangle in triangles:
        _, selected_edges = triangle_g_vectors_and_edges(
            vertices,
            triangle,
            g_vector_choice,
            triangles,
        )
        edges.extend(selected_edges)
    return edges


def getG(
    vertices: dict[int, np.ndarray],
    triangles: tuple[tuple[int, int, int], ...],
    g_vector_choice: str = G_VECTOR_CHOICE_SHORTEST,
) -> np.ndarray:
    """Return one Gram matrix per triangle using the selected edge-vector rule.

    The selected edges are oriented away from their shared triangle vertex before
    computing G_ij = v_i dot v_j. The result has shape (len(triangles), 2, 2).
    """
    G_values = []
    for triangle in triangles:
        vectors, _ = triangle_g_vectors_and_edges(
            vertices,
            triangle,
            g_vector_choice,
            triangles,
        )
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
    parameter_values: tuple[float, ...] | list[float] | np.ndarray,
    triangles: tuple[tuple[int, int, int], ...] | None = None,
    shared_edge: tuple[int, int] | None = None,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    g_vector_choice: str = G_VECTOR_CHOICE_SHORTEST,
) -> dict[str, np.ndarray]:
    """Return C++ inRegion and failure-reason masks on a parameter/L grid.

    The returned array has shape (len(parameter_values), len(L_values)), matching the
    heatmap orientation used by build_flip_energy_heatmap. The inside mask is
    True when every triangle satisfies 0 <= G_12 <= min(G_11, G_22), and when
    the specified shared edge is a longest edge of every triangle. G is computed
    from the configured edge-vector choice.
    """
    L_values = _as_1d_float_array(L_values, "L_values")
    parameter_values = _as_1d_float_array(parameter_values, "parameter_values")
    validate_parameterization(parameterization)
    validate_g_vector_choice(g_vector_choice)
    if triangles is None:
        triangles = flip_mode.current_triangles
    if shared_edge is None:
        shared_edge = flip_mode.current_diagonal
    shape = (len(parameter_values), len(L_values))
    inside_mask = np.zeros(shape, dtype=bool)
    g12_negative_mask = np.zeros(shape, dtype=bool)
    g12_too_large_mask = np.zeros(shape, dtype=bool)
    shared_edge_not_longest_mask = np.zeros(shape, dtype=bool)

    for row, col in _progress_grid_indices(
        shape,
        L_values,
        parameter_values,
        f"Reconnection masks: {g_vector_choice}",
    ):
        parameter_value = parameter_values[row]
        L = L_values[col]
        vertices = t23_parameterized_vertices_from_parameter(
            float(L),
            float(parameter_value),
            parameterization=parameterization,
            flip_mode=flip_mode,
        )
        G_values = getG(
            vertices,
            triangles,
            g_vector_choice=g_vector_choice,
        )
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
    parameter_values: tuple[float, ...] | list[float] | np.ndarray,
    triangles: tuple[tuple[int, int, int], ...] | None = None,
    shared_edge: tuple[int, int] | None = None,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    g_vector_choice: str = G_VECTOR_CHOICE_SHORTEST,
) -> np.ndarray:
    """Return a Boolean mask for the C++ inRegion condition on a parameter/L grid."""
    return reconnection_condition_masks(
        L_values,
        parameter_values,
        triangles=triangles,
        shared_edge=shared_edge,
        parameterization=parameterization,
        flip_mode=flip_mode,
        g_vector_choice=g_vector_choice,
    )["inside"]


def t23_parameterized_vertices_array(
    L: np.ndarray,
    u: np.ndarray,
    *,
    v: np.ndarray | float = REFERENCE_V,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    L = np.asarray(L, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if v.shape == ():
        v = np.full_like(u, float(v))
    if np.any(L <= 0.0):
        raise ValueError("All L values must be positive.")
    if np.any(np.abs(u) >= 1.0):
        raise ValueError("All u values must be in (-1, 1).")
    if L.shape != u.shape or L.shape != v.shape:
        raise ValueError(
            "L, u, and v must have the same shape, "
            f"got {L.shape}, {u.shape}, and {v.shape}."
        )
    validate_flip_mode(flip_mode)

    vertices = np.empty(L.shape + (4, 2), dtype=float)
    s, t = uv_to_st(u, v)
    vertices[..., 0, 0] = s * L
    vertices[..., 0, 1] = -1.0 / L
    vertices[..., 1, :] = 0.0
    vertices[..., 2, 0] = L
    vertices[..., 2, 1] = 0.0
    vertices[..., 3, 0] = t * L
    vertices[..., 3, 1] = 1.0 / L
    return vertices


def t23_parameterized_vertices_array_from_parameter(
    L: np.ndarray,
    parameter_values: np.ndarray,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    u, v = parameter_values_to_uv_arrays(parameter_values, parameterization)
    return t23_parameterized_vertices_array(L, u, v=v, flip_mode=flip_mode)


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


def _progress_grid_indices(
    shape: tuple[int, int],
    L_values: np.ndarray,
    parameter_values: np.ndarray,
    desc: str,
):
    indices = np.ndindex(shape)
    if (
        len(L_values) >= PROGRESS_GRID_SIZE_THRESHOLD
        and len(parameter_values) >= PROGRESS_GRID_SIZE_THRESHOLD
    ):
        return tqdm(
            indices,
            total=shape[0] * shape[1],
            desc=desc,
            leave=False,
            dynamic_ncols=True,
        )
    return indices


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


def horizontal_simple_shear_F(shear: float) -> np.ndarray:
    """Return horizontal simple shear F for x' = x + shear*y."""
    if not np.isfinite(shear):
        raise ValueError(f"Shear must be finite, got {shear}.")
    return np.array([[1.0, shear], [0.0, 1.0]], dtype=float)


def reference_simple_shear_energy(
    material: MaterialConfig,
    config: ReferenceEnergyContourConfig,
) -> float:
    """Return the two-half-area-element reference energy for simple shear.

    This is 2 * (triangle area 1/2) * psi(F), matching the pair-energy area
    weighting used above.
    """
    F = horizontal_simple_shear_F(config.shear)
    energy_density = ContiEnergy.energy_from_F(
        F,
        beta=material.beta,
        K=material.K,
        noise=material.noise,
        zeroReference=True,
    )
    return float(2.0 * 0.5 * energy_density)


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
    parameter_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    L_grid, parameter_grid = np.meshgrid(L_values, parameter_values, indexing="xy")
    current_vertices = t23_parameterized_vertices_array_from_parameter(
        L_grid,
        parameter_grid,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
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
    parameter_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> tuple[np.ndarray, np.ndarray]:
    L_grid, parameter_grid = np.meshgrid(L_values, parameter_values, indexing="xy")
    current_vertices = t23_parameterized_vertices_array_from_parameter(
        L_grid,
        parameter_grid,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    reference_vertices = square_reference_vertices(flip_mode=flip_mode)
    reference_array = vertices_to_array(reference_vertices)
    F = deformation_gradients_from_vertex_array(reference_array, current_vertices, triangles)
    reference_areas = np.array(
        [triangle_reference_area(reference_vertices, triangle) for triangle in triangles]
    )
    return F, reference_areas


def pair_cauchy_stress_grid(
    L_values: np.ndarray,
    parameter_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    F, reference_areas = pair_deformation_gradient_grid(
        L_values,
        parameter_values,
        triangles,
        parameterization=parameterization,
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
    parameter_values: np.ndarray,
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    current_stress = pair_cauchy_stress_grid(
        L_values,
        parameter_values,
        flip_mode.current_triangles,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    flipped_stress = pair_cauchy_stress_grid(
        L_values,
        parameter_values,
        flip_mode.flipped_triangles,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    return flipped_stress - current_stress


def first_element_G_grid(
    L_values: np.ndarray,
    parameter_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    values = np.empty((len(parameter_values), len(L_values), 2, 2), dtype=float)
    if triangles is None:
        triangles = flip_mode.current_triangles
    first_triangle = (triangles[0],)
    shape = (len(parameter_values), len(L_values))
    for row, col in _progress_grid_indices(
        shape,
        L_values,
        parameter_values,
        "First element G",
    ):
        parameter_value = parameter_values[row]
        L = L_values[col]
        vertices = t23_parameterized_vertices_from_parameter(
            float(L),
            float(parameter_value),
            parameterization=parameterization,
            flip_mode=flip_mode,
        )
        values[row, col] = getG(vertices, first_triangle)[0]
    return values


def edge_flip_energy_difference_grid(
    L_values: np.ndarray,
    parameter_values: np.ndarray,
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_energy = pair_energy_grid(
        L_values,
        parameter_values,
        flip_mode.current_triangles,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    flipped_energy = pair_energy_grid(
        L_values,
        parameter_values,
        flip_mode.flipped_triangles,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    return flipped_energy - current_energy, current_energy, flipped_energy


def _all_vertices(
    L_values: np.ndarray,
    parameter_values: np.ndarray,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> list[dict[int, np.ndarray]]:
    vertex_sets = []
    shape = (len(parameter_values), len(L_values))
    for row, col in _progress_grid_indices(
        shape,
        L_values,
        parameter_values,
        "Element-pair vertices",
    ):
        parameter_value = parameter_values[row]
        L = L_values[col]
        current_vertices = t23_parameterized_vertices_from_parameter(
            float(L),
            float(parameter_value),
            parameterization=parameterization,
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
    parameter_range: tuple[float, float],
    config: HeatmapElementPairGridConfig = CONFIG.heatmap.element_pair_grid,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    active_diagonal: tuple[int, int] | None = None,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    g_vector_choice: str | None = None,
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
    validate_parameterization(parameterization)
    if g_vector_choice is not None:
        validate_g_vector_choice(g_vector_choice)

    L_min, L_max = L_range
    parameter_min, parameter_max = parameter_range
    L_span = L_max - L_min
    parameter_span = parameter_max - parameter_min
    L_centers = np.linspace(
        L_min + config.padding_fraction * L_span,
        L_max - config.padding_fraction * L_span,
        config.size,
    )
    parameter_centers = np.linspace(
        parameter_min + config.padding_fraction * parameter_span,
        parameter_max - config.padding_fraction * parameter_span,
        config.size,
    )
    target_size = config.scale_fraction * min(L_span, parameter_span)
    active_diagonal = tuple(sorted(active_diagonal))
    edges = sorted(_triangle_edges(triangles))
    sampled_geometries = []
    max_radius = 0.0

    for parameter_center in parameter_centers:
        for L_center in L_centers:
            vertices = t23_parameterized_vertices_from_parameter(
                float(L_center),
                float(parameter_center),
                parameterization=parameterization,
                flip_mode=flip_mode,
            )
            points = np.array(list(vertices.values()))
            centroid = points.mean(axis=0)
            radius = float(np.max(np.linalg.norm(points - centroid, axis=1)))
            sampled_geometries.append((L_center, parameter_center, vertices, centroid))
            max_radius = max(max_radius, radius)

    if max_radius <= 0.0:
        raise RuntimeError("Expected non-degenerate element pairs in heatmap overlay.")
    scale = target_size / max_radius

    for L_center, parameter_center, vertices, centroid in sampled_geometries:
        transformed_vertices = {
            index: np.array([L_center, parameter_center]) + scale * (point - centroid)
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

        if g_vector_choice is not None:
            for start, end in selected_g_vector_edges(
                transformed_vertices,
                triangles,
                g_vector_choice,
            ):
                arrow = FancyArrowPatch(
                    transformed_vertices[start],
                    transformed_vertices[end],
                    arrowstyle="-|>",
                    mutation_scale=config.g_vector_arrow_scale,
                    shrinkA=0.0,
                    shrinkB=0.0,
                    linewidth=config.g_vector_linewidth,
                    color=config.g_vector_color,
                    alpha=config.g_vector_alpha,
                    zorder=10,
                )
                ax.add_patch(arrow)


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


def build_parameterization_grid(
    config: PairGridConfig = CONFIG.pair_grid,
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    active_diagonal: tuple[int, int] | None = None,
    inactive_diagonal: tuple[int, int] | None = None,
    title: str | None = None,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
) -> plt.Figure:
    """Build a grid of two-triangle element pairs for the selected closure."""
    L_values = _as_1d_float_array(config.L_values, "L_values")
    parameter_values = _as_1d_float_array(config.parameter_values, "parameter_values")
    validate_parameterization(parameterization)
    if triangles is None:
        triangles = flip_mode.current_triangles
    if active_diagonal is None:
        active_diagonal = flip_mode.current_diagonal
    vertex_sets = _all_vertices(
        L_values,
        parameter_values,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    reference_vertices = square_reference_vertices(flip_mode=flip_mode)
    limits = _shared_limits(vertex_sets + [reference_vertices])
    axis_label = parameter_axis_label(parameterization)

    fig, axes = plt.subplots(
        len(parameter_values),
        len(L_values),
        figsize=(3.4 * len(L_values), 3.4 * len(parameter_values)),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for row, parameter_value in enumerate(parameter_values):
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
            ax.set_title(
                rf"$L$={L_label}, {axis_label}={parameter_value:g}"
                f"\nE={energy:.3e}",
                fontsize=11,
            )
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.22, linewidth=0.6)
            if row == len(parameter_values) - 1:
                ax.set_xlabel(r"$x$")
            if col == 0:
                ax.set_ylabel(r"$y$")

    if not remove_figure_title:
        if title is None:
            title = parameterization_title(parameterization)
        fig.suptitle(f"{title} ({flip_mode.name})", fontsize=14)
    return fig


def build_flipped_parameterization_grid(
    config: PairGridConfig = CONFIG.pair_grid,
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
) -> plt.Figure:
    return build_parameterization_grid(
        config=config,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
        triangles=flip_mode.flipped_triangles,
        active_diagonal=flip_mode.flipped_diagonal,
        title=f"{parameterization_title(parameterization)} after flip",
        remove_figure_title=remove_figure_title,
    )


def heatmap_color_norm(
    values: np.ndarray,
    config: HeatmapConfig = CONFIG.heatmap,
):
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Cannot build heatmap color scale from only non-finite values.")
    if config.color_scale == "linear":
        max_abs = float(np.max(np.abs(finite_values)))
        if max_abs <= 0.0:
            max_abs = 1.0
        return CenteredNorm(vcenter=0.0, halfrange=max_abs)
    if config.color_scale == "power":
        if config.power_gamma <= 0.0:
            raise ValueError(f"power_gamma must be positive, got {config.power_gamma}.")
        max_abs = float(np.max(np.abs(finite_values)))
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


def region_legend_patch(
    color: str,
    fill_alpha: float,
    label: str,
    hatch: str | None = None,
) -> Patch:
    return Patch(
        facecolor=to_rgba(color, fill_alpha),
        edgecolor=color,
        hatch=hatch,
        label=label,
    )


def add_hatched_region_overlay(
    ax: plt.Axes,
    L_values: np.ndarray,
    parameter_values: np.ndarray,
    mask: np.ndarray,
    color: str,
    hatch: str,
    level: float,
    zorder: int,
) -> None:
    if not np.any(mask):
        return
    hatched_region = ax.contourf(
        L_values,
        parameter_values,
        mask.astype(float),
        levels=[level, 1.5],
        colors="none",
        hatches=[hatch],
        zorder=zorder,
    )
    hatched_region.set_edgecolor(color)
    hatched_region.set_facecolor("none")


def reference_energy_contour_label(config: ReferenceEnergyContourConfig) -> str:
    if config.label is not None:
        return config.label
    return rf"$|\Delta E| = E_{{\gamma={config.shear:g}}}$"


def add_reference_energy_contour(
    ax: plt.Axes,
    L_values: np.ndarray,
    parameter_values: np.ndarray,
    delta_energy: np.ndarray,
    reference_energy: float,
    config: ReferenceEnergyContourConfig,
) -> bool:
    if not config.draw:
        return False
    if reference_energy < 0.0:
        raise ValueError(f"Reference energy must be non-negative, got {reference_energy}.")

    values = np.abs(delta_energy) if config.use_absolute_delta_energy else delta_energy
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return False
    if reference_energy < float(np.nanmin(finite_values)) or reference_energy > float(
        np.nanmax(finite_values)
    ):
        return False

    ax.contour(
        L_values,
        parameter_values,
        values,
        levels=[reference_energy],
        colors=config.color,
        linestyles=config.linestyle,
        linewidths=config.linewidth,
        zorder=config.zorder,
    )
    return True


def reference_energy_contour_handle(
    config: ReferenceEnergyContourConfig,
) -> Line2D:
    return Line2D(
        [0, 1],
        [0, 0],
        color=config.color,
        linestyle=config.linestyle,
        linewidth=config.linewidth,
        label=reference_energy_contour_label(config),
    )


def mask_scalar_field_to_region(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if values.shape != mask.shape:
        raise ValueError(f"values and mask shapes differ: {values.shape} vs {mask.shape}.")
    return np.where(mask, values, np.nan)


def mask_matrix_field_to_region(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if values.shape[:2] != mask.shape:
        raise ValueError(
            f"matrix field and mask grid shapes differ: {values.shape[:2]} vs {mask.shape}."
        )
    return np.where(mask[..., None, None], values, np.nan)


def sampled_parameter_values(
    resolution: int,
    L_range: tuple[float, float],
    parameter_range: tuple[float, float],
    parameterization: ParameterizationConfig = CONFIG.parameterization,
) -> tuple[np.ndarray, np.ndarray]:
    validate_parameterization(parameterization)
    if resolution < 2:
        raise ValueError(f"resolution must be at least 2, got {resolution}.")
    if L_range[0] <= 0.0 or L_range[1] <= 0.0:
        raise ValueError(f"L_range values must be positive, got {L_range}.")
    if L_range[0] >= L_range[1]:
        raise ValueError(f"L_range must be increasing, got {L_range}.")
    if not np.all(np.isfinite(parameter_range)):
        raise ValueError(f"parameter_range values must be finite, got {parameter_range}.")
    if parameter_range[0] >= parameter_range[1]:
        raise ValueError(
            f"parameter_range must be increasing, got {parameter_range}."
        )
    if (
        parameterization.mode == PARAMETERIZATION_SYMMETRIC
        and (parameter_range[0] <= -1.0 or parameter_range[1] >= 1.0)
    ):
        raise ValueError(
            "The symmetric u-range must stay inside (-1, 1) so the flipped "
            f"triangles stay positive, got {parameter_range}."
        )
    return (
        np.linspace(L_range[0], L_range[1], resolution),
        np.linspace(parameter_range[0], parameter_range[1], resolution),
    )


def matrix_field_color_norm(
    values: np.ndarray,
    config: MatrixFieldPlotConfig,
):
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Cannot build matrix color scale from only non-finite values.")
    if config.color_scale == "linear":
        if config.centered_colorbar:
            max_abs = float(np.max(np.abs(finite_values)))
            if max_abs <= 0.0:
                max_abs = 1.0
            return CenteredNorm(vcenter=0.0, halfrange=max_abs)
        return Normalize(vmin=float(np.min(finite_values)), vmax=float(np.max(finite_values)))
    if config.color_scale == "power":
        if config.power_gamma <= 0.0:
            raise ValueError(f"power_gamma must be positive, got {config.power_gamma}.")
        if config.centered_colorbar:
            max_abs = float(np.max(np.abs(finite_values)))
            if max_abs <= 0.0:
                max_abs = 1.0
            return PowerNorm(gamma=config.power_gamma, vmin=-max_abs, vmax=max_abs)
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        return PowerNorm(gamma=config.power_gamma, vmin=vmin, vmax=vmax)
    raise ValueError(
        f"Unsupported color_scale {config.color_scale!r}. Use 'linear' or 'power'."
    )


def validate_matrix_components(components: tuple[tuple[int, int], ...]) -> None:
    if len(components) == 0:
        raise ValueError("MatrixFieldPlotConfig.components must not be empty.")
    for component in components:
        if len(component) != 2 or any(index not in (0, 1) for index in component):
            raise ValueError(f"Unsupported matrix component {component}.")


def matrix_plot_columns(config: MatrixFieldPlotConfig) -> int:
    if config.columns is not None:
        if config.columns < 1:
            raise ValueError(f"columns must be positive, got {config.columns}.")
        return config.columns
    return 2 if len(config.components) == 4 else len(config.components)


def build_matrix_field_heatmaps(
    matrix_values: np.ndarray,
    L_values: np.ndarray,
    parameter_values: np.ndarray,
    config: MatrixFieldPlotConfig,
    current_no_flip_mask: np.ndarray | None = None,
    reconnection_contours: ReconnectionContourConfig = CONFIG.heatmap.reconnection_contours,
    delta_energy: np.ndarray | None = None,
    reference_energy: float | None = None,
    reference_energy_contour: ReferenceEnergyContourConfig | None = None,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
) -> plt.Figure:
    validate_parameterization(parameterization)
    validate_matrix_components(config.components)
    if matrix_values.shape != (len(parameter_values), len(L_values), 2, 2):
        raise ValueError(
            "matrix_values must have shape "
            f"({len(parameter_values)}, {len(L_values)}, 2, 2), "
            f"got {matrix_values.shape}."
        )
    if current_no_flip_mask is not None:
        matrix_values = mask_matrix_field_to_region(matrix_values, current_no_flip_mask)
    norm = matrix_field_color_norm(matrix_values, config)
    columns = matrix_plot_columns(config)
    rows = int(np.ceil(len(config.components) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.6 * columns, 3.5 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    reference_contour_drawn = False
    for component_index, (i, j) in enumerate(config.components):
        row = component_index // columns
        col = component_index % columns
        ax = axes[row, col]
        image = ax.imshow(
            matrix_values[..., i, j],
            origin="lower",
            extent=(
                L_values[0],
                L_values[-1],
                parameter_values[0],
                parameter_values[-1],
            ),
            aspect="auto",
            cmap=config.cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.axvline(REFERENCE_L, color="0.2", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axhline(
            REFERENCE_PARAMETER,
            color="0.2",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
        )
        if current_no_flip_mask is not None and has_region_boundary(
            current_no_flip_mask
        ):
            ax.contour(
                L_values,
                parameter_values,
                current_no_flip_mask.astype(float),
                levels=[reconnection_contours.level],
                colors=reconnection_contours.current_color,
                linewidths=reconnection_contours.linewidth,
                alpha=1.0,
                zorder=14,
            )
        if (
            delta_energy is not None
            and reference_energy is not None
            and reference_energy_contour is not None
        ):
            reference_contour_drawn = (
                add_reference_energy_contour(
                    ax,
                    L_values,
                    parameter_values,
                    delta_energy,
                    reference_energy,
                    reference_energy_contour,
                )
                or reference_contour_drawn
            )
        if config.element_pair_grid.draw:
            plot_heatmap_element_pair_grid(
                ax,
                L_range=(L_values[0], L_values[-1]),
                parameter_range=(parameter_values[0], parameter_values[-1]),
                config=config.element_pair_grid,
                triangles=flip_mode.current_triangles,
                active_diagonal=flip_mode.current_diagonal,
                parameterization=parameterization,
                flip_mode=flip_mode,
            )
        title = (
            rf"${config.component_symbol}_{{{i + 1}{j + 1}}}$"
            if config.component_symbol
            else f"[{i}, {j}]"
        )
        ax.set_title(title)
        if row == rows - 1:
            ax.set_xlabel(r"$L$")
        if col == 0:
            ax.set_ylabel(parameter_axis_label(parameterization))

    for empty_index in range(len(config.components), rows * columns):
        row = empty_index // columns
        col = empty_index % columns
        axes[row, col].set_visible(False)

    if image is None:
        raise RuntimeError("Expected at least one matrix component to plot.")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist())
    colorbar.set_label(config.colorbar_label)
    if current_no_flip_mask is not None or reference_contour_drawn:
        legend_handles = []
        if current_no_flip_mask is not None and has_region_boundary(current_no_flip_mask):
            legend_handles.append(
                Line2D(
                    [0, 1],
                    [0, 0],
                    color=reconnection_contours.current_color,
                    linewidth=reconnection_contours.linewidth,
                    label="current no-flip region",
                )
            )
        if reference_contour_drawn and reference_energy_contour is not None:
            legend_handles.append(reference_energy_contour_handle(reference_energy_contour))
        if legend_handles:
            legend = axes[0, 0].legend(
                handles=legend_handles,
                loc="upper right",
                framealpha=0.9,
            )
            legend.set_zorder(100)
    if not remove_figure_title:
        fig.suptitle(
            f"{config.title} ({config.resolution}x{config.resolution}, "
            f"{config.color_scale}, {parameterization.mode})"
        )
    return fig


def build_cauchy_stress_difference_heatmaps(
    config: MatrixFieldPlotConfig = CONFIG.cauchy_stress_difference,
    material: MaterialConfig = CONFIG.material,
    reference_energy_contour: ReferenceEnergyContourConfig = CONFIG.reference_energy_contour,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    L_values, parameter_values = sampled_parameter_values(
        config.resolution,
        config.L_range,
        config.parameter_range,
        parameterization=parameterization,
    )
    matrix_values = cauchy_stress_difference_grid(
        L_values,
        parameter_values,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    delta_energy, _, _ = edge_flip_energy_difference_grid(
        L_values,
        parameter_values,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    current_reconnection_masks = reconnection_condition_masks(
        L_values,
        parameter_values,
        triangles=flip_mode.current_triangles,
        shared_edge=flip_mode.current_diagonal,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    current_no_flip_mask = current_reconnection_masks["inside"]
    reference_energy = reference_simple_shear_energy(
        material,
        reference_energy_contour,
    )
    fig = build_matrix_field_heatmaps(
        matrix_values,
        L_values,
        parameter_values,
        config,
        current_no_flip_mask=current_no_flip_mask,
        reconnection_contours=CONFIG.heatmap.reconnection_contours,
        delta_energy=delta_energy,
        reference_energy=reference_energy,
        reference_energy_contour=reference_energy_contour,
        parameterization=parameterization,
        flip_mode=flip_mode,
        remove_figure_title=remove_figure_title,
    )
    return fig, {
        "L_values": L_values,
        "parameter_values": parameter_values,
        "matrix_values": matrix_values,
        "delta_energy": delta_energy,
        "inside_current_reconnection_zone": current_no_flip_mask,
        "reference_energy": reference_energy,
    }


def build_first_element_G_heatmaps(
    config: MatrixFieldPlotConfig = CONFIG.first_element_G,
    material: MaterialConfig = CONFIG.material,
    reference_energy_contour: ReferenceEnergyContourConfig = CONFIG.reference_energy_contour,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    L_values, parameter_values = sampled_parameter_values(
        config.resolution,
        config.L_range,
        config.parameter_range,
        parameterization=parameterization,
    )
    matrix_values = first_element_G_grid(
        L_values,
        parameter_values,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    delta_energy, _, _ = edge_flip_energy_difference_grid(
        L_values,
        parameter_values,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    current_reconnection_masks = reconnection_condition_masks(
        L_values,
        parameter_values,
        triangles=flip_mode.current_triangles,
        shared_edge=flip_mode.current_diagonal,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    current_no_flip_mask = current_reconnection_masks["inside"]
    reference_energy = reference_simple_shear_energy(
        material,
        reference_energy_contour,
    )
    fig = build_matrix_field_heatmaps(
        matrix_values,
        L_values,
        parameter_values,
        config,
        current_no_flip_mask=current_no_flip_mask,
        reconnection_contours=CONFIG.heatmap.reconnection_contours,
        delta_energy=delta_energy,
        reference_energy=reference_energy,
        reference_energy_contour=reference_energy_contour,
        parameterization=parameterization,
        flip_mode=flip_mode,
        remove_figure_title=remove_figure_title,
    )
    return fig, {
        "L_values": L_values,
        "parameter_values": parameter_values,
        "matrix_values": matrix_values,
        "delta_energy": delta_energy,
        "inside_current_reconnection_zone": current_no_flip_mask,
        "reference_energy": reference_energy,
    }


def build_flip_energy_heatmap(
    config: HeatmapConfig = CONFIG.heatmap,
    material: MaterialConfig = CONFIG.material,
    reference_energy_contour: ReferenceEnergyContourConfig = CONFIG.reference_energy_contour,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    content: str = HEATMAP_COMBINED,
    g_vector_choice: str = G_VECTOR_CHOICE_SHORTEST,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    validate_heatmap_content(content)
    validate_parameterization(parameterization)
    validate_g_vector_choice(g_vector_choice)
    L_values, parameter_values = sampled_parameter_values(
        config.resolution,
        config.L_range,
        config.parameter_range,
        parameterization=parameterization,
    )
    delta_energy, current_energy, flipped_energy = edge_flip_energy_difference_grid(
        L_values,
        parameter_values,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    abs_delta_energy = np.abs(delta_energy)
    reference_energy = reference_simple_shear_energy(
        material,
        reference_energy_contour,
    )
    show_energy = content in (HEATMAP_COMBINED, HEATMAP_ENERGY_ONLY)
    show_regions = content in (HEATMAP_COMBINED, HEATMAP_REGIONS_ONLY)
    current_reconnection_masks = reconnection_condition_masks(
        L_values,
        parameter_values,
        triangles=flip_mode.current_triangles,
        shared_edge=flip_mode.current_diagonal,
        parameterization=parameterization,
        flip_mode=flip_mode,
        g_vector_choice=g_vector_choice,
    )
    current_reconnection_zone_mask = current_reconnection_masks["inside"]
    visible_delta_energy = (
        mask_scalar_field_to_region(delta_energy, current_reconnection_zone_mask)
        if show_energy
        else delta_energy
    )
    norm = heatmap_color_norm(visible_delta_energy, config=config) if show_energy else None
    flipped_reconnection_zone_mask = insideReconnectionZone(
        L_values,
        parameter_values,
        triangles=flip_mode.flipped_triangles,
        shared_edge=flip_mode.flipped_diagonal,
        parameterization=parameterization,
        flip_mode=flip_mode,
        g_vector_choice=g_vector_choice,
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
            extent=(
                L_values[0],
                L_values[-1],
                parameter_values[0],
                parameter_values[-1],
            ),
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=3.0,
            interpolation="nearest",
        )
    elif show_energy:
        image = ax.imshow(
            visible_delta_energy,
            origin="lower",
            extent=(
                L_values[0],
                L_values[-1],
                parameter_values[0],
                parameter_values[-1],
            ),
            aspect="auto",
            cmap=config.cmap,
            norm=norm,
            interpolation="nearest",
        )
    else:
        ax.set_xlim(L_values[0], L_values[-1])
        ax.set_ylim(parameter_values[0], parameter_values[-1])
        ax.set_facecolor("white")

    if show_regions and not contours.debug_only:
        if (
            not show_energy
            and contours.draw_current
            and np.any(current_reconnection_zone_mask)
        ):
            ax.contourf(
                L_values,
                parameter_values,
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
                        parameter_values,
                        reason_mask.astype(float),
                        levels=[contours.level, 1.5],
                        colors=[color],
                        alpha=contours.failure_fill_alpha,
                        zorder=zorder,
                    )
        if contours.draw_flipped and np.any(flipped_reconnection_zone_mask):
            ax.contourf(
                L_values,
                parameter_values,
                flipped_reconnection_zone_mask.astype(float),
                levels=[contours.level, 1.5],
                colors=[contours.flipped_color],
                alpha=contours.fill_alpha,
                zorder=7,
            )
        if contours.draw_failure_reasons:
            add_hatched_region_overlay(
                ax,
                L_values,
                parameter_values,
                current_reconnection_masks["g12_too_large"],
                contours.g12_too_large_color,
                contours.g12_too_large_hatch,
                contours.level,
                zorder=13,
            )

    if overlay.draw:
        highlighted_g_vector_choice = (
            g_vector_choice if content == HEATMAP_REGIONS_ONLY else None
        )
        plot_heatmap_element_pair_grid(
            ax,
            L_range=(L_values[0], L_values[-1]),
            parameter_range=(parameter_values[0], parameter_values[-1]),
            config=overlay,
            triangles=flip_mode.current_triangles,
            active_diagonal=flip_mode.current_diagonal,
            parameterization=parameterization,
            flip_mode=flip_mode,
            g_vector_choice=highlighted_g_vector_choice,
        )

    legend_handles = []
    reference_contour_drawn = add_reference_energy_contour(
        ax,
        L_values,
        parameter_values,
        delta_energy,
        reference_energy,
        reference_energy_contour,
    )
    if reference_contour_drawn:
        legend_handles.append(reference_energy_contour_handle(reference_energy_contour))

    has_current_reconnection_boundary = has_region_boundary(
        current_reconnection_zone_mask
    )
    if (
        (show_regions or show_energy)
        and contours.draw_current
        and has_current_reconnection_boundary
    ):
        ax.contour(
            L_values,
            parameter_values,
            current_reconnection_zone_mask.astype(float),
            levels=[contours.level],
            colors=contours.current_color,
            linewidths=contours.linewidth,
            alpha=1.0,
            zorder=14,
        )
        if show_energy:
            legend_handles.append(
                Line2D(
                    [0, 1],
                    [0, 0],
                    color=contours.current_color,
                    linewidth=contours.linewidth,
                    label="current no-flip region",
                )
            )
        else:
            legend_handles.append(
                region_legend_patch(
                    contours.current_color,
                    contours.fill_alpha,
                    "current no-flip region: both triangles inside",
                )
            )

    if show_regions and contours.draw_failure_reasons:
        if content == HEATMAP_REGIONS_ONLY and overlay.draw:
            legend_handles.append(
                Line2D(
                    [0, 1],
                    [0, 0],
                    color=overlay.g_vector_color,
                    linewidth=overlay.g_vector_linewidth,
                    marker=">",
                    markersize=7,
                    markerfacecolor=overlay.g_vector_color,
                    markeredgecolor=overlay.g_vector_color,
                    label="edges used for $G$",
                )
            )
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
                    parameter_values,
                    reason_mask.astype(float),
                    levels=[contours.level],
                    colors=color,
                    linewidths=contours.linewidth,
                    alpha=1.0,
                    zorder=12,
                )
            if reason_is_present or contours.show_empty_failure_reasons:
                shown_label = label if reason_is_present else f"{label} (not present)"
                hatch = (
                    contours.g12_too_large_hatch
                    if reason_key == "g12_too_large"
                    else None
                )
                legend_handles.append(
                    region_legend_patch(
                        color,
                        contours.failure_fill_alpha,
                        shown_label,
                        hatch=hatch,
                    )
                )

    has_flipped_reconnection_boundary = has_region_boundary(
        flipped_reconnection_zone_mask
    )
    if show_regions and contours.draw_flipped and has_flipped_reconnection_boundary:
        ax.contour(
            L_values,
            parameter_values,
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
    ax.axhline(
        REFERENCE_PARAMETER,
        color="0.2",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
    )
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(parameter_axis_label(parameterization))
    if not remove_figure_title:
        if contours.debug_only:
            ax.set_title(
                rf"Reconnection-zone mask ({config.resolution}x{config.resolution})"
            )
        elif content == HEATMAP_REGIONS_ONLY:
            ax.set_title(
                "Reconnection regions "
                rf"({g_vector_choice_label(g_vector_choice, flip_mode.current_triangles)}, "
                rf"{config.resolution}x{config.resolution}, {parameterization.mode})"
            )
        else:
            ax.set_title(
                rf"$E_{{\mathrm{{flipped}}}} - E_{{\mathrm{{current}}}}$ "
                rf"({config.resolution}x{config.resolution}, {config.color_scale}, "
                rf"{parameterization.mode})"
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
        "parameter_values": parameter_values,
        "delta_energy": delta_energy,
        "visible_delta_energy": visible_delta_energy,
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
        "g_vector_choice": g_vector_choice,
        "parameterization_mode": parameterization.mode,
        "reference_energy": reference_energy,
    }
    return fig, data


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")


def flip_mode_output_path(path: Path, flip_mode: FlipMode) -> Path:
    return path.with_name(f"{path.stem}_{flip_mode.name}{path.suffix}")


def heatmap_content_output_path(path: Path, output_tag: str) -> Path:
    if output_tag == HEATMAP_COMBINED:
        return path
    return path.with_name(f"{path.stem}_{output_tag}{path.suffix}")


def build_heatmap_variants(
    config: HeatmapConfig,
    material: MaterialConfig,
    reference_energy_contour: ReferenceEnergyContourConfig,
    parameterization: ParameterizationConfig,
    flip_mode: FlipMode,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
) -> tuple[list[tuple[str, plt.Figure]], dict[str, dict[str, np.ndarray]]]:
    if len(config.contents) == 0:
        raise ValueError("HeatmapConfig.contents must contain at least one variant.")

    figures = []
    data_by_content = {}
    for content in config.contents:
        fig, data = build_flip_energy_heatmap(
            config=config,
            material=material,
            reference_energy_contour=reference_energy_contour,
            parameterization=parameterization,
            flip_mode=flip_mode,
            content=content,
            remove_figure_title=remove_figure_title,
        )
        figures.append((content, fig))
        data_by_content[content] = data

    for g_vector_choice in config.extra_region_g_vector_choices:
        validate_g_vector_choice(g_vector_choice)
        output_tag = f"{HEATMAP_REGIONS_ONLY}_{g_vector_choice}"
        fig, data = build_flip_energy_heatmap(
            config=config,
            material=material,
            reference_energy_contour=reference_energy_contour,
            parameterization=parameterization,
            flip_mode=flip_mode,
            content=HEATMAP_REGIONS_ONLY,
            g_vector_choice=g_vector_choice,
            remove_figure_title=remove_figure_title,
        )
        figures.append((output_tag, fig))
        data_by_content[output_tag] = data
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
    print(f"Parameterization: {config.parameterization.mode}")
    fig = build_parameterization_grid(
        config=config.pair_grid,
        material=config.material,
        parameterization=config.parameterization,
        flip_mode=flip_mode,
        remove_figure_title=config.remove_figure_titles,
    )
    flipped_fig = build_flipped_parameterization_grid(
        config=config.pair_grid,
        material=config.material,
        parameterization=config.parameterization,
        flip_mode=flip_mode,
        remove_figure_title=config.remove_figure_titles,
    )
    heatmap_figures, heatmap_data_by_content = build_heatmap_variants(
        config=config.heatmap,
        material=config.material,
        reference_energy_contour=config.reference_energy_contour,
        parameterization=config.parameterization,
        flip_mode=flip_mode,
        remove_figure_title=config.remove_figure_titles,
    )
    if config.plot_focused_heatmap:
        focused_heatmap_figures, focused_heatmap_data_by_content = build_heatmap_variants(
            config=config.focused_heatmap,
            material=config.material,
            reference_energy_contour=config.reference_energy_contour,
            parameterization=config.parameterization,
            flip_mode=flip_mode,
            remove_figure_title=config.remove_figure_titles,
        )
    else:
        focused_heatmap_figures = []
        focused_heatmap_data_by_content = {}
    cauchy_fig, cauchy_data = build_cauchy_stress_difference_heatmaps(
        config=config.cauchy_stress_difference,
        material=config.material,
        reference_energy_contour=config.reference_energy_contour,
        parameterization=config.parameterization,
        flip_mode=flip_mode,
        remove_figure_title=config.remove_figure_titles,
    )
    if config.plot_first_element_G:
        first_element_G_fig, first_element_G_data = build_first_element_G_heatmaps(
            config=config.first_element_G,
            material=config.material,
            reference_energy_contour=config.reference_energy_contour,
            parameterization=config.parameterization,
            flip_mode=flip_mode,
            remove_figure_title=config.remove_figure_titles,
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
    if config.plot_focused_heatmap:
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
    delta_energy = heatmap_data["delta_energy"]
    print(
        "Delta E range: "
        f"{float(np.nanmin(delta_energy)):.6e} to {float(np.nanmax(delta_energy)):.6e}"
    )
    if config.plot_focused_heatmap:
        focused_heatmap_data = representative_heatmap_data(
            focused_heatmap_data_by_content
        )
        focused_delta_energy = focused_heatmap_data["delta_energy"]
        print(
            "Focused Delta E range: "
            f"{float(np.nanmin(focused_delta_energy)):.6e} "
            f"to {float(np.nanmax(focused_delta_energy)):.6e}"
        )
    cauchy_values = cauchy_data["matrix_values"]
    cauchy_mask = cauchy_data["inside_current_reconnection_zone"]
    visible_cauchy_values = mask_matrix_field_to_region(cauchy_values, cauchy_mask)
    reference_energy = heatmap_data["reference_energy"]
    print(
        "Reference simple-shear energy: "
        f"{reference_energy:.6e} "
        f"(gamma={config.reference_energy_contour.shear:g})"
    )
    print(
        "Visible Cauchy stress difference range: "
        f"{float(np.nanmin(visible_cauchy_values)):.6e} "
        f"to {float(np.nanmax(visible_cauchy_values)):.6e}"
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
    if config.plot_focused_heatmap:
        focused_current_reconnection_zone_mask = focused_heatmap_data[
            "inside_current_reconnection_zone"
        ]
        focused_flipped_reconnection_zone_mask = focused_heatmap_data[
            "inside_flipped_reconnection_zone"
        ]
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
    validate_parameterization(config.parameterization)

    figures = []
    for flip_mode in config.flip_modes:
        figures.extend(build_and_save_for_flip_mode(config, flip_mode))

    if config.show:
        plt.show()
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
