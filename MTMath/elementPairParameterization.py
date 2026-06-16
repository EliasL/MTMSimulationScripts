"""Two-element edge-flip parameterization plots.

The T23 parameterization follows Sylvain's notation:

    x2 = (0, 0), x3 = (L, 0),
    x1 = (s L, -1 / L), x4 = (t L, 1 / L).

The full T23-controlled geometry is described by (L, u, v), where
u = s + t - 1 and v = s - t. The plotting config chooses a two-parameter
slice: either the symmetric slice v=0, or the antisymmetric slice u=0.

The figures use w = L - sqrt(2) as the horizontal control parameter. Since
x2 is fixed at the origin and x3=(L, 0), w is the direct change in shared-edge
length away from the reference state. The PDF Section 4 formulae are still
evaluated with the physical L(w)=sqrt(2)+w.

By default main() runs both the symmetric and antisymmetric two-parameter
slices for the first flip, where T23 is current and has det(F)=1. The second
flip can be enabled explicitly; then T14 is treated as current after the first
flip, its elements do not generally have det(F)=1, and that is intentional.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject

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
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import FancyArrowPatch, Patch, Polygon
from tqdm import tqdm

from MTMath.energyFunction import ContiEnergy

USE_LATEX_RENDERING = True
plt.rcParams["text.usetex"] = USE_LATEX_RENDERING
plt.rcParams["font.family"] = "serif" if USE_LATEX_RENDERING else "DejaVu Serif"

T23_TRIANGLES = ((1, 2, 3), (2, 3, 4))
T14_TRIANGLES = ((1, 2, 4), (1, 4, 3))
DIAGONAL_23 = (2, 3)
DIAGONAL_14 = (1, 4)
REFERENCE_L = float(np.sqrt(2.0))
REFERENCE_w = 0.0
REFERENCE_U = 0.0
REFERENCE_V = 0.0
REFERENCE_PARAMETER = 0.0
ROOT_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT_DIR / "Plots"
HEATMAP_COMBINED = "combined"
HEATMAP_ENERGY_ONLY = "energy_only"
HEATMAP_REGIONS_ONLY = "regions_only"
HEATMAP_CONTENTS = (HEATMAP_COMBINED, HEATMAP_ENERGY_ONLY, HEATMAP_REGIONS_ONLY)
DEFAULT_HEATMAP_CONTENTS = (HEATMAP_COMBINED, HEATMAP_ENERGY_ONLY)
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
STRESS_MEASURE_NORMAL_DIFFERENCE_HALF = "normal_difference_half"
STRESS_MEASURE_VON_MISES = "von_mises"
STRESS_MEASURES = (
    STRESS_MEASURE_NORMAL_DIFFERENCE_HALF,
    STRESS_MEASURE_VON_MISES,
)
STRESS_SELECTION_AVERAGE = "average"
STRESS_SELECTION_ELEMENT_1 = "element1"
STRESS_SELECTION_ELEMENT_2 = "element2"
STRESS_SELECTIONS = (
    STRESS_SELECTION_AVERAGE,
    STRESS_SELECTION_ELEMENT_1,
    STRESS_SELECTION_ELEMENT_2,
)
ELEMENT_STRESS_MEASURE_SHEAR = "shear"
ELEMENT_STRESS_MEASURE_VON_MISES_AVERAGE = "von_mises_average"
ELEMENT_STRESS_MEASURES = (
    ELEMENT_STRESS_MEASURE_SHEAR,
    ELEMENT_STRESS_MEASURE_VON_MISES_AVERAGE,
)
MATRIX_FIELD_COMPONENT = "component"
MATRIX_FIELD_STRESS_MEASURE = "stress_measure"
PROGRESS_GRID_SIZE_THRESHOLD = 400
MATRIX_PANEL_FIGSIZE = (3.6, 3.5)
STANDALONE_FIGSIZE = (MATRIX_PANEL_FIGSIZE[0] + 1.0, MATRIX_PANEL_FIGSIZE[1])
REGION_FILL_ZORDER = 3
REGION_OUTLINE_ZORDER = 5
REGION_HATCH_ZORDER = 6
CURRENT_REGION_OUTLINE_ZORDER = 7
DELAUNAY_CONTOUR_ZORDER = 7.5
ELEMENT_PAIR_GRID_ZORDER = 8

# =============================================================================
# User-editable plotting defaults
# =============================================================================
# Edit these values directly when running the script from an IDE or with:
#     python MTMath/elementPairParameterization.py

RESOLUTION=100
# Main heatmap window. w=L-sqrt(2) directly controls the shared-edge length;
# w=(-0.60, 0.60) corresponds to L=(0.814..., 2.014...).
# The parameter axis is u for symmetric mode and v for antisymmetric mode.
MAIN_VIEW_w_RANGE = (-0.60, 0.60)
MAIN_VIEW_PARAMETER_RANGE = (0.0, 0.5)

# Keep False to color the full parameterization domain. Set True to only color
# values inside the current no-flip region while keeping the region outline.
MASK_COLOR_OUTSIDE_NO_FLIP_REGION = False


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
DEFAULT_PLOTTED_FLIP_MODES = (FIRST_FLIP_T23_TO_T14,)
DEFAULT_STRESS_SELECTIONS = (
    STRESS_SELECTION_AVERAGE,
    STRESS_SELECTION_ELEMENT_1,
    STRESS_SELECTION_ELEMENT_2,
)
DEFAULT_PARAMETERIZATIONS = (
    ParameterizationConfig(PARAMETERIZATION_SYMMETRIC),
    ParameterizationConfig(PARAMETERIZATION_ANTISYMMETRIC),
)


@dataclass(frozen=True)
class PairGridConfig:
    w_values: tuple[float, ...]
    parameter_values: tuple[float, ...]
    output_path: Path
    flipped_output_path: Path


@dataclass(frozen=True)
class ReconnectionContourConfig:
    draw_current: bool = True
    draw_flipped: bool = False
    draw_failure_reasons: bool = True
    draw_delaunay: bool = True
    show_empty_failure_reasons: bool = True
    debug_only: bool = False
    level: float = 0.5
    current_color: str = "cyan"
    flipped_color: str = "lime"
    delaunay_color: str = "green"
    delaunay_linewidth: float = 1.6
    delaunay_linestyle: str = "--"
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
    size: int = 3
    padding_fraction: float = 0.16
    scale_fraction: float = 0.075
    alpha: float = 0.55
    linewidth: float = 0.8
    g_vector_color: str = "gold"
    g_vector_alpha: float = 0.9
    g_vector_linewidth: float = 1.8
    g_vector_arrow_scale: float = 7.0


@dataclass(frozen=True)
class ReferenceContourConfig:
    draw: bool = True
    shear: float = 0.5
    color: str = "black"
    linestyle: str = "--"
    linewidth: float = 1.4
    zorder: int = REGION_OUTLINE_ZORDER
    use_absolute_delta_energy: bool = True
    label: str | None = None


@dataclass(frozen=True)
class HeatmapConfig:
    output_path: Path
    resolution: int
    w_range: tuple[float, float]
    parameter_range: tuple[float, float]
    cmap: str = "coolwarm"
    color_scale: str = "linear"  # "linear" or "power"
    power_gamma: float = 0.5
    centered_colorbar: bool = False
    color_limits_from_delaunay_switch_region: bool = True
    reconnection_contours: ReconnectionContourConfig = ReconnectionContourConfig()
    element_pair_grid: HeatmapElementPairGridConfig = HeatmapElementPairGridConfig()
    contents: tuple[str, ...] = DEFAULT_HEATMAP_CONTENTS
    extra_region_g_vector_choices: tuple[str, ...] = ()
    mask_color_outside_no_flip_region: bool = MASK_COLOR_OUTSIDE_NO_FLIP_REGION


@dataclass(frozen=True)
class MatrixFieldPlotConfig:
    output_path: Path
    resolution: int
    w_range: tuple[float, float]
    parameter_range: tuple[float, float]
    title: str
    colorbar_label: str
    component_symbol: str = ""
    cmap: str = "coolwarm"
    color_scale: str = "linear"  # "linear" or "power"
    power_gamma: float = 0.5
    centered_colorbar: bool = False
    color_limits_from_delaunay_switch_region: bool = True
    components: tuple[tuple[int, int], ...] = (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )
    fields: tuple[tuple[str, tuple[int, int] | str], ...] | None = None
    columns: int | None = None
    element_pair_grid: HeatmapElementPairGridConfig = HeatmapElementPairGridConfig()
    mask_color_outside_no_flip_region: bool = MASK_COLOR_OUTSIDE_NO_FLIP_REGION


@dataclass(frozen=True)
class ScalarFieldPlotConfig:
    output_path: Path
    resolution: int
    w_range: tuple[float, float]
    parameter_range: tuple[float, float]
    title: str
    colorbar_label: str
    cmap: str = "coolwarm"
    color_scale: str = "linear"  # "linear" or "power"
    power_gamma: float = 0.5
    centered_colorbar: bool = False
    color_limits_from_delaunay_switch_region: bool = True
    element_pair_grid: HeatmapElementPairGridConfig = HeatmapElementPairGridConfig()
    mask_color_outside_no_flip_region: bool = MASK_COLOR_OUTSIDE_NO_FLIP_REGION


@dataclass(frozen=True)
class PlotConfig:
    flip_mode: FlipMode
    parameterization: ParameterizationConfig
    material: MaterialConfig
    reference_contour: ReferenceContourConfig
    pair_grid: PairGridConfig
    heatmap: HeatmapConfig
    focused_heatmap: HeatmapConfig
    cauchy_stress_difference: MatrixFieldPlotConfig
    cauchy_stress_measures: MatrixFieldPlotConfig
    element_shear_stress: ScalarFieldPlotConfig
    element_von_mises_stress: ScalarFieldPlotConfig
    first_element_G: MatrixFieldPlotConfig
    flip_modes: tuple[FlipMode, ...] = DEFAULT_PLOTTED_FLIP_MODES
    parameterizations: tuple[ParameterizationConfig, ...] = DEFAULT_PARAMETERIZATIONS
    stress_selections: tuple[str, ...] = DEFAULT_STRESS_SELECTIONS
    assert_element_stress_component_signs: bool = False
    combine_element_pdfs: bool = True
    remove_figure_titles: bool = True
    plot_focused_heatmap: bool = False
    plot_first_element_G: bool = False
    show: bool = False


CONFIG = PlotConfig(
    flip_mode=FIRST_FLIP_T23_TO_T14,
    # Active slice used by helper functions. main() replaces this while looping
    # over parameterizations below.
    parameterization=ParameterizationConfig(),
    material=MaterialConfig(),
    reference_contour=ReferenceContourConfig(),
    pair_grid=PairGridConfig(
        w_values=(-0.60, REFERENCE_w, 0.60),
        parameter_values=(0.8, 0.4, 0.0),
        output_path=PLOTS_DIR / "two_element_parameterization_grid.pdf",
        flipped_output_path=PLOTS_DIR / "two_element_parameterization_flipped_grid.pdf",
    ),
    heatmap=HeatmapConfig(
        output_path=PLOTS_DIR / "two_element_parameterization_flip_energy_heatmap.pdf",
        resolution=RESOLUTION,
        w_range=MAIN_VIEW_w_RANGE,
        parameter_range=MAIN_VIEW_PARAMETER_RANGE,
    ),
    focused_heatmap=HeatmapConfig(
        output_path=PLOTS_DIR
        / "two_element_parameterization_flip_energy_heatmap_focused.pdf",
        resolution=RESOLUTION,
        w_range=(REFERENCE_w, 2.0 - REFERENCE_L),
        parameter_range=(0.0, 0.3),
    ),
    cauchy_stress_difference=MatrixFieldPlotConfig(
        output_path=PLOTS_DIR
        / "two_element_parameterization_cauchy_stress_difference.pdf",
        resolution=RESOLUTION,
        w_range=MAIN_VIEW_w_RANGE,
        parameter_range=MAIN_VIEW_PARAMETER_RANGE,
        title=(
            r"$\Delta \langle \sigma \rangle = "
            r"\langle \sigma_{\mathrm{flipped}}"
            r" - \sigma_{\mathrm{current}} \rangle$"
        ),
        colorbar_label=r"$\Delta \left\langle \sigma \right\rangle$",
        component_symbol=r"\Delta\sigma",
        fields=(
            (MATRIX_FIELD_COMPONENT, (0, 0)),
            (MATRIX_FIELD_COMPONENT, (1, 1)),
            (MATRIX_FIELD_COMPONENT, (0, 1)),
        ),
        columns=3,
    ),
    cauchy_stress_measures=MatrixFieldPlotConfig(
        output_path=PLOTS_DIR
        / "two_element_parameterization_cauchy_stress_measures.pdf",
        resolution=RESOLUTION,
        w_range=MAIN_VIEW_w_RANGE,
        parameter_range=MAIN_VIEW_PARAMETER_RANGE,
        title="Cauchy stress difference measures",
        colorbar_label=r"$\Delta$ stress measure",
        fields=(
            (MATRIX_FIELD_STRESS_MEASURE, STRESS_MEASURE_NORMAL_DIFFERENCE_HALF),
            (MATRIX_FIELD_STRESS_MEASURE, STRESS_MEASURE_VON_MISES),
        ),
        columns=2,
    ),
    element_shear_stress=ScalarFieldPlotConfig(
        output_path=PLOTS_DIR
        / "two_element_parameterization_element_shear_stress_difference.pdf",
        resolution=RESOLUTION,
        w_range=MAIN_VIEW_w_RANGE,
        parameter_range=MAIN_VIEW_PARAMETER_RANGE,
        title=r"Element-averaged shear stress difference",
        colorbar_label=r"$\Delta \left\langle \sigma_{12} \right\rangle$",
    ),
    element_von_mises_stress=ScalarFieldPlotConfig(
        output_path=PLOTS_DIR
        / "two_element_parameterization_element_von_mises_stress_difference.pdf",
        resolution=RESOLUTION,
        w_range=MAIN_VIEW_w_RANGE,
        parameter_range=MAIN_VIEW_PARAMETER_RANGE,
        title=r"Element-averaged von Mises stress difference",
        colorbar_label=r"$\Delta \left\langle \sigma_{\mathrm{vM}} \right\rangle$",
    ),
    first_element_G=MatrixFieldPlotConfig(
        output_path=PLOTS_DIR / "two_element_parameterization_first_element_G.pdf",
        resolution=RESOLUTION,
        w_range=MAIN_VIEW_w_RANGE,
        parameter_range=(0.0, 0.9),
        title="First current element G",
        colorbar_label="G",
        component_symbol="G",
        cmap="viridis",
        centered_colorbar=False,
    ),
    flip_modes=DEFAULT_PLOTTED_FLIP_MODES,
    # Default to both the symmetric slice v=0 from PDF Section 6.1 and the
    # antisymmetric affine slice u=0 from PDF Section 5.
    parameterizations=DEFAULT_PARAMETERIZATIONS,
    stress_selections=DEFAULT_STRESS_SELECTIONS,
    assert_element_stress_component_signs=False,
    combine_element_pdfs=True,
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


def validate_parameterizations(
    parameterizations: tuple[ParameterizationConfig, ...],
) -> None:
    if len(parameterizations) == 0:
        raise ValueError("config.parameterizations must contain at least one entry.")
    for parameterization in parameterizations:
        validate_parameterization(parameterization)


def material_kwargs(material: MaterialConfig) -> dict[str, float]:
    return {"beta": material.beta, "K": material.K, "noise": material.noise}


def L_from_w(w: np.ndarray | float) -> np.ndarray:
    """Convert plotted displacement w to physical L for PDF Section 4 geometry."""
    return REFERENCE_L + np.asarray(w, dtype=float)


def w_from_L(L: np.ndarray | float) -> np.ndarray:
    """Convert physical L to plotted displacement w=L-sqrt(2)."""
    return np.asarray(L, dtype=float) - REFERENCE_L


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


def t23_parameterized_vertices_from_w_parameter(
    w: float,
    parameter_value: float,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> dict[int, np.ndarray]:
    u, v = parameter_value_to_uv(parameter_value, parameterization)
    L = float(L_from_w(w))
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
    w_values: tuple[float, ...] | list[float] | np.ndarray,
    parameter_values: tuple[float, ...] | list[float] | np.ndarray,
    triangles: tuple[tuple[int, int, int], ...] | None = None,
    shared_edge: tuple[int, int] | None = None,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    g_vector_choice: str = G_VECTOR_CHOICE_SHORTEST,
) -> dict[str, np.ndarray]:
    """Return C++ inRegion and failure-reason masks on a parameter/w grid.

    The returned array has shape (len(parameter_values), len(w_values)), matching the
    heatmap orientation used by build_flip_energy_heatmap. The inside mask is
    True when every triangle satisfies the selected local G_12 criterion, and
    when the specified shared edge is a longest edge of every triangle. G is
    computed from the configured edge-vector choice.
    """
    useDoubleTriangleRegion = False
    w_values = _as_1d_float_array(w_values, "w_values")
    parameter_values = _as_1d_float_array(parameter_values, "parameter_values")
    validate_parameterization(parameterization)
    validate_g_vector_choice(g_vector_choice)
    if triangles is None:
        triangles = flip_mode.current_triangles
    if shared_edge is None:
        shared_edge = flip_mode.current_diagonal
    shape = (len(parameter_values), len(w_values))
    inside_mask = np.zeros(shape, dtype=bool)
    g12_negative_mask = np.zeros(shape, dtype=bool)
    g12_too_large_mask = np.zeros(shape, dtype=bool)
    shared_edge_not_longest_mask = np.zeros(shape, dtype=bool)

    for row, col in _progress_grid_indices(
        shape,
        w_values,
        parameter_values,
        f"Reconnection masks: {g_vector_choice}",
    ):
        parameter_value = parameter_values[row]
        w = w_values[col]
        vertices = t23_parameterized_vertices_from_w_parameter(
            float(w),
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
        if useDoubleTriangleRegion:
            g12_negative_per_triangle = np.zeros_like(G12, dtype=bool)
            g12_too_large_per_triangle = (np.abs(G12) > G11) | (
                np.abs(G12) > G22
            )
        else:
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


def other_diagonal(diagonal: tuple[int, int]) -> tuple[int, int]:
    diagonal = tuple(sorted(diagonal))
    if diagonal == DIAGONAL_23:
        return DIAGONAL_14
    if diagonal == DIAGONAL_14:
        return DIAGONAL_23
    raise ValueError(f"Unsupported diagonal {diagonal}.")


def delaunay_edges(vertices: dict[int, np.ndarray]) -> set[tuple[int, int]]:
    """Return all Delaunay edges for the four current nodes using SciPy/Qhull."""
    try:
        from scipy.spatial import Delaunay, QhullError
    except ImportError as error:
        raise RuntimeError(
            "The Delaunay region requires scipy. Install scipy or disable "
            "ReconnectionContourConfig.draw_delaunay."
        ) from error

    points = vertices_to_array(vertices)
    try:
        triangulation = Delaunay(points)
    except QhullError as error:
        raise RuntimeError(
            f"SciPy Delaunay failed for vertices {vertices}."
        ) from error

    edges = set()
    for simplex in triangulation.simplices:
        node_indices = [int(index) + 1 for index in simplex]
        a, b, c = node_indices
        edges.update(tuple(sorted(edge)) for edge in ((a, b), (a, c), (b, c)))
    return edges


def delaunay_keeps_current_diagonal(
    vertices: dict[int, np.ndarray],
    current_diagonal: tuple[int, int],
) -> bool:
    current_diagonal = tuple(sorted(current_diagonal))
    flipped_diagonal = other_diagonal(current_diagonal)
    edges = delaunay_edges(vertices)
    has_current = current_diagonal in edges
    has_flipped = flipped_diagonal in edges
    if has_current == has_flipped:
        raise RuntimeError(
            "Expected Delaunay triangulation to choose exactly one diagonal, "
            f"got has_current={has_current}, has_flipped={has_flipped}, "
            f"edges={sorted(edges)}."
        )
    return has_current


def delaunay_current_diagonal_mask(
    w_values: tuple[float, ...] | list[float] | np.ndarray,
    parameter_values: tuple[float, ...] | list[float] | np.ndarray,
    current_diagonal: tuple[int, int],
    parameterization: ParameterizationConfig = CONFIG.parameterization,
) -> np.ndarray:
    """Return True where SciPy Delaunay keeps the current diagonal."""
    w_values = _as_1d_float_array(w_values, "w_values")
    parameter_values = _as_1d_float_array(parameter_values, "parameter_values")
    validate_parameterization(parameterization)
    return _delaunay_current_diagonal_mask_cached(
        tuple(float(value) for value in w_values),
        tuple(float(value) for value in parameter_values),
        tuple(sorted(current_diagonal)),
        parameterization.mode,
    ).copy()


@lru_cache(maxsize=32)
def _delaunay_current_diagonal_mask_cached(
    w_values_tuple: tuple[float, ...],
    parameter_values_tuple: tuple[float, ...],
    current_diagonal: tuple[int, int],
    parameterization_mode: str,
) -> np.ndarray:
    w_values = np.asarray(w_values_tuple, dtype=float)
    parameter_values = np.asarray(parameter_values_tuple, dtype=float)
    parameterization = ParameterizationConfig(mode=parameterization_mode)
    shape = (len(parameter_values), len(w_values))
    mask = np.zeros(shape, dtype=bool)
    for row, col in _progress_grid_indices(
        shape,
        w_values,
        parameter_values,
        f"Delaunay mask: diagonal {current_diagonal[0]}{current_diagonal[1]}",
    ):
        w = float(w_values[col])
        parameter_value = float(parameter_values[row])
        vertices = t23_parameterized_vertices_from_w_parameter(
            w,
            parameter_value,
            parameterization=parameterization,
        )
        try:
            mask[row, col] = delaunay_keeps_current_diagonal(
                vertices,
                current_diagonal,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"Delaunay classification failed at w={w:g}, "
                f"parameter={parameter_value:g}."
            ) from error
    return mask


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


def t23_parameterized_vertices_array_from_w_parameter(
    w: np.ndarray,
    parameter_values: np.ndarray,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    u, v = parameter_values_to_uv_arrays(parameter_values, parameterization)
    L = L_from_w(w)
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
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    desc: str,
):
    indices = np.ndindex(shape)
    if (
        len(w_values) >= PROGRESS_GRID_SIZE_THRESHOLD
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
        **material_kwargs(material),
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
    config: ReferenceContourConfig,
    element_selection: str = STRESS_SELECTION_AVERAGE,
) -> float:
    """Return the area-weighted reference energy for simple shear.

    The default is the two-half-area-element value, matching the total pair
    energy. Element-specific plots use one half-area element instead.
    """
    validate_stress_selection(element_selection)
    F = horizontal_simple_shear_F(config.shear)
    energy_density = ContiEnergy.energy_from_F(
        F,
        **material_kwargs(material),
        zeroReference=True,
    )
    element_count = 2.0 if element_selection == STRESS_SELECTION_AVERAGE else 1.0
    return float(element_count * 0.5 * energy_density)


def reference_simple_shear_cauchy_stress(
    material: MaterialConfig,
    config: ReferenceContourConfig,
) -> np.ndarray:
    """Return the Cauchy stress for the same simple shear as the energy contour."""
    F = horizontal_simple_shear_F(config.shear)
    return ContiEnergy.cauchy_from_F(
        F,
        **material_kwargs(material),
    )


def pair_energy_grid(
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    return np.sum(
        element_energy_grid(
            w_values,
            parameter_values,
            triangles,
            material=material,
            parameterization=parameterization,
            flip_mode=flip_mode,
        ),
        axis=-1,
    )


def element_energy_grid(
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    F, reference_areas = pair_deformation_gradient_grid(
        w_values,
        parameter_values,
        triangles,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    energy_density = ContiEnergy.energy_from_F(
        F,
        **material_kwargs(material),
        zeroReference=True,
    )
    return reference_areas * energy_density


def select_energy_values(
    element_energy: np.ndarray,
    element_selection: str,
) -> np.ndarray:
    validate_stress_selection(element_selection)
    if element_energy.shape[-1] != 2:
        raise ValueError(
            "element_energy must have exactly two values on the last axis, "
            f"got shape {element_energy.shape}."
        )
    if element_selection == STRESS_SELECTION_AVERAGE:
        return np.sum(element_energy, axis=-1)
    return select_two_element_values(
        element_energy,
        element_selection,
        element_axis=-1,
    )


def pair_deformation_gradient_grid(
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> tuple[np.ndarray, np.ndarray]:
    w_grid, parameter_grid = np.meshgrid(w_values, parameter_values, indexing="xy")
    current_vertices = t23_parameterized_vertices_array_from_w_parameter(
        w_grid,
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


def assert_same_sign_across_two_elements(
    values: np.ndarray,
    label: str,
    tolerance: float = 1e-12,
) -> None:
    if values.shape[-1] != 2:
        raise ValueError(
            f"{label} must have exactly two element values on the last axis, "
            f"got shape {values.shape}."
        )
    opposite_sign = values[..., 0] * values[..., 1] < -tolerance
    if not np.any(opposite_sign):
        return

    first_index = tuple(int(index) for index in np.argwhere(opposite_sign)[0])
    first_pair = values[first_index]
    raise RuntimeError(
        f"{label} has opposite signs between the two elements at grid index "
        f"{first_index}: {first_pair}. This would allow cancellation when averaging."
    )


def warn_if_normal_difference_signs_differ(
    stress: np.ndarray,
    label: str,
    tolerance: float = 1e-12,
) -> None:
    if stress.shape[-3:] != (2, 2, 2):
        raise ValueError(
            "stress must end with shape (2 elements, 2, 2), "
            f"got {stress.shape}."
        )
    normal_difference_half = 0.5 * (stress[..., 0, 0] - stress[..., 1, 1])
    opposite_sign = (
        normal_difference_half[..., 0] * normal_difference_half[..., 1]
        < -tolerance
    )
    if not np.any(opposite_sign):
        return

    first_index = tuple(int(index) for index in np.argwhere(opposite_sign)[0])
    first_pair = normal_difference_half[first_index]
    warnings.warn(
        f"{label}: (sigma_11 - sigma_22) / 2 has opposite signs between "
        f"the two elements at {int(np.sum(opposite_sign))} grid points. "
        f"First at grid index {first_index}: {first_pair}. Averaging this "
        "signed quantity may hide element-level stress magnitude.",
        RuntimeWarning,
        stacklevel=2,
    )


def element_cauchy_stress_grid(
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    assert_component_signs: bool = False,
) -> np.ndarray:
    F, _ = pair_deformation_gradient_grid(
        w_values,
        parameter_values,
        triangles,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    sigma = ContiEnergy.cauchy_from_F(
        F,
        **material_kwargs(material),
    )
    if assert_component_signs:
        for i in range(2):
            for j in range(2):
                assert_same_sign_across_two_elements(
                    sigma[..., i, j],
                    rf"Cauchy stress component sigma_{i + 1}{j + 1}",
                )
    J = np.linalg.det(F)
    if np.any(J <= 0.0):
        raise RuntimeError(
            f"Expected positive element Jacobians for Cauchy stress, got min J={J.min()}."
        )
    return sigma


def pair_cauchy_stress_grid(
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    assert_component_signs: bool = False,
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> np.ndarray:
    sigma = element_cauchy_stress_grid(
        w_values,
        parameter_values,
        triangles,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
        assert_component_signs=assert_component_signs,
    )
    return select_two_element_values(
        sigma,
        stress_selection,
        element_axis=-3,
    )


def cauchy_stress_difference_grid(
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    assert_component_signs: bool = False,
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> np.ndarray:
    validate_stress_selection(stress_selection)
    current_stress = element_cauchy_stress_grid(
        w_values,
        parameter_values,
        flip_mode.current_triangles,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
        assert_component_signs=assert_component_signs,
    )
    flipped_stress = element_cauchy_stress_grid(
        w_values,
        parameter_values,
        flip_mode.flipped_triangles,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
        assert_component_signs=assert_component_signs,
    )
    if stress_selection == STRESS_SELECTION_AVERAGE:
        warn_if_normal_difference_signs_differ(
            current_stress,
            f"{flip_mode.name} current",
        )
        warn_if_normal_difference_signs_differ(
            flipped_stress,
            f"{flip_mode.name} flipped",
        )
    return select_two_element_values(
        flipped_stress - current_stress,
        stress_selection,
        element_axis=-3,
    )


def stress_measure_difference_grid(
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    measure: str,
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Delta measure = measure(flipped) - measure(current)."""
    validate_stress_measures((measure,))
    validate_stress_selection(stress_selection)
    current_stress = element_cauchy_stress_grid(
        w_values,
        parameter_values,
        flip_mode.current_triangles,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    flipped_stress = element_cauchy_stress_grid(
        w_values,
        parameter_values,
        flip_mode.flipped_triangles,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    if (
        measure == STRESS_MEASURE_NORMAL_DIFFERENCE_HALF
        and stress_selection == STRESS_SELECTION_AVERAGE
    ):
        warn_if_normal_difference_signs_differ(
            current_stress,
            f"{flip_mode.name} current",
        )
        warn_if_normal_difference_signs_differ(
            flipped_stress,
            f"{flip_mode.name} flipped",
        )
    current_values = select_two_element_values(
        stress_measure_values(current_stress, measure),
        stress_selection,
        element_axis=-1,
    )
    flipped_values = select_two_element_values(
        stress_measure_values(flipped_stress, measure),
        stress_selection,
        element_axis=-1,
    )
    return flipped_values - current_values, current_values, flipped_values


def element_stress_measure_difference_grid(
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    measure: str,
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_element_stress_measures((measure,))
    validate_stress_selection(stress_selection)
    current_stress = element_cauchy_stress_grid(
        w_values,
        parameter_values,
        flip_mode.current_triangles,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    flipped_stress = element_cauchy_stress_grid(
        w_values,
        parameter_values,
        flip_mode.flipped_triangles,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    current_values = element_stress_measure_values(
        current_stress,
        measure,
        stress_selection=stress_selection,
    )
    flipped_values = element_stress_measure_values(
        flipped_stress,
        measure,
        stress_selection=stress_selection,
    )
    return flipped_values - current_values, current_values, flipped_values


def first_element_G_grid(
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> np.ndarray:
    values = np.empty((len(parameter_values), len(w_values), 2, 2), dtype=float)
    if triangles is None:
        triangles = flip_mode.current_triangles
    first_triangle = (triangles[0],)
    shape = (len(parameter_values), len(w_values))
    for row, col in _progress_grid_indices(
        shape,
        w_values,
        parameter_values,
        "First element G",
    ):
        parameter_value = parameter_values[row]
        w = w_values[col]
        vertices = t23_parameterized_vertices_from_w_parameter(
            float(w),
            float(parameter_value),
            parameterization=parameterization,
            flip_mode=flip_mode,
        )
        values[row, col] = getG(vertices, first_triangle)[0]
    return values


def edge_flip_energy_difference_grid(
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    material: MaterialConfig = CONFIG.material,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    element_selection: str = STRESS_SELECTION_AVERAGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_stress_selection(element_selection)
    current_element_energy = element_energy_grid(
        w_values,
        parameter_values,
        flip_mode.current_triangles,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    flipped_element_energy = element_energy_grid(
        w_values,
        parameter_values,
        flip_mode.flipped_triangles,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    current_energy = select_energy_values(current_element_energy, element_selection)
    flipped_energy = select_energy_values(flipped_element_energy, element_selection)
    return flipped_energy - current_energy, current_energy, flipped_energy


def _all_vertices(
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
) -> list[dict[int, np.ndarray]]:
    vertex_sets = []
    shape = (len(parameter_values), len(w_values))
    for row, col in _progress_grid_indices(
        shape,
        w_values,
        parameter_values,
        "Element-pair vertices",
    ):
        parameter_value = parameter_values[row]
        w = w_values[col]
        current_vertices = t23_parameterized_vertices_from_w_parameter(
            float(w),
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


def _draw_element_pair_topology_glyph(
    drawing: DrawingArea,
    vertices: dict[int, np.ndarray],
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    active_diagonal: tuple[int, int],
    config: HeatmapElementPairGridConfig,
    selected_element_index: int | None,
) -> None:
    active_diagonal = tuple(sorted(active_diagonal))
    if selected_element_index is None:
        edge_groups = [(None, sorted(_triangle_edges(triangles)))]
    else:
        if not 0 <= selected_element_index < len(triangles):
            raise ValueError(
                "selected_element_index must be a valid zero-based triangle "
                f"index, got {selected_element_index} for {len(triangles)} "
                "triangles."
            )
        triangle_edges = [
            sorted(_triangle_edges((triangle, triangle)))
            for triangle in triangles
        ]
        draw_order = [
            index for index in range(len(triangles))
            if index != selected_element_index
        ] + [selected_element_index]
        edge_groups = [(index, triangle_edges[index]) for index in draw_order]

    for element_index, edge_group in edge_groups:
        is_selected_element = (
            selected_element_index is None
            or element_index == selected_element_index
        )
        linestyle = "-" if is_selected_element else "--"
        edge_alpha = config.alpha if is_selected_element else 0.65 * config.alpha
        for edge in edge_group:
            edge_points = np.array([vertices[edge[0]], vertices[edge[1]]])
            is_active_diagonal = tuple(sorted(edge)) == active_diagonal
            drawing.add_artist(
                Line2D(
                    edge_points[:, 0],
                    edge_points[:, 1],
                    color="black",
                    linewidth=config.linewidth
                    * (1.7 if is_active_diagonal else 1.0),
                    alpha=edge_alpha,
                    linestyle=linestyle,
                )
            )


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
            rf"$x_{{{index}}}$",
            fontsize=9,
            color="black",
        )


# === Helper: Draw a grid of element-pair glyphs over a heatmap axis ===
def plot_heatmap_element_pair_grid(
    ax: plt.Axes,
    w_range: tuple[float, float],
    parameter_range: tuple[float, float],
    config: HeatmapElementPairGridConfig = CONFIG.heatmap.element_pair_grid,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    active_diagonal: tuple[int, int] | None = None,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    g_vector_choice: str | None = None,
    selected_element_index: int | None = None,
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
    if selected_element_index is not None and not (
        0 <= selected_element_index < len(triangles)
    ):
        raise ValueError(
            "selected_element_index must be a valid zero-based triangle index, "
            f"got {selected_element_index} for {len(triangles)} triangles."
        )
    validate_parameterization(parameterization)
    if g_vector_choice is not None:
        validate_g_vector_choice(g_vector_choice)

    w_min, w_max = w_range
    parameter_min, parameter_max = parameter_range
    w_span = w_max - w_min
    parameter_span = parameter_max - parameter_min
    w_centers = np.linspace(
        w_min + config.padding_fraction * w_span,
        w_max - config.padding_fraction * w_span,
        config.size,
    )
    parameter_centers = np.linspace(
        parameter_min + config.padding_fraction * parameter_span,
        parameter_max - config.padding_fraction * parameter_span,
        config.size,
    )
    active_diagonal = tuple(sorted(active_diagonal))
    sampled_geometries = []
    max_radius = 0.0

    for parameter_center in parameter_centers:
        for w_center in w_centers:
            vertices = t23_parameterized_vertices_from_w_parameter(
                float(w_center),
                float(parameter_center),
                parameterization=parameterization,
                flip_mode=flip_mode,
            )
            points = np.array(list(vertices.values()))
            centroid = points.mean(axis=0)
            radius = float(np.max(np.linalg.norm(points - centroid, axis=1)))
            sampled_geometries.append((w_center, parameter_center, vertices, centroid))
            max_radius = max(max_radius, radius)

    if max_radius <= 0.0:
        raise RuntimeError("Expected non-degenerate element pairs in heatmap overlay.")

    figure_width, figure_height = ax.figure.get_size_inches()
    axis_box = ax.get_position()
    axis_size_points = 72.0 * min(
        axis_box.width * figure_width,
        axis_box.height * figure_height,
    )
    target_radius = config.scale_fraction * axis_size_points
    if target_radius <= 0.0:
        raise RuntimeError("Expected a positive heatmap overlay glyph size.")
    glyph_radius = 0.82 * target_radius
    drawing_width = 4.8 * target_radius
    drawing_height = 2.4 * target_radius
    drawing_center = np.array([0.5 * drawing_width, 0.5 * drawing_height])
    left_center = drawing_center + np.array([-1.25 * target_radius, 0.0])
    right_center = drawing_center + np.array([1.25 * target_radius, 0.0])

    for w_center, parameter_center, vertices, centroid in sampled_geometries:
        current_vertices = {
            index: left_center + glyph_radius / max_radius * (point - centroid)
            for index, point in vertices.items()
        }
        flipped_vertices = {
            index: right_center + glyph_radius / max_radius * (point - centroid)
            for index, point in vertices.items()
        }
        drawing = DrawingArea(drawing_width, drawing_height, clip=False)
        _draw_element_pair_topology_glyph(
            drawing,
            current_vertices,
            triangles,
            active_diagonal,
            config,
            selected_element_index,
        )
        _draw_element_pair_topology_glyph(
            drawing,
            flipped_vertices,
            flip_mode.flipped_triangles,
            flip_mode.flipped_diagonal,
            config,
            selected_element_index,
        )
        drawing.add_artist(
            FancyArrowPatch(
                drawing_center + np.array([-0.24 * target_radius, 0.0]),
                drawing_center + np.array([0.24 * target_radius, 0.0]),
                arrowstyle="-|>",
                mutation_scale=4.0,
                shrinkA=0.0,
                shrinkB=0.0,
                linewidth=0.6,
                color="0.2",
                alpha=config.alpha,
                zorder=8,
            )
        )

        if g_vector_choice is not None:
            for start, end in selected_g_vector_edges(
                current_vertices,
                triangles,
                g_vector_choice,
            ):
                arrow = FancyArrowPatch(
                    current_vertices[start],
                    current_vertices[end],
                    arrowstyle="-|>",
                    mutation_scale=config.g_vector_arrow_scale,
                    shrinkA=0.0,
                    shrinkB=0.0,
                    linewidth=config.g_vector_linewidth,
                    color=config.g_vector_color,
                    alpha=config.g_vector_alpha,
                    zorder=10,
                )
                drawing.add_artist(arrow)

        annotation = AnnotationBbox(
            drawing,
            (w_center, parameter_center),
            xycoords="data",
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0.0,
        )
        annotation.set_zorder(ELEMENT_PAIR_GRID_ZORDER)
        ax.add_artist(annotation)


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
    w_values = _as_1d_float_array(config.w_values, "w_values")
    parameter_values = _as_1d_float_array(config.parameter_values, "parameter_values")
    validate_parameterization(parameterization)
    if triangles is None:
        triangles = flip_mode.current_triangles
    if active_diagonal is None:
        active_diagonal = flip_mode.current_diagonal
    vertex_sets = _all_vertices(
        w_values,
        parameter_values,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    reference_vertices = square_reference_vertices(flip_mode=flip_mode)
    limits = _shared_limits(vertex_sets + [reference_vertices])
    axis_label = parameter_axis_label(parameterization)

    fig, axes = plt.subplots(
        len(parameter_values),
        len(w_values),
        figsize=(3.4 * len(w_values), 3.4 * len(parameter_values)),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for row, parameter_value in enumerate(parameter_values):
        for col, w in enumerate(w_values):
            ax = axes[row, col]
            vertices = vertex_sets[row * len(w_values) + col]
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
            ax.set_title(
                rf"$w$={w:g}, {axis_label}={parameter_value:g}"
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
    color_limit_mask: np.ndarray | None = None,
):
    return field_color_norm(
        values,
        color_scale=config.color_scale,
        power_gamma=config.power_gamma,
        centered_colorbar=config.centered_colorbar,
        color_limit_mask=color_limit_mask,
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
    w_values: np.ndarray,
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
        w_values,
        parameter_values,
        mask.astype(float),
        levels=[level, 1.5],
        colors="none",
        hatches=[hatch],
        zorder=zorder,
    )
    hatched_region.set_edgecolor(color)
    hatched_region.set_facecolor("none")


def reference_energy_label(
    config: ReferenceContourConfig,
    element_selection: str = STRESS_SELECTION_AVERAGE,
) -> str:
    validate_stress_selection(element_selection)
    if config.label is not None:
        return config.label
    element_suffix = (
        ""
        if element_selection == STRESS_SELECTION_AVERAGE
        else rf"^{{({element_selection_index(element_selection)})}}"
    )
    return (
        rf"$|\Delta E{element_suffix}| = "
        rf"E_{{\gamma={config.shear:g}}}{element_suffix}$"
    )


def contour_level_label(level: float) -> str:
    if np.isclose(level, 0.0, atol=1e-12):
        return "0"
    return f"{level:.2f}".rstrip("0").rstrip(".")


def add_reference_contour(
    ax: plt.Axes,
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    values: np.ndarray,
    level: float,
    config: ReferenceContourConfig,
    *,
    require_nonnegative_level: bool = False,
    symmetric_levels: bool = False,
) -> bool:
    if not config.draw:
        return False
    if values.shape != (len(parameter_values), len(w_values)):
        raise ValueError(
            "values grid shape does not match w/parameter axes: "
            f"{values.shape} vs {(len(parameter_values), len(w_values))}."
        )
    if require_nonnegative_level and level < 0.0:
        raise ValueError(f"Reference level must be non-negative, got {level}.")
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return False
    min_value = float(np.nanmin(finite_values))
    max_value = float(np.nanmax(finite_values))
    if symmetric_levels:
        reference_magnitude = abs(float(level))
        if reference_magnitude <= 1e-12:
            candidate_levels = [0.0]
        else:
            candidate_levels = [-reference_magnitude, reference_magnitude]
        levels = sorted(
            {
                float(candidate_level)
                for candidate_level in candidate_levels
                if min_value <= candidate_level <= max_value
            }
        )
    else:
        levels = [float(level)] if min_value <= level <= max_value else []
    if not levels:
        return False

    contour = ax.contour(
        w_values,
        parameter_values,
        values,
        levels=levels,
        colors=config.color,
        linestyles=config.linestyle,
        linewidths=config.linewidth,
        zorder=config.zorder,
    )
    ax.clabel(
        contour,
        contour.levels,
        fmt=contour_level_label,
        inline=True,
        inline_spacing=2,
        fontsize=8,
    )
    return True


def add_delaunay_contour(
    ax: plt.Axes,
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    mask: np.ndarray,
    config: ReconnectionContourConfig,
) -> bool:
    if not config.draw_delaunay or not has_region_boundary(mask):
        return False
    if mask.shape != (len(parameter_values), len(w_values)):
        raise ValueError(
            "Delaunay mask shape does not match w/parameter axes: "
            f"{mask.shape} vs {(len(parameter_values), len(w_values))}."
        )
    ax.contour(
        w_values,
        parameter_values,
        mask.astype(float),
        levels=[config.level],
        colors=config.delaunay_color,
        linestyles=config.delaunay_linestyle,
        linewidths=config.delaunay_linewidth,
        zorder=DELAUNAY_CONTOUR_ZORDER,
    )
    return True


def reference_contour_handle(config: ReferenceContourConfig, label: str) -> Line2D:
    return Line2D(
        [0, 1],
        [0, 0],
        color=config.color,
        linestyle=config.linestyle,
        linewidth=config.linewidth,
        label=label,
    )


def delaunay_contour_handle(config: ReconnectionContourConfig) -> Line2D:
    return Line2D(
        [0, 1],
        [0, 0],
        color=config.delaunay_color,
        linestyle=config.delaunay_linestyle,
        linewidth=config.delaunay_linewidth,
        label="Delaunay boundary",
    )


def heatmap_extent(
    w_values: np.ndarray,
    parameter_values: np.ndarray,
) -> tuple[float, float, float, float]:
    return (
        float(w_values[0]),
        float(w_values[-1]),
        float(parameter_values[0]),
        float(parameter_values[-1]),
    )


def add_reference_parameter_axes(ax: plt.Axes) -> None:
    ax.axvline(REFERENCE_w, color="0.2", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(
        REFERENCE_PARAMETER,
        color="0.2",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
    )


def delaunay_mask_and_color_limit(
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    reconnection_contours: ReconnectionContourConfig,
    color_limits_from_delaunay_switch_region: bool,
    parameterization: ParameterizationConfig,
    flip_mode: FlipMode,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    needs_delaunay_mask = (
        reconnection_contours.draw_delaunay
        or color_limits_from_delaunay_switch_region
    )
    delaunay_current_mask = (
        delaunay_current_diagonal_mask(
            w_values,
            parameter_values,
            flip_mode.current_diagonal,
            parameterization=parameterization,
        )
        if needs_delaunay_mask
        else None
    )
    color_limit_mask = (
        ~delaunay_current_mask
        if color_limits_from_delaunay_switch_region
        and delaunay_current_mask is not None
        else None
    )
    return delaunay_current_mask, color_limit_mask


def add_current_no_flip_contour(
    ax: plt.Axes,
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    current_no_flip_mask: np.ndarray | None,
    reconnection_contours: ReconnectionContourConfig,
) -> bool:
    if current_no_flip_mask is None or not has_region_boundary(current_no_flip_mask):
        return False
    ax.contour(
        w_values,
        parameter_values,
        current_no_flip_mask.astype(float),
        levels=[reconnection_contours.level],
        colors=reconnection_contours.current_color,
        linewidths=reconnection_contours.linewidth,
        alpha=1.0,
        zorder=CURRENT_REGION_OUTLINE_ZORDER,
    )
    return True


def add_heatmap_element_pair_overlay(
    ax: plt.Axes,
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    config: HeatmapElementPairGridConfig,
    parameterization: ParameterizationConfig,
    flip_mode: FlipMode,
    g_vector_choice: str | None = None,
    selected_element_index: int | None = None,
) -> None:
    if not config.draw:
        return
    plot_heatmap_element_pair_grid(
        ax,
        w_range=(float(w_values[0]), float(w_values[-1])),
        parameter_range=(float(parameter_values[0]), float(parameter_values[-1])),
        config=config,
        triangles=flip_mode.current_triangles,
        active_diagonal=flip_mode.current_diagonal,
        parameterization=parameterization,
        flip_mode=flip_mode,
        g_vector_choice=g_vector_choice,
        selected_element_index=selected_element_index,
    )


def draw_field_heatmap_panel(
    ax: plt.Axes,
    color_values: np.ndarray,
    reference_values: np.ndarray,
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    *,
    cmap: str,
    norm,
    title: str,
    show_xlabel: bool,
    show_ylabel: bool,
    current_no_flip_mask: np.ndarray | None,
    reconnection_contours: ReconnectionContourConfig,
    delaunay_current_mask: np.ndarray | None,
    reference_level: float | None,
    reference_contour: ReferenceContourConfig | None,
    reference_contour_symmetric: bool = False,
    element_pair_grid: HeatmapElementPairGridConfig,
    parameterization: ParameterizationConfig,
    flip_mode: FlipMode,
    selected_element_index: int | None = None,
) -> tuple[object, bool, bool, bool]:
    image = ax.imshow(
        color_values,
        origin="lower",
        extent=heatmap_extent(w_values, parameter_values),
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    add_reference_parameter_axes(ax)
    current_contour_drawn = add_current_no_flip_contour(
        ax,
        w_values,
        parameter_values,
        current_no_flip_mask,
        reconnection_contours,
    )
    delaunay_contour_drawn = (
        add_delaunay_contour(
            ax,
            w_values,
            parameter_values,
            delaunay_current_mask,
            reconnection_contours,
        )
        if delaunay_current_mask is not None
        else False
    )
    reference_contour_drawn = (
        add_reference_contour(
            ax,
            w_values,
            parameter_values,
            reference_values,
            reference_level,
            reference_contour,
            symmetric_levels=reference_contour_symmetric,
        )
        if reference_level is not None and reference_contour is not None
        else False
    )
    add_heatmap_element_pair_overlay(
        ax,
        w_values,
        parameter_values,
        element_pair_grid,
        parameterization,
        flip_mode,
        selected_element_index=selected_element_index,
    )
    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel(r"$w$")
    if show_ylabel:
        ax.set_ylabel(parameter_axis_label(parameterization))
    return (
        image,
        current_contour_drawn,
        reference_contour_drawn,
        delaunay_contour_drawn,
    )


def add_field_overlay_legend(
    ax: plt.Axes,
    *,
    current_contour_drawn: bool,
    reference_contour_drawn: bool,
    delaunay_contour_drawn: bool,
    reconnection_contours: ReconnectionContourConfig,
    reference_contour: ReferenceContourConfig | None,
    reference_label: str | None = None,
) -> None:
    legend_handles = []
    if current_contour_drawn:
        legend_handles.append(
            Line2D(
                [0, 1],
                [0, 0],
                color=reconnection_contours.current_color,
                linewidth=reconnection_contours.linewidth,
                label="current no-flip region",
            )
        )
    if reference_contour_drawn and reference_contour is not None:
        legend_handles.append(
            reference_contour_handle(
                reference_contour,
                reference_label
                or rf"simple-shear stress at $\gamma={reference_contour.shear:g}$",
            )
        )
    if delaunay_contour_drawn:
        legend_handles.append(delaunay_contour_handle(reconnection_contours))
    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            loc="upper right",
            framealpha=0.9,
        )
        legend.set_zorder(100)


def hide_unused_panel_axes(
    axes: np.ndarray,
    used_count: int,
) -> None:
    rows, columns = axes.shape
    for empty_index in range(used_count, rows * columns):
        row = empty_index // columns
        col = empty_index % columns
        axes[row, col].set_visible(False)


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
    w_range: tuple[float, float],
    parameter_range: tuple[float, float],
    parameterization: ParameterizationConfig = CONFIG.parameterization,
) -> tuple[np.ndarray, np.ndarray]:
    validate_parameterization(parameterization)
    if resolution < 2:
        raise ValueError(f"resolution must be at least 2, got {resolution}.")
    if not np.all(np.isfinite(w_range)):
        raise ValueError(f"w_range values must be finite, got {w_range}.")
    if w_range[0] >= w_range[1]:
        raise ValueError(f"w_range must be increasing, got {w_range}.")
    L_limits = L_from_w(np.asarray(w_range, dtype=float))
    if np.any(L_limits <= 0.0):
        raise ValueError(
            "w_range makes physical L(w)=sqrt(2)+w non-positive, "
            f"got w_range={w_range}, L_limits={tuple(L_limits)}."
        )
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
        np.linspace(w_range[0], w_range[1], resolution),
        np.linspace(parameter_range[0], parameter_range[1], resolution),
    )


def field_color_norm(
    values: np.ndarray,
    *,
    color_scale: str,
    power_gamma: float,
    centered_colorbar: bool,
    color_limit_mask: np.ndarray | None = None,
):
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Cannot build color scale from only non-finite values.")
    if color_limit_mask is not None:
        color_limit_mask = np.asarray(color_limit_mask, dtype=bool)
        if color_limit_mask.shape != values.shape[: color_limit_mask.ndim]:
            raise ValueError(
                "color_limit_mask shape must match the leading dimensions of "
                f"values, got {color_limit_mask.shape} vs {values.shape}."
            )
        broadcast_mask = color_limit_mask.reshape(
            color_limit_mask.shape + (1,) * (values.ndim - color_limit_mask.ndim)
        )
        finite_limit_values = values[
            np.broadcast_to(broadcast_mask, values.shape) & np.isfinite(values)
        ]
        if finite_limit_values.size == 0:
            warnings.warn(
                "No finite values in the requested color-limit region; using "
                "all finite plotted values instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            finite_values = finite_limit_values
    if color_scale == "linear":
        if centered_colorbar:
            max_abs = float(np.max(np.abs(finite_values)))
            if max_abs <= 0.0:
                max_abs = 1.0
            return CenteredNorm(vcenter=0.0, halfrange=max_abs)
        return Normalize(
            vmin=float(np.min(finite_values)),
            vmax=float(np.max(finite_values)),
        )
    if color_scale == "power":
        if power_gamma <= 0.0:
            raise ValueError(f"power_gamma must be positive, got {power_gamma}.")
        if centered_colorbar:
            max_abs = float(np.max(np.abs(finite_values)))
            if max_abs <= 0.0:
                max_abs = 1.0
            return PowerNorm(gamma=power_gamma, vmin=-max_abs, vmax=max_abs)
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        return PowerNorm(gamma=power_gamma, vmin=vmin, vmax=vmax)
    raise ValueError(
        f"Unsupported color_scale {color_scale!r}. Use 'linear' or 'power'."
    )


def field_cmap_for_range(
    values: np.ndarray,
    default_cmap: str,
    color_limit_mask: np.ndarray | None = None,
) -> str:
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if color_limit_mask is not None:
        color_limit_mask = np.asarray(color_limit_mask, dtype=bool)
        if color_limit_mask.shape != values.shape[: color_limit_mask.ndim]:
            raise ValueError(
                "color_limit_mask shape must match the leading dimensions of "
                f"values, got {color_limit_mask.shape} vs {values.shape}."
            )
        broadcast_mask = color_limit_mask.reshape(
            color_limit_mask.shape + (1,) * (values.ndim - color_limit_mask.ndim)
        )
        finite_limit_values = values[
            np.broadcast_to(broadcast_mask, values.shape) & np.isfinite(values)
        ]
        if finite_limit_values.size > 0:
            finite_values = finite_limit_values
    if finite_values.size == 0:
        return default_cmap

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if np.isclose(vmin, 0.0) and np.isclose(vmax, 0.0):
        return default_cmap
    if vmin >= 0.0:
        return "Reds"
    if vmax <= 0.0:
        return "Blues_r"

    positive_range = vmax
    negative_range = -vmin
    if positive_range >= 2.0 * negative_range:
        return "Reds"
    if negative_range >= 2.0 * positive_range:
        return "Blues_r"
    return default_cmap


def validate_matrix_components(components: tuple[tuple[int, int], ...]) -> None:
    if len(components) == 0:
        raise ValueError("MatrixFieldPlotConfig.components must not be empty.")
    for component in components:
        if len(component) != 2 or any(index not in (0, 1) for index in component):
            raise ValueError(f"Unsupported matrix component {component}.")


def validate_stress_measures(measures: tuple[str, ...]) -> None:
    if len(measures) == 0:
        raise ValueError("Stress measure list must not be empty.")
    invalid_measures = [measure for measure in measures if measure not in STRESS_MEASURES]
    if invalid_measures:
        raise ValueError(
            f"Unsupported stress measures {invalid_measures}. Use {STRESS_MEASURES}."
        )


def validate_stress_selection(stress_selection: str) -> None:
    if stress_selection not in STRESS_SELECTIONS:
        raise ValueError(
            f"Unsupported stress selection {stress_selection!r}. "
            f"Use one of {STRESS_SELECTIONS}."
        )


def validate_stress_selections(stress_selections: tuple[str, ...]) -> None:
    if len(stress_selections) == 0:
        raise ValueError("stress_selections must contain at least one value.")
    for stress_selection in stress_selections:
        validate_stress_selection(stress_selection)


def select_two_element_values(
    values: np.ndarray,
    stress_selection: str,
    *,
    element_axis: int,
) -> np.ndarray:
    validate_stress_selection(stress_selection)
    values = np.asarray(values, dtype=float)
    if element_axis < 0:
        element_axis += values.ndim
    if element_axis < 0 or element_axis >= values.ndim:
        raise ValueError(
            f"element_axis {element_axis} is outside values shape {values.shape}."
        )
    if values.shape[element_axis] != 2:
        raise ValueError(
            "Expected exactly two element values along axis "
            f"{element_axis}, got shape {values.shape}."
        )
    if stress_selection == STRESS_SELECTION_AVERAGE:
        return np.mean(values, axis=element_axis)
    element_index = 0 if stress_selection == STRESS_SELECTION_ELEMENT_1 else 1
    return np.take(values, element_index, axis=element_axis)


def stress_selection_label(stress_selection: str) -> str:
    validate_stress_selection(stress_selection)
    if stress_selection == STRESS_SELECTION_AVERAGE:
        return "element average"
    if stress_selection == STRESS_SELECTION_ELEMENT_1:
        return "element 1"
    if stress_selection == STRESS_SELECTION_ELEMENT_2:
        return "element 2"
    raise RuntimeError(f"Unhandled stress selection {stress_selection!r}.")


def element_selection_index(stress_selection: str) -> int:
    validate_stress_selection(stress_selection)
    if stress_selection == STRESS_SELECTION_ELEMENT_1:
        return 1
    if stress_selection == STRESS_SELECTION_ELEMENT_2:
        return 2
    raise ValueError(f"{stress_selection!r} does not select one element.")


def selected_element_index_for_plot(element_selection: str) -> int | None:
    validate_stress_selection(element_selection)
    if element_selection == STRESS_SELECTION_AVERAGE:
        return None
    return element_selection_index(element_selection) - 1


def label_with_stress_selection(label: str, stress_selection: str) -> str:
    validate_stress_selection(stress_selection)
    return label


def scalar_measure_plot_config_for_selection(
    config: ScalarFieldPlotConfig,
    measure: str,
    stress_selection: str,
) -> ScalarFieldPlotConfig:
    validate_element_stress_measures((measure,))
    validate_stress_selection(stress_selection)
    if stress_selection == STRESS_SELECTION_AVERAGE:
        return config

    element_index = element_selection_index(stress_selection)
    element_label = stress_selection_label(stress_selection).capitalize()
    if measure == ELEMENT_STRESS_MEASURE_SHEAR:
        return replace(
            config,
            title=f"{element_label} shear stress difference",
            colorbar_label=rf"$\Delta \sigma_{{12}}^{{({element_index})}}$",
        )
    if measure == ELEMENT_STRESS_MEASURE_VON_MISES_AVERAGE:
        return replace(
            config,
            title=f"{element_label} von Mises stress difference",
            colorbar_label=(
                rf"$\Delta \sigma_{{\mathrm{{vM}}}}^{{({element_index})}}$"
            ),
        )
    raise RuntimeError(f"Unhandled element stress measure {measure!r}.")


def validate_element_stress_measures(measures: tuple[str, ...]) -> None:
    if len(measures) == 0:
        raise ValueError("Element stress measure list must not be empty.")
    invalid_measures = [
        measure for measure in measures if measure not in ELEMENT_STRESS_MEASURES
    ]
    if invalid_measures:
        raise ValueError(
            "Unsupported element stress measures "
            f"{invalid_measures}. Use {ELEMENT_STRESS_MEASURES}."
        )


def matrix_component_label(
    config: MatrixFieldPlotConfig,
    component: tuple[int, int],
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> str:
    validate_stress_selection(stress_selection)
    i, j = component
    if config.component_symbol == r"\Delta\sigma":
        if stress_selection != STRESS_SELECTION_AVERAGE:
            element_index = element_selection_index(stress_selection)
            return rf"$\Delta \sigma_{{{i + 1}{j + 1}}}^{{({element_index})}}$"
        return rf"$\Delta \left\langle \sigma_{{{i + 1}{j + 1}}} \right\rangle$"
    if config.component_symbol:
        return rf"${config.component_symbol}_{{{i + 1}{j + 1}}}$"
    return f"[{i}, {j}]"


def stress_measure_label(
    measure: str,
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> str:
    validate_stress_measures((measure,))
    validate_stress_selection(stress_selection)
    if measure == STRESS_MEASURE_NORMAL_DIFFERENCE_HALF:
        if stress_selection != STRESS_SELECTION_AVERAGE:
            element_index = element_selection_index(stress_selection)
            return (
                r"$\Delta \left("
                r"\frac{\sigma_{11}-\sigma_{22}}{2} \right)"
                rf"^{{({element_index})}}$"
            )
        return (
            r"$\Delta \left\langle "
            r"\frac{\sigma_{11}-\sigma_{22}}{2} \right\rangle$"
        )
    if measure == STRESS_MEASURE_VON_MISES:
        if stress_selection != STRESS_SELECTION_AVERAGE:
            element_index = element_selection_index(stress_selection)
            return rf"$\Delta\sigma_{{\mathrm{{vM}}}}^{{({element_index})}}$"
        return r"$\Delta\sigma_{\mathrm{vM}}$"
    raise RuntimeError(f"Unhandled stress measure {measure!r}.")


def stress_reference_component_label(
    component: tuple[int, int] | None = None,
) -> str:
    if component is None:
        return (
            r"$|\Delta\sigma_{ij}| = "
            r"|\sigma^{\gamma=0.5}_{ij}|$"
        )
    i, j = component
    return (
        rf"$|\Delta\sigma_{{{i + 1}{j + 1}}}| = "
        rf"|\sigma^{{\gamma=0.5}}_{{{i + 1}{j + 1}}}|$"
    )


def stress_reference_measure_label(measure: str | None = None) -> str:
    if measure is None:
        return (
            r"$|\Delta\sigma_\bullet| = "
            r"|\sigma^{\gamma=0.5}_\bullet|$"
        )
    validate_stress_measures((measure,))
    if measure == STRESS_MEASURE_NORMAL_DIFFERENCE_HALF:
        subscript = r"{(11-22)/2}"
    elif measure == STRESS_MEASURE_VON_MISES:
        subscript = r"{\mathrm{vM}}"
    else:
        raise RuntimeError(f"Unhandled stress measure {measure!r}.")
    return (
        rf"$|\Delta\sigma_{subscript}| = "
        rf"|\sigma^{{\gamma=0.5}}_{subscript}|$"
    )


def matrix_reference_stress_label(
    fields: tuple[tuple[str, tuple[int, int] | str], ...],
) -> str:
    component_fields = [
        field_value
        for field_kind, field_value in fields
        if field_kind == MATRIX_FIELD_COMPONENT
    ]
    measure_fields = [
        field_value
        for field_kind, field_value in fields
        if field_kind == MATRIX_FIELD_STRESS_MEASURE
    ]
    if component_fields and not measure_fields:
        if len(component_fields) == 1 and isinstance(component_fields[0], tuple):
            return stress_reference_component_label(component_fields[0])
        return stress_reference_component_label()
    if measure_fields and not component_fields:
        if len(measure_fields) == 1 and isinstance(measure_fields[0], str):
            return stress_reference_measure_label(measure_fields[0])
        return stress_reference_measure_label()
    return (
        r"$|\Delta\sigma_\bullet| = "
        r"|\sigma^{\gamma=0.5}_\bullet|$"
    )


def draw_matrix_reference_contour(
    config: MatrixFieldPlotConfig,
    field: tuple[str, tuple[int, int] | str],
) -> bool:
    """Skip normal-component reference contours in the Cauchy component row."""
    field_kind, field_value = field
    if (
        config.component_symbol == r"\Delta\sigma"
        and field_kind == MATRIX_FIELD_COMPONENT
    ):
        return field_value == (0, 1)
    return True


def element_stress_reference_label(measure: str) -> str:
    validate_element_stress_measures((measure,))
    if measure == ELEMENT_STRESS_MEASURE_SHEAR:
        return stress_reference_component_label((0, 1))
    if measure == ELEMENT_STRESS_MEASURE_VON_MISES_AVERAGE:
        return stress_reference_measure_label(STRESS_MEASURE_VON_MISES)
    raise RuntimeError(f"Unhandled element stress measure {measure!r}.")


def matrix_plot_fields(
    config: MatrixFieldPlotConfig,
) -> tuple[tuple[str, tuple[int, int] | str], ...]:
    if config.fields is None:
        validate_matrix_components(config.components)
        return tuple((MATRIX_FIELD_COMPONENT, component) for component in config.components)
    validate_matrix_plot_fields(config.fields)
    return config.fields


def validate_matrix_plot_fields(
    fields: tuple[tuple[str, tuple[int, int] | str], ...],
) -> None:
    if len(fields) == 0:
        raise ValueError("MatrixFieldPlotConfig.fields must not be empty.")
    for field_kind, field_value in fields:
        if field_kind == MATRIX_FIELD_COMPONENT:
            if (
                not isinstance(field_value, tuple)
                or len(field_value) != 2
                or any(index not in (0, 1) for index in field_value)
            ):
                raise ValueError(f"Unsupported matrix component field {field_value}.")
        elif field_kind == MATRIX_FIELD_STRESS_MEASURE:
            if not isinstance(field_value, str):
                raise ValueError(f"Stress-measure field must be a string, got {field_value}.")
            validate_stress_measures((field_value,))
        else:
            raise ValueError(
                f"Unsupported matrix plot field kind {field_kind!r}. "
                f"Use {MATRIX_FIELD_COMPONENT!r} or {MATRIX_FIELD_STRESS_MEASURE!r}."
            )


def matrix_plot_field_label(
    config: MatrixFieldPlotConfig,
    field: tuple[str, tuple[int, int] | str],
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> str:
    field_kind, field_value = field
    if field_kind == MATRIX_FIELD_COMPONENT:
        if not isinstance(field_value, tuple):
            raise RuntimeError(f"Expected component tuple, got {field_value}.")
        return matrix_component_label(config, field_value, stress_selection)
    if field_kind == MATRIX_FIELD_STRESS_MEASURE:
        if not isinstance(field_value, str):
            raise RuntimeError(f"Expected stress-measure string, got {field_value}.")
        return stress_measure_label(field_value, stress_selection)
    raise RuntimeError(f"Unhandled matrix plot field kind {field_kind!r}.")


def matrix_plot_field_values(
    matrix_values: np.ndarray,
    field: tuple[str, tuple[int, int] | str],
) -> np.ndarray:
    field_kind, field_value = field
    if field_kind == MATRIX_FIELD_COMPONENT:
        if not isinstance(field_value, tuple):
            raise RuntimeError(f"Expected component tuple, got {field_value}.")
        i, j = field_value
        return matrix_values[..., i, j]
    if field_kind == MATRIX_FIELD_STRESS_MEASURE:
        if not isinstance(field_value, str):
            raise RuntimeError(f"Expected stress-measure string, got {field_value}.")
        return stress_measure_values(matrix_values, field_value)
    raise RuntimeError(f"Unhandled matrix plot field kind {field_kind!r}.")


def stress_measure_values(stress_values: np.ndarray, measure: str) -> np.ndarray:
    validate_stress_measures((measure,))
    sigma11 = stress_values[..., 0, 0]
    sigma22 = stress_values[..., 1, 1]
    sigma12 = 0.5 * (stress_values[..., 0, 1] + stress_values[..., 1, 0])
    normal_difference_half = 0.5 * (sigma11 - sigma22)
    if measure == STRESS_MEASURE_NORMAL_DIFFERENCE_HALF:
        return normal_difference_half
    if measure == STRESS_MEASURE_VON_MISES:
        # 2D in-plane deviatoric magnitude. This makes the no-shear limit match
        # |(sigma_11 - sigma_22) / 2|, which is the comparison used here.
        vm_squared = normal_difference_half**2 + sigma12**2
        return np.sqrt(np.maximum(vm_squared, 0.0))
    raise RuntimeError(f"Unhandled stress measure {measure!r}.")


def element_stress_measure_values(
    element_stress_values: np.ndarray,
    measure: str,
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> np.ndarray:
    validate_element_stress_measures((measure,))
    validate_stress_selection(stress_selection)
    if element_stress_values.shape[-3:] != (2, 2, 2):
        raise ValueError(
            "element_stress_values must end with shape (2 elements, 2, 2), "
            f"got {element_stress_values.shape}."
        )
    if measure == ELEMENT_STRESS_MEASURE_SHEAR:
        sigma12 = 0.5 * (
            element_stress_values[..., 0, 1]
            + element_stress_values[..., 1, 0]
        )
        return select_two_element_values(
            sigma12,
            stress_selection,
            element_axis=-1,
        )
    if measure == ELEMENT_STRESS_MEASURE_VON_MISES_AVERAGE:
        return select_two_element_values(
            stress_measure_values(element_stress_values, STRESS_MEASURE_VON_MISES),
            stress_selection,
            element_axis=-1,
        )
    raise RuntimeError(f"Unhandled element stress measure {measure!r}.")


def matrix_plot_columns(config: MatrixFieldPlotConfig) -> int:
    if config.columns is not None:
        if config.columns < 1:
            raise ValueError(f"columns must be positive, got {config.columns}.")
        return config.columns
    field_count = len(matrix_plot_fields(config))
    return 2 if field_count == 4 else field_count


def build_matrix_field_heatmaps(
    matrix_values: np.ndarray,
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    config: MatrixFieldPlotConfig,
    current_no_flip_mask: np.ndarray | None = None,
    reconnection_contours: ReconnectionContourConfig = CONFIG.heatmap.reconnection_contours,
    reference_matrix: np.ndarray | None = None,
    reference_contour: ReferenceContourConfig | None = None,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> plt.Figure:
    validate_parameterization(parameterization)
    validate_stress_selection(stress_selection)
    plot_fields = matrix_plot_fields(config)
    if matrix_values.shape != (len(parameter_values), len(w_values), 2, 2):
        raise ValueError(
            "matrix_values must have shape "
            f"({len(parameter_values)}, {len(w_values)}, 2, 2), "
            f"got {matrix_values.shape}."
        )
    full_matrix_values = matrix_values
    color_matrix_values = (
        mask_matrix_field_to_region(matrix_values, current_no_flip_mask)
        if config.mask_color_outside_no_flip_region
        and current_no_flip_mask is not None
        else matrix_values
    )
    delaunay_current_mask, color_limit_mask = delaunay_mask_and_color_limit(
        w_values,
        parameter_values,
        reconnection_contours,
        config.color_limits_from_delaunay_switch_region,
        parameterization,
        flip_mode,
    )
    columns = matrix_plot_columns(config)
    rows = int(np.ceil(len(plot_fields) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            MATRIX_PANEL_FIGSIZE[0] * columns + 1.0,
            MATRIX_PANEL_FIGSIZE[1] * rows,
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    current_contour_drawn = False
    reference_contour_drawn = False
    delaunay_contour_drawn = False
    reference_label_fields = []
    selected_element_index = selected_element_index_for_plot(stress_selection)
    for field_index, field in enumerate(plot_fields):
        row = field_index // columns
        col = field_index % columns
        ax = axes[row, col]
        values = matrix_plot_field_values(color_matrix_values, field)
        norm = field_color_norm(
            values,
            color_scale=config.color_scale,
            power_gamma=config.power_gamma,
            centered_colorbar=config.centered_colorbar,
            color_limit_mask=color_limit_mask,
        )
        cmap = field_cmap_for_range(
            values,
            config.cmap,
            color_limit_mask=color_limit_mask,
        )
        draw_reference = draw_matrix_reference_contour(config, field)
        reference_level = (
            float(matrix_plot_field_values(reference_matrix, field))
            if reference_matrix is not None
            and reference_contour is not None
            and draw_reference
            else None
        )
        if reference_level is not None:
            reference_label_fields.append(field)
        (
            image,
            panel_current_contour_drawn,
            panel_reference_contour_drawn,
            panel_delaunay_contour_drawn,
        ) = draw_field_heatmap_panel(
            ax,
            values,
            matrix_plot_field_values(full_matrix_values, field),
            w_values,
            parameter_values,
            cmap=cmap,
            norm=norm,
            title=matrix_plot_field_label(config, field, stress_selection),
            show_xlabel=row == rows - 1,
            show_ylabel=col == 0,
            current_no_flip_mask=current_no_flip_mask,
            reconnection_contours=reconnection_contours,
            delaunay_current_mask=delaunay_current_mask,
            reference_level=reference_level,
            reference_contour=reference_contour,
            reference_contour_symmetric=True,
            element_pair_grid=config.element_pair_grid,
            parameterization=parameterization,
            flip_mode=flip_mode,
            selected_element_index=selected_element_index,
        )
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(
            label_with_stress_selection(config.colorbar_label, stress_selection)
        )
        current_contour_drawn = (
            panel_current_contour_drawn or current_contour_drawn
        )
        reference_contour_drawn = (
            panel_reference_contour_drawn or reference_contour_drawn
        )
        delaunay_contour_drawn = (
            panel_delaunay_contour_drawn or delaunay_contour_drawn
        )

    hide_unused_panel_axes(axes, len(plot_fields))

    add_field_overlay_legend(
        axes[0, 0],
        current_contour_drawn=current_contour_drawn,
        reference_contour_drawn=reference_contour_drawn,
        delaunay_contour_drawn=delaunay_contour_drawn,
        reconnection_contours=reconnection_contours,
        reference_contour=reference_contour,
        reference_label=matrix_reference_stress_label(tuple(reference_label_fields)),
    )
    if not remove_figure_title:
        fig.suptitle(
            f"{config.title} ({config.resolution}x{config.resolution}, "
            f"{config.color_scale}, {parameterization.mode})"
        )
    return fig


def build_scalar_field_heatmap(
    values: np.ndarray,
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    config: ScalarFieldPlotConfig,
    current_no_flip_mask: np.ndarray | None = None,
    reconnection_contours: ReconnectionContourConfig = CONFIG.heatmap.reconnection_contours,
    reference_level: float | None = None,
    reference_contour: ReferenceContourConfig | None = None,
    reference_label: str | None = None,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> plt.Figure:
    validate_parameterization(parameterization)
    validate_stress_selection(stress_selection)
    if values.shape != (len(parameter_values), len(w_values)):
        raise ValueError(
            "values must have shape "
            f"({len(parameter_values)}, {len(w_values)}), got {values.shape}."
        )
    full_values = values
    color_values = (
        mask_scalar_field_to_region(values, current_no_flip_mask)
        if config.mask_color_outside_no_flip_region
        and current_no_flip_mask is not None
        else values
    )
    delaunay_current_mask, color_limit_mask = delaunay_mask_and_color_limit(
        w_values,
        parameter_values,
        reconnection_contours,
        config.color_limits_from_delaunay_switch_region,
        parameterization,
        flip_mode,
    )
    norm = field_color_norm(
        color_values,
        color_scale=config.color_scale,
        power_gamma=config.power_gamma,
        centered_colorbar=config.centered_colorbar,
        color_limit_mask=color_limit_mask,
    )
    cmap = field_cmap_for_range(
        color_values,
        config.cmap,
        color_limit_mask=color_limit_mask,
    )

    fig, ax = plt.subplots(figsize=STANDALONE_FIGSIZE, constrained_layout=True)
    (
        image,
        current_contour_drawn,
        reference_contour_drawn,
        delaunay_contour_drawn,
    ) = draw_field_heatmap_panel(
        ax,
        color_values,
        full_values,
        w_values,
        parameter_values,
        cmap=cmap,
        norm=norm,
        title=config.title,
        show_xlabel=True,
        show_ylabel=True,
        current_no_flip_mask=current_no_flip_mask,
        reconnection_contours=reconnection_contours,
        delaunay_current_mask=delaunay_current_mask,
        reference_level=reference_level,
        reference_contour=reference_contour,
        reference_contour_symmetric=True,
        element_pair_grid=config.element_pair_grid,
        parameterization=parameterization,
        flip_mode=flip_mode,
        selected_element_index=selected_element_index_for_plot(stress_selection),
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(label_with_stress_selection(config.colorbar_label, stress_selection))

    add_field_overlay_legend(
        ax,
        current_contour_drawn=current_contour_drawn,
        reference_contour_drawn=reference_contour_drawn,
        delaunay_contour_drawn=delaunay_contour_drawn,
        reconnection_contours=reconnection_contours,
        reference_contour=reference_contour,
        reference_label=reference_label,
    )

    if not remove_figure_title:
        fig.suptitle(
            f"{config.title} ({config.resolution}x{config.resolution}, "
            f"{config.color_scale}, {parameterization.mode})"
        )
    return fig


def build_scalar_field_panel_heatmaps(
    field_values: tuple[np.ndarray, ...],
    field_labels: tuple[str, ...],
    reference_levels: tuple[float, ...],
    w_values: np.ndarray,
    parameter_values: np.ndarray,
    config: MatrixFieldPlotConfig,
    current_no_flip_mask: np.ndarray | None = None,
    reconnection_contours: ReconnectionContourConfig = CONFIG.heatmap.reconnection_contours,
    reference_contour: ReferenceContourConfig | None = None,
    reference_label: str | None = None,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> plt.Figure:
    validate_parameterization(parameterization)
    validate_stress_selection(stress_selection)
    if not (
        len(field_values) == len(field_labels) == len(reference_levels)
    ):
        raise ValueError(
            "field_values, field_labels, and reference_levels must have equal "
            f"lengths, got {len(field_values)}, {len(field_labels)}, "
            f"{len(reference_levels)}."
        )
    if len(field_values) == 0:
        raise ValueError("Expected at least one scalar field to plot.")
    for values in field_values:
        if values.shape != (len(parameter_values), len(w_values)):
            raise ValueError(
                "scalar field values must have shape "
                f"({len(parameter_values)}, {len(w_values)}), got {values.shape}."
            )

    color_fields = tuple(
        mask_scalar_field_to_region(values, current_no_flip_mask)
        if config.mask_color_outside_no_flip_region
        and current_no_flip_mask is not None
        else values
        for values in field_values
    )
    delaunay_current_mask, color_limit_mask = delaunay_mask_and_color_limit(
        w_values,
        parameter_values,
        reconnection_contours,
        config.color_limits_from_delaunay_switch_region,
        parameterization,
        flip_mode,
    )
    columns = matrix_plot_columns(config)
    rows = int(np.ceil(len(field_values) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            MATRIX_PANEL_FIGSIZE[0] * columns + 1.0,
            MATRIX_PANEL_FIGSIZE[1] * rows,
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    current_contour_drawn = False
    reference_contour_drawn = False
    delaunay_contour_drawn = False
    selected_element_index = selected_element_index_for_plot(stress_selection)
    for field_index, (values, color_values, label, reference_level) in enumerate(
        zip(field_values, color_fields, field_labels, reference_levels)
    ):
        row = field_index // columns
        col = field_index % columns
        ax = axes[row, col]
        norm = field_color_norm(
            color_values,
            color_scale=config.color_scale,
            power_gamma=config.power_gamma,
            centered_colorbar=config.centered_colorbar,
            color_limit_mask=color_limit_mask,
        )
        cmap = field_cmap_for_range(
            color_values,
            config.cmap,
            color_limit_mask=color_limit_mask,
        )
        (
            image,
            panel_current_contour_drawn,
            panel_reference_contour_drawn,
            panel_delaunay_contour_drawn,
        ) = draw_field_heatmap_panel(
            ax,
            color_values,
            values,
            w_values,
            parameter_values,
            cmap=cmap,
            norm=norm,
            title=label,
            show_xlabel=row == rows - 1,
            show_ylabel=col == 0,
            current_no_flip_mask=current_no_flip_mask,
            reconnection_contours=reconnection_contours,
            delaunay_current_mask=delaunay_current_mask,
            reference_level=reference_level,
            reference_contour=reference_contour,
            reference_contour_symmetric=True,
            element_pair_grid=config.element_pair_grid,
            parameterization=parameterization,
            flip_mode=flip_mode,
            selected_element_index=selected_element_index,
        )
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(
            label_with_stress_selection(config.colorbar_label, stress_selection)
        )
        current_contour_drawn = (
            panel_current_contour_drawn or current_contour_drawn
        )
        reference_contour_drawn = (
            panel_reference_contour_drawn or reference_contour_drawn
        )
        delaunay_contour_drawn = (
            panel_delaunay_contour_drawn or delaunay_contour_drawn
        )

    hide_unused_panel_axes(axes, len(field_values))

    add_field_overlay_legend(
        axes[0, 0],
        current_contour_drawn=current_contour_drawn,
        reference_contour_drawn=reference_contour_drawn,
        delaunay_contour_drawn=delaunay_contour_drawn,
        reconnection_contours=reconnection_contours,
        reference_contour=reference_contour,
        reference_label=reference_label,
    )
    if not remove_figure_title:
        fig.suptitle(
            f"{config.title} ({config.resolution}x{config.resolution}, "
            f"{config.color_scale}, {parameterization.mode})"
        )
    return fig


def stress_data_grid_matches_config(
    stress_data: dict[str, np.ndarray],
    config: MatrixFieldPlotConfig,
    stress_selection: str,
) -> bool:
    w_values = stress_data.get("w_values")
    if w_values is None:
        return False
    parameter_values = stress_data["parameter_values"]
    return (
        len(w_values) == config.resolution
        and len(parameter_values) == config.resolution
        and np.allclose((w_values[0], w_values[-1]), config.w_range)
        and np.allclose((parameter_values[0], parameter_values[-1]), config.parameter_range)
        and stress_data.get("stress_selection") == stress_selection
    )


def build_cauchy_stress_measure_heatmaps(
    config: MatrixFieldPlotConfig = CONFIG.cauchy_stress_measures,
    material: MaterialConfig = CONFIG.material,
    reference_contour: ReferenceContourConfig = CONFIG.reference_contour,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
    stress_data: dict[str, np.ndarray] | None = None,
    assert_component_signs: bool = CONFIG.assert_element_stress_component_signs,
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    validate_stress_selection(stress_selection)
    fields = matrix_plot_fields(config)
    for field_kind, field_value in fields:
        if field_kind != MATRIX_FIELD_STRESS_MEASURE or not isinstance(
            field_value,
            str,
        ):
            raise ValueError(
                "build_cauchy_stress_measure_heatmaps only supports "
                f"{MATRIX_FIELD_STRESS_MEASURE!r} fields, got "
                f"{field_kind!r}, {field_value!r}."
            )

    if stress_data is not None and stress_data_grid_matches_config(
        stress_data,
        config,
        stress_selection,
    ):
        w_values = stress_data["w_values"]
        parameter_values = stress_data["parameter_values"]
        current_no_flip_mask = stress_data["inside_current_reconnection_zone"]
        reference_cauchy_stress = stress_data["reference_cauchy_stress"]
    else:
        w_values, parameter_values = sampled_parameter_values(
            config.resolution,
            config.w_range,
            config.parameter_range,
            parameterization=parameterization,
        )
        current_reconnection_masks = reconnection_condition_masks(
            w_values,
            parameter_values,
            triangles=flip_mode.current_triangles,
            shared_edge=flip_mode.current_diagonal,
            parameterization=parameterization,
            flip_mode=flip_mode,
        )
        current_no_flip_mask = current_reconnection_masks["inside"]
        reference_cauchy_stress = reference_simple_shear_cauchy_stress(
            material,
            reference_contour,
        )

    field_values = []
    field_labels = []
    reference_levels = []
    data = {
        "w_values": w_values,
        "physical_L_values": L_from_w(w_values),
        "parameter_values": parameter_values,
        "inside_current_reconnection_zone": current_no_flip_mask,
        "reference_cauchy_stress": reference_cauchy_stress,
        "stress_selection": stress_selection,
    }
    for _, measure in fields:
        if not isinstance(measure, str):
            raise RuntimeError(f"Expected stress measure string, got {measure}.")
        values, current_values, flipped_values = stress_measure_difference_grid(
            w_values,
            parameter_values,
            measure,
            material=material,
            parameterization=parameterization,
            flip_mode=flip_mode,
            stress_selection=stress_selection,
        )
        reference_level = float(stress_measure_values(reference_cauchy_stress, measure))
        field_values.append(values)
        field_labels.append(stress_measure_label(measure, stress_selection))
        reference_levels.append(reference_level)
        data[measure] = values
        data[f"current_{measure}"] = current_values
        data[f"flipped_{measure}"] = flipped_values
        data[f"visible_{measure}"] = mask_scalar_field_to_region(
            values,
            current_no_flip_mask,
        )
        data[f"reference_{measure}"] = np.asarray(reference_level)

    fig = build_scalar_field_panel_heatmaps(
        tuple(field_values),
        tuple(field_labels),
        tuple(reference_levels),
        w_values,
        parameter_values,
        config,
        current_no_flip_mask=current_no_flip_mask,
        reconnection_contours=CONFIG.heatmap.reconnection_contours,
        reference_contour=reference_contour,
        reference_label=matrix_reference_stress_label(fields),
        parameterization=parameterization,
        flip_mode=flip_mode,
        remove_figure_title=remove_figure_title,
        stress_selection=stress_selection,
    )
    return fig, data


def build_cauchy_stress_difference_heatmaps(
    config: MatrixFieldPlotConfig = CONFIG.cauchy_stress_difference,
    material: MaterialConfig = CONFIG.material,
    reference_contour: ReferenceContourConfig = CONFIG.reference_contour,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
    assert_component_signs: bool = CONFIG.assert_element_stress_component_signs,
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    validate_stress_selection(stress_selection)
    w_values, parameter_values = sampled_parameter_values(
        config.resolution,
        config.w_range,
        config.parameter_range,
        parameterization=parameterization,
    )
    matrix_values = cauchy_stress_difference_grid(
        w_values,
        parameter_values,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
        assert_component_signs=assert_component_signs,
        stress_selection=stress_selection,
    )
    current_reconnection_masks = reconnection_condition_masks(
        w_values,
        parameter_values,
        triangles=flip_mode.current_triangles,
        shared_edge=flip_mode.current_diagonal,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    current_no_flip_mask = current_reconnection_masks["inside"]
    reference_cauchy_stress = reference_simple_shear_cauchy_stress(
        material,
        reference_contour,
    )
    fig = build_matrix_field_heatmaps(
        matrix_values,
        w_values,
        parameter_values,
        config,
        current_no_flip_mask=current_no_flip_mask,
        reconnection_contours=CONFIG.heatmap.reconnection_contours,
        reference_matrix=reference_cauchy_stress,
        reference_contour=reference_contour,
        parameterization=parameterization,
        flip_mode=flip_mode,
        remove_figure_title=remove_figure_title,
        stress_selection=stress_selection,
    )
    return fig, {
        "w_values": w_values,
        "physical_L_values": L_from_w(w_values),
        "parameter_values": parameter_values,
        "matrix_values": matrix_values,
        "inside_current_reconnection_zone": current_no_flip_mask,
        "reference_cauchy_stress": reference_cauchy_stress,
        "stress_selection": stress_selection,
    }


def reference_element_stress_measure_value(
    material: MaterialConfig,
    reference_contour: ReferenceContourConfig,
    measure: str,
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> float:
    reference_cauchy_stress = reference_simple_shear_cauchy_stress(
        material,
        reference_contour,
    )
    two_element_reference_stress = np.stack(
        (reference_cauchy_stress, reference_cauchy_stress),
        axis=0,
    )
    value = float(
        element_stress_measure_values(
            two_element_reference_stress,
            measure,
            stress_selection=stress_selection,
        )
    )
    return 0.0 if np.isclose(value, 0.0, atol=1e-12) else value


def build_element_stress_measure_heatmap(
    config: ScalarFieldPlotConfig,
    measure: str,
    material: MaterialConfig = CONFIG.material,
    reference_contour: ReferenceContourConfig = CONFIG.reference_contour,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
    stress_selection: str = STRESS_SELECTION_AVERAGE,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    validate_element_stress_measures((measure,))
    validate_stress_selection(stress_selection)
    w_values, parameter_values = sampled_parameter_values(
        config.resolution,
        config.w_range,
        config.parameter_range,
        parameterization=parameterization,
    )
    values, current_values, flipped_values = element_stress_measure_difference_grid(
        w_values,
        parameter_values,
        measure,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
        stress_selection=stress_selection,
    )
    current_reconnection_masks = reconnection_condition_masks(
        w_values,
        parameter_values,
        triangles=flip_mode.current_triangles,
        shared_edge=flip_mode.current_diagonal,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    current_no_flip_mask = current_reconnection_masks["inside"]
    reference_value = reference_element_stress_measure_value(
        material,
        reference_contour,
        measure,
        stress_selection=stress_selection,
    )
    plot_config = scalar_measure_plot_config_for_selection(
        config,
        measure,
        stress_selection,
    )
    fig = build_scalar_field_heatmap(
        values,
        w_values,
        parameter_values,
        plot_config,
        current_no_flip_mask=current_no_flip_mask,
        reconnection_contours=CONFIG.heatmap.reconnection_contours,
        reference_level=reference_value,
        reference_contour=reference_contour,
        reference_label=element_stress_reference_label(measure),
        parameterization=parameterization,
        flip_mode=flip_mode,
        remove_figure_title=remove_figure_title,
        stress_selection=stress_selection,
    )
    return fig, {
        "w_values": w_values,
        "physical_L_values": L_from_w(w_values),
        "parameter_values": parameter_values,
        "values": values,
        "current_values": current_values,
        "flipped_values": flipped_values,
        "visible_values": mask_scalar_field_to_region(values, current_no_flip_mask),
        "inside_current_reconnection_zone": current_no_flip_mask,
        "reference_value": np.asarray(reference_value),
        "measure": np.asarray(measure),
        "stress_selection": stress_selection,
    }


def build_first_element_G_heatmaps(
    config: MatrixFieldPlotConfig = CONFIG.first_element_G,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    w_values, parameter_values = sampled_parameter_values(
        config.resolution,
        config.w_range,
        config.parameter_range,
        parameterization=parameterization,
    )
    matrix_values = first_element_G_grid(
        w_values,
        parameter_values,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    current_reconnection_masks = reconnection_condition_masks(
        w_values,
        parameter_values,
        triangles=flip_mode.current_triangles,
        shared_edge=flip_mode.current_diagonal,
        parameterization=parameterization,
        flip_mode=flip_mode,
    )
    current_no_flip_mask = current_reconnection_masks["inside"]
    fig = build_matrix_field_heatmaps(
        matrix_values,
        w_values,
        parameter_values,
        config,
        current_no_flip_mask=current_no_flip_mask,
        reconnection_contours=CONFIG.heatmap.reconnection_contours,
        parameterization=parameterization,
        flip_mode=flip_mode,
        remove_figure_title=remove_figure_title,
    )
    return fig, {
        "w_values": w_values,
        "physical_L_values": L_from_w(w_values),
        "parameter_values": parameter_values,
        "matrix_values": matrix_values,
        "inside_current_reconnection_zone": current_no_flip_mask,
    }


def build_flip_energy_heatmap(
    config: HeatmapConfig = CONFIG.heatmap,
    material: MaterialConfig = CONFIG.material,
    reference_contour: ReferenceContourConfig = CONFIG.reference_contour,
    parameterization: ParameterizationConfig = CONFIG.parameterization,
    flip_mode: FlipMode = CONFIG.flip_mode,
    content: str = HEATMAP_COMBINED,
    g_vector_choice: str = G_VECTOR_CHOICE_SHORTEST,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
    element_selection: str = STRESS_SELECTION_AVERAGE,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    validate_heatmap_content(content)
    validate_parameterization(parameterization)
    validate_g_vector_choice(g_vector_choice)
    validate_stress_selection(element_selection)
    w_values, parameter_values = sampled_parameter_values(
        config.resolution,
        config.w_range,
        config.parameter_range,
        parameterization=parameterization,
    )
    delta_energy, current_energy, flipped_energy = edge_flip_energy_difference_grid(
        w_values,
        parameter_values,
        material=material,
        parameterization=parameterization,
        flip_mode=flip_mode,
        element_selection=element_selection,
    )
    abs_delta_energy = np.abs(delta_energy)
    reference_energy = reference_simple_shear_energy(
        material,
        reference_contour,
        element_selection=element_selection,
    )
    show_energy = content in (HEATMAP_COMBINED, HEATMAP_ENERGY_ONLY)
    show_regions = content in (HEATMAP_COMBINED, HEATMAP_REGIONS_ONLY)
    contours = config.reconnection_contours
    current_reconnection_masks = reconnection_condition_masks(
        w_values,
        parameter_values,
        triangles=flip_mode.current_triangles,
        shared_edge=flip_mode.current_diagonal,
        parameterization=parameterization,
        flip_mode=flip_mode,
        g_vector_choice=g_vector_choice,
    )
    current_no_flip_mask = current_reconnection_masks["inside"]
    visible_delta_energy = (
        mask_scalar_field_to_region(delta_energy, current_no_flip_mask)
        if show_energy and config.mask_color_outside_no_flip_region
        else delta_energy
    )
    delaunay_current_mask, color_limit_mask = delaunay_mask_and_color_limit(
        w_values,
        parameter_values,
        contours,
        config.color_limits_from_delaunay_switch_region,
        parameterization,
        flip_mode,
    )
    norm = (
        heatmap_color_norm(
            visible_delta_energy,
            config=config,
            color_limit_mask=color_limit_mask,
        )
        if show_energy
        else None
    )
    flipped_no_flip_mask = reconnection_condition_masks(
        w_values,
        parameter_values,
        triangles=flip_mode.flipped_triangles,
        shared_edge=flip_mode.flipped_diagonal,
        parameterization=parameterization,
        flip_mode=flip_mode,
        g_vector_choice=g_vector_choice,
    )["inside"]

    fig, ax = plt.subplots(figsize=STANDALONE_FIGSIZE, constrained_layout=True)
    overlay = config.element_pair_grid
    image = None

    if contours.debug_only:
        debug_mask = np.zeros_like(current_no_flip_mask, dtype=float)
        debug_mask[current_no_flip_mask] += 1.0
        debug_mask[flipped_no_flip_mask] += 2.0
        image = ax.imshow(
            debug_mask,
            origin="lower",
            extent=heatmap_extent(w_values, parameter_values),
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
            extent=heatmap_extent(w_values, parameter_values),
            aspect="auto",
            cmap=config.cmap,
            norm=norm,
            interpolation="nearest",
        )
    else:
        ax.set_xlim(w_values[0], w_values[-1])
        ax.set_ylim(parameter_values[0], parameter_values[-1])
        ax.set_facecolor("white")

    if show_regions and not contours.debug_only:
        if (
            not show_energy
            and contours.draw_current
            and np.any(current_no_flip_mask)
        ):
            ax.contourf(
                w_values,
                parameter_values,
                current_no_flip_mask.astype(float),
                levels=[contours.level, 1.5],
                colors=[contours.current_color],
                alpha=contours.fill_alpha,
                zorder=REGION_FILL_ZORDER,
            )
        if contours.draw_failure_reasons:
            failure_reason_styles = (
                (
                    "g12_negative",
                    contours.g12_negative_color,
                    REGION_FILL_ZORDER + 1,
                ),
                (
                    "g12_too_large",
                    contours.g12_too_large_color,
                    REGION_FILL_ZORDER + 1,
                ),
                (
                    "shared_edge_not_longest",
                    contours.shared_edge_not_longest_color,
                    REGION_FILL_ZORDER + 1,
                ),
            )
            for reason_key, color, zorder in failure_reason_styles:
                reason_mask = current_reconnection_masks[reason_key]
                if np.any(reason_mask):
                    ax.contourf(
                        w_values,
                        parameter_values,
                        reason_mask.astype(float),
                        levels=[contours.level, 1.5],
                        colors=[color],
                        alpha=contours.failure_fill_alpha,
                        zorder=zorder,
                    )
        if contours.draw_flipped and np.any(flipped_no_flip_mask):
            ax.contourf(
                w_values,
                parameter_values,
                flipped_no_flip_mask.astype(float),
                levels=[contours.level, 1.5],
                colors=[contours.flipped_color],
                alpha=contours.fill_alpha,
                zorder=REGION_FILL_ZORDER,
            )
        if contours.draw_failure_reasons:
            add_hatched_region_overlay(
                ax,
                w_values,
                parameter_values,
                current_reconnection_masks["g12_too_large"],
                contours.g12_too_large_color,
                contours.g12_too_large_hatch,
                contours.level,
                zorder=REGION_HATCH_ZORDER,
            )

    if overlay.draw:
        highlighted_g_vector_choice = (
            g_vector_choice if content == HEATMAP_REGIONS_ONLY else None
        )
        add_heatmap_element_pair_overlay(
            ax,
            w_values,
            parameter_values,
            overlay,
            parameterization=parameterization,
            flip_mode=flip_mode,
            g_vector_choice=highlighted_g_vector_choice,
            selected_element_index=selected_element_index_for_plot(element_selection),
        )

    legend_handles = []
    reference_contour_values = (
        np.abs(delta_energy)
        if reference_contour.use_absolute_delta_energy
        else delta_energy
    )
    reference_contour_drawn = add_reference_contour(
        ax,
        w_values,
        parameter_values,
        reference_contour_values,
        reference_energy,
        reference_contour,
        require_nonnegative_level=True,
    )
    if reference_contour_drawn:
        legend_handles.append(
            reference_contour_handle(
                reference_contour,
                reference_energy_label(reference_contour, element_selection),
            )
        )

    delaunay_contour_drawn = (
        add_delaunay_contour(
            ax,
            w_values,
            parameter_values,
            delaunay_current_mask,
            contours,
        )
        if delaunay_current_mask is not None
        else False
    )
    if delaunay_contour_drawn:
        legend_handles.append(delaunay_contour_handle(contours))

    has_current_no_flip_boundary = has_region_boundary(
        current_no_flip_mask
    )
    if (
        (show_regions or show_energy)
        and contours.draw_current
        and has_current_no_flip_boundary
    ):
        ax.contour(
            w_values,
            parameter_values,
            current_no_flip_mask.astype(float),
            levels=[contours.level],
            colors=contours.current_color,
            linewidths=contours.linewidth,
            alpha=1.0,
            zorder=CURRENT_REGION_OUTLINE_ZORDER,
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
                    w_values,
                    parameter_values,
                    reason_mask.astype(float),
                    levels=[contours.level],
                    colors=color,
                    linewidths=contours.linewidth,
                    alpha=1.0,
                    zorder=REGION_OUTLINE_ZORDER,
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

    has_flipped_no_flip_boundary = has_region_boundary(
        flipped_no_flip_mask
    )
    if show_regions and contours.draw_flipped and has_flipped_no_flip_boundary:
        ax.contour(
            w_values,
            parameter_values,
            flipped_no_flip_mask.astype(float),
            levels=[contours.level],
            colors=contours.flipped_color,
            linewidths=contours.linewidth,
            linestyles="--",
            alpha=1.0,
            zorder=REGION_OUTLINE_ZORDER,
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

    add_reference_parameter_axes(ax)
    ax.set_xlabel(r"$w$")
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
            element_title_suffix = (
                ""
                if element_selection == STRESS_SELECTION_AVERAGE
                else rf", {stress_selection_label(element_selection)}"
            )
            ax.set_title(
                rf"$E_{{\mathrm{{flipped}}}} - E_{{\mathrm{{current}}}}$ "
                rf"({config.resolution}x{config.resolution}, {config.color_scale}, "
                rf"{parameterization.mode}{element_title_suffix})"
            )
    if image is not None:
        colorbar = fig.colorbar(image, ax=ax)
        energy_label = (
            r"$\Delta E = E_{\mathrm{flipped}} - E_{\mathrm{current}}$"
            if element_selection == STRESS_SELECTION_AVERAGE
            else (
                rf"$\Delta E^{{({element_selection_index(element_selection)})}}"
                rf" = E_{{\mathrm{{flipped}}}}^{{({element_selection_index(element_selection)})}}"
                rf" - E_{{\mathrm{{current}}}}^{{({element_selection_index(element_selection)})}}$"
            )
        )
        colorbar.set_label(
            "reconnection zone: 1=current, 2=flipped, 3=both"
            if contours.debug_only
            else energy_label
        )

    data = {
        "w_values": w_values,
        "physical_L_values": L_from_w(w_values),
        "parameter_values": parameter_values,
        "delta_energy": delta_energy,
        "visible_delta_energy": visible_delta_energy,
        "abs_delta_energy": abs_delta_energy,
        "current_energy": current_energy,
        "flipped_energy": flipped_energy,
        "inside_current_reconnection_zone": current_no_flip_mask,
        "inside_flipped_reconnection_zone": flipped_no_flip_mask,
        "inside_reconnection_zone": current_no_flip_mask,
        "delaunay_keeps_current_diagonal": delaunay_current_mask,
        "current_g12_negative": current_reconnection_masks["g12_negative"],
        "current_g12_too_large": current_reconnection_masks["g12_too_large"],
        "current_shared_edge_not_longest": current_reconnection_masks[
            "shared_edge_not_longest"
        ],
        "g_vector_choice": g_vector_choice,
        "parameterization_mode": parameterization.mode,
        "reference_energy": reference_energy,
        "element_selection": element_selection,
    }
    return fig, data


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")


def parameterization_output_path(
    path: Path,
    parameterization: ParameterizationConfig,
) -> Path:
    validate_parameterization(parameterization)
    return path.with_name(f"{path.stem}_{parameterization.mode}{path.suffix}")


def flip_mode_output_path(
    path: Path,
    flip_mode: FlipMode,
    parameterization: ParameterizationConfig | None = None,
) -> Path:
    output_path = path.with_name(f"{path.stem}_{flip_mode.name}{path.suffix}")
    if parameterization is None:
        return output_path
    return parameterization_output_path(output_path, parameterization)


def heatmap_content_output_path(path: Path, output_tag: str) -> Path:
    if output_tag == HEATMAP_COMBINED:
        return path
    return path.with_name(f"{path.stem}_{output_tag}{path.suffix}")


def heatmap_element_output_tag(content: str, element_selection: str) -> str:
    validate_heatmap_content(content)
    validate_stress_selection(element_selection)
    if element_selection == STRESS_SELECTION_AVERAGE:
        return content
    return f"{content}_{element_selection}"


def stress_selection_output_path(path: Path, stress_selection: str) -> Path:
    validate_stress_selection(stress_selection)
    if stress_selection == STRESS_SELECTION_AVERAGE:
        return path
    return path.with_name(f"{path.stem}_{stress_selection}{path.suffix}")


def combined_elements_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_elements{path.suffix}")


def render_pdf_first_page_to_png(
    pdf_path: Path,
    output_prefix: Path,
    render_dpi: int,
) -> Path:
    result = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-r",
            str(render_dpi),
            "-f",
            "1",
            "-l",
            "1",
            str(pdf_path),
            str(output_prefix),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to render {pdf_path} with pdftoppm:\n{result.stderr.strip()}"
        )
    png_path = output_prefix.with_suffix(".png")
    if not png_path.exists():
        raise RuntimeError(f"Expected pdftoppm to create {png_path}.")
    return png_path


def combine_two_element_pdfs(
    first_element_pdf: Path,
    second_element_pdf: Path,
    output_path: Path,
    *,
    render_dpi: int = 180,
) -> None:
    """Create a tightly cropped, vertically stacked PDF from two element PDFs."""
    del render_dpi  # Kept for backwards-compatible call sites.

    if not first_element_pdf.exists():
        raise FileNotFoundError(first_element_pdf)
    if not second_element_pdf.exists():
        raise FileNotFoundError(second_element_pdf)

    first_page = PdfReader(str(first_element_pdf)).pages[0]
    second_page = PdfReader(str(second_element_pdf)).pages[0]

    first_width = float(first_page.cropbox.width)
    first_height = float(first_page.cropbox.height)
    second_width = float(second_page.cropbox.width)
    second_height = float(second_page.cropbox.height)

    output_width = max(first_width, second_width)
    output_height = first_height + second_height

    combined_page = PageObject.create_blank_page(
        width=output_width,
        height=output_height,
    )

    first_x_offset = 0.5 * (output_width - first_width)
    second_x_offset = 0.5 * (output_width - second_width)

    combined_page.merge_transformed_page(
        first_page,
        Transformation().translate(
            tx=first_x_offset - float(first_page.cropbox.left),
            ty=second_height - float(first_page.cropbox.bottom),
        ),
    )
    combined_page.merge_transformed_page(
        second_page,
        Transformation().translate(
            tx=second_x_offset - float(second_page.cropbox.left),
            ty=-float(second_page.cropbox.bottom),
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_page(combined_page)
    with output_path.open("wb") as file:
        writer.write(file)

    print(f"Saved {output_path}")


def combine_element_specific_stress_pdfs(
    base_output_path: Path,
    flip_mode: FlipMode,
    parameterization: ParameterizationConfig,
) -> None:
    flipped_base_path = flip_mode_output_path(
        base_output_path,
        flip_mode,
        parameterization,
    )
    combine_two_element_pdfs(
        stress_selection_output_path(flipped_base_path, STRESS_SELECTION_ELEMENT_1),
        stress_selection_output_path(flipped_base_path, STRESS_SELECTION_ELEMENT_2),
        combined_elements_output_path(flipped_base_path),
    )


def build_heatmap_variants(
    config: HeatmapConfig,
    material: MaterialConfig,
    reference_contour: ReferenceContourConfig,
    parameterization: ParameterizationConfig,
    flip_mode: FlipMode,
    remove_figure_title: bool = CONFIG.remove_figure_titles,
) -> tuple[list[tuple[str, plt.Figure]], dict[str, dict[str, np.ndarray]]]:
    if len(config.contents) == 0:
        raise ValueError("HeatmapConfig.contents must contain at least one variant.")

    figures = []
    data_by_content = {}
    variants = []
    for content in config.contents:
        element_selections = (
            STRESS_SELECTIONS
            if content in (HEATMAP_COMBINED, HEATMAP_ENERGY_ONLY)
            else (STRESS_SELECTION_AVERAGE,)
        )
        variants.extend(
            (
                heatmap_element_output_tag(content, element_selection),
                content,
                G_VECTOR_CHOICE_SHORTEST,
                element_selection,
            )
            for element_selection in element_selections
        )
    variants.extend(
        (
            f"{HEATMAP_REGIONS_ONLY}_{g_vector_choice}",
            HEATMAP_REGIONS_ONLY,
            g_vector_choice,
            STRESS_SELECTION_AVERAGE,
        )
        for g_vector_choice in config.extra_region_g_vector_choices
    )
    for output_tag, content, g_vector_choice, element_selection in variants:
        validate_g_vector_choice(g_vector_choice)
        fig, data = build_flip_energy_heatmap(
            config=config,
            material=material,
            reference_contour=reference_contour,
            parameterization=parameterization,
            flip_mode=flip_mode,
            content=content,
            g_vector_choice=g_vector_choice,
            remove_figure_title=remove_figure_title,
            element_selection=element_selection,
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


def build_stress_plot_set(
    config: PlotConfig,
    flip_mode: FlipMode,
    stress_selection: str,
) -> tuple[list[tuple[Path, plt.Figure]], dict[str, dict[str, np.ndarray]]]:
    validate_stress_selection(stress_selection)
    cauchy_fig, cauchy_data = build_cauchy_stress_difference_heatmaps(
        config=config.cauchy_stress_difference,
        material=config.material,
        reference_contour=config.reference_contour,
        parameterization=config.parameterization,
        flip_mode=flip_mode,
        remove_figure_title=config.remove_figure_titles,
        assert_component_signs=config.assert_element_stress_component_signs,
        stress_selection=stress_selection,
    )
    stress_measure_fig, stress_measure_data = build_cauchy_stress_measure_heatmaps(
        config=config.cauchy_stress_measures,
        material=config.material,
        reference_contour=config.reference_contour,
        parameterization=config.parameterization,
        flip_mode=flip_mode,
        remove_figure_title=config.remove_figure_titles,
        stress_data=cauchy_data,
        assert_component_signs=config.assert_element_stress_component_signs,
        stress_selection=stress_selection,
    )
    shear_fig, shear_data = build_element_stress_measure_heatmap(
        config=config.element_shear_stress,
        measure=ELEMENT_STRESS_MEASURE_SHEAR,
        material=config.material,
        reference_contour=config.reference_contour,
        parameterization=config.parameterization,
        flip_mode=flip_mode,
        remove_figure_title=config.remove_figure_titles,
        stress_selection=stress_selection,
    )
    element_vm_fig, element_vm_data = build_element_stress_measure_heatmap(
        config=config.element_von_mises_stress,
        measure=ELEMENT_STRESS_MEASURE_VON_MISES_AVERAGE,
        material=config.material,
        reference_contour=config.reference_contour,
        parameterization=config.parameterization,
        flip_mode=flip_mode,
        remove_figure_title=config.remove_figure_titles,
        stress_selection=stress_selection,
    )
    figures = [
        (config.cauchy_stress_difference.output_path, cauchy_fig),
        (config.cauchy_stress_measures.output_path, stress_measure_fig),
        (config.element_shear_stress.output_path, shear_fig),
        (config.element_von_mises_stress.output_path, element_vm_fig),
    ]
    data = {
        "cauchy": cauchy_data,
        "stress_measure": stress_measure_data,
        "shear": shear_data,
        "element_vm": element_vm_data,
    }
    return figures, data


def build_and_save_for_flip_mode(
    config: PlotConfig,
    flip_mode: FlipMode,
) -> list[plt.Figure]:
    validate_stress_selections(config.stress_selections)
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
        reference_contour=config.reference_contour,
        parameterization=config.parameterization,
        flip_mode=flip_mode,
        remove_figure_title=config.remove_figure_titles,
    )
    if config.plot_focused_heatmap:
        focused_heatmap_figures, focused_heatmap_data_by_content = build_heatmap_variants(
            config=config.focused_heatmap,
            material=config.material,
            reference_contour=config.reference_contour,
            parameterization=config.parameterization,
            flip_mode=flip_mode,
            remove_figure_title=config.remove_figure_titles,
        )
    else:
        focused_heatmap_figures = []
        focused_heatmap_data_by_content = {}
    stress_figures = []
    stress_data_by_selection = {}
    for stress_selection in config.stress_selections:
        print(f"Stress plots: {stress_selection_label(stress_selection)}")
        plot_specs, stress_data = build_stress_plot_set(
            config,
            flip_mode,
            stress_selection,
        )
        stress_figures.extend(
            (stress_selection, base_path, stress_fig)
            for base_path, stress_fig in plot_specs
        )
        stress_data_by_selection[stress_selection] = stress_data
    if config.plot_first_element_G:
        first_element_G_fig, first_element_G_data = build_first_element_G_heatmaps(
            config=config.first_element_G,
            parameterization=config.parameterization,
            flip_mode=flip_mode,
            remove_figure_title=config.remove_figure_titles,
        )
    else:
        first_element_G_fig = None
        first_element_G_data = None

    save_figure(
        fig,
        flip_mode_output_path(
            config.pair_grid.output_path,
            flip_mode,
            config.parameterization,
        ),
    )
    save_figure(
        flipped_fig,
        flip_mode_output_path(
            config.pair_grid.flipped_output_path,
            flip_mode,
            config.parameterization,
        ),
    )
    for content, heatmap_fig in heatmap_figures:
        save_figure(
            heatmap_fig,
            flip_mode_output_path(
                heatmap_content_output_path(config.heatmap.output_path, content),
                flip_mode,
                config.parameterization,
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
                    config.parameterization,
                ),
            )
    for stress_selection, base_path, stress_fig in stress_figures:
        save_figure(
            stress_fig,
            stress_selection_output_path(
                flip_mode_output_path(
                    base_path,
                    flip_mode,
                    config.parameterization,
                ),
                stress_selection,
            ),
        )
    if (
        config.combine_element_pdfs
        and STRESS_SELECTION_ELEMENT_1 in config.stress_selections
        and STRESS_SELECTION_ELEMENT_2 in config.stress_selections
    ):
        for base_path in (
            config.cauchy_stress_difference.output_path,
            config.cauchy_stress_measures.output_path,
            config.element_shear_stress.output_path,
            config.element_von_mises_stress.output_path,
        ):
            combine_element_specific_stress_pdfs(
                base_path,
                flip_mode,
                config.parameterization,
            )
    if first_element_G_fig is not None:
        save_figure(
            first_element_G_fig,
            flip_mode_output_path(
                config.first_element_G.output_path,
                flip_mode,
                config.parameterization,
            ),
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
    summary_stress_selection = (
        STRESS_SELECTION_AVERAGE
        if STRESS_SELECTION_AVERAGE in stress_data_by_selection
        else next(iter(stress_data_by_selection))
    )
    summary_stress_data = stress_data_by_selection[summary_stress_selection]
    cauchy_data = summary_stress_data["cauchy"]
    stress_measure_data = summary_stress_data["stress_measure"]
    shear_data = summary_stress_data["shear"]
    element_vm_data = summary_stress_data["element_vm"]
    print(
        "Stress summary selection: "
        f"{stress_selection_label(summary_stress_selection)}"
    )
    cauchy_values = cauchy_data["matrix_values"]
    cauchy_mask = cauchy_data["inside_current_reconnection_zone"]
    visible_cauchy_values = mask_matrix_field_to_region(cauchy_values, cauchy_mask)
    reference_energy = heatmap_data["reference_energy"]
    print(
        "Reference simple-shear energy: "
        f"{reference_energy:.6e} "
        f"(gamma={config.reference_contour.shear:g})"
    )
    print(
        "Visible Cauchy stress difference range: "
        f"{float(np.nanmin(visible_cauchy_values)):.6e} "
        f"to {float(np.nanmax(visible_cauchy_values)):.6e}"
    )
    stress_measure_names = [
        field_value
        for field_kind, field_value in matrix_plot_fields(config.cauchy_stress_measures)
        if field_kind == MATRIX_FIELD_STRESS_MEASURE
    ]
    if stress_measure_names:
        print(
            "Visible Cauchy stress measure ranges: "
            + ", ".join(
                f"{measure}="
                f"{float(np.nanmin(stress_measure_data[f'visible_{measure}'])):.6e}"
                f" to "
                f"{float(np.nanmax(stress_measure_data[f'visible_{measure}'])):.6e}"
                for measure in stress_measure_names
            )
        )
    print(
        "Visible element stress measure ranges: "
        f"shear="
        f"{float(np.nanmin(shear_data['visible_values'])):.6e} to "
        f"{float(np.nanmax(shear_data['visible_values'])):.6e}, "
        f"von_mises_average="
        f"{float(np.nanmin(element_vm_data['visible_values'])):.6e} to "
        f"{float(np.nanmax(element_vm_data['visible_values'])):.6e}"
    )
    if first_element_G_data is not None:
        first_element_G_values = first_element_G_data["matrix_values"]
        print(
            "First element G range: "
            f"{float(np.nanmin(first_element_G_values)):.6e} "
            f"to {float(np.nanmax(first_element_G_values)):.6e}"
        )
    current_no_flip_mask = heatmap_data["inside_current_reconnection_zone"]
    flipped_no_flip_mask = heatmap_data["inside_flipped_reconnection_zone"]
    print(
        "Current no-flip occupancy: "
        f"{int(np.count_nonzero(current_no_flip_mask))} / "
        f"{current_no_flip_mask.size}"
    )
    print(
        "Flipped no-flip occupancy: "
        f"{int(np.count_nonzero(flipped_no_flip_mask))} / "
        f"{flipped_no_flip_mask.size}"
    )
    if config.plot_focused_heatmap:
        focused_current_no_flip_mask = focused_heatmap_data[
            "inside_current_reconnection_zone"
        ]
        focused_flipped_no_flip_mask = focused_heatmap_data[
            "inside_flipped_reconnection_zone"
        ]
        print(
            "Focused current no-flip occupancy: "
            f"{int(np.count_nonzero(focused_current_no_flip_mask))} / "
            f"{focused_current_no_flip_mask.size}"
        )
        print(
            "Focused flipped no-flip occupancy: "
            f"{int(np.count_nonzero(focused_flipped_no_flip_mask))} / "
            f"{focused_flipped_no_flip_mask.size}"
        )

    figures = [
        fig,
        flipped_fig,
        *[heatmap_fig for _, heatmap_fig in heatmap_figures],
        *[
            focused_heatmap_fig
            for _, focused_heatmap_fig in focused_heatmap_figures
        ],
        *[stress_fig for _, _, stress_fig in stress_figures],
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
    validate_parameterizations(config.parameterizations)
    validate_stress_selections(config.stress_selections)

    figures = []
    for parameterization in config.parameterizations:
        parameterization_config = replace(config, parameterization=parameterization)
        for flip_mode in parameterization_config.flip_modes:
            figures.extend(
                build_and_save_for_flip_mode(parameterization_config, flip_mode)
            )

    if config.show:
        plt.show()
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
