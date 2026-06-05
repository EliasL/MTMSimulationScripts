"""Two-element edge-flip parameterization plots.

The current geometry follows Sylvain's notation:

    x2 = (0, 0), x3 = (L, 0),
    x1 = (s L, -1 / L), x4 = (t L, 1 / L).

For this first plotting pass we use the symmetric closure v = s - t = 0, so
s = t = (1 + u) / 2.
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
from matplotlib.colors import CenteredNorm, PowerNorm
from matplotlib.patches import Patch, Polygon

from MTMath.energyFunction import ContiEnergy

OLD_TRIANGLES = ((1, 2, 3), (2, 3, 4))
FLIPPED_TRIANGLES = ((1, 2, 4), (1, 4, 3))
OLD_DIAGONAL = (2, 3)
FLIPPED_DIAGONAL = (1, 4)
REFERENCE_L = float(np.sqrt(2.0))
REFERENCE_U = 0.0
ROOT_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT_DIR / "Plots"

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
class PairGridConfig:
    L_values: tuple[float, ...]
    u_values: tuple[float, ...]
    output_path: Path
    flipped_output_path: Path


@dataclass(frozen=True)
class ReconnectionContourConfig:
    draw_current: bool = True
    draw_flipped: bool = True
    debug_only: bool = False
    level: float = 0.5
    current_color: str = "cyan"
    flipped_color: str = "lime"
    linewidth: float = 1.4
    fill_alpha: float = 0.12


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


@dataclass(frozen=True)
class PlotConfig:
    material: MaterialConfig
    pair_grid: PairGridConfig
    heatmap: HeatmapConfig
    focused_heatmap: HeatmapConfig
    show: bool = False


CONFIG = PlotConfig(
    material=MaterialConfig(),
    pair_grid=PairGridConfig(
        L_values=(REFERENCE_L, REFERENCE_L * 2, REFERENCE_L * 3),
        u_values=(0.9, 0.5, 0.0),
        output_path=PLOTS_DIR / "two_element_parameterization_grid.png",
        flipped_output_path=PLOTS_DIR / "two_element_parameterization_flipped_grid.png",
    ),
    heatmap=HeatmapConfig(
        output_path=PLOTS_DIR / "two_element_parameterization_flip_energy_heatmap.png",
        resolution=100,
        L_range=(0.75, REFERENCE_L * 3),
        u_range=(-0.9, 0.9),
    ),
    focused_heatmap=HeatmapConfig(
        output_path=PLOTS_DIR / "two_element_parameterization_flip_energy_heatmap_focused.png",
        resolution=100,
        L_range=(REFERENCE_L, REFERENCE_L * 2),
        u_range=(0.0, 0.2),
    ),
    show=False,
)


def symmetric_current_vertices(L: float, u: float) -> dict[int, np.ndarray]:
    """Return current vertices for the v=0 two-parameter subfamily."""
    if L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")
    if abs(u) >= 1.0:
        raise ValueError(
            f"u must be in (-1, 1) so the flipped triangles stay positive, got {u}."
        )

    middle_x = 0.5 * (1.0 + u) * L
    return {
        1: np.array([middle_x, -1.0 / L]),
        2: np.array([0.0, 0.0]),
        3: np.array([L, 0.0]),
        4: np.array([middle_x, 1.0 / L]),
    }


def square_reference_vertices() -> dict[int, np.ndarray]:
    return symmetric_current_vertices(REFERENCE_L, REFERENCE_U)


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


def insideReconnectionZone(
    L_values: tuple[float, ...] | list[float] | np.ndarray,
    u_values: tuple[float, ...] | list[float] | np.ndarray,
    triangles: tuple[tuple[int, int, int], ...] = FLIPPED_TRIANGLES,
    shared_edge: tuple[int, int] = FLIPPED_DIAGONAL,
) -> np.ndarray:
    """Return a Boolean mask for the C++ inRegion condition on an (u, L) grid.

    The returned array has shape (len(u_values), len(L_values)), matching the
    heatmap orientation used by build_flip_energy_heatmap. A grid point is marked
    True when every triangle satisfies 0 <= G_12 <= min(G_11, G_22), and when
    the specified shared edge is a longest edge of every triangle. G is computed
    from the two shortest edge vectors of each triangle.
    """
    L_values = _as_1d_float_array(L_values, "L_values")
    u_values = _as_1d_float_array(u_values, "u_values")
    mask = np.zeros((len(u_values), len(L_values)), dtype=bool)

    for row, u in enumerate(u_values):
        for col, L in enumerate(L_values):
            vertices = symmetric_current_vertices(float(L), float(u))
            G_values = getG(vertices, triangles)
            G11 = G_values[:, 0, 0]
            G12 = G_values[:, 0, 1]
            G22 = G_values[:, 1, 1]
            inside_per_triangle = (G12 >= 0.0) & (G12 <= np.minimum(G11, G22))
            shared_edge_longest_per_triangle = np.array(
                [
                    shared_edge_is_longest(vertices, triangle, shared_edge)
                    for triangle in triangles
                ],
                dtype=bool,
            )
            mask[row, col] = bool(
                np.all(inside_per_triangle & shared_edge_longest_per_triangle)
            )

    return mask


def symmetric_current_vertices_array(L: np.ndarray, u: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=float)
    u = np.asarray(u, dtype=float)
    if np.any(L <= 0.0):
        raise ValueError("All L values must be positive.")
    if np.any(np.abs(u) >= 1.0):
        raise ValueError("All u values must be in (-1, 1).")
    if L.shape != u.shape:
        raise ValueError(f"L and u must have the same shape, got {L.shape} and {u.shape}.")

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


def verify_old_triangles_are_area_preserving(current_vertices: dict[int, np.ndarray]) -> None:
    reference_vertices = square_reference_vertices()
    determinants = [
        np.linalg.det(
            triangle_deformation_gradient(reference_vertices, current_vertices, triangle)
        )
        for triangle in OLD_TRIANGLES
    ]
    if not np.allclose(determinants, 1.0, rtol=1e-12, atol=1e-12):
        raise RuntimeError(f"Expected old-triangle det(F)=1, got {determinants}.")


def _as_1d_float_array(values: tuple[float, ...] | list[float] | np.ndarray, name: str):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence, got shape {array.shape}.")
    return array


def pair_energy(
    current_vertices: dict[int, np.ndarray],
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    material: MaterialConfig = CONFIG.material,
) -> float:
    """Return reference-area weighted energy for a two-triangle topology."""
    reference_vertices = square_reference_vertices()
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


def old_pair_energy(
    current_vertices: dict[int, np.ndarray],
    material: MaterialConfig = CONFIG.material,
) -> float:
    return pair_energy(current_vertices, OLD_TRIANGLES, material=material)


def flipped_pair_energy(
    current_vertices: dict[int, np.ndarray],
    material: MaterialConfig = CONFIG.material,
) -> float:
    return pair_energy(current_vertices, FLIPPED_TRIANGLES, material=material)


def pair_energy_grid(
    L_values: np.ndarray,
    u_values: np.ndarray,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]],
    material: MaterialConfig = CONFIG.material,
) -> np.ndarray:
    L_grid, u_grid = np.meshgrid(L_values, u_values, indexing="xy")
    current_vertices = symmetric_current_vertices_array(L_grid, u_grid)
    reference_vertices = square_reference_vertices()
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


def edge_flip_energy_difference_grid(
    L_values: np.ndarray,
    u_values: np.ndarray,
    material: MaterialConfig = CONFIG.material,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_energy = pair_energy_grid(
        L_values,
        u_values,
        OLD_TRIANGLES,
        material=material,
    )
    flipped_energy = pair_energy_grid(
        L_values,
        u_values,
        FLIPPED_TRIANGLES,
        material=material,
    )
    return flipped_energy - old_energy, old_energy, flipped_energy


def _all_vertices(
    L_values: np.ndarray,
    u_values: np.ndarray,
) -> list[dict[int, np.ndarray]]:
    geometries = []
    for u in u_values:
        for L in L_values:
            current_vertices = symmetric_current_vertices(float(L), float(u))
            verify_old_triangles_are_area_preserving(current_vertices)
            geometries.append(current_vertices)
    return geometries


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
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]] = OLD_TRIANGLES,
    active_diagonal: tuple[int, int] = OLD_DIAGONAL,
    inactive_diagonal: tuple[int, int] = FLIPPED_DIAGONAL,
    reference_vertices: dict[int, np.ndarray] | None = None,
) -> None:
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


# === Helper: Draw a grid of element-pair geometries over a heatmap axis ===
def plot_heatmap_element_pair_grid(
    ax: plt.Axes,
    L_range: tuple[float, float],
    u_range: tuple[float, float],
    config: HeatmapElementPairGridConfig = CONFIG.heatmap.element_pair_grid,
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]] = OLD_TRIANGLES,
    active_diagonal: tuple[int, int] = OLD_DIAGONAL,
) -> None:
    """Draw a small grid of element-pair geometries over a heatmap axis."""
    if config.size < 1:
        raise ValueError(f"grid_size must be at least 1, got {config.size}.")
    if not 0.0 <= config.padding_fraction < 0.5:
        raise ValueError(
            "padding_fraction must be in [0, 0.5), "
            f"got {config.padding_fraction}."
        )

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

    for u_center in u_centers:
        for L_center in L_centers:
            vertices = symmetric_current_vertices(float(L_center), float(u_center))
            points = np.array(list(vertices.values()))
            centroid = points.mean(axis=0)
            max_radius = float(np.max(np.linalg.norm(points - centroid, axis=1)))
            if max_radius <= 0.0:
                continue
            scale = target_size / max_radius
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


def _shared_limits(geometries: list[dict[int, np.ndarray]]) -> tuple[float, float, float, float]:
    points = np.vstack([np.array(list(vertices.values())) for vertices in geometries])
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
    triangles: tuple[tuple[int, int, int], tuple[int, int, int]] = OLD_TRIANGLES,
    active_diagonal: tuple[int, int] = OLD_DIAGONAL,
    inactive_diagonal: tuple[int, int] = FLIPPED_DIAGONAL,
    title: str = "Symmetric two-element parameterization (v=0)",
) -> plt.Figure:
    """Build a grid of two-triangle element pairs for the v=0 closure."""
    L_values = _as_1d_float_array(config.L_values, "L_values")
    u_values = _as_1d_float_array(config.u_values, "u_values")
    geometries = _all_vertices(L_values, u_values)
    reference_vertices = square_reference_vertices()
    limits = _shared_limits(geometries + [reference_vertices])

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
            vertices = geometries[row * len(L_values) + col]
            energy = pair_energy(vertices, triangles, material=material)
            plot_element_pair(
                ax,
                vertices,
                triangles=triangles,
                active_diagonal=active_diagonal,
                inactive_diagonal=inactive_diagonal,
                reference_vertices=reference_vertices,
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

    fig.suptitle(title, fontsize=14)
    return fig


def build_flipped_parameterization_grid(
    config: PairGridConfig = CONFIG.pair_grid,
    material: MaterialConfig = CONFIG.material,
) -> plt.Figure:
    return build_symmetric_parameterization_grid(
        config=config,
        material=material,
        triangles=FLIPPED_TRIANGLES,
        active_diagonal=FLIPPED_DIAGONAL,
        inactive_diagonal=OLD_DIAGONAL,
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


def build_flip_energy_heatmap(
    config: HeatmapConfig = CONFIG.heatmap,
    material: MaterialConfig = CONFIG.material,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    if config.resolution < 2:
        raise ValueError(f"resolution must be at least 2, got {config.resolution}.")
    if config.L_range[0] <= 0.0 or config.L_range[1] <= 0.0:
        raise ValueError(f"L_range values must be positive, got {config.L_range}.")
    if config.L_range[0] >= config.L_range[1]:
        raise ValueError(f"L_range must be increasing, got {config.L_range}.")
    if config.u_range[0] <= -1.0 or config.u_range[1] >= 1.0:
        raise ValueError(f"u_range must stay inside (-1, 1), got {config.u_range}.")
    if config.u_range[0] >= config.u_range[1]:
        raise ValueError(f"u_range must be increasing, got {config.u_range}.")

    L_values = np.linspace(config.L_range[0], config.L_range[1], config.resolution)
    u_values = np.linspace(config.u_range[0], config.u_range[1], config.resolution)
    delta_energy, old_energy, flipped_energy = edge_flip_energy_difference_grid(
        L_values,
        u_values,
        material=material,
    )
    abs_delta_energy = np.abs(delta_energy)
    norm = heatmap_color_norm(delta_energy, config=config)
    current_reconnection_zone_mask = insideReconnectionZone(
        L_values,
        u_values,
        triangles=OLD_TRIANGLES,
        shared_edge=OLD_DIAGONAL,
    )
    flipped_reconnection_zone_mask = insideReconnectionZone(
        L_values,
        u_values,
        triangles=FLIPPED_TRIANGLES,
        shared_edge=FLIPPED_DIAGONAL,
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    contours = config.reconnection_contours
    overlay = config.element_pair_grid

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
    else:
        image = ax.imshow(
            delta_energy,
            origin="lower",
            extent=(L_values[0], L_values[-1], u_values[0], u_values[-1]),
            aspect="auto",
            cmap=config.cmap,
            norm=norm,
            interpolation="nearest",
        )

    if not contours.debug_only:
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
            triangles=OLD_TRIANGLES,
            active_diagonal=OLD_DIAGONAL,
        )

    legend_handles = []

    has_current_reconnection_boundary = np.any(current_reconnection_zone_mask) and not np.all(
        current_reconnection_zone_mask
    )
    if contours.draw_current and has_current_reconnection_boundary:
        ax.contour(
            L_values,
            u_values,
            current_reconnection_zone_mask.astype(float),
            levels=[contours.level],
            colors=contours.current_color,
            linewidths=contours.linewidth,
            zorder=10,
        )
        legend_handles.append(
            Patch(
                facecolor=contours.current_color,
                edgecolor=contours.current_color,
                alpha=contours.fill_alpha,
                label="current no-flip region: both triangles inside",
            )
        )

    has_flipped_reconnection_boundary = np.any(flipped_reconnection_zone_mask) and not np.all(
        flipped_reconnection_zone_mask
    )
    if contours.draw_flipped and has_flipped_reconnection_boundary:
        ax.contour(
            L_values,
            u_values,
            flipped_reconnection_zone_mask.astype(float),
            levels=[contours.level],
            colors=contours.flipped_color,
            linewidths=contours.linewidth,
            linestyles="--",
            zorder=11,
        )
        legend_handles.append(
            Patch(
                facecolor=contours.flipped_color,
                edgecolor=contours.flipped_color,
                alpha=contours.fill_alpha,
                label="flipped no-flip region: both triangles inside",
            )
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)

    ax.axvline(REFERENCE_L, color="0.2", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(REFERENCE_U, color="0.2", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("L")
    ax.set_ylabel("u")
    if contours.debug_only:
        ax.set_title(rf"Reconnection-zone mask ({config.resolution}x{config.resolution})")
    else:
        ax.set_title(
            rf"$E_{{flipped}} - E_{{current}}$ "
            rf"({config.resolution}x{config.resolution}, {config.color_scale})"
        )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(
        "reconnection zone: 1=current, 2=flipped, 3=both"
        if contours.debug_only
        else r"$\Delta E = E_{flipped} - E_{current}$"
    )

    data = {
        "L_values": L_values,
        "u_values": u_values,
        "delta_energy": delta_energy,
        "abs_delta_energy": abs_delta_energy,
        "old_energy": old_energy,
        "flipped_energy": flipped_energy,
        "inside_current_reconnection_zone": current_reconnection_zone_mask,
        "inside_flipped_reconnection_zone": flipped_reconnection_zone_mask,
        "inside_reconnection_zone": current_reconnection_zone_mask,
    }
    return fig, data


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")


def main(config: PlotConfig = CONFIG) -> None:
    fig = build_symmetric_parameterization_grid(
        config=config.pair_grid,
        material=config.material,
    )
    flipped_fig = build_flipped_parameterization_grid(
        config=config.pair_grid,
        material=config.material,
    )
    heatmap_fig, heatmap_data = build_flip_energy_heatmap(
        config=config.heatmap,
        material=config.material,
    )
    focused_heatmap_fig, focused_heatmap_data = build_flip_energy_heatmap(
        config=config.focused_heatmap,
        material=config.material,
    )

    save_figure(fig, config.pair_grid.output_path)
    save_figure(flipped_fig, config.pair_grid.flipped_output_path)
    save_figure(heatmap_fig, config.heatmap.output_path)
    save_figure(focused_heatmap_fig, config.focused_heatmap.output_path)

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
    if config.show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(flipped_fig)
        plt.close(heatmap_fig)
        plt.close(focused_heatmap_fig)


if __name__ == "__main__":
    main()
