"""Small, composable drawing helpers for illustrative triangular meshes.

These functions operate on NumPy node and connectivity arrays and an existing
Matplotlib axis.  They deliberately do not know about VTU files, periodic
images, scalar fields, animations, figure layout, or saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.text import Text

from MTMath.meshUtils import mesh_edge_segments, unique_mesh_edges


Color = str | tuple[float, float, float] | tuple[float, float, float, float]


@dataclass(frozen=True)
class MeshStyle:
    """Visual settings shared by the faces, edges, and nodes of one mesh."""

    color: Color = "0.25"
    face_alpha: float = 0.14
    edge_alpha: float = 1.0
    linewidth: float = 1.5
    linestyle: str = "-"
    node_facecolor: Color | None = None
    node_edgecolor: Color | None = None
    node_alpha: float = 1.0
    node_size: float = 18.0
    node_linewidth: float = 0.8
    draw_faces: bool = True
    draw_edges: bool = True
    draw_nodes: bool = True
    zorder: float = 1.0


@dataclass(frozen=True)
class MeshArtists:
    """Artists created by :func:`draw_triangle_mesh`."""

    faces: PolyCollection | None
    edges: LineCollection | None
    nodes: PathCollection | None


def _validate_nodes(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"nodes must have shape (n_nodes, 2), got {nodes.shape}.")
    return nodes


def _validate_connectivity(
    connectivity: np.ndarray,
    *,
    n_nodes: int,
) -> np.ndarray:
    connectivity = np.asarray(connectivity, dtype=int)
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(
            "connectivity must have shape (n_elements, 3), "
            f"got {connectivity.shape}."
        )
    if connectivity.size and (
        connectivity.min() < 0 or connectivity.max() >= n_nodes
    ):
        raise ValueError("connectivity contains a node index outside the nodes array.")
    return connectivity


def draw_mesh_faces(
    ax: Axes,
    nodes: np.ndarray,
    connectivity: np.ndarray,
    *,
    color: Color | Sequence[Color] = "0.25",
    alpha: float = 0.14,
    zorder: float = 1.0,
) -> PolyCollection:
    """Draw one filled polygon per triangular element."""
    nodes = _validate_nodes(nodes)
    connectivity = _validate_connectivity(connectivity, n_nodes=len(nodes))
    collection = PolyCollection(
        nodes[connectivity],
        closed=True,
        facecolors=color,
        edgecolors="none",
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_collection(collection)
    return collection


def draw_mesh_edges(
    ax: Axes,
    nodes: np.ndarray,
    edges: np.ndarray,
    *,
    color: Color | Sequence[Color] = "0.25",
    alpha: float = 1.0,
    linewidth: float = 1.5,
    linestyle: str = "-",
    zorder: float = 2.0,
) -> LineCollection:
    """Draw explicit node-index pairs as a line collection."""
    nodes = _validate_nodes(nodes)
    segments = mesh_edge_segments(nodes, edges)
    collection = LineCollection(
        segments,
        colors=color,
        alpha=alpha,
        linewidths=linewidth,
        linestyles=linestyle,
        zorder=zorder,
    )
    ax.add_collection(collection)
    return collection


def draw_mesh_nodes(
    ax: Axes,
    nodes: np.ndarray,
    *,
    node_ids: np.ndarray | Sequence[int] | None = None,
    facecolor: Color = "0.25",
    edgecolor: Color = "0.25",
    alpha: float = 1.0,
    size: float = 18.0,
    linewidth: float = 0.8,
    zorder: float = 3.0,
) -> PathCollection:
    """Draw all nodes, or an explicitly selected subset of nodes."""
    nodes = _validate_nodes(nodes)
    if node_ids is None:
        selected = nodes
    else:
        node_ids = np.asarray(node_ids, dtype=int)
        if node_ids.ndim != 1:
            raise ValueError(f"node_ids must be one-dimensional, got {node_ids.shape}.")
        if node_ids.size and (node_ids.min() < 0 or node_ids.max() >= len(nodes)):
            raise ValueError("node_ids contains an index outside the nodes array.")
        selected = nodes[node_ids]
    return ax.scatter(
        selected[:, 0],
        selected[:, 1],
        s=size,
        facecolors=facecolor,
        edgecolors=edgecolor,
        alpha=alpha,
        linewidths=linewidth,
        zorder=zorder,
    )


def draw_triangle_mesh(
    ax: Axes,
    nodes: np.ndarray,
    connectivity: np.ndarray,
    *,
    style: MeshStyle = MeshStyle(),
) -> MeshArtists:
    """Draw a triangular mesh without duplicating shared element edges."""
    nodes = _validate_nodes(nodes)
    connectivity = _validate_connectivity(connectivity, n_nodes=len(nodes))

    faces = None
    if style.draw_faces:
        faces = draw_mesh_faces(
            ax,
            nodes,
            connectivity,
            color=style.color,
            alpha=style.face_alpha,
            zorder=style.zorder,
        )

    edges = None
    if style.draw_edges:
        edges = draw_mesh_edges(
            ax,
            nodes,
            unique_mesh_edges(connectivity),
            color=style.color,
            alpha=style.edge_alpha,
            linewidth=style.linewidth,
            linestyle=style.linestyle,
            zorder=style.zorder + 1,
        )

    node_artist = None
    if style.draw_nodes:
        node_ids = np.unique(connectivity) if connectivity.size else np.empty(0, dtype=int)
        node_artist = draw_mesh_nodes(
            ax,
            nodes,
            node_ids=node_ids,
            facecolor=style.node_facecolor or style.color,
            edgecolor=style.node_edgecolor or style.color,
            alpha=style.node_alpha,
            size=style.node_size,
            linewidth=style.node_linewidth,
            zorder=style.zorder + 2,
        )

    return MeshArtists(faces=faces, edges=edges, nodes=node_artist)


def draw_node_labels(
    ax: Axes,
    nodes: np.ndarray,
    labels: Sequence[str],
    *,
    node_ids: np.ndarray | Sequence[int] | None = None,
    offsets: np.ndarray | Sequence[float] = (0.0, 0.0),
    **text_kwargs,
) -> list[Text]:
    """Place caller-provided labels next to selected mesh nodes."""
    nodes = _validate_nodes(nodes)
    if node_ids is None:
        node_ids = np.arange(len(nodes), dtype=int)
    else:
        node_ids = np.asarray(node_ids, dtype=int)
    if node_ids.ndim != 1:
        raise ValueError(f"node_ids must be one-dimensional, got {node_ids.shape}.")
    if node_ids.size and (node_ids.min() < 0 or node_ids.max() >= len(nodes)):
        raise ValueError("node_ids contains an index outside the nodes array.")
    if len(labels) != len(node_ids):
        raise ValueError(
            f"Expected {len(node_ids)} labels for the selected nodes, got {len(labels)}."
        )

    offsets = np.asarray(offsets, dtype=float)
    if offsets.shape == (2,):
        offsets = np.broadcast_to(offsets, (len(node_ids), 2))
    elif offsets.shape != (len(node_ids), 2):
        raise ValueError(
            "offsets must have shape (2,) or (n_selected_nodes, 2), "
            f"got {offsets.shape}."
        )

    kwargs = {"ha": "center", "va": "center", "zorder": 4, **text_kwargs}
    return [
        ax.text(*(nodes[node_id] + offset), label, **kwargs)
        for node_id, offset, label in zip(node_ids, offsets, labels)
    ]


def configure_mesh_axis(
    ax: Axes,
    *node_sets: np.ndarray,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    padding_fraction: float = 0.08,
    equal_aspect: bool = True,
    hide_axes: bool = True,
) -> None:
    """Apply equal aspect, optional automatic limits, and illustration styling."""
    if padding_fraction < 0:
        raise ValueError(
            f"padding_fraction must be non-negative, got {padding_fraction}."
        )
    if xlim is None or ylim is None:
        if not node_sets:
            raise ValueError("node_sets are required when an axis limit is omitted.")
        points = np.concatenate([_validate_nodes(nodes) for nodes in node_sets], axis=0)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        span = float(np.max(maxs - mins))
        padding = padding_fraction * span if span > 0 else padding_fraction
        if xlim is None:
            xlim = (float(mins[0] - padding), float(maxs[0] + padding))
        if ylim is None:
            ylim = (float(mins[1] - padding), float(maxs[1] + padding))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


class MeshFigure:
    """Axis-bound convenience wrapper, analogous to ``LatticeFigure``."""

    def __init__(self, ax: Axes) -> None:
        self.ax = ax
        self._node_sets: list[np.ndarray] = []

    def draw_mesh(
        self,
        nodes: np.ndarray,
        connectivity: np.ndarray,
        *,
        style: MeshStyle = MeshStyle(),
    ) -> MeshArtists:
        nodes = _validate_nodes(nodes)
        self._node_sets.append(nodes)
        return draw_triangle_mesh(self.ax, nodes, connectivity, style=style)

    def draw_edges(self, nodes: np.ndarray, edges: np.ndarray, **kwargs) -> LineCollection:
        nodes = _validate_nodes(nodes)
        self._node_sets.append(nodes)
        return draw_mesh_edges(self.ax, nodes, edges, **kwargs)

    def draw_node_labels(self, nodes: np.ndarray, labels: Sequence[str], **kwargs) -> list[Text]:
        nodes = _validate_nodes(nodes)
        self._node_sets.append(nodes)
        return draw_node_labels(self.ax, nodes, labels, **kwargs)

    def configure_axis(self, **kwargs) -> None:
        configure_mesh_axis(self.ax, *self._node_sets, **kwargs)
