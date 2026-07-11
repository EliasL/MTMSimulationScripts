"""Track one element through VTU output and plot matrix paths on a Poincare disk."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from MTMath.poincareEnergy import C2PoincareDisk, drawPoincareGrid, prepPoincareFig
from Plotting.vtuDataForSylvain import VTUData


@dataclass
class ElementMatrixHistory:
    """A time-ordered 2x2 cell-matrix history for one finite element."""

    paths: tuple[Path, ...]
    matrices: np.ndarray
    element_index: int
    matrix_name: str = "T"

    def __post_init__(self) -> None:
        self.paths = tuple(Path(path) for path in self.paths)
        self.matrices = np.asarray(self.matrices, dtype=float)
        if self.element_index < 0:
            raise ValueError("element_index must be non-negative.")
        if self.matrices.shape != (len(self.paths), 2, 2):
            raise ValueError(
                "matrices must have shape (number of paths, 2, 2); "
                f"received {self.matrices.shape} for {len(self.paths)} paths."
            )
        if not self.paths:
            raise ValueError("At least one VTU state is required.")
        if not np.all(np.isfinite(self.matrices)):
            raise ValueError("The matrix history contains non-finite values.")
        determinants = np.linalg.det(self.matrices)
        if np.any(np.abs(determinants) < 1e-12):
            bad = int(np.flatnonzero(np.abs(determinants) < 1e-12)[0])
            raise ValueError(f"Singular matrix at history index {bad}: {self.paths[bad]}")

    @classmethod
    def from_vtu_files(
        cls,
        vtu_files: Iterable[str | Path],
        element_index: int,
        matrix_name: str = "T",
        *,
        symmetric: bool = False,
    ) -> "ElementMatrixHistory":
        """Read one 2x2 cell matrix from each VTU file in the supplied order."""
        paths = tuple(Path(path) for path in vtu_files)
        if not paths:
            raise ValueError("No VTU files were supplied.")
        matrices = np.stack(
            [
                read_element_matrix(
                    path,
                    element_index,
                    matrix_name,
                    symmetric=symmetric,
                )
                for path in paths
            ]
        )
        return cls(paths, matrices, element_index, matrix_name)

    def subset(self, indices: Sequence[int] | np.ndarray) -> "ElementMatrixHistory":
        indices = np.asarray(indices, dtype=int)
        return ElementMatrixHistory(
            tuple(self.paths[index] for index in indices),
            self.matrices[indices],
            self.element_index,
            self.matrix_name,
        )

    def consecutive_unique(self) -> "ElementMatrixHistory":
        """Remove repeated consecutive matrices while preserving path order."""
        keep = np.ones(len(self.matrices), dtype=bool)
        keep[1:] = np.any(self.matrices[1:] != self.matrices[:-1], axis=(1, 2))
        return self.subset(np.flatnonzero(keep))

    def right_cauchy_green(self) -> np.ndarray:
        return np.swapaxes(self.matrices, -1, -2) @ self.matrices

    def poincare_coordinates(
        self, transformation: np.ndarray | None = None
    ) -> np.ndarray:
        x, y = C2PoincareDisk(
            self.right_cauchy_green(), transformation=transformation
        )
        coordinates = np.column_stack((x, y))
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("The matrix history produced non-finite Poincare coordinates.")
        return coordinates

    def centering_matrix(self, history_index: int) -> np.ndarray:
        """Return M=T_0^-1, so T -> T M maps the selected state to I."""
        self._validate_history_index(history_index)
        return np.linalg.inv(self.matrices[history_index])

    def hyperbolic_step_distances(self, *, unique: bool = True) -> np.ndarray:
        history = self.consecutive_unique() if unique else self
        coordinates = history.poincare_coordinates()
        z = coordinates[:, 0] + 1j * coordinates[:, 1]
        relative_radius = np.abs((z[1:] - z[:-1]) / (1 - np.conjugate(z[:-1]) * z[1:]))
        if np.any(relative_radius >= 1):
            raise ValueError("A relative Poincare coordinate lies outside the unit disk.")
        return 2 * np.arctanh(relative_radius)

    def index_of(self, path_or_name: str | Path) -> int:
        requested = Path(path_or_name)
        matches = [
            index
            for index, path in enumerate(self.paths)
            if path == requested or path.name == requested.name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one history entry matching {path_or_name!s}, found {len(matches)}."
            )
        return matches[0]

    def write_csv(self, output_path: str | Path) -> None:
        coordinates = self.poincare_coordinates()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "sequence",
                    "element",
                    "matrix",
                    "M11",
                    "M12",
                    "M21",
                    "M22",
                    "poincare_x",
                    "poincare_y",
                    "file",
                ]
            )
            for sequence, (path, matrix, coordinate) in enumerate(
                zip(self.paths, self.matrices, coordinates)
            ):
                writer.writerow(
                    [
                        sequence,
                        self.element_index,
                        self.matrix_name,
                        *matrix.reshape(-1),
                        *coordinate,
                        path.name,
                    ]
                )

    def _validate_history_index(self, history_index: int) -> None:
        if not 0 <= history_index < len(self.paths):
            raise IndexError(
                f"History index {history_index} is outside [0, {len(self.paths)})."
            )


def read_element_matrix(
    vtu_file: str | Path,
    element_index: int,
    matrix_name: str = "T",
    *,
    symmetric: bool = False,
) -> np.ndarray:
    """Read one 2x2 cell matrix from one VTU file."""
    if element_index < 0:
        raise ValueError("element_index must be non-negative.")
    data = VTUData(vtu_file)
    components = {
        (i, j): read_matrix_component(data, matrix_name, i, j, symmetric=symmetric)
        for i in (1, 2)
        for j in (1, 2)
    }
    shapes = {index: values.shape for index, values in components.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Matrix component shapes do not match: {shapes}")
    number_of_elements = next(iter(components.values())).shape[0]
    if element_index >= number_of_elements:
        raise IndexError(
            f"Element {element_index} is outside a VTU with {number_of_elements} cells: "
            f"{vtu_file}"
        )
    return np.array(
        [
            [components[(1, 1)][element_index], components[(1, 2)][element_index]],
            [components[(2, 1)][element_index], components[(2, 2)][element_index]],
        ],
        dtype=float,
    )


def read_matrix_component(
    data: VTUData,
    matrix_name: str,
    i: int,
    j: int,
    *,
    symmetric: bool,
) -> np.ndarray:
    candidates = (
        f"{matrix_name}{i}{j}",
        f"{matrix_name}_{i}{j}",
        f"{matrix_name}{i}_{j}",
        f"{matrix_name}_{i}_{j}",
    )
    field_name = next(
        (candidate for candidate in candidates if candidate in data.cell_field_names),
        None,
    )
    if field_name is None and symmetric and i != j:
        return read_matrix_component(data, matrix_name, j, i, symmetric=False)
    if field_name is None:
        available = [
            name for name in data.cell_field_names if name.startswith(matrix_name)
        ]
        raise KeyError(
            f"Missing {matrix_name}[{i},{j}]. Tried {candidates}. "
            f"Available matching cell fields: {available}"
        )
    values, location, _ = data.field(field_name)
    if location != "cell":
        raise ValueError(f"Expected {field_name!r} to be cell data, found {location} data.")
    return np.asarray(values)


def plot_poincare_path(
    history: ElementMatrixHistory,
    *,
    transformation: np.ndarray | None = None,
    unique: bool = True,
    label_endpoints: bool = True,
    ax: plt.Axes | None = None,
    grid_size: int = 800,
    grid_depth: int | None = None,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """Plot a single-element matrix history on the Poincare disk."""
    plotted = history.consecutive_unique() if unique else history
    fig, ax = prepare_poincare_axes(
        ax=ax,
        transformation=transformation,
        grid_size=grid_size,
        grid_depth=grid_depth,
    )
    coordinates = plotted.poincare_coordinates(transformation)
    xy = disk_to_plot(coordinates, grid_size)
    ax.plot(xy[:, 0], xy[:, 1], color="0.15", linewidth=1.5, zorder=5)
    ax.scatter(xy[:, 0], xy[:, 1], color="0.15", s=11, linewidths=0, zorder=6)

    if label_endpoints:
        ax.scatter(
            *xy[0], s=75, marker="o", facecolor="white", edgecolor="black", zorder=12
        )
        ax.scatter(
            *xy[-1], s=80, marker="s", facecolor="black", edgecolor="black", zorder=12
        )
        ax.annotate(
            "Start", xy=xy[0], xytext=(8, 8), textcoords="offset points", zorder=13
        )
        ax.annotate(
            "End",
            xy=xy[-1],
            xytext=(-8, -8),
            textcoords="offset points",
            ha="right",
            va="top",
            zorder=13,
        )
    return fig, ax, xy


def plot_centered_poincare_transition(
    history: ElementMatrixHistory,
    before_index: int,
    after_index: int,
    *,
    before_label: str = "before",
    after_label: str = "after",
    unique: bool = True,
    use_tex: bool = True,
    grid_size: int = 800,
    grid_depth: int = 10,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """Center the disk on one history state and highlight a selected transition."""
    history._validate_history_index(before_index)
    history._validate_history_index(after_index)
    centering = history.centering_matrix(before_index)
    fig, ax, _ = plot_poincare_path(
        history,
        transformation=centering,
        unique=unique,
        label_endpoints=False,
        grid_size=grid_size,
        grid_depth=grid_depth,
    )
    selected = history.subset([before_index, after_index])
    selected_xy = disk_to_plot(selected.poincare_coordinates(centering), grid_size)
    before_xy, after_xy = selected_xy

    ax.scatter(
        *before_xy,
        s=85,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=1,
        zorder=120,
    )
    ax.scatter(
        *after_xy,
        s=95,
        marker="s",
        facecolor="#d73027",
        edgecolor="black",
        linewidth=1,
        zorder=120,
    )
    ax.annotate(
        "",
        xy=after_xy,
        xytext=before_xy,
        arrowprops={"arrowstyle": "-|>", "color": "#d73027", "linewidth": 2.2},
        zorder=110,
    )
    ax.set_xlabel(r"$x_p(\widetilde{C})$")
    ax.set_ylabel(r"$y_p(\widetilde{C})$")
    add_centering_description(
        ax,
        history.matrices[before_index],
        centering,
        use_tex=use_tex,
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="black",
                label=before_label,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#d73027",
                markeredgecolor="black",
                label=after_label,
            ),
        ],
        loc="lower right",
        frameon=False,
    )
    return fig, ax, centering


def prepare_poincare_axes(
    *,
    ax: plt.Axes | None,
    transformation: np.ndarray | None,
    grid_size: int,
    grid_depth: int | None,
) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.2, 8.5))
    else:
        fig = ax.figure
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    prepPoincareFig(
        grid_size=grid_size,
        ax=ax,
        withCircle=True,
        withGrid=False,
        withYieldSurface=False,
        minimalTicks=False,
    )
    drawPoincareGrid(
        ax,
        grid_size=grid_size,
        depth=grid_depth if grid_depth is not None else (10 if transformation is not None else 6),
        c="gray",
        alpha=0.55,
        zorder=1,
        transformation=transformation,
    )
    return fig, ax


def add_centering_description(
    ax: plt.Axes,
    reference: np.ndarray,
    centering: np.ndarray,
    *,
    use_tex: bool,
) -> None:
    if use_tex:
        preamble = str(matplotlib.rcParams["text.latex.preamble"])
        if "amsmath" not in preamble:
            matplotlib.rcParams["text.latex.preamble"] = preamble + r"\usepackage{amsmath}"
        text = (
            "Pre-jump reference and centering\n"
            rf"$T_0={matrix_latex(reference)},\qquad "
            rf"M=T_0^{{-1}}={matrix_latex(centering)}$"
            "\n"
            r"$\widetilde{T}=T M,\qquad \widetilde{C}=M^{\mathsf{T}} C M$"
        )
    else:
        text = (
            "Pre-jump reference and centering\n"
            f"T0 = {matrix_text(reference)}\n"
            f"M = inv(T0) = {matrix_text(centering)}\n"
            "T_tilde = T M,    C_tilde = M^T C M"
        )
    ax.text(
        0.015,
        0.985,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.45,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 1},
        usetex=use_tex,
        zorder=200,
    )


def disk_to_plot(coordinates: np.ndarray, grid_size: int) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim == 1:
        coordinates = coordinates[None, :]
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"Expected Poincare coordinates with shape (N, 2), got {coordinates.shape}.")
    half = grid_size / 2
    return coordinates * half + half


def matrix_text(matrix: np.ndarray) -> str:
    values = [[format_number(value) for value in row] for row in np.asarray(matrix)]
    return f"[[{', '.join(values[0])}], [{', '.join(values[1])}]]"


def matrix_latex(matrix: np.ndarray) -> str:
    values = [[format_number(value) for value in row] for row in np.asarray(matrix)]
    return (
        r"\begin{pmatrix}"
        + " & ".join(values[0])
        + r" \\ "
        + " & ".join(values[1])
        + r"\end{pmatrix}"
    )


def format_number(value: float) -> str:
    rounded = round(float(value))
    return str(rounded) if np.isclose(value, rounded, atol=1e-10) else f"{value:.5g}"
