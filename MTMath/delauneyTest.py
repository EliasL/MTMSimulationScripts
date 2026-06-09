#!/usr/bin/env python3
"""Plot final Sylvain mesh element Gram matrices in the Poincare disk."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.tri as mtri
import numpy as np
from scipy.spatial import Delaunay, QhullError
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT = PROJECT_ROOT / "Plots" / "delauneyTest.png"

from Management.jobs import sylvainBatches
from Plotting.dataFunctions import VTUData
from Plotting.remotePlotting import getFinalMesh
from MTMath.poincareEnergy import C2Plane, drawC, plotPoincareDisk, prepPoincareFig


@dataclass(frozen=True)
class MeshSample:
    label: str
    folder: Path
    vtu_file: Path
    points: np.ndarray
    triangles: np.ndarray
    triangulation_mode: str
    original_triangle_count: int


def sylvain_configs(
    batches: list[int],
    nr_seeds: int,
    size: int,
    threads: int,
    reconnection: str,
):
    configs = []
    labels = []
    for batch in batches:
        batch_configs, batch_labels = sylvainBatches(
            batch,
            nrSeeds=nr_seeds,
            size=size,
            threads=threads,
            group_by_variant=False,
            reconnection=reconnection,
        )
        configs.extend(batch_configs)
        labels.extend(
            f"batch={batch}" + (f", {label}" if label else "")
            for label in batch_labels
        )

    if not configs:
        raise RuntimeError(f"No configs found for Sylvain batches {batches}.")
    return configs, labels


def filtered_delaunay_triangles(points: np.ndarray, original_triangles: np.ndarray, progress=None) -> np.ndarray:
    try:
        candidate_triangles = np.asarray(Delaunay(points).simplices, dtype=int)
    except QhullError as exc:
        raise RuntimeError("Delaunay triangulation failed for the mesh node positions.") from exc

    original_mesh = mtri.Triangulation(points[:, 0], points[:, 1], original_triangles)
    finder = original_mesh.get_trifinder()
    triangle_points = points[candidate_triangles]
    edge_midpoints = 0.5 * (
        triangle_points[:, [0, 1, 2], :] + triangle_points[:, [1, 2, 0], :]
    )
    test_points = np.concatenate(
        [triangle_points.mean(axis=1)[:, None, :], edge_midpoints],
        axis=1,
    )
    flat_test_points = test_points.reshape(-1, 2)
    inside = finder(flat_test_points[:, 0], flat_test_points[:, 1]) != -1
    inside_triangles = inside.reshape(len(candidate_triangles), -1).all(axis=1)
    triangles = candidate_triangles[inside_triangles]
    if len(triangles) == 0:
        raise RuntimeError("Delaunay triangulation produced no triangles inside the original mesh.")

    if progress is not None:
        progress.set_postfix_str(f"kept {len(triangles)}/{len(candidate_triangles)} Delaunay")
    return triangles


def choose_triangles(
    points: np.ndarray,
    original_triangles: np.ndarray,
    triangulation_mode: str,
    progress=None,
) -> np.ndarray:
    if triangulation_mode == "mesh":
        return original_triangles
    if triangulation_mode == "delaunay":
        return filtered_delaunay_triangles(points, original_triangles, progress=progress)
    raise ValueError(f"Unsupported triangulation mode: {triangulation_mode}")


def load_final_meshes(
    vtu_files: list[str],
    labels: list[str],
    triangulation_mode: str,
) -> list[MeshSample]:
    if len(vtu_files) != len(labels):
        raise RuntimeError(f"Expected one label per VTU file, got {len(labels)} labels and {len(vtu_files)} VTUs.")

    samples = []
    progress = tqdm(
        list(zip(vtu_files, labels)),
        desc="Reading VTU meshes",
        unit="mesh",
    )
    for vtu_file, label in progress:
        final_vtu = Path(vtu_file)
        folder = final_vtu.parent.parent
        data = VTUData(final_vtu)
        points = np.asarray(data.get_nodes(), dtype=float)
        if points.ndim != 2 or points.shape[1] < 2:
            raise RuntimeError(f"Expected node coordinates with shape (n, >=2), got {points.shape}.")
        points = points[:, :2]

        original_triangles = np.asarray(data.get_connectivity(), dtype=int)
        if original_triangles.ndim != 2 or original_triangles.shape[1] != 3:
            raise RuntimeError(f"Expected triangle connectivity with shape (n, 3), got {original_triangles.shape}.")
        triangles = choose_triangles(points, original_triangles, triangulation_mode, progress=progress)

        samples.append(
            MeshSample(
                label=label,
                folder=folder,
                vtu_file=final_vtu,
                points=points,
                triangles=triangles,
                triangulation_mode=triangulation_mode,
                original_triangle_count=len(original_triangles),
            )
        )

    if not samples:
        raise RuntimeError("No final meshes loaded.")
    return samples


def gram_from_anchor(points: np.ndarray, triangle: np.ndarray, anchor_pos: int) -> np.ndarray:
    anchor = int(triangle[anchor_pos])
    other_vertices = [int(vertex) for pos, vertex in enumerate(triangle) if pos != anchor_pos]
    vectors = [points[vertex] - points[anchor] for vertex in other_vertices]
    vectors = sorted(vectors, key=lambda vector: float(vector @ vector))
    vector_matrix = np.column_stack(vectors)
    return vector_matrix.T @ vector_matrix


def element_gram_matrix_groups(points: np.ndarray, triangles: np.ndarray, progress=None) -> dict[str, np.ndarray]:
    groups = {"shortest pair": [], "other pair 1": [], "other pair 2": []}
    progress_step = 1000
    pending_progress = 0
    for triangle in triangles:
        vertex_G = [gram_from_anchor(points, triangle, anchor_pos) for anchor_pos in range(3)]
        edge_lengths = [float(G[0, 0] + G[1, 1]) for G in vertex_G]
        shortest_anchor_pos = int(np.argmin(edge_lengths))

        groups["shortest pair"].append(vertex_G[shortest_anchor_pos])
        other_anchor_positions = [pos for pos in range(3) if pos != shortest_anchor_pos]
        groups["other pair 1"].append(vertex_G[other_anchor_positions[0]])
        groups["other pair 2"].append(vertex_G[other_anchor_positions[1]])

        if progress is not None:
            pending_progress += 1
            if pending_progress >= progress_step:
                progress.update(pending_progress)
                pending_progress = 0

    if progress is not None and pending_progress:
        progress.update(pending_progress)

    G_groups = {name: np.asarray(values) for name, values in groups.items()}
    for name, G_values in G_groups.items():
        if G_values.shape != (len(triangles), 2, 2):
            raise RuntimeError(
                f"Expected {name} G values with shape ({len(triangles)}, 2, 2), got {G_values.shape}."
            )

        det = np.linalg.det(G_values)
        if np.all(det > 0.0):
            continue
        bad = np.flatnonzero(det <= 0.0)
        raise RuntimeError(f"Expected positive definite {name} Gram matrices; invalid triangle indices: {bad}.")
    return G_groups


def combined_gram_matrix_groups(samples: list[MeshSample]) -> dict[str, np.ndarray]:
    combined = {"shortest pair": [], "other pair 1": [], "other pair 2": []}
    total_triangles = sum(len(sample.triangles) for sample in samples)
    if total_triangles == 0:
        raise RuntimeError("No triangles available for Gram matrix computation.")

    with tqdm(total=total_triangles, desc="Computing element G", unit="tri") as progress:
        for sample in samples:
            groups = element_gram_matrix_groups(sample.points, sample.triangles, progress=progress)
            for name, G_values in groups.items():
                combined[name].append(G_values)
    return {name: np.concatenate(values, axis=0) for name, values in combined.items()}


def plot_triangulation_and_poincare(
    samples: list[MeshSample],
    G_groups: dict[str, np.ndarray],
    grid_size: int,
    transformation: str,
) -> plt.Figure:
    print("Checking Poincare coordinates for G groups...")
    for name, G_values in G_groups.items():
        x, y = C2Plane(G_values, transformation=transformation)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise RuntimeError(f"Expected every {name} Gram matrix to map to a finite Poincare point.")

    print("Drawing mesh preview and Poincare disk...")
    fig, (ax_mesh, ax_disk) = plt.subplots(1, 2, figsize=(11, 5))

    first_sample = samples[0]
    points = first_sample.points
    triangles = first_sample.triangles
    ax_mesh.triplot(points[:, 0], points[:, 1], triangles, color="0.35", linewidth=0.8)
    ax_mesh.scatter(points[:, 0], points[:, 1], s=2, color="tab:blue", zorder=3)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    padding = 0.03 * float(np.max(maxs - mins))
    ax_mesh.set_xlim(mins[0] - padding, maxs[0] + padding)
    ax_mesh.set_ylim(mins[1] - padding, maxs[1] + padding)
    ax_mesh.set_aspect("equal", adjustable="box")
    ax_mesh.set_title(
        f"Final {first_sample.triangulation_mode} example: "
        f"{len(points)} nodes, {len(triangles)} triangles"
    )
    ax_mesh.set_xlabel("x")
    ax_mesh.set_ylabel("y")

    prepPoincareFig(
        ax=ax_disk,
        grid_size=grid_size,
        withGrid=False,
        withYieldSurface=False,
        transformation=transformation,
    )
    plotPoincareDisk(
        ax=ax_disk,
        save=False,
        grid_size=grid_size,
        depth=4,
        transformation=transformation,
    )
    colors = {"shortest pair": "tab:red", "other pair 1": "tab:green", "other pair 2": "tab:purple"}
    for zorder, (name, G_values) in enumerate(tqdm(G_groups.items(), desc="Drawing G groups", unit="group"), start=20):
        drawC(
            ax=ax_disk,
            C=G_values,
            grid_size=grid_size,
            transformation=transformation,
            scatter=True,
            c=colors[name],
            s=14,
            alpha=0.7,
            edgecolors="none",
            zorder=zorder,
        )
    ax_disk.set_title(f"Element G values from {len(samples)} final meshes")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=name, markersize=5)
        for name, color in colors.items()
    ]
    ax_disk.legend(handles=handles, loc="upper right", frameon=False)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load final Sylvain batch meshes and plot each element G in the Poincare disk."
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[-2, -1], help="Sylvain batch numbers to load.")
    parser.add_argument("--nr-seeds", type=int, default=4, help="Number of seeds per Sylvain batch.")
    parser.add_argument("--size", type=int, default=100, help="Mesh size used by sylvainBatches.")
    parser.add_argument("--threads", type=int, default=3, help="Thread count used by sylvainBatches.")
    parser.add_argument("--reconnection", default="none", help="Reconnection method passed to sylvainBatches.")
    parser.add_argument("--max-simulations", type=int, help="Optional limit for quick test plots.")
    parser.add_argument("--force-update", action="store_true", help="Re-download final VTU files even if cached.")
    parser.add_argument(
        "--triangulation",
        default="delaunay",
        choices=("delaunay", "mesh"),
        help="Use filtered Delaunay triangles from node positions, or the original mesh connectivity.",
    )
    parser.add_argument("--grid-size", type=int, default=500, help="Poincare disk plotting grid size.")
    parser.add_argument("--transformation", default="none", choices=("none", "triangular"))
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for saving the figure. Default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)}",
    )
    parser.add_argument("--no-show", action="store_true", help="Build the plot without opening an interactive window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configs, labels = sylvain_configs(
        args.batches,
        nr_seeds=args.nr_seeds,
        size=args.size,
        threads=args.threads,
        reconnection=args.reconnection,
    )
    if args.max_simulations is not None:
        if args.max_simulations <= 0:
            raise ValueError(f"max_simulations must be positive, got {args.max_simulations}.")
        configs = configs[: args.max_simulations]
        labels = labels[: args.max_simulations]

    print(f"Prepared {len(configs)} Sylvain config(s).")
    print(f"Using {args.triangulation} triangulation.")
    print("Locating final mesh VTU files...")
    vtu_files = getFinalMesh(configs, forceUpdate=args.force_update)
    print(f"Loading {len(vtu_files)} final mesh VTU file(s).")
    samples = load_final_meshes(vtu_files, labels, args.triangulation)
    print("Computing Gram matrices from triangle edge pairs.")
    G_groups = combined_gram_matrix_groups(samples)
    print("Building figure.")
    fig = plot_triangulation_and_poincare(
        samples,
        G_groups,
        args.grid_size,
        args.transformation,
    )

    total_triangles = sum(len(sample.triangles) for sample in samples)
    print(f"Loaded {len(samples)} final meshes with {total_triangles} total triangles.")
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.write("Saving figure...")
    sys.stdout.flush()
    fig.savefig(output_path, dpi=300)
    sys.stdout.write(f"\rSaved plot to {output_path}\n")
    sys.stdout.flush()
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
