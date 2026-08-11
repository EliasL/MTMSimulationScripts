"""Rotate the reference mesh while keeping one simple-shear current mesh fixed."""

import argparse
import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from MTMath.energyFunction import ContiEnergy, rotation
from MTMath.meshUtils import element_deformation_gradients, perfect_grid_nodes
from MTMath.miniMTM import simpleShearSystem2


def _element_connectivity(elements) -> np.ndarray:
    connectivity = []
    for element in elements:
        node_ids = getattr(element, "node_ids", element)
        connectivity.append(node_ids)
    connectivity = np.asarray(connectivity, dtype=int)
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(
            f"Expected triangular connectivity with shape (n_elements, 3), got {connectivity.shape}."
        )
    return connectivity


def _rotate_points(points: np.ndarray, theta: float, center: np.ndarray) -> np.ndarray:
    R = rotation(theta)
    return (points - center) @ R.T + center


def _assert_equal_across_elements(values: np.ndarray, name: str) -> None:
    if values.shape[1] <= 1:
        return
    ref = values[:, :1, :, :]
    if not np.allclose(values, ref, rtol=1e-9, atol=1e-12):
        diff = np.max(np.abs(values - ref))
        raise RuntimeError(f"{name} differs across elements; max |diff|={diff:.3e}")


def build_rotation_study(
    mesh_cells: int = 2,
    shear: float = 0.8,
    n_angles: int = 181,
) -> dict[str, np.ndarray | float | int]:
    if mesh_cells < 1:
        raise ValueError(f"mesh_cells must be at least 1, got {mesh_cells}.")
    if n_angles < 2:
        raise ValueError(f"n_angles must be at least 2, got {n_angles}.")

    nodes_per_side = mesh_cells + 1
    shear_values = np.array([shear], dtype=float)
    position_history, elements, _, _ = simpleShearSystem2(
        L=nodes_per_side,
        shearValues=shear_values,
    )

    reference_positions = perfect_grid_nodes((nodes_per_side, nodes_per_side))
    current_positions = np.asarray(position_history[0], dtype=float)
    connectivity = _element_connectivity(elements)

    if current_positions.shape != reference_positions.shape:
        raise RuntimeError(
            "miniMTM returned an unexpected number of node positions: "
            f"{current_positions.shape} vs reference {reference_positions.shape}."
        )

    thetas = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=True)
    center = reference_positions.mean(axis=0)

    rotated_reference_positions = np.empty((n_angles, len(reference_positions), 2))
    F_values = np.empty((n_angles, len(connectivity), 2, 2))

    for i, theta in enumerate(thetas):
        ref_rot = _rotate_points(reference_positions, theta, center)
        rotated_reference_positions[i] = ref_rot
        F_values[i] = element_deformation_gradients(
            ref_rot, current_positions, connectivity
        )

    P = ContiEnergy.P_from_F(F_values)
    sigma = ContiEnergy.cauchy_from_F(F_values)

    _assert_equal_across_elements(F_values, "F")
    _assert_equal_across_elements(P, "P")
    _assert_equal_across_elements(sigma, "sigma")
    return {
        "mesh_cells": mesh_cells,
        "nodes_per_side": nodes_per_side,
        "shear": shear,
        "rotation_name": "Reference",
        "thetas": thetas,
        "theta_deg": np.rad2deg(thetas),
        "reference_positions": reference_positions,
        "rotated_reference_positions": rotated_reference_positions,
        "current_positions": current_positions,
        "connectivity": connectivity,
        "F": F_values,
        "P": P,
        "sigma": sigma,
    }


def build_current_rotation_study(
    mesh_cells: int = 2,
    shear: float = 0.8,
    n_angles: int = 181,
) -> dict[str, np.ndarray | float | int]:
    """Rotate the current mesh while keeping the reference mesh fixed.

    The resulting deformation gradients are, up to numerical roundoff,
    ``F(theta) = R(theta) @ F(0)``.  Therefore the right Cauchy--Green tensor
    is fixed while the spatial components of both PK1 and Cauchy stress rotate.
    """
    if mesh_cells < 1:
        raise ValueError(f"mesh_cells must be at least 1, got {mesh_cells}.")
    if n_angles < 2:
        raise ValueError(f"n_angles must be at least 2, got {n_angles}.")

    nodes_per_side = mesh_cells + 1
    shear_values = np.array([shear], dtype=float)
    position_history, elements, _, _ = simpleShearSystem2(
        L=nodes_per_side,
        shearValues=shear_values,
    )

    reference_positions = perfect_grid_nodes((nodes_per_side, nodes_per_side))
    current_positions = np.asarray(position_history[0], dtype=float)
    connectivity = _element_connectivity(elements)

    if current_positions.shape != reference_positions.shape:
        raise RuntimeError(
            "miniMTM returned an unexpected number of node positions: "
            f"{current_positions.shape} vs reference {reference_positions.shape}."
        )

    thetas = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=True)
    center = current_positions.mean(axis=0)

    rotated_current_positions = np.empty(
        (n_angles, len(current_positions), 2), dtype=float
    )
    F_values = np.empty((n_angles, len(connectivity), 2, 2), dtype=float)

    for i, theta in enumerate(thetas):
        current_rot = _rotate_points(current_positions, theta, center)
        rotated_current_positions[i] = current_rot
        F_values[i] = element_deformation_gradients(
            reference_positions, current_rot, connectivity
        )

    P = ContiEnergy.P_from_F(F_values)
    sigma = ContiEnergy.cauchy_from_F(F_values)

    _assert_equal_across_elements(F_values, "F")
    _assert_equal_across_elements(P, "P")
    _assert_equal_across_elements(sigma, "sigma")

    # Rotating the current configuration must preserve C=F^T F.
    C_values = np.einsum("...ji,...jk->...ik", F_values, F_values)
    if not np.allclose(C_values, C_values[0:1], rtol=1e-9, atol=1e-12):
        diff = np.max(np.abs(C_values - C_values[0:1]))
        raise RuntimeError(
            "Current rotation changed C; expected a rigid spatial rotation. "
            f"max |delta C|={diff:.3e}"
        )

    return {
        "mesh_cells": mesh_cells,
        "nodes_per_side": nodes_per_side,
        "shear": shear,
        "rotation_name": "Current",
        "thetas": thetas,
        "theta_deg": np.rad2deg(thetas),
        "reference_positions": reference_positions,
        "current_positions": current_positions,
        "rotated_current_positions": rotated_current_positions,
        "connectivity": connectivity,
        "F": F_values,
        "P": P,
        "sigma": sigma,
    }


def plot_stress_components(
    study: dict[str, np.ndarray | float | int],
    element_index: int = 0,
    save_path: str | None = None,
):
    P = np.asarray(study["P"])
    sigma = np.asarray(study["sigma"])
    theta_deg = np.asarray(study["theta_deg"])
    shear = float(study["shear"])
    rotation_name = str(study.get("rotation_name", "Reference"))

    if not 0 <= element_index < P.shape[1]:
        raise IndexError(
            f"element_index must be in [0, {P.shape[1] - 1}], got {element_index}."
        )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    components = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (i, j) in components:
        axes[0].plot(theta_deg, P[:, element_index, i, j], label=rf"$P_{{{i+1}{j+1}}}$")
        axes[1].plot(
            theta_deg,
            sigma[:, element_index, i, j],
            label=rf"$\sigma_{{{i+1}{j+1}}}$",
        )

    axes[0].set_ylabel("PK1 stress")
    axes[0].set_title(
        f"First Piola-Kirchhoff stress, element {element_index}, "
        f"simple shear gamma={shear:.3g}"
    )
    axes[1].set_ylabel("Cauchy stress")
    axes[1].set_xlabel(f"{rotation_name} rotation (deg)")
    axes[1].set_title(
        f"Cauchy stress for the same element under {rotation_name.lower()} rotation"
    )

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)

    fig.tight_layout()

    if save_path is not None:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes


def _set_mesh_lines(lines, coords: np.ndarray, connectivity: np.ndarray) -> None:
    for line, node_ids in zip(lines, connectivity):
        triangle = coords[node_ids]
        closed = np.vstack([triangle, triangle[0]])
        line.set_data(closed[:, 0], closed[:, 1])


def _make_animation_writer(save_path: str, fps: int):
    ext = os.path.splitext(save_path)[1].lower()
    if ext == ".gif":
        return animation.PillowWriter(fps=fps)
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is not available. Use a .gif save path instead.")
    return animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )


def animate_reference_rotation(
    study: dict[str, np.ndarray | float | int],
    element_index: int = 0,
    save_path: str | None = None,
    fps: int = 20,
):
    theta_deg = np.asarray(study["theta_deg"])
    rotated_reference_positions = np.asarray(study["rotated_reference_positions"])
    current_positions = np.asarray(study["current_positions"])
    connectivity = np.asarray(study["connectivity"])
    shear = float(study["shear"])
    mesh_cells = int(study["mesh_cells"])

    if not 0 <= element_index < connectivity.shape[0]:
        raise IndexError(
            f"element_index must be in [0, {connectivity.shape[0] - 1}], got {element_index}."
        )

    all_points = np.concatenate(
        [rotated_reference_positions.reshape(-1, 2), current_positions],
        axis=0,
    )
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    span = np.max(maxs - mins)
    pad = 0.1 * span if span > 0 else 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ref_ax, cur_ax = axes

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(mins[0] - pad, maxs[0] + pad)
        ax.set_ylim(mins[1] - pad, maxs[1] + pad)
        ax.grid(True, alpha=0.3)

    ref_ax.set_title("Reference mesh")
    cur_ax.set_title("Current mesh")
    ref_ax.set_xlabel("x")
    ref_ax.set_ylabel("y")
    cur_ax.set_xlabel("x")

    ref_lines = []
    cur_lines = []
    for e_idx in range(connectivity.shape[0]):
        is_focus = e_idx == element_index
        style = {
            "color": "tab:orange" if is_focus else "0.7",
            "linewidth": 2.5 if is_focus else 1.0,
            "zorder": 3 if is_focus else 1,
        }
        ref_line, = ref_ax.plot([], [], **style)
        cur_line, = cur_ax.plot([], [], **style)
        ref_lines.append(ref_line)
        cur_lines.append(cur_line)

    ref_points = ref_ax.scatter([], [], c="tab:blue", s=25, zorder=4)
    cur_points = cur_ax.scatter(
        current_positions[:, 0],
        current_positions[:, 1],
        c="tab:blue",
        s=25,
        zorder=4,
    )

    _set_mesh_lines(cur_lines, current_positions, connectivity)

    def update(frame: int):
        coords = rotated_reference_positions[frame]
        _set_mesh_lines(ref_lines, coords, connectivity)
        ref_points.set_offsets(coords)
        fig.suptitle(
            f"{mesh_cells}x{mesh_cells} cell mesh, gamma={shear:.3g}, reference rotation={theta_deg[frame]:.1f} deg"
        )
        return [*ref_lines, *cur_lines, ref_points, cur_points]

    update(0)
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(theta_deg),
        interval=1000 / fps,
        blit=False,
    )

    if save_path is not None:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        writer = _make_animation_writer(save_path, fps=fps)
        anim.save(save_path, writer=writer, dpi=150)

    return fig, anim


def _default_animation_path() -> str:
    ext = ".mp4" if animation.writers.is_available("ffmpeg") else ".gif"
    return os.path.join("Plots", f"rotation_of_reference_mesh{ext}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rotate the reference mesh while keeping one simple-shear current mesh fixed, "
            "then plot PK1 and Cauchy stress components."
        )
    )
    parser.add_argument("--mesh-cells", type=int, default=2)
    parser.add_argument("--shear", type=float, default=0.8)
    parser.add_argument("--n-angles", type=int, default=181)
    parser.add_argument("--element", type=int, default=0)
    parser.add_argument(
        "--stress-path",
        default=os.path.join("Plots", "rotation_of_reference_stress_components.pdf"),
    )
    parser.add_argument(
        "--animation-path",
        default=_default_animation_path(),
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    study = build_rotation_study(
        mesh_cells=args.mesh_cells,
        shear=args.shear,
        n_angles=args.n_angles,
    )

    stress_fig, _ = plot_stress_components(
        study,
        element_index=args.element,
        save_path=args.stress_path,
    )
    animation_fig, _ = animate_reference_rotation(
        study,
        element_index=args.element,
        save_path=args.animation_path,
        fps=args.fps,
    )

    sigma = np.asarray(study["sigma"])[:, args.element]
    sigma_variation = np.max(np.abs(sigma - sigma[:1]))
    print(f"Saved stress plot to {args.stress_path}")
    print(f"Saved mesh animation to {args.animation_path}")
    print(f"Max Cauchy component variation over rotation: {sigma_variation:.3e}")

    if args.show:
        plt.show()
    else:
        plt.close(stress_fig)
        plt.close(animation_fig)


if __name__ == "__main__":
    main()
