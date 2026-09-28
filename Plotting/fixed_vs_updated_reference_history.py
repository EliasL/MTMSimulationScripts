"""Compare fixed and refreshed element references through a reconnection history.

The two columns are loaded from separate archived 8x8 double-dislocation
simulations.  The left run suppresses reference-element replacement during
edge flips; the right run uses the normal reference update.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Plotting.fixed_reference_shear_rotation_illustration import (
    as_matrix,
    as_triangle,
    matrix_latex,
)
from Plotting.matrix_visualization import draw_matrix_columns
from Plotting.mesh_plotting import MeshFigure, MeshStyle


DEFAULT_OUTPUT_DIR = ROOT / "Plots" / "reconnecting_reference"
DEFAULT_NORMAL_DIR = (
    ROOT
    / "MTS2D_reversibility_dump"
    / "test_data"
    / "doubleDislocation8x8UpdatedReferenceRerun"
)
DEFAULT_FIXED_DIR = (
    ROOT
    / "MTS2D_reversibility_dump"
    / "test_data"
    / "doubleDislocation8x8FixedReferenceInspection"
)
TRIANGLE = np.array([[0, 1, 2]], dtype=int)

TEXT = "#000000"
MUTED = "#000000"
GRID = "#CBD2D8"
REFERENCE = "#087E8B"
CURRENT = "#D95F02"
UPDATED = "#000000"
STALE = "#000000"
COLUMN_COLORS = (REFERENCE, CURRENT)
MATRIX_LIMITS = (-1.42, 1.42)


def bottom_aligned(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float)
    aligned = nodes.copy()
    aligned[:, 0] -= aligned[:, 0].mean()
    aligned[:, 1] -= aligned[:, 1].min() + 0.88
    return aligned


def load_payload(history_path: Path) -> dict:
    payload = json.loads(history_path.read_text())
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{history_path} does not contain non-empty 'rows' data.")
    return payload


def snapshot_data(
    row: dict, *, label: str, stage: str = "after"
) -> dict[str, np.ndarray]:
    after = row.get(stage, row)
    if not isinstance(after, dict):
        raise ValueError(
            f"{label} row at gamma={row.get('gamma')} has no {stage} snapshot."
        )
    data = {
        "reference_nodes": as_triangle(
            after["reference_nodes"], name=f"{label} reference triangle"
        ),
        "current_nodes": as_triangle(
            after["current_nodes"], name=f"{label} current triangle"
        ),
        "F": as_matrix(after["F"], name=f"{label} F"),
        "F_E": as_matrix(after["F_E"], name=f"{label} F_E"),
        "H": as_matrix(after["H"], name=f"{label} H"),
        "T_P": as_matrix(after["T_P"], name=f"{label} T_P"),
        "T": as_matrix(after["T"], name=f"{label} T"),
    }
    if not np.allclose(data["T"], data["F_E"] @ data["T_P"], atol=2e-12):
        raise ValueError(f"{label} T does not equal F_E @ T_P at gamma={row.get('gamma')}.")
    return data


def load_histories(
    normal_history_path: Path, fixed_history_path: Path, *, edge_flips_only: bool
) -> list[dict[str, object]]:
    normal_payload = load_payload(normal_history_path)
    fixed_payload = load_payload(fixed_history_path)
    if "Fixed-reference" not in str(fixed_payload.get("label", "")):
        raise ValueError(f"{fixed_history_path} is not labeled as a fixed-reference run.")

    normal_rows = normal_payload["rows"]
    fixed_rows = fixed_payload["rows"]
    if len(normal_rows) != len(fixed_rows):
        raise ValueError("Normal and fixed-reference histories have different lengths.")

    states = []
    fixed_reference_initial = None
    previous_normal_reference = None
    for normal_row, fixed_row in zip(normal_rows, fixed_rows):
        normal_gamma = float(normal_row["gamma"])
        fixed_gamma = float(fixed_row["gamma"])
        if not np.isclose(normal_gamma, fixed_gamma):
            raise ValueError(
                f"History gamma mismatch: normal={normal_gamma}, fixed={fixed_gamma}."
            )
        fixed_or_normal_changed = (
            bool(normal_row.get("element_changed", False)),
            bool(fixed_row.get("element_changed", False)),
        )
        if edge_flips_only and fixed_or_normal_changed[0] != fixed_or_normal_changed[1]:
            raise ValueError(
                f"Paired edge-flip mismatch at gamma={fixed_gamma:g}: "
                f"normal={fixed_or_normal_changed[0]}, fixed={fixed_or_normal_changed[1]}."
            )
        is_initial_condition = edge_flips_only and np.isclose(fixed_gamma, 0.0)
        if edge_flips_only and not fixed_or_normal_changed[0] and not is_initial_condition:
            continue

        stages = (("before", "before edge flip"), ("after", "after edge flip"))
        if is_initial_condition:
            stages = (("after", "initial condition"),)
        if not edge_flips_only:
            stages = (("after", None),)
        for stage, event_phase in stages:
            normal = snapshot_data(normal_row, label="normal", stage=stage)
            fixed = snapshot_data(fixed_row, label="fixed-reference", stage=stage)
            if fixed_reference_initial is None:
                fixed_reference_initial = fixed["reference_nodes"].copy()
            if not np.allclose(fixed["reference_nodes"], fixed_reference_initial):
                raise ValueError(
                    f"Fixed-reference run changed its reference triangle at "
                    f"gamma={fixed_gamma:g}."
                )
            current_difference = float(
                np.max(np.abs(normal["current_nodes"] - fixed["current_nodes"]))
            )
            if current_difference > 5e-4:
                raise ValueError(
                    f"Paired current triangles differ by {current_difference:g} "
                    f"at gamma={fixed_gamma:g} ({stage})."
                )
            normal_reference_changed = (
                previous_normal_reference is not None
                and not np.allclose(normal["reference_nodes"], previous_normal_reference)
            )
            states.append(
                {
                    "gamma": fixed_gamma,
                    "edge_flip": bool(
                        fixed_or_normal_changed[0]
                        or fixed_or_normal_changed[1]
                        or normal_reference_changed
                    ),
                    "event_phase": event_phase,
                    "fixed": fixed,
                    "normal": normal,
                    "current_difference": current_difference,
                }
            )
            previous_normal_reference = normal["reference_nodes"].copy()
    return states


def draw_geometry(ax, reference_nodes: np.ndarray, current_nodes: np.ndarray) -> None:
    reference_style = MeshStyle(
        color=REFERENCE,
        face_alpha=0.035,
        linewidth=2.0,
        linestyle="--",
        node_size=22,
        node_linewidth=0.0,
        zorder=2,
    )
    current_style = MeshStyle(
        color=CURRENT,
        face_alpha=0.12,
        linewidth=2.0,
        linestyle="-",
        node_size=22,
        node_linewidth=0.0,
        zorder=3,
    )
    mesh = MeshFigure(ax)
    mesh.draw_mesh(bottom_aligned(reference_nodes), TRIANGLE, style=reference_style)
    mesh.draw_mesh(bottom_aligned(current_nodes), TRIANGLE, style=current_style)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_matrix(ax, matrix: np.ndarray, *, symbol: str, exact: bool) -> None:
    draw_matrix_columns(
        ax,
        matrix,
        limits=MATRIX_LIMITS,
        colors=COLUMN_COLORS,
        linestyles=("-", "-"),
        linewidth=2.5,
        mutation_scale=18,
        grid_color=GRID,
        show_ticks=False,
    )
    ax.set_title(
        rf"${symbol}$", fontsize=10.5, pad=7, fontweight="semibold", color=TEXT
    )
    equality = "=" if exact else r"\simeq"
    ax.text(
        0.5,
        -0.16,
        rf"${equality} {matrix_latex(matrix)}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.8,
        color=TEXT,
    )


def draw_history_column(
    cell,
    state: dict[str, object],
    *,
    fixed_reference: bool,
    first_row: bool,
) -> object:
    axes = cell.subgridspec(
        1,
        4,
        width_ratios=(1.55, 1.0, 1.0, 1.0),
        wspace=0.05,
    ).subplots()
    geometry_axis, F_axis, H_axis, T_axis = axes

    if fixed_reference:
        snapshot = state["fixed"]
        symbols = (r"F_{\rm fixed}", r"H_{\rm fixed}", r"T_{\rm fixed}")
        heading_color = STALE
    else:
        snapshot = state["normal"]
        symbols = (r"F", r"H", r"T")
        heading_color = UPDATED
    if not isinstance(snapshot, dict):
        raise TypeError("History state snapshot must be a dictionary.")

    draw_geometry(
        geometry_axis,
        reference_nodes=np.asarray(snapshot["reference_nodes"]),
        current_nodes=np.asarray(snapshot["current_nodes"]),
    )
    if first_row:
        geometry_axis.set_title(
            r"reference $\mathbf{X}$ and current $\mathbf{x}$",
            fontsize=9.2,
            pad=7,
            color=heading_color,
            fontweight="semibold",
        )
    draw_matrix(
        F_axis, np.asarray(snapshot["F"]), symbol=symbols[0], exact=False
    )
    draw_matrix(
        H_axis, np.asarray(snapshot["H"]), symbol=symbols[1], exact=True
    )
    draw_matrix(
        T_axis, np.asarray(snapshot["T"]), symbol=symbols[2], exact=False
    )
    return geometry_axis


def make_figure(states: list[dict[str, object]], *, element_index: int):
    if not states:
        raise ValueError("At least one history state is required.")

    figure = plt.figure(
        figsize=(0.64 * 15.8, 0.64 * (2.6 * len(states) + 3.1)), facecolor="white"
    )
    figure.text(
        0.295,
        0.965,
        r"Fixed initial reference $\mathbf{X}^0$",
        ha="center",
        va="top",
        fontsize=13,
        color=TEXT,
        fontweight="semibold",
    )
    figure.text(
        0.30,
        0.940,
        r"actual rerun with edge-flip reference replacement suppressed",
        ha="center",
        va="top",
        fontsize=9.5,
        color=TEXT,
    )
    figure.text(
        0.745,
        0.965,
        r"Reference refreshed at each edge flip",
        ha="center",
        va="top",
        fontsize=13,
        color=TEXT,
        fontweight="semibold",
    )
    figure.text(
        0.75,
        0.940,
        r"normal simulation rerun",
        ha="center",
        va="top",
        fontsize=9.5,
        color=TEXT,
    )

    grid = figure.add_gridspec(
        len(states),
        2,
        left=0.085,
        right=0.99,
        top=0.875,
        bottom=0.035,
        hspace=0.31,
        wspace=0.05,
    )
    for index, state in enumerate(states):
        fixed_geometry_axis = draw_history_column(
            grid[index, 0], state, fixed_reference=True, first_row=index == 0
        )
        draw_history_column(grid[index, 1], state, fixed_reference=False, first_row=index == 0)
        gamma = float(state["gamma"])
        event_phase = state.get("event_phase")
        if event_phase is not None:
            event_text = str(event_phase)
            event_color = TEXT
            bbox = dict(boxstyle="round,pad=0.28", facecolor="#F2F8F3", edgecolor=UPDATED)
        else:
            event_text = ""
            event_color = TEXT
            bbox = None
        fixed_geometry_axis.text(
            -0.42,
            0.52,
            rf"$\gamma={gamma:g}$" + (f"\n{event_text}" if event_text else ""),
            transform=fixed_geometry_axis.transAxes,
            ha="center",
            va="center",
            fontsize=10 if event_text else 11,
            color=event_color,
            fontweight="semibold",
            bbox=bbox,
            clip_on=False,
        )

    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--element", type=int, default=48)
    parser.add_argument("--normal-history", type=Path)
    parser.add_argument("--fixed-history", type=Path)
    parser.add_argument("--output-stem", type=Path)
    parser.add_argument("--edge-flips-only", action="store_true")
    args = parser.parse_args()

    if args.element < 0:
        raise ValueError("--element must be non-negative.")
    history_name = (
        f"element_{args.element}_edge_flip_history.json"
        if args.edge_flips_only
        else f"element_{args.element}_matrix_history.json"
    )
    normal_history = args.normal_history or (DEFAULT_NORMAL_DIR / history_name)
    fixed_history = args.fixed_history or (DEFAULT_FIXED_DIR / history_name)
    output_stem = args.output_stem or (
        DEFAULT_OUTPUT_DIR
        / f"element_{args.element}_fixed_vs_updated_reference_history"
    )
    states = load_histories(
        normal_history, fixed_history, edge_flips_only=args.edge_flips_only
    )
    figure = make_figure(states, element_index=args.element)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
