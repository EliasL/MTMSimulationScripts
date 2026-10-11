"""Paper-style mesh-locking figure for the double-dislocation test.

The left panel summarizes the piecewise boundary loading.  The right panel
compares the same mesh without reconnection and with edge
flipping at four horizontal/vertical displacement pairs ``(u_x, u_y)``.
The exported scalar load selects snapshots: u_x = min(load, switch_load)
and u_y = max(load - switch_load, 0) for this horizontal-first protocol.

Run this file directly to write PNG and PDF copies to ``Plots/``.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from Plotting.dataFunctions import VTUData, infer_strain_from_vtu, resolve_vtu_files
from Plotting.mesh_plotting import (
    MeshFigure,
    MeshStyle,
    draw_mesh_nodes,
)
from Plotting.pyplotFunctions import plot_mesh


DEFAULT_DATA_ROOT = ROOT / "GeneratedData" / "MTS2D_output"
DEFAULT_LOADS = (0.4, 1.0, 1.4, 2.0)
DEFAULT_SIZE = 10
DEFAULT_SWITCH_LOAD = 1.0

FIXED_COLOR = "#2F343B"
X_LOAD_COLOR = "#0072B2"
Y_LOAD_COLOR = "#D55E00"
FREE_COLOR = "#009E73"
MESH_COLOR = "#7F858D"


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def find_double_dislocation_run(
    data_root: Path,
    *,
    edge_flip: bool,
    size: int = DEFAULT_SIZE,
    switch_load: float = DEFAULT_SWITCH_LOAD,
    mesh_diagonal: str = "minor",
) -> Path:
    """Find the requested mesh orientation without inferring it from a label."""

    data_root = Path(data_root)
    prefix = f"doubleDislocationTest,s{size}x{size}"
    switch_token = f"GP3{switch_load:g}"
    matches = [
        path
        for path in data_root.iterdir()
        if path.is_dir()
        and path.name.startswith(prefix)
        and f"meshDiagonal{mesh_diagonal}" in path.name
        and switch_token in path.name
        and ("edgeFlip" in path.name) == edge_flip
        and (path / "collection.pvd").exists()
    ]
    if len(matches) != 1:
        method = "edge flipping" if edge_flip else "no reconnection"
        raise ValueError(
            f"Expected one {size}x{size} {mesh_diagonal}-mesh run with {method} and "
            f"GP3={switch_load:g}; found {len(matches)} in {data_root}."
        )
    return matches[0]


def select_vtu_at_load(run_path: Path, target_load: float) -> Path:
    """Return the VTU snapshot whose exported load matches ``target_load``."""

    candidates = []
    for vtu_file in resolve_vtu_files(run_path):
        load = infer_strain_from_vtu(vtu_file)
        if load is not None and np.isfinite(load):
            candidates.append((abs(float(load) - target_load), float(load), Path(vtu_file)))
    if not candidates:
        raise FileNotFoundError(f"No load-labelled VTU files found in {run_path}.")
    error, load, vtu_file = min(candidates, key=lambda item: item[0])
    if error > 5e-7:
        raise ValueError(
            f"Closest snapshot to protocol load {target_load:g} is {load:g} "
            f"in {run_path}."
        )
    return vtu_file


def validate_horizontal_first_runs(
    fixed_run: Path, reconnecting_run: Path, *, switch_load: float = 1.0,
    final_load: float = 4.0, mesh_diagonal: str | None = None,
    alternating: bool = False,
) -> dict[str, str]:
    """Reject comparisons with different settings or incorrectly labelled loads."""
    configs = []
    for path in (fixed_run, reconnecting_run):
        config = {}
        for line in (Path(path) / "config.conf").read_text().splitlines():
            line = line.split("#", 1)[0]
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
        configs.append(config)
        if float(config.get("GP1", 0)) != 0 or float(config.get("GP2", 0)) != (2 if alternating else 0):
            raise ValueError("The simulation loading protocol differs from the requested figure.")
        actual_switch = float(config.get("GP3", 0)) or 1.0
        if not np.isclose(actual_switch, switch_load):
            raise ValueError("The illustrated loading switch differs from the simulation.")
        if float(config["maxLoad"]) < final_load:
            raise ValueError("The simulation does not cover the illustrated loading range.")
        if mesh_diagonal is not None and config["meshDiagonal"] != mesh_diagonal:
            raise ValueError(f"Requested {mesh_diagonal} mesh, found {config['meshDiagonal']} in {path}.")
        initial = VTUData(select_vtu_at_load(Path(path), 0.0))
        triangles = initial.get_connectivity()
        shared = sorted(set(triangles[0]) & set(triangles[1]))
        if len(shared) != 2:
            raise ValueError("Cannot identify the initial square's shared diagonal.")
        reference = initial.get_reference_nodes()[:, :2]
        diagonal = reference[shared[1]] - reference[shared[0]]
        actual_diagonal = "major" if np.prod(diagonal) < 0 else "minor"
        if actual_diagonal != config["meshDiagonal"]:
            raise ValueError("Initial VTU connectivity disagrees with the configured mesh orientation.")
    matching = ("rows", "cols", "usingPBC", "seed", "QDSD", "initialGuessNoise",
                "meshDiagonal", "energyFunction", "bulkModulus", "startLoad",
                "loadIncrement", "maxLoad", "GP1", "GP2", "GP3", "minimizer",
                "epsR", "LBFGSNrCorrections", "LBFGSScale", "LBFGSEpsg",
                "LBFGSEpsf", "LBFGSEpsx", "LBFGSMaxIterations")
    for key in matching:
        if configs[0].get(key) != configs[1].get(key):
            raise ValueError(f"Comparison runs have different {key} settings.")
    if configs[0]["reconnectionMethod"] != "none" or configs[1]["reconnectionMethod"] != "edgeFlip":
        raise ValueError("Expected a fixed-connectivity run and an edge-flipping run.")
    return configs[0]


def loading_displacements(load: float, switch_load: float = 1.0, *, alternating: bool = False):
    """Map accumulated displacement to the two prescribed boundary components."""
    if alternating:
        cycles = np.floor(load / (2 * switch_load))
        remainder = load - cycles * 2 * switch_load
        return (float(cycles * switch_load + min(remainder, switch_load)),
                float(cycles * switch_load + max(remainder - switch_load, 0)))
    return min(load, switch_load), max(load - switch_load, 0)


def classify_protocol_nodes(
    reference_nodes: np.ndarray,
    fixed_status: np.ndarray,
) -> dict[str, np.ndarray]:
    """Split constrained boundary nodes by their role in the loading protocol."""

    reference_nodes = np.asarray(reference_nodes, dtype=float)[:, :2]
    fixed_status = np.asarray(fixed_status).ravel().astype(bool)
    if len(reference_nodes) != len(fixed_status):
        raise ValueError("reference_nodes and fixed_status must have matching lengths.")

    x = reference_nodes[:, 0]
    y = reference_nodes[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    atol = 1e-7 * max(1.0, xmax - xmin, ymax - ymin)

    on_left = np.isclose(x, xmin, atol=atol)
    on_bottom = np.isclose(y, ymin, atol=atol)
    x_loaded = fixed_status & on_left & (y > ymid)
    y_loaded = fixed_status & on_bottom & (x > xmid)
    fixed = fixed_status & ~x_loaded & ~y_loaded
    top = np.isclose(y, ymax, atol=atol)
    right = np.isclose(x, xmax, atol=atol)

    return {
        "fixed": np.flatnonzero(fixed),
        "x_loaded": np.flatnonzero(x_loaded),
        "y_loaded": np.flatnonzero(y_loaded),
        "top": np.flatnonzero(top),
        "right": np.flatnonzero(right),
    }


def _chain_edges(nodes: np.ndarray, node_ids: np.ndarray, coordinate: int) -> np.ndarray:
    """Connect a boundary-node subset in coordinate order."""

    node_ids = np.asarray(node_ids, dtype=int)
    order = np.argsort(nodes[node_ids, coordinate])
    ordered = node_ids[order]
    if len(ordered) < 2:
        return np.empty((0, 2), dtype=int)
    return np.column_stack([ordered[:-1], ordered[1:]])


def _draw_loading_protocol(
    ax: plt.Axes,
    reference_nodes: np.ndarray,
    connectivity: np.ndarray,
    fixed_status: np.ndarray,
    *,
    switch_load: float,
    final_load: float,
    alternating: bool = False,
) -> None:
    reference_nodes = np.asarray(reference_nodes, dtype=float)[:, :2]
    groups = classify_protocol_nodes(reference_nodes, fixed_status)
    xmin, ymin = np.min(reference_nodes, axis=0)
    xmax, ymax = np.max(reference_nodes, axis=0)
    span = max(float(xmax - xmin), float(ymax - ymin))

    mesh = MeshFigure(ax)
    mesh.draw_mesh(
        reference_nodes,
        connectivity,
        style=MeshStyle(
            color=MESH_COLOR,
            face_alpha=0.0,
            edge_alpha=0.52,
            linewidth=0.36,
            draw_faces=False,
            draw_nodes=False,
            zorder=0,
        ),
    )

    fixed_edges = np.vstack(
        [
            _chain_edges(
                reference_nodes,
                groups["fixed"][
                    np.isclose(reference_nodes[groups["fixed"], 1], ymin)
                ],
                0,
            ),
            _chain_edges(
                reference_nodes,
                groups["fixed"][
                    np.isclose(reference_nodes[groups["fixed"], 0], xmin)
                ],
                1,
            ),
        ]
    )
    mesh.draw_edges(
        reference_nodes,
        fixed_edges,
        color=FIXED_COLOR,
        linewidth=2.2,
        zorder=4,
    )
    mesh.draw_edges(
        reference_nodes,
        _chain_edges(reference_nodes, groups["top"], 0),
        color=FREE_COLOR,
        linewidth=2.0,
        linestyle=(0, (4.0, 2.5)),
        zorder=3,
    )
    mesh.draw_edges(
        reference_nodes,
        _chain_edges(reference_nodes, groups["right"], 1),
        color=FREE_COLOR,
        linewidth=2.0,
        linestyle=(0, (4.0, 2.5)),
        zorder=3,
    )

    draw_mesh_nodes(
        ax,
        reference_nodes,
        node_ids=groups["fixed"],
        facecolor=FIXED_COLOR,
        edgecolor="white",
        size=17,
        linewidth=0.45,
        zorder=6,
    )
    draw_mesh_nodes(
        ax,
        reference_nodes,
        node_ids=groups["x_loaded"],
        facecolor=X_LOAD_COLOR,
        edgecolor="white",
        size=21,
        linewidth=0.55,
        zorder=7,
    )
    draw_mesh_nodes(
        ax,
        reference_nodes,
        node_ids=groups["y_loaded"],
        facecolor=Y_LOAD_COLOR,
        edgecolor="white",
        size=21,
        linewidth=0.55,
        zorder=7,
    )

    for node_id in groups["x_loaded"]:
        y = reference_nodes[node_id, 1]
        ax.annotate(
            "",
            xy=(xmin - 0.015 * span, y),
            xytext=(xmin - 0.115 * span, y),
            arrowprops={"arrowstyle": "-|>", "color": X_LOAD_COLOR, "lw": 1.5},
            zorder=8,
        )
    for node_id in groups["y_loaded"]:
        x = reference_nodes[node_id, 0]
        ax.annotate(
            "",
            xy=(x, ymin - 0.015 * span),
            xytext=(x, ymin - 0.115 * span),
            arrowprops={"arrowstyle": "-|>", "color": Y_LOAD_COLOR, "lw": 1.5},
            zorder=8,
        )

    ax.text(
        xmin + 0.25 * (xmax - xmin),
        ymin + 0.84 * (ymax - ymin),
        "1. horizontal\n" + rf"$0\leq u_x\leq {switch_load:g}$" + "\n" + r"$u_y=0$",
        color=X_LOAD_COLOR,
        ha="center",
        va="center",
        fontsize=16,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.text(
        xmin + 0.75 * (xmax - xmin),
        ymin + 0.16 * (ymax - ymin),
        "2. vertical\n" + rf"$u_x={switch_load:g}$" + "\n"
        + rf"$0\leq u_y\leq {(switch_load if alternating else max(final_load - switch_load, 0)):g}$",
        color=Y_LOAD_COLOR,
        ha="center",
        va="center",
        fontsize=16,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.text(
        xmin + 0.20 * (xmax - xmin),
        ymin + 0.12 * (ymax - ymin),
        "fixed nodes",
        color=FIXED_COLOR,
        ha="center",
        va="center",
        fontsize=16,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.text(
        xmin + 0.76 * (xmax - xmin),
        ymin + 0.88 * (ymax - ymin),
        "free edges",
        color=FREE_COLOR,
        ha="center",
        va="center",
        fontsize=16,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )

    if alternating:
        ux, uy = loading_displacements(final_load, switch_load, alternating=True)
        ax.text(xmin + .5 * span, ymin + .5 * span,
                "Repeat horizontal / vertical\n" + rf"to $u_x={ux:g},\;u_y={uy:g}$",
                ha="center", va="center", fontsize=14,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 2})
    ax.text(-0.04, 1.045, r"\textbf{(a)}", transform=ax.transAxes, fontsize=16)
    mesh.configure_axis(
        xlim=(xmin - 0.16 * span, xmax + 0.035 * span),
        ylim=(ymin - 0.16 * span, ymax + 0.035 * span),
        equal_aspect=True,
        hide_axes=True,
    )


def _nice_energy_max(maximum: float) -> float:
    if not np.isfinite(maximum) or maximum <= 0:
        return 1.0
    scale = 10.0 ** np.floor(np.log10(maximum))
    return float(np.ceil(2.0 * maximum / scale) * 0.5 * scale)


def relabel_mesh_locking_pdf(
    source: Path,
    output_stem: Path,
    *,
    switch_load: float = DEFAULT_SWITCH_LOAD,
    final_load: float = 4.0,
    energy_max: float | None = None,
) -> None:
    """Reuse original vector panels when the corresponding VTUs are unavailable.

    Accept the original gamma-labelled mesh-locking figure. Mesh geometry and
    Energy colours can be remapped from the original 0–0.15 coolwarm scale.
    This uses the centres of the original 256 colour bins, so the recovered
    energies retain the source palette's quantization rather than VTU precision.
    """
    import fitz

    original = fitz.open(source)
    page = original[0]
    colorbar_rect = None
    if energy_max is not None:
        if not np.isfinite(energy_max) or energy_max <= 0:
            raise ValueError("Energy maximum must be finite and positive.")
        image_rects = [rect for item in page.get_images(full=True)
                       for rect in page.get_image_rects(item[0])]
        colorbar_rect = next(rect for rect in image_rects
                            if rect.height > 10 * rect.width)
        cmap = mpl.colormaps["coolwarm"]
        palette = cmap(np.arange(256))[:, :3]

        def remap_color(match):
            rgb = np.array([float(value) for value in match.groups()[:3]])
            distances = np.sum((palette - rgb) ** 2, axis=1)
            index = int(np.argmin(distances))
            if distances[index] > 1e-16:
                return match.group(0)
            energy = 0.15 * (index + 0.5) / 256
            mapped = cmap(min(energy / energy_max, 1.0))[:3]
            return (" ".join(f"{channel:.10f}" for channel in mapped).encode()
                    + b" " + match.group(4))

        pattern = rb"(?<!\S)([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(rg|RG)\b"
        for xref in page.get_contents():
            original.update_stream(xref, re.sub(pattern, remap_color,
                                                original.xref_stream(xref)))
    lines = [
        (line, unicodedata.normalize(
            "NFKC", "".join(span["text"] for span in line["spans"])
        ))
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
    ]
    titles = [(line, text) for line, text in lines if "γ" in text
              and not text.startswith(("1.", "2."))]
    if len(titles) != 4:
        raise ValueError("Expected the original figure with four gamma-labelled titles.")
    width, height = page.rect.width + 18, page.rect.height + 4
    _configure_matplotlib()
    overlay = plt.figure(figsize=(width / 72, height / 72))
    title_index = 0
    protocol_box = {"facecolor": "white", "edgecolor": "none",
                    "alpha": 0.78, "pad": 1.5}
    for line, text in lines:
        box = fitz.Rect(line["bbox"])
        x, y = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        rotation, color, size, bbox = 0, "black", 16, None
        if text.startswith("1."):
            text = ("1. horizontal\n" + rf"$0\leq u_x\leq {switch_load:g}$"
                    + "\n" + r"$u_y=0$")
            y += 8
            color, bbox = X_LOAD_COLOR, protocol_box
        elif text.startswith("2."):
            text = ("2. vertical\n" + rf"$u_x={switch_load:g}$" + "\n"
                    + rf"$0\leq u_y\leq {final_load - switch_load:g}$")
            y -= 12
            color, bbox = Y_LOAD_COLOR, protocol_box
        elif "fixed" in text and "nodes" in text:
            color, bbox = FIXED_COLOR, protocol_box
        elif "free" in text and "edges" in text:
            color, bbox = FREE_COLOR, protocol_box
        elif "γ" in text:
            load = DEFAULT_LOADS[title_index]
            letter = "ABCD"[title_index]
            ux, uy = min(load, switch_load), max(load - switch_load, 0)
            text = rf"$\mathbf{{{letter}}}\quad u_x={ux:g},\;u_y={uy:g}$"
            size = 15
            title_index += 1
        elif "No" in text and "reconnection" in text:
            text, rotation, size = "Fixed connectivity", 90, 15
        elif "With" in text and "reconnecting" in text:
            text, rotation, size = "Reconnection", 90, 15
        elif text in ("(a)", "(b)"):
            pass
        elif text == "Ei":
            text, size = r"$E_i$", 15
        else:
            size = 14  # Shared colour-bar tick labels.
            if energy_max is not None:
                page.add_redact_annot(box, fill=False)
                continue
        overlay.text(x / width, 1 - y / height, text, ha="center", va="center",
                     rotation=rotation, fontsize=size, color=color, bbox=bbox)
        page.add_redact_annot(box, fill=False)
    page.apply_redactions(images=0, graphics=0, text=0)
    if colorbar_rect is not None:
        rect = colorbar_rect
        overlay.patches.append(mpl.patches.Rectangle(
            ((rect.x0 - 2) / width, (height - rect.y1 - 12) / height),
            (width - rect.x0 + 2) / width, (rect.height + 14) / height,
            transform=overlay.transFigure, facecolor="white", edgecolor="none",
            zorder=-1,
        ))
        bar_ax = overlay.add_axes((rect.x0 / width, (height - rect.y1) / height,
                                   rect.width / width, rect.height / height))
        bar = overlay.colorbar(ScalarMappable(norm=Normalize(0, energy_max),
                                              cmap="coolwarm"), cax=bar_ax,
                               ticks=np.linspace(0, energy_max, 7), format="%.2f")
        bar.ax.tick_params(labelsize=14, pad=3)
        bar.outline.set_linewidth(0.6)
    stream = io.BytesIO()
    overlay.savefig(stream, format="pdf", transparent=True)
    plt.close(overlay)
    lettering = fitz.open(stream=stream.getvalue(), filetype="pdf")
    result = fitz.open()
    output = result.new_page(width=width, height=height)
    output.show_pdf_page(fitz.Rect(0, 0, page.rect.width, page.rect.height), original, 0)
    output.show_pdf_page(output.rect, lettering, 0)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_stem.with_suffix(".pdf"), garbage=4, deflate=True)
    output.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72)).save(
        output_stem.with_suffix(".png")
    )
    result.close()
    lettering.close()
    original.close()


def make_mesh_locking_figure(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    loads: tuple[float, ...] = DEFAULT_LOADS,
    size: int = DEFAULT_SIZE,
    switch_load: float = DEFAULT_SWITCH_LOAD,
    energy_limits: tuple[float, float] | None = None,
    final_load: float = 4.0,
    no_reconnection_run: Path | None = None,
    edge_flipping_run: Path | None = None,
    mesh_diagonal: str = "minor",
    alternating: bool = False,
) -> tuple[plt.Figure, dict[str, object]]:
    """Build the complete loading-protocol and mesh-evolution figure."""

    if len(loads) != 4:
        raise ValueError(f"Expected four load snapshots, got {len(loads)}.")
    if final_load < max(loads) or final_load < switch_load:
        raise ValueError("Final protocol load must include all snapshots and the switch.")
    _configure_matplotlib()

    if (no_reconnection_run is None) != (edge_flipping_run is None):
        raise ValueError("Supply both explicit run paths, or neither.")
    run_paths = {
        "no_reconnection": find_double_dislocation_run(
            data_root, edge_flip=False, size=size, switch_load=switch_load,
            mesh_diagonal=mesh_diagonal,
        ),
        "edge_flipping": find_double_dislocation_run(
            data_root, edge_flip=True, size=size, switch_load=switch_load,
            mesh_diagonal=mesh_diagonal,
        ),
    } if no_reconnection_run is None else {
        "no_reconnection": Path(no_reconnection_run),
        "edge_flipping": Path(edge_flipping_run),
    }
    validate_horizontal_first_runs(
        run_paths["no_reconnection"], run_paths["edge_flipping"],
        switch_load=switch_load, final_load=final_load, mesh_diagonal=mesh_diagonal,
        alternating=alternating,
    )
    snapshots = {
        method: [select_vtu_at_load(path, load) for load in loads]
        for method, path in run_paths.items()
    }

    all_energies = np.concatenate(
        [
            VTUData(vtu_file).get_energy_field().ravel()
            for method_snapshots in snapshots.values()
            for vtu_file in method_snapshots
        ]
    )
    if energy_limits is None:
        energy_limits = (0.0, _nice_energy_max(float(np.nanmax(all_energies))))

    all_nodes = np.concatenate(
        [
            VTUData(vtu_file).get_nodes()[:, :2]
            for method_snapshots in snapshots.values()
            for vtu_file in method_snapshots
        ],
        axis=0,
    )
    panel_min = float(np.min(all_nodes))
    panel_max = float(np.max(all_nodes))
    panel_padding = 0.035 * (panel_max - panel_min)
    panel_limits = (panel_min - panel_padding, panel_max + panel_padding)

    initial_vtu = select_vtu_at_load(run_paths["no_reconnection"], 0.0)
    initial_data = VTUData(initial_vtu)

    figure = plt.figure(figsize=(15.8, 6.15))
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 2.0),
        left=0.025,
        right=0.992,
        bottom=0.105,
        top=0.94,
        wspace=0.055,
    )
    protocol_ax = figure.add_subplot(outer[0, 0])
    evolution_grid = outer[0, 1].subgridspec(
        2,
        5,
        width_ratios=(1.0, 1.0, 1.0, 1.0, 0.065),
        hspace=0.075,
        wspace=0.05,
    )
    evolution_axes = np.empty((2, 4), dtype=object)
    for row in range(2):
        for column in range(4):
            evolution_axes[row, column] = figure.add_subplot(
                evolution_grid[row, column]
            )
    colorbar_ax = figure.add_subplot(evolution_grid[:, 4])

    _draw_loading_protocol(
        protocol_ax,
        initial_data.get_reference_nodes(),
        initial_data.get_connectivity(),
        initial_data.get_fixed_status(),
        switch_load=switch_load,
        final_load=final_load,
        alternating=alternating,
    )

    method_rows = (
        ("no_reconnection", "Fixed connectivity"),
        ("edge_flipping", "Reconnection"),
    )
    cmap = None
    norm = None
    for row, (method, row_label) in enumerate(method_rows):
        for column, (load, vtu_file) in enumerate(zip(loads, snapshots[method])):
            ax = evolution_axes[row, column]
            _, cmap, norm = plot_mesh(
                str(vtu_file),
                e_lims=energy_limits,
                mesh_property="energy",
                ax=ax,
                add_colorbar=False,
                add_rombus=False,
            )
            ax.set_xlim(*panel_limits)
            ax.set_ylim(*panel_limits)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ux, uy = loading_displacements(load, switch_load, alternating=alternating)
                letter = "ABCD"[column]
                ax.set_title(rf"$\mathbf{{{letter}}}\quad u_x={ux:g},\;u_y={uy:g}$", pad=8)
            if column == 0:
                ax.set_ylabel(row_label, labelpad=5, fontsize=15)

    evolution_axes[0, 0].text(
        -0.13,
        1.08,
        r"\textbf{(b)}",
        transform=evolution_axes[0, 0].transAxes,
        fontsize=16,
    )
    if cmap is None or norm is None:
        raise RuntimeError("No mesh panels were drawn.")
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(*energy_limits), cmap=cmap),
        cax=colorbar_ax,
        orientation="vertical",
    )
    colorbar.ax.set_title(r"$E_i$", pad=6)
    colorbar.ax.tick_params(labelsize=14)
    colorbar.outline.set_linewidth(0.6)

    metadata = {
        "run_paths": run_paths,
        "snapshots": snapshots,
        "energy_limits": energy_limits,
        "panel_limits": panel_limits,
        "protocol_axis": protocol_ax,
        "evolution_axes": evolution_axes,
    }
    return figure, metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the double-dislocation mesh-locking comparison."
    )
    parser.add_argument(
        "--source-figure",
        type=Path,
        help="Reuse vector panels from the original gamma-labelled PDF instead of VTUs.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing MTS2D output folders.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "Plots" / "mesh_locking_double_dislocation",
        help="Output path without an extension.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="Number of nodes along each side of the square mesh.",
    )
    parser.add_argument(
        "--no-reconnection-run",
        type=Path,
        help="Explicit no-reconnection run folder (supply both run paths).",
    )
    parser.add_argument(
        "--edge-flipping-run",
        type=Path,
        help="Explicit edge-flipping run folder (supply both run paths).",
    )
    parser.add_argument(
        "--alternating", action="store_true",
        help="Alternate horizontal and vertical loading every switch-load interval.",
    )
    parser.add_argument(
        "--mesh-diagonal", choices=("major", "minor"), default="minor",
        help="Solver diagonal orientation, checked against both run configurations.",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        help="Upper limit of the shared energy colour scale (lower limit is zero).",
    )
    parser.add_argument(
        "--final-load",
        type=float,
        default=4.0,
        help="Final protocol load for the loading illustration, independent of snapshots.",
    )
    parser.add_argument(
        "--switch-load",
        type=float,
        default=DEFAULT_SWITCH_LOAD,
        help="Protocol load at which horizontal loading switches to vertical.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.source_figure is not None:
        relabel_mesh_locking_pdf(
            args.source_figure, args.output_stem,
            switch_load=args.switch_load, final_load=args.final_load,
            energy_max=args.energy_max,
        )
        return
    figure, _ = make_mesh_locking_figure(
        args.data_root,
        size=args.size,
        switch_load=args.switch_load,
        final_load=args.final_load,
        energy_limits=None if args.energy_max is None else (0.0, args.energy_max),
        no_reconnection_run=args.no_reconnection_run,
        edge_flipping_run=args.edge_flipping_run,
        mesh_diagonal=args.mesh_diagonal,
        alternating=args.alternating,
    )
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
