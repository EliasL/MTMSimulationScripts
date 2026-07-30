"""Build a real-data flowchart for extracting a truncated power-law exponent.

The slow mode reads the simulation CSVs, recomputes the energy drops and fits,
and caches five plot panels.  The fast mode only recomposes those cached panels,
which makes layout iteration nearly instantaneous.

Examples
--------
Regenerate the analysis and every plot panel::

    .venv/bin/python -m Plotting.truncated_powerlaw_flowchart --regenerate-subplots

Move panels or change their widths in ``PANEL_LAYOUT`` below, then recompose
quickly::

    .venv/bin/python -m Plotting.truncated_powerlaw_flowchart --reuse-subplots

Try the configured cluster download before falling back to local real data::

    .venv/bin/python -m Plotting.truncated_powerlaw_flowchart \
        --regenerate-subplots --download-l250
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox
import numpy as np
from pypdf import PageObject, PdfReader, PdfWriter, Transformation

from Management.jobs import size_scaling_job
from Management.updateCSV import read_macrodata_csv
from MTMath.evaluatePowerlawFit import (
    Fit,
    Truncated_Power_Law,
    evaluate_xmin_distances,
)
from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import calculate_energy_step_data
from Plotting.findXmin import (
    find_xmin_refined_global_min_from_results,
    find_xmin_simple_drop_from_results,
    select_global_min_from_search_details,
)
from Plotting.plotPowerLaw import (
    _resolve_strain_lim,
    dist_from_fit,
    get_energy_drops,
    plot_data_and_fit,
    plot_energy_drop_trace,
    plot_ks_distance,
)


# =============================================================================
# USER-TUNABLE SETTINGS
# =============================================================================

# Workflow mode. CLI flags override this value.
REGENERATE_SUBPLOTS = False

# Preferred data: the ten non-reconnecting L=250 size-scaling simulations.
REQUESTED_SYSTEM_SIZE = 250
REQUESTED_SEEDS = tuple(range(10))
REQUIRE_ALL_REQUESTED_SEEDS = False
DOWNLOAD_L250_IF_MISSING = False

# Explicit CSVs take precedence when this tuple is non-empty.
CSV_PATHS: tuple[Path, ...] = ()

# Places that may contain ``<config>.csv``, ``<config>/macroData.csv``, or an
# ``MTS2D_output/<config>/macroData.csv`` tree.
DATA_SEARCH_ROOTS = (
    Path("/tmp/MTS2D"),
    Path("~/Work/PhD/remoteData/macro").expanduser(),
    Path("/Volumes/data/remoteData/macro"),
    Path("~/Work/PhD/Code/localData").expanduser(),
)

# A real local dataset used only for a visibly marked preview when no requested
# L=250 CSV is reachable.  Set ALLOW_REAL_DATA_FALLBACK=False for strict runs.
ALLOW_REAL_DATA_FALLBACK = True
FALLBACK_CSV_PATHS = (
    Path(
        "~/Work/PhD/Code/localData/MTS2D_output/"
        "simpleShear,s51x51l0.15,1e-05,1.0PBCedgeFlipt3"
        "reconnectRevert1reconnectEdgeLocking0LBFGSEpsx1e-06"
        "logDuringMinimization1energyDropThreshold0.0001s0/"
        "macroData.csv"
    ).expanduser(),
)

# Energy-drop analysis.
# "auto" uses the existing stress-peak split and keeps only the post-yield
# region: gamma > gamma_yield + 0.01, separately for every simulation.
STRAIN_LIMIT = "auto"
AVERAGE_ENERGY = False
DROP_COLUMN = "stress_corrected_drop_second_order"
# Keep every positive drop in the displayed empirical distribution.  xmin is
# selected later and limits the likelihood fit, not the raw data shown.
MIN_DROP = 0.0
XMIN_CANDIDATE_COUNT = 100
# Scan the complete admissible positive-drop range. The upper endpoint is the
# largest cutoff that still retains the requested number of tail events.
MIN_TAIL_COUNT = 25
PARALLEL_XMIN_FITS = False
SIMPLE_DROP_LOCAL_REFINEMENTS = 10
SIMPLE_DROP_LOCAL_MAX_ITERATIONS = 64
GLOBAL_MIN_LOCAL_REFINEMENTS = 10
GLOBAL_MIN_LOCAL_MAX_ITERATIONS = 64

# Raw energy panel and inset.
ZOOM_CENTER = 0.805
ZOOM_WIDTH = 0.010
ENERGY_INSET_BOUNDS = (0.28, 0.69, 0.64, 0.24)
ENERGY_DROP_COLOR = "C1"
ENERGY_DROP_LOG_SCALE = False
ENERGY_DROP_LINESTYLE = "-"
ENERGY_DROP_MARKER = None
ENERGY_INSET_BACKGROUND_ALPHA = 0.9

# Panel-specific colors.
XMIN_SCAN_COLOR = "C3"
DISTRIBUTION_COLOR = "C2"
EMPIRICAL_PDF_MARKER_SIZE = 3.2
KS_DISTANCE_COLOR = "red"
KS_INSET_BOUNDS = (0.57, 0.33, 0.39, 0.34)
KS_INSET_X_FACTOR = 1.15
SHOW_GRID = False

# Landscape A4 layout. Set only (left, bottom, width) for each panel. Heights
# are derived below, so resizing a panel cannot change its physical aspect ratio.
# Entries in PANEL_TOP_EDGES keep their top fixed while their derived height
# changes; their configured bottom value is ignored.
A4_LANDSCAPE_INCHES = (11.69, 8.27)
width = 0.4
h1 = 0.0
h2=0.55
PANEL_LAYOUT = {
    "energy": (-0.02,h1, width),
    "raw_pdf": (0.01, h2, width-0.08),
    "ccdf_ks": (0.34,h2-0.01, width-0.1),
    "xmin_scan": (0.65, h2, width-0.06),
    "mle_fit": (0.63,h1, width),
}
PANEL_TOP_EDGES = {
    "energy": 0.5122,
    "mle_fit": 0.5122,
}
# Higher values are drawn later and therefore appear above overlapping panels.
PANEL_ZORDERS = {
    "energy": 2.2,
    "raw_pdf": 2.5,
    "ccdf_ks": 2.4,
    "xmin_scan": 2.3,
    "mle_fit": 2.2,
}
# Equation nodes are positioned by their centers. Their width and height are
# measured from the rendered text, then expanded by EQUATION_AUTO_BOX_PADDING.
EQUATION_CENTERS = {
    "energy_equation": (0.50, 0.31),
    "ks_equation": (0.50, 0.44),
    "fit_equation": (0.50, 0.21),
}
EQUATION_FONT_SIZE = 8.0
# Automatic mode measures the equations with LaTeX when it is available.
# Padding is in normalized A4 figure coordinates and is added on every side.
EQUATION_USE_LATEX = True
EQUATION_AUTO_BOX_PADDING = (0.015, 0.006)
# Replace None with (width, height) to override one automatically measured box.
EQUATION_MANUAL_BOX_SIZES = {
    "energy_equation": None,
    "ks_equation": None,
    "fit_equation": None,
}
EQUATION_BOX_FACE_COLORS = {
    "energy_equation": "C1",
    "ks_equation": "C3",
    "fit_equation": "C2",
}
EQUATION_BOX_FACE_ALPHA = 0.14
EQUATION_BOX_EDGE_COLOR = "0.65"
EQUATION_BOX_LINEWIDTH = 0.8
EQUATION_BOX_ROUNDING = 0.004
EQUATION_LINE_SPACING = 1.45
# Small directional arrows displayed just above the equation boxes. These are
# independent of SHOW_ARROWS, which controls the flowchart connection arrows.
SHOW_EQUATION_BOX_ARROWS = True
EQUATION_BOX_ARROW_DIRECTIONS = {
    "energy_equation": "left",
    "ks_equation": "up",
    "fit_equation": "right",
}
EQUATION_BOX_ARROW_LENGTH = 0.035
EQUATION_BOX_ARROW_GAP = 0.008
EQUATION_BOX_ARROW_LINEWIDTH = 1.6
EQUATION_BOX_ARROW_MUTATION_SCALE = 10
EQUATION_BOX_ARROW_ALPHA = 0.9
PANEL_LABELS = {
    "energy": r"(a) Energy drops",
    "raw_pdf": r"(b) Raw PDF",
    "ccdf_ks": r"(c) KS at $\Delta E_{\min}$",
    "xmin_scan": r"(d) Select $\Delta E_{\min}$",
    "mle_fit": r"(e) Fitting",
}
PANEL_TITLE_GAP = 0.008
SHOW_SOURCE_NOTE = False

# Exact box anchors and connections. To reroute an arrow, edit only the anchor
# names here. Valid anchors: top, bottom, left, right, topleft, topright,
# bottomleft, bottomright.
ANCHOR_FRACTIONS = {
    "top": (0.5, 1.0),
    "bottom": (0.5, 0.0),
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
    "topleft": (0.2, 1.0),
    "topright": (.9, 1.0),
    "bottomleft": (.1, 0.1),
    "bottomright": (.8, 0.1),
}
PLOT_CONNECTIONS = (
    ("energy", "topright", "raw_pdf", "bottom"),
    ("raw_pdf", "bottomright", "ccdf_ks", "topleft"),
    ("ccdf_ks", "topright", "xmin_scan", "bottomleft"),
    ("xmin_scan", "bottomright", "mle_fit", "topright"),
)
EQUATION_CONNECTIONS = (
    ("energy_equation", "topleft", "energy", "bottomleft"),
    ("ks_equation", "top", "ccdf_ks", "bottom"),
    ("fit_equation", "right", "mle_fit", "bottomright"),
)
PLOT_ARROW_COLOR = "0.18"
EQUATION_ARROW_COLOR = "0.55"
PLOT_ARROW_LINEWIDTH = 1.8
EQUATION_ARROW_LINEWIDTH = 1.6
ARROW_MUTATION_SCALE = 13
SHOW_ARROWS = False
ARROW_ZORDER = 4.0
EQUATION_BOX_ZORDER = 3.0
FLOWCHART_TEXT_ZORDER = 5.0

# Original physical width-to-height ratio of each independent subplot. These
# keep regeneration from collapsing all five panels onto one shared ratio.
PANEL_SOURCE_ASPECT_RATIOS = {
    "energy": 1.22,
    "raw_pdf": 1.3223466,
    "ccdf_ks": 1.1079120,
    "xmin_scan": 1.3891715,
    "mle_fit": 1.31,
}
PANEL_RENDER_SCALE = 1.0
PANEL_DPI = 350
FINAL_DPI = 300
PANEL_FONT_SIZE = 7.5
LEGEND_FONT_SIZE = 6.0
KS_LEGEND_FONT_SIZE = 6.5
FLOWCHART_CROP_PAD_INCHES = 0.02
PANEL_CROP_PAD_INCHES = 0.02
PANEL_SUBPLOT_MARGINS = {
    "left": 0.13,
    "right": 0.90,
    "bottom": 0.18,
    "top": 0.96,
}


# =============================================================================
# DERIVED PATHS -- usually no need to edit below this line
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "Plots" / "powerLaw" / "truncated_powerlaw_flowchart"
PANEL_CACHE_DIR = OUTPUT_DIR / "panels"
DOWNLOAD_CACHE_DIR = OUTPUT_DIR / "data"
ANALYSIS_SUMMARY_PATH = OUTPUT_DIR / "analysis_summary.json"
FINAL_PNG = OUTPUT_DIR / "truncated_powerlaw_exponent_flowchart.png"
FINAL_PDF = OUTPUT_DIR / "truncated_powerlaw_exponent_flowchart.pdf"

PANEL_FILES = {
    name: PANEL_CACHE_DIR / f"{name}.png" for name in PANEL_LAYOUT
}
PANEL_PDF_FILES = {
    name: PANEL_CACHE_DIR / f"{name}.pdf" for name in PANEL_LAYOUT
}


def _configure_matplotlib():
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": PANEL_FONT_SIZE,
            "axes.labelsize": PANEL_FONT_SIZE,
            "xtick.labelsize": PANEL_FONT_SIZE - 1.0,
            "ytick.labelsize": PANEL_FONT_SIZE - 1.0,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.0,
        }
    )


def _l250_configs_and_labels():
    groups, label_groups = size_scaling_job(reconnection="none")
    for configs, labels in zip(groups, label_groups):
        if configs and int(configs[0].rows) == REQUESTED_SYSTEM_SIZE:
            requested = [
                (config, label)
                for config, label in zip(configs, labels)
                if int(config.seed) in REQUESTED_SEEDS
            ]
            return [item[0] for item in requested], [item[1] for item in requested]
    raise RuntimeError(f"No L={REQUESTED_SYSTEM_SIZE} size-scaling group is defined.")


def _candidate_paths(root: Path, config_name: str):
    return (
        root / f"{config_name}_fixed.csv",
        root / f"{config_name}.csv",
        root / config_name / "macroData.csv",
        root / "MTS2D_output" / config_name / "macroData.csv",
    )


def _find_requested_csvs(configs):
    found = []
    for config in configs:
        for root in DATA_SEARCH_ROOTS + (DOWNLOAD_CACHE_DIR,):
            match = next(
                (path for path in _candidate_paths(root, config.name) if path.is_file()),
                None,
            )
            if match is not None:
                found.append(match)
                break
    return found


def _download_requested_csvs(configs, labels, *, force_update=False):
    """Use the repository downloader, caching refreshed files for this figure."""

    from Plotting import remotePlotting

    DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    original_macro_path = remotePlotting.MACRO_PATH
    try:
        remotePlotting.MACRO_PATH = str(DOWNLOAD_CACHE_DIR)
        paths, _ = remotePlotting.get_csv_files(
            configs,
            labels=labels,
            useOldFiles=not force_update,
            forceUpdate=force_update,
            debug_download=True,
            fix_files=True,
        )
    finally:
        remotePlotting.MACRO_PATH = original_macro_path
    return [Path(path) for path in paths if Path(path).is_file()]


def resolve_csv_paths(*, download_if_missing=False, allow_fallback=True):
    if CSV_PATHS:
        explicit = [Path(path).expanduser() for path in CSV_PATHS]
        missing = [path for path in explicit if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Explicit CSV paths do not exist: {missing}")
        return explicit, False

    configs, labels = _l250_configs_and_labels()
    paths = _find_requested_csvs(configs)
    if download_if_missing:
        _download_requested_csvs(
            configs,
            labels,
            force_update=True,
        )
        paths = _find_requested_csvs(configs)

    if paths and (not REQUIRE_ALL_REQUESTED_SEEDS or len(paths) == len(configs)):
        return paths, False
    if paths and REQUIRE_ALL_REQUESTED_SEEDS:
        raise RuntimeError(
            f"Found {len(paths)} of {len(configs)} requested L={REQUESTED_SYSTEM_SIZE} "
            "runs, but REQUIRE_ALL_REQUESTED_SEEDS=True."
        )

    if allow_fallback and ALLOW_REAL_DATA_FALLBACK:
        fallback = [path for path in FALLBACK_CSV_PATHS if path.is_file()]
        if fallback:
            warnings.warn(
                f"No L={REQUESTED_SYSTEM_SIZE} CSV was reachable; using the configured "
                "real-data fallback for a clearly marked preview.",
                stacklevel=2,
            )
            return fallback, True

    raise FileNotFoundError(
        f"No non-reconnecting L={REQUESTED_SYSTEM_SIZE} CSV was found. Run with "
        "--download-l250 while the cluster is reachable, populate CSV_PATHS, or "
        "enable the real-data fallback."
    )


def _read_steps(path: Path):
    df = read_macrodata_csv(
        str(path),
        fix_mixed=True,
        update_header=True,
        warn_on_dtype=True,
    )
    steps, info = calculate_energy_step_data(
        path,
        df=df,
        metadata=get_metadata(str(path)),
        average_energy=AVERAGE_ENERGY,
    )
    return df, steps, info


def _strain_mask(values, strain_limit):
    values = np.asarray(values, dtype=float)
    if strain_limit is None:
        return np.isfinite(values)
    low, high = strain_limit
    return np.isfinite(values) & (values > low) & (values < high)


def _collect_analysis(csv_paths: list[Path], used_fallback: bool):
    # Keep the first run as a trace for the energy panel, but extract the
    # combined sample through the same reusable path as the standard power-law
    # analysis.  This includes every positive drop below the eventual xmin.
    raw_df, raw_steps, raw_info = _read_steps(csv_paths[0])
    first_strain_limit = tuple(
        float(value)
        for value in _resolve_strain_lim(
            STRAIN_LIMIT,
            df=raw_df,
            postRegime=True,
        )
    )
    configured_strain_limit = (
        list(STRAIN_LIMIT)
        if isinstance(STRAIN_LIMIT, (list, tuple))
        else STRAIN_LIMIT
    )
    drops, data_info = get_energy_drops(
        [str(path) for path in csv_paths],
        strainLim=configured_strain_limit,
        debug=False,
        postRegime=True,
        averageEnergy=AVERAGE_ENERGY,
        stress_corrected=True,
        stress_correction_order=2,
        stress_tangent="current",
        min_drop=MIN_DROP,
    )
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > MIN_DROP)]
    if drops.size < max(3, MIN_TAIL_COUNT):
        raise RuntimeError(
            f"Only {drops.size} drops exceed MIN_DROP={MIN_DROP:g}; lower MIN_DROP "
            "or provide more simulation CSVs."
        )

    sorted_drops = np.sort(drops)
    candidate_lo = float(sorted_drops[0])
    candidate_hi = float(sorted_drops[-MIN_TAIL_COUNT])
    if not (candidate_hi > candidate_lo > 0.0):
        raise RuntimeError("Could not form a positive xmin candidate interval.")

    xmins = np.geomspace(candidate_lo, candidate_hi, XMIN_CANDIDATE_COUNT)
    distances, param_vals, _ = evaluate_xmin_distances(
        drops,
        xmins,
        distType=Truncated_Power_Law,
        parallel=PARALLEL_XMIN_FITS,
    )
    distances = np.asarray(distances, dtype=float)
    tail_counts = np.asarray([np.count_nonzero(drops >= xmin) for xmin in xmins])
    selectable = np.isfinite(distances) & (tail_counts >= MIN_TAIL_COUNT)
    if not np.any(selectable):
        raise RuntimeError("No finite xmin candidate retains the requested tail count.")
    selected_xmin, simple_drop_details = find_xmin_simple_drop_from_results(
        drops,
        xmins,
        distances,
        min_tail_count=MIN_TAIL_COUNT,
        local_refinements=SIMPLE_DROP_LOCAL_REFINEMENTS,
        local_max_iterations=SIMPLE_DROP_LOCAL_MAX_ITERATIONS,
        distType=Truncated_Power_Law,
        parallel=PARALLEL_XMIN_FITS,
    )
    selected_distance = float(simple_drop_details["selected_distance"])
    _, global_search_details = find_xmin_refined_global_min_from_results(
        drops,
        xmins,
        distances,
        min_tail_count=MIN_TAIL_COUNT,
        local_refinements=GLOBAL_MIN_LOCAL_REFINEMENTS,
        local_max_iterations=GLOBAL_MIN_LOCAL_MAX_ITERATIONS,
        distType=Truncated_Power_Law,
        parallel=PARALLEL_XMIN_FITS,
    )

    # Select the global minimum only after both independent refinement
    # strategies have completed. This ensures that a lower simpleDrop
    # evaluation cannot sit below a prematurely chosen "global" minimum.
    (
        global_min_xmin,
        global_min_distance,
        all_evaluations,
    ) = select_global_min_from_search_details(
        simple_drop_details,
        global_search_details,
    )

    fit = Fit(
        drops,
        xmin=selected_xmin,
        xmin_distribution=Truncated_Power_Law.name,
        verbose=0,
    )
    dist = dist_from_fit(fit)

    source_meta = get_metadata(str(csv_paths[0]))
    actual_size = source_meta.get("L")
    if actual_size is None:
        dims = source_meta.get("dims") or source_meta.get("N")
        actual_size = dims[0] if dims else None
    reconnecting = any("edgeFlip" in str(path) for path in csv_paths)

    return {
        "csv_paths": csv_paths,
        "used_fallback": used_fallback,
        "actual_size": int(actual_size) if actual_size is not None else None,
        "reconnecting": reconnecting,
        "raw_df": raw_df,
        "raw_steps": raw_steps,
        "raw_info": raw_info,
        "first_strain_limit": first_strain_limit,
        "data_info": data_info,
        "drops": drops,
        "xmins": xmins,
        "distances": distances,
        "param_vals": param_vals,
        "tail_counts": tail_counts,
        "global_min_xmin": float(global_min_xmin),
        "global_min_distance": float(global_min_distance),
        "global_search_details": global_search_details,
        "all_xmin_evaluations": all_evaluations,
        "simple_drop_details": simple_drop_details,
        "simple_drop_distance": selected_distance,
        "selected_xmin": selected_xmin,
        "fit": fit,
        "alpha": float(dist.alpha),
        "lambda": float(dist.Lambda),
        "ks_distance": float(dist.D),
    }


def _new_panel(name):
    _, _, width = PANEL_LAYOUT[name]
    physical_width = width * A4_LANDSCAPE_INCHES[0] * PANEL_RENDER_SCALE
    figsize = (
        physical_width,
        physical_width / PANEL_SOURCE_ASPECT_RATIOS[name],
    )
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(**PANEL_SUBPLOT_MARGINS)
    ax.grid(SHOW_GRID, which="both")
    ax.set_title("")
    return fig, ax


def _save_panel(fig, name):
    fig.savefig(
        PANEL_FILES[name],
        dpi=PANEL_DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=PANEL_CROP_PAD_INCHES,
    )
    fig.savefig(
        PANEL_PDF_FILES[name],
        facecolor="white",
        bbox_inches="tight",
        pad_inches=PANEL_CROP_PAD_INCHES,
    )
    plt.close(fig)


def _energy_panel(analysis):
    df = analysis["raw_df"]
    steps = analysis["raw_steps"]
    info = analysis["raw_info"]
    energy_col = info["energy_col"]

    strain = np.asarray(df["load"], dtype=float)
    energy = np.asarray(df[energy_col], dtype=float)
    drop_strain = np.asarray(steps["load_ip1"], dtype=float)
    drop_values = np.asarray(steps[DROP_COLUMN], dtype=float)
    first_strain_limit = analysis["first_strain_limit"]
    curve_mask = _strain_mask(strain, first_strain_limit)
    drop_mask = _strain_mask(drop_strain, first_strain_limit)

    fig, ax = _new_panel("energy")
    drop_ax, inset_ax, inset_drop_ax = plot_energy_drop_trace(
        ax,
        strain[curve_mask],
        energy[curve_mask],
        drop_strain[drop_mask],
        drop_values[drop_mask],
        energy_label=r"$E$",
        drop_label=r"$\Delta E_S$",
        color_drop=ENERGY_DROP_COLOR,
        min_drop=MIN_DROP,
        log_drop_axis=ENERGY_DROP_LOG_SCALE,
        drop_marker=ENERGY_DROP_MARKER,
        drop_linestyle=ENERGY_DROP_LINESTYLE,
        zoom_center=ZOOM_CENTER,
        zoom_width=ZOOM_WIDTH,
        inset_bounds=ENERGY_INSET_BOUNDS,
        inset_background_alpha=ENERGY_INSET_BACKGROUND_ALPHA,
        inset_show_y_ticks=False,
        set_title=False,
        show_legend=True,
    )
    drop_ax.grid(False, which="both")
    inset_ax.grid(False, which="both")
    inset_drop_ax.grid(False, which="both")
    _save_panel(fig, "energy")


def _raw_pdf_panel(analysis):
    fig, ax = _new_panel("raw_pdf")
    plot_data_and_fit(
        analysis["fit"],
        ax=ax,
        data_info=analysis["data_info"],
        data_color=DISTRIBUTION_COLOR,
        addFit=False,
        save=False,
        show=False,
        close=False,
        show_fit_region=False,
        show_cutoff=False,
        show_title=False,
        show_legend=False,
    )
    ax.lines[0].set_label("Empirical PDF")
    ax.lines[0].set_markersize(EMPIRICAL_PDF_MARKER_SIZE)
    ax.set_xlabel(r"$\Delta E_S$")
    ax.set_ylabel(r"$p(\Delta E_S)$")
    ax.legend(loc="best")
    _save_panel(fig, "raw_pdf")


def _ccdf_ks_panel(analysis):
    fig, ax = _new_panel("ccdf_ks")
    plot_ks_distance(
        analysis["drops"],
        analysis["selected_xmin"],
        ax=ax,
        save=False,
        close=False,
        set_title=False,
        show_legend=True,
        empirical_color=DISTRIBUTION_COLOR,
        ks_color=KS_DISTANCE_COLOR,
        show_inset=True,
        inset_bounds=KS_INSET_BOUNDS,
        inset_x_factor=KS_INSET_X_FACTOR,
        inset_grid=SHOW_GRID,
        legend_usetex=False,
        tight_layout=False,
    )
    ax.set_xlabel(r"$\Delta E_S$")
    ax.set_ylabel(r"$P(\Delta E_S>x)$")
    ax.legend(loc="best", fontsize=KS_LEGEND_FONT_SIZE)
    _save_panel(fig, "ccdf_ks")


def _xmin_scan_panel(analysis):
    fig, ax = _new_panel("xmin_scan")
    xmins = analysis["xmins"]
    distances = analysis["distances"]
    global_xmin = analysis["global_min_xmin"]
    global_distance = analysis["global_min_distance"]
    selected_xmin = analysis["selected_xmin"]
    selected_distance = analysis["simple_drop_distance"]
    global_differs_from_simple_drop = not np.isclose(
        global_xmin,
        selected_xmin,
        rtol=1e-6,
        atol=0.0,
    )
    ax.plot(
        xmins,
        distances,
        marker="o",
        markersize=2.8,
        color=XMIN_SCAN_COLOR,
        label=r"$D(\Delta E_{\min})$",
    )
    rough_local_minima = analysis["global_search_details"]["rough_local_minima"]
    if global_differs_from_simple_drop and rough_local_minima:
        ax.scatter(
            [item["xmin"] for item in rough_local_minima],
            [item["distance"] for item in rough_local_minima],
            marker="s",
            s=12,
            facecolor="none",
            edgecolor="0.35",
            linewidth=0.6,
            zorder=4,
            label="Coarse local minima",
        )
    ax.scatter(
        [global_xmin],
        [global_distance],
        marker="X",
        s=25,
        facecolor="white",
        edgecolor="0.25",
        linewidth=0.8,
        zorder=5,
        label=rf"Global min.: $\Delta E_{{\min}}={global_xmin:.1e}$",
    )
    if global_differs_from_simple_drop:
        ax.scatter(
            [selected_xmin],
            [selected_distance],
            marker="D",
            s=22,
            color=XMIN_SCAN_COLOR,
            edgecolor="white",
            linewidth=0.5,
            zorder=6,
            label=rf"simpleDrop: $\Delta E_{{\min}}={selected_xmin:.1e}$",
        )
        ax.axvline(
            global_xmin,
            color="0.45",
            linestyle="--",
            linewidth=0.8,
        )
    ax.axvline(
        selected_xmin,
        color=XMIN_SCAN_COLOR,
        linestyle="--",
        linewidth=1.0,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta E_{\min}$")
    ax.set_ylabel(r"$D$")
    ax.legend(loc="best")
    _save_panel(fig, "xmin_scan")


def _mle_fit_panel(analysis):
    fig, ax = _new_panel("mle_fit")
    fit = analysis["fit"]
    alpha = analysis["alpha"]
    lambda_value = analysis["lambda"]

    plot_data_and_fit(
        fit,
        ax=ax,
        data_info=analysis["data_info"],
        color=DISTRIBUTION_COLOR,
        data_color=DISTRIBUTION_COLOR,
        addFit=True,
        save=False,
        show=False,
        close=False,
        show_fit_region=True,
        show_cutoff=True,
        show_title=False,
        show_legend=False,
    )
    ax.lines[0].set_label("Empirical PDF")
    ax.lines[0].set_markersize(EMPIRICAL_PDF_MARKER_SIZE)
    ax.lines[1].set_label(
        rf"MLE: $\hat{{\alpha}}={alpha:.2f}$, "
        rf"$\hat{{\lambda}}={lambda_value:.1e}$"
    )
    if ax.patches:
        ax.patches[-1].set_label(r"$\mathcal{X}_{\min}$ (fit region)")
    ax.axvline(
        analysis["selected_xmin"],
        color="0.35",
        linestyle="--",
        linewidth=0.8,
        label=r"$\Delta E_{\min}$",
    )
    ax.set_xlabel(r"$\Delta E_S$")
    ax.set_ylabel(r"$p(\Delta E_S)$")
    ax.legend(loc="best")
    _save_panel(fig, "mle_fit")


def _write_analysis_summary(analysis):
    summary = {
        "csv_paths": [str(path) for path in analysis["csv_paths"]],
        "used_fallback": bool(analysis["used_fallback"]),
        "actual_size": analysis["actual_size"],
        "reconnecting": bool(analysis["reconnecting"]),
        "strain_limit": STRAIN_LIMIT,
        "first_run_post_yield_limit": list(analysis["first_strain_limit"]),
        "minimum_drop": MIN_DROP,
        "number_of_drops": int(analysis["drops"].size),
        "xmin_method": "simpleDrop",
        "global_min_xmin": analysis["global_min_xmin"],
        "global_min_distance": analysis["global_min_distance"],
        "global_min_differs_from_simple_drop": not np.isclose(
            analysis["global_min_xmin"],
            analysis["selected_xmin"],
            rtol=1e-6,
            atol=0.0,
        ),
        "global_search_rough_local_minima": analysis["global_search_details"][
            "rough_local_minima"
        ],
        "global_search_refined_local_minima": analysis["global_search_details"][
            "local_minima"
        ],
        "global_search_evaluation_count": len(
            analysis["global_search_details"]["evaluated_xmins"]
        ),
        "selected_xmin": analysis["selected_xmin"],
        "simple_drop_distance": analysis["simple_drop_distance"],
        "simple_drop_largest_interval": list(
            analysis["simple_drop_details"]["largest_drop_interval"]
        ),
        "simple_drop_local_minima": analysis["simple_drop_details"]["local_minima"],
        "alpha": analysis["alpha"],
        "lambda": analysis["lambda"],
        "inverse_lambda": (
            1.0 / analysis["lambda"] if analysis["lambda"] > 0.0 else None
        ),
        "ks_distance": analysis["ks_distance"],
        "xmin_candidates": analysis["xmins"].tolist(),
        "xmin_distances": analysis["distances"].tolist(),
        "tail_counts": analysis["tail_counts"].tolist(),
    }
    ANALYSIS_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def regenerate_panels(*, download_if_missing=False, allow_fallback=True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths, used_fallback = resolve_csv_paths(
        download_if_missing=download_if_missing,
        allow_fallback=allow_fallback,
    )
    analysis = _collect_analysis(csv_paths, used_fallback)
    _energy_panel(analysis)
    _raw_pdf_panel(analysis)
    _ccdf_ks_panel(analysis)
    _xmin_scan_panel(analysis)
    _mle_fit_panel(analysis)
    return _write_analysis_summary(analysis)


def _load_summary():
    if not ANALYSIS_SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {ANALYSIS_SUMMARY_PATH}; run --regenerate-subplots first."
        )
    missing_panels = [
        path
        for path in (*PANEL_FILES.values(), *PANEL_PDF_FILES.values())
        if not path.is_file()
    ]
    if missing_panels:
        raise FileNotFoundError(
            f"Missing cached plot panels {missing_panels}; run --regenerate-subplots."
        )
    return json.loads(ANALYSIS_SUMMARY_PATH.read_text())


def _calculate_panel_positions():
    """Calculate panel heights from each cached subplot's native aspect ratio."""

    page_width, page_height = A4_LANDSCAPE_INCHES
    positions = {}
    for name, (left, bottom, width) in PANEL_LAYOUT.items():
        image = mpimg.imread(PANEL_FILES[name])
        pixel_height, pixel_width = image.shape[:2]
        native_aspect_ratio = pixel_width / pixel_height
        height = width * page_width / (native_aspect_ratio * page_height)
        if name in PANEL_TOP_EDGES:
            bottom = PANEL_TOP_EDGES[name] - height
        positions[name] = (left, bottom, width, height)
    return positions


def _anchor(node_positions, node_name, anchor_name):
    """Return an exact named anchor on a configured node bounding box."""

    if node_name not in node_positions:
        raise KeyError(f"Unknown node {node_name!r}.")
    if anchor_name not in ANCHOR_FRACTIONS:
        valid = ", ".join(ANCHOR_FRACTIONS)
        raise KeyError(f"Unknown anchor {anchor_name!r}; choose from {valid}.")
    left, bottom, width, height = node_positions[node_name]
    x_fraction, y_fraction = ANCHOR_FRACTIONS[anchor_name]
    return (
        left + x_fraction * width,
        bottom + y_fraction * height,
    )


def _add_arrow(fig, start, end, *, color, linewidth):
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=ARROW_MUTATION_SCALE,
        linewidth=linewidth,
        color=color,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=ARROW_ZORDER,
    )
    fig.add_artist(arrow)


def _add_equation_box_arrows(fig, equation_positions):
    """Draw short, independently configurable arrows above equation boxes."""

    if not SHOW_EQUATION_BOX_ARROWS:
        return

    valid_directions = {"left", "right", "up", "down"}
    for name, direction in EQUATION_BOX_ARROW_DIRECTIONS.items():
        if name not in equation_positions:
            raise KeyError(f"Unknown equation box {name!r}.")
        if direction not in valid_directions:
            valid = ", ".join(sorted(valid_directions))
            raise ValueError(
                f"Unknown equation-box arrow direction {direction!r}; "
                f"choose from {valid}."
            )

        left, bottom, width, height = equation_positions[name]
        center_x = left + 0.5 * width
        arrow_y = bottom + height + EQUATION_BOX_ARROW_GAP
        half_length = 0.5 * EQUATION_BOX_ARROW_LENGTH
        if direction == "left":
            start = (center_x + half_length, arrow_y)
            end = (center_x - half_length, arrow_y)
        elif direction == "right":
            start = (center_x - half_length, arrow_y)
            end = (center_x + half_length, arrow_y)
        elif direction == "up":
            start = (center_x, arrow_y)
            end = (center_x, arrow_y + EQUATION_BOX_ARROW_LENGTH)
        else:
            start = (center_x, arrow_y + EQUATION_BOX_ARROW_LENGTH)
            end = (center_x, arrow_y)

        arrow = FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=EQUATION_BOX_ARROW_MUTATION_SCALE,
            linewidth=EQUATION_BOX_ARROW_LINEWIDTH,
            color=EQUATION_BOX_FACE_COLORS[name],
            alpha=EQUATION_BOX_ARROW_ALPHA,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=ARROW_ZORDER,
        )
        fig.add_artist(arrow)


def _source_note(summary):
    size = summary.get("actual_size")
    runs = len(summary.get("csv_paths", []))
    n = summary.get("number_of_drops")
    if summary.get("used_fallback"):
        reconnect = ", reconnecting" if summary.get("reconnecting") else ""
        return rf"Post-yield preview: $L={size}$ fallback{reconnect}; {runs} run; $n={n}$"
    return rf"Post-yield size scaling: $L={size}$; {runs} runs; $n={n}$"


def _write_vector_pdf(fig, panel_axes, panel_positions, crop_bbox_inches):
    """Compose cached vector panel PDFs over the vector flowchart layout."""

    layout_pdf = OUTPUT_DIR / "_flowchart_layout.pdf"
    try:
        for ax in panel_axes:
            ax.set_visible(False)
        # Save a transparent overlay containing titles, equations, and arrows.
        # It is merged after the panel PDFs so arrows stay above every plot.
        fig.savefig(layout_pdf, transparent=True)

        layout_reader = PdfReader(str(layout_pdf))
        overlay_page = layout_reader.pages[0]
        page_width = float(overlay_page.mediabox.width)
        page_height = float(overlay_page.mediabox.height)
        figure_width, figure_height = fig.get_size_inches()
        crop_left = crop_bbox_inches.x0 / figure_width * page_width
        crop_bottom = crop_bbox_inches.y0 / figure_height * page_height
        crop_width = crop_bbox_inches.width / figure_width * page_width
        crop_height = crop_bbox_inches.height / figure_height * page_height
        output_page = PageObject.create_blank_page(
            width=crop_width,
            height=crop_height,
        )

        panel_items = sorted(
            panel_positions.items(),
            key=lambda item: PANEL_ZORDERS[item[0]],
        )
        for name, (left, bottom, width, _height) in panel_items:
            panel_page = PdfReader(str(PANEL_PDF_FILES[name])).pages[0]
            panel_width = float(panel_page.mediabox.width)
            scale = (width * page_width) / panel_width
            transform = (
                Transformation()
                .scale(scale)
                .translate(
                    left * page_width - crop_left,
                    bottom * page_height - crop_bottom,
                )
            )
            output_page.merge_transformed_page(panel_page, transform)

        output_page.merge_transformed_page(
            overlay_page,
            Transformation().translate(-crop_left, -crop_bottom),
        )
        writer = PdfWriter()
        writer.add_page(output_page)
        with FINAL_PDF.open("wb") as stream:
            writer.write(stream)
    finally:
        for ax in panel_axes:
            ax.set_visible(True)
        layout_pdf.unlink(missing_ok=True)


def _add_equations(fig):
    """Draw equations and return their visible connection-box bounds."""

    equations = {
        "energy_equation": (
            r"$\Delta E_S=\widehat E_{n+1}-E_{n+1}$"
            "\n"
            r"$\widehat E_{n+1}=E_n"
            r"+V_0\langle\sigma_{12}\rangle_n\Delta\gamma_n"
            r"+\frac{V_0}{2}\mathfrak{a}_{1212,n}(\Delta\gamma_n)^2$"
        ),
        "ks_equation": (
            r"$D(\Delta E_{\min})="
            r"\sup_{x\geq\Delta E_{\min}}"
            r"\left|\widehat P_{>}(x)-P_{>}^{\rm TPL}(x)\right|$"
            "\n"
            r"$\widehat P_{>}(x)=\frac{1}{|\mathcal{X}_{\min}|}"
            r"\sum_{\Delta E_S\in\mathcal{X}_{\min}}\mathbf{1}(\Delta E_S>x)$"
            "\n"
            r"$P_{>}^{\rm TPL}(x)=\int_x^\infty\!"
            r"p(u\mid\hat\alpha,\hat\lambda,\Delta E_{\min})\,du$"
            "\n"
            r"$\mathcal{X}_{\min}\equiv"
            r"\{\Delta E_S:\Delta E_S\geq\Delta E_{\min}\}$"
        ),
        "fit_equation": (
            r"$p(\Delta E_S\mid\alpha,\lambda,\Delta E_{\min})"
            r"=\frac{\Delta E_S^{-\alpha}e^{-\lambda\Delta E_S}}"
            r"{Z(\alpha,\lambda,\Delta E_{\min})}$"
            "\n"
            r"$\ell(\alpha,\lambda)="
            r"\sum_{\Delta E_S\in\mathcal{X}_{\min}}"
            r"\ln p(\Delta E_S\mid\alpha,\lambda,\Delta E_{\min})$"
            "\n"
            r"$(\hat\alpha,\hat\lambda)="
            r"\arg\max_{\alpha,\lambda}\ell(\alpha,\lambda)$"
        ),
    }

    artists = {}
    for name, equation in equations.items():
        artists[name] = fig.text(
            *EQUATION_CENTERS[name],
            equation,
            ha="center",
            va="center",
            fontsize=EQUATION_FONT_SIZE,
            linespacing=EQUATION_LINE_SPACING,
            usetex=EQUATION_USE_LATEX,
            zorder=FLOWCHART_TEXT_ZORDER,
        )

    try:
        fig.canvas.draw()
    except RuntimeError:
        if not EQUATION_USE_LATEX:
            raise
        warnings.warn(
            "LaTeX equation measurement failed; falling back to MathText. "
            "Use EQUATION_MANUAL_BOX_SIZES if a box needs adjustment.",
            stacklevel=2,
        )
        for artist in artists.values():
            artist.set_usetex(False)
        fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    inverse_figure_transform = fig.transFigure.inverted()
    pad_x, pad_y = EQUATION_AUTO_BOX_PADDING
    positions = {}
    for name, artist in artists.items():
        bounds = artist.get_window_extent(renderer).transformed(
            inverse_figure_transform
        )
        manual_size = EQUATION_MANUAL_BOX_SIZES[name]
        if manual_size is None:
            width = bounds.width + 2.0 * pad_x
            height = bounds.height + 2.0 * pad_y
        else:
            width, height = manual_size
        center_x, center_y = EQUATION_CENTERS[name]
        position = (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            width,
            height,
        )
        positions[name] = position
        left, bottom, width, height = position
        box = FancyBboxPatch(
            (left, bottom),
            width,
            height,
            boxstyle=(
                "round,pad=0.0,"
                f"rounding_size={EQUATION_BOX_ROUNDING}"
            ),
            transform=fig.transFigure,
            facecolor=mpl.colors.to_rgba(
                EQUATION_BOX_FACE_COLORS[name],
                EQUATION_BOX_FACE_ALPHA,
            ),
            edgecolor=EQUATION_BOX_EDGE_COLOR,
            linewidth=EQUATION_BOX_LINEWIDTH,
            zorder=EQUATION_BOX_ZORDER,
        )
        fig.add_artist(box)
    return positions


def _flowchart_crop_bbox(fig):
    """Measure a padded bounding box around all visible flowchart artists."""

    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    return Bbox.from_extents(
        tight_bbox.x0 - FLOWCHART_CROP_PAD_INCHES,
        tight_bbox.y0 - FLOWCHART_CROP_PAD_INCHES,
        tight_bbox.x1 + FLOWCHART_CROP_PAD_INCHES,
        tight_bbox.y1 + FLOWCHART_CROP_PAD_INCHES,
    )


def compose_flowchart(summary):
    FINAL_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=A4_LANDSCAPE_INCHES, facecolor="white")
    panel_positions = _calculate_panel_positions()
    equation_positions = _add_equations(fig)
    node_positions = {**panel_positions, **equation_positions}

    _add_equation_box_arrows(fig, equation_positions)

    if SHOW_ARROWS:
        for source, source_anchor, target, target_anchor in PLOT_CONNECTIONS:
            _add_arrow(
                fig,
                _anchor(node_positions, source, source_anchor),
                _anchor(node_positions, target, target_anchor),
                color=PLOT_ARROW_COLOR,
                linewidth=PLOT_ARROW_LINEWIDTH,
            )
        for source, source_anchor, target, target_anchor in EQUATION_CONNECTIONS:
            _add_arrow(
                fig,
                _anchor(node_positions, source, source_anchor),
                _anchor(node_positions, target, target_anchor),
                color=EQUATION_ARROW_COLOR,
                linewidth=EQUATION_ARROW_LINEWIDTH,
            )

    panel_axes = []
    for name, position in panel_positions.items():
        ax = fig.add_axes(position, zorder=PANEL_ZORDERS[name])
        ax.imshow(mpimg.imread(PANEL_FILES[name]), aspect="equal")
        ax.set_axis_off()
        panel_axes.append(ax)
        x, y, w, h = position
        fig.text(
            x + 0.5 * w,
            y + h + PANEL_TITLE_GAP,
            PANEL_LABELS[name],
            ha="center",
            va="bottom",
            fontsize=11.0,
            fontweight="semibold",
            zorder=FLOWCHART_TEXT_ZORDER,
        )

    if SHOW_SOURCE_NOTE:
        fig.text(
            0.5,
            0.025,
            _source_note(summary),
            ha="center",
            va="center",
            fontsize=7.0,
            color="0.35",
        )

    crop_bbox_inches = _flowchart_crop_bbox(fig)
    fig.savefig(
        FINAL_PNG,
        dpi=FINAL_DPI,
        facecolor="white",
        bbox_inches=crop_bbox_inches,
        pad_inches=0.0,
    )
    _write_vector_pdf(fig, panel_axes, panel_positions, crop_bbox_inches)
    return fig


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--regenerate-subplots",
        action="store_true",
        help="Recompute the data analysis and overwrite cached plot panels.",
    )
    mode.add_argument(
        "--reuse-subplots",
        action="store_true",
        help="Reuse cached panels and only recompose the flowchart.",
    )
    parser.add_argument(
        "--download-l250",
        action="store_true",
        help="Force-refresh the configured L=250 CSV files from the servers.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail instead of creating a marked preview from fallback real data.",
    )
    parser.add_argument("--show", action="store_true", help="Show the final figure.")
    return parser.parse_args()


def generate_flowchart(
    *,
    regenerate_subplots=None,
    download_l250=None,
    allow_fallback=True,
    show=False,
):
    """Generate the flowchart using the same settings as the command-line script.

    ``regenerate_subplots=None`` follows ``REGENERATE_SUBPLOTS``. Set it to
    ``False`` for the fast layout-only mode that reuses the cached panels.
    """

    _configure_matplotlib()
    regenerate = (
        REGENERATE_SUBPLOTS
        if regenerate_subplots is None
        else bool(regenerate_subplots)
    )
    download = (
        DOWNLOAD_L250_IF_MISSING if download_l250 is None else bool(download_l250)
    )

    if regenerate:
        summary = regenerate_panels(
            download_if_missing=download,
            allow_fallback=allow_fallback,
        )
    else:
        summary = _load_summary()

    fig = compose_flowchart(summary)
    print(f"Saved {FINAL_PNG}")
    print(f"Saved {FINAL_PDF}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return FINAL_PNG, FINAL_PDF


def main():
    args = _parse_args()
    if args.regenerate_subplots:
        regenerate = True
    elif args.reuse_subplots:
        regenerate = False
    else:
        regenerate = None

    generate_flowchart(
        regenerate_subplots=regenerate,
        download_l250=(True if args.download_l250 else None),
        allow_fallback=not args.no_fallback,
        show=args.show,
    )


if __name__ == "__main__":
    main()
