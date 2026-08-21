"""Build a reversibility-aware truncated-power-law flowchart.

This is a separate version of the ordinary flowchart.  It uses the
post-yield relaxation-energy drops to form
``kappa = Delta E_R/(rho V_0 Delta gamma^2)`` and makes the reversible /
irreversible split at ``kappa_det = mu/2``.  It then fits only the
irreversible stress-corrected ``Delta E_S`` drops.

For the default small-data case, the final ``Delta E_S,min^KS`` is selected by
evaluating every observed candidate in the irreversible ``Delta E_S``
population and choosing the true global KS minimum before the maximum-
likelihood fit.  The coarse/local search is an explicit approximation for
larger populations.

Slow mode::

    .venv/bin/python -m Plotting.truncated_powerlaw_reversibility_flowchart \
        --regenerate-subplots

Fast layout mode::

    .venv/bin/python -m Plotting.truncated_powerlaw_reversibility_flowchart \
        --reuse-subplots
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import warnings
from pathlib import Path

# Allow ``python Plotting/truncated_powerlaw_reversibility_flowchart.py`` as
# well as ``python -m Plotting.truncated_powerlaw_reversibility_flowchart``.
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

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
    POWERLAW_STANDARD_WORKFLOW,
    Truncated_Power_Law,
    evaluate_xmin_distances,
)
from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import (
    extract_energy_drops_from_dataframe,
)
from Plotting.findXmin import plot_xmin_analysis
from Plotting.plotPowerLaw import (
    _resolve_strain_lim,
    dist_from_fit,
    get_energy_drops,
    make_fit,
    plot_data_pdf,
    plot_data_and_fit,
    plot_energy_drop_trace,
    plot_ks_distance,
)
from Plotting.standardPowerlaw import (
    kappa_detection_threshold,
    kappa_from_relaxation_energy,
)


# =============================================================================
# USER-TUNABLE SETTINGS
# =============================================================================

# Running the script without arguments refreshes the data panels.  Use
# ``--reuse-subplots`` for fast layout-only iterations.
REGENERATE_SUBPLOTS = False
REQUESTED_SYSTEM_SIZE = 250
REQUESTED_SEEDS = tuple(range(10))
REQUIRE_ALL_REQUESTED_SEEDS = False

CSV_PATHS: tuple[Path, ...] = ()
DATA_SEARCH_ROOTS = (
    Path("/tmp/MTS2D"),
    Path("~/Work/PhD/remoteData/macro").expanduser(),
    Path("/Volumes/data/remoteData/macro"),
    Path("~/Work/PhD/Code/localData").expanduser(),
)
DOWNLOAD_L250_IF_MISSING = False

STRAIN_LIMIT = "auto"
AVERAGE_ENERGY = False
MIN_DROP = 0.0
XMIN_CANDIDATE_COUNT = 100
XMIN_MIN_TAIL_COUNT = 25
TRUE_GLOBAL_EVENT_THRESHOLD = 10_000
USE_TRUE_GLOBAL_FOR_SMALL_DATA = True
TRUE_GLOBAL_PARALLEL = False
# These affect only how densely markers are drawn in the vector panels.  All
# data points remain in the analysis and in the connecting line paths.
XMIN_MAX_MARKERS = 250
ALPHA_MAX_MARKERS = 250
FIT_PDF_MAX_MARKERS = 400

# Leave this as None to use the material threshold ``kappa_det = mu/2``.
# Set a number explicitly only for a frozen data release or layout comparison.
KAPPA_DET_OVERRIDE = None

# The event split is based on kappa = Delta E_R/(rho V_0 Delta gamma^2).
KAPPA_DETECTION_METHOD_LABEL = "kappa_det = mu/2 with rho=1"
REVERSIBLE_COLOR = "#b9dff2"       # light blue
IRREVERSIBLE_COLOR = "#f6c28b"      # light orange
ER_ALL_COLOR = "0.55"
ES_COLOR = "C2"
KS_COLOR = "red"
ALPHA_COLOR = "C4"
DROP_DETECTION_LINE_COLOR = "0.25"

# Raw energy trace (kept consistent with the original flowchart).
ZOOM_CENTER = 0.805
ZOOM_WIDTH = 0.010
ENERGY_INSET_BOUNDS = (0.32, 0.69, 0.6, 0.24)
ENERGY_DROP_COLOR = "C1"
ENERGY_DROP_LOG_SCALE = False
ENERGY_DROP_LINESTYLE = "-"
ENERGY_DROP_MARKER = None
ENERGY_INSET_BACKGROUND_ALPHA = 0.9

SHOW_GRID = False
PANEL_DPI = 350
FINAL_DPI = 300
PANEL_FONT_SIZE = 7.5
LEGEND_FONT_SIZE = 6.0
FLOWCHART_CROP_PAD_INCHES = 0.02
PANEL_CROP_PAD_INCHES = 0.02
PANEL_SUBPLOT_MARGINS = {
    "left": 0.13,
    "right": 0.90,
    "bottom": 0.18,
    "top": 0.93,
}

A4_LANDSCAPE_INCHES = (9.35, 6.62)

# Panel geometry.  Edit one tuple per panel: (left, bottom, width), all in
# normalized figure coordinates.  The height is calculated automatically from
# that panel's native aspect ratio, so changing a width cannot stretch it.
# For panels whose top edge should stay fixed, edit PANEL_TOP_EDGES below.
w = 0.27
h = 0.0  # vertical offset applied to the entire top row
top_row_bottom = 0.52 + h
# If True, PANEL_TOP_EDGES fixes the upper edge and overrides the bottom value
# in PANEL_LAYOUT. Leave False when using h to move the top row vertically.
USE_FIXED_PANEL_TOP_EDGES = False
PANEL_LAYOUT = {
    "energy": (0.02, top_row_bottom, w+0.03),
    "reversibility_er": (0.33, top_row_bottom, w),
    "reversibility_es": (0.64, top_row_bottom, w),
    "ccdf_ks": (0.02, 0.02, w),
    "xmin_scan": (0.295, 0.02, w+0.04),
    "mle_fit": (0.61, 0.02, w+0.1),
}
h=0.8
PANEL_TOP_EDGES = {
    "energy": h,
    "reversibility_er": h,
    "reversibility_es": h,
}
PANEL_SOURCE_ASPECT_RATIOS = {
    "energy": 1.22,
    "reversibility_er": 1.32,
    "reversibility_es": 1.32,
    "ccdf_ks": 1.11,
    "xmin_scan": 1.39,
    "mle_fit": 1.38,
}
PANEL_ZORDERS = {
    "energy": 2.2,
    "reversibility_er": 2.5,
    "reversibility_es": 2.5,
    "ccdf_ks": 2.4,
    "xmin_scan": 2.3,
    "mle_fit": 2.2,
}
PANEL_LABELS = {
    "energy": r"(a) Energy drops",
    "reversibility_er": r"(b) Drop detection: $\kappa$",
    "reversibility_es": r"(c) Reversible / irreversible $\Delta E_S$",
    "ccdf_ks": r"(d) KS distance: irreversible $\Delta E_S$",
    "xmin_scan": r"(e) Global $\Delta E_{\min}$",
    "mle_fit": r"(f) MLE fitting",
}
PANEL_TITLE_GAP = 0.008

# Equation geometry.  Edit ``center`` and ``size`` directly in normalized
# figure coordinates.  ``size`` is (width, height); no automatic text sizing
# is applied, which makes iterative layout adjustments predictable.
eqh=0.45
EQUATION_BOXES = {
    "energy_eq": {
        "center": (0.2, eqh),
        "size": (0.18, 0.070),
        "text": (
            r"$\Delta E_S=\widehat E_{n+1}-E_{n+1}$"
            "\n"
            r"$\widehat E_{n+1}=E_n+V_0\langle\sigma_{12}\rangle_n\Delta\gamma_n"
            r"+\frac{V_0}{2}a_{1212,n}(\Delta\gamma_n)^2$"
        ),
        "color": "C1",
    },
    "ks_eq": {
        "center": (0.4, eqh),
        "size": (0.18, 0.12),
        "text": (
            r"$D(\Delta E_{\min})=\sup_{x\geq\Delta E_{\min}}"
            r"|\widehat P_{>}(x)-P_{>}^{\rm TPL}(x)|$"
            "\n"
            r"$\widehat P_{>}(x)=|\mathcal{X}_{\min}|^{-1}"
            r"\sum_{\Delta E_S\in\mathcal{X}_{\min}}\mathbf{1}(\Delta E_S>x)$"
            "\n"
            r"$P_{>}^{\rm TPL}(x)=\int_x^\infty p(u\mid\hat\alpha,\hat\lambda,"
            r"\Delta E_{\min})\,du$"
            "\n"
            r"$\mathcal{X}_{\min}=\{\Delta E_S:\Delta E_S\geq\Delta E_{\min}\}$"
        ),
        "color": "C3",
    },
    "split_eq": {
        "center": (0.6, eqh),
        "size": (0.17, 0.065),
        "text": (
            r"$\Delta E_R<\Delta E_{R,\min}:\ \mathrm{reversible}$"
            "\n"
            r"$\Delta E_R\geq\Delta E_{R,\min}:\ \mathrm{irreversible}$"
        ),
        "color": "C4",
    },
    "fit_eq": {
        "center": (0.8, eqh+0.007),
        "size": (0.18, 0.105),
        "text": (
            r"$p(\Delta E_S\mid\alpha,\lambda,\Delta E_{\min})"
            r"=\frac{\Delta E_S^{-\alpha}e^{-\lambda\Delta E_S}}"
            r"{Z(\alpha,\lambda,\Delta E_{\min})}$"
            "\n"
            r"$\ell(\alpha,\lambda)=\sum_{\Delta E_S\in\mathcal{X}_{\min}}"
            r"\ln p(\Delta E_S\mid\alpha,\lambda,\Delta E_{\min})$"
            "\n"
            r"$(\hat\alpha,\hat\lambda)=\arg\max_{\alpha,\lambda}\,\ell(\alpha,\lambda)$"
        ),
        "color": "C2",
    },
}
EQUATION_BOX_ALPHA = 0.14
EQUATION_BOX_EDGE_COLOR = "0.65"
EQUATION_BOX_LINEWIDTH = 0.8
EQUATION_FONT_SIZE = 5.8

ANCHOR_FRACTIONS = {
    "top": (0.5, 1.0),
    "bottom": (0.5, 0.0),
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
    "topleft": (0.2, 1.0),
    "topright": (1.0, 1.0),
    "figbottomleft": (0.1, 0.1),
    "bottomleft": (0.0, 0.0),
    "bottomright": (1.0, 0.1),
}
PLOT_CONNECTIONS = (
    ("energy", "topright", "reversibility_er", "bottomleft"),
    ("reversibility_er", "right", "reversibility_es", "left"),
    ("reversibility_es", "bottomleft", "ccdf_ks", "topright"),
    ("ccdf_ks", "right", "xmin_scan", "left"),
    ("xmin_scan", "right", "mle_fit", "left"),
)
# Each equation arrow is (source box, source anchor, destination panel,
# destination anchor).  Valid anchors are top, bottom, left, right, topleft,
# topright, bottomleft, and bottomright.
EQUATION_CONNECTIONS = (
    ("energy_eq", "top", "energy", "bottom"),
    ("split_eq", "top", "reversibility_er", "bottomright"),
    ("split_eq", "top", "reversibility_es", "figbottomleft"),
    ("ks_eq", "bottom", "ccdf_ks", "topright"),
    ("ks_eq", "bottom", "xmin_scan", "topleft"),
    ("fit_eq", "bottomleft", "mle_fit", "topleft"),
)
SHOW_PLOT_ARROWS = False
SHOW_EQUATION_ARROWS = True
PLOT_ARROW_COLOR = "0.18"
EQUATION_ARROW_COLOR = "0.55"
PLOT_ARROW_LINEWIDTH = 1.5
EQUATION_ARROW_LINEWIDTH = 1.2
ARROW_MUTATION_SCALE = 10
ARROW_ZORDER = 8.0
EQUATION_BOX_ZORDER = 7.0
FLOWCHART_TEXT_ZORDER = 9.0


# =============================================================================
# PATHS AND DATA
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "Plots" / "powerLaw" / "truncated_powerlaw_reversibility_flowchart"
PANEL_CACHE_DIR = OUTPUT_DIR / "panels"
XMIN_FIT_CACHE_DIR = OUTPUT_DIR / "xmin_fit_cache"
EXHAUSTIVE_XMIN_CACHE_DIR = OUTPUT_DIR / "exhaustive_xmin_cache"
ANALYSIS_SUMMARY_PATH = OUTPUT_DIR / "analysis_summary.json"
FINAL_PNG = OUTPUT_DIR / "truncated_powerlaw_reversibility_flowchart.png"
FINAL_PDF = OUTPUT_DIR / "truncated_powerlaw_reversibility_flowchart.pdf"
PANEL_FILES = {name: PANEL_CACHE_DIR / f"{name}.png" for name in PANEL_LAYOUT}
PANEL_PDF_FILES = {name: PANEL_CACHE_DIR / f"{name}.pdf" for name in PANEL_LAYOUT}


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
    source_data_dir = (
        Path(__file__).resolve().parents[1]
        / "Plots"
        / "powerLaw"
        / "truncated_powerlaw_flowchart"
        / "data"
    )
    found = []
    for config in configs:
        for root in DATA_SEARCH_ROOTS + (source_data_dir, OUTPUT_DIR / "data"):
            match = next(
                (path for path in _candidate_paths(root, config.name) if path.is_file()),
                None,
            )
            if match is not None:
                found.append(match)
                break
    return found


def _resolve_csv_paths():
    if CSV_PATHS:
        paths = [Path(path).expanduser() for path in CSV_PATHS]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing explicit CSV paths: {missing}")
        return paths
    configs, _ = _l250_configs_and_labels()
    paths = _find_requested_csvs(configs)
    if paths and (not REQUIRE_ALL_REQUESTED_SEEDS or len(paths) == len(configs)):
        return paths
    raise FileNotFoundError(
        f"Found {len(paths)} of {len(configs)} requested L={REQUESTED_SYSTEM_SIZE} CSVs."
    )


def _read_steps(path):
    from Plotting.truncated_powerlaw_flowchart import _read_steps as read_steps

    return read_steps(path)


def _fit(data, *, refine=True, use_cache=True):
    return make_fit(
        np.asarray(data, dtype=float),
        distType=Truncated_Power_Law,
        use_cache=bool(use_cache),
        cache_dir=str(XMIN_FIT_CACHE_DIR),
        parallel_xmin=False,
        xmin_search_kwargs={
            "nr_initial": XMIN_CANDIDATE_COUNT,
            "min_tail_count": XMIN_MIN_TAIL_COUNT,
            "refine": bool(refine),
        },
    )


def _marker_stride(point_count, max_markers):
    """Return a Matplotlib ``markevery`` stride for dense vector plots."""
    point_count = int(point_count)
    max_markers = max(1, int(max_markers))
    return max(1, int(np.ceil(point_count / max_markers)))


def _exhaustive_xmin_analysis(data):
    """Evaluate every observed xmin retaining the requested tail size."""
    drops = np.asarray(data, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > MIN_DROP)]
    if drops.size < XMIN_MIN_TAIL_COUNT:
        raise RuntimeError(
            f"Need at least {XMIN_MIN_TAIL_COUNT} events for an exhaustive xmin search."
        )

    sorted_drops = np.sort(drops)
    signature = hashlib.sha1()
    signature.update(np.ascontiguousarray(sorted_drops, dtype=np.float64).tobytes())
    signature.update(
        f"|{XMIN_MIN_TAIL_COUNT}|{Truncated_Power_Law.name}|{TRUE_GLOBAL_PARALLEL}".encode()
    )
    cache_path = EXHAUSTIVE_XMIN_CACHE_DIR / f"{signature.hexdigest()}.npz"

    candidate_max = float(sorted_drops[-XMIN_MIN_TAIL_COUNT])
    xmins = np.unique(sorted_drops[sorted_drops <= candidate_max])
    if xmins.size < 2:
        raise RuntimeError("Fewer than two exhaustive xmin candidates are available.")

    cached = False
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                cached_xmins = np.asarray(cache["xmins"], dtype=float)
                cached_distances = np.asarray(cache["distances"], dtype=float)
                cached_valid = np.asarray(cache["valid_fits"], dtype=bool)
                cached_alphas = np.asarray(cache["alphas"], dtype=float)
            if (
                cached_xmins.shape == xmins.shape
                and np.array_equal(cached_xmins, xmins)
                and cached_distances.shape == xmins.shape
                and cached_valid.shape == xmins.shape
                and cached_alphas.shape == xmins.shape
            ):
                distances = cached_distances
                valid_fits = cached_valid
                alphas = cached_alphas
                cached = True
        except (OSError, ValueError, KeyError):
            cached = False

    if cached:
        print(f"Loaded exhaustive xmin cache: {cache_path}")
    else:
        distances, param_vals, valid_fits = evaluate_xmin_distances(
            drops,
            xmins,
            distType=Truncated_Power_Law,
            parallel=TRUE_GLOBAL_PARALLEL,
        )
        distances = np.asarray(distances, dtype=float)
        valid_fits = np.asarray(valid_fits, dtype=bool)
        alphas = np.asarray(
            [
                values[0] if values is not None and len(values) else np.nan
                for values in param_vals
            ],
            dtype=float,
        )
        EXHAUSTIVE_XMIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            xmins=xmins,
            distances=distances,
            valid_fits=valid_fits,
            alphas=alphas,
        )
        print(f"Saved exhaustive xmin cache: {cache_path}")

    finite = np.isfinite(distances)
    preferred = finite & valid_fits
    if not np.any(preferred):
        preferred = finite
    if not np.any(preferred):
        raise RuntimeError("The exhaustive xmin search produced no finite distances.")
    candidate_indices = np.flatnonzero(preferred)
    best_index = int(candidate_indices[np.argmin(distances[preferred])])
    tail_counts = np.asarray(
        [np.count_nonzero(drops >= xmin) for xmin in xmins],
        dtype=int,
    )
    evaluated_xmins = xmins[finite]
    evaluated_distances = distances[finite]
    return {
        "method": "exhaustive_global_min",
        "selection_mode": "all_observed_xmins",
        "xmins": xmins,
        "distances": distances,
        "param_vals": [[alpha] for alpha in alphas],
        "alphas": alphas,
        "sigmas": np.full(xmins.shape, np.nan),
        "valid_fits": valid_fits,
        "tail_counts": tail_counts,
        "simple_drop_xmin": np.nan,
        "simple_drop_distance": np.nan,
        "simple_drop_details": {},
        "global_min_xmin": float(xmins[best_index]),
        "global_min_distance": float(distances[best_index]),
        "global_search_details": {
            "mode": "exhaustive",
            "xmins": xmins,
            "distances": distances,
            "valid_fits": valid_fits,
            "evaluated_xmins": evaluated_xmins.tolist(),
            "evaluated_distances": evaluated_distances.tolist(),
            "rough_local_minima": [],
            "local_minima": [],
            "candidate_count": int(xmins.size),
            "candidate_max": candidate_max,
        },
        "all_evaluations": {
            float(xmin): float(distance)
            for xmin, distance in zip(evaluated_xmins, evaluated_distances)
        },
        "nr_initial": int(xmins.size),
        "min_tail_count": int(XMIN_MIN_TAIL_COUNT),
        "max_xmin": None,
        "refinement": "exhaustive",
    }


def _collect_analysis(csv_paths):
    raw_df, raw_steps, raw_info = _read_steps(csv_paths[0])
    first_strain_limit = tuple(
        float(value)
        for value in _resolve_strain_lim(STRAIN_LIMIT, df=raw_df, postRegime=True)
    )

    # Keep the original panel-a trace, while collecting paired E_R/E_S rows
    # for the cross-quantity reversibility split.
    records = []
    kappa_all = []
    for path in csv_paths:
        df = read_macrodata_csv(str(path), L=REQUESTED_SYSTEM_SIZE)
        load = np.asarray(df["load"], dtype=float)
        delta_e_r_rows = -np.asarray(
            df["total_e_change_from_init"], dtype=float
        )[1:]
        kappa_rows = np.full(load.shape, np.nan, dtype=float)
        kappa_rows[1:] = kappa_from_relaxation_energy(
            delta_e_r_rows,
            np.diff(load),
            float(REQUESTED_SYSTEM_SIZE * REQUESTED_SYSTEM_SIZE),
        )
        strain_lim = _resolve_strain_lim(STRAIN_LIMIT, df=df, postRegime=True)
        er_values, er_mask, er_signed, er_info = extract_energy_drops_from_dataframe(
            df,
            csv_file_path=str(path),
            metadata=get_metadata(str(path)),
            strain_lim=strain_lim,
            energy_key="total_e_change_from_init",
            average_energy=False,
            stress_corrected=False,
            drop_sign="negative",
            min_drop=MIN_DROP,
        )
        _, es_mask, es_signed, es_info = extract_energy_drops_from_dataframe(
            df,
            csv_file_path=str(path),
            metadata=get_metadata(str(path)),
            strain_lim=strain_lim,
            average_energy=AVERAGE_ENERGY,
            stress_corrected=True,
            correction_order=2,
            tangent="current",
            drop_sign="negative",
            min_drop=MIN_DROP,
        )
        er_rows = np.asarray(er_mask, dtype=bool)
        kappa_all.extend(kappa_rows[er_rows].tolist())
        records.append(
            {
                "path": str(path),
                "er_mask": np.asarray(er_mask, dtype=bool),
                "er_signed": np.asarray(er_signed, dtype=float),
                "kappa": kappa_rows,
                "es_mask": np.asarray(es_mask, dtype=bool),
                "es_signed": np.asarray(es_signed, dtype=float),
                "er_info": er_info,
                "es_info": es_info,
            }
        )

    kappa_all = np.asarray(kappa_all, dtype=float)
    kappa_all = kappa_all[np.isfinite(kappa_all) & (kappa_all > MIN_DROP)]
    kappa_det = (
        float(KAPPA_DET_OVERRIDE)
        if KAPPA_DET_OVERRIDE is not None
        else kappa_detection_threshold()
    )
    kappa_rev = kappa_all[kappa_all < kappa_det]
    kappa_irrev = kappa_all[kappa_all >= kappa_det]
    es_rev = []
    es_irrev = []
    for record in records:
        er_valid = (
            record["er_mask"]
            & np.isfinite(record["er_signed"])
            & np.isfinite(record["kappa"])
            & (record["kappa"] > MIN_DROP)
        )
        rev_rows = er_valid & (record["kappa"] < kappa_det)
        irrev_rows = er_valid & (record["kappa"] >= kappa_det)
        es_rev_rows = record["es_mask"] & rev_rows
        es_irrev_rows = record["es_mask"] & irrev_rows
        es_rev.extend((-record["es_signed"][es_rev_rows]).tolist())
        es_irrev.extend((-record["es_signed"][es_irrev_rows]).tolist())
    es_rev = np.asarray(es_rev, dtype=float)
    es_rev = es_rev[
        np.isfinite(es_rev) & (es_rev > MIN_DROP)
    ]
    es_irrev = np.asarray(es_irrev, dtype=float)
    es_irrev = es_irrev[
        np.isfinite(es_irrev) & (es_irrev > MIN_DROP)
    ]
    if es_irrev.size < XMIN_MIN_TAIL_COUNT:
        raise RuntimeError(
            f"Only {es_irrev.size} irreversible Delta E_S drops were found."
        )

    use_true_global = (
        USE_TRUE_GLOBAL_FOR_SMALL_DATA
        and es_irrev.size < TRUE_GLOBAL_EVENT_THRESHOLD
    )
    if use_true_global:
        es_fit = None
        es_analysis = _exhaustive_xmin_analysis(es_irrev)
    else:
        es_fit = _fit(es_irrev)
        es_analysis = es_fit.xmin_analysis
    es_xmin_ks = float(es_analysis["global_min_xmin"])
    es_fit_fixed = Fit(
        data=es_irrev,
        xmin=es_xmin_ks,
        xmin_distribution=Truncated_Power_Law.name,
        verbose=0,
    )

    return {
        "csv_paths": [str(path) for path in csv_paths],
        "raw_df": raw_df,
        "raw_steps": raw_steps,
        "raw_info": raw_info,
        "first_strain_limit": first_strain_limit,
        "kappa_all": kappa_all,
        "kappa_det": kappa_det,
        "kappa_rev": kappa_rev,
        "kappa_irrev": kappa_irrev,
        "es_rev": es_rev,
        "es_irrev": es_irrev,
        "es_fit": es_fit,
        "es_analysis": es_analysis,
        "xmin_selection_mode": (
            "exhaustive" if use_true_global else "coarse_local_search"
        ),
        "xmin_selection_is_approximate": not use_true_global,
        "es_xmin_ks": es_xmin_ks,
        "es_fit_fixed": es_fit_fixed,
    }


# =============================================================================
# PANEL GENERATION
# =============================================================================

def _new_panel(name, *, figsize=None):
    if figsize is None:
        _, _, width = PANEL_LAYOUT[name]
        physical_width = width * A4_LANDSCAPE_INCHES[0]
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
    PANEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
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
    strain = np.asarray(df["load"], dtype=float)
    energy = np.asarray(df[info["energy_col"]], dtype=float)
    drop_strain = np.asarray(steps["load_ip1"], dtype=float)
    drop_values = np.asarray(steps["stress_corrected_drop_second_order"], dtype=float)
    limit = analysis["first_strain_limit"]
    curve_mask = (strain > limit[0]) & (strain < limit[1])
    drop_mask = (drop_strain > limit[0]) & (drop_strain < limit[1])
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
    for axis in (drop_ax, inset_ax, inset_drop_ax):
        axis.grid(False, which="both")
    _save_panel(fig, "energy")


def _set_pdf_axes(ax, quantity):
    """Apply log-PDF axes with labels tied to the plotted quantity."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"${quantity}$")
    ax.set_ylabel(rf"$p({quantity})$")
    ax.grid(False, which="both")


def _format_scientific_math(value, *, decimals=1):
    """Format a number as compact MathText, e.g. ``4.9\times10^{-6}``."""
    mantissa, exponent = f"{float(value):.{decimals}e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def _reversibility_er_panel(analysis):
    """Show the kappa PDF and its material detection threshold."""
    fig, ax = _new_panel("reversibility_er")
    kappa_all = analysis["kappa_all"]
    kappa_det = analysis["kappa_det"]
    plot_data_pdf(
        ax,
        kappa_all,
        label="All $\kappa$ events",
        color=ER_ALL_COLOR,
        drop_label=r"\kappa",
        show_legend=False,
    )
    _set_pdf_axes(ax, r"\kappa")
    lo, hi = ax.get_xlim()
    ax.axvspan(lo, kappa_det, color=REVERSIBLE_COLOR, alpha=0.34, zorder=0)
    ax.axvspan(kappa_det, hi, color=IRREVERSIBLE_COLOR, alpha=0.30, zorder=0)
    ax.axvline(
        kappa_det,
        color=DROP_DETECTION_LINE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label=rf"$\kappa_{{\det}}={_format_scientific_math(kappa_det)}$",
        zorder=4,
    )
    ax.legend(loc="best", fontsize=5.4)
    _save_panel(fig, "reversibility_er")


def _reversibility_es_panel(analysis):
    """Show reversible and irreversible PDFs after the kappa split."""
    fig, ax = _new_panel("reversibility_es")
    reversible = analysis["es_rev"]
    irreversible = analysis["es_irrev"]
    plot_data_pdf(
        ax,
        reversible,
        label=rf"reversible ($n={reversible.size}$)",
        color=REVERSIBLE_COLOR,
        drop_label=r"\Delta E_S",
        show_legend=False,
    )
    ax.lines[-1].set_markersize(2.4)
    plot_data_pdf(
        ax,
        irreversible,
        label=rf"irreversible ($n={irreversible.size}$)",
        color=IRREVERSIBLE_COLOR,
        drop_label=r"\Delta E_S",
        show_legend=False,
    )
    ax.lines[-1].set_markersize(2.4)
    _set_pdf_axes(ax, r"\Delta E_S")
    ax.legend(loc="best", fontsize=5.4)
    _save_panel(fig, "reversibility_es")


def _ccdf_panel(analysis):
    fig, ax = _new_panel("ccdf_ks")
    es_xmin_ks = analysis["es_xmin_ks"]
    plot_ks_distance(
        analysis["es_irrev"],
        es_xmin_ks,
        ax=ax,
        save=False,
        close=False,
        set_title=False,
        show_legend=True,
        empirical_color=ES_COLOR,
        model_color="0.35",
        ks_color=KS_COLOR,
        show_inset=True,
        inset_bounds=(0.57, 0.33, 0.39, 0.34),
        inset_x_factor=1.15,
        inset_grid=False,
        tight_layout=False,
    )
    ax.set_xlabel(r"$\Delta E_S$ (irreversible events)")
    ax.set_ylabel(r"$P(\Delta E_S>x)$")
    ax.legend(loc="best", fontsize=5.5)
    _save_panel(fig, "ccdf_ks")


def _xmin_panel(analysis):
    """Plot raw D(x_min), with search markers only for approximate scans."""
    fig, ax = _new_panel("xmin_scan")
    scan = analysis["es_analysis"]
    xmins = np.asarray(scan["xmins"], dtype=float)
    distances = np.asarray(scan["distances"], dtype=float)
    finite = np.isfinite(xmins) & np.isfinite(distances) & (xmins > 0)
    order = np.argsort(xmins[finite])
    xmins = xmins[finite][order]
    distances = distances[finite][order]
    ax.plot(
        xmins,
        distances,
        color="0.35",
        marker="o",
        markersize=1.8,
        markevery=_marker_stride(xmins.size, XMIN_MAX_MARKERS),
        linewidth=0.65,
        alpha=0.82,
        label=r"$D(x_{\min})$",
    )

    details = scan.get("global_search_details", {})
    rough = details.get("rough_local_minima", [])
    searched = details.get("local_minima", [])
    is_approximate = bool(analysis.get("xmin_selection_is_approximate", False))
    if is_approximate and rough:
        rough_x = np.asarray([entry["xmin"] for entry in rough], dtype=float)
        rough_d = np.asarray([entry["distance"] for entry in rough], dtype=float)
        ax.scatter(
            rough_x,
            rough_d,
            marker="+",
            s=18,
            linewidths=0.65,
            color="0.35",
            alpha=0.55,
            label="coarse local minima",
            zorder=4,
        )
    if is_approximate and searched:
        local_x = np.asarray([entry["xmin"] for entry in searched], dtype=float)
        local_d = np.asarray([entry["distance"] for entry in searched], dtype=float)
        ax.scatter(
            local_x,
            local_d,
            marker="x",
            s=16,
            linewidths=0.65,
            color="0.35",
            alpha=0.55,
            label="searched local minima",
            zorder=4,
        )

    es_xmin_ks = float(analysis["es_xmin_ks"])
    global_distance = float(analysis["es_analysis"]["global_min_distance"])
    ax.scatter(
        [es_xmin_ks],
        [global_distance],
        marker="o",
        s=20,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        label="global minimum*" if is_approximate else "global minimum",
        zorder=6,
    )
    ax.axvline(es_xmin_ks, color="0.20", linestyle=":", linewidth=0.9, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta E_{S,\min}^{KS}$")
    ax.set_ylabel(r"$D$")
    ax.grid(False, which="both")
    ax.legend(loc="best", fontsize=5.0)
    _save_panel(fig, "xmin_scan")


def _fit_panel(analysis):
    """Fit panel with alpha(xmin) on a twin y-axis."""
    fig, ax1 = _new_panel("mle_fit")
    fit = analysis["es_fit_fixed"]
    alpha_fit = dist_from_fit(fit)
    plot_data_and_fit(
        fit,
        ax=ax1,
        data_info={"drop_label": r"E_S"},
        color=ES_COLOR,
        data_color=ES_COLOR,
        addFit=True,
        save=False,
        show=False,
        close=False,
        show_fit_region=True,
        show_cutoff=True,
        show_title=False,
        show_legend=False,
        xmin_analysis=False,
    )
    ax1.lines[0].set_label("Irreversible $\u0394E_S$ PDF")
    ax1.lines[0].set_markersize(1.6)
    ax1.lines[0].set_markevery(
        _marker_stride(len(ax1.lines[0].get_xdata()), FIT_PDF_MAX_MARKERS)
    )
    if len(ax1.lines) > 1:
        ax1.lines[1].set_label(
            rf"MLE: $\hat{{\alpha}}={alpha_fit.alpha:.2f}$, "
            rf"$\hat{{\lambda}}={alpha_fit.Lambda:.1e}$"
        )
    # Keep the gray fit-region shading, but omit its verbose label from the
    # combined legend.
    for patch in ax1.patches:
        patch.set_label("_nolegend_")
    ax1.set_xlabel(r"$\Delta E$")
    ax1.set_ylabel(r"$p(\Delta E_S)$")
    ax1.grid(False, which="both")

    ax2 = ax1.twinx()
    # The secondary-axis curves should remain visible over the combined
    # legend.  A transparent patch lets them draw above ax1's legend.
    ax2.patch.set_visible(False)
    scan = analysis["es_analysis"]
    x = np.asarray(scan["xmins"], dtype=float)
    alpha = np.asarray(scan["alphas"], dtype=float)
    valid = np.asarray(scan.get("valid_fits", np.ones_like(x, dtype=bool)), dtype=bool)
    valid &= np.isfinite(x) & np.isfinite(alpha) & (x > 0)
    order = np.argsort(x[valid])
    x_valid = x[valid][order]
    alpha_valid = alpha[valid][order]
    ax2.plot(
        x_valid,
        alpha_valid,
        color=ALPHA_COLOR,
        marker="s",
        markersize=2.8,
        markevery=_marker_stride(x_valid.size, ALPHA_MAX_MARKERS),
        linewidth=1.1,
        label=r"$\alpha(\Delta E_{S,\min}^{KS})$",
        zorder=4,
    )
    es_xmin_ks = analysis["es_xmin_ks"]
    ax2.axvline(
        es_xmin_ks,
        color="0.25",
        linestyle=":",
        linewidth=1.0,
        label=(
            rf"$\Delta E_{{S,\min}}^{{KS,*}}={es_xmin_ks:.2e}$"
            if analysis.get("xmin_selection_is_approximate", False)
            else rf"$\Delta E_{{S,\min}}^{{KS}}={es_xmin_ks:.2e}$"
        ),
        zorder=3,
    )
    ax2.set_xscale("log")
    # Keep the alpha curve colored, but use the normal black axis styling so
    # the twin axis does not dominate the panel visually.
    ax2.set_ylabel(r"$\alpha(\Delta E_{S,\min}^{KS})$")
    ax2.tick_params(axis="y", colors="black")
    ax2.spines["right"].set_color("black")
    ax2.grid(False, which="both")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="center left",
        fontsize=4.2,
        framealpha=0.82,
    )
    legend.set_zorder(1.0)
    _save_panel(fig, "mle_fit")


def _write_analysis_summary(analysis):
    es_fit_dist = dist_from_fit(analysis["es_fit_fixed"])
    summary = {
        "csv_paths": analysis["csv_paths"],
        "post_yield": True,
        "number_of_kappa_events": int(analysis["kappa_all"].size),
        "kappa_det": analysis["kappa_det"],
        "mu": float(2.0 * analysis["kappa_det"]),
        "reversible_kappa_events": int(analysis["kappa_rev"].size),
        "irreversible_kappa_events": int(analysis["kappa_irrev"].size),
        "reversible_delta_E_S_events": int(analysis["es_rev"].size),
        "irreversible_delta_E_S_events": int(analysis["es_irrev"].size),
        "xmin_selection_mode": analysis["xmin_selection_mode"],
        "xmin_selection_is_approximate": bool(
            analysis["xmin_selection_is_approximate"]
        ),
        "xmin_candidate_count": int(analysis["es_analysis"]["xmins"].size),
        "true_global_event_threshold": TRUE_GLOBAL_EVENT_THRESHOLD,
        "kappa_detection_method": KAPPA_DETECTION_METHOD_LABEL,
        "delta_E_S_xmin_ks": analysis["es_xmin_ks"],
        "delta_E_S_xmin_ks_distance": analysis["es_analysis"]["global_min_distance"],
        "delta_E_S_xmin_ks_alpha": float(es_fit_dist.alpha),
        "delta_E_S_xmin_ks_lambda": float(es_fit_dist.Lambda),
        "delta_E_S_xmin_ks_D": float(es_fit_dist.D),
        "fit_population": "irreversible_delta_E_S",
    }
    ANALYSIS_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def regenerate_panels():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = _resolve_csv_paths()
    analysis = _collect_analysis(paths)
    _energy_panel(analysis)
    _reversibility_er_panel(analysis)
    _reversibility_es_panel(analysis)
    _ccdf_panel(analysis)
    _xmin_panel(analysis)
    _fit_panel(analysis)
    return _write_analysis_summary(analysis)


# =============================================================================
# FLOWCHART COMPOSITION
# =============================================================================

def _load_summary():
    missing = [
        path
        for path in (*PANEL_FILES.values(), *PANEL_PDF_FILES.values())
        if not path.is_file()
    ]
    if missing or not ANALYSIS_SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            "Missing cached reversibility panels; run --regenerate-subplots first."
        )
    return json.loads(ANALYSIS_SUMMARY_PATH.read_text())


def _calculate_panel_positions():
    page_width, page_height = A4_LANDSCAPE_INCHES
    positions = {}
    for name, (left, bottom, width) in PANEL_LAYOUT.items():
        image = mpimg.imread(PANEL_FILES[name])
        pixel_height, pixel_width = image.shape[:2]
        native_aspect_ratio = pixel_width / pixel_height
        height = width * page_width / (native_aspect_ratio * page_height)
        if USE_FIXED_PANEL_TOP_EDGES and name in PANEL_TOP_EDGES:
            bottom = PANEL_TOP_EDGES[name] - height
        positions[name] = (left, bottom, width, height)
    return positions


def _equation_positions():
    positions = {}
    for name, spec in EQUATION_BOXES.items():
        cx, cy = spec["center"]
        width, height = spec["size"]
        positions[name] = (cx - width / 2, cy - height / 2, width, height)
    return positions


def _anchor(node_positions, node_name, anchor_name):
    left, bottom, width, height = node_positions[node_name]
    fx, fy = ANCHOR_FRACTIONS[anchor_name]
    return left + fx * width, bottom + fy * height


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


def _add_equation_boxes(fig):
    positions = _equation_positions()
    for name, spec in EQUATION_BOXES.items():
        left, bottom, width, height = positions[name]
        fig.add_artist(
            FancyBboxPatch(
                (left, bottom),
                width,
                height,
                boxstyle="round,pad=0.0,rounding_size=0.004",
                transform=fig.transFigure,
                facecolor=mpl.colors.to_rgba(spec["color"], EQUATION_BOX_ALPHA),
                edgecolor=EQUATION_BOX_EDGE_COLOR,
                linewidth=EQUATION_BOX_LINEWIDTH,
                zorder=EQUATION_BOX_ZORDER,
            )
        )
        cx, cy = spec["center"]
        fig.text(
            cx,
            cy,
            spec["text"],
            ha="center",
            va="center",
            fontsize=EQUATION_FONT_SIZE,
            linespacing=1.12,
            zorder=FLOWCHART_TEXT_ZORDER,
        )
    return positions


def _flowchart_crop_bbox(fig):
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    return Bbox.from_extents(
        tight_bbox.x0 - FLOWCHART_CROP_PAD_INCHES,
        tight_bbox.y0 - FLOWCHART_CROP_PAD_INCHES,
        tight_bbox.x1 + FLOWCHART_CROP_PAD_INCHES,
        tight_bbox.y1 + FLOWCHART_CROP_PAD_INCHES,
    )


def _write_vector_pdf(fig, panel_axes, panel_positions, crop_bbox_inches):
    layout_pdf = OUTPUT_DIR / "_reversibility_flowchart_layout.pdf"
    try:
        for ax in panel_axes:
            ax.set_visible(False)
        fig.savefig(layout_pdf, transparent=True)
        overlay_page = PdfReader(str(layout_pdf)).pages[0]
        page_width = float(overlay_page.mediabox.width)
        page_height = float(overlay_page.mediabox.height)
        fig_width, fig_height = fig.get_size_inches()
        crop_left = crop_bbox_inches.x0 / fig_width * page_width
        crop_bottom = crop_bbox_inches.y0 / fig_height * page_height
        crop_width = crop_bbox_inches.width / fig_width * page_width
        crop_height = crop_bbox_inches.height / fig_height * page_height
        output_page = PageObject.create_blank_page(width=crop_width, height=crop_height)
        for name, (left, bottom, width, _height) in sorted(
            panel_positions.items(), key=lambda item: PANEL_ZORDERS[item[0]]
        ):
            panel_page = PdfReader(str(PANEL_PDF_FILES[name])).pages[0]
            panel_width = float(panel_page.mediabox.width)
            scale = width * page_width / panel_width
            output_page.merge_transformed_page(
                panel_page,
                Transformation()
                .scale(scale)
                .translate(
                    left * page_width - crop_left,
                    bottom * page_height - crop_bottom,
                ),
            )
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


def compose_flowchart(summary):
    fig = plt.figure(figsize=A4_LANDSCAPE_INCHES, facecolor="white")
    panel_positions = _calculate_panel_positions()
    equation_positions = _add_equation_boxes(fig)
    nodes = {**panel_positions, **equation_positions}

    if SHOW_PLOT_ARROWS:
        for source, source_anchor, target, target_anchor in PLOT_CONNECTIONS:
            _add_arrow(
                fig,
                _anchor(nodes, source, source_anchor),
                _anchor(nodes, target, target_anchor),
                color=PLOT_ARROW_COLOR,
                linewidth=PLOT_ARROW_LINEWIDTH,
            )
    if SHOW_EQUATION_ARROWS:
        for source, source_anchor, target, target_anchor in EQUATION_CONNECTIONS:
            _add_arrow(
                fig,
                _anchor(nodes, source, source_anchor),
                _anchor(nodes, target, target_anchor),
                color=EQUATION_ARROW_COLOR,
                linewidth=EQUATION_ARROW_LINEWIDTH,
            )

    panel_axes = []
    for name, position in panel_positions.items():
        ax = fig.add_axes(position, zorder=PANEL_ZORDERS[name])
        ax.imshow(mpimg.imread(PANEL_FILES[name]), aspect="equal")
        ax.set_axis_off()
        panel_axes.append(ax)
        x, y, width, height = position
        fig.text(
            x + 0.5 * width,
            y + height + PANEL_TITLE_GAP,
            PANEL_LABELS[name],
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="semibold",
            zorder=FLOWCHART_TEXT_ZORDER,
        )

    crop_bbox_inches = _flowchart_crop_bbox(fig)
    # fig.savefig(
    #     FINAL_PNG,
    #     dpi=FINAL_DPI,
    #     facecolor="white",
    #     bbox_inches=crop_bbox_inches,
    #     pad_inches=0.0,
    # )
    _write_vector_pdf(fig, panel_axes, panel_positions, crop_bbox_inches)
    return fig


def generate_flowchart(*, regenerate_subplots=None, show=False):
    _configure_matplotlib()
    regenerate = (
        REGENERATE_SUBPLOTS
        if regenerate_subplots is None
        else bool(regenerate_subplots)
    )
    if regenerate:
        summary = regenerate_panels()
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


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=POWERLAW_STANDARD_WORKFLOW,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--regenerate-subplots", action="store_true")
    mode.add_argument("--reuse-subplots", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.regenerate_subplots:
        regenerate = True
    elif args.reuse_subplots:
        regenerate = False
    else:
        # Running the file without arguments always composes the current
        # flowchart.  By default it reuses the cached panels; set
        # REGENERATE_SUBPLOTS=True above when the data panels should refresh.
        regenerate = REGENERATE_SUBPLOTS
    generate_flowchart(regenerate_subplots=regenerate, show=args.show)


if __name__ == "__main__":
    main()
