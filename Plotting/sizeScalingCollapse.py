"""Finite-size collapse of size-scaling energy-drop protocols.

Implements the logarithmic overlap-variance procedure in Salman et al. (2025),
Eq. (11)-(12). Run from the repository root with

    .venv/bin/python -m Plotting.sizeScalingCollapse

Each stage is cached below ``Plots/powerLaw/size_collapse/cache``.
By default, classify paired E_R/E_S events with kappa_det and use only finite
positive irreversible E_S. The explicit ``--population-mode all`` alternative
uses all positive drops and writes into a separate ``all_events`` directory.
Collapse curves are normalized tail PDFs above each size's global KS cutoff.
Use ``--stage shared --common-cutoff-size 100 --skip-pvalue`` for an alternate
collapse using the L=100 KS cutoff for every size (``*_common_cutoff.pdf``).
This is a cutoff-sensitivity comparison, not a replacement for per-size fits.
For lambda-derived scaling without collapse optimization, use
``--stage lambda-scaling --lambda-cutoff common --common-cutoff-size 100
--minimum-size 50 --yield-regime post --skip-pvalue`` with a separate output
folder. It writes the lambda-scaling, fixed-alpha collapse, raw PDFs with
per-size cutoffs, and KS scan as separate PDFs. The cross-size lambda
regression is not an individual-size flowchart fit.
Use ``--lambda-normalization full --lambda-cutoff per-size`` to retain the
full-population PDF normalization while hiding bins below each fitted cutoff.
Use ``--standard-cache-root`` to share the classification analysis caches.
For quick parameter plots using the completed cutoff fits, pass
``--stage parameters --protocols second_order --parameter-xmin-methods global_min
--yield-regime post --largest-first --skip-pvalue`` with the same ``--output``
as the earlier run. These versions are saved under
``xmin_global/parameters_per_size/without_uncertainties``.
These plots compare sizes; for individual-size classification, split, and
fitted-PDF panels use
``truncated_powerlaw_reversibility_flowchart.generate_single_size_panels``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from Management.jobs import nonreconnecting_size_scaling_job
from Management.updateCSV import HEADER_RENAME_MAP, update_df_header
from Plotting.energyDropCalculations import (
    SIGMA12_RESCUE_SENTINEL,
    calculate_energy_step_data,
    validate_sigma12_column,
)
from Plotting.plotPowerLaw import dist_from_fit, getHist, make_fit
from Plotting.standardPowerlaw import (
    EventDrops, POST_YIELD_RULE, SIZE_SCALING_MIN_SIZE, yield_regimes,
    kappa_detection_threshold, kappa_from_relaxation_energy, positive_es, split_by_kappa,
)


PROTOCOLS = (
    "second_order",
    "previous_energy",
    "initial_guess_energy",
)
PROTOCOL_LABELS = {
    "second_order": r"$E_S$ (stress corrected)",
    "previous_energy": r"$E_I$ (inter-strain)",
    "initial_guess_energy": r"$E_R$ (relaxation)",
}
PROTOCOL_DROP_SYMBOLS = {
    "second_order": r"\Delta E_S",
    "previous_energy": r"\Delta E_I",
    "initial_guess_energy": r"\Delta E_R",
}
REGIMES = ("pre", "post")
YIELD_REGIME_NAMES = ("pre", "post")
CACHE_VERSION = 6  # fixed gamma=0 tangent and first-Piola peak yield windows
COLLAPSE_CACHE_VERSION = 5  # invalidate curves selected by the former yield rule
EXPONENT_RANGE = (0.5, 2.5)
DIMENSION_RANGE = (0.25, 2.25)
XMIN_METHOD = "global"


def _last_load(path: Path) -> float:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        position = stream.tell() - 2
        while position > 0:
            stream.seek(position)
            if stream.read(1) == b"\n":
                break
            position -= 1
        last_line = stream.readline().decode(errors="replace").strip()
    header = pd.read_csv(path, nrows=0).columns.tolist()
    values = last_line.split(",")
    return float(values[header.index("load")])


def size_scaling_paths(data_root: Path, seeds_per_size=None, minimum_end_load=None,
                       *, minimum_size=SIZE_SCALING_MIN_SIZE):
    """Use all runs with data after their own first-Piola peak +0.01."""
    if minimum_size < 1:
        raise ValueError("minimum_size must be positive.")
    groups, _ = nonreconnecting_size_scaling_job()
    paths = {}
    inventory = {}
    for group in groups:
        size = int(group[0].rows)
        if size < minimum_size:
            continue
        available = []
        for config in sorted(group, key=lambda item: item.seed):
            path = data_root / f"{config.name}.csv"
            if path.exists():
                if minimum_end_load is None:
                    low, high = stress_peak_regimes(path)["post"]
                    usable = low < high
                else:
                    usable = _last_load(path) > minimum_end_load
                if usable:
                    available.append(path)
        inventory[size] = len(available)
        if not available or (seeds_per_size is not None and len(available) < seeds_per_size):
            raise RuntimeError(
                f"L={size} has {len(available)} usable runs; "
                f"need {seeds_per_size or 'at least one'}."
            )
        paths[size] = available[:seeds_per_size]
    return dict(sorted(paths.items())), inventory


def completed_size_scaling_paths(data_root: Path, seeds_per_size: int, post_hi: float):
    """Compatibility entry point for explicitly completion-limited analyses."""
    return size_scaling_paths(data_root, seeds_per_size, np.nextafter(post_hi, -np.inf))


def _file_cache_path(path: Path, size: int, regimes, cache_dir: Path) -> Path:
    stat = path.stat()
    signature = repr(
        (CACHE_VERSION, POST_YIELD_RULE, str(path), stat.st_size, stat.st_mtime_ns, size, regimes)
    )
    return cache_dir / f"{hashlib.sha1(signature.encode()).hexdigest()}.npz"


def _has_header_transition(path: Path) -> bool:
    with path.open("rb") as stream:
        return any(line.lstrip().lower().startswith(b"#header:") for line in stream)


def _read_mixed_selected(path: Path, wanted: set[str]) -> pd.DataFrame:
    values = {column: [] for column in wanted}
    header = None
    selected_indices = None
    with path.open(newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if not row:
                continue
            token = row[0].strip()
            if header is None or token.lower().startswith("#header:"):
                if header is None:
                    header = [column.strip() for column in row]
                else:
                    header = [token.split(":", 1)[1].strip()] + [
                        column.strip() for column in row[1:]
                    ]
                canonical = [HEADER_RENAME_MAP.get(column, column) for column in header]
                duplicates = {
                    column for column in canonical if canonical.count(column) > 1
                }
                if duplicates:
                    raise ValueError(
                        f"Duplicate canonical columns {sorted(duplicates)} in {path} "
                        f"at line {line_number}."
                    )
                selected_indices = {
                    column: canonical.index(column)
                    for column in wanted
                    if column in canonical
                }
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"Row length mismatch in {path} at line {line_number}: "
                    f"expected {len(header)}, got {len(row)}."
                )
            for column in wanted:
                index = selected_indices.get(column)
                values[column].append(np.nan if index is None else float(row[index]))
    return pd.DataFrame(values)


def _read_energy_step_values(path: Path, size: int):
    wanted = {
        "load_step",
        "load",
        "total_energy",
        "total_energy_change",
        "total_e_change_from_init",
        "avg_sigma12",
        "avg_P12",
    }
    if _has_header_transition(path):
        df = _read_mixed_selected(path, wanted)
    else:
        raw_wanted = wanted | {"avg_sigmaxy", "avg_Pxy"}
        df = pd.read_csv(
            path,
            usecols=lambda column: column in raw_wanted,
            low_memory=False,
        )
        df = update_df_header(df, add_total_columns=False, L=size)
    required = {
        "load",
        "total_energy",
        "total_energy_change",
        "total_e_change_from_init",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}.")
    nonfinite = [column for column in required if not np.isfinite(df[column]).all()]
    if nonfinite:
        raise ValueError(f"Non-finite required columns {nonfinite} in {path}.")
    if "avg_sigma12" not in df:
        raise ValueError(
            f"Missing native avg_sigma12 in {path}; refusing to use avg_P12 "
            "as a substitute."
        )
    validate_sigma12_column(df, context=str(path))
    load = np.asarray(df["load"], dtype=float)
    if not np.all(np.isfinite(load)) or np.any(np.diff(load) <= 0):
        raise ValueError(f"Load must be finite and strictly increasing: {path}")

    steps, _ = calculate_energy_step_data(
        path,
        df=df,
        metadata={"L": size},
        average_energy=False,
    )
    step_load = np.asarray(steps["load_ip1"], dtype=float)
    values = {
        "second_order": np.asarray(
            steps["stress_corrected_drop_second_order"], dtype=float
        ),
        "previous_energy": -np.asarray(df["total_energy_change"], dtype=float)[1:],
        "initial_guess_energy": -np.asarray(
            df["total_e_change_from_init"], dtype=float
        )[1:],
    }
    values["kappa"] = kappa_from_relaxation_energy(
        values["initial_guess_energy"],
        np.asarray(steps["delta_gamma"], dtype=float),
        float(size * size),
    )
    invalid_rows = (
        df["avg_sigma12"].to_numpy(dtype=float) == SIGMA12_RESCUE_SENTINEL
    )
    invalid_steps = invalid_rows[:-1] | invalid_rows[1:]
    for protocol in ("previous_energy", "initial_guess_energy"):
        values[protocol] = values[protocol].copy()
        values[protocol][invalid_steps] = np.nan
    values["kappa"] = values["kappa"].copy()
    values["kappa"][invalid_steps] = np.nan

    for protocol, drops in values.items():
        if drops.shape != step_load.shape:
            raise ValueError(
                f"Shape mismatch for {protocol} in {path}: "
                f"{drops.shape} vs {step_load.shape}."
            )
    return step_load, values


def stress_peak_regimes(path: Path) -> dict[str, tuple[float, float]]:
    """Read only loads and P12 to resolve the shared yield windows."""
    return yield_regimes(_read_mixed_selected(path, {"load", "avg_P12"}))


def extract_run(path: Path, size: int, regimes, cache_dir: Path, force=False):
    if regimes is None or not hasattr(regimes, "items"):
        windows = stress_peak_regimes(path)
        regimes = windows if regimes is None else {name: windows[name] for name in regimes}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _file_cache_path(path, size, regimes, cache_dir)
    if cache_path.exists() and not force:
        with np.load(cache_path) as cached:
            return {key: cached[key] for key in cached.files}

    step_load, values = _read_energy_step_values(path, size)

    extracted = {}
    for protocol, drops in values.items():
        for regime, (low, high) in regimes.items():
            mask = (
                np.isfinite(drops)
                & (drops > 0)
                & (step_load > low)
                & (step_load < high)
            )
            extracted[f"{protocol}_{regime}"] = drops[mask]
    np.savez_compressed(cache_path, **extracted)
    return extracted


def extract_event_pairs(path: Path, size: int, regimes, cache_dir: Path, force=False):
    """Cache E_R/E_S values for the same positive-E_R events.

    The E_R mask is intentionally the only event-selection mask here.  The
    E_S array retains the corresponding position for every event, including
    non-positive or non-finite values, so a later positive-E_S filter cannot
    change which events were labeled irreversible.
    """
    if regimes is None or not hasattr(regimes, "items"):
        windows = stress_peak_regimes(path)
        regimes = windows if regimes is None else {name: windows[name] for name in regimes}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _file_cache_path(path, size, regimes, cache_dir)
    if cache_path.exists() and not force:
        with np.load(cache_path) as cached:
            return {key: cached[key] for key in cached.files}

    step_load, values = _read_energy_step_values(path, size)
    e_r = values["initial_guess_energy"]
    e_s = values["second_order"]
    extracted = {}
    for regime, (low, high) in regimes.items():
        mask = (
            np.isfinite(e_r)
            & (e_r > 0)
            & (step_load > low)
            & (step_load < high)
        )
        extracted[f"initial_guess_energy_{regime}"] = e_r[mask]
        extracted[f"second_order_{regime}"] = e_s[mask]
        extracted[f"kappa_{regime}"] = values["kappa"][mask]
    np.savez_compressed(cache_path, **extracted)
    return extracted


def pool_event_pairs(paths_by_size, regimes, cache_dir: Path, force=False):
    """Pool E_R/E_S event pairs by system size and strain regime."""
    regime_names = tuple(regimes) if regimes is not None else YIELD_REGIME_NAMES
    pooled = {
        regime: {size: {"initial_guess_energy": [], "second_order": [], "kappa": []}
                 for size in paths_by_size}
        for regime in regime_names
    }
    for size, paths in paths_by_size.items():
        print(
            f"Extracting/caching event pairs L={size}: {len(paths)} available runs",
            flush=True,
        )
        per_run = [
            extract_event_pairs(
                path, size, regimes, cache_dir / "runs", force=force
            )
            for path in paths
        ]
        for regime in regime_names:
            for protocol in ("initial_guess_energy", "second_order"):
                arrays = [run[f"{protocol}_{regime}"] for run in per_run]
                pooled[regime][size][protocol] = np.concatenate(arrays)
            pooled[regime][size]["kappa"] = np.concatenate(
                [run[f"kappa_{regime}"] for run in per_run]
            )
            e_r = pooled[regime][size]["initial_guess_energy"]
            e_s = pooled[regime][size]["second_order"]
            kappa = pooled[regime][size]["kappa"]
            if e_r.shape != e_s.shape or e_r.shape != kappa.shape:
                raise RuntimeError(
                    f"Aligned E_R/E_S/kappa shape mismatch for L={size}, {regime}."
                )
    return pooled


def pool_drops(paths_by_size, regimes, cache_dir: Path, force=False):
    regime_names = tuple(regimes) if regimes is not None else YIELD_REGIME_NAMES
    pooled = {
        protocol: {regime: {} for regime in regime_names} for protocol in PROTOCOLS
    }
    for size, paths in paths_by_size.items():
        print(f"Extracting/caching L={size}: {len(paths)} available runs", flush=True)
        per_run = [
            extract_run(path, size, regimes, cache_dir / "runs", force=force)
            for path in paths
        ]
        for protocol in PROTOCOLS:
            for regime in regime_names:
                arrays = [run[f"{protocol}_{regime}"] for run in per_run]
                pooled[protocol][regime][size] = np.concatenate(arrays)
    return pooled


def pool_irreversible_es(paths_by_size, regimes, cache_dir: Path, force=False):
    """Select positive E_S only after classifying paired events with kappa."""
    events = pool_event_pairs(paths_by_size, regimes, cache_dir, force=force)
    pooled = {"second_order": {regime: {} for regime in events}}
    kappa_det = kappa_detection_threshold()
    for regime, sizes in events.items():
        for size, event in sizes.items():
            drops = EventDrops(er=event["initial_guess_energy"],
                               es=event["second_order"], kappa=event["kappa"])
            split = split_by_kappa(drops, kappa_det)
            pooled["second_order"][regime][size] = positive_es(drops, split.is_irrev)
    return pooled


def log_histogram(data, bins_per_decade=5):
    """Use the same logarithmic histogram as the flowchart PDF panels."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size < 3:
        raise ValueError("Need at least three positive drops for a histogram.")
    centers, density = getHist(data, bins_per_decade=bins_per_decade)
    mask = np.isfinite(density) & (density > 0)
    return centers[mask], density[mask]


def fit_xmins(
    size_drops,
    parallel,
    cache_dir: Path,
    description="xmin",
    refine=True,
):
    fits = {}
    for size, drops in size_drops.items():
        mode = "parallel" if parallel else "serial"
        print(f"Fitting xmin ({mode}): {description}, L={size}", flush=True)
        xmin_search_kwargs = {
            "progress": True,
            "progress_label": f"{description}, L={size}",
            "refine": refine,
        }
        fits[size] = make_fit(
            drops,
            cache_dir=str(cache_dir),
            parallel_xmin=parallel,
            xmin_search_kwargs=xmin_search_kwargs,
        )
        print(f"Finished xmin: {description}, L={size}", flush=True)
    return fits


def histogram_curves(size_drops, bins_per_decade):
    curves = {}
    for size, drops in size_drops.items():
        curves[size] = log_histogram(drops, bins_per_decade=bins_per_decade)
    return curves


def tail_histogram_curves(size_drops, xmins, bins_per_decade, *, normalization="tail"):
    """Show complete bins above xmin with tail or full-population normalization.

    Bin the full population first, as in the flowchart. Discard the bin
    straddling xmin rather than drawing an artificially partly filled bin.
    """
    if normalization not in ("tail", "full"):
        raise ValueError(f"Unknown PDF normalization: {normalization}.")
    curves = {}
    for size, drops in size_drops.items():
        xmin = float(xmins[size])
        tail = np.asarray(drops, dtype=float)
        tail = tail[np.isfinite(tail) & (tail >= xmin)]
        if tail.size < 3:
            raise ValueError(
                f"L={size} has only {tail.size} drops at or above xmin={xmin:.3e}."
            )
        drop, density = log_histogram(drops, bins_per_decade=bins_per_decade)
        lower_edges = drop / 10 ** (0.5 / bins_per_decade)
        complete = lower_edges >= xmin
        if np.count_nonzero(complete) < 2:
            raise ValueError(f"L={size} has too few complete histogram bins above xmin.")
        factor = len(drops) / len(tail) if normalization == "tail" else 1.0
        curves[size] = (drop[complete], density[complete] * factor)
    return curves


def exclude_size(curves, excluded_size):
    """Return a collapse dataset without one explicitly requested system size."""
    if excluded_size not in curves:
        raise ValueError(f"Cannot exclude missing system size L={excluded_size}.")
    filtered = {size: curve for size, curve in curves.items() if size != excluded_size}
    if len(filtered) < 3:
        raise ValueError("A collapse comparison requires at least three system sizes.")
    return filtered


def collapse_variance(curves, exponent, dimension, n_points=80):
    transformed = []
    for size, (drop, density) in curves.items():
        log_size = np.log10(size)
        u = np.log10(drop) - dimension * log_size
        v = np.log10(density) + dimension * exponent * log_size
        transformed.append((u, v))
    low = max(u.min() for u, _ in transformed)
    high = min(u.max() for u, _ in transformed)
    if not high > low:
        return np.inf
    common = np.linspace(low, high, n_points)
    interpolated = np.vstack([np.interp(common, u, v) for u, v in transformed])
    return float(np.mean(np.var(interpolated, axis=0, ddof=1)))


def evaluate_landscape(curves, exponents, dimensions):
    quality = np.empty((len(exponents), len(dimensions)), dtype=float)
    for i, exponent in enumerate(exponents):
        for j, dimension in enumerate(dimensions):
            quality[i, j] = collapse_variance(curves, exponent, dimension)
    return quality


def _curve_signature(curves) -> str:
    digest = hashlib.sha1()
    for size, (drop, density) in sorted(curves.items()):
        digest.update(np.asarray(size, dtype=np.int64).tobytes())
        digest.update(np.asarray(drop, dtype=float).tobytes())
        digest.update(np.asarray(density, dtype=float).tobytes())
    return digest.hexdigest()


def optimize_collapse(curves, cache_path: Path, force=False):
    signature = _curve_signature(curves)
    if cache_path.exists() and not force:
        with np.load(cache_path) as cached:
            if (
                int(cached.get("cache_version", -1)) == COLLAPSE_CACHE_VERSION
                and str(cached.get("curve_signature", "")) == signature
            ):
                return {key: cached[key] for key in cached.files}

    coarse_x = np.linspace(*EXPONENT_RANGE, 61)
    coarse_d = np.linspace(*DIMENSION_RANGE, 61)
    coarse_q = evaluate_landscape(curves, coarse_x, coarse_d)
    coarse_i, coarse_j = np.unravel_index(np.nanargmin(coarse_q), coarse_q.shape)
    dx = coarse_x[1] - coarse_x[0]
    dd = coarse_d[1] - coarse_d[0]
    fine_x = np.linspace(
        max(EXPONENT_RANGE[0], coarse_x[coarse_i] - 2 * dx),
        min(EXPONENT_RANGE[1], coarse_x[coarse_i] + 2 * dx),
        81,
    )
    fine_d = np.linspace(
        max(DIMENSION_RANGE[0], coarse_d[coarse_j] - 2 * dd),
        min(DIMENSION_RANGE[1], coarse_d[coarse_j] + 2 * dd),
        81,
    )
    fine_q = evaluate_landscape(curves, fine_x, fine_d)
    fine_i, fine_j = np.unravel_index(np.nanargmin(fine_q), fine_q.shape)
    boundary = fine_i in {0, len(fine_x) - 1} or fine_j in {
        0,
        len(fine_d) - 1,
    }
    result = {
        "cache_version": np.asarray(COLLAPSE_CACHE_VERSION),
        "curve_signature": np.asarray(signature),
        "coarse_x": coarse_x,
        "coarse_d": coarse_d,
        "coarse_q": coarse_q,
        "fine_x": fine_x,
        "fine_d": fine_d,
        "fine_q": fine_q,
        "x": np.asarray(float(fine_x[fine_i])),
        "dimension": np.asarray(float(fine_d[fine_j])),
        "quality": np.asarray(float(fine_q[fine_i, fine_j])),
        "boundary": np.asarray(boundary),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    return result


def plot_raw_and_xmin(
    raw_curves,
    fits,
    protocol,
    regime,
    path: Path,
    applied_xmins=None,
    xmin_note=None,
    *,
    save_png=True,
):
    pdf_fig, ax_pdf = plt.subplots(figsize=(4.4, 3.36))
    xmin_fig, ax_xmin = plt.subplots(figsize=(4.4, 3.36))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for curve_index, (size, (drop, density)) in enumerate(raw_curves.items()):
        analysis = getattr(fits[size], "xmin_analysis", None)
        if analysis is None:
            raise RuntimeError(f"Missing xmin analysis for L={size}.")
        global_xmin = float(analysis["global_min_xmin"])
        global_distance = float(analysis["global_min_distance"])
        color = colors[curve_index % len(colors)]
        ax_pdf.axvline(
            global_xmin if applied_xmins is None else applied_xmins[size],
            color=color,
            linestyle=":",
            linewidth=1.0,
            alpha=0.85,
            zorder=0.1,
        )
        ax_pdf.plot(
            drop,
            density,
            marker="o",
            ms=3,
            linestyle="none",
            label=f"L={size}",
            color=color,
            zorder=3,
        )
        results = analysis["global_search_details"]
        xmins = np.asarray(results.get("xmins", []), dtype=float)
        distances = np.asarray(results.get("distances", []), dtype=float)
        mask = (np.asarray(results["valid_fits"], dtype=bool)
                & np.isfinite(xmins) & np.isfinite(distances) & (xmins > 0))
        if mask.any():
            order = np.argsort(xmins[mask])
            valid_xmins = xmins[mask][order]
            valid_distances = distances[mask][order]
            ax_xmin.plot(
                valid_xmins,
                valid_distances,
                color=color,
                label=f"L={size}",
                zorder=3,
            )
            ax_xmin.scatter(
                [global_xmin],
                [global_distance],
                marker="x",
                color=color,
                s=55,
                linewidths=1.8,
                zorder=20,
            )
    drop_symbol = PROTOCOL_DROP_SYMBOLS[protocol]
    ax_pdf.set(
        xscale="log",
        yscale="log",
        xlabel=rf"${drop_symbol}$",
        ylabel=rf"$p({drop_symbol})$",
    )
    legend_extras = [
        Line2D(
            [],
            [],
            color="black",
            marker="x",
            linestyle="none",
            markersize=7,
            markeredgewidth=1.8,
            label="global min",
        ),
    ]
    for axis in (ax_pdf, ax_xmin):
        handles, labels = axis.get_legend_handles_labels()
        if axis is ax_pdf:
            cutoff_label = "KS cutoff" if applied_xmins is None else "Common cutoff"
            extra = Line2D([], [], color="black", linestyle=":", label=cutoff_label)
            axis.legend(handles + [extra], labels + [cutoff_label])
        else:
            axis.legend(handles + legend_extras, labels + ["global min"])
    ax_xmin.set(xscale="log", xlabel=r"$\Delta E_{\min}$", ylabel=r"$D$")
    for fig, axis, suffix in ((pdf_fig, ax_pdf, "pdf"), (xmin_fig, ax_xmin, "xmin_scan")):
        fig.tight_layout()
        target = path.with_name(f"{path.stem}_{suffix}.pdf")
        fig.savefig(target, bbox_inches="tight")
        if save_png:
            fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_collapse(curves, result, protocol, regime, path: Path, xmin_note=None,
                  *, save_quality=True, save_png=True,
                  parameter_symbols=("x", "D_x"), drop_symbol="s", exponent_format=".3f"):
    exponent = float(result["x"])
    dimension = float(result["dimension"])
    collapse_fig, ax_collapse = plt.subplots(figsize=(4.4, 3.36))
    exponent_symbol, dimension_symbol = parameter_symbols
    parameters = rf"${exponent_symbol}={exponent:{exponent_format}},\ {dimension_symbol}={dimension:.3f}$"
    figures = [(collapse_fig, path)]
    if save_quality:
        quality_fig, ax_map = plt.subplots(figsize=(4.4, 3.36))
        image = ax_map.pcolormesh(
            result["fine_d"], result["fine_x"], np.log10(result["fine_q"]),
            shading="auto",
        )
        ax_map.scatter([dimension], [exponent], marker="*", s=90, color="red")
        ax_map.set(xlabel="$D_x$", ylabel="$x$")
        quality_fig.colorbar(image, ax=ax_map, label="$\\log_{10} Q$")
        ax_map.legend(handles=[Line2D([], [], marker="*", color="red", linestyle="none",
                                      markersize=10, label=parameters)])
        figures.append((quality_fig, path.with_name(f"{path.stem}_quality.pdf")))
    for size, (drop, density) in curves.items():
        scaled_x = drop / size**dimension
        scaled_y = density * size ** (dimension * exponent)
        ax_collapse.plot(scaled_x, scaled_y, marker="o", ms=3, linestyle="none", label=f"L={size}")
    ax_collapse.set(
        xscale="log",
        yscale="log",
        xlabel=rf"${drop_symbol}/L^{{{dimension_symbol}}}$",
        ylabel=rf"$P({drop_symbol};L)L^{{{dimension_symbol}\,{exponent_symbol}}}$",
    )
    ax_collapse.legend(title=parameters)
    for fig, target in figures:
        fig.tight_layout()
        fig.savefig(target, bbox_inches="tight")
        if save_png:
            fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_lambda_scaling(fits, path: Path):
    """Fit lambda=A L^(-D_lambda); take alpha directly from the L=250 fit."""
    if 250 not in fits or len(fits) < 3:
        raise ValueError("Lambda scaling requires L=250 and at least three sizes.")
    sizes = np.asarray(sorted(fits), dtype=float)
    distributions = {size: dist_from_fit(fit) for size, fit in fits.items()}
    rates = np.asarray([distributions[int(size)].Lambda for size in sizes], dtype=float)
    alpha = float(distributions[250].alpha)
    if not np.isfinite(alpha) or not np.all(np.isfinite(rates) & (rates > 0)):
        raise ValueError("Invalid fitted alpha or lambda for size scaling.")
    slope, intercept = np.polyfit(np.log(sizes), np.log(rates), 1)
    dimension = float(-slope)
    prefactor = float(np.exp(intercept))
    grid = np.geomspace(sizes.min(), sizes.max(), 200)
    fig, ax = plt.subplots(figsize=(4.4, 3.36))
    ax.plot(sizes, rates, "o", label="Per-size KS fits")
    ax.plot(grid, prefactor * grid**(-dimension),
            label=rf"$\lambda\propto L^{{-{dimension:.3f}}}$")
    ax.set(xscale="log", yscale="log", xlabel=r"System size $L$", ylabel=r"$\lambda$")
    ax.set_xticks(sizes, [str(int(size)) for size in sizes])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize="small")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"x": alpha, "dimension": dimension, "lambda_prefactor": prefactor,
            "alpha_source_size": 250, "collapse_optimized": False,
            "lambda_fit": "unweighted least squares: log(lambda) = log(A) - D_lambda log(L)",
            "parameter_fit_cutoffs": {size: float(fit.xmin) for size, fit in fits.items()},
            "lambda_per_size": dict(zip(sizes.astype(int).tolist(), rates.tolist()))}


def fixed_xmin_parameter_fits(
    size_drops,
    xmin,
    *,
    fit_cache_dir: Path,
    evaluation_cache_dir: Path,
    uncertainty_accuracy,
    parallel,
    description,
    selected_fits=None,
    skip_pvalue=False,
):
    records = {}
    for size, drops in size_drops.items():
        size_xmin = (
            float(xmin[size])
            if isinstance(xmin, dict)
            else float(xmin)
        )
        print(
            f"Fitting parameters: {description}, L={size}, xmin={size_xmin:.6g}",
            flush=True,
        )
        if selected_fits is None:
            fit = make_fit(
                drops, xmin_range=size_xmin, cache_dir=str(fit_cache_dir),
                xmin_selection="simpleDrop",
            )
        else:
            fit = selected_fits[size]
            if float(fit.xmin) != size_xmin:
                raise ValueError(f"Selected-fit xmin mismatch for {description}, L={size}.")
        if not skip_pvalue:
            fit.evaluate_fit(
                data=drops,
                confidence=uncertainty_accuracy,
                parallel=parallel,
                cache_dir=str(evaluation_cache_dir),
                tqdmDesc=f"{description}, L={size}",
            )
        print(f"Finished parameters: {description}, L={size}", flush=True)
        distribution = dist_from_fit(fit)
        alpha = float(distribution.alpha)
        cutoff = float(getattr(distribution, "Lambda", np.nan))
        alpha_std = None if skip_pvalue else float(getattr(fit, "alpha_std", np.nan))
        cutoff_std = None if skip_pvalue else float(getattr(fit, "Lambda_std", np.nan))
        if not np.isfinite(alpha) or not np.isfinite(cutoff) or cutoff <= 0:
            raise RuntimeError(f"Invalid parameters for {description}, L={size}.")
        if not skip_pvalue and not (np.isfinite(alpha_std) and np.isfinite(cutoff_std)):
            raise RuntimeError(f"Invalid parameter uncertainties for {description}, L={size}.")
        records[size] = {
            "alpha": alpha,
            "alpha_std": alpha_std,
            "Lambda": cutoff,
            "Lambda_std": cutoff_std,
            "xmin": size_xmin,
            "tail_count": int(np.count_nonzero(np.asarray(drops) >= size_xmin)),
            "p": None if skip_pvalue else float(fit.p),
            "alpha_bootstrap_mean": None if skip_pvalue else float(fit.alpha_mean),
        }
        if selected_fits is not None and not skip_pvalue:
            records[size].update(
                xmin_bootstrap_mean=float(fit.bootstrap_xmin_mean),
                xmin_bootstrap_std=float(fit.bootstrap_xmin_std),
            )
    return records


def plot_parameter_vs_size(parameter_results, xmins, parameter, path: Path, skip_pvalue=False):
    del xmins
    if parameter not in {"alpha", "Lambda"}:
        raise ValueError("parameter must be 'alpha' or 'Lambda'.")
    markers = ("o", "s", "^", "D")
    protocols = tuple(parameter_results)
    regimes = tuple(parameter_results[protocols[0]])
    for regime in regimes:
        fig, ax = plt.subplots(figsize=(4.4, 3.52))
        for marker, protocol in zip(markers, protocols):
            records = parameter_results[protocol][regime]
            sizes = np.asarray(sorted(records), dtype=float)
            values = np.asarray(
                [records[int(size)][parameter] for size in sizes], dtype=float
            )
            yerr = None
            if not skip_pvalue:
                errors = np.asarray(
                    [records[int(size)][f"{parameter}_std"] for size in sizes],
                    dtype=float,
                )
                yerr = errors
                if parameter == "Lambda":
                    yerr = np.vstack((np.minimum(errors, 0.99 * values), errors))
            ax.errorbar(
                sizes,
                values,
                yerr=yerr,
                marker=marker,
                capsize=3,
                label=PROTOCOL_LABELS[protocol],
            )
        ax.set_xlabel("System size $L$")
        ax.set_xticks(sizes)
        ax.grid(alpha=0.2)
        if parameter == "Lambda":
            ax.set_yscale("log")
            ax.set_ylabel(r"Cutoff rate $\lambda$")
        else:
            ax.set_ylabel(r"Exponent $\alpha$")
        ax.legend(fontsize="small")
        target = path if len(regimes) == 1 else path.with_name(f"{path.stem}_{regime}.pdf")
        if skip_pvalue:
            fig.text(0.5, 0.01, "Bootstrap omitted: no p-value or uncertainties",
                     ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04 if skip_pvalue else 0, 1, 1))
        fig.savefig(target, bbox_inches="tight")
        fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_xmin_vs_size(xmins, method: str, path: Path, skip_pvalue=False):
    if method not in xmins:
        raise ValueError(f"Unknown xmin method {method!r}.")
    markers = ("o", "s", "^")
    colors = ("#0072B2", "#E69F00", "#009E73")
    protocols = tuple(xmins[method])
    regimes = tuple(xmins[method][protocols[0]])
    for regime in regimes:
        fig, ax = plt.subplots(figsize=(4.4, 3.52))
        for marker, color, protocol in zip(markers, colors, protocols):
            values = xmins[method][protocol][regime]
            sizes = np.asarray(sorted(values), dtype=float)
            y = np.asarray([values[int(size)] for size in sizes], dtype=float)
            ax.plot(
                sizes,
                y,
                marker=marker,
                color=color,
                linewidth=1.1,
                label=PROTOCOL_LABELS[protocol],
            )
        ax.set_xlabel("System size $L$")
        ax.set_xticks(sizes)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.set_ylabel(r"$\Delta E_{\min}$")
        ax.legend(fontsize="small")
        target = path if len(regimes) == 1 else path.with_name(f"{path.stem}_{regime}.pdf")
        if skip_pvalue:
            fig.text(0.5, 0.01, "Bootstrap omitted: no p-value or uncertainties",
                     ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04 if skip_pvalue else 0, 1, 1))
        fig.savefig(target, bbox_inches="tight")
        fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


def run(args):
    if args.common_cutoff_size is None:
        args.common_cutoff_size = 100 if args.stage == "lambda-scaling" else 250
    if args.stage == "lambda-scaling" and (not args.skip_pvalue or args.yield_regime != "post"):
        raise ValueError("Lambda scaling requires --skip-pvalue and --yield-regime post.")
    protocols = tuple(args.protocols)
    if args.stage == "lambda-scaling" and (protocols != ("second_order",) or args.population_mode != "classified"):
        raise ValueError("Lambda scaling uses the standard positive irreversible Delta E_S population.")
    if args.population_mode == "classified" and protocols != ("second_order",):
        raise ValueError("The standard classified population fits only second_order. "
                         "Select --population-mode all explicitly for other protocols.")
    if args.standard_cache_root and args.population_mode != "classified":
        raise ValueError("--standard-cache-root requires the classified population.")
    if len(set(protocols)) != len(protocols):
        raise ValueError("--protocols must not contain duplicates.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.seeds_per_size is not None and args.seeds_per_size < 1:
        raise ValueError("--seeds-per-size must be positive.")
    if args.bins_per_decade < 1:
        raise ValueError("--bins-per-decade must be positive.")
    if not 0 < args.uncertainty_accuracy < 0.5:
        raise ValueError("--uncertainty-accuracy must be between 0 and 0.5.")
    regimes = None if args.yield_regime == "both" else (args.yield_regime,)
    paths, inventory = size_scaling_paths(Path(args.data_root), args.seeds_per_size,
                                        minimum_size=args.minimum_size)
    if args.largest_first:
        paths = dict(sorted(paths.items(), reverse=True))
    standard_cache = Path(args.standard_cache_root) if args.standard_cache_root else None
    if args.population_mode == "classified":
        pair_cache = (standard_cache or output / "cache") / "event_pairs"
        pooled = pool_irreversible_es(paths, regimes, pair_cache, args.force)
    else:
        pooled = pool_drops(paths, regimes, output / "cache" / "extracted", args.force)
    if regimes is None:
        regimes = YIELD_REGIME_NAMES
    if args.stage == "extract":
        return

    analysis_dir = output if args.stage == "lambda-scaling" else output / f"xmin_{XMIN_METHOD}"
    if args.population_mode == "all":
        analysis_dir /= "all_events"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    xmin_cache = ((standard_cache / "global_min") if standard_cache else
                  output / "cache" / "xmin" / args.population_mode)

    do_local = args.stage in {"collapse", "all"}
    do_shared = args.stage in {"shared", "parameters", "all"}
    do_parameters = args.stage in {"parameters", "all"}
    all_fits = {protocol: {} for protocol in protocols}
    all_raw_curves = {protocol: {} for protocol in protocols}
    all_curves = {protocol: {} for protocol in protocols}
    all_results = {protocol: {} for protocol in protocols}
    summary = {
        "inventory": inventory,
        "seeds_per_size": args.seeds_per_size,
        "minimum_size": args.minimum_size,
        "source_csvs": {size: [str(path) for path in files] for size, files in paths.items()},
        "sample_counts": {size: len(files) for size, files in paths.items()},
        "sample_selection": "available runs extending past their avg_P12 peak +0.01; partial runs included",
        "histogram_method": "Plotting.plotPowerLaw.getHist",
        "bins_per_decade": args.bins_per_decade,
        "collapse_pdf": "conditional tail density; only bins wholly above the KS cutoff",
        "yield_regimes": list(regimes),
        "fit_size_order": list(paths),
        "protocols": protocols,
        "population_mode": args.population_mode,
        "fit_population": ("positive irreversible Delta E_S" if args.population_mode == "classified"
                           else "all positive drops; alternative analysis"),
        "post_yield_rule": POST_YIELD_RULE,
        "delta_E_S_tangent": "gamma0",
        "kappa_det": kappa_detection_threshold() if args.population_mode == "classified" else None,
        "results": {},
    }
    for protocol in protocols:
        summary["results"][protocol] = {}
        for regime in regimes:
            print(
                f"Fitting xmin: {protocol}, {regime}-yield, "
                f"strategy={XMIN_METHOD}",
                flush=True,
            )
            size_drops = pooled[protocol][regime]
            fits = fit_xmins(
                size_drops,
                parallel=args.parallel_xmin,
                cache_dir=xmin_cache / protocol / regime,
                description=f"{protocol}, {regime}-yield",
            )
            all_fits[protocol][regime] = fits
            raw_curves = histogram_curves(size_drops, args.bins_per_decade)
            all_raw_curves[protocol][regime] = raw_curves
            if args.stage == "lambda-scaling":
                result = plot_lambda_scaling(fits, analysis_dir / "lambda_scaling.pdf")
                if args.lambda_cutoff == "common":
                    if args.common_cutoff_size not in fits:
                        raise ValueError(f"Missing common cutoff source L={args.common_cutoff_size}.")
                    cutoff = float(fits[args.common_cutoff_size].xmin)
                    xmins = {size: cutoff for size in fits}
                else:
                    xmins = {size: float(fit.xmin) for size, fit in fits.items()}
                curves = tail_histogram_curves(
                    size_drops, xmins, args.bins_per_decade,
                    normalization=args.lambda_normalization,
                )
                plot_collapse(
                    curves, result, protocol, regime,
                    analysis_dir / f"{protocol}_{regime}_collapse_fixed_alpha.pdf",
                    save_quality=False, save_png=False,
                    parameter_symbols=(r"\alpha_{250}", r"D_\lambda"),
                    exponent_format=".2f",
                    drop_symbol=PROTOCOL_DROP_SYMBOLS[protocol],
                )
                plot_raw_and_xmin(raw_curves, fits, protocol, regime,
                                  analysis_dir / f"{protocol}_{regime}.pdf",
                                  save_png=False)
                summary["results"][protocol][regime] = {
                    **result, "collapse_cutoff_rule": args.lambda_cutoff,
                    "collapse_normalization": args.lambda_normalization,
                    "cutoff_source_size": args.common_cutoff_size if args.lambda_cutoff == "common" else None,
                    "applied_cutoffs": xmins,
                    "drop_count": {size: len(values) for size, values in size_drops.items()},
                    "tail_count": {size: int(np.count_nonzero(values >= xmins[size]))
                                   for size, values in size_drops.items()},
                }
                continue
            plot_raw_and_xmin(
                raw_curves,
                fits,
                protocol,
                regime,
                analysis_dir / f"{protocol}_{regime}_raw_xmin.pdf",
                xmin_note=f"Per-size {XMIN_METHOD} xmin (vertical lines)",
            )
            if not do_local:
                continue
            xmins = {size: float(fit.xmin) for size, fit in fits.items()}
            curves = tail_histogram_curves(
                size_drops, xmins, args.bins_per_decade
            )
            all_curves[protocol][regime] = curves
            result = optimize_collapse(
                curves,
                analysis_dir / "cache" / "collapse" / f"{protocol}_{regime}.npz",
                force=args.force,
            )
            all_results[protocol][regime] = result
            print(
                f"Collapse: x={float(result['x']):.3f}, "
                f"D={float(result['dimension']):.3f}, "
                f"Q={float(result['quality']):.3g}",
                flush=True,
            )
            plot_collapse(
                curves,
                result,
                protocol,
                regime,
                analysis_dir / f"{protocol}_{regime}_collapse.pdf",
                xmin_note=f"Per-size {XMIN_METHOD} xmin; tail PDFs renormalized",
            )
            summary["results"][protocol][regime] = {
                "x": float(result["x"]),
                "dimension": float(result["dimension"]),
                "quality": float(result["quality"]),
                "optimum_on_search_boundary": bool(result["boundary"]),
                "xmin": {str(size): xmin for size, xmin in xmins.items()},
                "drop_count": {str(size): int(len(values)) for size, values in size_drops.items()},
                "tail_count": {
                    str(size): int(np.count_nonzero(values >= xmins[size]))
                    for size, values in size_drops.items()
                },
            }
    if args.stage == "lambda-scaling":
        summary["bootstrap_performed"] = False
        summary["raw_pdf"] = "full positive irreversible E_S population; per-size KS cutoff markers"
        summary["collapse_pdf"] = (
            "full-population density; masking does not renormalize the tail"
            if args.lambda_normalization == "full" else "conditional tail density"
        ) + "; only complete bins above the applied cutoff"
        (analysis_dir / "lambda_scaling_results.json").write_text(json.dumps(summary, indent=2))
        return
    if args.stage == "xmin":
        return

    if do_local:
        (analysis_dir / "collapse_results.json").write_text(
            json.dumps(summary, indent=2)
        )

    if not (do_shared or do_parameters):
        return

    if do_parameters:
        parameter_dir = analysis_dir / "parameters_per_size"
        if args.skip_pvalue:
            parameter_dir /= "without_uncertainties"
        parameter_dir.mkdir(parents=True, exist_ok=True)
        parameter_xmins = {
            method: {protocol: {} for protocol in protocols}
            for method in args.parameter_xmin_methods
        }
        parameter_results = {
            method: {protocol: {} for protocol in protocols}
            for method in parameter_xmins
        }
        for method in parameter_xmins:
            for protocol in protocols:
                for regime in regimes:
                    size_drops = pooled[protocol][regime]
                    parameter_xmins[method][protocol][regime] = {
                        size: float(
                            all_fits[protocol][regime][size].xmin_analysis[
                                "simple_drop_xmin"
                                if method == "simple_drop"
                                else "global_min_xmin"
                            ]
                        )
                        for size in size_drops
                    }
                    parameter_results[method][protocol][regime] = (
                        fixed_xmin_parameter_fits(
                            size_drops,
                            parameter_xmins[method][protocol][regime],
                            fit_cache_dir=(
                                parameter_dir
                                / "cache"
                                / "fixed_xmin"
                                / method
                                / protocol
                                / regime
                            ),
                            evaluation_cache_dir=(standard_cache / "evaluation" / protocol / regime
                                                  if standard_cache and method == "global_min" else
                                parameter_dir
                                / "cache"
                                / "evaluation"
                                / method
                                / protocol
                                / regime
                            ),
                            uncertainty_accuracy=args.uncertainty_accuracy,
                            parallel=args.parallel_uncertainty,
                            description=f"{method}, {protocol}, {regime}-yield",
                            selected_fits=(all_fits[protocol][regime]
                                           if method == "global_min" else None),
                            skip_pvalue=args.skip_pvalue,
                        )
                    )
        for method in parameter_results:
            plot_parameter_vs_size(
                parameter_results[method],
                parameter_xmins[method],
                "alpha",
                parameter_dir / f"alpha_vs_size_{method}.pdf",
                skip_pvalue=args.skip_pvalue,
            )
            plot_parameter_vs_size(
                parameter_results[method],
                parameter_xmins[method],
                "Lambda",
                parameter_dir / f"lambda_vs_size_{method}.pdf",
                skip_pvalue=args.skip_pvalue,
            )
            plot_xmin_vs_size(
                parameter_xmins,
                method,
                parameter_dir / f"xmin_vs_size_{method}.pdf",
                skip_pvalue=args.skip_pvalue,
            )
        (parameter_dir / "parameter_results.json").write_text(
            json.dumps(
                {
                    "inventory": inventory,
                    "seeds_per_size": args.seeds_per_size,
                    "yield_regimes": list(regimes),
                    "fit_size_order": list(paths),
                    "sizes": sorted(paths),
                    "protocols": protocols,
                    "bootstrap_performed": not args.skip_pvalue,
                    "population_mode": args.population_mode,
                    "fit_population": summary["fit_population"],
                    "post_yield_rule": POST_YIELD_RULE,
                    "delta_E_S_tangent": "gamma0",
                    "kappa_det": summary["kappa_det"],
                    "xmins": parameter_xmins,
                    "records": parameter_results,
                },
                indent=2,
            )
        )

    if args.stage == "parameters":
        return

    source_size = args.common_cutoff_size
    shared_suffix = "shared_xmin" if source_size == 250 else "common_cutoff"
    shared_dir = analysis_dir / "shared_L250_xmin" if source_size == 250 else analysis_dir
    shared_dir.mkdir(parents=True, exist_ok=True)
    shared_xmins = {protocol: {} for protocol in protocols}
    shared_curves = {protocol: {} for protocol in protocols}
    shared_results = {protocol: {} for protocol in protocols}
    parameter_results = {protocol: {} for protocol in protocols}
    shared_summary = {
        "inventory": inventory,
        "seeds_per_size": args.seeds_per_size,
        "minimum_size": args.minimum_size,
        "yield_regimes": list(regimes),
        "fit_size_order": list(paths),
        "xmin_source_size": source_size,
        "analysis_kind": "common-cutoff collapse sensitivity; per-size KS fits unchanged",
        "histogram_method": summary["histogram_method"],
        "bins_per_decade": args.bins_per_decade,
        "collapse_pdf": "conditional tail density; only bins wholly above the common cutoff",
        "xmin_strategy": XMIN_METHOD,
        "uncertainty_accuracy": args.uncertainty_accuracy,
        "bootstrap_performed": not args.skip_pvalue,
        "bootstrap_sets": 0 if args.skip_pvalue else max(1, int(1 / (4 * args.uncertainty_accuracy**2))),
        "post_yield_rule": POST_YIELD_RULE,
        "delta_E_S_tangent": "gamma0",
        "fit_population": summary["fit_population"],
        "source_csvs": summary["source_csvs"],
        "sample_counts": summary["sample_counts"],
        "results": {},
    }
    for protocol in protocols:
        shared_summary["results"][protocol] = {}
        for regime in regimes:
            size_drops = pooled[protocol][regime]
            if source_size not in size_drops:
                raise RuntimeError(f"Common-cutoff analysis requires L={source_size} data.")
            shared_xmin = float(all_fits[protocol][regime][source_size].xmin)
            xmins = {size: shared_xmin for size in size_drops}
            shared_xmins[protocol][regime] = shared_xmin
            note = (
                f"Shared xmin={shared_xmin:.3e} from L={source_size} "
                f"({XMIN_METHOD})"
            )
            regime_summary = {
                "shared_xmin": shared_xmin,
                "tail_count": {
                    str(size): int(np.count_nonzero(values >= shared_xmin))
                    for size, values in size_drops.items()
                },
            }

            if do_shared:
                plot_raw_and_xmin(
                    all_raw_curves[protocol][regime],
                    all_fits[protocol][regime],
                    protocol,
                    regime,
                    shared_dir / f"{protocol}_{regime}_raw_{shared_suffix}.pdf",
                    applied_xmins=xmins,
                    xmin_note=note,
                )
                curves = tail_histogram_curves(
                    size_drops, xmins, args.bins_per_decade
                )
                shared_curves[protocol][regime] = curves
                result = optimize_collapse(
                    curves,
                    shared_dir / "cache" / "collapse" / f"{protocol}_{regime}_{shared_suffix}.npz",
                    force=args.force,
                )
                shared_results[protocol][regime] = result
                plot_collapse(
                    curves,
                    result,
                    protocol,
                    regime,
                    shared_dir / f"{protocol}_{regime}_collapse_{shared_suffix}.pdf",
                    xmin_note=note,
                )
                regime_summary["collapse"] = {
                    "x": float(result["x"]),
                    "dimension": float(result["dimension"]),
                    "quality": float(result["quality"]),
                    "optimum_on_search_boundary": bool(result["boundary"]),
                }
            shared_summary["results"][protocol][regime] = regime_summary

    (shared_dir / f"{shared_suffix}_results.json").write_text(
        json.dumps(shared_summary, indent=2)
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/Volumes/data/remoteData/macro",
    )
    parser.add_argument(
        "--output",
        default="Plots/powerLaw/size_collapse_piola_peak_post_only",
    )
    parser.add_argument(
        "--stage",
        choices=("extract", "xmin", "collapse", "shared", "parameters", "lambda-scaling", "all"),
        default="all",
    )
    parser.add_argument("--seeds-per-size", type=int,
                        help="Optional sample cap; default uses all available post-yield runs.")
    parser.add_argument("--minimum-size", type=int, default=SIZE_SCALING_MIN_SIZE,
                        help="Minimum system size (default: 100; use 50 for an explicitly requested comparison).")
    parser.add_argument("--parallel-xmin", action="store_true")
    parser.add_argument("--uncertainty-accuracy", type=float, default=0.05)
    parser.add_argument("--parallel-uncertainty", action="store_true")
    parser.add_argument("--bins-per-decade", type=int, default=5)
    parser.add_argument("--common-cutoff-size", type=int,
                        help="Common-cutoff reference size (default: 100 for lambda scaling, 250 for other stages).")
    parser.add_argument("--lambda-cutoff", choices=("common", "per-size"), default="common",
                        help="Cutoff rule for the lambda-scaling stage.")
    parser.add_argument("--lambda-normalization", choices=("tail", "full"), default="tail",
                        help="Normalize to the retained tail or full population before masking bins.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--yield-regime", choices=("both", "pre", "post"), default="post")
    parser.add_argument("--largest-first", action="store_true")
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=("second_order",),
                        help="Energy-drop protocols to fit (default: second_order).")
    parser.add_argument("--population-mode", choices=("classified", "all"), default="classified",
                        help="Default: kappa-classified irreversible E_S; 'all' is an alternative analysis.")
    parser.add_argument("--standard-cache-root", type=Path,
                        help="Reuse event pairs, KS searches and bootstraps from sizeScalingERClassification/cache.")
    parser.add_argument("--parameter-xmin-methods", nargs="+",
                        choices=("global_min", "simple_drop"),
                        default=("global_min",))
    parser.add_argument("--skip-pvalue", action="store_true",
                        help="Omit parameter bootstrap; save plots under without_uncertainties.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
