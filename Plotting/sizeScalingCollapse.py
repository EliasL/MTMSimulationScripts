"""Finite-size collapse of size-scaling energy-drop protocols.

Implements the logarithmic overlap-variance procedure in Salman et al. (2025),
Eq. (11)-(12). Run from the repository root with

    .venv/bin/python -m Plotting.sizeScalingCollapse

Each stage is cached below ``Plots/powerLaw/size_collapse/cache``.
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

from Management.jobs import size_scaling_job
from Management.updateCSV import HEADER_RENAME_MAP, update_df_header
from Plotting.energyDropCalculations import calculate_energy_step_data
from Plotting.plotPowerLaw import dist_from_fit, make_fit


PROTOCOLS = (
    "second_order",
    "first_order",
    "previous_energy",
    "initial_guess_energy",
)
PROTOCOL_LABELS = {
    "second_order": "Second-order elastic continuation",
    "first_order": "First-order elastic continuation",
    "previous_energy": "Previous relaxed energy",
    "initial_guess_energy": "Affine initial-guess energy",
}
REGIMES = {"pre": (0.15, 0.5), "post": (0.7, 1.0)}
CACHE_VERSION = 2
COLLAPSE_CACHE_VERSION = 3
EXPONENT_RANGE = (0.5, 2.5)
DIMENSION_RANGE = (0.25, 2.25)


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


def completed_size_scaling_paths(data_root: Path, seeds_per_size: int, post_hi: float):
    groups, _ = size_scaling_job()
    paths = {}
    inventory = {}
    for group in groups:
        size = int(group[0].rows)
        completed = []
        for config in sorted(group, key=lambda item: item.seed):
            path = data_root / f"{config.name}.csv"
            if path.exists() and _last_load(path) >= post_hi:
                completed.append(path)
        inventory[size] = len(completed)
        if len(completed) < seeds_per_size:
            raise RuntimeError(
                f"L={size} has {len(completed)} completed runs; "
                f"need {seeds_per_size}."
            )
        paths[size] = completed[:seeds_per_size]
    return dict(sorted(paths.items())), inventory


def _file_cache_path(path: Path, size: int, regimes, cache_dir: Path) -> Path:
    stat = path.stat()
    signature = repr(
        (CACHE_VERSION, str(path), stat.st_size, stat.st_mtime_ns, size, regimes)
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


def extract_run(path: Path, size: int, regimes, cache_dir: Path, force=False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _file_cache_path(path, size, regimes, cache_dir)
    if cache_path.exists() and not force:
        with np.load(cache_path) as cached:
            return {key: cached[key] for key in cached.files}

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
    stress_columns = [
        column
        for column in ("avg_sigma12", "avg_P12")
        if column in df and np.isfinite(df[column]).all()
    ]
    if not stress_columns:
        raise ValueError(f"No complete stress column in {path}.")
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
        "first_order": np.asarray(
            steps["stress_corrected_drop_first_order"], dtype=float
        ),
        "previous_energy": -np.asarray(df["total_energy_change"], dtype=float)[1:],
        "initial_guess_energy": -np.asarray(
            df["total_e_change_from_init"], dtype=float
        )[1:],
    }

    extracted = {}
    for protocol, drops in values.items():
        if drops.shape != step_load.shape:
            raise ValueError(
                f"Shape mismatch for {protocol} in {path}: "
                f"{drops.shape} vs {step_load.shape}."
            )
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


def pool_drops(paths_by_size, regimes, cache_dir: Path, force=False):
    pooled = {
        protocol: {regime: {} for regime in regimes} for protocol in PROTOCOLS
    }
    for size, paths in paths_by_size.items():
        print(f"Extracting/caching L={size}: {len(paths)} completed runs", flush=True)
        per_run = [
            extract_run(path, size, regimes, cache_dir / "runs", force=force)
            for path in paths
        ]
        for protocol in PROTOCOLS:
            for regime in regimes:
                arrays = [run[f"{protocol}_{regime}"] for run in per_run]
                pooled[protocol][regime][size] = np.concatenate(arrays)
    return pooled


def log_histogram(data, bins_per_decade=10):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size < 3:
        raise ValueError("Need at least three positive drops for a histogram.")
    low, high = np.log10(data.min()), np.log10(data.max())
    n_bins = max(10, int(np.ceil((high - low) * bins_per_decade)))
    edges = np.logspace(low, high, n_bins + 1)
    density, edges = np.histogram(data, bins=edges, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = np.isfinite(density) & (density > 0)
    return centers[mask], density[mask]


def fit_xmins(size_drops, strategy, accuracy, parallel, cache_dir: Path):
    fits = {}
    for size, drops in size_drops.items():
        if strategy in {"dip", "ks"}:
            fit = make_fit(
                drops,
                cache_dir=str(cache_dir),
                fast_xmin=strategy == "dip",
                xmin_accuracy=accuracy,
                parallel_xmin=parallel,
            )
        else:
            strategy_kwargs = {}
            if accuracy is not None and strategy in {"plateau", "dks"}:
                strategy_kwargs["samples_per_decade"] = 1.0 / accuracy
            elif accuracy is not None and strategy == "sizer":
                strategy_kwargs["samplesPerDecade"] = 1.0 / accuracy
            fit = make_fit(
                drops,
                cache_dir=str(cache_dir),
                xmin_strategy=strategy,
                xmin_accuracy=accuracy,
                parallel_xmin=parallel,
                xmin_strategy_kwargs=strategy_kwargs,
            )
        fits[size] = fit
    return fits


def histogram_curves(size_drops, bins_per_decade):
    curves = {}
    for size, drops in size_drops.items():
        curves[size] = log_histogram(drops, bins_per_decade=bins_per_decade)
    return curves


def tail_histogram_curves(size_drops, xmins, bins_per_decade):
    """Build normalized PDFs using only observations at or above xmin."""
    curves = {}
    for size, drops in size_drops.items():
        xmin = float(xmins[size])
        tail = np.asarray(drops, dtype=float)
        tail = tail[np.isfinite(tail) & (tail >= xmin)]
        if tail.size < 3:
            raise ValueError(
                f"L={size} has only {tail.size} drops at or above xmin={xmin:.3e}."
            )
        curves[size] = log_histogram(tail, bins_per_decade=bins_per_decade)
    return curves


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
):
    applied_xmins = applied_xmins or {
        size: float(fit.xmin) for size, fit in fits.items()
    }
    fig, (ax_pdf, ax_xmin) = plt.subplots(1, 2, figsize=(11, 4.2))
    for size, (drop, density) in raw_curves.items():
        ax_pdf.plot(drop, density, marker="o", ms=3, linestyle="none", label=f"L={size}")
        ax_pdf.axvline(applied_xmins[size], alpha=0.25)
        results = getattr(fits[size], "xmin_fitting_results", None) or {}
        xmins = np.asarray(results.get("xmins", []), dtype=float)
        distances = np.asarray(results.get("distances", []), dtype=float)
        mask = np.isfinite(xmins) & np.isfinite(distances) & (xmins > 0)
        if mask.any():
            order = np.argsort(xmins[mask])
            valid_xmins = xmins[mask][order]
            valid_distances = distances[mask][order]
            ax_xmin.plot(valid_xmins, valid_distances, label=f"L={size}")
            ax_xmin.scatter(
                [applied_xmins[size]],
                [np.interp(applied_xmins[size], valid_xmins, valid_distances)],
                s=20,
            )
    ax_pdf.set(xscale="log", yscale="log", xlabel="Energy drop $s$", ylabel="$P(s;L)$")
    ax_pdf.legend()
    ax_xmin.set(xscale="log", xlabel="$x_{min}$ candidate", ylabel="KS distance")
    ax_xmin.legend()
    title = f"{PROTOCOL_LABELS[protocol]} - {regime}-yield"
    if xmin_note:
        title += f"\n{xmin_note}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_collapse(curves, result, protocol, regime, path: Path, xmin_note=None):
    exponent = float(result["x"])
    dimension = float(result["dimension"])
    fig, (ax_map, ax_collapse) = plt.subplots(1, 2, figsize=(11, 4.2))
    image = ax_map.pcolormesh(
        result["fine_d"],
        result["fine_x"],
        np.log10(result["fine_q"]),
        shading="auto",
    )
    ax_map.scatter([dimension], [exponent], marker="*", s=90, color="red")
    ax_map.set(xlabel="$D_x$", ylabel="$x$", title="Collapse quality")
    fig.colorbar(image, ax=ax_map, label="$\\log_{10} Q$")
    for size, (drop, density) in curves.items():
        scaled_x = drop / size**dimension
        scaled_y = density * size ** (dimension * exponent)
        ax_collapse.plot(scaled_x, scaled_y, marker="o", ms=3, linestyle="none", label=f"L={size}")
    ax_collapse.set(
        xscale="log",
        yscale="log",
        xlabel="$s/L^{D_x}$",
        ylabel="$P(s;L)L^{D_x x}$",
        title=f"$x={exponent:.3f}$, $D_x={dimension:.3f}$",
    )
    ax_collapse.legend()
    title = f"{PROTOCOL_LABELS[protocol]} - {regime}-yield"
    if xmin_note:
        title += f"\n{xmin_note}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_protocol_comparison(
    all_curves, results, output_path: Path, shared_xmins=None
):
    fig, axes = plt.subplots(len(PROTOCOLS), len(REGIMES), figsize=(10, 13), squeeze=False)
    for row, protocol in enumerate(PROTOCOLS):
        for col, regime in enumerate(REGIMES):
            ax = axes[row, col]
            result = results[protocol][regime]
            exponent = float(result["x"])
            dimension = float(result["dimension"])
            for size, (drop, density) in all_curves[protocol][regime].items():
                ax.plot(
                    drop / size**dimension,
                    density * size ** (dimension * exponent),
                    marker="o",
                    ms=2.5,
                    linestyle="none",
                    label=f"L={size}" if row == 0 and col == 0 else None,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            title = (
                f"{PROTOCOL_LABELS[protocol]}\n"
                f"{regime}: x={exponent:.3f}, D={dimension:.3f}"
            )
            if shared_xmins is not None:
                title += f", xmin={shared_xmins[protocol][regime]:.2e} (L=250)"
            ax.set_title(title)
            if row == len(PROTOCOLS) - 1:
                ax.set_xlabel("$s/L^{D_x}$")
            if col == 0:
                ax.set_ylabel("$P(s;L)L^{D_x x}$")
    axes[0, 0].legend(ncol=3, fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def fixed_xmin_parameter_fits(
    size_drops,
    xmin,
    *,
    fit_cache_dir: Path,
    evaluation_cache_dir: Path,
    uncertainty_accuracy,
    parallel,
    description,
):
    records = {}
    for size, drops in size_drops.items():
        fit = make_fit(
            drops,
            xmin_range=float(xmin),
            cache_dir=str(fit_cache_dir),
        )
        fit.evaluate_fit(
            data=drops,
            confidence=uncertainty_accuracy,
            parallel=parallel,
            cache_dir=str(evaluation_cache_dir),
            tqdmDesc=f"{description}, L={size}",
        )
        distribution = dist_from_fit(fit)
        alpha = float(distribution.alpha)
        cutoff = float(getattr(distribution, "Lambda", np.nan))
        alpha_std = float(getattr(fit, "alpha_std", np.nan))
        cutoff_std = float(getattr(fit, "Lambda_std", np.nan))
        if not np.isfinite(alpha) or not np.isfinite(alpha_std):
            raise RuntimeError(f"Invalid alpha uncertainty for {description}, L={size}.")
        if not np.isfinite(cutoff) or cutoff <= 0 or not np.isfinite(cutoff_std):
            raise RuntimeError(f"Invalid Lambda uncertainty for {description}, L={size}.")
        records[size] = {
            "alpha": alpha,
            "alpha_std": alpha_std,
            "Lambda": cutoff,
            "Lambda_std": cutoff_std,
            "tail_count": int(np.count_nonzero(np.asarray(drops) >= xmin)),
        }
    return records


def plot_parameter_vs_size(parameter_results, shared_xmins, parameter, path: Path):
    if parameter not in {"alpha", "Lambda"}:
        raise ValueError("parameter must be 'alpha' or 'Lambda'.")
    markers = ("o", "s", "^", "D")
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(11, 4.4), sharex=True)
    for ax, regime in zip(axes, REGIMES):
        for marker, protocol in zip(markers, PROTOCOLS):
            records = parameter_results[protocol][regime]
            sizes = np.asarray(sorted(records), dtype=float)
            values = np.asarray(
                [records[int(size)][parameter] for size in sizes], dtype=float
            )
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
                label=(
                    f"{PROTOCOL_LABELS[protocol]} "
                    f"($x_{{min}}={shared_xmins[protocol][regime]:.2e}$)"
                ),
            )
        ax.set_title(f"{regime}-yield")
        ax.set_xlabel("System size $L$")
        ax.grid(alpha=0.2)
        if parameter == "Lambda":
            ax.set_yscale("log")
            ax.set_ylabel(r"Cutoff rate $\lambda$")
        else:
            ax.set_ylabel(r"Exponent $\alpha$")
        ax.legend(fontsize="small")
    fig.suptitle(
        r"Shared $x_{min}$ from $L=250$; error bars are one bootstrap standard deviation"
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.seeds_per_size < 1:
        raise ValueError("--seeds-per-size must be positive.")
    if args.bins_per_decade < 1:
        raise ValueError("--bins-per-decade must be positive.")
    if not 0 < args.uncertainty_accuracy < 0.5:
        raise ValueError("--uncertainty-accuracy must be between 0 and 0.5.")
    regimes = {
        "pre": (args.pre[0], args.pre[1]),
        "post": (args.post[0], args.post[1]),
    }
    paths, inventory = completed_size_scaling_paths(
        Path(args.data_root), args.seeds_per_size, regimes["post"][1]
    )
    pooled = pool_drops(paths, regimes, output / "cache" / "extracted", args.force)
    if args.stage == "extract":
        return

    accuracy_tag = "default" if args.xmin_accuracy is None else f"{args.xmin_accuracy:g}"
    analysis_dir = output / f"xmin_{args.xmin_strategy}_accuracy_{accuracy_tag}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    do_local = args.stage in {"collapse", "all"}
    do_shared = args.stage in {"shared", "parameters", "all"}
    do_parameters = args.stage in {"parameters", "all"}
    all_fits = {protocol: {} for protocol in PROTOCOLS}
    all_raw_curves = {protocol: {} for protocol in PROTOCOLS}
    all_curves = {protocol: {} for protocol in PROTOCOLS}
    all_results = {protocol: {} for protocol in PROTOCOLS}
    summary = {"inventory": inventory, "seeds_per_size": args.seeds_per_size, "results": {}}
    for protocol in PROTOCOLS:
        summary["results"][protocol] = {}
        for regime in regimes:
            print(
                f"Fitting xmin: {protocol}, {regime}-yield, "
                f"strategy={args.xmin_strategy}",
                flush=True,
            )
            size_drops = pooled[protocol][regime]
            fits = fit_xmins(
                size_drops,
                strategy=args.xmin_strategy,
                accuracy=args.xmin_accuracy,
                parallel=args.parallel_xmin,
                cache_dir=output / "cache" / "xmin" / protocol / regime,
            )
            all_fits[protocol][regime] = fits
            raw_curves = histogram_curves(size_drops, args.bins_per_decade)
            all_raw_curves[protocol][regime] = raw_curves
            plot_raw_and_xmin(
                raw_curves,
                fits,
                protocol,
                regime,
                analysis_dir / f"{protocol}_{regime}_raw_xmin.pdf",
                xmin_note=f"Per-size {args.xmin_strategy} xmin (vertical lines)",
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
                xmin_note=f"Per-size {args.xmin_strategy} xmin; tail PDFs renormalized",
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
    if args.stage == "xmin":
        return

    if do_local:
        plot_protocol_comparison(
            all_curves,
            all_results,
            analysis_dir / "protocol_collapse_comparison.pdf",
        )
        (analysis_dir / "collapse_results.json").write_text(
            json.dumps(summary, indent=2)
        )

    if not (do_shared or do_parameters):
        return

    shared_dir = analysis_dir / "shared_L250_xmin"
    shared_dir.mkdir(parents=True, exist_ok=True)
    shared_xmins = {protocol: {} for protocol in PROTOCOLS}
    shared_curves = {protocol: {} for protocol in PROTOCOLS}
    shared_results = {protocol: {} for protocol in PROTOCOLS}
    parameter_results = {protocol: {} for protocol in PROTOCOLS}
    shared_summary = {
        "inventory": inventory,
        "seeds_per_size": args.seeds_per_size,
        "xmin_source_size": 250,
        "xmin_strategy": args.xmin_strategy,
        "uncertainty_accuracy": args.uncertainty_accuracy,
        "bootstrap_sets": max(1, int(1 / (4 * args.uncertainty_accuracy**2))),
        "results": {},
    }
    for protocol in PROTOCOLS:
        shared_summary["results"][protocol] = {}
        for regime in regimes:
            size_drops = pooled[protocol][regime]
            if 250 not in size_drops:
                raise RuntimeError("Shared-xmin analysis requires L=250 data.")
            shared_xmin = float(all_fits[protocol][regime][250].xmin)
            xmins = {size: shared_xmin for size in size_drops}
            shared_xmins[protocol][regime] = shared_xmin
            note = (
                f"Shared xmin={shared_xmin:.3e} from L=250 "
                f"({args.xmin_strategy})"
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
                    shared_dir / f"{protocol}_{regime}_raw_shared_xmin.pdf",
                    applied_xmins=xmins,
                    xmin_note=note,
                )
                curves = tail_histogram_curves(
                    size_drops, xmins, args.bins_per_decade
                )
                shared_curves[protocol][regime] = curves
                result = optimize_collapse(
                    curves,
                    shared_dir / "cache" / "collapse" / f"{protocol}_{regime}.npz",
                    force=args.force,
                )
                shared_results[protocol][regime] = result
                plot_collapse(
                    curves,
                    result,
                    protocol,
                    regime,
                    shared_dir / f"{protocol}_{regime}_collapse_shared_xmin.pdf",
                    xmin_note=note,
                )
                regime_summary["collapse"] = {
                    "x": float(result["x"]),
                    "dimension": float(result["dimension"]),
                    "quality": float(result["quality"]),
                    "optimum_on_search_boundary": bool(result["boundary"]),
                }

            if do_parameters:
                records = fixed_xmin_parameter_fits(
                    size_drops,
                    shared_xmin,
                    fit_cache_dir=(
                        output / "cache" / "fixed_xmin" / protocol / regime
                    ),
                    evaluation_cache_dir=(
                        shared_dir / "cache" / "evaluation" / protocol / regime
                    ),
                    uncertainty_accuracy=args.uncertainty_accuracy,
                    parallel=args.parallel_uncertainty,
                    description=f"{protocol}, {regime}-yield",
                )
                parameter_results[protocol][regime] = records
                regime_summary["parameters"] = {
                    str(size): record for size, record in records.items()
                }
            shared_summary["results"][protocol][regime] = regime_summary

    if do_shared:
        plot_protocol_comparison(
            shared_curves,
            shared_results,
            shared_dir / "protocol_collapse_comparison_shared_xmin.pdf",
            shared_xmins=shared_xmins,
        )
    if do_parameters:
        plot_parameter_vs_size(
            parameter_results,
            shared_xmins,
            "alpha",
            shared_dir / "alpha_vs_size_shared_xmin.pdf",
        )
        plot_parameter_vs_size(
            parameter_results,
            shared_xmins,
            "Lambda",
            shared_dir / "lambda_vs_size_shared_xmin.pdf",
        )
    (shared_dir / "shared_xmin_results.json").write_text(
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
        default="Plots/powerLaw/size_collapse",
    )
    parser.add_argument(
        "--stage",
        choices=("extract", "xmin", "collapse", "shared", "parameters", "all"),
        default="all",
    )
    parser.add_argument("--seeds-per-size", type=int, default=6)
    parser.add_argument("--pre", type=float, nargs=2, default=REGIMES["pre"])
    parser.add_argument("--post", type=float, nargs=2, default=REGIMES["post"])
    parser.add_argument("--xmin-strategy", default="plateau")
    parser.add_argument("--xmin-accuracy", type=float, default=0.1)
    parser.add_argument("--parallel-xmin", action="store_true")
    parser.add_argument("--uncertainty-accuracy", type=float, default=0.05)
    parser.add_argument("--parallel-uncertainty", action="store_true")
    parser.add_argument("--bins-per-decade", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
