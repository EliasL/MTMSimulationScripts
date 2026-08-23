"""Temporary finite-size-scaling plots for a sigma-rescue snapshot.

The default run uses L=150, 200, 250 and produces results for Delta E_S,
Delta E_R, and Delta E_I. It applies the paired kappa classification in the
post-yield window, chooses the exhaustive observed-candidate global minimum
of the KS distance independently for each size and observable, and applies
that xmin to every PDF, collapse, and moment plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MTMath.finiteSizeScaling import (
    build_log_pdf_curves,
    collapsed_curves,
    filter_by_xmin,
    fit_moment_scaling,
    optimize_collapse,
)
from Plotting.plotPowerLaw import make_fit
from Plotting.standardPowerlaw import (
    EventDrops,
    kappa_detection_threshold,
    split_by_kappa,
)


DEFAULT_SNAPSHOT = Path("sigma_rescue_interim/snapshots/20260819T100206Z")
DEFAULT_OUTPUT = Path("Plots/powerLaw/sigma_rescue_finite_size_scaling_L150_L250")
POST_YIELD = (0.7, 1.0)
DEFAULT_SIZES = (150, 200, 250)
OBSERVABLES = ("delta_E_S", "delta_E_R", "delta_E_I")
OBSERVABLE_LABELS = {
    "delta_E_S": r"$\Delta E_S$",
    "delta_E_R": r"$\Delta E_R$",
    "delta_E_I": r"$\Delta E_I$",
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _load_and_classify(snapshot_root: Path) -> tuple[pd.DataFrame, float]:
    table = Path(snapshot_root).resolve() / "tables" / "drops_usable.csv.gz"
    if not table.is_file():
        raise FileNotFoundError(f"Missing usable drop table: {table}")
    columns = [
        "size",
        "seed",
        "load_ip1",
        "delta_gamma",
        "reference_volume",
        "delta_E_I",
        "delta_E_R",
        "delta_E_S",
    ]
    frame = pd.read_csv(table, usecols=columns, low_memory=False)
    if set(frame.columns) != set(columns):
        raise RuntimeError(f"Unexpected columns in {table}: {sorted(frame.columns)}")
    frame = frame.loc[
        (frame["load_ip1"] > POST_YIELD[0]) & (frame["load_ip1"] < POST_YIELD[1])
    ].copy()
    if frame.empty:
        raise ValueError(f"No rows in the post-yield window {POST_YIELD}.")
    for column in columns:
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values found in required column {column}.")
    if np.any(frame["delta_gamma"].to_numpy(dtype=float) <= 0):
        raise ValueError("delta_gamma must be finite and positive.")
    if np.any(frame["reference_volume"].to_numpy(dtype=float) <= 0):
        raise ValueError("reference_volume must be finite and positive.")

    frame["kappa"] = frame["delta_E_R"] / (
        frame["reference_volume"] * frame["delta_gamma"] ** 2
    )
    frame["is_reversible"] = False
    frame["is_irreversible"] = False
    kappa_det = kappa_detection_threshold()
    for size in sorted(frame["size"].unique()):
        selected = frame.loc[frame["size"] == size]
        drops = EventDrops(
            er=selected["delta_E_R"].to_numpy(dtype=float),
            es=selected["delta_E_S"].to_numpy(dtype=float),
            kappa=selected["kappa"].to_numpy(dtype=float),
        )
        split = split_by_kappa(drops, kappa_det)
        frame.loc[selected.index, "is_reversible"] = split.is_rev
        frame.loc[selected.index, "is_irreversible"] = split.is_irrev
    return frame, kappa_det


def _coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    coverage = frame.groupby(["size", "seed"], sort=True).agg(
        event_rows=("size", "size"),
        load_low=("load_ip1", "min"),
        load_high=("load_ip1", "max"),
        reversible_events=("is_reversible", "sum"),
        irreversible_events=("is_irreversible", "sum"),
    )
    for observable in OBSERVABLES:
        frame[f"positive_{observable}_irreversible"] = (
            frame["is_irreversible"]
            & np.isfinite(frame[observable])
            & (frame[observable] > 0)
        )
        coverage[f"positive_{observable}_irreversible"] = frame.groupby(
            ["size", "seed"], sort=True
        )[f"positive_{observable}_irreversible"].sum()
    return coverage.reset_index()


def _samples_by_size(
    frame: pd.DataFrame, observable: str, sizes: tuple[int, ...]
) -> dict[int, np.ndarray]:
    if observable not in OBSERVABLES:
        raise ValueError(f"Unknown observable {observable!r}.")
    samples = {}
    for size in sizes:
        values = frame.loc[
            (frame["size"] == size)
            & frame["is_irreversible"]
            & np.isfinite(frame[observable])
            & (frame[observable] > 0),
            observable,
        ].to_numpy(dtype=float)
        if values.size < 3:
            raise ValueError(
                f"L={size}, {observable} has fewer than three positive irreversible values."
            )
        samples[int(size)] = values
    return samples


def _fit_global_xmins(
    samples: dict[int, np.ndarray],
    observable: str,
    output_dir: Path,
    *,
    force: bool,
) -> tuple[dict[int, float], dict[int, float], dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Choose the exhaustive observed-candidate global minimum of D per tail."""

    xmins = {}
    distances = {}
    scans = {}
    for size, values in sorted(samples.items()):
        fit = make_fit(
            values,
            cache_dir=str(output_dir / "xmin_cache" / observable / f"L{size}"),
            # The cache key includes the full data and xmin-search settings;
            # --force controls output regeneration without repeating valid scans.
            use_cache=True,
            parallel_xmin=False,
            xmin_selection="global",
            xmin_search_mode="full",
        )
        analysis = fit.xmin_fitting_results
        if analysis is None or "global_min_xmin" not in analysis:
            raise RuntimeError(f"Global xmin analysis is missing for L={size}, {observable}.")
        xmins[size] = float(analysis["global_min_xmin"])
        distances[size] = float(analysis["global_min_distance"])
        details = analysis["global_search_details"]
        scans[size] = (
            np.asarray(details["xmins"], dtype=float),
            np.asarray(details["distances"], dtype=float),
        )
    return xmins, distances, scans


def _plot_xmin_scans(
    scans: dict[int, tuple[np.ndarray, np.ndarray]],
    xmins: dict[int, float],
    observable: str,
    output: Path,
) -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    label = OBSERVABLE_LABELS[observable]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for index, size in enumerate(sorted(scans)):
        candidates, distances = scans[size]
        color = colors[index % len(colors)]
        mask = np.isfinite(candidates) & np.isfinite(distances) & (candidates > 0)
        ax.semilogx(candidates[mask], distances[mask], color=color, lw=0.8, label=f"L={size}")
        selected = xmins[size]
        selected_distance = float(np.interp(np.log(selected), np.log(candidates[mask]), distances[mask]))
        ax.plot(selected, selected_distance, "x", color=color, ms=7, mew=1.5)
    ax.set_xlabel(rf"$x_{{\min}}$ for {label}")
    ax.set_ylabel("KS distance $D$")
    ax.set_title(rf"Global-minimum-$D$ selection for {label}")
    ax.grid(alpha=0.2)
    ax.legend(fontsize="small")
    fig.tight_layout()
    _save(fig, output)


def _plot_collapse(
    curves,
    result: dict[str, object],
    observable: str,
    xmins: dict[int, float],
    output: Path,
) -> None:
    exponent = float(result["exponent"])
    dimension = float(result["dimension"])
    transformed = collapsed_curves(curves, exponent, dimension)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    label = OBSERVABLE_LABELS[observable]
    label_plain = label[1:-1]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    raw_ax, collapsed_ax, landscape_ax, residual_ax = axes.flat
    for index, (size, curve) in enumerate(sorted(curves.items())):
        color = colors[index % len(colors)]
        raw_ax.loglog(
            curve.x,
            curve.density,
            "o-",
            ms=3,
            lw=0.9,
            color=color,
            label=rf"L={int(size)}; $x_{{\min}}={xmins[int(size)]:.1e}$",
        )
        x, y = transformed[size]
        collapsed_ax.loglog(x, y, "o-", ms=3, lw=0.9, color=color, label=f"L={int(size)}")
    raw_ax.set_xlabel(label)
    raw_ax.set_ylabel(rf"$p({label_plain})$")
    raw_ax.set_title("Positive irreversible PDFs above global-min-D xmin")
    collapsed_ax.set_xlabel(rf"${label_plain}/L^{{d_E}}$")
    collapsed_ax.set_ylabel(rf"$L^{{d_E\tau}}p({label_plain},L)$")
    collapsed_ax.set_title(rf"Collapse: $\tau={exponent:.3f}$, $d_E={dimension:.3f}$")
    raw_ax.legend(fontsize="small")
    collapsed_ax.legend(fontsize="small")
    for ax in (raw_ax, collapsed_ax):
        ax.grid(alpha=0.2, which="both")

    coarse_exp = np.asarray(result["coarse_exponents"])
    coarse_dim = np.asarray(result["coarse_dimensions"])
    coarse_quality = np.asarray(result["coarse_quality"])
    mesh = landscape_ax.pcolormesh(
        coarse_exp,
        coarse_dim,
        np.maximum(coarse_quality.T, np.finfo(float).tiny),
        shading="auto",
    )
    landscape_ax.plot(exponent, dimension, "wx", ms=8, mew=2)
    landscape_ax.set_xlabel(r"$\tau$")
    landscape_ax.set_ylabel(r"$d_E$")
    landscape_ax.set_title("Coarse collapse objective")
    fig.colorbar(mesh, ax=landscape_ax, label="log-PDF overlap variance")

    scaled_domains = [curve.x / size**dimension for size, curve in curves.items()]
    low = max(values.min() for values in scaled_domains)
    high = min(values.max() for values in scaled_domains)
    common = np.logspace(np.log10(low), np.log10(high), 80)
    log_values = []
    for size, curve in sorted(curves.items()):
        x, y = transformed[size]
        log_values.append(np.interp(np.log(common), np.log(x), np.log(y)))
    residual_ax.semilogx(common, np.std(np.asarray(log_values), axis=0), color="black")
    residual_ax.set_xlabel(rf"${label_plain}/L^{{d_E}}$")
    residual_ax.set_ylabel("log-density spread")
    residual_ax.set_title("Residual spread across sizes")
    residual_ax.grid(alpha=0.2)
    fig.suptitle(
        rf"Finite-size scaling of {label}; $0.7<\gamma<1.0$, "
        rf"$\kappa_{{det}}=\mu/2$"
    )
    fig.tight_layout()
    _save(fig, output)


def _plot_moments(moment: dict[str, object], observable: str, output: Path) -> None:
    sizes = np.asarray(moment["sizes"], dtype=float)
    orders = np.asarray(moment["orders"], dtype=float)
    moments = np.asarray(moment["moments"], dtype=float)
    exponents = np.asarray(moment["moment_exponents"], dtype=float)
    errors = np.asarray(moment["moment_exponent_stderr"], dtype=float)
    dimension = float(moment["dimension"])
    intercept = float(moment["moment_order_intercept"])
    label = OBSERVABLE_LABELS[observable]
    fig, (moment_ax, exponent_ax) = plt.subplots(1, 2, figsize=(11, 4.3))
    for column, order in enumerate(orders):
        moment_ax.loglog(sizes, moments[:, column], "o-", label=rf"$q={order:g}$")
    moment_ax.set_xlabel("System size $L$")
    moment_ax.set_ylabel(rf"Moment $\langle({label[1:-1]})^q\rangle$")
    moment_ax.set_title("Moment scaling above global-min-D xmin")
    moment_ax.grid(alpha=0.2, which="both")
    moment_ax.legend(fontsize="small")
    exponent_ax.errorbar(
        orders,
        exponents,
        yerr=errors,
        fmt="o",
        capsize=3,
        label="fitted moment exponents",
    )
    order_grid = np.linspace(orders.min(), orders.max(), 100)
    exponent_ax.plot(order_grid, dimension * order_grid + intercept, label=rf"$d_E={dimension:.3f}$")
    exponent_ax.set_xlabel("Moment order $q$")
    exponent_ax.set_ylabel(r"$z_q$ in $\langle E^q\rangle\sim L^{z_q}$")
    exponent_ax.set_title(rf"{label} moment-order fit: $\tau={float(moment['exponent']):.3f}$")
    exponent_ax.grid(alpha=0.2)
    exponent_ax.legend(fontsize="small")
    fig.tight_layout()
    _save(fig, output)


def _json_number(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def run(
    snapshot_root: Path = DEFAULT_SNAPSHOT,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    observables: tuple[str, ...] = OBSERVABLES,
    bins_per_decade: int = 12,
    force: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists() and (output_dir / "summary.json").exists() and not force:
        raise FileExistsError(f"Output already exists: {output_dir}; use --force to overwrite.")
    frame, kappa_det = _load_and_classify(Path(snapshot_root))
    sizes = tuple(sorted({int(size) for size in sizes}))
    observables = tuple(observables)
    if len(sizes) < 2:
        raise ValueError("Finite-size scaling requires at least two requested sizes.")
    available = set(frame["size"].astype(int).unique())
    missing_sizes = sorted(set(sizes) - available)
    if missing_sizes:
        raise ValueError(f"Requested sizes are absent from the snapshot: {missing_sizes}")
    unknown = sorted(set(observables) - set(OBSERVABLES))
    if unknown:
        raise ValueError(f"Unknown observables: {unknown}")
    coverage = _coverage_table(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / "coverage.csv", index=False)

    results = {}
    xmin_rows = []
    for observable in observables:
        observable_dir = output_dir / observable
        samples = _samples_by_size(frame, observable, sizes)
        xmins, distances, scans = _fit_global_xmins(
            samples, observable, output_dir, force=force
        )
        tail_samples = filter_by_xmin(samples, xmins)
        curves = build_log_pdf_curves(tail_samples, bins_per_decade=bins_per_decade)
        collapse = optimize_collapse(curves)
        moment = fit_moment_scaling(tail_samples)
        observable_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            observable_dir / "collapse_landscape.npz",
            coarse_exponents=collapse["coarse_exponents"],
            coarse_dimensions=collapse["coarse_dimensions"],
            coarse_quality=collapse["coarse_quality"],
            fine_exponents=collapse["fine_exponents"],
            fine_dimensions=collapse["fine_dimensions"],
            fine_quality=collapse["fine_quality"],
        )
        _plot_collapse(
            curves,
            collapse,
            observable,
            xmins,
            observable_dir / "finite_size_collapse.pdf",
        )
        _plot_moments(moment, observable, observable_dir / "moment_scaling.pdf")
        _plot_xmin_scans(
            scans,
            xmins,
            observable,
            observable_dir / "xmin_global_D_scans.pdf",
        )
        xmin_rows.extend(
            {
                "observable": observable,
                "size": int(size),
                "xmin": float(xmins[size]),
                "global_min_D": float(distances[size]),
                "tail_count": int(tail_samples[size].size),
                "min_tail_count_for_scan": 100,
            }
            for size in sizes
        )
        results[observable] = {
            "xmins": {str(size): float(value) for size, value in xmins.items()},
            "global_min_D": {str(size): float(value) for size, value in distances.items()},
            "tail_counts": {str(size): int(tail_samples[size].size) for size in sizes},
            "collapse": {
                "tau": float(collapse["exponent"]),
                "d_E": float(collapse["dimension"]),
                "quality": float(collapse["quality"]),
                "boundary": bool(collapse["boundary"]),
            },
            "moments": {
                "orders": [_json_number(value) for value in moment["orders"]],
                "moment_exponents": [_json_number(value) for value in moment["moment_exponents"]],
                "dimension": float(moment["dimension"]),
                "dimension_stderr": float(moment["dimension_stderr"]),
                "tau": float(moment["exponent"]),
                "r_squared": [_json_number(value) for value in moment["moment_r_squared"]],
                "order_fit_r_squared": float(moment["moment_order_r_squared"]),
            },
        }

    pd.DataFrame(xmin_rows).to_csv(output_dir / "xmin_selection.csv", index=False)
    summary = {
        "snapshot_root": str(Path(snapshot_root).resolve()),
        "post_yield_window": list(POST_YIELD),
        "classification": "kappa_det = mu/2; kappa = Delta E_R/(V0 Delta gamma^2), rho=1",
        "kappa_det": float(kappa_det),
        "population": "finite positive irreversible observable values, after paired event classification",
        "bins_per_decade": int(bins_per_decade),
        "sizes": list(sizes),
        "observables": list(observables),
        "xmin_selection": "global minimum KS distance over every eligible observed candidate, independently per size and observable; minimum tail count=100",
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--observables", nargs="+", choices=OBSERVABLES, default=list(OBSERVABLES))
    parser.add_argument("--bins-per-decade", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.bins_per_decade < 1:
        raise ValueError("--bins-per-decade must be positive.")
    run(
        args.snapshot_root,
        args.output_dir,
        sizes=tuple(args.sizes),
        observables=tuple(args.observables),
        bins_per_decade=args.bins_per_decade,
        force=args.force,
    )


if __name__ == "__main__":
    main()
