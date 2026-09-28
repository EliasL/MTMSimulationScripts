"""Per-size power-law diagnostics for the local sigma-rescue snapshot.

This is deliberately a diagnostic stage, not a scaling fit.  It uses the
standard event pairing contract: post-yield positive ``Delta E_R`` values are
classified with ``kappa_det = mu/(2 rho)`` for ``rho=N/V_0=2``; that event mask is transferred to paired
``Delta E_I``/``Delta E_S`` values before positive filtering.  Each size then
gets one figure containing fitted PDFs and xmin/KS scans for the three drop
measures, plus a separate PDF showing the reversible/irreversible split.
The Clauset refit bootstrap and 95% intervals are computed for the standard
irreversible ``Delta E_S`` fit.

Run from the repository root with::

    .venv/bin/python -m Plotting.sigmaRescueSizeScalingPowerlawDiagnostics
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from MTMath.evaluatePowerlawFit import Truncated_Power_Law
from Plotting.plotPowerLaw import (
    dist_from_fit,
    find_best_xmin,
    make_fit,
    plot_data_pdf,
    plot_fit_pdf,
)
from Plotting.standardPowerlaw import (
    DEFAULT_KAPPA_RHO,
    EventDrops,
    kappa_detection_threshold,
    kappa_from_relaxation_energy,
    positive_es,
    load_sigma_rescue_post_yield,
    require_gamma0_snapshot,
    split_by_kappa,
)


DEFAULT_SNAPSHOT = Path("sigma_rescue_interim/current_gamma0")
DEFAULT_OUTPUT = Path(
    "Plots/powerLaw/sigma_rescue_size_scaling_individual"
)
PROTOCOLS = ("delta_E_I", "delta_E_R", "delta_E_S")
PROTOCOL_SYMBOLS = {
    "delta_E_I": r"$\Delta E_I$",
    "delta_E_R": r"$\Delta E_R$",
    "delta_E_S": r"$\Delta E_S$",
}
PLOT_DROP_LABELS = {
    "delta_E_I": r"E_I",
    "delta_E_R": r"E_R",
    "delta_E_S": r"E_S",
}
REVERSIBLE_COLOR = "#b9dff2"
IRREVERSIBLE_COLOR = "#f6c28b"
ER_ALL_COLOR = "0.55"
DROP_DETECTION_LINE_COLOR = "0.25"
PVALUE_NR_EVALUATION = 20
PVALUE_MIN = 0.1
PVALUE_START_ACCURACY = 0.1
PVALUE_MAX_ACCURACY = 0.01
PVALUE_CONFIDENCE = 0.01
DEFAULT_PVALUE_WORKERS = min(4, os.cpu_count() or 1)


def _positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def _load_post_yield(snapshot_root: Path) -> pd.DataFrame:
    require_gamma0_snapshot(snapshot_root)
    table_path = Path(snapshot_root).resolve() / "tables" / "drops_usable.csv.gz"
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing combined usable table: {table_path}")
    columns = ["size", "load_ip1", "delta_gamma", "reference_volume", *PROTOCOLS]
    frame = load_sigma_rescue_post_yield(snapshot_root, columns)
    if frame.empty:
        raise ValueError(f"The usable drop table is empty: {table_path}")
    if frame.empty:
        raise ValueError(f"No post-yield rows found in {table_path}")
    frame["kappa"] = kappa_from_relaxation_energy(
        frame["delta_E_R"].to_numpy(dtype=float),
        frame["delta_gamma"].to_numpy(dtype=float),
        frame["reference_volume"].to_numpy(dtype=float),
        rho=DEFAULT_KAPPA_RHO,
    )
    return frame


def _fit_population(
    data: np.ndarray,
    *,
    size: int,
    protocol: str,
    output_dir: Path,
    parallel_xmin: bool,
    xmin_selection: str,
    force_recompute: bool,
):
    if data.size < 3:
        raise ValueError(f"L={size}, {protocol} has fewer than 3 fit values.")
    fit = make_fit(
        data,
        cache_dir=str(output_dir / "cache" / protocol / f"L{size}"),
        parallel_xmin=parallel_xmin,
        use_cache=not force_recompute,
        xmin_selection=xmin_selection,
        xmin_search_mode="rapid" if xmin_selection == "rapidGlobal" else "full",
        xmin_search_kwargs={
            "refine": True,
        },
    )
    analysis = fit.xmin_fitting_results
    for key in ("simple_drop_xmin", "global_min_xmin", "global_min_distance"):
        if key not in analysis:
            raise RuntimeError(f"Missing {key} for L={size}, {protocol}.")
    return fit


def _plot_fit_panel(ax, fit, data: np.ndarray, protocol: str):
    symbol = PROTOCOL_SYMBOLS[protocol]
    plot_data_pdf(
        ax,
        data,
        label="irreversible events",
        color="black",
        drop_label=PLOT_DROP_LABELS[protocol],
        drop_sign="positive",
        show_legend=False,
    )
    plot_fit_pdf(
        ax,
        fit,
        color="tab:red",
        label="global-min truncated power law",
        drop_label=PLOT_DROP_LABELS[protocol],
        drop_sign="positive",
        show_legend=False,
        set_title=False,
        x_grid_mode="smooth",
        xmin_only=True,
        linewidth=1.3,
    )
    analysis = fit.xmin_fitting_results
    global_xmin = float(analysis["global_min_xmin"])
    ax.axvline(
        global_xmin,
        color="0.2",
        linestyle=":",
        linewidth=1.0,
        label="global minimum",
    )
    distribution = dist_from_fit(fit)
    lambda_value = float(getattr(distribution, "Lambda", np.nan))
    annotation = (
        rf"$\Delta E_{{\min}}^{{global}}={global_xmin:.2e}$" "\n"
        rf"$\alpha={float(distribution.alpha):.2f}$, "
        rf"$\lambda={lambda_value:.2e}$" "\n"
        rf"$D={float(fit.D):.3f}$, $n={data.size}$"
    )
    ax.text(
        0.04,
        0.04,
        annotation,
        transform=ax.transAxes,
        va="bottom",
        fontsize="x-small",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    ax.set_title(symbol)
    ax.set_xlabel(symbol)
    ax.set_ylabel(rf"$p({symbol[1:-1]})$")
    ax.grid(alpha=0.18)


def _set_pdf_axes(ax, quantity: str) -> None:
    """Use the same log-PDF presentation as the reversibility flowchart."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"${quantity}$")
    ax.set_ylabel(rf"$p({quantity})$")
    ax.grid(False, which="both")


def _save_separation_figure(
    size: int,
    kappa_all: np.ndarray,
    kappa_det: float,
    er_all: np.ndarray,
    er_rev: np.ndarray,
    er_irrev: np.ndarray,
    es_rev: np.ndarray,
    es_irrev: np.ndarray,
    output_dir: Path,
) -> None:
    """Save the kappa classification, Delta E_R, and paired Delta E_S PDFs."""
    if kappa_all.size == 0:
        raise ValueError(f"L={size} has no positive kappa values to plot.")
    if er_all.size == 0 or er_rev.size == 0 or er_irrev.size == 0:
        raise ValueError(
            f"L={size} needs positive Delta E_R values in both split classes; "
            f"got all={er_all.size}, reversible={er_rev.size}, "
            f"irreversible={er_irrev.size}."
        )
    if es_rev.size == 0 or es_irrev.size == 0:
        raise ValueError(
            f"L={size} needs positive Delta E_S values in both split classes; "
            f"got reversible={es_rev.size}, irreversible={es_irrev.size}."
        )

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), squeeze=False)
    kappa_ax, er_ax, es_ax = axes[0]
    plot_data_pdf(
        kappa_ax,
        kappa_all,
        label=rf"all $\kappa$ events ($n={kappa_all.size}$)",
        color=ER_ALL_COLOR,
        drop_label=r"\kappa",
        drop_sign="positive",
        show_legend=False,
    )
    _set_pdf_axes(kappa_ax, r"\kappa")
    lo, hi = kappa_ax.get_xlim()
    kappa_ax.axvspan(lo, kappa_det, color=REVERSIBLE_COLOR, alpha=0.34, zorder=0)
    kappa_ax.axvspan(kappa_det, hi, color=IRREVERSIBLE_COLOR, alpha=0.30, zorder=0)
    kappa_ax.axvline(
        kappa_det,
        color=DROP_DETECTION_LINE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label=rf"$\kappa_{{\det}}={kappa_det:.2e}$",
        zorder=4,
    )
    kappa_ax.set_title(r"$\kappa$ split")
    kappa_ax.legend(loc="best", fontsize="x-small")

    plot_data_pdf(
        er_ax,
        er_all,
        label=rf"all events ($n={er_all.size}$)",
        color=ER_ALL_COLOR,
        drop_label=r"\Delta E_R",
        drop_sign="positive",
        show_legend=False,
    )
    plot_data_pdf(
        er_ax,
        er_rev,
        label=rf"reversible ($n={er_rev.size}$)",
        color=REVERSIBLE_COLOR,
        drop_label=r"\Delta E_R",
        drop_sign="positive",
        show_legend=False,
    )
    plot_data_pdf(
        er_ax,
        er_irrev,
        label=rf"irreversible ($n={er_irrev.size}$)",
        color=IRREVERSIBLE_COLOR,
        drop_label=r"\Delta E_R",
        drop_sign="positive",
        show_legend=False,
    )
    _set_pdf_axes(er_ax, r"\Delta E_R")
    er_ax.set_title(r"$\Delta E_R$ and $\kappa_{\det}$ split")
    er_ax.legend(loc="best", fontsize="x-small")

    plot_data_pdf(
        es_ax,
        es_rev,
        label=rf"reversible ($n={es_rev.size}$)",
        color=REVERSIBLE_COLOR,
        drop_label=r"E_S",
        drop_sign="positive",
        show_legend=False,
    )
    plot_data_pdf(
        es_ax,
        es_irrev,
        label=rf"irreversible ($n={es_irrev.size}$)",
        color=IRREVERSIBLE_COLOR,
        drop_label=r"E_S",
        drop_sign="positive",
        show_legend=False,
    )
    _set_pdf_axes(es_ax, r"\Delta E_S")
    es_ax.set_title(r"$\Delta E_S$ after $\kappa_{\det}$ split")
    es_ax.legend(loc="best", fontsize="x-small")

    fig.suptitle(rf"Post-yield separation, $L={size}$")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.84, wspace=0.30)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"L{size}_post_yield_separation_pdfs.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(pdf_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_full_xmin_scan(
    ax,
    fit,
    data: np.ndarray,
    *,
    size: int,
    protocol: str,
    output_dir: Path,
    pvalue_workers: int,
    bootstrap_xmin_mode: str,
    pvalue_confidence: float,
    force_pvalue: bool,
) -> None:
    """Draw the empirical xmin scan and bootstrap the standard Delta E_S fit."""
    if fit.xmin_fitting_results is None:
        raise RuntimeError(f"Missing xmin results for L={size}, {protocol}.")
    parallel = int(pvalue_workers) > 1
    eval_cache_dir = output_dir / "cache" / "xmin_bootstrap" / protocol / f"L{size}"
    if protocol == "delta_E_S":
        fit.evaluate_clauset_pvalue(
            data=data,
            confidence=pvalue_confidence,
            parallel=parallel,
            max_workers=pvalue_workers,
            use_cache=not force_pvalue,
            cache_dir=str(eval_cache_dir),
            tqdmDesc=f"{protocol}, post-yield, L={size}: Clauset p-value",
            xmin_mode=bootstrap_xmin_mode,
        )
    scan = fit.xmin_fitting_results
    xmins = np.asarray(scan["xmins"], dtype=float)
    distances = np.asarray(scan["distances"], dtype=float)
    alphas = np.asarray(scan["alphas"], dtype=float)
    valid = np.isfinite(xmins) & np.isfinite(distances) & np.isfinite(alphas)
    if "valid_fits" in scan:
        valid &= np.asarray(scan["valid_fits"], dtype=bool)
    if not np.any(valid):
        raise ValueError(f"No valid xmin scan for L={size}, {protocol}.")
    ax.plot(xmins[valid], distances[valid], color="tab:blue", label="KS distance")
    alpha_ax = ax.twinx()
    alpha_ax.plot(xmins[valid], alphas[valid], color="tab:orange", alpha=0.7,
                  label=r"empirical $\alpha(x_{\min})$")
    ax.axvline(float(scan["global_min_xmin"]), color="black", linestyle=":")
    ax.set(xscale="log", xlabel=r"$\Delta E_{\min}$", ylabel="KS distance")
    alpha_ax.set_ylabel(r"$\alpha$")
    title = PROTOCOL_SYMBOLS[protocol]
    if protocol == "delta_E_S":
        title += rf"; selected-fit $p={float(fit.p):.3f}$"
    ax.set_title(title)


def _save_size_figure(
    size: int,
    fits: dict[str, object],
    data: dict[str, np.ndarray],
    output_dir: Path,
    protocols: tuple[str, ...],
    pvalue_workers: int,
    bootstrap_xmin_mode: str,
    pvalue_confidence: float,
    force_pvalue: bool,
) -> None:
    fig, axes = plt.subplots(
        2,
        len(protocols),
        figsize=(6.0 * len(protocols), 8.8),
        squeeze=False,
    )
    for column, protocol in enumerate(protocols):
        _plot_fit_panel(axes[0, column], fits[protocol], data[protocol], protocol)
        _draw_full_xmin_scan(
            axes[1, column],
            fits[protocol],
            data[protocol],
            size=size,
            protocol=protocol,
            output_dir=output_dir,
            pvalue_workers=pvalue_workers,
            bootstrap_xmin_mode=bootstrap_xmin_mode,
            pvalue_confidence=pvalue_confidence,
            force_pvalue=force_pvalue,
        )

    handles = [
        Line2D([], [], color="black", marker="o", linestyle="None", label="irreversible data"),
        Line2D([], [], color="tab:red", label="global-min truncated power law"),
        Line2D([], [], color="black", marker="x", linestyle="None", label="global minimum"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize="small",
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.suptitle(
        rf"Post-yield, $L={size}$; irreversible population from "
        rf"$\kappa_{{\det}}$ split",
        y=1.02,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.97,
        bottom=0.08,
        top=0.88,
        wspace=0.52,
        hspace=0.44,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"L{size}_post_yield_fits_and_xmin_scans.pdf"
    png_path = pdf_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(
    snapshot_root: Path = DEFAULT_SNAPSHOT,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    parallel_xmin: bool = False,
    force: bool = False,
    resume: bool = False,
    sizes: tuple[int, ...] | None = None,
    protocols: tuple[str, ...] = PROTOCOLS,
    pvalue_workers: int = DEFAULT_PVALUE_WORKERS,
    bootstrap_xmin_mode: str = "global",
    pvalue_confidence: float = PVALUE_CONFIDENCE,
    observed_xmin_mode: str = "global",
) -> dict:
    protocols = tuple(protocols)
    if not protocols:
        raise ValueError("At least one diagnostic protocol is required.")
    unknown_protocols = sorted(set(protocols) - set(PROTOCOLS))
    if unknown_protocols:
        raise ValueError(f"Unknown diagnostic protocols: {unknown_protocols}")
    if bootstrap_xmin_mode in {"full", "rapid"}:
        bootstrap_xmin_mode = {"full": "global", "rapid": "rapidGlobal"}[
            bootstrap_xmin_mode
        ]
    if bootstrap_xmin_mode not in {"global", "rapidGlobal"}:
        raise ValueError(
            "bootstrap_xmin_mode must be either 'global' or 'rapidGlobal'."
        )
    if observed_xmin_mode in {"full", "rapid"}:
        observed_xmin_mode = {"full": "global", "rapid": "rapidGlobal"}[
            observed_xmin_mode
        ]
    if observed_xmin_mode not in {"global", "rapidGlobal"}:
        raise ValueError(
            "observed_xmin_mode must be either 'global' or 'rapidGlobal'."
        )
    frame = _load_post_yield(snapshot_root)
    if sizes is not None:
        requested = {int(size) for size in sizes}
        available = set(frame["size"].astype(int).unique())
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Requested sizes are absent from the snapshot: {missing}")
        frame = frame.loc[frame["size"].isin(requested)].copy()
    if (
        not force
        and not resume
        and output_dir.exists()
        and any(output_dir.glob("L*_post_yield_fits_and_xmin_scans.pdf"))
    ):
        raise FileExistsError(
            f"Diagnostic output already exists in {output_dir}; use --force to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(pvalue_workers) != pvalue_workers or pvalue_workers < 1:
        raise ValueError("pvalue_workers must be a positive integer.")
    pvalue_workers = int(pvalue_workers)
    if not 0.0 < pvalue_confidence <= 0.5:
        raise ValueError("pvalue_confidence must lie in (0, 0.5].")

    summary = {
        "snapshot_root": str(Path(snapshot_root).resolve()),
        "delta_E_S_tangent": "gamma0",
        "post_yield_window": "per-run validated sigma12 peak + 0.01, load < 1.0",
        "yield_peak_audit": str((Path(snapshot_root).resolve() / "tables" / "yield_peaks.csv")),
        "population": "kappa_det = mu/(2 rho) irreversible event mask, rho=N/V_0=2",
        "kappa_detection": (
            "kappa = Delta E_R/(rho V_0 Delta gamma^2), "
            "rho=N/V_0=2"
        ),
        "xmin_selection": (
            "global minimum after exhaustive observed-candidate scan"
            if observed_xmin_mode == "global"
            else "rapidGlobal minimum after 100-point coarse scan and local refinement"
        ),
        "selected_fit_pvalue": (
            "Clauset semiparametric bootstrap v2: empirical resampling below "
            "the observed xmin and "
            + (
                "global-xmin refitting"
                if bootstrap_xmin_mode == "global"
                else "rapidGlobal-xmin refitting"
            )
            + " for every synthetic Delta E_S set; Delta E_I and Delta E_R are cutoff diagnostics"
        ),
        "bootstrap_xmin_mode": bootstrap_xmin_mode,
        "bootstrap_xmin_strategy": (
            "exhaustive observed-candidate global search for every synthetic set"
            if bootstrap_xmin_mode == "global"
            else "100-point coarse KS grid plus local-minimum refinement for "
            "every synthetic set"
        ),
        "xmin_scan_pvalues": "fixed-xmin diagnostic values",
        "pvalue_confidence": float(pvalue_confidence),
        "pvalue_bootstrap_replicates": max(
            1, int(1 / (4 * pvalue_confidence**2))
        ),
        "protocols": list(protocols),
        "fits": {},
    }
    for size in sorted(frame["size"].unique()):
        size_frame = frame.loc[frame["size"] == size].reset_index(drop=True)
        kappa_all = _positive(size_frame["kappa"].to_numpy(dtype=float))
        if kappa_all.size < 3:
            raise ValueError(f"L={size} has fewer than 3 positive kappa events.")
        kappa_det = kappa_detection_threshold()
        paired = EventDrops(
            er=size_frame["delta_E_R"].to_numpy(dtype=float),
            es=size_frame["delta_E_S"].to_numpy(dtype=float),
            kappa=size_frame["kappa"].to_numpy(dtype=float),
        )
        split = split_by_kappa(paired, kappa_det)
        event_split = split.is_rev | split.is_irrev
        er_all = _positive(paired.er[event_split])
        er_rev = _positive(paired.er[split.is_rev])
        er_irrev = _positive(paired.er[split.is_irrev])
        data = {
            "delta_E_I": _positive(
                size_frame.loc[split.is_irrev, "delta_E_I"].to_numpy(dtype=float)
            ),
            "delta_E_R": _positive(paired.er[split.is_irrev]),
            "delta_E_S": positive_es(paired, split.is_irrev),
        }
        es_rev = positive_es(paired, split.is_rev)
        _save_separation_figure(
            int(size),
            kappa_all,
            kappa_det,
            er_all,
            er_rev,
            er_irrev,
            es_rev,
            data["delta_E_S"],
            output_dir,
        )
        fits = {
            protocol: _fit_population(
                data[protocol],
                size=int(size),
                protocol=protocol,
                output_dir=output_dir,
                parallel_xmin=parallel_xmin,
                xmin_selection=observed_xmin_mode,
                force_recompute=force,
            )
            for protocol in protocols
        }
        _save_size_figure(
            int(size),
            fits,
            data,
            output_dir,
            protocols,
            pvalue_workers,
            bootstrap_xmin_mode,
            pvalue_confidence,
            force_pvalue=force,
        )
        summary["fits"][str(int(size))] = {
            "post_yield_rows": int(len(size_frame)),
            "kappa_det": kappa_det,
            "reversible_events": int(np.count_nonzero(split.is_rev)),
            "irreversible_events": int(np.count_nonzero(split.is_irrev)),
            "positive_delta_E_S_reversible_count": int(es_rev.size),
            "positive_delta_E_S_irreversible_count": int(data["delta_E_S"].size),
            "protocols": {
                protocol: {
                    "positive_data_count": int(data[protocol].size),
                    "simple_drop_xmin": float(fits[protocol].xmin_fitting_results["simple_drop_xmin"]),
                    "global_min_xmin": float(fits[protocol].xmin_fitting_results["global_min_xmin"]),
                    "global_min_D": float(fits[protocol].xmin_fitting_results["global_min_distance"]),
                    "clauset_pvalue_v2": (
                        float(fits[protocol].p) if protocol == "delta_E_S" else None
                    ),
                    "bootstrap_xmin_ci_95": (
                        [float(fits[protocol].bootstrap_xmin_ci_low),
                         float(fits[protocol].bootstrap_xmin_ci_high)]
                        if protocol == "delta_E_S" else None
                    ),
                    "bootstrap_alpha_ci_95": (
                        [float(fits[protocol].alpha_ci_low),
                         float(fits[protocol].alpha_ci_high)]
                        if protocol == "delta_E_S" else None
                    ),
                }
                for protocol in protocols
            },
        }
        print(f"Saved L={int(size)} post-yield fit and xmin diagnostics", flush=True)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parallel-xmin", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse cached fits and bootstrap samples in an existing output directory.",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument(
        "--protocols",
        choices=PROTOCOLS,
        nargs="+",
        default=PROTOCOLS,
        help="Drop populations to fit and plot (default: all three).",
    )
    parser.add_argument(
        "--observed-xmin-mode",
        choices=("global", "rapidGlobal"),
        default="global",
        help=(
            "xmin search used for the observed data: exhaustive global or "
            "100-point rapidGlobal (default: global)."
        ),
    )
    parser.add_argument(
        "--pvalue-workers",
        type=int,
        default=DEFAULT_PVALUE_WORKERS,
        help=f"Delta E_S bootstrap worker processes (default: {DEFAULT_PVALUE_WORKERS}).",
    )
    parser.add_argument(
        "--bootstrap-xmin-mode",
        choices=("global", "rapidGlobal"),
        default="global",
        help=(
            "xmin search used while refitting bootstrap samples: exhaustive "
            "global or 100-point rapidGlobal (default: global)."
        ),
    )
    parser.add_argument(
        "--pvalue-confidence",
        type=float,
        default=PVALUE_CONFIDENCE,
        help=(
            "Bootstrap confidence parameter; 0.01 gives 2500 replicates, "
            "larger values are faster diagnostics (default: 0.01)."
        ),
    )
    args = parser.parse_args()
    run(
        args.snapshot_root,
        args.output_dir,
        parallel_xmin=args.parallel_xmin,
        force=args.force,
        resume=args.resume,
        sizes=None if args.sizes is None else tuple(args.sizes),
        protocols=tuple(args.protocols),
        pvalue_workers=args.pvalue_workers,
        bootstrap_xmin_mode=args.bootstrap_xmin_mode,
        pvalue_confidence=args.pvalue_confidence,
        observed_xmin_mode=args.observed_xmin_mode,
    )


if __name__ == "__main__":
    main()
