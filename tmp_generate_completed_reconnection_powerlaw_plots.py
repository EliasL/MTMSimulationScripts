"""Generate standard kappa-classified power-law plots for the two stopped runs."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Management.updateCSV import (
    RECONNECT_REVERSIBILITY_MACRODATA_HEADER,
    update_df_header,
)
from MTMath.evaluatePowerlawFit import Fit, Truncated_Power_Law
from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import calculate_energy_step_data
from Plotting.findXmin import analyze_xmin
from Plotting.plotPowerLaw import (
    dist_from_fit,
    plot_data_pdf,
    plot_fit_pdf,
    plot_fits_over_xmin,
)
from Plotting.standardPowerlaw import (
    EventDrops,
    kappa_detection_threshold,
    kappa_from_relaxation_energy,
    positive_es,
    split_by_kappa,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Plots/current_reversibility_progress/kappa_standard_completed"
CACHE_DIR = OUTPUT_DIR / "fit_cache"
MIN_TAIL_COUNT = 100
COARSE_CANDIDATES = 40
# The global-xmin Clauset bootstrap is expensive; this interim extraction uses
# 100 replicates while preserving global xmin re-evaluation in every replicate.
FIT_CONFIDENCE = 0.05
BOOTSTRAP_WORKERS = min(4, os.cpu_count() or 1)
COARSE_CONFIDENCE = 0.1

SOURCES = {
    "edgeFlip": Path(
        "/Volumes/data/MTS2D_output/"
        "reversibilityProtocolTest,s200x200l1.0,1e-05,5.1PBCedgeFlipt2"
        "epsR0.0LBFGSEpsg0.0LBFGSEpsx1e-06s0/macroData.csv"
    ),
    "delaunay": Path(
        "/Volumes/data/MTS2D_output/"
        "reversibilityProtocolTest,s200x200l1.0,1e-05,5.1PBCdelaunayt2"
        "epsR0.0LBFGSEpsg0.0LBFGSEpsx1e-06s0/macroData.csv"
    ),
}


def read_complete_csv(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    last_newline = raw.rfind(b"\n")
    if last_newline < 0:
        raise ValueError(f"No complete CSV line found in {path}.")
    frame = pd.read_csv(
        io.BytesIO(raw[: last_newline + 1]),
        names=RECONNECT_REVERSIBILITY_MACRODATA_HEADER,
        skiprows=1,
        low_memory=False,
    )
    return update_df_header(frame, add_total_columns=False)


def remove_corrupt_duplicate_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    load_step = np.asarray(frame["load_step"], dtype=float)
    duplicate_positions = np.flatnonzero(np.diff(load_step) == 0)
    removed = []
    for position in duplicate_positions:
        if position in {item["row"] for item in removed}:
            continue
        if load_step[position] != load_step[position + 1]:
            raise RuntimeError("Duplicate load-step detection is inconsistent.")
        rows = [position, position + 1]
        energies = np.abs(np.asarray(frame["total_energy"].iloc[rows], dtype=float))
        if not np.all(np.isfinite(energies)) or np.max(energies) < 1.0e10 * max(np.min(energies), 1.0):
            raise RuntimeError(
                "Found a duplicate load step without an unambiguous corrupt-energy row: "
                f"rows={rows}, energies={energies.tolist()}"
            )
        bad = rows[int(np.argmax(energies))]
        removed.append(
            {
                "row": int(bad),
                "load_step": int(load_step[bad]),
                "load": float(frame["load"].iloc[bad]),
                "total_energy": float(frame["total_energy"].iloc[bad]),
            }
        )
    if not removed:
        return frame.reset_index(drop=True), removed
    cleaned = frame.drop(index=[item["row"] for item in removed]).reset_index(drop=True)
    clean_steps = np.asarray(cleaned["load_step"], dtype=float)
    clean_load = np.asarray(cleaned["load"], dtype=float)
    if not np.all(np.diff(clean_steps) > 0) or not np.all(np.diff(clean_load) > 0):
        raise RuntimeError("Removing corrupt duplicate rows did not restore monotone loads.")
    return cleaned, removed


def extract_events(path: Path) -> tuple[EventDrops, dict]:
    frame = read_complete_csv(path)
    original_rows = len(frame)
    frame, removed_rows = remove_corrupt_duplicate_rows(frame)
    metadata = get_metadata(str(path))
    step_data, info = calculate_energy_step_data(
        str(path), df=frame, metadata=metadata, average_energy=False
    )
    volume = float(info["reference_volume"])
    delta_er = -np.asarray(frame["total_e_change_from_init"].iloc[1:], dtype=float) / volume
    delta_es = np.asarray(step_data["stress_corrected_drop_second_order"], dtype=float) / volume
    delta_gamma = np.asarray(step_data["delta_gamma"], dtype=float)
    if not (delta_er.shape == delta_es.shape == delta_gamma.shape):
        raise RuntimeError("Delta E_R, Delta E_S, and Delta gamma are not aligned.")
    if np.any(~np.isfinite(delta_gamma)) or np.any(delta_gamma <= 0):
        raise RuntimeError("Completed transitions contain non-positive Delta gamma.")
    kappa = kappa_from_relaxation_energy(delta_er, delta_gamma, volume, rho=1.0)
    drops = EventDrops(er=delta_er, es=delta_es, kappa=kappa)
    details = {
        "original_rows": original_rows,
        "clean_rows": len(frame),
        "removed_corrupt_duplicate_rows": removed_rows,
        "volume": volume,
        "load_start": float(frame["load"].iloc[0]),
        "load_end": float(frame["load"].iloc[-1]),
        "transition_count": int(delta_er.size),
        "post_yield_rule": "all complete transitions after the requested load-1 start are treated as post-yield",
    }
    return drops, details


def diagnostic_fixed_xmin_fit(
    data: np.ndarray, xmin: float, confidence: float, description: str
) -> Fit:
    fit = Fit(
        data=data,
        xmin=float(xmin),
        xmin_distribution=Truncated_Power_Law.name,
        verbose=0,
    )
    fit.evaluate_fit(
        data=data,
        confidence=confidence,
        parallel=False,
        cache_dir=str(CACHE_DIR / "evaluation"),
        tqdmDesc=description,
    )
    return fit


def selected_global_fit(
    data: np.ndarray,
    xmin: float,
    analysis: dict,
    confidence: float,
    description: str,
) -> Fit:
    """Evaluate the selected fit with global-xmin bootstrap refitting only."""
    fit = Fit(
        data=data,
        xmin=float(xmin),
        xmin_distribution=Truncated_Power_Law.name,
        verbose=0,
    )
    fit.xmin_analysis = analysis
    fit.xmin_fitting_results = analysis
    fit.xmin_selection = "global"
    fit.xmin_search_mode = "full"
    fit.evaluate_fit(
        data=data,
        confidence=confidence,
        parallel=True,
        max_workers=BOOTSTRAP_WORKERS,
        cache_dir=str(CACHE_DIR / "evaluation_global_xmin"),
        tqdmDesc=description,
    )
    if getattr(fit, "pvalue_xmin_mode", None) != "global":
        raise RuntimeError(
            "Selected standard fit did not use global-xmin bootstrap refitting."
        )
    return fit


def diagnostic_analysis(analysis: dict) -> dict:
    global_details = analysis["global_search_details"]
    diagnostic = dict(analysis)
    for key in ("xmins", "distances", "param_vals", "valid_fits", "tail_counts"):
        diagnostic[key] = global_details[key]
    return diagnostic


def write_fit_plot(kind: str, data: np.ndarray, fit: Fit, split, details: dict, analysis: dict) -> None:
    distribution = dist_from_fit(fit)
    global_xmin = float(analysis["global_min_xmin"])
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    plot_data_pdf(
        ax,
        data,
        label=rf"irreversible $\Delta E_S$ ($n={data.size}$)",
        color="tab:blue",
        drop_label=r"E_S/V_0",
        drop_sign="positive",
        show_legend=False,
    )
    plot_fit_pdf(
        ax,
        fit,
        color="tab:orange",
        label="truncated power-law MLE",
        drop_label=r"E_S/V_0",
        drop_sign="positive",
        show_legend=False,
        set_title=False,
        x_grid_mode="smooth",
        xmin_only=True,
        linewidth=1.5,
    )
    ax.axvline(
        global_xmin,
        color="0.15",
        linestyle=":",
        linewidth=1.2,
        label=rf"$\Delta E_{{S,\min}}^{{KS}}={global_xmin:.2e}$",
    )
    removed = len(details["removed_corrupt_duplicate_rows"])
    note = (
        rf"$\kappa_{{\det}}={split.kappa_det:.4g}$" + "\n"
        rf"$\alpha={distribution.alpha:.3f}$, $\lambda={distribution.Lambda:.3g}$" + "\n"
        rf"$D={fit.D:.3f}$, $p={fit.p:.3f}$" + "\n"
        rf"$n_{{\rm irrev}}={data.size}$"
    )
    if removed:
        note += f"\n{removed} crash-recovery row(s) excluded in memory"
    ax.text(
        0.04,
        0.04,
        note,
        transform=ax.transAxes,
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 3},
    )
    ax.set_title(rf"{kind}: irreversible $\Delta E_S/V_0$ power-law extraction")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 240})): 
        fig.savefig(OUTPUT_DIR / f"{kind}_irreversible_deltaES_fit.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_diagnostic_plot(kind: str, data: np.ndarray, fit: Fit, analysis: dict, details: dict) -> None:
    coarse_xmins = np.asarray(analysis["xmins"], dtype=float)
    coarse_fits = [
        diagnostic_fixed_xmin_fit(
            data, xmin, COARSE_CONFIDENCE, f"{kind} coarse xmin"
        )
        for xmin in coarse_xmins
    ]
    coarse_fits.append(fit)
    diagnostic = diagnostic_analysis(analysis)
    fig, ax = plt.subplots(figsize=(12.0, 4.2))
    plot_fits_over_xmin(
        coarse_fits,
        best_fit=fit,
        title=rf"{kind}: $\Delta E_S$ KS cutoff diagnostic after $\kappa_{{\det}}$ split",
        data_info={
            "drops": data,
            "drop_label": r"E_S/V_0",
            "xmin_axis_label": r"$\Delta E_{S,\min}/V_0$",
            "xmin_min_tail_count": MIN_TAIL_COUNT,
        },
        xmin_results=diagnostic,
        selected_xmin=None,
        ks_xmin=float(analysis["global_min_xmin"]),
        global_xmin=float(analysis["global_min_xmin"]),
        global_distance=float(analysis["global_min_distance"]),
        ax=ax,
    )
    removed = len(details["removed_corrupt_duplicate_rows"])
    if removed:
        fig.text(0.01, 0.01, f"Analysis-only exclusion: {removed} corrupt crash-recovery duplicate row(s); source CSV unchanged.", fontsize=8)
    fig.tight_layout()
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 240})): 
        fig.savefig(OUTPUT_DIR / f"{kind}_irreversible_deltaES_xmin_diagnostic.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kappa_det = kappa_detection_threshold()
    summary = {
        "workflow": "standard kappa_det=mu/2 protocol",
        "kappa_det": float(kappa_det),
        "selected_fit_pvalue": (
            "Clauset semiparametric bootstrap with exhaustive global-xmin "
            "re-evaluation for every bootstrap sample"
        ),
        "selected_fit_bootstrap_xmin_mode": "global",
        "selected_fit_bootstrap_replicates": max(
            1, int(1 / (4 * FIT_CONFIDENCE**2))
        ),
        "selected_fit_bootstrap_workers": BOOTSTRAP_WORKERS,
        "fits": {},
    }
    for kind, path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        drops, details = extract_events(path)
        split = split_by_kappa(drops, kappa_det)
        es_irrev = positive_es(drops, split.is_irrev)
        if es_irrev.size < MIN_TAIL_COUNT:
            raise RuntimeError(f"Only {es_irrev.size} irreversible Delta E_S values for {kind}.")
        analysis = analyze_xmin(
            es_irrev,
            nr_initial=COARSE_CANDIDATES,
            min_tail_count=MIN_TAIL_COUNT,
            refine=False,
            global_mode="global",
            parallel=False,
            progress=False,
        )
        fit = selected_global_fit(
            es_irrev,
            analysis["global_min_xmin"],
            analysis,
            FIT_CONFIDENCE,
            f"{kind} selected global xmin",
        )
        distribution = dist_from_fit(fit)
        write_fit_plot(kind, es_irrev, fit, split, details, analysis)
        write_diagnostic_plot(kind, es_irrev, fit, analysis, details)
        details.update(
            {
                "kappa_det": float(kappa_det),
                "reversible_event_count": int(split.is_rev.sum()),
                "irreversible_event_count": int(split.is_irrev.sum()),
                "positive_irreversible_delta_E_S_count": int(es_irrev.size),
                "es_xmin_ks": float(analysis["global_min_xmin"]),
                "es_xmin_ks_distance": float(analysis["global_min_distance"]),
                "exhaustive_observed_xmin_candidates": int(
                    analysis["global_search_details"]["fine_candidate_count"]
                ),
                "alpha": float(distribution.alpha),
                "lambda": float(distribution.Lambda),
                "D": float(fit.D),
                "p": float(fit.p),
            }
        )
        summary["fits"][kind] = details
        print(
            f"{kind}: load={details['load_end']:.6g}, "
            f"irrev_ES={es_irrev.size}, xmin={details['es_xmin_ks']:.6g}, "
            f"alpha={details['alpha']:.6g}, lambda={details['lambda']:.6g}, "
            f"D={details['D']:.6g}, p={details['p']:.6g}",
            flush=True,
        )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote plots and summary to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
