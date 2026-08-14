#!/usr/bin/env python3
"""Fit irreversible energy-drop tails for available reconnecting size jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Management.reconnectionJobSelection import discover_simulation_jobs
from Management.updateCSV import read_macrodata_csv
from MTMath.evaluatePowerlawFit import Fit, POWERLAW_STANDARD_WORKFLOW, Truncated_Power_Law
from Plotting.energyDropCalculations import calculate_energy_step_data
from Plotting.findXmin import analyze_xmin
from Plotting.plotPowerLaw import dist_from_fit, plot_data_pdf, plot_fit_pdf


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/Volumes/data/MTS2D_output/sizeScalingJobs")
OUTPUT_DIR = ROOT / "output/pdf/reconnecting_size_scaling"
CACHE_DIR = OUTPUT_DIR / "cache"
POST_YIELD_LOW = 0.7
POST_YIELD_HIGH = 1.0
MIN_TAIL_COUNT = 100
CONFIDENCE = 0.1


def _cache_path(path: Path, size: int) -> Path:
    stat = path.stat()
    signature = f"{path}:{stat.st_size}:{stat.st_mtime_ns}:{size}:aligned-events-v2"
    return CACHE_DIR / "runs" / f"{hashlib.sha1(signature.encode()).hexdigest()}.npz"


def extract_aligned_events(path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    cache_path = _cache_path(path, size)
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return cached["e_r"], cached["e_s"]

    df = read_macrodata_csv(path)
    required = {"load", "total_e_change_from_init"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing columns {missing} in {path}")

    energy_steps, _ = calculate_energy_step_data(
        str(path), df=df, metadata={"L": size}, average_energy=False
    )
    if len(energy_steps) != len(df) - 1:
        raise RuntimeError(f"Step-data length mismatch in {path}")

    load = np.asarray(df["load"], dtype=float)
    event_mask = (
        (load[1:] > POST_YIELD_LOW)
        & (load[1:] < POST_YIELD_HIGH)
    )
    volume = float(size**2)
    e_r = -np.asarray(df["total_e_change_from_init"].iloc[1:], dtype=float) / volume
    e_s = (
        np.asarray(energy_steps["stress_corrected_drop_second_order"], dtype=float)
        / volume
    )
    e_r = e_r[event_mask]
    e_s = e_s[event_mask]
    if e_r.shape != e_s.shape:
        raise RuntimeError(f"Unaligned Delta E_R/Delta E_S arrays in {path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, e_r=e_r, e_s=e_s)
    return e_r, e_s


def positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def fit_fixed_xmin(data: np.ndarray, xmin: float, description: str) -> tuple[Fit, dict]:
    data = positive(data)
    fit = Fit(
        data=data,
        xmin=float(xmin),
        xmin_distribution=Truncated_Power_Law.name,
        verbose=0,
    )
    fit.evaluate_fit(
        data=data,
        confidence=CONFIDENCE,
        parallel=False,
        cache_dir=str(CACHE_DIR / "evaluation"),
        tqdmDesc=description,
    )
    dist = dist_from_fit(fit)
    result = {
        "xmin": float(xmin),
        "tail_count": int(np.count_nonzero(data >= xmin)),
        "alpha": float(dist.alpha),
        "alpha_std": float(getattr(fit, "alpha_std", np.nan)),
        "Lambda": float(dist.Lambda),
        "Lambda_std": float(getattr(fit, "Lambda_std", np.nan)),
        "D": float(fit.D),
        "p": float(getattr(fit, "p", np.nan)),
        "p_std": float(getattr(fit, "p_std", np.nan)),
    }
    if not np.isfinite(result["alpha"]) or not np.isfinite(result["Lambda"]):
        raise RuntimeError(f"Invalid fit parameters for {description}: {result}")
    return fit, result


def compact_xmin_plot(ax, analysis: dict, title: str, color: str) -> None:
    xmins = np.asarray(analysis["xmins"], dtype=float)
    distances = np.asarray(analysis["distances"], dtype=float)
    tail_counts = np.asarray(analysis["tail_counts"], dtype=int)
    eligible = (
        np.isfinite(xmins)
        & np.isfinite(distances)
        & (tail_counts >= MIN_TAIL_COUNT)
    )
    ax.plot(xmins, distances, color="0.65", linewidth=0.8)
    ax.scatter(xmins[eligible], distances[eligible], s=9, color=color, label="eligible")
    ax.axvline(
        analysis["simple_drop_xmin"],
        color="tab:blue",
        linestyle="--",
        linewidth=1,
        label=rf"simpleDrop={analysis['simple_drop_xmin']:.1e}",
    )
    ax.axvline(
        analysis["global_min_xmin"],
        color="0.2",
        linestyle=":",
        linewidth=1.2,
        label=rf"global={analysis['global_min_xmin']:.1e}",
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 0.5)
    ax.set_title(title, fontsize="small")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=6, frameon=False, loc="best")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = discover_simulation_jobs(DATA_ROOT, job_type="size-scaling")
    by_size = {}
    for job in jobs:
        by_size.setdefault(job.size, []).append(job)

    coverage = []
    fit_objects = []
    diagnostics = []
    records = []
    for size in sorted(by_size):
        size_jobs = sorted(by_size[size], key=lambda job: job.seed)
        pooled_er = []
        pooled_es = []
        post_jobs = 0
        max_loads = []
        for job in size_jobs:
            path = job.folder / "macroData.csv"
            if not path.exists():
                continue
            df = read_macrodata_csv(path, update_header=False)
            last_load = float(np.asarray(df["load"], dtype=float)[-1])
            max_loads.append(last_load)
            e_r, e_s = extract_aligned_events(path, size)
            if e_r.size:
                post_jobs += 1
                pooled_er.append(e_r)
                pooled_es.append(e_s)

        coverage.append(
            {
                "size": size,
                "existing_jobs": len(size_jobs),
                "post_yield_jobs_with_data": post_jobs,
                "max_last_load": max(max_loads) if max_loads else np.nan,
            }
        )
        if not pooled_er:
            continue

        e_r = np.concatenate(pooled_er)
        e_s = np.concatenate(pooled_es)
        positive_er = positive(e_r)
        if positive_er.size < MIN_TAIL_COUNT:
            continue
        er_analysis = analyze_xmin(
            positive_er,
            nr_initial=100,
            min_tail_count=MIN_TAIL_COUNT,
            refine=True,
            parallel=False,
            progress=False,
        )
        er_xmin = float(er_analysis["simple_drop_xmin"])
        irreversible = np.isfinite(e_r) & (e_r > 0) & (e_r >= er_xmin)
        irreversible_es = positive(e_s[irreversible])
        if irreversible_es.size < MIN_TAIL_COUNT:
            continue
        es_analysis = analyze_xmin(
            irreversible_es,
            nr_initial=100,
            min_tail_count=MIN_TAIL_COUNT,
            refine=True,
            parallel=False,
            progress=False,
        )
        es_xmin = float(es_analysis["global_min_xmin"])
        fit, fit_values = fit_fixed_xmin(
            irreversible_es,
            es_xmin,
            f"reconnecting size scaling L={size}",
        )
        record = {
            "size": size,
            "existing_jobs": len(size_jobs),
            "post_yield_jobs_with_data": post_jobs,
            "aligned_post_yield_events": int(e_r.size),
            "positive_delta_E_R_events": int(positive_er.size),
            "delta_E_R_simpleDrop_xmin": er_xmin,
            "irreversible_event_count": int(np.count_nonzero(irreversible)),
            "positive_irreversible_delta_E_S_events": int(irreversible_es.size),
            "delta_E_S_global_xmin": es_xmin,
            **fit_values,
        }
        records.append(record)
        fit_objects.append((fit, irreversible_es, record))
        diagnostics.append((er_analysis, es_analysis, record))
        print(
            f"L={size}: jobs={len(size_jobs)}, post_jobs={post_jobs}, "
            f"events={e_r.size}, irreversible_ES={irreversible_es.size}, "
            f"xmin={es_xmin:.6g}, alpha={fit_values['alpha']:.5g}, "
            f"Lambda={fit_values['Lambda']:.5g}, p={fit_values['p']:.4g}",
            flush=True,
        )

    if not fit_objects:
        raise RuntimeError("No size-scaling setting has enough post-yield edge-flip data.")

    ncols = len(fit_objects)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.4), squeeze=False)
    axes = axes.ravel()
    for ax, (fit, data, record) in zip(axes, fit_objects):
        plot_data_pdf(
            ax,
            data,
            label=r"irreversible $\Delta E_S$ data",
            color="tab:blue",
            drop_label=r"E_S/V_0",
            drop_sign="positive",
            show_legend=False,
        )
        plot_fit_pdf(
            ax,
            fit,
            color="tab:orange",
            label="truncated power-law fit",
            drop_label=r"E_S/V_0",
            drop_sign="positive",
            show_legend=False,
            set_title=False,
            x_grid_mode="smooth",
            xmin_only=True,
            linewidth=1.4,
        )
        ax.axvline(record["delta_E_S_global_xmin"], color="0.2", linestyle=":", linewidth=1)
        ax.set_title(rf"$L={record['size']}$", fontsize="medium")
        ax.text(
            0.04,
            0.04,
            rf"$\alpha={record['alpha']:.3f}\pm{record['alpha_std']:.3f}$" + "\n"
            rf"$\lambda={record['Lambda']:.2e}\pm{record['Lambda_std']:.2e}$" + "\n"
            rf"$x_{{\min}}={record['delta_E_S_global_xmin']:.2e}$, $n={record['tail_count']}$" + "\n"
            rf"$D={record['D']:.3f}$, $p={record['p']:.3f}$",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2},
        )
        ax.grid(alpha=0.18)
    handles = [
        Line2D([], [], marker="o", linestyle="None", color="tab:blue", label=r"irreversible $\Delta E_S$ data"),
        Line2D([], [], color="tab:orange", label="truncated power-law fit"),
        Line2D([], [], color="0.2", linestyle=":", label=r"global $\Delta E_{S,\min}$"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
        fontsize="small",
    )
    fig.suptitle(
        r"Reconnecting size-scaling jobs: irreversible $\Delta E_S/V_0$ PDFs",
        y=0.995,
    )
    fig.subplots_adjust(top=0.82, wspace=0.28, left=0.08, right=0.99, bottom=0.12)
    fit_pdf = OUTPUT_DIR / "reconnecting_size_scaling_deltaES_irreversible_pdf_fits.pdf"
    fig.savefig(fit_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 7.0), squeeze=False)
    for column, (er_analysis, es_analysis, record) in enumerate(diagnostics):
        compact_xmin_plot(axes[0, column], er_analysis, r"$\Delta E_R$ cutoff", "tab:blue")
        compact_xmin_plot(axes[1, column], es_analysis, r"irreversible $\Delta E_S$ cutoff", "tab:orange")
        axes[0, column].set_title(rf"$L={record['size']}$", fontsize="medium")
        axes[1, column].set_xlabel(r"candidate $x_{\min}$")
        axes[0, column].set_ylabel("KS distance")
        axes[1, column].set_ylabel("KS distance")
    fig.suptitle("Reconnecting size-scaling xmin diagnostics", y=0.995)
    fig.subplots_adjust(top=0.92, hspace=0.38, wspace=0.30, left=0.08, right=0.99, bottom=0.10)
    diagnostic_pdf = OUTPUT_DIR / "reconnecting_size_scaling_xmin_diagnostics.pdf"
    fig.savefig(diagnostic_pdf, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(coverage).to_csv(OUTPUT_DIR / "reconnecting_size_scaling_data_coverage.csv", index=False)
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "reconnecting_size_scaling_fit_results.csv", index=False)
    summary = {
        "data_root": str(DATA_ROOT),
        "reconnection": "edgeFlip",
        "post_yield_range": [POST_YIELD_LOW, POST_YIELD_HIGH],
        "event_selection": (
            "aligned post-yield macro transitions, without an nr_edge_flips filter. "
            "Reversible/irreversible classification is made by simpleDrop on "
            "aligned Delta E_R."
        ),
        "workflow": POWERLAW_STANDARD_WORKFLOW,
        "coverage": coverage,
        "fits": records,
    }
    (OUTPUT_DIR / "reconnecting_size_scaling_fit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {fit_pdf}", flush=True)
    print(f"wrote {diagnostic_pdf}", flush=True)


if __name__ == "__main__":
    main()
