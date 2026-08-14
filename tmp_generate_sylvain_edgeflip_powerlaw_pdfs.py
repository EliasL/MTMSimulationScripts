#!/usr/bin/env python3
"""Fit irreversible edge-flip energy drops for the Sylvain batch jobs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from Management.updateCSV import read_macrodata_csv
from MTMath.evaluatePowerlawFit import Fit, POWERLAW_STANDARD_WORKFLOW, Truncated_Power_Law
from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import calculate_energy_step_data, volume_from_metadata
from Plotting.findXmin import analyze_xmin
from Plotting.plotPowerLaw import dist_from_fit, plot_data_pdf, plot_fit_pdf


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/Volumes/data/remoteData/macro")
OUTPUT_DIR = ROOT / "output/pdf/sylvain_edgeflip_reconnecting"
CACHE_DIR = OUTPUT_DIR / "cache"

# The central configuration is present in both Sylvain batch listings but is
# one simulation group and is deliberately plotted once.
SPECS = (
    ("batch -1", "1e-06", "1e-06"),
    ("batch -1", "5e-06", "1e-06"),
    ("batches -1/-2", "1e-05", "1e-06"),
    ("batch -1", "5e-05", "1e-06"),
    ("batch -1", "0.0001", "1e-06"),
    ("batch -2", "1e-05", "1e-05"),
    ("batch -2", "1e-05", "0.0001"),
)
SEEDS = range(4)
MIN_TAIL_COUNT = 100
CONFIDENCE = 0.1


def csv_path(load_increment: str, eps_x: str, seed: int) -> Path:
    return DATA_ROOT / (
        "reversibilityProtocolTest,s100x100l0.14,"
        f"{load_increment},1.0PBCedgeFlipt3LBFGSEpsx{eps_x}"
        f"energyDropThreshold1e-05s{seed}.csv"
    )


def extract_edge_flip_events(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = read_macrodata_csv(path)
    required = {"load", "avg_sigma12", "total_e_change_from_init", "nr_edge_flips"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing columns {missing} in {path}")

    metadata = get_metadata(str(path))
    volume = volume_from_metadata(metadata)
    if volume is None or not np.isfinite(volume) or volume <= 0:
        raise ValueError(f"Could not infer a positive volume from {path}")
    volume = float(volume)
    if not np.isclose(volume, 10000.0):
        raise ValueError(f"Unexpected volume {volume} in {path}")

    energy_steps, _ = calculate_energy_step_data(
        str(path), df=df, metadata=metadata, average_energy=False
    )
    if len(energy_steps) != len(df) - 1:
        raise RuntimeError(f"Step-data length mismatch in {path}")

    load = np.asarray(df["load"], dtype=float)
    stress = np.asarray(df["avg_sigma12"], dtype=float)
    yield_load = float(load[int(np.nanargmax(stress))])
    event_rows = (
        (load[1:] > yield_load)
        & (np.asarray(df["nr_edge_flips"].iloc[1:], dtype=float) > 0)
    )
    e_r = -np.asarray(df["total_e_change_from_init"].iloc[1:], dtype=float) / volume
    e_s = (
        np.asarray(energy_steps["stress_corrected_drop_second_order"], dtype=float)
        / volume
    )
    e_r = e_r[event_rows]
    e_s = e_s[event_rows]
    if e_r.shape != e_s.shape:
        raise RuntimeError(f"Unaligned Delta E_R/Delta E_S arrays in {path}")
    return e_r, e_s


def pooled_events(load_increment: str, eps_x: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    e_r_values = []
    e_s_values = []
    sources = []
    for seed in SEEDS:
        path = csv_path(load_increment, eps_x, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        e_r, e_s = extract_edge_flip_events(path)
        e_r_values.append(e_r)
        e_s_values.append(e_s)
        sources.append(str(path))
    e_r = np.concatenate(e_r_values)
    e_s = np.concatenate(e_s_values)
    if e_r.size < MIN_TAIL_COUNT or e_r.shape != e_s.shape:
        raise RuntimeError(
            f"Unexpected pooled event data for load_increment={load_increment}, "
            f"eps_x={eps_x}: shapes={e_r.shape}, {e_s.shape}"
        )
    return e_r, e_s, sources


def positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def fit_result(data: np.ndarray, xmin: float, description: str) -> tuple[Fit, dict]:
    data = positive(data)
    fit = Fit(data=data, xmin=float(xmin), xmin_distribution=Truncated_Power_Law.name, verbose=0)
    fit.evaluate_fit(
        data=data,
        confidence=CONFIDENCE,
        parallel=False,
        cache_dir=str(CACHE_DIR),
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
    eligible = np.isfinite(xmins) & np.isfinite(distances) & (tail_counts >= MIN_TAIL_COUNT)
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    fit_objects = []
    xmin_analyses = []

    for batch, load_increment, eps_x in SPECS:
        e_r, e_s, sources = pooled_events(load_increment, eps_x)
        positive_er = positive(e_r)
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
            raise RuntimeError(
                f"Too few irreversible Delta E_S events for {load_increment}, {eps_x}: "
                f"{irreversible_es.size}"
            )

        es_analysis = analyze_xmin(
            irreversible_es,
            nr_initial=100,
            min_tail_count=MIN_TAIL_COUNT,
            refine=True,
            parallel=False,
            progress=False,
        )
        es_xmin = float(es_analysis["global_min_xmin"])
        fit, fit_values = fit_result(
            irreversible_es,
            es_xmin,
            f"Delta E_S {load_increment} eps_x={eps_x}",
        )
        record = {
            "batch": batch,
            "load_increment": load_increment,
            "eps_x": eps_x,
            "seed_count": len(sources),
            "aligned_edge_flip_events": int(e_r.size),
            "positive_delta_E_R_events": int(positive_er.size),
            "delta_E_R_simpleDrop_xmin": er_xmin,
            "irreversible_event_count": int(np.count_nonzero(irreversible)),
            "positive_irreversible_delta_E_S_events": int(irreversible_es.size),
            "delta_E_S_global_xmin": es_xmin,
            **fit_values,
            "source_paths": sources,
        }
        records.append(record)
        fit_objects.append((fit, irreversible_es, record))
        xmin_analyses.append((er_analysis, es_analysis, record))
        print(
            f"{batch}: dg={load_increment}, eps_x={eps_x}; "
            f"n={irreversible_es.size}, xmin={es_xmin:.6g}, "
            f"alpha={fit_values['alpha']:.5g}, Lambda={fit_values['Lambda']:.5g}, "
            f"D={fit_values['D']:.4g}, p={fit_values['p']:.4g}",
            flush=True,
        )

    n_panels = len(records)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 6.1), squeeze=False)
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
            label=r"truncated power-law fit",
            drop_label=r"E_S/V_0",
            drop_sign="positive",
            show_legend=False,
            set_title=False,
            x_grid_mode="smooth",
            xmin_only=True,
            linewidth=1.4,
        )
        ax.axvline(record["delta_E_S_global_xmin"], color="0.2", linestyle=":", linewidth=1)
        ax.set_title(
            rf"{record['batch']}: $\Delta\gamma={float(record['load_increment']):.0e}$, "
            rf"$\epsilon_x={float(record['eps_x']):.0e}$",
            fontsize="small",
        )
        ax.text(
            0.04,
            0.04,
            rf"$\alpha={record['alpha']:.3f}\pm{record['alpha_std']:.3f}$" + "\n"
            rf"$\lambda={record['Lambda']:.2e}\pm{record['Lambda_std']:.2e}$" + "\n"
            rf"$x_{{\min}}={record['delta_E_S_global_xmin']:.2e}$, $n={record['tail_count']}$" + "\n"
            rf"$D={record['D']:.3f}$, $p={record['p']:.3f}$",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2},
        )
        ax.grid(alpha=0.18)
    for ax in axes[n_panels:]:
        ax.remove()
    handles = [
        plt.Line2D([], [], marker="o", linestyle="None", color="tab:blue", label=r"irreversible $\Delta E_S$ data"),
        plt.Line2D([], [], color="tab:orange", label="truncated power-law fit"),
        plt.Line2D([], [], color="0.2", linestyle=":", label=r"global $\Delta E_{S,\min}$"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
        fontsize="small",
    )
    fig.suptitle(r"Sylvain reconnecting edge-flip jobs: irreversible $\Delta E_S/V_0$ PDFs", y=0.995)
    fig.subplots_adjust(top=0.84, hspace=0.42, wspace=0.28, left=0.06, right=0.99, bottom=0.08)
    fit_pdf = OUTPUT_DIR / "sylvain_edgeflip_reconnecting_deltaES_irreversible_pdf_fits.pdf"
    fig.savefig(fit_pdf, bbox_inches="tight")
    plt.close(fig)

    diagnostic_blocks = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        2 * diagnostic_blocks,
        ncols,
        figsize=(12.5, 5.2 * diagnostic_blocks),
        squeeze=False,
    )
    for column, (er_analysis, es_analysis, record) in enumerate(xmin_analyses):
        block = column // ncols
        col = column % ncols
        er_row = 2 * block
        es_row = er_row + 1
        compact_xmin_plot(axes[er_row, col], er_analysis, r"$\Delta E_R$ cutoff", "tab:blue")
        compact_xmin_plot(axes[es_row, col], es_analysis, r"irreversible $\Delta E_S$ cutoff", "tab:orange")
        axes[er_row, col].set_title(
            rf"{record['batch']}: $\Delta\gamma={float(record['load_increment']):.0e}$, "
            rf"$\epsilon_x={float(record['eps_x']):.0e}$",
            fontsize="small",
        )
        axes[es_row, col].set_xlabel(r"candidate $x_{\min}$")
        axes[er_row, col].set_ylabel("KS distance")
        axes[es_row, col].set_ylabel("KS distance")
    for group in range(n_panels, diagnostic_blocks * ncols):
        block = group // ncols
        col = group % ncols
        axes[2 * block, col].remove()
        axes[2 * block + 1, col].remove()
    fig.suptitle("Sylvain edge-flip xmin diagnostics", y=0.995)
    fig.subplots_adjust(top=0.96, hspace=0.42, wspace=0.32, left=0.06, right=0.99, bottom=0.06)
    diagnostic_pdf = OUTPUT_DIR / "sylvain_edgeflip_reconnecting_xmin_diagnostics.pdf"
    fig.savefig(diagnostic_pdf, bbox_inches="tight")
    plt.close(fig)

    result_df = pd.DataFrame(
        [{key: value for key, value in record.items() if key != "source_paths"} for record in records]
    )
    result_df.to_csv(OUTPUT_DIR / "sylvain_edgeflip_reconnecting_fit_results.csv", index=False)
    metadata = {
        "workflow": POWERLAW_STANDARD_WORKFLOW,
        "event_selection": (
            "candidate events are post-yield macro rows with nr_edge_flips > 0; "
            "this only selects edge-flip rows and does not assign reversibility. "
            "Reversible/irreversible classification is made separately by applying "
            "simpleDrop to Delta E_R, then transferring that label to aligned "
            "Delta E_S; these CSV-only exports do not include the reversibilityData "
            "event directories"
        ),
        "energy_scale": "Delta E_R=-total_e_change_from_init/V0; Delta E_S=stress_corrected_drop_second_order/V0",
        "V0": 10000.0,
        "confidence": CONFIDENCE,
        "results": records,
    }
    (OUTPUT_DIR / "sylvain_edgeflip_reconnecting_fit_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {fit_pdf}", flush=True)
    print(f"wrote {diagnostic_pdf}", flush=True)


if __name__ == "__main__":
    main()
