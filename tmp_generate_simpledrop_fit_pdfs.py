"""Plot empirical PDFs and truncated-power-law fits at simpleDrop cutoffs."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from MTMath.evaluatePowerlawFit import Truncated_Power_Law
from Plotting.plotPowerLaw import make_fit, plot_data_and_fit
from tmp_generate_example_pvalue_plots import (
    EVAL_CACHE,
    ROOT,
    WORKERS,
    XMIN_CACHE,
    _drops_and_info,
    _examples,
)


OUTPUT_DIR = ROOT / "Plots/powerLaw/pvalue_examples"


def _analyses():
    analyses = []
    for example in _examples():
        if not (
            example.get("flowchart")
            or example.get("reconnecting_200")
            or example.get("large_500")
        ):
            continue
        drops, data_info = _drops_and_info(example)
        fit = make_fit(
            drops,
            distType=Truncated_Power_Law,
            use_cache=True,
            cache_dir=str(XMIN_CACHE),
            parallel_xmin=False,
            xmin_search_kwargs={"nr_initial": 100, "min_tail_count": 25},
        )
        fit.evaluate_fit(
            data=drops,
            confidence=0.01,
            parallel=True,
            max_workers=WORKERS,
            use_cache=True,
            cache_dir=str(EVAL_CACHE),
            max_synthetic_samples=5e6,
            tqdmDesc=f"{example['name']} p-value",
        )
        analyses.append((example, drops, data_info, fit))
    if len(analyses) != 3:
        raise RuntimeError(f"Expected three simpleDrop analyses, found {len(analyses)}")
    return analyses


def _plot_panel(ax, example, drops, data_info, fit):
    plot_data_and_fit(
        fit,
        ax=ax,
        data_info=data_info,
        color="C1",
        data_color="C0",
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
    ax.lines[0].set_label(f"Empirical PDF ({drops.size:,} drops)")
    ax.lines[0].set_markersize(4.0)
    ax.lines[1].set_label("Truncated power-law fit")
    ax.axvline(
        fit.xmin,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=rf"simpleDrop $\Delta E_{{\min}}={fit.xmin:.3g}$",
    )
    tail_count = int(np.count_nonzero(drops >= fit.xmin))
    ax.text(
        0.03,
        0.04,
        rf"$n_{{\rm tail}}={tail_count:,}$" "\n"
        rf"$D={fit.D:.4f}$, $p={fit.p:.4f}$",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    if example.get("flowchart"):
        ax.set_title(r"Non-reconnecting, $L=250$, 10 samples")
    elif example.get("reconnecting_200"):
        ax.set_title(r"Reconnecting, $L=200$, $\gamma_T=5.1$")
    else:
        ax.set_title(r"Non-reconnecting, $L=500$")
    ax.set_xlabel(r"$\Delta E$")
    ax.set_ylabel(r"$p(\Delta E)$")
    ax.legend(loc="upper right", ncol=1, fontsize=8)
    ax.grid(False, which="both")


def main() -> None:
    os.chdir(ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analyses = _analyses()

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8))
    for ax, analysis in zip(axes, analyses):
        _plot_panel(ax, *analysis)
    fig.tight_layout()

    pdf_path = OUTPUT_DIR / "simpledrop_pdf_fits_L200_L250_L500.pdf"
    png_path = OUTPUT_DIR / "simpledrop_pdf_fits_L200_L250_L500.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
