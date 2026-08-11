"""Generate corrected stress-drop PDF, p-value, and D-vs-cutoff plots."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from MTMath.evaluatePowerlawFit import Truncated_Power_Law
from Plotting.findXmin import plot_xmin_analysis
from Plotting.plotPowerLaw import (
    PLOTPATH,
    find_best_xmin,
    get_stress_drops,
    make_fit,
    plot_data_and_fit,
)
from tmp_generate_example_pvalue_plots import (
    EVAL_CACHE,
    ROOT,
    WORKERS,
    XMIN_CACHE,
    _examples,
    _render_png,
)


OUTPUT_DIR = ROOT / PLOTPATH / "pvalue_examples"
STRESS_PVALUE_DIR = "pvalue_examples_stress/"


def _known_examples():
    examples = [
        example
        for example in _examples()
        if (
            example.get("flowchart")
            or example.get("reconnecting_200")
            or example.get("large_500")
        )
    ]
    if len(examples) != 3:
        raise RuntimeError(f"Expected three known examples, found {len(examples)}")
    return examples


def _fit_stress(example):
    paths = [str(path) for path in example["paths"]]
    drops, data_info = get_stress_drops(
        paths,
        strainLim="auto",
        postRegime=True,
        stress_type="stress_corrected",
        label=example["title"],
    )
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > 0.0)]
    if drops.size < 25:
        raise RuntimeError(f"Only {drops.size} stress drops found for {example['name']}")
    data_info["customTitle"] = example["title"]
    fit = make_fit(
        drops,
        distType=Truncated_Power_Law,
        use_cache=True,
        cache_dir=str(XMIN_CACHE),
        parallel_xmin=True,
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
        tqdmDesc=f"{example['name']} stress p-value",
    )
    return drops, data_info, fit


def _plot_stress_pdf(ax, example, drops, data_info, fit):
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
        label=rf"simpleDrop $\Delta\sigma_{{S,\min}}={fit.xmin:.3g}$",
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
    ax.set_xlabel(r"$\Delta\sigma_S$")
    ax.set_ylabel(r"$p(\Delta\sigma_S)$")
    ax.legend(loc="upper right", ncol=1, fontsize=8)
    ax.grid(False, which="both")


def _plot_D_panels(energy_fits, stress_fits):
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 8.5))
    for col, (example, energy_fit, stress_fit) in enumerate(
        zip(_known_examples(), energy_fits, stress_fits)
    ):
        plot_xmin_analysis(energy_fit.xmin_analysis, ax=axes[0, col])
        axes[0, col].set_xlabel(r"$\Delta E_{\min}/V_0$")
        axes[0, col].set_ylabel(r"$D$")
        if example.get("flowchart"):
            axes[0, col].set_title(r"Energy drops: $L=250$")
        elif example.get("reconnecting_200"):
            axes[0, col].set_title(r"Energy drops: $L=200$ reconnecting")
        else:
            axes[0, col].set_title(r"Energy drops: $L=500$")
        plot_xmin_analysis(stress_fit.xmin_analysis, ax=axes[1, col])
        axes[1, col].set_xlabel(r"$\Delta\sigma_{S,\min}$")
        axes[1, col].set_ylabel(r"$D$")
        if example.get("flowchart"):
            axes[1, col].set_title(r"Stress drops: $L=250$")
        elif example.get("reconnecting_200"):
            axes[1, col].set_title(r"Stress drops: $L=200$ reconnecting")
        else:
            axes[1, col].set_title(r"Stress drops: $L=500$")
    fig.tight_layout()
    return fig


def main() -> None:
    os.chdir(ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / PLOTPATH / STRESS_PVALUE_DIR).mkdir(parents=True, exist_ok=True)
    examples = _known_examples()

    energy_fits = []
    for example in examples:
        from tmp_generate_example_pvalue_plots import _drops_and_info

        drops, data_info = _drops_and_info(example)
        fit = make_fit(
            drops,
            distType=Truncated_Power_Law,
            use_cache=True,
            cache_dir=str(XMIN_CACHE),
            parallel_xmin=False,
            xmin_search_kwargs={"nr_initial": 100, "min_tail_count": 25},
        )
        if fit.xmin_analysis is not None:
            fit.xmin_analysis.setdefault("data_max", float(np.max(drops)))
            fit.xmin_analysis.setdefault(
                "tail_valid_max",
                float(np.sort(drops)[-fit.xmin_analysis["min_tail_count"]]),
            )
            fit.xmin_analysis["xmin_scale"] = float(data_info["reference_volume"])
            fit.xmin_analysis["xmin_axis_label"] = r"$\Delta E_{\min}/V_0$"
        energy_fits.append(fit)

    stress_results = [_fit_stress(example) for example in examples]
    stress_fits = [result[2] for result in stress_results]

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8))
    for ax, example, result in zip(axes, examples, stress_results):
        _plot_stress_pdf(ax, example, *result)
    fig.tight_layout()
    stress_pdf = OUTPUT_DIR / "simpledrop_stress_pdf_fits_L200_L250_L500.pdf"
    stress_png = OUTPUT_DIR / "simpledrop_stress_pdf_fits_L200_L250_L500.png"
    fig.savefig(stress_pdf, bbox_inches="tight")
    fig.savefig(stress_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    stress_pvalue_paths = []
    for example, (stress_drops, data_info, fit) in zip(examples, stress_results):
        best_fit = find_best_xmin(
            np.asarray(stress_drops),
            nr_evaluation=20,
            min_p=0.1,
            start_accuracy=0.1,
            max_accuracy=0.01,
            DistType=Truncated_Power_Law,
            data_info=data_info,
            selected_fit=fit,
            parallel=True,
            max_workers=WORKERS,
            use_memmap=True,
            extraPath=STRESS_PVALUE_DIR,
        )
        stress_pvalue_paths.append(Path(best_fit.xmin_plot_path))
        _render_png(Path(best_fit.xmin_plot_path))

    D_fig = _plot_D_panels(energy_fits, stress_fits)
    D_pdf = OUTPUT_DIR / "simpledrop_D_vs_cutoff_L200_L250_L500.pdf"
    D_png = OUTPUT_DIR / "simpledrop_D_vs_cutoff_L200_L250_L500.png"
    D_fig.savefig(D_pdf, bbox_inches="tight")
    D_fig.savefig(D_png, dpi=220, bbox_inches="tight")
    plt.close(D_fig)

    summary = {}
    for example, energy_fit, stress_fit in zip(examples, energy_fits, stress_fits):
        energy_p = float(energy_fit.p) if hasattr(energy_fit, "p") else None
        energy_summary = {
            "xmin": float(energy_fit.xmin),
            "D": float(energy_fit.D),
        }
        if energy_p is not None:
            energy_summary["p"] = energy_p
        summary[example["name"]] = {
            "energy": energy_summary,
            "stress_corrected": {
                "xmin": float(stress_fit.xmin),
                "D": float(stress_fit.D),
                "p": float(stress_fit.p),
            },
        }
    summary_path = OUTPUT_DIR / "simpledrop_energy_stress_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(stress_pdf)
    print(stress_png)
    print(D_pdf)
    print(D_png)
    for path in stress_pvalue_paths:
        print(path)
    print(summary_path)


if __name__ == "__main__":
    main()
