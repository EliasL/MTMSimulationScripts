"""Generate representative p-value-versus-xmin diagnostic plots.

This deliberately uses the same ``make_fit``/``find_best_xmin`` path as the
earlier Sylvain p-value run.  It is a small example-only driver rather than a
replacement for the all-settings batch script.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from MTMath.evaluatePowerlawFit import Truncated_Power_Law
from Plotting.plotPowerLaw import (
    PLOTPATH,
    find_best_xmin,
    get_energy_drops,
    make_fit,
)


ROOT = Path(__file__).resolve().parent
SYLVAIN_ROOT = Path("/Volumes/data/remoteData/macro")
OUTPUT_SUBDIR = "pvalue_examples/"
XMIN_CACHE = ROOT / ".xmin_values"
EVAL_CACHE = ROOT / ".eval_cache"
WORKERS = min(8, os.cpu_count() or 1)


def _render_png(pdf_path: Path) -> Path | None:
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        return None
    png_path = pdf_path.with_suffix(".png")
    subprocess.run(
        [
            pdftoppm,
            "-png",
            "-r",
            "160",
            "-singlefile",
            str(pdf_path),
            str(png_path.with_suffix("")),
        ],
        check=True,
    )
    return png_path


def _sylvain_path(load_increment: str, eps_x: str, seed: int) -> Path:
    path = SYLVAIN_ROOT / (
        "reversibilityProtocolTest,s100x100l0.14,"
        f"{load_increment},1.0PBCt3LBFGSEpsx{eps_x}"
        f"energyDropThreshold1e-05s{seed}.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _examples() -> list[dict]:
    return [
        {
            "name": "sylvain_batch-2_epsx-1e-6_post",
            "title": r"Sylvain batch -2, $\epsilon_x=10^{-6}$, post-yield",
            "paths": [_sylvain_path("1e-05", "1e-06", seed) for seed in range(4)],
            "post_regime": True,
        },
        {
            "name": "sylvain_batch-1_deltagamma-5e-6_post",
            "title": r"Sylvain batch -1, $\Delta\gamma=5\times10^{-6}$, post-yield",
            "paths": [_sylvain_path("5e-06", "1e-06", seed) for seed in range(4)],
            "post_regime": True,
        },
        {
            "name": "flowchart_L250_10_samples_post",
            "title": r"Flowchart data, $L=250$, 10 samples, post-yield",
            "paths": sorted(
                (ROOT / "Plots/powerLaw/truncated_powerlaw_flowchart/data").glob(
                    "*_fixed.csv"
                )
            ),
            "post_regime": True,
            "flowchart": True,
        },
        {
            "name": "reconnecting_L200_gammaT-5.1_post",
            "title": r"Reconnecting data, $L=200$, $\gamma_T=5.1$, post-yield",
            "paths": [
                Path(
                    "/Volumes/data/MTS2D_output/"
                    "simpleShear,s200x200l0.15,1e-05,5.1PBCedgeFlipt5"
                    "epsR1e-05LBFGSEpsg1e-08LBFGSEpsx1e-06s0/macroData.csv"
                )
            ],
            "post_regime": True,
            "reconnecting_200": True,
        },
        {
            "name": "nonreconnecting_L500_post",
            "title": r"Non-reconnecting data, $L=500$, post-yield",
            "paths": [
                Path(
                    "/Volumes/data/MTS2D_output/"
                    "simpleShear,s500x500l0.138,2e-05,1.0PBCt8"
                    "initialGuessNoise0.04LBFGSEpsx1e-05s0/macroData.csv"
                )
            ],
            "post_regime": True,
            "large_500": True,
        },
    ]


def _drops_and_info(example: dict):
    paths = [str(path) for path in example["paths"]]
    if not paths:
        raise RuntimeError(f"No CSV paths found for {example['name']}")
    kwargs = {
        "strainLim": "auto",
        "debug": False,
        "postRegime": example["post_regime"],
        "averageEnergy": False,
        "stress_corrected": True,
        "stress_correction_order": 2,
        "stress_tangent": "current",
        "min_drop": 0.0,
    }
    drops, data_info = get_energy_drops(paths, **kwargs)
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > 0.0)]
    if drops.size < 25:
        raise RuntimeError(f"Only {drops.size} positive drops found for {example['name']}")
    data_info["customTitle"] = example["title"]
    return drops, data_info


def _run(example: dict) -> tuple[Path, Path | None]:
    drops, data_info = _drops_and_info(example)
    plot_data_info = dict(data_info)
    plot_data_info["drop_label"] = r"E"
    plot_data_info["xmin_scale"] = float(data_info["reference_volume"])
    plot_data_info["xmin_axis_label"] = r"$\Delta E_{\min}/V_0$"
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
        tqdmDesc=f"{example['name']} final p-value",
    )
    output_dir = ROOT / PLOTPATH / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    best_fit = find_best_xmin(
        drops,
        nr_evaluation=20,
        min_p=0.1,
        start_accuracy=0.1,
        max_accuracy=0.01,
        DistType=Truncated_Power_Law,
        data_info=plot_data_info,
        selected_fit=fit,
        parallel=True,
        max_workers=WORKERS,
        use_memmap=True,
        extraPath=OUTPUT_SUBDIR,
    )
    print(
        f"{example['name']}: n={drops.size}, xmin={fit.xmin:.8g}, "
        f"tail_n={np.count_nonzero(drops >= fit.xmin)}, D={fit.D:.6g}, "
        f"p={fit.p:.6g}, p_local_max_found="
        f"{best_fit.p_value_local_max_found}"
    )
    pdf_path = Path(best_fit.xmin_plot_path)
    if not pdf_path.is_absolute():
        pdf_path = ROOT / pdf_path
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    return pdf_path, _render_png(pdf_path)


def main() -> None:
    os.chdir(ROOT)
    for example in _examples():
        # The two Sylvain examples already exist in the earlier complete
        # p-value output directory; regenerating those scans is redundant.
        if not (
            example.get("flowchart", False)
            or example.get("reconnecting_200", False)
            or example.get("large_500", False)
        ):
            continue
        pdf_path, png_path = _run(example)
        print(f"{example['name']}: {pdf_path}")
        if png_path is not None:
            print(f"{example['name']}: {png_path}")


if __name__ == "__main__":
    main()
