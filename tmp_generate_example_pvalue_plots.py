"""Generate irreversible-event p-value-versus-xmin diagnostic plots.

The empirical xmin is the exhaustive observed-candidate global minimum. Two
Clauset bootstrap variants are evaluated on identical deterministic samples:
an exhaustive xmin refit and a 100-point coarse scan with local refinement.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from MTMath.evaluatePowerlawFit import Truncated_Power_Law
from Management.updateCSV import read_macrodata_csv
from Plotting.sizeScalingCollapse import extract_event_pairs
from Plotting.standardPowerlaw import (
    EventDrops,
    kappa_detection_threshold,
    positive_es,
    split_by_kappa,
)
from Plotting.plotPowerLaw import (
    PLOTPATH,
    _resolve_strain_lim,
    find_best_xmin,
    get_energy_drops,
    make_fit,
)


ROOT = Path(__file__).resolve().parent
SYLVAIN_ROOT = Path("/Volumes/data/remoteData/macro")
OUTPUT_SUBDIR = "pvalue_examples/"
# Keep the irreversible-population analysis separate from all-drop and older
# simpleDrop/fixed-xmin results. Caching remains enabled throughout.
XMIN_CACHE = ROOT / ".xmin_values_irreversible_global"
EVAL_CACHE = ROOT / ".eval_cache_irreversible"
EVENT_PAIR_CACHE = ROOT / ".event_pairs_irreversible"
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
            "name": "flowchart_L250_10_samples_post_irreversible",
            "title": r"Irreversible events, $L=250$, 10 samples, post-yield",
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
    if example.get("flowchart", False):
        data_info = {
            "customTitle": example["title"],
            "reference_volume": float(250**2),
        }
        er_parts = []
        es_parts = []
        kappa_parts = []
        for path in example["paths"]:
            frame = read_macrodata_csv(str(path), L=250)
            strain_low, strain_high = _resolve_strain_lim(
                "auto", df=frame, postRegime=True
            )
            paired = extract_event_pairs(
                Path(path),
                250,
                {"post": (float(strain_low), float(strain_high))},
                EVENT_PAIR_CACHE,
            )
            er_parts.append(np.asarray(paired["initial_guess_energy_post"], dtype=float))
            es_parts.append(np.asarray(paired["second_order_post"], dtype=float))
            kappa_parts.append(np.asarray(paired["kappa_post"], dtype=float))
        event_drops = EventDrops(
            er=np.concatenate(er_parts),
            es=np.concatenate(es_parts),
            kappa=np.concatenate(kappa_parts),
        )
        kappa_det = kappa_detection_threshold()
        split = split_by_kappa(event_drops, kappa_det)
        drops = positive_es(event_drops, split.is_irrev)
        data_info["kappa_det"] = kappa_det
        data_info["reversible_event_count"] = int(np.count_nonzero(split.is_rev))
        data_info["irreversible_event_count"] = int(np.count_nonzero(split.is_irrev))
        data_info["fit_population"] = "positive irreversible Delta E_S"
        if drops.size < 100:
            raise RuntimeError(
                f"Only {drops.size} positive irreversible Delta E_S drops were found."
            )
        return drops, data_info

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


def _plot_evaluated_fit(example, drops, data_info, fit, bootstrap_mode):
    plot_data_info = dict(data_info)
    mode_label = (
        "full-global bootstrap"
        if bootstrap_mode == "global"
        else "100-point coarse-local bootstrap"
    )
    plot_data_info["customTitle"] = f"{example['title']}, {mode_label}"
    plot_data_info["drop_label"] = r"E_S"
    plot_data_info["xmin_scale"] = float(data_info["reference_volume"])
    plot_data_info["xmin_axis_label"] = r"$\Delta E_{S,\min}/V_0$"
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
        show_selected_xmin=False,
        show_title=False,
    )
    pdf_path = Path(best_fit.xmin_plot_path)
    if not pdf_path.is_absolute():
        pdf_path = ROOT / pdf_path
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    return pdf_path, _render_png(pdf_path)


def _write_bootstrap_comparison(example, rapid_fit, global_fit):
    rapid_xmins = np.asarray(rapid_fit.bootstrap_xmins, dtype=float)
    global_xmins = np.asarray(global_fit.bootstrap_xmins, dtype=float)
    rapid_distances = np.asarray(rapid_fit.bootstrap_D_values, dtype=float)
    global_distances = np.asarray(global_fit.bootstrap_D_values, dtype=float)
    if not (
        rapid_xmins.shape
        == global_xmins.shape
        == rapid_distances.shape
        == global_distances.shape
    ):
        raise RuntimeError("Bootstrap mode comparison arrays are not aligned.")

    exact_match = rapid_xmins == global_xmins
    xmin_ratio = rapid_xmins / global_xmins
    log10_xmin_ratio = np.log10(xmin_ratio)
    distance_difference = rapid_distances - global_distances
    xmin_fold_difference = np.maximum(xmin_ratio, 1.0 / xmin_ratio)
    distance_fold_difference = np.maximum(
        rapid_distances / global_distances,
        global_distances / rapid_distances,
    )
    mismatched = ~exact_match

    def stats(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return {"mean": None, "std": None, "max_abs": None}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "max_abs": float(np.max(np.abs(values))),
        }

    def fold_stats(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return {"median": None, "mean": None, "max": None}
        return {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }

    summary = {
        "dataset": example["name"],
        "fit_population": "positive irreversible Delta E_S",
        "replicate_count": int(rapid_xmins.size),
        "identical_bootstrap_samples": True,
        "exact_xmin_match_count": int(np.count_nonzero(exact_match)),
        "exact_xmin_match_fraction": float(np.mean(exact_match)),
        "xmin_log10_ratio_all": stats(log10_xmin_ratio),
        "xmin_log10_ratio_when_mismatched": stats(log10_xmin_ratio[mismatched]),
        "xmin_fold_difference_all": fold_stats(xmin_fold_difference),
        "xmin_fold_difference_when_mismatched": fold_stats(
            xmin_fold_difference[mismatched]
        ),
        "D_rapid_minus_global_all": stats(distance_difference),
        "D_rapid_minus_global_when_mismatched": stats(
            distance_difference[mismatched]
        ),
        "D_fold_difference_all": fold_stats(distance_fold_difference),
        "D_fold_difference_when_mismatched": fold_stats(
            distance_fold_difference[mismatched]
        ),
        "D_within_1e-4_fraction": float(
            np.mean(np.abs(distance_difference) <= 1e-4)
        ),
        "D_within_1e-3_fraction": float(
            np.mean(np.abs(distance_difference) <= 1e-3)
        ),
        "rapid_p": float(rapid_fit.p),
        "global_p": float(global_fit.p),
    }

    output_dir = ROOT / PLOTPATH / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{example['name']}_bootstrap_xmin_comparison"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (
            "replicate",
            "rapid_xmin",
            "global_xmin",
            "exact_xmin_match",
            "xmin_ratio",
            "log10_xmin_ratio",
            "xmin_fold_difference",
            "rapid_D",
            "global_D",
            "D_rapid_minus_global",
            "D_fold_difference",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(rapid_xmins.size):
            writer.writerow(
                {
                    "replicate": index,
                    "rapid_xmin": rapid_xmins[index],
                    "global_xmin": global_xmins[index],
                    "exact_xmin_match": bool(exact_match[index]),
                    "xmin_ratio": xmin_ratio[index],
                    "log10_xmin_ratio": log10_xmin_ratio[index],
                    "xmin_fold_difference": xmin_fold_difference[index],
                    "rapid_D": rapid_distances[index],
                    "global_D": global_distances[index],
                    "D_rapid_minus_global": distance_difference[index],
                    "D_fold_difference": distance_fold_difference[index],
                }
            )
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return csv_path, json_path


def _run(example: dict):
    drops, data_info = _drops_and_info(example)
    fit = make_fit(
        drops,
        distType=Truncated_Power_Law,
        use_cache=True,
        cache_dir=str(XMIN_CACHE),
        parallel_xmin=True,
        xmin_search_kwargs={"nr_initial": 100, "min_tail_count": 100},
        xmin_selection="global",
        xmin_search_mode="full",
    )
    if getattr(fit, "xmin_selection", None) != "global":
        raise RuntimeError("The p-value fit is not configured for global xmin selection.")
    outputs = []
    evaluated_fits = {}
    for bootstrap_mode in ("rapidGlobal", "global"):
        evaluated = copy.deepcopy(fit)
        evaluated.evaluate_clauset_pvalue(
            data=drops,
            confidence=0.01,
            parallel=True,
            max_workers=WORKERS,
            use_cache=True,
            cache_dir=str(EVAL_CACHE),
            tqdmDesc=f"{example['name']} {bootstrap_mode} p-value",
            xmin_mode=bootstrap_mode,
        )
        if getattr(evaluated, "pvalue_xmin_mode", None) != bootstrap_mode:
            raise RuntimeError(
                f"The p-value fit did not use {bootstrap_mode} xmin refitting."
            )
        evaluated_fits[bootstrap_mode] = evaluated
        pdf_path, png_path = _plot_evaluated_fit(
            example, drops, data_info, evaluated, bootstrap_mode
        )
        outputs.append((bootstrap_mode, pdf_path, png_path))
        print(
            f"{example['name']} {bootstrap_mode}: n={drops.size}, "
            f"xmin={evaluated.xmin:.8g}, "
            f"tail_n={np.count_nonzero(drops >= evaluated.xmin)}, "
            f"D={evaluated.D:.6g}, p={evaluated.p:.6g}",
            flush=True,
        )
    comparison_paths = _write_bootstrap_comparison(
        example,
        evaluated_fits["rapidGlobal"],
        evaluated_fits["global"],
    )
    return outputs, comparison_paths


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
        outputs, comparison_paths = _run(example)
        for bootstrap_mode, pdf_path, png_path in outputs:
            print(f"{example['name']} {bootstrap_mode}: {pdf_path}")
            if png_path is not None:
                print(f"{example['name']} {bootstrap_mode}: {png_path}")
        for path in comparison_paths:
            print(f"{example['name']} comparison: {path}")


if __name__ == "__main__":
    main()
