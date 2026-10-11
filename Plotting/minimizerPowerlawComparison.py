#!/usr/bin/env python3
"""Compare irreversible Delta E_R power laws between minimizers.

Events are restricted to the post-yield region, classified with
``kappa_det = mu/(2 rho)`` for ``rho=N/V_0=2``, and only finite positive
irreversible Delta E_R values are fitted.  Each minimizer independently uses
the exhaustive observed-candidate global minimum of the KS distance.  The
legacy simpleDrop cutoff is neither selected nor displayed.

The post-yield start is each run's avg_P12 peak +0.01.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from Management.updateCSV import update_df_header
from MTMath.evaluatePowerlawFit import Truncated_Power_Law
from Plotting.plotPowerLaw import (
    MINIMIZER_COLORS,
    dist_from_fit,
    make_fit,
    plot_fitted_population_comparison,
    plot_xmin_d_alpha_comparison,
)
from Plotting.standardPowerlaw import (
    POST_YIELD_RULE, yield_regimes,
    EventDrops,
    kappa_detection_threshold,
    kappa_from_relaxation_energy,
    split_by_kappa,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "Plots/powerLaw/minimizer_irreversible_er"
DEFAULT_SIZE = 200
MIN_TAIL_COUNT = 100
NR_DIAGNOSTIC_XMINS = 160
ALGORITHMS = ("FIRE", "CG", "L-BFGS")


def _seed_number(path: Path) -> int:
    match = re.search(r"s(\d+)(?:/macroData)?\.csv$", str(path))
    if match is None:
        raise ValueError(f"Could not determine seed from {path}")
    return int(match.group(1))


def _available_load_end(path: Path) -> float:
    """Read the final load without loading an entire macrodata file."""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 8192))
        lines = stream.read().decode(errors="ignore").splitlines()
    for line in reversed(lines):
        if line.strip() and not line.lstrip().lower().startswith("#header:"):
            fields = line.split(",")
            if len(fields) < 2:
                break
            return float(fields[1])
    raise ValueError(f"Could not read the final load from {path}")


def _most_complete_per_seed(paths: list[Path]) -> list[Path]:
    """Choose one source per seed by load end, then mtime and file size."""
    selected = {}
    for path in paths:
        seed = _seed_number(path)
        rank = (_available_load_end(path), path.stat().st_mtime_ns, path.stat().st_size)
        previous = selected.get(seed)
        if previous is None or rank > previous[0]:
            selected[seed] = (rank, path)
    return [selected[seed][1] for seed in sorted(selected)]


def default_source_paths() -> dict[str, list[Path]]:
    """Discover the supplied local L=200 file for each minimizer/seed."""
    local = Path("/Volumes/data/MTS2D_output")
    resumed = Path("/Volumes/data/MTS2D_local_resume")
    patterns = {
        "FIRE": [
            (
                local,
                "simpleShear,s200x200l0.15,1e-05,1.0PBCt3minimizerFIRE"
                "LBFGSEpsg1e-05CGEpsg1e-05eps1e-05s*/macroData.csv",
            )
        ],
        "CG": [
            (
                resumed,
                "simpleShear,s200x200l0.15,1e-05,1.0PBCt3minimizerCG"
                "LBFGSEpsg1e-05CGEpsg1e-05eps1e-05s*/macroData.csv",
            ),
        ],
        "L-BFGS": [
            (
                resumed,
                "simpleShear,s200x200l0.15,1e-05,1.0PBCt3"
                "LBFGSEpsg1e-05CGEpsg1e-05eps1e-05s*/macroData.csv",
            ),
        ],
    }
    sources = {}
    for label, searches in patterns.items():
        candidates = [
            path
            for directory, pattern in searches
            for path in directory.glob(pattern)
        ]
        paths = _most_complete_per_seed(candidates)
        if not paths:
            descriptions = [str(directory / pattern) for directory, pattern in searches]
            raise FileNotFoundError(
                f"No source files found for {label}: {descriptions}"
            )
        sources[label] = paths
    return sources


def _read_relaxation_steps(path: Path, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    wanted = {
        "load",
        "avg_init_energy_change",
        "avg_e_change_from_init",
        "total_e_change_from_init",
        "avg_RSS",
        "avg_P12",
        "avg_sigma12",
    }
    frame = pd.read_csv(path, usecols=lambda column: column in wanted, low_memory=False)
    frame = update_df_header(frame, L=size)
    required = {"load", "total_e_change_from_init"}
    missing = required - set(frame)
    if missing:
        raise KeyError(f"Missing required columns {sorted(missing)} in {path}")
    load = frame["load"].to_numpy(dtype=float)
    if load.size < 3 or not np.all(np.isfinite(load)) or np.any(np.diff(load) <= 0):
        raise ValueError(f"Load must be finite and strictly increasing in {path}")

    post_yield_start = yield_regimes(frame)["post"][0]
    if load[-1] <= post_yield_start:
        raise ValueError(f"No data after avg_P12 peak +0.01 in {path}")

    delta_gamma = np.diff(load)
    delta_e_r = -frame["total_e_change_from_init"].to_numpy(dtype=float)[1:]
    step_load = load[1:]
    if not (delta_e_r.shape == delta_gamma.shape == step_load.shape):
        raise RuntimeError(f"Unaligned step arrays in {path}")
    mask = (
        np.isfinite(delta_e_r)
        & (delta_e_r > 0)
        & np.isfinite(delta_gamma)
        & (delta_gamma > 0)
        & (step_load > post_yield_start)
        & (step_load < load[-1])
    )
    delta_e_r = delta_e_r[mask]
    delta_gamma = delta_gamma[mask]
    kappa = kappa_from_relaxation_energy(
        delta_e_r,
        delta_gamma,
        float(size**2),
    )
    return delta_e_r, delta_gamma, kappa, {
        "path": str(path),
        "seed": _seed_number(path),
        "available_load_end": float(load[-1]),
        "complete_to_load_1": bool(load[-1] >= 1.0),
        "post_yield_start": post_yield_start,
        "yield_method": POST_YIELD_RULE,
        "positive_post_yield_delta_E_R_count": int(delta_e_r.size),
    }


def collect_irreversible_delta_er(
    paths: list[Path],
    *,
    size: int,
    kappa_det: float,
) -> tuple[np.ndarray, dict]:
    if not paths:
        raise ValueError("At least one macrodata path is required.")
    er_parts = []
    kappa_parts = []
    runs = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        er, _, kappa, record = _read_relaxation_steps(path, size)
        er_parts.append(er)
        kappa_parts.append(kappa)
        runs.append(record)
    er = np.concatenate(er_parts)
    kappa = np.concatenate(kappa_parts)
    events = EventDrops(er=er, es=np.full(er.shape, np.nan), kappa=kappa)
    split = split_by_kappa(events, kappa_det)
    er_irrev = events.er[split.is_irrev]
    er_irrev = er_irrev[np.isfinite(er_irrev) & (er_irrev > 0)]
    if er_irrev.size < MIN_TAIL_COUNT:
        raise RuntimeError(
            f"Only {er_irrev.size} irreversible Delta E_R values; "
            f"need at least {MIN_TAIL_COUNT}."
        )
    for record, er_run, kappa_run in zip(runs, er_parts, kappa_parts):
        record["irreversible_delta_E_R_count"] = int(
            np.count_nonzero(np.isfinite(kappa_run) & (kappa_run >= kappa_det))
        )
    return er_irrev, {
        "csv_count": len(paths),
        "completed_csv_count": sum(record["complete_to_load_1"] for record in runs),
        "positive_post_yield_delta_E_R_count": int(er.size),
        "irreversible_delta_E_R_count": int(er_irrev.size),
        "runs": runs,
    }


def run_analysis(
    sources: dict[str, list[Path]],
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    size: int = DEFAULT_SIZE,
) -> tuple[Path, Path, Path]:
    if int(size) != size or size <= 0:
        raise ValueError("size must be a positive integer.")
    size = int(size)
    missing_labels = set(ALGORITHMS) - set(sources)
    if missing_labels:
        raise ValueError(f"Missing minimizer sources: {sorted(missing_labels)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kappa_det = kappa_detection_threshold()
    populations = {}
    population_summaries = {}
    fits = {}
    for label in ALGORITHMS:
        print(f"[{label}] collecting {len(sources[label])} available runs", flush=True)
        population, summary = collect_irreversible_delta_er(
            sources[label],
            size=size,
            kappa_det=kappa_det,
        )
        populations[label] = population
        population_summaries[label] = summary
        print(
            f"[{label}] exhaustive global KS search over "
            f"{population.size} irreversible events",
            flush=True,
        )
        fits[label] = make_fit(
            population,
            distType=Truncated_Power_Law,
            use_cache=True,
            cache_dir=str(output_dir / "xmin_cache" / label.replace("-", "")),
            parallel_xmin=False,
            xmin_selection="global",
            xmin_search_mode="full",
            xmin_search_kwargs={
                "nr_initial": NR_DIAGNOSTIC_XMINS,
                "min_tail_count": MIN_TAIL_COUNT,
                "refine": True,
                "progress": False,
            },
        )
        if getattr(fits[label], "xmin_selection", None) != "global":
            raise RuntimeError(f"{label} did not use global xmin selection.")

    colors = {
        "FIRE": MINIMIZER_COLORS["FIRE"],
        "CG": MINIMIZER_COLORS["CG"],
        "L-BFGS": MINIMIZER_COLORS["L-BFGS"],
    }
    pdf_path = output_dir / f"L{size}_irreversible_ER_pdf_minimizer_comparison.pdf"
    figure, _ = plot_fitted_population_comparison(
        fits,
        colors=colors,
        drop_label=r"E_R",
    )
    figure.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(figure)

    scan_path = output_dir / f"L{size}_irreversible_ER_D_alpha_vs_Emin.pdf"
    figure, _ = plot_xmin_d_alpha_comparison(
        fits,
        colors=colors,
    )
    figure.savefig(scan_path, format="pdf", bbox_inches="tight")
    plt.close(figure)

    fit_summaries = {}
    for label, fit in fits.items():
        dist = dist_from_fit(fit)
        fit_summaries[label] = {
            "delta_E_R_xmin_ks": float(fit.xmin),
            "ks_distance": float(fit.D),
            "alpha": float(dist.alpha),
            "lambda": float(dist.Lambda),
            "tail_count": int(np.count_nonzero(populations[label] >= fit.xmin)),
            "xmin_selection": "exhaustive observed-candidate global KS minimum",
        }
    summary_path = output_dir / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "size": size,
                "reference_volume": float(size**2),
                "rho": 2.0,
                "kappa_det": float(kappa_det),
                "classification": (
                    "post-yield events classified by kappa; finite positive "
                    "irreversible Delta E_R fitted"
                ),
                "yield_criterion": POST_YIELD_RULE,
                "simple_drop_used": False,
                "populations": population_summaries,
                "fits": fit_summaries,
                "outputs": [str(pdf_path), str(scan_path)],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return pdf_path, scan_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in run_analysis(
        default_source_paths(),
        output_dir=args.output_dir,
        size=DEFAULT_SIZE,
    ):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
