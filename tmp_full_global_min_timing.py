"""Time the exhaustive observed-xmin search on the flowchart dataset."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from MTMath.evaluatePowerlawFit import Truncated_Power_Law, evaluate_xmin_distances
from Plotting.findXmin import analyze_xmin
from Plotting.plotPowerLaw import get_energy_drops
import Plotting.truncated_powerlaw_flowchart as flow


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    paths = tuple(sorted(flow.DOWNLOAD_CACHE_DIR.glob("*_fixed.csv")))
    if not paths:
        raise RuntimeError(f"No repaired flowchart CSVs found in {flow.DOWNLOAD_CACHE_DIR}")

    result_path = Path("Plots/powerLaw/truncated_powerlaw_flowchart/full_global_min_timing.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "started_at": timestamp(),
        "dataset": "flowchart repaired L=250 CSVs",
        "csv_count": len(paths),
        "parallel": False,
        "status": "loading_drops",
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    drops, _ = get_energy_drops(
        [str(path) for path in paths],
        strainLim=flow.STRAIN_LIMIT,
        debug=False,
        postRegime=True,
        averageEnergy=flow.AVERAGE_ENERGY,
        stress_corrected=True,
        stress_correction_order=2,
        stress_tangent="current",
        min_drop=flow.MIN_DROP,
    )
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > flow.MIN_DROP)]
    if drops.size < flow.MIN_TAIL_COUNT:
        raise RuntimeError(f"Only {drops.size} valid drops were found.")

    candidate_max = float(np.sort(drops)[-flow.MIN_TAIL_COUNT])
    candidates = np.unique(drops[drops <= candidate_max])
    if candidates.size < 2:
        raise RuntimeError("Fewer than two exhaustive xmin candidates were found.")

    result.update(
        {
            "drop_count": int(drops.size),
            "candidate_count": int(candidates.size),
            "candidate_max": candidate_max,
            "status": "running_exhaustive_global_min",
            "exhaustive_started_at": timestamp(),
        }
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    exhaustive_start = time.perf_counter()
    distances, _, valid = evaluate_xmin_distances(
        drops,
        candidates,
        distType=Truncated_Power_Law,
        parallel=False,
    )
    exhaustive_seconds = time.perf_counter() - exhaustive_start
    finite = np.isfinite(distances)
    if not finite.any():
        raise RuntimeError("The exhaustive search produced no finite KS distances.")
    best_index = int(np.flatnonzero(finite)[np.argmin(distances[finite])])

    result.update(
        {
            "exhaustive_seconds": exhaustive_seconds,
            "exhaustive_finished_at": timestamp(),
            "global_min_xmin": float(candidates[best_index]),
            "global_min_distance": float(distances[best_index]),
            "finite_fit_count": int(np.count_nonzero(valid)),
            "status": "running_simpleDrop_comparison",
        }
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    simple_start = time.perf_counter()
    simple = analyze_xmin(
        drops,
        nr_initial=flow.XMIN_CANDIDATE_COUNT,
        min_tail_count=flow.MIN_TAIL_COUNT,
        distType=Truncated_Power_Law,
        parallel=False,
    )
    simple_seconds = time.perf_counter() - simple_start
    result.update(
        {
            "simpleDrop_seconds": simple_seconds,
            "simpleDrop_xmin": float(simple["simple_drop_xmin"]),
            "simpleDrop_distance": float(simple["simple_drop_distance"]),
            "speedup": exhaustive_seconds / simple_seconds,
            "seconds_saved": exhaustive_seconds - simple_seconds,
            "status": "complete",
            "completed_at": timestamp(),
        }
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
