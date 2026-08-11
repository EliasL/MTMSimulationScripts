"""Run fixed-xmin bootstrap p-value scans for all Sylvain analyses.

The canonical xmin is selected by ``make_fit`` (the exhaustive simpleDrop
selection).  The bootstrap p-value is then evaluated at that fixed xmin, and
``find_best_xmin`` is used only to produce the diagnostic p-value-vs-xmin
plot.  It does not replace the canonical xmin selection.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import numpy as np

from Plotting.plotPowerLaw import (
    PLOTPATH,
    find_best_xmin,
    get_energy_drops,
    make_fit,
)
from MTMath.evaluatePowerlawFit import Truncated_Power_Law


DATA_ROOT = Path("/Volumes/data/remoteData/macro")
PLOT_DIR = ROOT / PLOTPATH / "sylvain_pvalues"
STATUS_DIR = ROOT / PLOTPATH / "sylvain_pvalue_status"
XMIN_CACHE = ROOT / ".xmin_values"
EVAL_CACHE = ROOT / ".eval_cache"
WORKERS = 2
FINAL_CONFIDENCE = 0.01
ROUGH_CONFIDENCE = 0.1


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_id(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def _csv_path(load_increment: str, eps_x: str, seed: int) -> Path:
    filename = (
        "reversibilityProtocolTest,s100x100l0.14,"
        f"{load_increment},1.0PBCt3LBFGSEpsx{eps_x}"
        f"energyDropThreshold1e-05s{seed}.csv"
    )
    path = DATA_ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _analysis_specs() -> list[dict]:
    specs = []

    # In batch -2, loadIncrement is fixed and LBFGS epsx varies.
    for eps_x in ("0.0001", "1e-05", "1e-06", "1e-07"):
        specs.append({"batch": -2, "load_increment": "1e-05", "eps_x": eps_x})

    # In batch -1, epsx is fixed and loadIncrement varies.
    for load_increment in ("0.0001", "5e-05", "1e-05", "5e-06", "1e-06"):
        specs.append({"batch": -1, "load_increment": load_increment, "eps_x": "1e-06"})

    analyses = []
    for spec in specs:
        for post_regime in (True, False):
            regime = "post" if post_regime else "pre"
            label = (
                f"batch={spec['batch']}, loadIncrement={spec['load_increment']}, "
                f"LBFGSEpsx={spec['eps_x']}"
            )
            analysis_id = _safe_id(f"batch{spec['batch']}_{regime}_{spec['load_increment']}_{spec['eps_x']}")
            paths = [
                _csv_path(spec["load_increment"], spec["eps_x"], seed)
                for seed in range(4)
            ]
            analyses.append(
                {
                    **spec,
                    "id": analysis_id,
                    "regime": regime,
                    "post_regime": post_regime,
                    "label": label,
                    "paths": paths,
                }
            )
    if len(analyses) != 18:
        raise RuntimeError(f"Expected 18 Sylvain analyses, found {len(analyses)}")
    return analyses


def _render_png(pdf_path: Path) -> Path | None:
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        print("pdftoppm not found; retaining PDF only", flush=True)
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


def _run_analysis(spec: dict) -> dict:
    analysis_id = spec["id"]
    paths = [str(path) for path in spec["paths"]]
    labels = [f"{spec['label']}, seed={seed}" for seed in range(4)]

    drops, data_info = get_energy_drops(
        paths,
        strainLim="auto",
        debug=False,
        label=labels,
        postRegime=spec["post_regime"],
    )
    drops = np.asarray(drops, dtype=float)
    if drops.ndim != 1 or drops.size == 0:
        raise ValueError(f"{analysis_id}: expected a non-empty 1-D drop array")
    if not np.all(np.isfinite(drops)) or not np.all(drops > 0):
        raise ValueError(f"{analysis_id}: drops contain non-finite or non-positive values")

    fit = make_fit(
        data=drops,
        distType=Truncated_Power_Law,
        use_cache=True,
        cache_dir=str(XMIN_CACHE),
        parallel_xmin=False,
        xmin_search_kwargs={"nr_initial": 100, "min_tail_count": 100},
    )

    parallel_fallback = False
    try:
        fit.evaluate_fit(
            data=drops,
            confidence=FINAL_CONFIDENCE,
            parallel=True,
            max_workers=WORKERS,
            use_cache=True,
            cache_dir=str(EVAL_CACHE),
            max_synthetic_samples=5e6,
            tqdmDesc=f"{analysis_id} final xmin bootstrap",
        )
    except Exception as parallel_error:
        parallel_fallback = True
        print(
            f"{analysis_id}: parallel final bootstrap failed; retrying serial: "
            f"{parallel_error!r}",
            flush=True,
        )
        fit.evaluate_fit(
            data=drops,
            confidence=FINAL_CONFIDENCE,
            parallel=False,
            use_cache=True,
            cache_dir=str(EVAL_CACHE),
            max_synthetic_samples=5e6,
            tqdmDesc=f"{analysis_id} final xmin bootstrap (serial fallback)",
        )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    best_fit = find_best_xmin(
        drops,
        nr_evaluation=20,
        start_accuracy=ROUGH_CONFIDENCE,
        max_accuracy=FINAL_CONFIDENCE,
        DistType=Truncated_Power_Law,
        data_info=data_info,
        selected_fit=fit,
        parallel=True,
        max_workers=WORKERS,
        use_memmap=True,
        extraPath="sylvain_pvalues/",
    )

    pdf_path = Path(best_fit.xmin_plot_path)
    if not pdf_path.is_absolute():
        pdf_path = ROOT / pdf_path
    if not pdf_path.is_file():
        raise FileNotFoundError(f"p-value plot was not created: {pdf_path}")
    png_path = _render_png(pdf_path)

    details = getattr(fit, "xmin_analysis", {}) or {}
    simple_details = details.get("simple_drop_details", {}) or {}
    result = {
        "status": "complete",
        "id": analysis_id,
        "batch": spec["batch"],
        "regime": spec["regime"],
        "load_increment": spec["load_increment"],
        "eps_x": spec["eps_x"],
        "paths": paths,
        "total_drop_count": int(drops.size),
        "selected_xmin": float(fit.xmin),
        "selected_tail_count": int(np.count_nonzero(drops >= fit.xmin)),
        "selected_ks_distance": float(fit.D),
        "final_p_value": float(fit.p),
        "final_p_value_accuracy": FINAL_CONFIDENCE,
        "final_bootstrap_sets": int(1 / (4 * FINAL_CONFIDENCE**2)),
        "final_parallel_requested": True,
        "final_max_workers": WORKERS,
        "final_parallel_fallback_to_serial": parallel_fallback,
        "p_scan_best_xmin": float(best_fit.xmin),
        "p_scan_best_p_value": float(best_fit.p),
        "p_scan_rough_bootstrap_sets": int(1 / (4 * ROUGH_CONFIDENCE**2)),
        "p_scan_refined_bootstrap_sets": int(1 / (4 * (ROUGH_CONFIDENCE / 2) ** 2)),
        "p_scan_pdf": str(pdf_path),
        "p_scan_png": str(png_path) if png_path is not None else None,
        "simple_drop_region_candidate_count": int(
            len(simple_details.get("region_xmins", []))
        ),
        "simple_drop_region_best_xmin": float(
            simple_details["region_best_xmin"]
        ),
        "simple_drop_local_minimum": float(simple_details["local_minimum"]),
        "completed_at": _timestamp(),
    }
    return result


def main() -> int:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    specs = _analysis_specs()
    summary_path = STATUS_DIR / "summary.json"
    statuses = {}

    for index, spec in enumerate(specs, start=1):
        status_path = STATUS_DIR / f"{spec['id']}.json"
        if status_path.is_file():
            try:
                previous = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
            previous_plot = previous.get("p_scan_pdf")
            if (
                previous.get("status") == "complete"
                and previous_plot
                and Path(previous_plot).is_file()
            ):
                statuses[spec["id"]] = previous
                print(
                    f"[{index}/18] SKIP {spec['id']} (already complete)",
                    flush=True,
                )
                continue

        running = {
            "status": "running",
            "id": spec["id"],
            "batch": spec["batch"],
            "regime": spec["regime"],
            "load_increment": spec["load_increment"],
            "eps_x": spec["eps_x"],
            "started_at": _timestamp(),
        }
        _write_json(status_path, running)
        statuses[spec["id"]] = running
        _write_json(
            summary_path,
            {
                "status": "running",
                "updated_at": _timestamp(),
                "completed": sum(s.get("status") == "complete" for s in statuses.values()),
                "errors": sum(s.get("status") == "error" for s in statuses.values()),
                "total": 18,
                "analyses": statuses,
            },
        )

        print(f"[{index}/18] START {spec['id']}", flush=True)
        try:
            result = _run_analysis(spec)
        except Exception as error:
            result = {
                **running,
                "status": "error",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "completed_at": _timestamp(),
            }
            print(f"[{index}/18] ERROR {spec['id']}: {error!r}", flush=True)
        else:
            print(
                f"[{index}/18] DONE {spec['id']}: "
                f"xmin={result['selected_xmin']:.6g}, p={result['final_p_value']:.4g}",
                flush=True,
            )

        statuses[spec["id"]] = result
        _write_json(status_path, result)
        _write_json(
            summary_path,
            {
                "status": "running",
                "updated_at": _timestamp(),
                "completed": sum(s.get("status") == "complete" for s in statuses.values()),
                "errors": sum(s.get("status") == "error" for s in statuses.values()),
                "total": 18,
                "analyses": statuses,
            },
        )

    errors = [status for status in statuses.values() if status.get("status") == "error"]
    final_status = "complete" if not errors and len(statuses) == 18 else "error"
    _write_json(
        summary_path,
        {
            "status": final_status,
            "updated_at": _timestamp(),
            "completed": sum(s.get("status") == "complete" for s in statuses.values()),
            "errors": len(errors),
            "total": 18,
            "analyses": statuses,
        },
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
