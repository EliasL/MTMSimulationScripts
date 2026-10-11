"""Regenerate post-yield power-law figures using avg_P12 peak +0.01, largest sizes first.

Run from the repository root. Completed stages are recorded in
``Plots/powerLaw/piola_peak_post_only_redraw/status.json`` and skipped on rerun.
Each stage writes its own log; failures are recorded and reported at the end.
Old figures and caches stay in their original directories.
Use ``--second-order-error-bars`` to run only the post-yield second-order
parameter/error-bar plots, reusing the completed empirical cutoff fits.
Use ``--size-scaling-redraw --skip-pvalue`` for splits, collapse and parameter
plots without uncertainty bootstraps.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from Plotting.standardPowerlaw import POST_YIELD_RULE


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "Plots/powerLaw/piola_peak_post_only_redraw"
PYTHON = sys.executable
MACRO = Path("/Volumes/data/remoteData/macro")
RECONNECTING = Path("/Volumes/data/MTS2D_output/sizeScalingJobs")


def stages() -> list[tuple[str, list[str]]]:
    nonreconnecting_100 = sorted(MACRO.glob(
        "simpleShear,s100x100l0.15,1e-05,1.0PBCt3LBFGSEpsx1e-06s[0-9].csv"
    ))
    reconnecting_100 = sorted(RECONNECTING.glob(
        "simpleShear,s100x100l0.15,1e-05,1.0PBCedgeFlipt3LBFGSEpsx1e-06s[0-9]/macroData.csv"
    ))
    commands = [
        ("size_scaling_classification", [PYTHON, "-m", "Plotting.sizeScalingERClassification",
            "--data-root", str(MACRO), "--yield-regime", "post", "--largest-first",
            "--output-dir",
            "Plots/reversible_event_analysis/size_scaling_E_R_classification_piola_peak_post_only"]),
        ("reconnecting_size_scaling", [PYTHON, "tmp_generate_reconnecting_size_scaling_powerlaw_pdfs.py",
            "--data-root", str(RECONNECTING), "--largest-first", "--output-dir",
            "output/pdf/reconnecting_size_scaling/flowchart_panels_piola_peak_post_only"]),
        ("size_scaling_collapse", [PYTHON, "-m", "Plotting.sizeScalingCollapse",
            "--data-root", str(MACRO), "--yield-regime", "post", "--largest-first",
            "--output", "Plots/powerLaw/size_collapse_piola_peak_post_only"]),
        ("sigma_rescue", [PYTHON, "rebuild_sigma_rescue_gamma0.py"]),
        ("minimizer_comparison", [PYTHON, "-m", "Plotting.minimizerPowerlawComparison",
            "--output-dir", "Plots/powerLaw/minimizer_irreversible_er_piola_peak"]),
        ("reversibility_flowchart", [PYTHON, "-m", "Plotting.truncated_powerlaw_reversibility_flowchart",
            "--regenerate-subplots"]),
        ("legacy_flowchart", [PYTHON, "-m", "Plotting.truncated_powerlaw_flowchart",
            "--regenerate-subplots"]),
        ("kappa_event_diagnostic", [PYTHON, "-m", "Plotting.kappaEventClassification",
            "--output", "output/pdf/kappa_event_classification_diagnostics_piola_peak.pdf"]),
        ("kappa_size_diagnostic", [PYTHON, "-m", "Plotting.kappaSizeScalingClassification",
            "--data-root", str(MACRO), "--largest-first", "--size-summary",
            "Plots/reversible_event_analysis/size_scaling_E_R_classification_piola_peak_post_only/classification_summary.csv",
            "--output", "output/pdf/kappa_event_classification_by_size_piola_peak_post_only.pdf"]),
        ("real_space_xmin_atlas", [PYTHON, "-m", "Plotting.real_space_events.xmin_atlas",
            "--yield-regime", "post", "--output-dir",
            "output/pdf/post_yield_irreversible_xmin_event_atlas_piola_peak"]),
    ]
    if nonreconnecting_100 and reconnecting_100:
        commands.append(("reconnection_comparison", [PYTHON, "-m",
            "Plotting.plot_size_scaling_irreversible_delta_es_comparison",
            "--reconnecting-csv", *map(str, reconnecting_100),
            "--non-reconnecting-csv", *map(str, nonreconnecting_100),
            "--output", "Plots/powerLaw/irreversible_delta_es_comparison_piola_peak.pdf",
            "--cache-dir", "Plots/powerLaw/irreversible_delta_es_comparison_piola_peak_cache"] ))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print the planned stages and exit.")
    parser.add_argument("--second-order-error-bars", action="store_true",
                        help="Run only post-yield second-order parameter uncertainties.")
    parser.add_argument("--size-scaling-redraw", action="store_true",
                        help="Redraw splits and provisional size-scaling plots, then second-order uncertainties.")
    parser.add_argument("--skip-pvalue", action="store_true",
                        help="With --size-scaling-redraw, omit the uncertainty stage entirely.")
    args = parser.parse_args()
    if args.skip_pvalue and (not args.size_scaling_redraw or args.second_order_error_bars):
        parser.error("--skip-pvalue requires --size-scaling-redraw without --second-order-error-bars")
    commands = stages()
    if args.second_order_error_bars or args.size_scaling_redraw:
        planned = dict(commands)
        command = planned["size_scaling_collapse"]
        cache_option = ["--standard-cache-root",
                        "Plots/reversible_event_analysis/size_scaling_E_R_classification_piola_peak_post_only/cache"]
        error_bars = ("second_order_error_bars", command + [
            "--stage", "parameters", "--protocols", "second_order",
            "--parameter-xmin-methods", "global_min", *cache_option])
        commands = ([
            ("size_scaling_split", planned["size_scaling_classification"] + ["--split-only"]),
            ("size_scaling_quick", command + [*cache_option, "--skip-pvalue"]),
        ] if args.size_scaling_redraw else []) + ([] if args.skip_pvalue else [error_bars])
    if args.list:
        for name, command in commands:
            print(name, " ".join(map(str, command)))
        return
    WORK.mkdir(parents=True, exist_ok=True)
    status_path = WORK / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status and status.get("post_yield_rule") != POST_YIELD_RULE:
        raise ValueError(f"Existing status has another yield rule: {status_path}")
    if status and status.get("analysis_scope") != "post-only; largest-first":
        raise ValueError(f"Existing status has another analysis scope: {status_path}")
    status["post_yield_rule"] = POST_YIELD_RULE
    status["analysis_scope"] = "post-only; largest-first"
    status["bootstrap_requested"] = not args.skip_pvalue
    status["requested_stages"] = [name for name, _ in commands]
    status.setdefault("stages", {})
    sizes = [100, 150, 200, 250]
    if status.get("size_scaling_sizes") != sizes:
        for name, state in status["stages"].items():
            if state == "complete":
                status["stages"][name] = "needs redraw with current sizes"
    status["size_scaling_sizes"] = sizes
    if status.get("energy_drop_population") != "positive irreversible Delta E_S":
        for name in ("second_order_error_bars", "size_scaling_collapse"):
            status["stages"][name] = "needs standard-population rerun"
    status["energy_drop_population"] = "positive irreversible Delta E_S"
    selection = "all available post-yield samples, including partial runs"
    if status.get("sample_selection") != selection:
        for name in ("second_order_error_bars", "size_scaling_collapse", "size_scaling_classification",
                     "size_scaling_split", "size_scaling_quick"):
            status["stages"][name] = "needs redraw with all available samples"
    status["sample_selection"] = selection
    environment = os.environ.copy()
    environment.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(ROOT / ".matplotlib-cache"),
                       PYTHONPATH=str(ROOT))
    for name, command in commands:
        if status["stages"].get(name) == "complete":
            print(f"Skipping completed stage: {name}", flush=True)
            continue
        print(f"Running {name}; log: {WORK / (name + '.log')}", flush=True)
        with (WORK / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        status["stages"][name] = "complete" if result.returncode == 0 else f"failed ({result.returncode})"
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)
        print(f"{name}: {status['stages'][name]}", flush=True)
    failed = {name: status["stages"][name] for name, _ in commands
              if status["stages"][name] != "complete"}
    if failed:
        raise RuntimeError(f"Power-law redraw stages failed: {failed}")


if __name__ == "__main__":
    main()
