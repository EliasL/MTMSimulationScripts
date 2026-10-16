"""Rebuild current sigma-rescue power-law figures from the frozen gamma=0 snapshot.

Run from the repository root with ``.venv/bin/python -u rebuild_sigma_rescue_gamma0.py``.
Per-size fits currently use 100 full-global refits for 95% percentile
intervals. Set ``BOOTSTRAP_SAMPLES`` to 1000 for the final-data run. Reruns
reuse the matching cache. Every stage fails loudly and is logged by the calling
shell.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))



SNAPSHOT = ROOT / "sigma_rescue_interim/current_gamma0_piola_peak"
OUTPUT = ROOT / "Plots/powerLaw/sigma_rescue_size_scaling_piola_peak"
BOOTSTRAP_SAMPLES = 100


def bootstrap_confidence(samples: int) -> float:
    """Map the evaluator's confidence argument to the requested sample count."""
    if samples < 1:
        raise ValueError("Bootstrap sample count must be positive.")
    # The evaluator truncates 1 / (4 confidence**2) to an integer.
    return (1.0 / (4.0 * samples)) ** 0.5 * (1.0 - 1e-9)


def stage(*args: str) -> None:
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(ROOT / ".matplotlib-cache"),
               PYTHONPATH=str(ROOT))
    command = [sys.executable, *args]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    from Management.sigmaRescueDropSnapshot import build_piola_yield_view
    build_piola_yield_view(ROOT / "sigma_rescue_interim/current_gamma0", SNAPSHOT)
    stage("-m", "Plotting.sigmaRescueSizeScalingPowerlawDiagnostics",
          "--snapshot-root", str(SNAPSHOT), "--pvalue-confidence",
          str(bootstrap_confidence(BOOTSTRAP_SAMPLES)),
          "--bootstrap-xmin-mode", "global", "--observed-xmin-mode", "global",
          "--pvalue-workers", "4", "--resume", "--largest-first")
    stage("-m", "Plotting.plotSigmaRescueSizeScaling", "--snapshot-root", str(SNAPSHOT))
    stage("-m", "Plotting.plotSigmaRescueEsExponentSizeScaling",
          "--snapshot-root", str(SNAPSHOT), "--output-dir",
          str(ROOT / "Plots/powerLaw/sigma_rescue_es_exponent_size_scaling_piola_peak"),
          "--force", "--largest-first")
    stage("tmp_sigma_rescue_finite_size_scaling.py", "--snapshot-root", str(SNAPSHOT),
          "--force", "--largest-first")
    expected = [OUTPUT / f"{name}.pdf"
                for name in ("alpha", "lambda", "event_coverage", "xmin")]
    for path in expected:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing generated PDF: {path}")
    print("COMPLETE: all gamma=0 sigma-rescue PDFs generated", flush=True)


if __name__ == "__main__":
    main()
