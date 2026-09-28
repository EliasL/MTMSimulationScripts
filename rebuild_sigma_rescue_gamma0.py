"""Rebuild current sigma-rescue power-law figures from the frozen gamma=0 snapshot.

Run from the repository root with ``.venv/bin/python -u rebuild_sigma_rescue_gamma0.py``.
The per-size ΔE_S fit uses a full-global 99-sample refit bootstrap; reruns
reuse its cache. Every stage fails loudly and is logged by the calling shell.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from Plotting.plotSigmaRescueSizeScaling import (
    COLORS, COUNT_COLOR, FRACTION_COLOR, MARKERS, PROTOCOL_LABELS, PROTOCOLS,
)


SNAPSHOT = ROOT / "sigma_rescue_interim/current_gamma0"
OUTPUT = ROOT / "Plots/powerLaw/sigma_rescue_size_scaling"


def stage(*args: str) -> None:
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(ROOT / ".matplotlib-cache"),
               PYTHONPATH=str(ROOT))
    command = [sys.executable, *args]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def split_panels() -> None:
    source = OUTPUT / "size_scaling_parameters.json"
    data = json.loads(source.read_text())
    records = data["records"]
    sizes = sorted(int(size) for size in data["paired_event_counts"])
    counts = {int(k): int(v) for k, v in data["paired_event_counts"].items()}
    fractions = {int(k): float(v) for k, v in data["rescue_fractions"].items()}
    seeds = {int(k): int(v) for k, v in data["established_seed_counts"].items()}
    outdir = OUTPUT / "separate_panels"
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("alpha", r"$\alpha$", False, "alpha"),
        ("Lambda", r"$\lambda$", True, "lambda"),
        ("event_count", "paired irreversible events", False, "event_coverage"),
        ("xmin", r"$\Delta E_{\min}$", True, "xmin"),
    )
    for metric, ylabel, log_y, stem in metrics:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        if metric == "event_count":
            fraction_ax = ax.twinx()
            x = np.asarray(sizes, dtype=float)
            y = np.asarray([counts[size] for size in sizes], dtype=float)
            f = np.asarray([fractions[size] for size in sizes], dtype=float)
            ax.plot(x, y, color=COUNT_COLOR, marker="o", linewidth=1.2)
            fraction_ax.plot(x, f, color=FRACTION_COLOR, linestyle="--", marker="D")
            for size, count in zip(sizes, y):
                ax.annotate(f"{seeds[size]} seeds", (size, count), xytext=(0, 5),
                            textcoords="offset points", ha="center", fontsize="x-small")
            ax.set_ylabel(ylabel, color=COUNT_COLOR)
            fraction_ax.set_ylabel(r"fraction of audited post-yield transitions with valid $\sigma_{12}$",
                                   color=FRACTION_COLOR)
            fraction_ax.set_ylim(0, 1.05)
            ax.legend(handles=[
                Line2D([], [], color=COUNT_COLOR, marker="o", label="paired irreversible events"),
                Line2D([], [], color=FRACTION_COLOR, marker="D", linestyle="--",
                       label="valid stress fraction"),
            ], loc="best", fontsize="small", frameon=False)
        else:
            for protocol in PROTOCOLS:
                rows = sorted((row for row in records if row["protocol"] == protocol),
                              key=lambda row: row["size"])
                x = np.asarray([row["size"] for row in rows], dtype=float)
                y = np.asarray([row[metric] for row in rows], dtype=float)
                ax.plot(x, y, marker=MARKERS[protocol], color=COLORS[protocol],
                        linewidth=1.1, label=PROTOCOL_LABELS[protocol])
                if protocol == "delta_E_S" and metric in {"alpha", "xmin"}:
                    key = "alpha_ci_95" if metric == "alpha" else "xmin_ci_95"
                    ci = np.asarray([row[key] for row in rows], dtype=float)
                    ax.vlines(x, ci[:, 0], ci[:, 1], color=COLORS[protocol],
                              alpha=0.55, linewidth=1.5)
            ax.set_ylabel(ylabel)
            ax.legend(loc="best", fontsize="small", frameon=False)
            if log_y:
                ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xlabel("System size $L$")
        ax.grid(alpha=0.2)
        ax.set_title("Post-yield " + stem.replace("_", " "))
        fig.tight_layout()
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"Saved four panels in {outdir}", flush=True)


def main() -> None:
    stage("-m", "Plotting.sigmaRescueSizeScalingPowerlawDiagnostics",
          "--snapshot-root", str(SNAPSHOT), "--pvalue-confidence", "0.05",
          "--bootstrap-xmin-mode", "global", "--observed-xmin-mode", "global",
          "--pvalue-workers", "8", "--resume")
    stage("-m", "Plotting.plotSigmaRescueSizeScaling", "--snapshot-root", str(SNAPSHOT))
    split_panels()
    stage("-m", "Plotting.plotSigmaRescueEsExponentSizeScaling",
          "--snapshot-root", str(SNAPSHOT), "--output-dir",
          str(ROOT / "Plots/powerLaw/sigma_rescue_es_exponent_size_scaling"))
    stage("tmp_sigma_rescue_finite_size_scaling.py", "--snapshot-root", str(SNAPSHOT),
          "--force")
    expected = [OUTPUT / "post_yield_size_scaling_parameters.pdf",
                *(OUTPUT / "separate_panels" / f"{name}.pdf"
                  for name in ("alpha", "lambda", "event_coverage", "xmin"))]
    for path in expected:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing generated PDF: {path}")
    print("COMPLETE: all gamma=0 sigma-rescue PDFs generated", flush=True)


if __name__ == "__main__":
    main()
