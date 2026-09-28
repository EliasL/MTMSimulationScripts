"""Plot paired coarse/global bootstrap-xmin diagnostics from cached results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".eval_cache_irreversible"
OUTPUT_DIR = ROOT / "Plots/powerLaw/pvalue_examples"
OUTPUT_STEM = "irreversible_bootstrap_xmin_comparison"
REFERENCE_VOLUME = 250.0**2


def _load_paired_results():
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in CACHE_DIR.glob("*.json")]
    rapid = next(
        payload for payload in payloads if payload.get("xmin_mode") == "rapidGlobal"
    )
    global_payloads = [
        payload
        for payload in payloads
        if payload.get("xmin_mode") == "global"
    ]
    if len(global_payloads) != 1:
        raise RuntimeError(
            f"Expected exactly one full-global cache, found {len(global_payloads)}."
        )
    global_payload = global_payloads[0]
    if global_payload.get("complete", False):
        global_xmins = np.asarray(global_payload["bootstrap_xmins"], dtype=float)
        global_distances = np.asarray(
            global_payload["bootstrap_D_values"], dtype=float
        )
    else:
        results = global_payload.get("results", [])
        if not results:
            raise RuntimeError("The global-bootstrap cache has no completed replicates.")
        global_xmins = np.asarray([result[2] for result in results], dtype=float)
        global_distances = np.asarray([result[0] for result in results], dtype=float)
    count = global_xmins.size
    rapid_xmins = np.asarray(rapid["bootstrap_xmins"][:count], dtype=float)
    rapid_distances = np.asarray(rapid["bootstrap_D_values"][:count], dtype=float)
    if not (
        rapid_xmins.shape
        == rapid_distances.shape
        == global_xmins.shape
        == global_distances.shape
    ):
        raise RuntimeError("The paired bootstrap arrays have inconsistent shapes.")
    if np.any(global_distances > rapid_distances + 1e-12):
        raise RuntimeError("The purported global search exceeded a coarse-search KS value.")
    return rapid_xmins, rapid_distances, global_xmins, global_distances


def _fold_difference(first, second):
    ratio = np.asarray(first, dtype=float) / np.asarray(second, dtype=float)
    return np.maximum(ratio, 1.0 / ratio)


def main():
    rapid_xmins, rapid_d, global_xmins, global_d = _load_paired_results()
    replicate = np.arange(1, rapid_d.size + 1)
    exact_xmin = rapid_xmins == global_xmins
    d_fold = _fold_difference(rapid_d, global_d)
    xmin_fold = _fold_difference(rapid_xmins, global_xmins)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10.0), sharex=True)

    axes[0].plot(replicate, global_d, color="0.15", lw=1.0, label="true global minimum")
    axes[0].plot(replicate, rapid_d, color="tab:blue", lw=0.8, alpha=0.85, label="100-point coarse-local minimum")
    axes[0].set_ylabel(r"minimum $D$")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    axes[1].scatter(
        replicate,
        global_xmins / REFERENCE_VOLUME,
        s=7,
        color="0.15",
        alpha=0.7,
        label=r"true global $x_{\min}$",
    )
    axes[1].scatter(
        replicate,
        rapid_xmins / REFERENCE_VOLUME,
        s=7,
        color="tab:blue",
        alpha=0.55,
        label=r"coarse-local $x_{\min}$",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$\Delta E_{S,\min}/V_0$")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")

    axes[2].scatter(replicate, xmin_fold, s=8, color="tab:purple", alpha=0.7)
    axes[2].axhline(1.0, color="0.15", lw=1.0, linestyle="--")
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$x_{\min}$ fold difference")
    axes[2].text(
        0.01,
        0.95,
        (
            f"exact $x_{{\\min}}$ matches: {np.count_nonzero(exact_xmin)}/{replicate.size}"
            f"\nmedian $x_{{\\min}}$ fold difference: {np.median(xmin_fold):.5g}"
            f"\nmaximum $x_{{\\min}}$ fold difference: {np.max(xmin_fold):.5g}"
        ),
        transform=axes[2].transAxes,
        va="top",
        ha="left",
    )

    axes[3].scatter(replicate, d_fold, s=8, color="tab:red", alpha=0.7)
    axes[3].axhline(1.0, color="0.15", lw=1.0, linestyle="--")
    axes[3].set_yscale("log")
    axes[3].set_ylabel(r"$D$ fold difference")
    axes[3].set_xlabel("bootstrap replicate")

    fig.suptitle(
        f"Coarse versus global bootstrap comparison ({replicate.size} paired replicates)"
    )
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}{suffix}", dpi=220, bbox_inches="tight")
    print(OUTPUT_DIR / f"{OUTPUT_STEM}.png")


if __name__ == "__main__":
    main()
