"""Diagnose the fixed-mu kappa classifier for relaxation energy drops.

For an AQS transition, ``Delta E_R = U_aff - U_0`` and

    kappa = Delta E_R / (rho * V_0 * Delta gamma**2).

The default detector is ``kappa_det = mu/2`` for ``rho=1``.  This module
compares it with the historical ``simpleDrop`` split and the recorded
element-level plastic-change indicator.  It is deliberately diagnostic; the
standard protocol is implemented elsewhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Management.updateCSV import read_macrodata_csv
from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import (
    infer_plastic_event_column,
)
from Plotting.standardPowerlaw import (
    kappa_detection_threshold,
    kappa_from_relaxation_energy,
)


DEFAULT_DATA_DIR = (
    _REPO_ROOT / "Plots/powerLaw/truncated_powerlaw_flowchart/data"
)
DEFAULT_ANALYSIS_SUMMARY = (
    _REPO_ROOT
    / "Plots/powerLaw/truncated_powerlaw_reversibility_flowchart/analysis_summary.json"
)
DEFAULT_OUTPUT = _REPO_ROOT / "output/pdf/kappa_event_classification_diagnostics.pdf"


def mu_kappa_threshold(mu, *, rho=1.0):
    """Return ``mu / (2 rho)`` for a scalar or array of moduli."""

    mu = np.asarray(mu, dtype=float)
    rho = float(rho)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive.")
    if np.any(~np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError("mu must contain only finite positive values.")
    return mu / (2.0 * rho)


def classification_metrics(predicted, recorded_plastic):
    """Return confusion counts and standard binary-classification metrics."""

    predicted = np.asarray(predicted, dtype=bool)
    recorded_plastic = np.asarray(recorded_plastic, dtype=bool)
    if predicted.shape != recorded_plastic.shape:
        raise ValueError("Predicted and recorded masks must have the same shape.")
    tp = int(np.count_nonzero(predicted & recorded_plastic))
    fp = int(np.count_nonzero(predicted & ~recorded_plastic))
    fn = int(np.count_nonzero(~predicted & recorded_plastic))
    tn = int(np.count_nonzero(~predicted & ~recorded_plastic))

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else np.nan

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "selected_fraction": ratio(tp + fp, predicted.size),
    }


def _reference_mu():
    return float(2.0 * kappa_detection_threshold())


def _discover_csv_paths(data_dir):
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*_fixed.csv"))
    if not paths:
        paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}.")
    return paths


def _simple_drop_er_threshold(summary_path):
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        return None
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    value = summary.get("delta_E_R_drop_detection_xmin")
    if value is None:
        return None
    threshold = float(value)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"Invalid simpleDrop Delta E_R threshold: {threshold}")
    return threshold


def collect_kappa_data(csv_paths, *, rho=1.0):
    """Collect aligned post-yield kappa and recorded event labels."""

    fields = {
        name: []
        for name in (
            "kappa",
            "load",
            "seed",
            "recorded_plastic",
            "simple_drop_scale",
        )
    }
    for path in map(Path, csv_paths):
        metadata = get_metadata(str(path))
        size = int(metadata["L"])
        seed = int(metadata["seed"])
        df = read_macrodata_csv(path, L=size)
        required = {"load", "avg_sigma12", "total_e_change_from_init"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"Missing columns {missing} in {path}.")

        delta_e_r = -df["total_e_change_from_init"].to_numpy(dtype=float)[1:]
        load = df["load"].to_numpy(dtype=float)
        delta_gamma = np.diff(load)
        reference_volume = float(size * size)
        kappa = kappa_from_relaxation_energy(
            delta_e_r,
            delta_gamma,
            reference_volume,
            rho=rho,
        )
        yield_load = float(df["load"].iloc[int(np.argmax(df["avg_sigma12"]))])
        event_load = load[1:]
        post_yield = (event_load > yield_load + 1e-2) & (
            event_load < float(df["load"].max())
        )
        plastic_col = infer_plastic_event_column(df)
        recorded_plastic = df[plastic_col].to_numpy()[1:] >= 1
        valid = post_yield & np.isfinite(kappa) & (kappa > 0)

        fields["kappa"].append(kappa[valid])
        fields["load"].append(event_load[valid])
        fields["seed"].append(np.full(np.count_nonzero(valid), seed, dtype=int))
        fields["recorded_plastic"].append(recorded_plastic[valid])
        fields["simple_drop_scale"].append(
            1.0
            / (
                rho
                * reference_volume
                * delta_gamma[valid] ** 2
            )
        )

    return {name: np.concatenate(values) for name, values in fields.items()}


def _log_pdf(values, bins):
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    valid = np.isfinite(hist) & (hist > 0)
    return centers[valid], hist[valid]


def _metric_curve(kappa, recorded_plastic, thresholds):
    metrics = [
        classification_metrics(kappa >= threshold, recorded_plastic)
        for threshold in thresholds
    ]
    return {
        key: np.asarray([metric[key] for metric in metrics], dtype=float)
        for key in ("precision", "recall", "specificity", "selected_fraction")
    }


def make_diagnostic_figure(data, *, simple_drop_er_det=None, rho=1.0):
    """Create the four-panel kappa diagnostic figure and return its summary."""

    kappa = data["kappa"]
    recorded = data["recorded_plastic"].astype(bool)
    seeds = data["seed"]
    if kappa.size == 0:
        raise ValueError("No positive post-yield Delta E_R events were found.")

    mu_reference = _reference_mu()
    reference_threshold = float(mu_kappa_threshold(mu_reference, rho=rho))
    classifiers = {
        r"$\kappa_{\det}=\mu/2$": (
            kappa >= reference_threshold,
            np.ones(kappa.shape, dtype=bool),
        ),
    }
    simple_threshold = None
    if simple_drop_er_det is not None:
        simple_thresholds = simple_drop_er_det * data["simple_drop_scale"]
        simple_threshold = float(np.median(simple_thresholds))
        if not np.allclose(simple_thresholds, simple_threshold, rtol=1e-8, atol=0.0):
            raise ValueError("simpleDrop does not map to one common kappa threshold.")
        classifiers = {
            "historical simpleDrop": (
                kappa >= simple_threshold,
                np.ones(kappa.shape, dtype=bool),
            ),
            **classifiers,
        }
    summaries = {
        label: classification_metrics(predicted[eligible], recorded[eligible])
        | {"eligible": int(np.count_nonzero(eligible))}
        for label, (predicted, eligible) in classifiers.items()
    }

    lower = max(float(np.min(kappa)), np.finfo(float).tiny)
    upper = float(np.max(kappa))
    bins = np.geomspace(lower, upper, 90)
    curve_thresholds = np.geomspace(lower, upper, 240)
    curve = _metric_curve(kappa, recorded, curve_thresholds)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    ax = axes[0, 0]
    for values, label, color in (
        (kappa, "all positive events", "0.2"),
        (kappa[~recorded], "recorded elastic", "tab:blue"),
        (kappa[recorded], "recorded plastic", "tab:orange"),
    ):
        x, y = _log_pdf(values, bins)
        ax.plot(x, y, label=f"{label} (n={values.size:,})", color=color)
    if simple_threshold is not None:
        ax.axvline(simple_threshold, color="tab:purple", linestyle="--", label="historical simpleDrop")
    ax.axvline(reference_threshold, color="tab:red", linestyle=":", label=r"$\kappa_{\det}=\mu/2$")
    ax.set(xscale="log", yscale="log", xlabel=r"$\kappa=\Delta E_R/(V_0\Delta\gamma^2)$", ylabel=r"PDF $p(\kappa)$", title="Post-yield kappa distribution")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for key, color in (
        ("precision", "tab:green"),
        ("recall", "tab:orange"),
        ("specificity", "tab:blue"),
        ("selected_fraction", "0.35"),
    ):
        ax.plot(curve_thresholds, curve[key], label=key.replace("_", " "), color=color)
    if simple_threshold is not None:
        ax.axvline(simple_threshold, color="tab:purple", linestyle="--")
    ax.axvline(reference_threshold, color="tab:red", linestyle=":")
    ax.set(xscale="log", ylim=(-0.02, 1.02), xlabel=r"candidate $\kappa_{\rm det}$", ylabel="fraction", title="Threshold trade-off against recorded plastic changes")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    unique_seeds = np.unique(seeds)
    width = 0.24
    x_seed = np.arange(unique_seeds.size)
    colors = ("tab:purple", "tab:red")
    for offset, ((label, (predicted, eligible)), color) in enumerate(
        zip(classifiers.items(), colors)
    ):
        fractions = []
        for seed in unique_seeds:
            mask = (seeds == seed) & eligible
            fractions.append(
                float(np.mean(predicted[mask])) if np.any(mask) else np.nan
            )
        ax.bar(
            x_seed + (offset - 1) * width,
            fractions,
            width=width,
            color=color,
            label=label,
        )
    ax.set_xticks(x_seed, [str(seed) for seed in unique_seeds])
    ax.set(xlabel="seed", ylabel="selected event fraction", ylim=(0, None), title="Classifier stability across seeds")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    labels = list(classifiers)
    x_classifier = np.arange(len(labels))
    metric_names = ("precision", "recall", "specificity")
    metric_colors = ("tab:green", "tab:orange", "tab:blue")
    for index, (metric_name, color) in enumerate(zip(metric_names, metric_colors)):
        values = [summaries[label][metric_name] for label in labels]
        ax.bar(
            x_classifier + (index - 1) * width,
            values,
            width=width,
            color=color,
            label=metric_name,
        )
    ax.set_xticks(x_classifier, labels)
    ax.set(ylabel="fraction", ylim=(0, 1.05), title="Classification metrics")
    ax.legend(fontsize=8)
    note = (
        f"rho={rho:g}; mu(F=I)={mu_reference:.4g}; "
        f"kappa_det={reference_threshold:.4g}"
        + (
            f"; historical kappa_simpleDrop={simple_threshold:.4g}"
            if simple_threshold is not None
            else "; historical simpleDrop comparison unavailable"
        )
        + "\n"
        f"positive post-yield events={kappa.size:,}; recorded plastic={np.count_nonzero(recorded):,}; "
        "local a_1212,i branch is intentionally excluded"
    )
    fig.suptitle("Kappa event-classification diagnostic", fontsize=14)
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    return fig, {
        "rho": float(rho),
        "mu_reference": mu_reference,
        "kappa_det": reference_threshold,
        "historical_simple_drop_threshold": simple_threshold,
        "event_count": int(kappa.size),
        "recorded_plastic_count": int(np.count_nonzero(recorded)),
        "classifiers": summaries,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--analysis-summary", type=Path, default=DEFAULT_ANALYSIS_SUMMARY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rho", type=float, default=1.0)
    args = parser.parse_args(argv)

    csv_paths = _discover_csv_paths(args.data_dir)
    data = collect_kappa_data(csv_paths, rho=args.rho)
    er_det = _simple_drop_er_threshold(args.analysis_summary)
    fig, summary = make_diagnostic_figure(
        data,
        simple_drop_er_det=er_det,
        rho=args.rho,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
