"""Compare a fixed ``mu / (2 rho)`` kappa split across system sizes.

The event-level quantity is taken directly from the macrodata convention used
by the power-law extraction,

    Delta E_R = U_aff - U_0 = -total_e_change_from_init,
    kappa = Delta E_R / (rho V_0 Delta gamma**2).

Only the constant reference ``mu(F=I)`` is used here.  In particular,
this diagnostic does not use the strain-dependent ``a_1212,i`` branch.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from Plotting.dataFunctions import get_metadata
from Plotting.kappaEventClassification import (
    classification_metrics,
    kappa_from_relaxation_energy,
    mu_kappa_threshold,
)
from Plotting.standardPowerlaw import (
    DEFAULT_KAPPA_RHO,
    kappa_detection_threshold,
)
from Plotting.sizeScalingCollapse import _read_mixed_selected


DEFAULT_DATA_ROOT = Path("/Volumes/data/remoteData/macro")
DEFAULT_SIZE_SUMMARY = (
    _REPO_ROOT
    / "Plots/reversible_event_analysis/size_scaling_E_R_classification/classification_summary.csv"
)
DEFAULT_OUTPUT = _REPO_ROOT / "output/pdf/kappa_event_classification_by_size.pdf"

_TARGET_RE = re.compile(
    r"simpleShear,s(?P<L>\d+)x(?P<L2>\d+)l0\.15,1e-05,1\.0PBC"
)


def _reference_mu():
    return float(
        2.0 * DEFAULT_KAPPA_RHO * kappa_detection_threshold()
    )


def _read_required_columns(path):
    """Read the required columns across old/new headers in one raw CSV."""

    wanted = {
        "load",
        "total_e_change_from_init",
        "nr_elements_with_m3_fix_change",
        "nr_elements_with_m3_change",
    }
    df = _read_mixed_selected(Path(path), wanted)
    plastic_name = next(
        (
            name
            for name in (
                "nr_elements_with_m3_fix_change",
                "nr_elements_with_m3_change",
            )
            if name in df and np.isfinite(df[name]).any()
        ),
        None,
    )
    if plastic_name is None:
        raise KeyError(f"Missing plastic-event column in {path}.")
    required = ("load", "total_e_change_from_init")
    missing = [name for name in required if name not in df]
    if missing:
        raise KeyError(f"Missing columns {missing} in {path}.")
    return {
        "load": df["load"].to_numpy(dtype=float),
        "total_e_change_from_init": df["total_e_change_from_init"].to_numpy(dtype=float),
        "plastic": df[plastic_name].to_numpy(dtype=float),
    }


def _constant_kappa_data(csv_paths, size, *, rho=DEFAULT_KAPPA_RHO):
    """Read aligned post-yield kappa values without local Born moduli."""

    kappa_parts = []
    plastic_parts = []
    seed_parts = []
    reference_volume = float(size * size)
    for path in map(Path, csv_paths):
        metadata = get_metadata(str(path))
        if int(metadata["L"]) != int(size):
            raise ValueError(f"Size mismatch for {path}: expected L={size}.")
        columns = _read_required_columns(path)
        load = columns["load"]
        delta_gamma = np.diff(load)
        delta_e_r = -columns["total_e_change_from_init"][1:]
        kappa = kappa_from_relaxation_energy(
            delta_e_r,
            delta_gamma,
            reference_volume,
            rho=rho,
        )
        load_ip1 = load[1:]
        # Match the size-scaling power-law protocol's post-yield window.
        post_yield = (load_ip1 > 0.7) & (load_ip1 < float(load.max()))
        recorded_plastic = columns["plastic"][1:] >= 1
        valid = post_yield & np.isfinite(kappa) & (kappa > 0)

        kappa_parts.append(kappa[valid])
        plastic_parts.append(recorded_plastic[valid])
        seed_parts.append(
            np.full(np.count_nonzero(valid), int(metadata["seed"]), dtype=int)
        )

    if not kappa_parts:
        raise ValueError(f"No data files found for L={size}.")
    return {
        "kappa": np.concatenate(kappa_parts),
        "recorded_plastic": np.concatenate(plastic_parts),
        "seed": np.concatenate(seed_parts),
    }


def _discover_paths(data_root, sizes, seeds_per_size):
    """Discover the standard size-scaling macrodata files."""

    data_root = Path(data_root)
    groups = {int(size): [] for size in sizes}
    for path in sorted(data_root.glob("*/macroData.csv")):
        match = _TARGET_RE.match(path.parent.name)
        if match is None:
            continue
        size = int(match.group("L"))
        if size in groups and int(match.group("L2")) == size:
            groups[size].append(path)

    if not any(groups.values()):
        from Plotting.sizeScalingCollapse import completed_size_scaling_paths

        discovered, _ = completed_size_scaling_paths(
            data_root,
            seeds_per_size,
            post_hi=1.0 - 1e-12,
        )
        groups = {
            int(size): list(discovered[size])
            for size in sizes
            if int(size) in discovered
        }

    selected = {}
    for size, paths in groups.items():
        if not paths:
            continue
        seeds = [int(get_metadata(str(path))["seed"]) for path in paths]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds found for L={size}.")
        selected[size] = sorted(paths, key=lambda path: int(get_metadata(str(path))["seed"]))
    if len(selected) < 2:
        raise FileNotFoundError(
            f"Expected at least two sizes under {data_root}."
        )
    return selected


def _simple_drop_thresholds(size_summary, sizes):
    """Return existing post-yield simpleDrop Delta E_R thresholds by size."""

    size_summary = Path(size_summary)
    if not size_summary.is_file():
        return None
    thresholds = {}
    with size_summary.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["regime"] != "post":
                continue
            size = int(row["size"])
            if size in sizes:
                value = row.get("xmin_delta_E_R")
                if value not in (None, ""):
                    thresholds[size] = float(value)

    missing = sorted(set(sizes) - set(thresholds))
    return thresholds if not missing else None


def _log_pdf(values, bins):
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    valid = np.isfinite(hist) & (hist > 0)
    return centers[valid], hist[valid]


def _summary_for_size(data, size, simple_drop_er_det=None, *, rho, reference_threshold):
    kappa = data["kappa"]
    recorded = data["recorded_plastic"].astype(bool)
    fixed = classification_metrics(kappa >= reference_threshold, recorded)
    summary = {
        "size": int(size),
        "event_count": int(kappa.size),
        "recorded_plastic_count": int(np.count_nonzero(recorded)),
        "fixed_threshold": float(reference_threshold),
        "fixed_mu_over_2": fixed,
    }
    if simple_drop_er_det is not None:
        simple_threshold = float(
            simple_drop_er_det / (rho * size * size * (1e-5**2))
        )
        summary.update(
            {
                "simple_drop_delta_e_r": float(simple_drop_er_det),
                "simple_drop_threshold": simple_threshold,
                "simpleDrop": classification_metrics(
                    kappa >= simple_threshold, recorded
                ),
            }
        )
    return summary


def make_size_diagnostic(
    data_by_size,
    simple_drop_er_det=None,
    *,
    rho=DEFAULT_KAPPA_RHO,
):
    """Create the multi-page size-scaling diagnostic and return its summary."""

    sizes = sorted(data_by_size)
    mu_reference = _reference_mu()
    reference_threshold = float(mu_kappa_threshold(mu_reference, rho=rho))
    summaries = {
        size: _summary_for_size(
            data_by_size[size],
            size,
            None
            if simple_drop_er_det is None
            else simple_drop_er_det.get(size),
            rho=rho,
            reference_threshold=reference_threshold,
        )
        for size in sizes
    }
    has_simple = all(
        "simple_drop_threshold" in summaries[size] for size in sizes
    )

    with PdfPages(_PDF_OUTPUT) as pdf:
        ncols = 3
        nrows = int(np.ceil(len(sizes) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(11.5, 3.65 * nrows),
            squeeze=False,
        )
        axes_flat = axes.ravel()
        for ax, size in zip(axes_flat, sizes):
            data = data_by_size[size]
            kappa = data["kappa"]
            recorded = data["recorded_plastic"].astype(bool)
            lower = max(float(np.min(kappa)), np.finfo(float).tiny)
            upper = float(np.max(kappa))
            bins = np.geomspace(lower, upper, 75)
            for values, label, color in (
                (kappa, "all", "0.2"),
                (kappa[~recorded], "recorded elastic", "tab:blue"),
                (kappa[recorded], "recorded plastic", "tab:orange"),
            ):
                x, y = _log_pdf(values, bins)
                ax.plot(x, y, color=color, label=f"{label} (n={values.size:,})")
            simple_threshold = summaries[size].get("simple_drop_threshold")
            if simple_threshold is not None:
                ax.axvline(
                    simple_threshold,
                    color="tab:purple",
                    linestyle="--",
                    label=r"historical simpleDrop $\kappa_{\rm det}$",
                )
            ax.axvline(
                reference_threshold,
                color="tab:red",
                linestyle=":",
                label=r"fixed $\mu/(2\rho)$",
            )
            ax.set(
                xscale="log",
                yscale="log",
                xlabel=r"$\kappa=\Delta E_R/(\rho V_0\Delta\gamma^2)$",
                ylabel=r"PDF $p(\kappa)$",
                title=f"L={size}",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        for ax in axes_flat[len(sizes) :]:
            ax.axis("off")
        fig.suptitle(
            "Kappa distributions: fixed $\mu/(2\rho)$"
            + (" versus historical simpleDrop" if has_simple else ""),
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        x = np.arange(len(sizes))
        labels = [str(size) for size in sizes]
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), squeeze=False)
        ax = axes[0, 0]
        if has_simple:
            simple_thresholds = [
                summaries[size]["simple_drop_threshold"] for size in sizes
            ]
            ax.plot(
                x,
                simple_thresholds,
                "o-",
                color="tab:purple",
                label="historical simpleDrop",
            )
        ax.axhline(reference_threshold, color="tab:red", linestyle=":", label=r"fixed $\mu/(2\rho)$")
        ax.set_xticks(x, labels)
        ax.set(xlabel="system size L", ylabel=r"$\kappa$ threshold", title="Threshold versus size")
        ax.legend()
        ax.grid(alpha=0.2)

        ax = axes[0, 1]
        methods = (("simpleDrop", "tab:purple"), ("fixed_mu_over_2", "tab:red")) if has_simple else (("fixed_mu_over_2", "tab:red"),)
        for method, color in methods:
            values = [summaries[size][method]["selected_fraction"] for size in sizes]
            ax.plot(x, values, "o-", color=color, label=method.replace("_", " "))
        values = [
            summaries[size]["recorded_plastic_count"] / summaries[size]["event_count"]
            for size in sizes
        ]
        ax.plot(x, values, "o-", color="tab:orange", label="recorded plastic fraction")
        ax.set_xticks(x, labels)
        ax.set(xlabel="system size L", ylabel="fraction", title="Selected-event fraction")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

        ax = axes[1, 0]
        for metric, color in (("precision", "tab:green"), ("recall", "tab:orange")):
            method_styles = (("simpleDrop", "--"), ("fixed_mu_over_2", "-")) if has_simple else (("fixed_mu_over_2", "-"),)
            for method, linestyle in method_styles:
                values = [summaries[size][method][metric] for size in sizes]
                ax.plot(
                    x,
                    values,
                    marker="o",
                    linestyle=linestyle,
                    color=color,
                    label=f"{method.replace('_', ' ')} {metric}",
                )
        ax.set_xticks(x, labels)
        ax.set(xlabel="system size L", ylabel="fraction", ylim=(0, 1.05), title="Precision and recall")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.2)

        ax = axes[1, 1]
        for method, color in methods:
            values = [summaries[size][method]["specificity"] for size in sizes]
            ax.plot(x, values, "o-", color=color, label=method.replace("_", " "))
        ax.set_xticks(x, labels)
        ax.set(xlabel="system size L", ylabel="fraction", ylim=(0, 1.05), title="Specificity")
        ax.legend()
        ax.grid(alpha=0.2)
        note = (
            f"rho={rho:g}; mu(F=I)={mu_reference:.6g}; fixed threshold={reference_threshold:.6g}\n"
            "Fixed classifier is kappa >= mu/(2 rho); no a_1212,i classifier is used.\n"
            "Post-yield is 0.7 < load < max(load); "
            + (
                "historical simpleDrop thresholds are shown when available."
                if has_simple
                else "no historical simpleDrop thresholds were supplied."
            )
        )
        fig.suptitle("Size-scaling classifier comparison", fontsize=14)
        fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return {
        "rho": float(rho),
        "mu_reference": mu_reference,
        "fixed_threshold": reference_threshold,
        "sizes": summaries,
    }


_PDF_OUTPUT = DEFAULT_OUTPUT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seeds-per-size", type=int, default=6)
    parser.add_argument("--size-summary", type=Path, default=DEFAULT_SIZE_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rho", type=float, default=DEFAULT_KAPPA_RHO)
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 150, 200, 250])
    args = parser.parse_args(argv)

    global _PDF_OUTPUT
    _PDF_OUTPUT = args.output
    path_groups = _discover_paths(args.data_root, args.sizes, args.seeds_per_size)
    sizes = sorted(path_groups)
    thresholds = _simple_drop_thresholds(args.size_summary, sizes)
    data_by_size = {}
    for size in sizes:
        print(f"Reading L={size} ({len(path_groups[size])} seeds)...", flush=True)
        data_by_size[size] = _constant_kappa_data(
            path_groups[size],
            size,
            rho=args.rho,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = make_size_diagnostic(
        data_by_size,
        thresholds,
        rho=args.rho,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
