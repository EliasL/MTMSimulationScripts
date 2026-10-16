"""Plot standard irreversible :math:`\\Delta E_S` exponents by system size.

The input is an immutable sigma-rescue snapshot produced by
``Management.sigmaRescueDropSnapshot``.  At each size, all usable,
non-reconnecting source trajectories are pooled after the post-yield filter.
The irreversible mask is determined from paired ``Delta E_R``/``Delta E_S``
events using ``kappa_det = mu / (2 rho)`` with ``rho = N / V0 = 2``.  Only
then are finite positive ``Delta E_S`` values fit with the exhaustive global
KS-minimum cutoff search.

Selected-fit p-values and bootstrap standard deviations come from the
canonical per-size diagnostics summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Plotting.plotPowerLaw import dist_from_fit, make_fit
from Plotting.standardPowerlaw import (
    SIGMA_RESCUE_POST_YIELD_WINDOW,
    DEFAULT_KAPPA_RHO,
    EventDrops,
    kappa_detection_threshold,
    kappa_from_relaxation_energy,
    positive_es,
    load_sigma_rescue_post_yield,
    require_gamma0_snapshot,
    split_by_kappa,
)


REQUIRED_COLUMNS = (
    "size",
    "seed",
    "load_ip1",
    "delta_gamma",
    "reference_volume",
    "delta_E_R",
    "delta_E_S",
)


def _load_post_yield(snapshot_root: Path) -> pd.DataFrame:
    require_gamma0_snapshot(snapshot_root)
    table_path = Path(snapshot_root) / "tables" / "drops_usable.csv.gz"
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing usable rescue table: {table_path}")
    frame = load_sigma_rescue_post_yield(snapshot_root, list(REQUIRED_COLUMNS))
    if frame.empty:
        raise ValueError("No rows after the per-run avg_P12 peak +0.01.")
    for column in REQUIRED_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite {column!r} value in {table_path}.")
    if np.any(frame["delta_gamma"].to_numpy(dtype=float) <= 0):
        raise ValueError("delta_gamma must be positive.")
    if np.any(frame["reference_volume"].to_numpy(dtype=float) <= 0):
        raise ValueError("reference_volume must be positive.")
    frame["kappa"] = kappa_from_relaxation_energy(
        frame["delta_E_R"].to_numpy(dtype=float),
        frame["delta_gamma"].to_numpy(dtype=float),
        frame["reference_volume"].to_numpy(dtype=float),
        rho=DEFAULT_KAPPA_RHO,
    )
    return frame


def _fit_size(frame: pd.DataFrame, size: int, cache_dir: Path) -> dict[str, float | int]:
    selected = frame.loc[frame["size"] == size]
    if selected.empty:
        raise ValueError(f"L={size} is absent from the rescue table.")
    drops = EventDrops(
        er=selected["delta_E_R"].to_numpy(dtype=float),
        es=selected["delta_E_S"].to_numpy(dtype=float),
        kappa=selected["kappa"].to_numpy(dtype=float),
    )
    kappa_det = kappa_detection_threshold(rho=DEFAULT_KAPPA_RHO)
    split = split_by_kappa(drops, kappa_det)
    es_irrev = positive_es(drops, split.is_irrev)
    if es_irrev.size < 3:
        raise ValueError(f"L={size} has fewer than three positive irreversible Delta E_S values.")
    fit = make_fit(
        es_irrev,
        cache_dir=str(cache_dir / f"L{size}"),
        use_cache=True,
        parallel_xmin=False,
        xmin_selection="global",
        xmin_search_mode="full",
        xmin_search_kwargs={
            "refine": True,
        },
    )
    analysis = fit.xmin_fitting_results
    if analysis is None:
        raise RuntimeError(f"L={size} has no xmin analysis.")
    xmin = float(analysis["global_min_xmin"])
    distance = float(analysis["global_min_distance"])
    tail_count = int(np.count_nonzero(es_irrev >= xmin))
    distribution = dist_from_fit(fit)
    alpha = float(distribution.alpha)
    cutoff_rate = float(getattr(distribution, "Lambda", np.nan))
    if not np.isfinite(alpha) or not np.isfinite(cutoff_rate) or cutoff_rate <= 0:
        raise RuntimeError(f"Invalid truncated-power-law fit for L={size}.")
    return {
        "size": int(size),
        "seed_count": int(selected["seed"].nunique()),
        "post_yield_event_count": int(len(selected)),
        "irreversible_event_count": int(np.count_nonzero(split.is_irrev)),
        "positive_delta_E_S_irreversible_count": int(es_irrev.size),
        "rho": DEFAULT_KAPPA_RHO,
        "kappa_det": kappa_det,
        "es_xmin_ks": xmin,
        "global_min_D": distance,
        "tail_count": tail_count,
        "alpha": alpha,
        "Lambda": cutoff_rate,
    }


def _write_summary(records: list[dict[str, float | int]], path: Path) -> None:
    fieldnames = list(records[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _plot(records: list[dict[str, float | int]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    sizes = np.asarray([row["size"] for row in records], dtype=float)
    alpha = np.asarray([row["alpha"] for row in records], dtype=float)
    ci_low = np.asarray([row["alpha"] - row["alpha_std"] for row in records], dtype=float)
    ci_high = np.asarray([row["alpha"] + row["alpha_std"] for row in records], dtype=float)
    seeds = np.asarray([row["seed_count"] for row in records], dtype=int)

    fig, alpha_ax = plt.subplots(figsize=(5.3, 3.1))
    alpha_ax.plot(sizes, alpha, color="#009E73", marker="o", linewidth=1.3)
    alpha_ax.vlines(sizes, ci_low, ci_high, color="#009E73", alpha=0.65,
                    label=r"fit $\pm1$ bootstrap SD")
    for index, (size, value, count) in enumerate(zip(sizes, alpha, seeds)):
        if index == 0:
            horizontal_alignment, horizontal_offset = "left", 5
        elif index == len(sizes) - 1:
            horizontal_alignment, horizontal_offset = "right", -5
        else:
            horizontal_alignment, horizontal_offset = "center", 0
        alpha_ax.annotate(
            rf"$n_s={count}$",
            (size, value),
            xytext=(horizontal_offset, 6),
            textcoords="offset points",
            ha=horizontal_alignment,
            fontsize=8,
        )
    alpha_ax.set_ylabel(r"Truncated-law exponent $\hat\alpha$")
    alpha_ax.margins(x=0.06, y=0.16)
    alpha_ax.grid(alpha=0.2)
    alpha_ax.legend(frameon=True, fontsize="small")
    alpha_ax.set_xlabel(r"System size $L$")
    alpha_ax.set_xticks(sizes.astype(int))

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(fig)


def run(snapshot_root: Path, output_dir: Path, *, sizes: tuple[int, ...] | None = None,
        largest_first: bool = False,
        fit_summary_path: Path = Path("Plots/powerLaw/sigma_rescue_size_scaling_individual_piola_peak/summary.json"),
        force: bool = False) -> list[dict[str, float | int]]:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.csv"
    if summary_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing summary: {summary_path}")
    frame = _load_post_yield(Path(snapshot_root))
    available = tuple(sorted(frame["size"].unique().astype(int)))
    selected_sizes = available if sizes is None else tuple(sorted(set(sizes)))
    selected_sizes = tuple(sorted(selected_sizes, reverse=largest_first))
    missing = sorted(set(selected_sizes) - set(available))
    if missing:
        raise ValueError(f"Requested sizes absent from the rescue table: {missing}")
    records = [
        _fit_size(frame, size, output_dir / "xmin_cache") for size in selected_sizes
    ]
    fit_summary = json.loads(Path(fit_summary_path).read_text())
    if (fit_summary.get("delta_E_S_tangent") != "gamma0"
            or fit_summary.get("post_yield_window") != SIGMA_RESCUE_POST_YIELD_WINDOW
            or Path(fit_summary["snapshot_root"]).resolve() != Path(snapshot_root).resolve()):
        raise ValueError("Bootstrap summary has the wrong data or Delta E_S convention.")
    if fit_summary.get("bootstrap_xmin_mode") != "global":
        raise ValueError("A full-global refit bootstrap is required.")
    for record in records:
        result = fit_summary["fits"][str(record["size"])]["protocols"]["delta_E_S"]
        if not np.isclose(record["es_xmin_ks"], result["global_min_xmin"]):
            raise ValueError(f"Cutoff mismatch for L={record['size']}.")
        record.update(
            pvalue=float(result["clauset_pvalue_v2"]),
            xmin_mean=float(result["bootstrap_xmin_mean"]),
            xmin_std=float(result["bootstrap_xmin_std"]),
            alpha_mean=float(result["bootstrap_alpha_mean"]),
            alpha_std=float(result["bootstrap_alpha_std"]),
        )
        record.update(snapshot_root=str(Path(snapshot_root).resolve()), delta_E_S_tangent="gamma0")
    _write_summary(records, summary_path)
    _plot(records, output_dir / "alpha_size_scaling.pdf")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--largest-first", action="store_true")
    parser.add_argument("--fit-summary", type=Path,
                        default=Path("Plots/powerLaw/sigma_rescue_size_scaling_individual_piola_peak/summary.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for record in run(args.snapshot_root, args.output_dir, sizes=args.sizes,
                      largest_first=args.largest_first,
                      fit_summary_path=args.fit_summary, force=args.force):
        print(record)


if __name__ == "__main__":
    main()
