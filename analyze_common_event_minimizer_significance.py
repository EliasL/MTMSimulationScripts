#!/usr/bin/env python3
"""Exact paired sign tests for common-event minimizer energies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ALGORITHMS = ("FIRE", "CG", "LBFGS")
PAIRS = (
    ("CG", "LBFGS", "CG lower than LBFGS"),
    ("FIRE", "CG", "FIRE higher than CG"),
    ("FIRE", "LBFGS", "FIRE higher than LBFGS"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path("common_event_minimizer_comparison_all_events.csv"),
    )
    parser.add_argument(
        "--metric",
        choices=("trajectory_total_energy_minimum", "trajectory_total_energy_final"),
        default="trajectory_total_energy_minimum",
        help="Energy quantity used for the paired comparisons.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fire-rerun-root",
        type=Path,
        help="Replace FIRE trajectory final/minimum energies with these uncapped reruns.",
    )
    return parser.parse_args()


def fire_rerun_energies(rerun_root: Path | None) -> dict[str, tuple[float, float]]:
    if rerun_root is None:
        return {}
    if not rerun_root.is_dir():
        raise FileNotFoundError(rerun_root)
    energies = {}
    for manifest_path in rerun_root.rglob("rerun_manifest.json"):
        event_id = str(manifest_path.parent.relative_to(rerun_root))
        payload = json.loads(manifest_path.read_text())
        result = payload.get("rerun_fire_result")
        if not isinstance(result, dict) or result.get("algorithm") != "FIRE":
            raise ValueError(f"Invalid FIRE rerun manifest: {manifest_path}")
        directories = result.get("minimization_directories")
        if not isinstance(directories, list) or len(directories) != 1:
            raise ValueError(f"Expected one FIRE trajectory in {manifest_path}")
        trajectory_path = Path(directories[0]) / "macroData.csv"
        with trajectory_path.open(newline="") as stream:
            trajectory = list(csv.DictReader(stream))
        if not trajectory or "total_energy" not in trajectory[0]:
            raise ValueError(f"Missing total-energy trajectory in {trajectory_path}")
        values = np.asarray([float(row["total_energy"]) for row in trajectory])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite FIRE trajectory energy in {trajectory_path}")
        if event_id in energies:
            raise ValueError(f"Duplicate FIRE rerun for {event_id}")
        energies[event_id] = (float(values[-1]), float(np.min(values)))
    if not energies:
        raise ValueError(f"No FIRE rerun manifests found in {rerun_root}")
    return energies


def selected_rows(
    csv_path: Path, metric: str, fire_rerun_root: Path | None
) -> tuple[list[dict[str, str]], int]:
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "seed",
        *(f"{a}_first_drop_load_step" for a in ALGORITHMS),
        *(f"{a}_{metric}" for a in ALGORITHMS),
    }
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Missing required columns in {csv_path}")
    reruns = fire_rerun_energies(fire_rerun_root)
    updated = 0
    for row in rows:
        replacement = reruns.get(row["event_relative_path"])
        if replacement is None:
            continue
        final, minimum = replacement
        row["FIRE_trajectory_total_energy_final"] = repr(final)
        row["FIRE_trajectory_total_energy_minimum"] = repr(minimum)
        updated += 1
    if reruns and updated != len(reruns):
        raise ValueError(
            f"Only matched {updated} of {len(reruns)} FIRE reruns to CSV rows"
        )
    selected = [
        row
        for row in rows
        if len({int(row[f"{a}_first_drop_load_step"]) for a in ALGORITHMS}) == 1
    ]
    if len(selected) != 200:
        raise ValueError(f"Expected 200 synchronized-load events, got {len(selected)}")
    return selected, updated


def test_pair(
    rows: list[dict[str, str]], left: str, right: str, claim: str, metric: str
) -> dict:
    left_energy = np.asarray([float(row[f"{left}_{metric}"]) for row in rows])
    right_energy = np.asarray([float(row[f"{right}_{metric}"]) for row in rows])
    differences = left_energy - right_energy

    if claim.startswith("CG lower"):
        wins = int(np.count_nonzero(differences < 0.0))
        losses = int(np.count_nonzero(differences > 0.0))
    else:
        wins = int(np.count_nonzero(differences > 0.0))
        losses = int(np.count_nonzero(differences < 0.0))
    ties = int(np.count_nonzero(differences == 0.0))
    resolved = wins + losses
    if resolved == 0:
        raise ValueError(f"No resolved comparisons for {left} vs {right}")

    one_sided_claim = binomtest(wins, resolved, 0.5, alternative="greater")
    one_sided_opposite = binomtest(losses, resolved, 0.5, alternative="greater")
    two_sided = binomtest(wins, resolved, 0.5, alternative="two-sided")
    z = 1.959963984540054
    proportion = wins / resolved
    denominator = 1.0 + z**2 / resolved
    center = (proportion + z**2 / (2.0 * resolved)) / denominator
    half_width = z * np.sqrt(
        proportion * (1.0 - proportion) / resolved + z**2 / (4.0 * resolved**2)
    ) / denominator
    return {
        "claim": claim,
        "left_minus_right": f"{left} minus {right}",
        "wins_supporting_claim": wins,
        "losses_opposing_claim": losses,
        "ties_at_csv_precision": ties,
        "resolved_events": resolved,
        "win_fraction": proportion,
        "win_fraction_wilson_ci_95": [center - half_width, center + half_width],
        "exact_one_sided_p_supporting_claim": one_sided_claim.pvalue,
        "exact_one_sided_p_supporting_opposite_order": one_sided_opposite.pvalue,
        "exact_two_sided_sign_test_p": two_sided.pvalue,
    }


def seed_majority_check(
    rows: list[dict[str, str]], left: str, right: str, claim: str, metric: str
) -> dict:
    seed_outcomes = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        differences = np.asarray(
            [
                float(row[f"{left}_{metric}"]) - float(row[f"{right}_{metric}"])
                for row in seed_rows
            ]
        )
        supports = int(np.count_nonzero(differences < 0.0)) if claim.startswith("CG lower") else int(np.count_nonzero(differences > 0.0))
        opposes = int(np.count_nonzero(differences > 0.0)) if claim.startswith("CG lower") else int(np.count_nonzero(differences < 0.0))
        if supports > opposes:
            outcome = "supports"
        elif supports < opposes:
            outcome = "opposes"
        else:
            outcome = "tie"
        seed_outcomes.append({"seed": seed, "outcome": outcome})
    supports = sum(item["outcome"] == "supports" for item in seed_outcomes)
    opposes = sum(item["outcome"] == "opposes" for item in seed_outcomes)
    resolved = supports + opposes
    result = binomtest(supports, resolved, 0.5, alternative="greater")
    return {
        "supports": supports,
        "opposes": opposes,
        "tied": len(seed_outcomes) - resolved,
        "resolved": resolved,
        "one_sided_p": result.pvalue,
        "outcomes": seed_outcomes,
    }


def main() -> None:
    args = parse_args()
    rows, reruns_applied = selected_rows(args.csv_path, args.metric, args.fire_rerun_root)
    pairs = [test_pair(rows, *pair, args.metric) for pair in PAIRS]
    for pair, pair_definition in zip(pairs, PAIRS):
        pair["bonferroni_adjusted_p_for_3_tests"] = min(
            1.0, 3.0 * pair["exact_two_sided_sign_test_p"]
        )
        pair["seed_majority_check"] = seed_majority_check(
            rows, *pair_definition, args.metric
        )
        pair["seed_majority_check"]["bonferroni_adjusted_p_for_3_tests"] = min(
            1.0, 3.0 * pair["seed_majority_check"]["one_sided_p"]
        )
    report = {
        "csv_path": str(args.csv_path.resolve()),
        "selection": {
            "n_selected": len(rows),
            "criterion": "all three algorithms have the same first_drop_load_step",
        },
        "metric": args.metric,
        "fire_reruns": {
            "applied": reruns_applied,
            "root": None if args.fire_rerun_root is None else str(args.fire_rerun_root.resolve()),
        },
        "method": {
            "name": "exact paired sign test",
            "null_hypothesis": "after excluding equal stored energies, either algorithm is lower with probability 0.5 on each event",
            "ties": "equal stored energies are excluded as unresolved at CSV precision",
            "multiple_comparisons": "Bonferroni correction across the three planned pairwise tests",
        },
        "pairwise_results": pairs,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
