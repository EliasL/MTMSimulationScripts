#!/usr/bin/env python3
"""Render the complete saved irreversible cohort with vectors binned at their starts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from Plotting.render_largest_reconnecting_events import (
    DEFAULT_JOB,
    ROOT,
    render_aggregate_largest_event_summary,
    select_events,
)
from Plotting.render_poincare_maxflow_heatmaps import render_maxflow_heatmaps
from Plotting.render_stress_maxflow_heatmaps import render_stress_maxflow_heatmaps


DEFAULT_OUTPUT = ROOT / "Plots/irreversible_events_all110_start_binned"
DEFAULT_NEGATIVE_OUTPUT = ROOT / "Plots/irreversible_events_negative66_start_binned"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=234)
    parser.add_argument("--stage", choices=("all", "flow", "density"), default="all")
    parser.add_argument("--negative-energy-change", action="store_true")
    args = parser.parse_args()
    if args.dpi <= 0:
        raise ValueError("dpi must be positive.")
    sign = "negative" if args.negative_energy_change else "both"
    events = select_events(args.job, number=None, saved_only=True, energy_change_sign=sign)
    if any(not event.state_directory.name.startswith("irrev_drop_l_") for event in events):
        raise ValueError("The selected cohort contains a state not marked irreversible.")
    expected_count = 66 if args.negative_energy_change else 110
    if args.job == DEFAULT_JOB and len(events) != expected_count:
        raise ValueError(f"Expected {expected_count} complete saved irreversible events, found {len(events)}.")
    if args.output_directory is None:
        args.output_directory = DEFAULT_NEGATIVE_OUTPUT if args.negative_energy_change else DEFAULT_OUTPUT
    args.output_directory.mkdir(parents=True, exist_ok=True)
    with (args.output_directory / "events.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("load_step", "start_load", "target_load", "equilibrium_energy_change", "state_directory"))
        for event in sorted(events, key=lambda item: item.start_load):
            writer.writerow((event.load_step, event.start_load, event.target_load, -event.energy_drop, event.state_directory))
    print(f"Selected {len(events)} saved irreversible events.", flush=True)

    if args.stage in {"all", "flow"}:
        magnitude_vmin = None if args.negative_energy_change else 1e-5
        print("Rendering Poincare flow: start bins, no final endpoint reduction.", flush=True)
        render_maxflow_heatmaps(
            events, args.output_directory / "poincare_maxflow_start_binned.png",
            dpi=args.dpi, zoom=3.5, coordinate_source="before",
            magnitude_vmin=magnitude_vmin,
        )
        print("Rendering direct-VTU stress flow: start bins.", flush=True)
        render_stress_maxflow_heatmaps(
            events, args.output_directory / "stress_maxflow_start_binned.png",
            dpi=args.dpi, vector_source="local_stress", coordinate_source="before",
            magnitude_vmin=magnitude_vmin,
        )
    if args.stage in {"all", "density"}:
        print("Rendering before densities and changes between fully reduced states.", flush=True)
        paths = render_aggregate_largest_event_summary(
            events, args.output_directory, dpi=args.dpi,
            fully_reduce_after=True,
        )
        names = (
            "poincare_density_after_full", "poincare_density_after", "stress_density_after",
            "poincare_density_before_full", "poincare_density_before", "stress_density_before",
            "poincare_density_change", "stress_density_change",
        )
        for path, name in zip(paths, names, strict=True):
            destination = args.output_directory
            if name.endswith("_full") or name.endswith("_after"):
                destination = destination / "supplementary"
                destination.mkdir(exist_ok=True)
            for suffix in (".png", ".pdf"):
                source = path.with_suffix(suffix)
                if source.is_file():
                    source.replace(destination / (name + suffix))
        cohort = "negative equilibrium-energy change" if args.negative_energy_change else "either sign of equilibrium-energy change"
        (args.output_directory / "README.md").write_text(
            f"# {len(events)} saved irreversible events\n\n"
            f"Source: `{args.job}`. All complete saved `irrev` events with {cohort} are used.\n\n"
            "Poincare vectors: reduce the before metric using M, evaluate both T_before @ M and T_after @ M, "
            "and subtract their disk coordinates. The after endpoint can leave the elastic well. "
            "There is no final endpoint reduction. Spatial bins use the before coordinates.\n\n"
            "Stress vectors: subtract the direct VTU laboratory-frame Cauchy-stress coordinates of the same "
            "element slot; bin at the before stress coordinates.\n\n"
            "Maxflow maps use 300 x 300 bins. Each flow value is the mean absolute projection on the "
            "cell's maximal-projection axis, not a vector count or a signed net flux. "
            "Zero-motion cells have no defined axis. Vectors are included when their starting points "
            "lie in the viewport, even when their endpoints lie outside it.\n\n"
            + ("The absolute projected flow colorbar starts at 1e-5; lower values use its lowest color.\n\n"
               if not args.negative_energy_change else "") +
            "The main density maps show the before states on 200 x 200 bins. "
            "Poincare density changes compare the reduced before state to the fully reduced after state. "
            "The final reduction is applied to density endpoints only; vector endpoints use the pre-event reduction. "
            "Changes are after minus before counts on a shared grid: red means an increase, blue a decrease, "
            "and zero is transparent. Only points inside the density viewport contribute.\n\n"
            "The Poincare zoom is 3.5; stress maxflow limits are +/-1.2. Supplementary figures include "
            "after-density and full-disk views. `events.csv` records every selected event.\n",
            encoding="utf-8",
        )
    print(args.output_directory, flush=True)


if __name__ == "__main__":
    main()
