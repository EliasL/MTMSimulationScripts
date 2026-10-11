"""Render downloaded, post-yield five-state examples with matching-load fields.

The scatter inset uses the full standard setting population from cached macro
CSVs. The highlighted event retains measurements from its downloaded states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Plotting import meshEventPlotting as mesh_plot
from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import calculate_energy_step_data
from .acquisition import state_paths_from_directory
from .catalog import build_standard_scatter_catalog
from .models import RenderOptions
from .legacy_fields import recover_legacy_fields, current_area_cauchy_mean
from .render import _energy_color_limit, _load_states, render_event_pdf


def legacy_starting_cauchy_mean(state, volume: float) -> float:
    """Recover spatial shear stress from the legacy fixed-reference PK1.

    Legacy P uses F_Fix's reference, whereas the exported sigma used F.
    Each triangle has reference area 1/2. Summing A0 (P F_Fix.T)_12
    and dividing by current volume gives the current-area Cauchy average,
    without assigning a negative physical Jacobian to clockwise local bases.
    This reader is deliberately restricted to the non-reconnecting atlas.
    """
    fields = state.cell_fields
    required = {"P11", "P12", "F_Fix21", "F_Fix22"}
    if not required.issubset(fields):
        raise ValueError(f"Missing legacy force-reference fields: {required-fields.keys()}")
    if len(state.triangles) != round(2*volume) or "T11" in fields:
        raise ValueError("Expected a non-reconnecting, equal-reference-area atlas mesh.")
    points = state.points[state.triangles]
    edges = points[:, 1:] - points[:, :1]
    areas = .5*np.abs(edges[:, 0, 0]*edges[:, 1, 1]-edges[:, 0, 1]*edges[:, 1, 0])
    if np.any(areas <= 0) or not np.isclose(areas.sum(), volume, rtol=1e-6):
        raise ValueError("Atlas mesh does not preserve its reference volume.")
    pf12 = fields["P11"]*fields["F_Fix21"] + fields["P12"]*fields["F_Fix22"]
    if not np.all(np.isfinite(pf12)):
        raise ValueError("Non-finite legacy force-reference stress.")
    return float(.5*np.sum(pf12)/areas.sum())


def rebuild_atlas(state_root: Path, selection_path: Path, output_root: Path,
                  *, labels: list[str] | None = None, dpi: int = 500,
                  csv_root: Path = Path(".plot-data/reversibility")) -> pd.DataFrame:
    selected = pd.read_csv(selection_path)
    excluded = selected.loc[~selected.yield_regime.eq("post"), "atlas_label"].tolist()
    selected = selected[selected.yield_regime.eq("post")].copy()
    paths_by_label = {}
    audits = []
    for index, event in selected.iterrows():
        label = str(event.atlas_label)
        paths = state_paths_from_directory(state_root / label)
        paths_by_label[label] = paths
        states = _load_states(paths)
        metadata = get_metadata(str(event.job_name))
        volume = float(metadata["L"]) ** 2
        raw_states = states
        states = {name: recover_legacy_fields(state, volume) for name, state in states.items()}
        start = states["state0_min_gamma"]
        end = states["state2_relaxed_gamma_plus"]
        raw_sigma0 = float(np.mean(start.cell_fields["legacy_exported_sigma12"]))
        sigma0 = current_area_cauchy_mean(start)
        energies = {name: float(np.sum(state.cell_fields["energy_field"]))
                    for name, state in states.items()}
        # Full-precision geometry must independently reproduce the recorded
        # macro increment; do not copy historical Delta E_S values.
        energy0 = energies["state0_min_gamma"]
        macro_change = -float(event.delta_E_inter_over_V0)*volume
        geometric_change = energies["state2_relaxed_gamma_plus"]-energy0
        if not np.isclose(geometric_change, macro_change, rtol=1e-10, atol=1e-8):
            raise ValueError(f"{label}: reconstructed energies disagree with the original macro increment.")
        frame = pd.DataFrame({
            "load": [event.event_start_load, event.load],
            "total_energy": [energy0, energies["state2_relaxed_gamma_plus"]],
            "avg_sigma12": [sigma0, current_area_cauchy_mean(end)],
        })
        steps, _ = calculate_energy_step_data(
            df=frame, metadata=metadata, average_energy=False)
        delta_es = float(steps.stress_corrected_drop_second_order.iloc[0]) / volume
        if not np.isfinite(delta_es) or delta_es <= 0:
            raise ValueError(f"{label} is not a positive gamma-zero Delta E_S event.")
        selected.loc[index, "delta_E_S_over_V0"] = delta_es
        selected.loc[index, "saved_event_directory"] = str(state_root / label)
        limits = {}
        for name, first, second in (
            ("forward", "state1_affine_gamma_plus", "state2_relaxed_gamma_plus"),
            ("backward", "state3_affine_gamma_minus", "state4_relaxed_gamma"),
        ):
            relation = mesh_plot.determine_topology_relation(states[first], states[second])
            values, _ = mesh_plot.calculate_energy_change_field(
                states[first], states[second], relation=relation)
            limit = _energy_color_limit(values, 0.97)
            limits[name] = {"symmetric_limit": limit,
                            "saturated_fraction": float(np.mean(np.abs(values) > limit))}
        shared_limit = max(field["symmetric_limit"] for field in limits.values())
        audits.append({"atlas_label": label, "avg_sigma12_start": sigma0,
                       "incorrect_unweighted_exported_sigma12": raw_sigma0,
                       "corrected_delta_E_S_over_V0": delta_es,
                       "recorded_macro_delta_E_inter_over_V0": float(event.delta_E_inter_over_V0),
                       "vtu_sum_delta_E_inter_over_V0": (energy0-energies["state2_relaxed_gamma_plus"])/volume,
                       "shared_color_limit": shared_limit,
                       "macro_vs_geometry_energy_change_error": geometric_change-macro_change,
                       "legacy_exported_total_energies": {name: float(np.sum(state.cell_fields["energy_field"])) for name, state in raw_states.items()},
                       "state_total_energies": energies, "color_limits": limits})

    output_root.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_root / "selection.csv", index=False)
    scatter_catalog = build_standard_scatter_catalog(csv_root=csv_root)
    scatter_catalog = scatter_catalog[scatter_catalog.yield_regime.eq("post")].copy()
    scatter_catalog["data_origin"] = "cached_macro"
    downloaded = selected.copy()
    downloaded["data_origin"] = "downloaded_states"
    scatter_catalog = pd.concat([scatter_catalog, downloaded], ignore_index=True)
    scatter_catalog.to_csv(output_root / "scatter_catalog.csv", index=False)
    excluded_gap_transitions = []
    for job in scatter_catalog.job_name.unique():
        loads = pd.read_csv(csv_root / job / "macroData.csv", usecols=["load"]).load
        gaps = loads.diff() > 1.5e-5
        for index in loads.index[gaps]:
            excluded_gap_transitions.append({"job_name": job,
                                             "start_load": float(loads.iloc[index-1]),
                                             "end_load": float(loads.iloc[index])})
    options = RenderOptions(output_root=output_root, rasterized_dpi=dpi,
                            energy_color_quantile=0.97, recover_legacy_geometry_fields=True)
    rendered = []
    for _, event in selected.iterrows():
        label = str(event.atlas_label)
        if labels is not None and label not in labels:
            continue
        for extension in ("pdf", "png"):
            output = output_root / (label + "." + extension)
            print(f"Rendering {output}", flush=True)
            render_event_pdf(event, paths_by_label[label], output, options,
                             setting_catalog=scatter_catalog)
        rendered.append({"atlas_label": label, "pdf": str(output_root/(label+".pdf")),
                         "png": str(output_root/(label+".png"))})
    result = pd.DataFrame(rendered)
    result.to_csv(output_root / "atlas_manifest.csv", index=False)
    (output_root / "provenance.json").write_text(json.dumps({
        "state_root": str(state_root.resolve()),
        "selection_source": str(selection_path.resolve()),
        "yield_selection": "original saved post-yield labels; no additional strain buffer",
        "excluded_pre_yield": excluded,
        "scatter_population": "full standard epsilon_x=1e-6 setting plus downloaded examples; post-yield only",
        "scatter_csv_root": str(csv_root.resolve()),
        "scatter_event_count": len(scatter_catalog),
        "highlight_source": "energy and Cauchy stress recomputed independently from full-precision saved geometry",
        "stress_recovery": "Conti-square Cauchy stress from triangle-edge deformation, current-area weighted",
        "energy_increment_source": "reconstructed geometry; validated against original macro increment",
        "mesh_energy_source": "full-precision geometry, never rounded legacy scalar fields",
        "excluded_gap_transitions": excluded_gap_transitions,
        "delta_E_S_tangent": "gamma0",
        "energy_field_pairs": [[1, 2], [3, 4]],
        "color_quantile_of_absolute_values": 0.97,
        "color_limit_policy": "max of the two 97th-percentile absolute-field limits, shared by both columns",
        "quiver_head_length_fraction": 0.30,
        "quiver_head_width_fraction": 0.24,
        "events": audits,
    }, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path,
                        default=Path.home()/"Downloads/atlas_nonreconnecting_states")
    parser.add_argument("--selection", type=Path,
                        default=Path("Plots/real_space_events/atlas_nonreconnecting/selection.csv"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("Plots/real_space_events/atlas_nonreconnecting_relaxation_postyield"))
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--dpi", type=int, default=500)
    parser.add_argument("--csv-root", type=Path, default=Path(".plot-data/reversibility"))
    args = parser.parse_args()
    rebuild_atlas(args.state_root, args.selection, args.output_root,
                  labels=args.labels, dpi=args.dpi, csv_root=args.csv_root)


if __name__ == "__main__":
    main()
