"""Construct and sample the event catalogue from macro data.

Classification must use the setting-specific Otsu cut in Delta_rev u and the
forward-step m3 count.  ``is_reversible`` and rev/irrev directory names are
never classification inputs.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import calculate_energy_step_data, volume_from_metadata
from Plotting.plotPowerLaw import findPrePostSplit
from Plotting import numericalParameterJustification as npj

from .models import AnalysisScope, EventClass, RepresentativeKind


CATALOG_COLUMNS = (
    "event_id",
    "job_name",
    "seed",
    "reconnection_mode",
    "epsilon_x",
    "delta_gamma",
    "load",
    "yield_regime",
    "rev_u_cut",
    "delta_rev_u",
    "reversibility_measured",
    "forward_m3_changes",
    "event_class",
    "delta_E_inter_over_V0",
    "delta_E_S_over_V0",
    "participation_fraction",
    "saved_event_directory",
)


def classify_event(
    *,
    delta_rev_u: float,
    rev_u_cut: float,
    forward_m3_changes: int,
    reversibility_measured: bool,
) -> EventClass:
    """Apply the independent reversibility and forward-plasticity definitions."""

    if forward_m3_changes < 0:
        raise ValueError("forward_m3_changes cannot be negative.")
    if not reversibility_measured:
        return EventClass.REVERSIBILITY_UNMEASURED
    closing = delta_rev_u <= rev_u_cut
    plastic = forward_m3_changes > 0
    if closing and plastic:
        return EventClass.REVERSIBLE_PLASTIC
    if closing:
        return EventClass.REVERSIBLE_ELASTIC
    if plastic:
        return EventClass.IRREVERSIBLE_PLASTIC
    return EventClass.IRREVERSIBLE_ELASTIC


def build_catalog(scope: AnalysisScope) -> pd.DataFrame:
    """Build one row per candidate event for non-reconnecting and reconnecting data.

    Reuse ``numericalParameterJustification`` for the second-order energy
    correction and unbinned log-Otsu split.  A zero reversibility record caused
    by the C++ early return is *unmeasured*, not reversible.  The implementation
    must also map saved event directories by job and load without trusting the
    directory's rev/irrev prefix.
    """

    from Management.configGenerator import ConfigGenerator
    from Management.jobs import sylvainBatches
    from Plotting.remotePlotting import get_csv_files

    rows = []
    for batch in scope.batches:
        configs, labels = sylvainBatches(batch, reconnection="none")
        grouped_configs, grouped_labels, _ = ConfigGenerator.group_by_settings(
            configs, labels=labels
        )
        csv_groups, _ = get_csv_files(
            grouped_configs,
            labels=grouped_labels,
            useOldFiles=False,
            forceUpdate=False,
        )
        if csv_groups is None:
            raise RuntimeError(f"No macro CSVs found for Sylvain batch {batch}.")
        for group in csv_groups:
            for csv_path in group:
                rows.extend(_catalog_rows_from_csv(Path(csv_path), scope, batch))
    result = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    if result.empty:
        raise RuntimeError("The event catalogue is empty.")
    return result


def build_catalog_from_job(job_directory: Path, *, batch: int = 0) -> pd.DataFrame:
    """Build a catalogue from one complete local simulation directory."""

    return pd.DataFrame(
        _catalog_rows_from_csv(Path(job_directory) / "macroData.csv", AnalysisScope(), batch),
        columns=CATALOG_COLUMNS,
    )


def build_standard_scatter_catalog(
    *, batch: int = -2, setting: float = 1e-6
) -> pd.DataFrame:
    """Build the exact positive-``Delta E_S`` population used by the standard scatter.

    This deliberately reuses ``numericalParameterJustification.load_batch`` and
    ``reversibleOnlyEnergyAnalysis.build_classifications``.  The resulting rows
    contain one row per plotted event, including the standard closing,
    non-closing and discarded-population labels.
    """

    from Plotting import numericalParameterJustification as npj
    from Plotting.reversibleOnlyEnergyAnalysis import build_classifications

    if batch not in (-2, -1):
        raise ValueError("The standard scatter catalogue supports batches -2 and -1.")
    samples = npj.load_batch(batch)
    attribute = "eps_x" if batch == -2 else "load_increment"
    groups = npj._setting_groups(samples, attribute)
    matching = [value for value in groups if np.isclose(value, setting)]
    if len(matching) != 1:
        raise ValueError(f"Expected one setting matching {setting:g}; found {matching}.")
    setting_value = matching[0]
    classifications = build_classifications(samples, attribute)
    classification = classifications[setting_value]
    rows = []
    for sample in groups[setting_value]:
        masks = (
            ("discarded", classification.discarded_masks[sample.path]),
            ("closing", classification.final_masks[sample.path]),
            ("nonclosing", classification.nonclosing_masks[sample.path]),
        )
        for population, population_mask in masks:
            valid = (
                population_mask
                & npj.real_energy_drop_mask(sample)
                & np.isfinite(sample.rev_u)
                & (sample.rev_u > 0)
                & np.isfinite(sample.energy_drop_density)
                & (sample.energy_drop_density > 0)
            )
            for index in np.flatnonzero(valid):
                rows.append(
                    {
                        "event_id": f"{sample.path}:{index}",
                        # ``load_batch`` may expose CSVs from the compact
                        # remote-data cache rather than from their original
                        # simulation directories.  Reuse the existing helper
                        # so the name remains the actual simulation job name
                        # in both layouts.
                        "job_name": npj._job_name(sample.path),
                        "seed": sample.seed,
                        "epsilon_x": sample.eps_x,
                        "delta_gamma": sample.load_increment,
                        "load": float(sample.gamma[index]),
                        "event_start_load": float(sample.gamma[index] - sample.load_increment),
                        "yield_regime": "post" if sample.post_yield[index] else "pre",
                        "rev_u_cut": classification.final_cut,
                        "delta_rev_u": float(sample.rev_u[index]),
                        "delta_E_inter_over_V0": float(
                            sample.inter_strain_energy_density[index]
                        ),
                        "delta_E_S_over_V0": float(sample.energy_drop_density[index]),
                        "forward_m3_changes": int(sample.m3_changes[index]),
                        "participation_fraction": float(
                            sample.participation_fraction[index]
                        ),
                        "population": population,
                        "event_class": (
                            "reversible_plastic"
                            if population == "closing" and sample.m3_changes[index] > 0
                            else "reversible_elastic"
                            if population == "closing"
                            else "irreversible_plastic"
                            if sample.m3_changes[index] > 0
                            else "irreversible_elastic"
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"No standard-scatter rows found for {attribute}={setting:g}.")
    return result


def _config_reconnection_mode(job_directory: Path) -> str:
    config_path = job_directory / "config.conf"
    if not config_path.is_file():
        return "unknown"
    matches = re.findall(
        r"^\s*reconnectionMethod\s*=\s*([^#\r\n]+)",
        config_path.read_text(),
        flags=re.MULTILINE,
    )
    return matches[0].strip() if len(matches) == 1 else "unknown"


def _complete_state_directory(path: Path) -> bool:
    prefixes = (
        "state0_min_gamma",
        "state1_affine_gamma_plus",
        "state2_relaxed_gamma_plus",
        "state3_affine_gamma_minus",
        "state4_relaxed_gamma",
    )
    return all(len(list(path.glob(f"{prefix}.*.vtu"))) == 1 for prefix in prefixes)


def _catalog_rows_from_csv(csv_path: Path, scope: AnalysisScope, batch: int) -> list[dict]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing macro CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {
        "load", "total_energy_change", "rev_u_diff", "rev_energy_diff",
        "rev_sigma_12_diff", "participationFraction",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing catalogue columns in {csv_path}: {missing}")
    m3_column = next(
        (name for name in ("nr_elements_with_m3_fix_change", "nr_elements_with_m3_change") if name in df),
        None,
    )
    if m3_column is None:
        raise KeyError(f"No forward m3-change column in {csv_path}.")
    metadata = get_metadata(str(csv_path))
    volume = volume_from_metadata(metadata)
    if volume is None or not np.isfinite(volume) or volume <= 0:
        raise ValueError(f"Could not infer mesh volume from {csv_path}.")
    energy_steps, _ = calculate_energy_step_data(
        str(csv_path), df=df, metadata=metadata, average_energy=False
    )
    if len(energy_steps) != len(df) - 1:
        raise RuntimeError(f"Energy-step length mismatch in {csv_path}.")
    load = df["load"].to_numpy(dtype=float)
    rev_u = df["rev_u_diff"].to_numpy(dtype=float)
    energy_drop_density = (
        energy_steps["stress_corrected_drop_second_order"].to_numpy(dtype=float)
        / float(volume)
    )
    positive_energy_drop = np.concatenate(
        ([False], np.isfinite(energy_drop_density) & (energy_drop_density > 0))
    )
    recorded = (
        (rev_u != 0)
        | (df["rev_energy_diff"].to_numpy(dtype=float) != 0)
        | (df["rev_sigma_12_diff"].to_numpy(dtype=float) != 0)
    ) & positive_energy_drop
    positive_rev_u = rev_u[recorded & (rev_u > 0)]
    if positive_rev_u.size < 50:
        raise ValueError(f"Too few recorded reversibility rows in {csv_path}.")
    rev_u_cut, _ = npj.unbinned_log_otsu_cut(positive_rev_u)
    yield_load = float(findPrePostSplit(df=df))
    load_increment = float(np.median(np.diff(load)))
    job_directory = csv_path.parent
    event_root = job_directory / "data" / "reversibilityData"
    event_directories = {}
    if event_root.is_dir():
        for path in event_root.iterdir():
            if not path.is_dir() or not _complete_state_directory(path):
                continue
            match = re.fullmatch(r"(?:rev|irrev)_drop_l_(?P<load>[0-9.eE+-]+)", path.name)
            if match is not None:
                event_directories[float(match.group("load"))] = path

    reconnection_mode = _config_reconnection_mode(job_directory)
    rows = []
    for row_index, row in df.iterrows():
        if row_index == 0:
            continue
        start_load = float(load[row_index - 1])
        target_load = float(load[row_index])
        saved_dir = event_directories.get(start_load)
        if not positive_energy_drop[row_index]:
            continue
        measured = bool(saved_dir is not None and recorded[row_index] and rev_u[row_index] > 0)
        forward_m3 = int(round(float(row[m3_column])))
        event_class = classify_event(
            delta_rev_u=float(rev_u[row_index]),
            rev_u_cut=rev_u_cut,
            forward_m3_changes=forward_m3,
            reversibility_measured=measured,
        )
        if not measured and float(row["total_energy_change"]) >= 0 and forward_m3 == 0:
            continue
        rows.append(
            {
                "event_id": f"{job_directory.name}:{target_load:.12g}",
                "job_name": job_directory.name,
                "seed": int(metadata.get("seed", -1)),
                "reconnection_mode": reconnection_mode,
                "epsilon_x": float(metadata.get("LBFGSEpsx", np.nan)),
                "delta_gamma": load_increment,
                "load": target_load,
                "yield_regime": "post" if target_load > yield_load else "pre",
                "rev_u_cut": float(rev_u_cut),
                "delta_rev_u": float(rev_u[row_index]),
                "reversibility_measured": measured,
                "forward_m3_changes": forward_m3,
                "event_class": event_class.value,
                "delta_E_inter_over_V0": -float(row["total_energy_change"]) / float(volume),
                "delta_E_S_over_V0": float(energy_drop_density[row_index - 1]),
                "participation_fraction": float(row["participationFraction"]),
                "saved_event_directory": str(saved_dir) if saved_dir is not None else "",
            }
        )
    return rows


def filter_catalog(catalog: pd.DataFrame, scope: AnalysisScope) -> pd.DataFrame:
    """Apply chosen settings and optional all/pre/post yield filtering."""

    required = set(CATALOG_COLUMNS)
    missing = required.difference(catalog.columns)
    if missing:
        raise KeyError(f"Catalogue is missing columns: {sorted(missing)}")
    result = catalog.copy()
    result = result[
        np.isfinite(result["delta_E_S_over_V0"])
        & (result["delta_E_S_over_V0"] > 0)
    ]
    if scope.epsilon_x is not None:
        result = result[
            np.isclose(result["epsilon_x"], scope.epsilon_x, equal_nan=False)
            | result["epsilon_x"].isna()
        ]
    if scope.delta_gamma is not None:
        result = result[
            np.isclose(result["delta_gamma"], scope.delta_gamma, equal_nan=False)
            | result["delta_gamma"].isna()
        ]
    if scope.yield_regime != "all":
        result = result[result["yield_regime"] == scope.yield_regime]
    return result.reset_index(drop=True)


def select_representatives(
    catalog: pd.DataFrame,
    *,
    examples_per_strategy: int = 1,
    strategies: tuple[RepresentativeKind, ...] = tuple(RepresentativeKind),
) -> pd.DataFrame:
    """Select typical, large-drop and high-participation examples per class.

    ``typical`` should be a medoid in robustly scaled event-feature space, not
    merely the event nearest the median of one variable.  Do not duplicate one
    event under multiple strategies unless its class has too few candidates.
    """

    if catalog.empty:
        raise ValueError("Cannot select representatives from an empty catalogue.")
    selected = []
    for event_class, class_rows in catalog.groupby("event_class", sort=True):
        candidates = class_rows[class_rows["reversibility_measured"]].copy()
        if candidates.empty:
            candidates = class_rows.copy()
        if candidates.empty:
            continue
        for strategy in strategies:
            if strategy is RepresentativeKind.TYPICAL:
                feature_names = (
                    "delta_rev_u", "delta_E_inter_over_V0",
                    "delta_E_S_over_V0", "participation_fraction",
                )
                values = np.log10(np.abs(candidates[list(feature_names)].to_numpy(float)) + 1e-300)
                values = (values - np.nanmedian(values, axis=0)) / (
                    np.nanstd(values, axis=0) + 1e-12
                )
                score = np.nansum(values**2, axis=1)
                order = np.argsort(score)
            elif strategy is RepresentativeKind.LARGE_INTERSTRAIN_DROP:
                order = np.argsort(-candidates["delta_E_inter_over_V0"].to_numpy(float))
            elif strategy is RepresentativeKind.HIGH_PARTICIPATION:
                order = np.argsort(-candidates["participation_fraction"].to_numpy(float))
            else:
                raise ValueError(f"Unhandled representative strategy {strategy}.")
            chosen = candidates.iloc[order[:examples_per_strategy]].copy()
            chosen["representative_kind"] = strategy.value
            selected.append(chosen)
    if not selected:
        raise RuntimeError("No representative events could be selected.")
    return pd.concat(selected, ignore_index=True).drop_duplicates("event_id")


def write_catalog(table: pd.DataFrame, path: Path) -> Path:
    """Write a stable CSV after validating the required schema."""

    path = Path(path)
    missing = set(CATALOG_COLUMNS).difference(table.columns)
    if missing:
        raise KeyError(f"Cannot write catalogue; missing {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path
