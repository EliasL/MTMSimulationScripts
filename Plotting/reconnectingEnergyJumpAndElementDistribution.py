#!/usr/bin/env python3
"""Plot element distributions and energy/stress jumps across reconnections.

The simulation folder must contain ``macroData.csv`` and the
``beforeReconnectionVtuData`` folder produced by
``Management.vtuBeforeReconnectionExtraction``.  Completed event folders are
snapshotted once at startup.  The live macro CSV is opened read-only, read only
up to its size at open time, and its possibly incomplete final line is dropped.
"""

import argparse
import io
import os
import re
import sys
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Management.updateCSV import update_df_header
from MTMath.energyFunction import ContiEnergy
from MTMath.poincareEnergy import C2PoincareDisk, prepPoincareFig
from Plotting.dataFunctions import VTUData
from Plotting.plotPowerLaw import findPrePostSplit
from Management.reconnectionJobSelection import SimulationJob, discover_simulation_jobs
from Management.vtuBeforeReconnectionExtraction import reconnection_pairs


EXTRACTION_FOLDER_NAME = "beforeReconnectionVtuData"
EVENT_FOLDER_PATTERN = re.compile(r"(?P<dump>.+)_step(?P<step>\d+)$")
LOAD_PATTERN = re.compile(r"_load=(?P<load>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)_")
MIN_STEP_PATTERN = re.compile(r"_minStep=(?P<iteration>\d+)\.(?P<call>\d+)_pre\.")
REGIMES = ("pre-yield", "post-yield")
STATES = ("before", "after")
REGIME_COLORS = {"pre-yield": "#3b75af", "post-yield": "#d97827"}
# Temporary while the size-scaling simulations are still incomplete.  Set this
# to ``None`` (or use ``--no-forced-pre-yield``) once all jobs are complete.
FORCED_PRE_YIELD_BELOW = 0.6
SINGLE_ELEMENT_REFERENCE_LOAD = 0.5
SINGLE_ELEMENT_REFERENCE_AREA = 0.5  # MTS2D's reference triangle area
LOSS_OF_ELLIPTICITY_LOAD = ContiEnergy.simpleShearStabilityLimit
SINGLE_ELEMENT_MAX_ENERGY = float(
    np.asarray(
        ContiEnergy.energy_from_simpleShear(
            SINGLE_ELEMENT_REFERENCE_LOAD,
            zeroReference=True,
        )
    ).squeeze()
    * SINGLE_ELEMENT_REFERENCE_AREA
)
LOSS_OF_ELLIPTICITY_SIGMA12 = float(
    ContiEnergy.cauchy_from_F(
        np.array([[1.0, LOSS_OF_ELLIPTICITY_LOAD], [0.0, 1.0]])
    )[0, 1]
)


@dataclass(frozen=True)
class PairSpec:
    event_folder: Path
    dump_name: str
    log_step: int
    reconnection_index: int
    min_iteration: int
    min_function_call: int
    load: float
    before: Path
    after: Path
    simulation_name: str = ""
    size: Optional[int] = None
    seed: Optional[int] = None
    yield_load: Optional[float] = None


@dataclass(frozen=True)
class VtuState:
    histogram: np.ndarray
    total_energy: float
    average_sigma12: float
    flipped_total_energy: float
    flipped_average_sigma12: float
    nr_elements: int
    nr_reconnected_elements: int


@dataclass(frozen=True)
class LocalFlipData:
    """Per-pair element values; columns preserve the C++ element indices."""

    element_pairs: np.ndarray
    before_energy: np.ndarray
    short_after_energy: np.ndarray
    long_after_energy: np.ndarray
    before_sigma12: np.ndarray
    short_after_sigma12: np.ndarray
    long_after_sigma12: np.ndarray


class NoReconnectionError(ValueError):
    """Raised for the final pre/post pair where reconnect() changed nothing."""


class TopologyDecompositionError(ValueError):
    """Raised when a VTU pair is not a unique two-element edge flip."""


def read_live_macro_snapshot(csv_path: Path) -> pd.DataFrame:
    """Read a fixed, complete-line snapshot without locking or writing files."""
    csv_path = Path(csv_path)
    with csv_path.open("rb") as stream:
        snapshot_size = os.fstat(stream.fileno()).st_size
        raw = stream.read(snapshot_size)
    final_newline = raw.rfind(b"\n")
    if final_newline < 0:
        raise ValueError(f"The macro CSV has no complete lines: {csv_path}")
    complete = raw[: final_newline + 1]

    wanted = {
        "load",
        "Load",
        "avg_sigma12",
        "avg_sigmaxy",
    }
    frame = pd.read_csv(
        io.BytesIO(complete),
        comment="#",
        usecols=lambda column: column in wanted,
        low_memory=False,
    )
    frame = update_df_header(frame, add_total_columns=False)
    if frame.empty:
        raise ValueError(f"The macro CSV snapshot contains no data rows: {csv_path}")
    if "load" not in frame:
        raise KeyError(f"No load column found in {csv_path}")
    frame["load"] = pd.to_numeric(frame["load"], errors="raise")
    if "avg_sigma12" not in frame:
        raise KeyError(f"No avg_sigma12 column found in {csv_path}")
    frame["avg_sigma12"] = pd.to_numeric(frame["avg_sigma12"], errors="raise")
    return frame


def determine_yield_load(macro_snapshot: pd.DataFrame) -> float:
    """Reuse the existing stress-maximum split, using sigma12 explicitly."""
    yield_load = float(findPrePostSplit(df=macro_snapshot))
    if not np.isfinite(yield_load):
        raise ValueError(f"Non-finite yield load: {yield_load}")
    return yield_load


def _warning_handle(forced_pre_yield_below: Optional[float]) -> Optional[Line2D]:
    if forced_pre_yield_below is None:
        return None
    return Line2D(
        [],
        [],
        linestyle="none",
        marker="",
        label=rf"Warn: $\gamma<{forced_pre_yield_below:g}$ pre-yield",
    )


def _scientific_count(count: int) -> str:
    """Format a non-zero sample count as one-decimal LaTeX scientific notation."""
    if count < 0:
        raise ValueError(f"Counts must be non-negative, got {count}.")
    if count == 0:
        return "0"
    mantissa, exponent = f"{int(count):.1e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def _legend_with_warning(
    ax: plt.Axes,
    forced_pre_yield_below: Optional[float],
    **kwargs,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    warning = _warning_handle(forced_pre_yield_below)
    if warning is not None:
        handles.append(warning)
        labels.append(warning.get_label())
    ax.legend(handles, labels, frameon=True, **kwargs)


def parse_reconnection_index(value: str) -> Union[int, str]:
    if value.lower() == "all":
        return "all"
    try:
        index = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a positive integer or 'all'.") from error
    if index < 1:
        raise argparse.ArgumentTypeError("Reconnection indices are one-based and positive.")
    return index


def _min_step_key(pair: Tuple[Path, Path]) -> Tuple[int, int]:
    match = MIN_STEP_PATTERN.search(pair[0].name)
    if match is None:
        raise ValueError(f"Could not parse minimization order from {pair[0].name}")
    return int(match.group("iteration")), int(match.group("call"))


def _load_from_vtu_name(path: Path) -> float:
    match = LOAD_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse load from {path.name}")
    return float(match.group("load"))


def snapshot_event_folders(extraction_folder: Path) -> List[Path]:
    """Return only atomically published event folders present at call time."""
    folders = []
    for path in list(extraction_folder.iterdir()):
        match = EVENT_FOLDER_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            folders.append(path)
    return sorted(folders, key=lambda path: int(EVENT_FOLDER_PATTERN.fullmatch(path.name).group("step")))


def select_pairs(
    event_folder: Path, selection: Union[int, str]
) -> List[Tuple[int, Tuple[Path, Path]]]:
    pairs = sorted(reconnection_pairs(event_folder), key=_min_step_key)
    if not pairs:
        raise ValueError(f"No matched pre/post VTU pairs in {event_folder}")
    if selection == "all":
        return list(enumerate(pairs, start=1))
    if selection > len(pairs):
        raise ValueError(
            f"Requested reconnection {selection}, but {event_folder.name} "
            f"contains only {len(pairs)} pair(s)."
        )
    return [(selection, pairs[selection - 1])]


def collect_pair_specs(
    extraction_folder: Path, selection: Union[int, str]
) -> Tuple[List[PairSpec], int]:
    event_folders = snapshot_event_folders(extraction_folder)
    if not event_folders:
        raise FileNotFoundError(f"No completed event folders found in {extraction_folder}")

    specs = []
    for event_folder in event_folders:
        folder_match = EVENT_FOLDER_PATTERN.fullmatch(event_folder.name)
        for ordinal, pair in select_pairs(event_folder, selection):
            before, after = pair
            load = _load_from_vtu_name(before)
            after_load = _load_from_vtu_name(after)
            if not np.isclose(load, after_load, rtol=0.0, atol=1e-12):
                raise ValueError(f"Pre/post load mismatch: {before.name}, {after.name}")
            min_iteration, min_call = _min_step_key(pair)
            specs.append(
                PairSpec(
                    event_folder=event_folder,
                    dump_name=folder_match.group("dump"),
                    log_step=int(folder_match.group("step")),
                    reconnection_index=ordinal,
                    min_iteration=min_iteration,
                    min_function_call=min_call,
                    load=load,
                    before=before,
                    after=after,
                )
            )
    return specs, len(event_folders)


def _reference_triangle_connectivity(data: VTUData, path: Path) -> np.ndarray:
    """Map VTU-local (including ghost) point indices to stable reference indices."""
    triangle_blocks = [cell.data for cell in data.mesh.cells if cell.type == "triangle"]
    if len(triangle_blocks) != 1:
        raise ValueError(
            f"Expected exactly one triangle block in {path}, found {len(triangle_blocks)}."
        )
    connectivity = np.asarray(triangle_blocks[0], dtype=np.int64)
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(f"Unexpected triangle connectivity shape in {path}: {connectivity.shape}")
    if "refIndex" not in data.mesh.point_data:
        raise KeyError(f"Missing point field 'refIndex' in {path}")
    ref_index = np.asarray(data.mesh.point_data["refIndex"], dtype=float).reshape(-1)
    if ref_index.shape != (len(data.mesh.points),):
        raise ValueError(f"Unexpected refIndex shape in {path}: {ref_index.shape}")
    if not np.all(np.isfinite(ref_index)) or not np.all(ref_index == np.rint(ref_index)):
        raise ValueError(f"Non-integer refIndex values in {path}")
    return ref_index.astype(np.int64)[connectivity]


def _edge_flip_element_pairs(
    before_connectivity: np.ndarray,
    after_connectivity: np.ndarray,
    before_path: Path,
    after_path: Path,
) -> np.ndarray:
    """Pair element rows that exchange a quadrilateral's diagonal.

    VTU cell rows follow ``TElement::eIndex``.  MTS2D recreates each selected
    child at its parent's existing index, so equal row indices are the
    shortest-node-move (``shortFlip``) lineage and swapping rows within a pair
    gives ``longFlip``.
    """
    changed = np.flatnonzero(
        np.any(
            np.sort(before_connectivity, axis=1)
            != np.sort(after_connectivity, axis=1),
            axis=1,
        )
    )
    if changed.size == 0:
        raise NoReconnectionError(
            f"No reconnected elements found in {before_path}, {after_path}"
        )

    edge_to_elements: Dict[Tuple[int, int], List[int]] = {}
    for element_index in changed:
        for edge in combinations(before_connectivity[element_index], 2):
            key = tuple(sorted(int(node) for node in edge))
            edge_to_elements.setdefault(key, []).append(int(element_index))

    candidates = {int(index): [] for index in changed}
    for old_shared_edge, element_indices in edge_to_elements.items():
        if len(element_indices) != 2:
            continue
        first, second = element_indices
        before_nodes = set(before_connectivity[first]) | set(
            before_connectivity[second]
        )
        after_nodes = set(after_connectivity[first]) | set(after_connectivity[second])
        new_shared_edge = set(after_connectivity[first]) & set(
            after_connectivity[second]
        )
        if (
            len(before_nodes) == 4
            and before_nodes == after_nodes
            and len(new_shared_edge) == 2
            and new_shared_edge != set(old_shared_edge)
        ):
            candidates[first].append(second)
            candidates[second].append(first)

    invalid = {index: partners for index, partners in candidates.items() if len(partners) != 1}
    if invalid:
        preview = dict(list(invalid.items())[:10])
        raise TopologyDecompositionError(
            "Could not decompose changed VTU cells into unique two-element "
            f"edge flips for {before_path}, {after_path}. Invalid candidates: {preview}"
        )

    pairs = np.asarray(
        sorted(
            (index, partners[0])
            for index, partners in candidates.items()
            if index < partners[0]
        ),
        dtype=np.int64,
    )
    if pairs.shape != (changed.size // 2, 2) or not np.array_equal(
        np.sort(pairs.reshape(-1)), changed
    ):
        raise TopologyDecompositionError(
            f"Flip pairs do not cover every changed element in {before_path}, {after_path}"
        )
    return pairs


def _reduce_vtu_state(
    data: VTUData,
    path: Path,
    matrix_name: str,
    edges: np.ndarray,
    reconnected_mask: np.ndarray,
) -> VtuState:
    if matrix_name == "C":
        matrices = data.get_C()
    elif matrix_name == "G":
        matrices = data.get_G()
    elif matrix_name == "T_total":
        T_total = data.get_T_total()
        matrices = T_total.swapaxes(-1, -2) @ T_total
    else:
        raise ValueError(f"Unknown Poincare matrix: {matrix_name}")
    energy = np.asarray(data.get_energy_field(), dtype=float)
    sigma12 = np.asarray(data.get_cell_data("sigma12"), dtype=float)
    nr_elements = len(matrices)
    if energy.shape != (nr_elements,) or sigma12.shape != (nr_elements,):
        raise ValueError(
            f"Inconsistent element-field shapes in {path}: "
            f"matrix={matrices.shape}, energy={energy.shape}, "
            f"sigma12={sigma12.shape}"
        )
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(sigma12)):
        raise ValueError(f"Non-finite energy or sigma12 values in {path}")
    if reconnected_mask.shape != (nr_elements,):
        raise ValueError(
            f"Reconnected-element mask has shape {reconnected_mask.shape}, "
            f"expected {(nr_elements,)} for {path}."
        )

    x, y = C2PoincareDisk(matrices[reconnected_mask])
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"Non-positive-definite {matrix_name} values in {path}")
    if np.any(x * x + y * y > 1.0 + 1e-10):
        raise ValueError(f"{matrix_name} mapped outside the Poincare disk in {path}")
    histogram = np.histogram2d(x, y, bins=(edges, edges))[0].T
    nr_reconnected = int(reconnected_mask.sum())
    if int(histogram.sum()) != nr_reconnected:
        raise ValueError(f"Poincare histogram lost elements from {path}")
    return VtuState(
        histogram=histogram,
        total_energy=float(np.sum(energy)),
        average_sigma12=float(np.mean(sigma12)),
        flipped_total_energy=float(np.sum(energy[reconnected_mask])),
        flipped_average_sigma12=float(np.mean(sigma12[reconnected_mask])),
        nr_elements=nr_elements,
        nr_reconnected_elements=nr_reconnected,
    )


def read_vtu_pair_details(
    before_path: Path, after_path: Path, matrix_name: str, edges: np.ndarray
) -> Tuple[VtuState, VtuState, LocalFlipData]:
    """Read a pair once and return aggregate and individual-flip values."""
    before_data = VTUData(str(before_path))
    after_data = VTUData(str(after_path))
    before_connectivity = _reference_triangle_connectivity(before_data, before_path)
    after_connectivity = _reference_triangle_connectivity(after_data, after_path)
    if before_connectivity.shape != after_connectivity.shape:
        raise ValueError(
            f"Triangle count changed across reconnection: {before_path}, {after_path}"
        )
    element_pairs = _edge_flip_element_pairs(
        before_connectivity,
        after_connectivity,
        before_path,
        after_path,
    )
    reconnected_mask = np.zeros(before_connectivity.shape[0], dtype=bool)
    reconnected_mask[element_pairs.reshape(-1)] = True
    before_energy = np.asarray(before_data.get_energy_field(), dtype=float)
    after_energy = np.asarray(after_data.get_energy_field(), dtype=float)
    before_sigma12 = np.asarray(before_data.get_cell_data("sigma12"), dtype=float)
    after_sigma12 = np.asarray(after_data.get_cell_data("sigma12"), dtype=float)
    for name, values in (
        ("before energy", before_energy),
        ("after energy", after_energy),
        ("before sigma12", before_sigma12),
        ("after sigma12", after_sigma12),
    ):
        if values.shape != (before_connectivity.shape[0],):
            raise ValueError(f"Unexpected {name} shape {values.shape} in {before_path}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite {name} values in {before_path}, {after_path}")

    short_after_indices = element_pairs
    long_after_indices = element_pairs[:, ::-1]
    local = LocalFlipData(
        element_pairs=element_pairs,
        before_energy=before_energy[element_pairs],
        short_after_energy=after_energy[short_after_indices],
        long_after_energy=after_energy[long_after_indices],
        before_sigma12=before_sigma12[element_pairs],
        short_after_sigma12=after_sigma12[short_after_indices],
        long_after_sigma12=after_sigma12[long_after_indices],
    )
    return (
        _reduce_vtu_state(before_data, before_path, matrix_name, edges, reconnected_mask),
        _reduce_vtu_state(after_data, after_path, matrix_name, edges, reconnected_mask),
        local,
    )


def read_vtu_pair(
    before_path: Path, after_path: Path, matrix_name: str, edges: np.ndarray
) -> Tuple[VtuState, VtuState]:
    """Read aggregate values for one pre/post reconnection pair."""
    before, after, _ = read_vtu_pair_details(
        before_path, after_path, matrix_name, edges
    )
    return before, after


def analyze_pairs(
    specs: List[PairSpec],
    yield_load: Optional[float],
    matrix_name: str,
    bins: int,
    skip_no_reconnection: bool = False,
    skip_invalid_topology: bool = False,
    forced_pre_yield_below: Optional[float] = FORCED_PRE_YIELD_BELOW,
) -> Tuple[
    pd.DataFrame,
    Dict[Tuple[str, str], np.ndarray],
    Dict[Tuple[str, str], int],
    pd.DataFrame,
    pd.DataFrame,
]:
    edges = np.linspace(-1.0, 1.0, bins + 1)
    histograms = {
        (regime, state): np.zeros((bins, bins), dtype=np.int64)
        for regime in REGIMES
        for state in STATES
    }
    sample_counts = {key: 0 for key in histograms}
    rows = []
    pair_rows = []
    lineage_rows = []

    for number, spec in enumerate(specs, start=1):
        event_yield_load = spec.yield_load if spec.yield_load is not None else yield_load
        if event_yield_load is None or not np.isfinite(event_yield_load):
            raise ValueError(f"No finite yield load is available for {spec.event_folder}")
        if (
            forced_pre_yield_below is not None
            and spec.load < forced_pre_yield_below
        ):
            regime = "pre-yield"
        else:
            regime = "pre-yield" if spec.load <= event_yield_load else "post-yield"
        print(
            f"[{number}/{len(specs)}] {spec.event_folder.name}, "
            f"reconnection {spec.reconnection_index}, load={spec.load:g}",
            flush=True,
        )
        try:
            before, after, local = read_vtu_pair_details(
                spec.before, spec.after, matrix_name, edges
            )
        except NoReconnectionError:
            if not skip_no_reconnection:
                raise
            print("  Skipping terminal no-change pre/post pair.", flush=True)
            continue
        except TopologyDecompositionError as error:
            if not skip_invalid_topology:
                raise
            print(f"  Skipping non-edge-flip topology: {error}", flush=True)
            continue
        if before.nr_elements != after.nr_elements:
            raise ValueError(
                f"Element count changed across reconnection: {spec.before}, {spec.after}"
            )
        histograms[(regime, "before")] += before.histogram.astype(np.int64)
        histograms[(regime, "after")] += after.histogram.astype(np.int64)
        sample_counts[(regime, "before")] += 1
        sample_counts[(regime, "after")] += 1
        rows.append(
            {
                "event_folder": spec.event_folder.name,
                "dump_name": spec.dump_name,
                "log_step": spec.log_step,
                "reconnection_index": spec.reconnection_index,
                "min_iteration": spec.min_iteration,
                "min_function_call": spec.min_function_call,
                "load": spec.load,
                "simulation_name": spec.simulation_name,
                "size": spec.size,
                "seed": spec.seed,
                "yield_load": event_yield_load,
                "regime": regime,
                "before_vtu": spec.before.name,
                "after_vtu": spec.after.name,
                "nr_elements": before.nr_elements,
                "nr_reconnected_elements": before.nr_reconnected_elements,
                "before_total_energy": before.total_energy,
                "after_total_energy": after.total_energy,
                "energy_shift": after.total_energy - before.total_energy,
                "before_avg_sigma12": before.average_sigma12,
                "after_avg_sigma12": after.average_sigma12,
                "stress_shift": after.average_sigma12 - before.average_sigma12,
                "before_flipped_total_energy": before.flipped_total_energy,
                "after_flipped_total_energy": after.flipped_total_energy,
                "flipped_energy_shift": (
                    after.flipped_total_energy - before.flipped_total_energy
                ),
                "before_flipped_avg_sigma12": before.flipped_average_sigma12,
                "after_flipped_avg_sigma12": after.flipped_average_sigma12,
                "flipped_stress_shift": (
                    after.flipped_average_sigma12 - before.flipped_average_sigma12
                ),
            }
        )
        common = {
            "event_folder": spec.event_folder.name,
            "dump_name": spec.dump_name,
            "log_step": spec.log_step,
            "reconnection_index": spec.reconnection_index,
            "min_iteration": spec.min_iteration,
            "min_function_call": spec.min_function_call,
            "load": spec.load,
            "simulation_name": spec.simulation_name,
            "size": spec.size,
            "seed": spec.seed,
            "yield_load": event_yield_load,
            "regime": regime,
        }
        for pair_index, element_indices in enumerate(local.element_pairs, start=1):
            local_index = pair_index - 1
            pair_before_energy = float(local.before_energy[local_index].sum())
            pair_after_energy = float(local.short_after_energy[local_index].sum())
            pair_before_sigma12 = float(local.before_sigma12[local_index].mean())
            pair_after_sigma12 = float(local.short_after_sigma12[local_index].mean())
            pair_rows.append(
                {
                    **common,
                    "pair_index": pair_index,
                    "first_element_index": int(element_indices[0]),
                    "second_element_index": int(element_indices[1]),
                    "before_total_energy": pair_before_energy,
                    "after_total_energy": pair_after_energy,
                    "energy_shift": pair_after_energy - pair_before_energy,
                    "before_avg_sigma12": pair_before_sigma12,
                    "after_avg_sigma12": pair_after_sigma12,
                    "stress_shift": pair_after_sigma12 - pair_before_sigma12,
                }
            )
            for element_in_pair, before_element_index in enumerate(element_indices):
                partner = 1 - element_in_pair
                before_energy_value = float(
                    local.before_energy[local_index, element_in_pair]
                )
                before_sigma12_value = float(
                    local.before_sigma12[local_index, element_in_pair]
                )
                short_energy = float(
                    local.short_after_energy[local_index, element_in_pair]
                )
                long_energy = float(
                    local.long_after_energy[local_index, element_in_pair]
                )
                short_sigma12 = float(
                    local.short_after_sigma12[local_index, element_in_pair]
                )
                long_sigma12 = float(
                    local.long_after_sigma12[local_index, element_in_pair]
                )
                lineage_rows.append(
                    {
                        **common,
                        "pair_index": pair_index,
                        "element_in_pair": element_in_pair + 1,
                        "before_element_index": int(before_element_index),
                        "short_after_element_index": int(before_element_index),
                        "long_after_element_index": int(element_indices[partner]),
                        "before_energy": before_energy_value,
                        "short_after_energy": short_energy,
                        "long_after_energy": long_energy,
                        "short_energy_shift": short_energy - before_energy_value,
                        "long_energy_shift": long_energy - before_energy_value,
                        "before_sigma12": before_sigma12_value,
                        "short_after_sigma12": short_sigma12,
                        "long_after_sigma12": long_sigma12,
                        "short_stress_shift": short_sigma12 - before_sigma12_value,
                        "long_stress_shift": long_sigma12 - before_sigma12_value,
                    }
                )
    if not rows:
        raise ValueError("None of the selected pre/post pairs contained a reconnection.")
    if not pair_rows or not lineage_rows:
        raise ValueError("No twin-pair or element-lineage samples were accumulated.")
    return (
        pd.DataFrame(rows),
        histograms,
        sample_counts,
        pd.DataFrame(pair_rows),
        pd.DataFrame(lineage_rows),
    )


def _save_figure(fig: plt.Figure, output_stem: Path, formats: List[str]) -> None:
    for extension in formats:
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def plot_poincare_distributions(
    histograms: Dict[Tuple[str, str], np.ndarray],
    sample_counts: Dict[Tuple[str, str], int],
    matrix_name: str,
    yield_load: Optional[float],
    output_stem: Path,
    formats: List[str],
    forced_pre_yield_below: Optional[float] = None,
) -> None:
    probabilities = {}
    positive = []
    for key, histogram in histograms.items():
        total = histogram.sum()
        probability = histogram.astype(float) / total if total else histogram.astype(float)
        probabilities[key] = probability
        positive.extend(probability[probability > 0])
    if not positive:
        raise ValueError("No Poincare histogram samples were accumulated.")
    vmin, vmax = float(np.min(positive)), float(np.max(positive))
    if vmin == vmax:
        vmin, vmax = vmin / 2.0, vmax * 2.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.8), constrained_layout=True)
    image = None
    for row, regime in enumerate(REGIMES):
        for column, state in enumerate(STATES):
            ax = axes[row, column]
            probability = probabilities[(regime, state)]
            masked = np.ma.masked_where(probability <= 0, probability)
            image = ax.imshow(
                masked,
                origin="lower",
                extent=(0, probability.shape[0], 0, probability.shape[1]),
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                zorder=0,
            )
            prepPoincareFig(
                grid_size=probability.shape[0],
                ax=ax,
                withCircle=True,
                withGrid=True,
                withYieldSurface=False,
            )
            count = sample_counts[(regime, state)]
            nr_elements = int(histograms[(regime, state)].sum())
            ax.set_title(
                f"{regime.capitalize()}, {state} "
                rf"($n={_scientific_count(count)}$ events, "
                rf"$N={_scientific_count(nr_elements)}$ elements)"
            )
            if count == 0:
                ax.text(0.5, 0.5, "No completed events", transform=ax.transAxes, ha="center")
    fig.colorbar(
        image, ax=axes, label="Reconnected-element probability per bin", shrink=0.82
    )
    matrix_label = {
        "C": r"\mathbf{C}",
        "G": r"\mathbf{G}",
        "T_total": (
            r"\mathbf{T}_{\mathrm{total}}^\mathsf{T}"
            r"\mathbf{T}_{\mathrm{total}}"
        ),
    }[matrix_name]
    if yield_load is None:
        title = rf"Element ${matrix_label}$ distributions around reconnection; per-job yield loads"
    else:
        title = (
            rf"Element ${matrix_label}$ distributions around reconnection; "
            rf"$\gamma_\mathrm{{yield}}={yield_load:.5g}$"
        )
    fig.suptitle(title)
    warning = _warning_handle(forced_pre_yield_below)
    if warning is not None:
        fig.legend(
            [warning],
            [warning.get_label()],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            frameon=True,
            fontsize="small",
        )
    _save_figure(fig, output_stem, formats)


def _log_jump_bins(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot make distribution bins without finite values.")
    positive = np.abs(finite)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        raise ValueError("Cannot make logarithmic bins when all jumps are zero.")
    lower, upper = float(positive.min()), float(positive.max())
    if lower == upper:
        lower, upper = lower / np.sqrt(10.0), upper * np.sqrt(10.0)
    nr_bins = max(5, min(20, int(np.ceil(np.sqrt(positive.size) * 2))))
    return np.geomspace(lower, upper, nr_bins + 1)


# Keep the earlier private name available to existing analysis tests/scripts.
_log_shift_bins = _log_jump_bins


def plot_energy_stress(
    summary: pd.DataFrame,
    output_stem: Path,
    formats: List[str],
    scope: str = "mesh",
    forced_pre_yield_below: Optional[float] = None,
    show_top_row: bool = False,
) -> None:
    labels = {
        "mesh": (
            "",
            r"Total energy $E$",
            r"Average $\sigma_{12}$",
            r"Energy jump $|E_\mathrm{after}-E_\mathrm{before}|$",
            r"Stress jump $|\sigma_{12,\mathrm{after}}-\sigma_{12,\mathrm{before}}|$",
        ),
        "flipped_elements": (
            "flipped_",
            r"Flipped-element total energy $E$",
            r"Average flipped-element $\sigma_{12}$",
            r"Flipped-element energy jump $|E_\mathrm{after}-E_\mathrm{before}|$",
            r"Flipped-element stress jump "
            r"$|\sigma_{12,\mathrm{after}}-\sigma_{12,\mathrm{before}}|$",
        ),
        "twin_pair": (
            "",
            r"Twin-pair total energy $E$",
            r"Twin-pair average $\sigma_{12}$",
            r"Twin-pair energy jump $|E_\mathrm{after}-E_\mathrm{before}|$",
            r"Twin-pair stress jump "
            r"$|\sigma_{12,\mathrm{after}}-\sigma_{12,\mathrm{before}}|$",
        ),
        "shortFlip": (
            "",
            r"Flip element energy $E$",
            r"Flip element $\sigma_{12}$",
            r"Flip energy jump $|E_\mathrm{after}-E_\mathrm{before}|$",
            r"Flip stress jump "
            r"$|\sigma_{12,\mathrm{after}}-\sigma_{12,\mathrm{before}}|$",
        ),
        "longFlip": (
            "",
            r"longFlip element energy $E$",
            r"longFlip element $\sigma_{12}$",
            r"longFlip energy jump $|E_\mathrm{after}-E_\mathrm{before}|$",
            r"longFlip stress jump "
            r"$|\sigma_{12,\mathrm{after}}-\sigma_{12,\mathrm{before}}|$",
        ),
    }
    if scope not in labels:
        raise ValueError(f"Unknown energy/stress plot scope: {scope}")
    prefix, energy_label, stress_label, energy_shift_label, stress_shift_label = (
        labels[scope]
    )
    if show_top_row:
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), constrained_layout=True)
        shift_axes = axes[1]
        value_specs = (
            (
                f"before_{prefix}total_energy",
                f"after_{prefix}total_energy",
                energy_label,
                axes[0, 0],
            ),
            (
                f"before_{prefix}avg_sigma12",
                f"after_{prefix}avg_sigma12",
                stress_label,
                axes[0, 1],
            ),
        )
        for before_key, after_key, ylabel, ax in value_specs:
            for regime in REGIMES:
                data = summary[summary["regime"] == regime].sort_values(
                    ["load", "reconnection_index"]
                )
                color = REGIME_COLORS[regime]
                for row in data.itertuples():
                    ax.plot(
                        [row.load, row.load],
                        [getattr(row, before_key), getattr(row, after_key)],
                        color=color,
                        alpha=0.35,
                        linewidth=0.8,
                    )
                ax.scatter(
                    data["load"],
                    data[before_key],
                    s=24,
                    facecolors="none",
                    edgecolors=color,
                    label=f"{regime}, before",
                )
                ax.scatter(
                    data["load"],
                    data[after_key],
                    s=18,
                    color=color,
                    marker="x",
                    label=f"{regime}, after",
                )
            ax.set_xlabel(r"Load $\gamma$")
            ax.set_ylabel(ylabel)
            _legend_with_warning(
                ax,
                forced_pre_yield_below,
                fontsize="small",
                loc="best",
            )
    else:
        fig, shift_axes = plt.subplots(
            1, 2, figsize=(8.5, 3.3), constrained_layout=True
        )

    shift_specs = (
        (
            f"{prefix}energy_shift",
            energy_shift_label,
            shift_axes[0],
            "energy",
        ),
        (
            f"{prefix}stress_shift",
            stress_shift_label,
            shift_axes[1],
            "stress",
        ),
    )
    for key, xlabel, ax, quantity in shift_specs:
        edges = _log_jump_bins(np.asarray(summary[key], dtype=float))
        total_zero_count = 0
        for regime in REGIMES:
            values = np.abs(
                np.asarray(summary.loc[summary["regime"] == regime, key], dtype=float)
            )
            if values.size == 0:
                continue
            positive = values[values > 0.0]
            if positive.size == 0:
                total_zero_count += int(values.size)
                continue
            total_zero_count += int(values.size - positive.size)
            weights = np.full(positive.size, 1.0 / positive.size)
            label = rf"{regime} ($n={_scientific_count(int(positive.size))}$)"
            ax.hist(
                positive,
                bins=edges,
                weights=weights,
                histtype="step",
                linewidth=1.8,
                color=REGIME_COLORS[regime],
                label=label,
            )
        if total_zero_count:
            ax.plot(
                [],
                [],
                linestyle="none",
                label=rf"Zeros omitted ($n={total_zero_count}$)",
            )
        if quantity == "energy":
            reference = SINGLE_ELEMENT_MAX_ENERGY
            reference_label = (
                rf"single-element $E_\mathrm{{max}}$ "
                rf"($\gamma={SINGLE_ELEMENT_REFERENCE_LOAD:g}$)"
            )
        else:
            reference = LOSS_OF_ELLIPTICITY_SIGMA12
            reference_label = (
                rf"$\sigma_{{12}}$ at loss of ellipticity "
                rf"($\gamma={LOSS_OF_ELLIPTICITY_LOAD:.4f}$)"
            )
        ax.axvline(
            reference,
            color="0.25",
            linestyle="--",
            linewidth=1.1,
            label=reference_label,
        )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability per logarithmic bin")
        _legend_with_warning(
            ax,
            forced_pre_yield_below,
            loc="best",
        )
    _save_figure(fig, output_stem, formats)


def _lineage_plot_frame(lineage: pd.DataFrame, lineage_name: str) -> pd.DataFrame:
    if lineage_name not in {"short", "long"}:
        raise ValueError("lineage_name must be 'short' or 'long'.")
    result = lineage[["load", "regime", "reconnection_index"]].copy()
    result["before_total_energy"] = lineage["before_energy"]
    result["after_total_energy"] = lineage[f"{lineage_name}_after_energy"]
    result["energy_shift"] = lineage[f"{lineage_name}_energy_shift"]
    result["before_avg_sigma12"] = lineage["before_sigma12"]
    result["after_avg_sigma12"] = lineage[f"{lineage_name}_after_sigma12"]
    result["stress_shift"] = lineage[f"{lineage_name}_stress_shift"]
    return result


def plot_size_comparison_distributions(
    lineages: pd.DataFrame,
    output_stem: Path,
    formats: List[str],
    forced_pre_yield_below: Optional[float] = None,
) -> None:
    """Compare Flip jump distributions across system sizes and regimes."""
    if "size" not in lineages:
        raise KeyError("Size-comparison plots require a 'size' column.")
    sizes = sorted(
        int(size)
        for size in lineages["size"].dropna().unique()
    )
    if not sizes:
        raise ValueError("No system sizes are available for comparison.")

    size_colors = {
        size: plt.get_cmap("tab10")(index % 10)
        for index, size in enumerate(sizes)
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    specs = (
        ("short_energy_shift", r"Flip energy jump", "energy"),
        ("short_stress_shift", r"Flip stress jump", "stress"),
    )
    for row, regime in enumerate(REGIMES):
        regime_data = lineages[lineages["regime"] == regime]
        for column, (key, xlabel, quantity) in enumerate(specs):
            ax = axes[row, column]
            all_values = np.abs(
                np.asarray(regime_data[key], dtype=float)
            )
            edges = _log_jump_bins(all_values)
            total_zero_count = 0
            for size in sizes:
                values = np.abs(
                    np.asarray(
                        regime_data.loc[regime_data["size"] == size, key],
                        dtype=float,
                    )
                )
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                positive = values[values > 0.0]
                total_zero_count += int(values.size - positive.size)
                if positive.size == 0:
                    continue
                ax.hist(
                    positive,
                    bins=edges,
                    weights=np.full(positive.size, 1.0 / positive.size),
                    histtype="step",
                    linewidth=1.6,
                    color=size_colors[size],
                    label=rf"L={size} ($n={_scientific_count(int(positive.size))}$)",
                )
            if total_zero_count:
                ax.plot(
                    [],
                    [],
                    linestyle="none",
                    label=rf"Zeros omitted ($n={total_zero_count}$)",
                )
            if quantity == "energy":
                reference = SINGLE_ELEMENT_MAX_ENERGY
                reference_label = (
                    rf"single-element $E_\mathrm{{max}}$ "
                    rf"($\gamma={SINGLE_ELEMENT_REFERENCE_LOAD:g}$)"
                )
            else:
                reference = LOSS_OF_ELLIPTICITY_SIGMA12
                reference_label = (
                    rf"$\sigma_{{12}}$ at loss of ellipticity "
                    rf"($\gamma={LOSS_OF_ELLIPTICITY_LOAD:.4f}$)"
                )
            ax.axvline(
                reference,
                color="0.25",
                linestyle="--",
                linewidth=1.1,
                label=reference_label,
            )
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Probability per logarithmic bin")
            ax.set_title(regime.capitalize())
            _legend_with_warning(ax, forced_pre_yield_below, loc="best")
    fig.suptitle("Flip jump distributions by system size")
    _save_figure(fig, output_stem, formats)


def _prepare_job_specs(
    job: SimulationJob,
    selection: Union[int, str],
    yield_load: Optional[float],
) -> Tuple[List[PairSpec], int, float]:
    extraction_folder = job.folder / EXTRACTION_FOLDER_NAME
    macro_path = job.folder / "macroData.csv"
    if not extraction_folder.is_dir():
        raise FileNotFoundError(f"Missing extraction folder: {extraction_folder}")
    if not macro_path.is_file():
        raise FileNotFoundError(f"Missing macro CSV: {macro_path}")
    macro_snapshot = read_live_macro_snapshot(macro_path)
    resolved_yield = determine_yield_load(macro_snapshot) if yield_load is None else yield_load
    if not np.isfinite(resolved_yield):
        raise ValueError(f"Non-finite yield load for {job.folder}: {resolved_yield}")
    specs, nr_event_folders = collect_pair_specs(extraction_folder, selection)
    specs = [
        replace(
            spec,
            simulation_name=job.folder.name,
            size=job.size,
            seed=job.seed,
            yield_load=resolved_yield,
        )
        for spec in specs
    ]
    print(
        f"{job.folder.name}: {len(macro_snapshot)} complete macro rows, "
        f"yield load={resolved_yield:g}, {nr_event_folders} completed event folders, "
        f"{len(specs)} selected pair(s).",
        flush=True,
    )
    return specs, nr_event_folders, resolved_yield


def run_analysis_many(
    jobs: List[SimulationJob],
    selection: Union[int, str] = "all",
    matrix_name: str = "C",
    bins: int = 180,
    yield_load: Optional[float] = None,
    output_folder: Optional[Path] = None,
    formats: Optional[List[str]] = None,
    output_prefix: Optional[str] = None,
    skip_invalid_topology: bool = False,
    forced_pre_yield_below: Optional[float] = FORCED_PRE_YIELD_BELOW,
    show_top_row: bool = False,
) -> pd.DataFrame:
    if not jobs:
        raise ValueError("At least one simulation job is required.")
    if matrix_name not in {"C", "G", "T_total"}:
        raise ValueError("matrix_name must be 'C', 'G', or 'T_total'.")
    if bins < 10:
        raise ValueError("bins must be at least 10.")
    formats = ["pdf"] if formats is None else formats
    if not formats or any(extension not in {"png", "pdf"} for extension in formats):
        raise ValueError("formats must contain 'png', 'pdf', or both.")
    if forced_pre_yield_below is not None and forced_pre_yield_below <= 0:
        raise ValueError("forced_pre_yield_below must be positive or None.")

    specs = []
    resolved_yields = []
    for job in jobs:
        job_specs, _, job_yield = _prepare_job_specs(job, selection, yield_load)
        specs.extend(job_specs)
        resolved_yields.append(job_yield)
    if not specs:
        raise ValueError("No selected reconnection pairs were found.")

    if output_folder is None:
        output_folder = (
            jobs[0].folder / EXTRACTION_FOLDER_NAME / "plots"
            if len(jobs) == 1 and jobs[0].job_type == "single"
            else jobs[0].folder.parent / "plots"
        )
    else:
        output_folder = output_folder.expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    common_yield = resolved_yields[0] if all(
        np.isclose(value, resolved_yields[0], rtol=0.0, atol=1e-12)
        for value in resolved_yields[1:]
    ) else None
    summary, histograms, sample_counts, twin_pairs, lineages = analyze_pairs(
        specs,
        common_yield,
        matrix_name,
        bins,
        skip_no_reconnection=(selection == "all"),
        skip_invalid_topology=skip_invalid_topology,
        forced_pre_yield_below=forced_pre_yield_below,
    )
    selection_tag = str(selection)
    stem = (
        f"{output_prefix}_reconnection_{selection_tag}"
        if output_prefix
        else f"reconnection_{selection_tag}"
    )
    summary_path = output_folder / f"{stem}_event_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    twin_pair_path = output_folder / f"{stem}_twin_pair_summary.csv"
    twin_pairs.to_csv(twin_pair_path, index=False)
    print(f"Saved {twin_pair_path}")
    lineage_path = output_folder / f"{stem}_element_lineage_summary.csv"
    lineages.to_csv(lineage_path, index=False)
    print(f"Saved {lineage_path}")
    plot_poincare_distributions(
        histograms,
        sample_counts,
        matrix_name,
        common_yield,
        output_folder / f"{stem}_poincare_{matrix_name}",
        formats,
        forced_pre_yield_below=forced_pre_yield_below,
    )
    plot_energy_stress(
        summary,
        output_folder / f"{stem}_energy_stress",
        formats,
        forced_pre_yield_below=forced_pre_yield_below,
        show_top_row=show_top_row,
    )
    plot_energy_stress(
        summary,
        output_folder / f"{stem}_flipped_elements_energy_stress",
        formats,
        scope="flipped_elements",
        forced_pre_yield_below=forced_pre_yield_below,
        show_top_row=show_top_row,
    )
    plot_energy_stress(
        twin_pairs,
        output_folder / f"{stem}_twin_pair_energy_stress",
        formats,
        scope="twin_pair",
        forced_pre_yield_below=forced_pre_yield_below,
        show_top_row=show_top_row,
    )
    for lineage_name in ("short", "long"):
        plot_energy_stress(
            _lineage_plot_frame(lineages, lineage_name),
            output_folder / f"{stem}_{lineage_name}Flip_energy_stress",
            formats,
            scope=f"{lineage_name}Flip",
            forced_pre_yield_below=forced_pre_yield_below,
            show_top_row=show_top_row,
        )
    if lineages["size"].notna().any():
        plot_size_comparison_distributions(
            lineages,
            output_folder / f"{stem}_shortFlip_size_comparison_distributions",
            formats,
            forced_pre_yield_below=forced_pre_yield_below,
        )
    return summary


def run_analysis(
    simulation_folder: Path,
    selection: Union[int, str] = "all",
    matrix_name: str = "C",
    bins: int = 180,
    yield_load: Optional[float] = None,
    output_folder: Optional[Path] = None,
    formats: Optional[List[str]] = None,
    forced_pre_yield_below: Optional[float] = FORCED_PRE_YIELD_BELOW,
    show_top_row: bool = False,
) -> pd.DataFrame:
    simulation_folder = simulation_folder.expanduser().resolve()
    return run_analysis_many(
        [SimulationJob(simulation_folder, "single", None, None)],
        selection=selection,
        matrix_name=matrix_name,
        bins=bins,
        yield_load=yield_load,
        output_folder=output_folder,
        formats=formats,
        forced_pre_yield_below=forced_pre_yield_below,
        show_top_row=show_top_row,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "simulation_folder",
        type=Path,
        nargs="?",
        help="single simulation folder (omit when using --job-type)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Volumes/data/MTS2D_output/sizeScalingJobs"),
        help="root containing job folders for aggregate analysis",
    )
    parser.add_argument(
        "--job-type",
        choices=("size-scaling",),
        default=None,
        help="analyze all discovered jobs of this type instead of one folder",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="restrict aggregate analysis to one system size (for example 100)",
    )
    parser.add_argument(
        "--reconnection-index",
        type=parse_reconnection_index,
        default="all",
        help="One-based reconnection within each event, or 'all' (default: all).",
    )
    parser.add_argument(
        "--matrix", choices=("C", "G", "T_total"), default="C"
    )
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument(
        "--yield-load",
        type=float,
        help="Override the stress-maximum yield load inferred from macroData.csv.",
    )
    parser.add_argument("--output-folder", type=Path)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=["pdf"],
    )
    parser.add_argument(
        "--skip-invalid-topology",
        action="store_true",
        help="skip VTU pairs that are not uniquely decomposable edge flips",
    )
    parser.add_argument(
        "--show-top-row",
        action="store_true",
        help="also show before/after values versus load above the jump distributions",
    )
    parser.set_defaults(force_pre_yield_below=FORCED_PRE_YIELD_BELOW)
    cutoff_group = parser.add_mutually_exclusive_group()
    cutoff_group.add_argument(
        "--force-pre-yield-below",
        dest="force_pre_yield_below",
        type=float,
        help=(
            "temporarily classify loads below this value as pre-yield "
            f"(default: {FORCED_PRE_YIELD_BELOW:g})"
        ),
    )
    cutoff_group.add_argument(
        "--no-forced-pre-yield",
        dest="force_pre_yield_below",
        action="store_const",
        const=None,
        help="use only the yield load inferred from each macro CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.job_type is not None:
        if args.simulation_folder is not None:
            raise ValueError("simulation_folder cannot be combined with --job-type")
        if args.size is not None and args.size <= 0:
            raise ValueError("--size must be positive")
        jobs = discover_simulation_jobs(
            args.data_root,
            job_type=args.job_type,
            size=args.size,
            require_extraction=True,
        )
        if not jobs:
            raise FileNotFoundError(
                f"No extracted {args.job_type} jobs found in {args.data_root}"
            )
        size_tag = "all" if args.size is None else str(args.size)
        output_prefix = f"{args.job_type.replace('-', '_')}_L{size_tag}"
        run_analysis_many(
            jobs,
            selection=args.reconnection_index,
            matrix_name=args.matrix,
            bins=args.bins,
            yield_load=args.yield_load,
            output_folder=args.output_folder,
            formats=args.formats,
            output_prefix=output_prefix,
            skip_invalid_topology=args.skip_invalid_topology,
            forced_pre_yield_below=args.force_pre_yield_below,
            show_top_row=args.show_top_row,
        )
        return

    if args.size is not None:
        raise ValueError("--size requires --job-type")
    if args.simulation_folder is None:
        raise ValueError("provide simulation_folder or use --job-type")
    run_analysis(
        args.simulation_folder,
        selection=args.reconnection_index,
        matrix_name=args.matrix,
        bins=args.bins,
        yield_load=args.yield_load,
        output_folder=args.output_folder,
        formats=args.formats,
        forced_pre_yield_below=args.force_pre_yield_below,
        show_top_row=args.show_top_row,
    )


if __name__ == "__main__":
    main()
