"""Zoom in on one energy drop and its two next-step predictions.

The plot is deliberately aligned to the *end* of a load step.  At relative
step ``0`` the measured solution is the post-drop ``E_{n+1}``, while

``E_S = E_{n+1} + Delta E_S`` and ``E_R = E_{n+1} + Delta E_R``

show the energies predicted for that same solution before the corresponding
drop is applied.  ``Delta E_S`` is obtained from the existing second-order
stress-corrected energy calculation.  ``Delta E_R`` keeps the existing
``total_e_change_from_init`` convention used by the reversibility analyses.

Run from the repository root, for example::

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python -m \
        Plotting.plot_single_energy_drop

The default is drop number 100, ordered by size (largest first) among
positive post-yield ``Delta E_S`` drops, in the bundled 250x250
non-reconnecting seed-0 sample.  Use ``--drop-number`` to select another
size-ranked drop, ``--drop-order chronological`` for chronological numbering,
or ``--drop-row``/``--drop-load`` for an exact transition.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd

from Management.updateCSV import read_macrodata_csv
from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import (
    calculate_energy_step_data,
    extract_energy_drops_from_dataframe,
)
from Plotting.findXmin import find_xmin_simple_drop
from Plotting.plotPowerLaw import findPrePostSplit
from Plotting.standardPowerlaw import EventDrops, EventSplit, split_by_er


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / (
    "Plots/energy_prediction_normal_data/"
    "simpleShear,s250x250l0.15,1e-05,1.0PBCt8LBFGSEpsx1e-06s0/"
    "macroData.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "Plots/single_energy_drop/energy_drop_zoom.png"
S_DROP_COLUMN = "stress_corrected_drop_second_order"


@dataclass(frozen=True)
class DropTrace:
    """The aligned data needed to plot one selected transition."""

    csv_path: Path
    frame: pd.DataFrame
    drop_row: int
    step_index: int
    yield_load: float
    delta_E_S: float
    delta_E_R: float
    selection_measure: str
    drop_number: int | None
    drop_order: str
    event_class: str | None = None
    er_det: float | None = None
    shared_es_region: tuple[float, float] | None = None
    selection_label: str | None = None


@dataclass(frozen=True)
class ClassifiedDropPool:
    """Paired post-yield events classified from ``Delta E_R``.

    ``transition_indices`` maps each paired event back to the corresponding
    row in ``calculate_energy_step_data``.  The ``Delta E_S`` region is the
    intersection of the finite-positive ranges of the two classes, so every
    selected value is in a range occupied by both classes.
    """

    csv_path: Path
    event_drops: EventDrops
    split: EventSplit
    transition_indices: np.ndarray
    er_det: float
    shared_es_region: tuple[float, float]
    yield_load: float


def _finite_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def _as_float_array(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _aligned_drop_series(df: pd.DataFrame, step_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return measured energy, Delta E_S, and Delta E_R aligned to macro rows.

    ``calculate_energy_step_data`` stores step ``n -> n+1`` in row ``n`` of
    its result.  The corresponding drop is therefore placed at macro row
    ``n+1``.  The first macro row is intentionally left undefined because it
    is the initial state, not the endpoint of an event transition.
    """

    energy = _as_float_array(df["total_energy"], name="total_energy")
    if len(step_df) != len(df) - 1:
        raise ValueError(
            "Energy-step data must have one row per macro transition: "
            f"{len(step_df)} != {len(df) - 1}."
        )

    delta_E_S = np.full(len(df), np.nan, dtype=float)
    delta_E_S[1:] = _as_float_array(
        step_df[S_DROP_COLUMN], name=S_DROP_COLUMN
    )

    if "total_e_change_from_init" not in df:
        raise KeyError("Missing 'total_e_change_from_init' for Delta E_R.")

    # This is the existing Delta E_R convention used by the reversibility
    # extraction code: a positive relaxation drop is -total_e_change_from_init.
    # Keep signed values here so the trace remains continuous even at steps
    # whose relaxation contribution is not a positive drop.
    delta_E_R = -_as_float_array(
        df["total_e_change_from_init"], name="total_e_change_from_init"
    )
    delta_E_R[0] = np.nan
    return energy, delta_E_S, delta_E_R


def _positive_relaxation_drop_mask(
    df: pd.DataFrame, *, csv_path: Path
) -> np.ndarray:
    """Use the existing extraction helper to identify positive Delta E_R rows."""

    _, mask, _, _ = extract_energy_drops_from_dataframe(
        df,
        csv_file_path=str(csv_path),
        energy_key="total_e_change_from_init",
        stress_corrected=False,
        drop_sign="negative",
        min_drop=0.0,
    )
    return np.asarray(mask, dtype=bool)


@lru_cache(maxsize=4)
def _load_drop_inputs(csv_path_string: str):
    """Load and calculate one simulation once for repeated drop selections."""

    csv_path = Path(csv_path_string)
    df = read_macrodata_csv(csv_path)
    metadata = get_metadata(str(csv_path))
    step_df, _ = calculate_energy_step_data(
        csv_path,
        df=df,
        metadata=metadata,
        average_energy=False,
    )
    energy, delta_E_S, delta_E_R = _aligned_drop_series(df, step_df)
    er_positive_mask = _positive_relaxation_drop_mask(df, csv_path=csv_path)
    yield_load = float(findPrePostSplit(df=df))
    return (
        df,
        step_df,
        energy,
        delta_E_S,
        delta_E_R,
        er_positive_mask,
        yield_load,
    )


@lru_cache(maxsize=4)
def _classify_drop_inputs(csv_path_string: str) -> ClassifiedDropPool:
    """Classify paired post-yield events using the existing simpleDrop path."""

    csv_path = Path(csv_path_string)
    (
        df,
        step_df,
        _,
        delta_E_S,
        delta_E_R,
        er_positive_mask,
        yield_load,
    ) = _load_drop_inputs(csv_path_string)

    post_yield = _as_float_array(step_df["load_ip1"], name="load_ip1") > yield_load
    transition_mask = (
        post_yield
        & np.asarray(er_positive_mask[1:], dtype=bool)
        & np.isfinite(delta_E_R[1:])
        & (delta_E_R[1:] > 0)
    )
    transition_indices = np.flatnonzero(transition_mask)
    if transition_indices.size < 100:
        raise ValueError(
            "At least 100 positive post-yield Delta E_R events are required "
            f"for simpleDrop; found {transition_indices.size}."
        )

    event_drops = EventDrops(
        er=delta_E_R[1:][transition_mask],
        es=delta_E_S[1:][transition_mask],
    )
    er_det, _ = find_xmin_simple_drop(
        event_drops.er,
        progress=False,
    )
    split = split_by_er(event_drops, er_det)

    rev_es = _finite_positive(event_drops.es[split.is_rev])
    irrev_es = _finite_positive(event_drops.es[split.is_irrev])
    if rev_es.size == 0 or irrev_es.size == 0:
        raise ValueError(
            "Both Delta E_R classes must contain finite-positive Delta E_S "
            "values to define a shared region."
        )

    shared_lo = max(float(np.min(rev_es)), float(np.min(irrev_es)))
    shared_hi = min(float(np.max(rev_es)), float(np.max(irrev_es)))
    if not shared_hi >= shared_lo:
        raise ValueError(
            "The reversible and irreversible Delta E_S ranges do not overlap."
        )

    return ClassifiedDropPool(
        csv_path=csv_path,
        event_drops=event_drops,
        split=split,
        transition_indices=transition_indices,
        er_det=float(er_det),
        shared_es_region=(shared_lo, shared_hi),
        yield_load=float(yield_load),
    )


def load_classified_drop_pool(csv_path: str | Path = DEFAULT_CSV) -> ClassifiedDropPool:
    """Return the cached Delta E_R classification and shared Delta E_S range."""

    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"macroData.csv not found: {csv_path}")
    return _classify_drop_inputs(str(csv_path))


def _choose_step_index(
    df: pd.DataFrame,
    step_df: pd.DataFrame,
    delta_E_S: np.ndarray,
    delta_E_R: np.ndarray,
    *,
    yield_load: float,
    drop_number: int,
    drop_order: str,
    drop_measure: str,
    drop_row: int | None,
    drop_load: float | None,
    er_positive_mask: np.ndarray,
) -> int:
    """Choose a transition, using the endpoint macro row for direct selection."""

    if drop_number < 1:
        raise ValueError("drop_number must be at least 1.")
    if drop_measure not in {"S", "R"}:
        raise ValueError("drop_measure must be 'S' or 'R'.")
    if drop_order not in {"largest", "chronological"}:
        raise ValueError("drop_order must be 'largest' or 'chronological'.")

    explicit_count = int(drop_row is not None) + int(drop_load is not None)
    if explicit_count > 1:
        raise ValueError("Use at most one of drop_row and drop_load.")

    if drop_row is not None:
        if not 1 <= drop_row < len(df):
            raise IndexError(
                f"drop_row must be between 1 and {len(df) - 1}; got {drop_row}."
            )
        return drop_row - 1

    if drop_load is not None:
        endpoint_loads = _as_float_array(
            step_df["load_ip1"], name="step endpoint loads"
        )
        candidates = np.flatnonzero(
            np.isclose(endpoint_loads, float(drop_load), rtol=1e-9, atol=1e-12)
        )
        if candidates.size != 1:
            raise ValueError(
                f"Expected one transition ending at load {drop_load:g}; "
                f"found {candidates.tolist()}."
            )
        return int(candidates[0])

    if drop_measure == "S":
        # Delta E_S is aligned to endpoint rows; convert to step rows.
        values = delta_E_S[1:]
        positive = np.isfinite(values) & (values > 0)
    else:
        values = delta_E_R[1:]
        positive = er_positive_mask[1:] & np.isfinite(values)

    post_yield = _as_float_array(step_df["load_ip1"], name="load_ip1") > yield_load
    # Keep the two drop definitions paired: the event mask comes from the
    # existing positive-Delta-E_R extraction, and Delta E_S is only checked
    # after that mask is applied.
    paired_event = er_positive_mask[1:]
    candidates = np.flatnonzero(post_yield & paired_event & positive)
    if candidates.size < drop_number:
        raise ValueError(
            f"Only {candidates.size} positive post-yield Delta E_{drop_measure} "
            f"drops are available; cannot select number {drop_number}."
        )

    if drop_order == "chronological":
        return int(candidates[drop_number - 1])
    order = np.argsort(values[candidates], kind="stable")[::-1]
    return int(candidates[order[drop_number - 1]])


def _choose_classified_step_index(
    df: pd.DataFrame,
    step_df: pd.DataFrame,
    pool: ClassifiedDropPool,
    *,
    drop_number: int,
    drop_order: str,
    event_class: str,
    drop_row: int | None,
    drop_load: float | None,
) -> int:
    """Select one class-labeled event inside the shared Delta E_S range."""

    if drop_number < 1:
        raise ValueError("drop_number must be at least 1.")
    if event_class not in {"reversible", "irreversible"}:
        raise ValueError("event_class must be 'reversible' or 'irreversible'.")
    if drop_order not in {"largest", "chronological"}:
        raise ValueError("drop_order must be 'largest' or 'chronological'.")
    explicit_count = int(drop_row is not None) + int(drop_load is not None)
    if explicit_count > 1:
        raise ValueError("Use at most one of drop_row and drop_load.")

    if drop_row is not None:
        if not 1 <= drop_row < len(df):
            raise IndexError(
                f"drop_row must be between 1 and {len(df) - 1}; got {drop_row}."
            )
        return drop_row - 1

    if drop_load is not None:
        endpoint_loads = _as_float_array(
            step_df["load_ip1"], name="step endpoint loads"
        )
        candidates = np.flatnonzero(
            np.isclose(endpoint_loads, float(drop_load), rtol=1e-9, atol=1e-12)
        )
        if candidates.size != 1:
            raise ValueError(
                f"Expected one transition ending at load {drop_load:g}; "
                f"found {candidates.tolist()}."
            )
        return int(candidates[0])

    class_mask = (
        pool.split.is_rev
        if event_class == "reversible"
        else pool.split.is_irrev
    )
    es = pool.event_drops.es
    region_lo, region_hi = pool.shared_es_region
    in_shared_region = (
        np.isfinite(es)
        & (es > 0)
        & (es >= region_lo)
        & (es <= region_hi)
    )
    candidates = np.flatnonzero(class_mask & in_shared_region)
    if candidates.size < drop_number:
        raise ValueError(
            f"Only {candidates.size} {event_class} drops are available in "
            f"the shared Delta E_S region; cannot select number {drop_number}."
        )

    if drop_order == "chronological":
        selected_event = int(candidates[drop_number - 1])
    else:
        order = np.argsort(es[candidates], kind="stable")[::-1]
        selected_event = int(candidates[order[drop_number - 1]])
    return int(pool.transition_indices[selected_event])


def load_drop_trace(
    csv_path: str | Path,
    *,
    pre_steps: int = 10,
    drop_number: int = 100,
    drop_order: str = "largest",
    drop_measure: str = "S",
    drop_row: int | None = None,
    drop_load: float | None = None,
    event_class: str | None = None,
    selection_label: str | None = None,
) -> DropTrace:
    """Load the data and return a relative-step trace for one drop.

    ``drop_number`` is one-based. By default it ranks positive post-yield
    drops from largest to smallest; with ``drop_order="chronological"`` it
    counts them in source order. ``drop_row`` is the zero-based macro-data row
    *after* the drop.  Thus the
    transition being shown is ``drop_row - 1 -> drop_row``.  ``drop_load`` is
    the load value at that same endpoint row.
    """

    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"macroData.csv not found: {csv_path}")
    if pre_steps < 1:
        raise ValueError("pre_steps must be at least 1.")

    (
        df,
        step_df,
        energy,
        delta_E_S,
        delta_E_R,
        er_positive_mask,
        yield_load,
    ) = _load_drop_inputs(str(csv_path))

    classified_pool = None
    if event_class is None:
        step_index = _choose_step_index(
            df,
            step_df,
            delta_E_S,
            delta_E_R,
            yield_load=yield_load,
            drop_number=drop_number,
            drop_order=drop_order,
            drop_measure=drop_measure,
            drop_row=drop_row,
            drop_load=drop_load,
            er_positive_mask=er_positive_mask,
        )
    else:
        classified_pool = _classify_drop_inputs(str(csv_path))
        step_index = _choose_classified_step_index(
            df,
            step_df,
            classified_pool,
            drop_number=drop_number,
            drop_order=drop_order,
            event_class=event_class,
            drop_row=drop_row,
            drop_load=drop_load,
        )
    endpoint_row = step_index + 1
    first_row = endpoint_row - pre_steps
    if first_row < 0:
        raise ValueError(
            f"Drop endpoint row {endpoint_row} has only {endpoint_row} preceding "
            f"steps; cannot show {pre_steps}."
        )

    rows = np.arange(first_row, endpoint_row + 1, dtype=int)
    relative_step = rows - endpoint_row
    frame = pd.DataFrame(
        {
            "relative_step": relative_step,
            "macro_row": rows,
            "load_step": np.asarray(df["load_step"], dtype=float)[rows]
            if "load_step" in df
            else rows,
            "load": _as_float_array(df["load"], name="load")[rows],
            "E": energy[rows],
            "delta_E_S": delta_E_S[rows],
            "E_S": energy[rows] + delta_E_S[rows],
            "delta_E_R": delta_E_R[rows],
            "E_R": energy[rows] + delta_E_R[rows],
        }
    )
    return DropTrace(
        csv_path=csv_path,
        frame=frame,
        drop_row=endpoint_row,
        step_index=step_index,
        yield_load=yield_load,
        delta_E_S=float(delta_E_S[endpoint_row]),
        delta_E_R=float(delta_E_R[endpoint_row]),
        selection_measure=drop_measure,
        drop_number=None if (drop_row is not None or drop_load is not None) else drop_number,
        drop_order=drop_order,
        event_class=event_class,
        er_det=None if classified_pool is None else classified_pool.er_det,
        shared_es_region=(
            None
            if classified_pool is None
            else classified_pool.shared_es_region
        ),
        selection_label=selection_label,
    )


def plot_drop_trace(
    trace: DropTrace,
    *,
    output_path: str | Path | None = DEFAULT_OUTPUT,
    ylim_margin: float = 0.08,
    show: bool = False,
):
    """Plot log difference traces with the energy trace as an inset.

    The endpoint at relative step zero is intentionally omitted from the
    difference plot.  It remains visible in the energy inset, but it is not
    plotted and does not contribute to the logarithmic y-limits of the main
    axes.
    """

    if not np.isfinite(ylim_margin) or ylim_margin < 0:
        raise ValueError("ylim_margin must be finite and nonnegative.")

    frame = trace.frame
    x = frame["relative_step"].to_numpy(dtype=float)
    drop_mask = frame["relative_step"].to_numpy(dtype=int) == 0
    plotted_mask = ~drop_mask

    differences = {
        "S": frame["delta_E_S"].to_numpy(dtype=float),
        "R": frame["delta_E_R"].to_numpy(dtype=float),
    }
    colors = {"S": "tab:blue", "R": "tab:orange"}
    labels = {
        "S": r"$|E_S-E|=|\Delta E_S|$",
        "R": r"$|E_R-E|=|\Delta E_R|$",
    }

    finite_nonzero = []
    for difference in differences.values():
        values = np.abs(difference[plotted_mask])
        finite_nonzero.extend(values[np.isfinite(values) & (values > 0)])
    finite_nonzero = np.asarray(finite_nonzero, dtype=float)
    if finite_nonzero.size == 0:
        raise ValueError("No finite nonzero pre-drop differences are available.")

    # The final drop is excluded before calculating these limits.  Padding is
    # applied in log space so the displayed range remains appropriate for a
    # log y-axis.
    log_min = float(np.log10(np.min(finite_nonzero)))
    log_max = float(np.log10(np.max(finite_nonzero)))
    log_padding = ylim_margin * max(log_max - log_min, 1.0)
    difference_ylim = (
        10 ** (log_min - log_padding),
        10 ** (log_max + log_padding),
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for key, difference in differences.items():
        valid = plotted_mask & np.isfinite(difference) & (difference != 0)
        ax.plot(
            x[valid],
            np.abs(difference[valid]),
            color=colors[key],
            linewidth=1.0,
            alpha=0.65,
            zorder=1,
        )
        positive = valid & (difference > 0)
        negative = valid & (difference < 0)
        ax.scatter(
            x[positive],
            np.abs(difference[positive]),
            color=colors[key],
            marker="o",
            s=30,
            zorder=3,
        )
        ax.scatter(
            x[negative],
            np.abs(difference[negative]),
            color=colors[key],
            marker="x",
            s=42,
            linewidths=1.8,
            zorder=4,
        )

    ax.axvline(-0.5, color="0.55", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlim(float(x[0]) - 0.35, 0.35)
    ax.set_ylim(*difference_ylim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Steps relative to drop (0 = omitted final drop)")
    ax.set_ylabel("Absolute energy difference")
    ax.grid(alpha=0.22, which="both")

    legend_handles = [
        Line2D(
            [], [], color=colors[key], marker="o", linewidth=1.2,
            label=labels[key],
        )
        for key in ("S", "R")
    ]
    legend_handles.extend(
        [
            Line2D(
                [], [], color="0.2", marker="o", linestyle="None",
                label="positive difference",
            ),
            Line2D(
                [], [], color="0.2", marker="x", linestyle="None",
                markersize=7, label="negative difference",
            ),
        ]
    )
    ax.legend(handles=legend_handles, loc="lower left", frameon=True)

    # Upper-left placement keeps the inset clear of the lower-left legend;
    # 38% is approximately 20% smaller than the previous 47% dimensions.
    inset = inset_axes(ax, width="38%", height="38%", loc="upper left", borderpad=1.2)
    _plot_energy_inset(inset, trace, ylim_margin=ylim_margin)

    if trace.selection_label is not None:
        selection_label = trace.selection_label
    elif trace.drop_number is not None:
        selection_label = f"drop #{trace.drop_number} ({trace.drop_order})"
    else:
        selection_label = "explicitly selected drop"
    if trace.event_class is not None and trace.selection_label is None:
        selection_label = f"{trace.event_class} {selection_label}"
    title = (
        "250x250 non-reconnecting energy-drop zoom\n"
        rf"{selection_label}, load={frame['load'].iloc[-1]:.5f}, "
        rf"yield load={trace.yield_load:.5f}"
        "\n"
        rf"$\Delta E_S$={trace.delta_E_S:.6g}, $\Delta E_R$={trace.delta_E_R:.6g}"
    )
    if trace.shared_es_region is not None:
        region_lo, region_hi = trace.shared_es_region
        title += (
            "\n"
            rf"shared $\Delta E_S\in[{region_lo:.3g},\ {region_hi:.3g}]$, "
            rf"$\Delta E_{{R,\mathrm{{det}}}}$={trace.er_det:.3g}"
        )
    fig.suptitle(
        title,
        y=0.86 if trace.shared_es_region is not None else 0.97,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.10,
        top=0.70 if trace.shared_es_region is not None else 0.78,
    )

    if output_path is not None:
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        print(f'Plot saved at: "{output_path.resolve()}"')
    if show:
        plt.show()
    return fig, ax, difference_ylim


def _plot_energy_inset(ax, trace: DropTrace, *, ylim_margin: float) -> tuple[float, float]:
    """Draw the original energy view and exclude the endpoint from its limits."""

    frame = trace.frame
    x = frame["relative_step"].to_numpy(dtype=float)
    drop_mask = frame["relative_step"].to_numpy(dtype=int) == 0
    pre_drop = frame.loc[~drop_mask, ["E", "E_S", "E_R"]].to_numpy(dtype=float)
    finite_pre = pre_drop[np.isfinite(pre_drop)]
    if finite_pre.size == 0:
        raise ValueError("No finite pre-drop energies are available for inset limits.")

    y_min = float(np.min(finite_pre))
    y_max = float(np.max(finite_pre))
    span = y_max - y_min
    padding = ylim_margin * (span if span > 0 else max(abs(y_max), 1.0))
    ylim = (y_min - padding, y_max + padding)

    ax.plot(x, frame["E"], color="black", marker="o", linewidth=1.2, label=r"$E$")
    ax.plot(
        x,
        frame["E_S"],
        color="tab:blue",
        marker="s",
        linewidth=1.0,
        label=r"$E_S$",
    )
    ax.plot(
        x,
        frame["E_R"],
        color="tab:orange",
        marker="D",
        linewidth=1.0,
        label=r"$E_R$",
    )
    ax.axvline(-0.5, color="0.55", linestyle="--", linewidth=0.8)
    ax.scatter(
        [0],
        [frame.loc[drop_mask, "E"].iloc[0]],
        color="black",
        marker="v",
        s=26,
        zorder=4,
    )
    ax.set_xlim(float(x[0]) - 0.35, 0.35)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("relative step", fontsize=8)
    ax.set_ylabel(r"$E$", fontsize=8)
    ax.set_title("Energy inset", fontsize=8)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", fontsize=6, frameon=True)
    return ylim


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--drop-row",
        type=int,
        help="Zero-based macro row after the drop; transition is row-1 -> row.",
    )
    selector.add_argument(
        "--drop-load",
        type=float,
        help="Load value at the post-drop endpoint row.",
    )
    selector.add_argument(
        "--drop-number",
        type=int,
        default=100,
        help="One-based size rank among positive post-yield drops (largest first; default: 100).",
    )
    parser.add_argument(
        "--drop-order",
        choices=("largest", "chronological"),
        default="largest",
        help="Order used by --drop-number (default: largest).",
    )
    parser.add_argument(
        "--drop-measure",
        choices=("S", "R"),
        default="S",
        help="Measure used to count automatic selections (default: S).",
    )
    parser.add_argument("--pre-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ylim-margin",
        type=float,
        default=0.08,
        help="Fractional padding around pre-drop energies only (default: 0.08).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    trace = load_drop_trace(
        args.csv,
        pre_steps=args.pre_steps,
        drop_number=args.drop_number,
        drop_order=args.drop_order,
        drop_measure=args.drop_measure,
        drop_row=args.drop_row,
        drop_load=args.drop_load,
    )
    print(
        f"Selected {trace.drop_number or 'explicit'} drop at macro row "
        f"{trace.drop_row} (transition step {trace.step_index}) "
        f"at load {trace.frame['load'].iloc[-1]:.8g}; "
        f"yield load={trace.yield_load:.8g}; "
        f"Delta E_S={trace.delta_E_S:.8g}; Delta E_R={trace.delta_E_R:.8g}"
    )
    plot_drop_trace(
        trace,
        output_path=args.output,
        ylim_margin=args.ylim_margin,
    )


if __name__ == "__main__":
    main()
