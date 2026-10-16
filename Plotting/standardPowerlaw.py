"""Small, explicit building blocks for the standard energy-drop protocol.

The low-level fitting routines remain generic.  This module holds the data
contract that must be respected before those routines are called: each
``Delta E_R`` value and ``Delta E_S`` value refers to the same event
transition, in the same position in the arrays.  Classification is performed
from the event-level ``kappa`` detector first; finite-positive filtering of
``Delta E_S`` happens only after the event labels are known.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


# MTS2D's structured triangular mesh has two elements per unit reference
# cell, so N/V_0 = (2 L^2)/(L^2) = 2 for the macrodata convention.
DEFAULT_KAPPA_RHO = 2.0
POST_YIELD_OFFSET = 0.01
PRE_YIELD_OFFSET = 1e-4
SIZE_SCALING_MIN_SIZE = 100
POST_YIELD_RULE = "avg_P12 global maximum per run; load_ip1 > peak_load + 0.01; pre: load_ip1 < peak_load - 0.0001"
SIGMA_RESCUE_POST_YIELD_WINDOW = "avg_P12 peak_load + 0.01 < load_ip1 < 1.0"


def yield_peak_load(df) -> float:
    """Return the load at the first global maximum of native first-Piola P12."""
    missing = {"load", "avg_P12"} - set(df)
    if missing:
        raise KeyError(f"Missing yield-peak columns: {sorted(missing)}")
    load = np.asarray(df["load"], dtype=float)
    stress = np.asarray(df["avg_P12"], dtype=float)
    if load.ndim != 1 or load.size < 2 or stress.shape != load.shape:
        raise ValueError("Yield-peak load and avg_P12 must be aligned 1D arrays.")
    if not np.all(np.isfinite(load)) or np.any(np.diff(load) <= 0):
        raise ValueError("Yield-peak loads must be finite and strictly increasing.")
    if not np.all(np.isfinite(stress)):
        raise ValueError("Yield-peak avg_P12 contains non-finite values.")
    return float(load[np.argmax(stress)])


def yield_regimes(df) -> dict[str, tuple[float, float]]:
    """Historical peak buffers, using P12 rather than Cauchy shear stress."""
    peak = yield_peak_load(df)
    return {"pre": (float(np.min(df["load"])), peak - PRE_YIELD_OFFSET),
            "post": (peak + POST_YIELD_OFFSET, float(np.max(df["load"])))}


def yield_region_mask(df, region, event_load=None):
    """Select pre/post events with the same buffered windows as extraction."""
    if region not in ("pre", "post"):
        raise ValueError(f"Unknown yield region: {region}")
    low, high = yield_regimes(df)[region]
    load = np.asarray(df["load"] if event_load is None else event_load, dtype=float)
    return np.isfinite(load) & (load > low) & (load < high)


def require_gamma0_snapshot(snapshot_root: Path) -> None:
    """Reject snapshots built with an old energy or yield convention."""
    manifest = Path(snapshot_root) / "tables" / "build_manifest.json"
    metadata = json.loads(manifest.read_text())
    if metadata.get("delta_E_S_tangent") != "gamma0":
        raise ValueError(f"Delta E_S table was not built with static gamma=0 mu: {manifest}")
    if metadata.get("post_yield_rule") != POST_YIELD_RULE:
        raise ValueError(f"Snapshot has an obsolete post-yield rule: {manifest}")


def load_sigma_rescue_post_yield(
    snapshot_root: Path, columns: list[str], *, audited: bool = False
) -> pd.DataFrame:
    """Select transitions after the audited per-run first-Piola stress peak."""
    require_gamma0_snapshot(snapshot_root)
    root = Path(snapshot_root) / "tables"
    filename = "drops_all_audited.csv.gz" if audited else "drops_usable.csv.gz"
    frame = pd.read_csv(root / filename, usecols=list(dict.fromkeys(["run_name", "size", "load_ip1", *columns])))
    peaks = pd.read_csv(root / "yield_peaks.csv", usecols=["run_name", "post_yield_start"])
    frame = frame.merge(peaks, on="run_name", how="left", validate="many_to_one")
    if frame["post_yield_start"].isna().any():
        raise ValueError("A source run has no first-Piola yield-peak audit.")
    frame = frame.loc[
        (frame["load_ip1"] > frame["post_yield_start"])
        & (frame["load_ip1"] < 1.0)
        & (frame["size"] >= SIZE_SCALING_MIN_SIZE)
    ].copy()
    if frame.empty:
        raise ValueError("No rows remain after each run's avg_P12 peak +0.01.")
    return frame


@dataclass(frozen=True)
class EventDrops:
    """Paired event-level drops and optional ``kappa`` values.

    The arrays are deliberately kept at event resolution, including entries
    whose ``es`` value is non-positive or non-finite.  That preserves the
    correspondence needed to transfer the ``er`` classification to ``es``.
    """

    er: np.ndarray
    es: np.ndarray
    kappa: np.ndarray | None = None

    def __post_init__(self) -> None:
        er = np.asarray(self.er, dtype=float)
        es = np.asarray(self.es, dtype=float)
        if er.shape != es.shape:
            raise ValueError(
                "EventDrops requires Delta E_R and Delta E_S arrays with "
                "the same shape."
            )
        object.__setattr__(self, "er", er)
        object.__setattr__(self, "es", es)
        if self.kappa is not None:
            kappa = np.asarray(self.kappa, dtype=float)
            if kappa.shape != er.shape:
                raise ValueError(
                    "EventDrops kappa must have the same shape as Delta E_R."
                )
            object.__setattr__(self, "kappa", kappa)


@dataclass(frozen=True)
class EventSplit:
    """Legacy event labels produced by a ``Delta E_R`` threshold."""

    er_det: float
    is_rev: np.ndarray
    is_irrev: np.ndarray

    def __post_init__(self) -> None:
        er_det = float(self.er_det)
        if not np.isfinite(er_det) or er_det <= 0:
            raise ValueError("er_det must be a finite positive threshold.")
        is_rev = np.asarray(self.is_rev, dtype=bool)
        is_irrev = np.asarray(self.is_irrev, dtype=bool)
        if is_rev.shape != is_irrev.shape:
            raise ValueError("EventSplit masks must have the same shape.")
        if np.any(is_rev & is_irrev):
            raise ValueError("An event cannot be both reversible and irreversible.")
        object.__setattr__(self, "er_det", er_det)
        object.__setattr__(self, "is_rev", is_rev)
        object.__setattr__(self, "is_irrev", is_irrev)


def split_by_er(drops: EventDrops, er_det: float) -> EventSplit:
    """Classify paired events using an explicit legacy ``Delta E_R`` cutoff.

    The default protocol uses :func:`split_by_kappa`; this helper remains for
    historical comparisons and tests.

    ``Delta E_S`` is intentionally not inspected here.  In particular, an
    event with an invalid ``es`` remains part of the event-level irreversible
    mask when its ``er`` value crosses the threshold; it is removed only when
    constructing the positive ``es_irrev`` fit population.
    """

    if not isinstance(drops, EventDrops):
        raise TypeError("split_by_er expects an EventDrops instance.")
    er_det = float(er_det)
    labelable = np.isfinite(drops.er) & (drops.er > 0)
    is_rev = labelable & (drops.er < er_det)
    is_irrev = labelable & (drops.er >= er_det)
    return EventSplit(er_det=er_det, is_rev=is_rev, is_irrev=is_irrev)


@dataclass(frozen=True)
class KappaEventSplit:
    """Event labels produced by the default ``kappa_det`` detector."""

    kappa_det: float
    is_rev: np.ndarray
    is_irrev: np.ndarray

    def __post_init__(self) -> None:
        kappa_det = float(self.kappa_det)
        if not np.isfinite(kappa_det) or kappa_det <= 0:
            raise ValueError("kappa_det must be a finite positive threshold.")
        is_rev = np.asarray(self.is_rev, dtype=bool)
        is_irrev = np.asarray(self.is_irrev, dtype=bool)
        if is_rev.shape != is_irrev.shape:
            raise ValueError("KappaEventSplit masks must have the same shape.")
        if np.any(is_rev & is_irrev):
            raise ValueError("An event cannot be both reversible and irreversible.")
        object.__setattr__(self, "kappa_det", kappa_det)
        object.__setattr__(self, "is_rev", is_rev)
        object.__setattr__(self, "is_irrev", is_irrev)


def split_by_kappa(drops: EventDrops, kappa_det: float) -> KappaEventSplit:
    """Classify paired events with ``kappa < kappa_det`` as reversible."""

    if not isinstance(drops, EventDrops):
        raise TypeError("split_by_kappa expects an EventDrops instance.")
    if drops.kappa is None:
        raise ValueError("split_by_kappa requires kappa values in EventDrops.")
    kappa_det = float(kappa_det)
    if not np.isfinite(kappa_det) or kappa_det <= 0:
        raise ValueError("kappa_det must be a finite positive threshold.")
    labelable = (
        np.isfinite(drops.er)
        & (drops.er > 0)
        & np.isfinite(drops.kappa)
        & (drops.kappa > 0)
    )
    is_rev = labelable & (drops.kappa < kappa_det)
    is_irrev = labelable & (drops.kappa >= kappa_det)
    return KappaEventSplit(kappa_det=kappa_det, is_rev=is_rev, is_irrev=is_irrev)


def kappa_from_relaxation_energy(
    delta_e_r,
    delta_gamma,
    reference_volume,
    *,
    rho=DEFAULT_KAPPA_RHO,
):
    """Return ``Delta E_R / (rho V_0 Delta gamma**2)`` without filtering."""

    delta_e_r = np.asarray(delta_e_r, dtype=float)
    delta_gamma = np.asarray(delta_gamma, dtype=float)
    if delta_e_r.shape != delta_gamma.shape:
        raise ValueError("Delta E_R and Delta gamma must have the same shape.")
    reference_volume = np.asarray(reference_volume, dtype=float)
    if reference_volume.ndim > 0 and reference_volume.shape != delta_e_r.shape:
        raise ValueError(
            "reference_volume must be a scalar or have the same shape as Delta E_R."
        )
    rho = float(rho)
    if np.any(~np.isfinite(reference_volume)) or np.any(reference_volume <= 0):
        raise ValueError("reference_volume must contain only finite positive values.")
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive.")
    if np.any(~np.isfinite(delta_gamma)) or np.any(delta_gamma <= 0):
        raise ValueError("Delta gamma must contain only finite positive values.")
    return delta_e_r / (rho * reference_volume * delta_gamma**2)


def kappa_detection_threshold(mu=None, *, rho=DEFAULT_KAPPA_RHO):
    """Return ``kappa_det = mu / (2 rho)`` for the material shear modulus.

    The cited articles denote this affine/Born shear modulus by ``G_B``;
    project code and plot labels use ``mu`` instead.
    """

    if mu is None:
        from MTMath.energyFunction import ContiEnergy

        mu = ContiEnergy.moduli_at_F(np.eye(2)).mu
    mu = float(np.asarray(mu, dtype=float).reshape(-1)[0])
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("mu must be finite and positive.")
    rho = float(rho)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive.")
    return mu / (2.0 * rho)


def positive_es(drops: EventDrops, mask: np.ndarray) -> np.ndarray:
    """Return finite-positive ``Delta E_S`` values for an event mask."""

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != drops.es.shape:
        raise ValueError("The event mask must match the paired event arrays.")
    values = drops.es[mask]
    return values[np.isfinite(values) & (values > 0)]
