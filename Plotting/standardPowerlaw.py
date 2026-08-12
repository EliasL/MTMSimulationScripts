"""Small, explicit building blocks for the standard energy-drop protocol.

The low-level fitting routines remain generic.  This module holds the data
contract that must be respected before those routines are called: each
``Delta E_R`` value and ``Delta E_S`` value refers to the same event
transition, in the same position in the arrays.  Classification is performed
from ``Delta E_R`` first; finite-positive filtering of ``Delta E_S`` happens
only after the event labels are known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventDrops:
    """Paired event-level ``Delta E_R`` and ``Delta E_S`` arrays.

    The arrays are deliberately kept at event resolution, including entries
    whose ``es`` value is non-positive or non-finite.  That preserves the
    correspondence needed to transfer the ``er`` classification to ``es``.
    """

    er: np.ndarray
    es: np.ndarray

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


@dataclass(frozen=True)
class EventSplit:
    """Event labels produced by the ``Delta E_R`` detection threshold."""

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
    """Classify paired events using only ``Delta E_R``.

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


def positive_es(drops: EventDrops, mask: np.ndarray) -> np.ndarray:
    """Return finite-positive ``Delta E_S`` values for an event mask."""

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != drops.es.shape:
        raise ValueError("The event mask must match the paired event arrays.")
    values = drops.es[mask]
    return values[np.isfinite(values) & (values > 0)]

