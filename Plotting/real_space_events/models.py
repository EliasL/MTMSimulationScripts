"""Shared data models for the real-space event workflow."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path


class EventClass(str, Enum):
    REVERSIBLE_PLASTIC = "reversible_plastic"
    REVERSIBLE_ELASTIC = "reversible_elastic"
    IRREVERSIBLE_PLASTIC = "irreversible_plastic"
    IRREVERSIBLE_ELASTIC = "irreversible_elastic"
    REVERSIBILITY_UNMEASURED = "reversibility_unmeasured"


class RepresentativeKind(str, Enum):
    TYPICAL = "typical"
    LARGE_INTERSTRAIN_DROP = "large_interstrain_drop"
    HIGH_PARTICIPATION = "high_participation"


@dataclass(frozen=True)
class AnalysisScope:
    """Default to the chosen numerical setting, without a yield restriction."""

    batches: tuple[int, ...] = (-2, -1)
    epsilon_x: float = 1e-6
    delta_gamma: float = 1e-5
    yield_regime: str = "all"  # all, pre, post
    reconnection_modes: tuple[str, ...] = ("none", "edgeFlip")


@dataclass(frozen=True)
class EventStatePaths:
    """The five states saved or replayed for one event."""

    state0_min_gamma: Path
    state1_affine_gamma_plus: Path
    state2_relaxed_gamma_plus: Path
    state3_affine_gamma_minus: Path
    state4_relaxed_gamma: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "state0": self.state0_min_gamma,
            "state1": self.state1_affine_gamma_plus,
            "state2": self.state2_relaxed_gamma_plus,
            "state3": self.state3_affine_gamma_minus,
            "state4": self.state4_relaxed_gamma,
        }


@dataclass(frozen=True)
class RemoteSource:
    host: str
    data_root: Path


@dataclass(frozen=True)
class DownloadRequest:
    event_id: str
    source: RemoteSource
    remote_event_directory: Path
    local_event_directory: Path


@dataclass(frozen=True)
class ReplayRequest:
    event_id: str
    source: RemoteSource
    job_directory: Path
    dump_path: Path
    target_load: float
    output_directory: Path
    force_backward_test: bool = True


@dataclass(frozen=True)
class EventSheetLayout:
    """Centralized geometry controls for the six-panel event sheet."""

    column_width_ratios: tuple[float, float, float] = (1.0, 1.0, 0.82)
    left: float = 0.055
    right: float = 0.985
    bottom: float = 0.09
    top: float = 0.90
    column_spacing: float = 0.08
    row_spacing: float = 0.34
    # Horizontal shifts are fractions of the full figure width.
    middle_column_shift: float = 0.020
    final_column_shift: float = 0.0
    scatter_ylabel_pad: float = 2.0
    scatter_ytick_pad: float = 1.0

    def validate(self) -> None:
        if len(self.column_width_ratios) != 3 or any(
            width <= 0 for width in self.column_width_ratios
        ):
            raise ValueError("column_width_ratios must contain three positive values.")
        if not 0 <= self.left < self.right <= 1:
            raise ValueError("Require 0 <= left < right <= 1.")
        if not 0 <= self.bottom < self.top <= 1:
            raise ValueError("Require 0 <= bottom < top <= 1.")
        if self.column_spacing < 0 or self.row_spacing < 0:
            raise ValueError("Row and column spacing must be non-negative.")
        values = (
            self.middle_column_shift,
            self.final_column_shift,
            self.scatter_ylabel_pad,
            self.scatter_ytick_pad,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Layout shifts and label padding must be finite.")


@dataclass(frozen=True)
class RenderOptions:
    output_root: Path
    activity_fraction: float = 0.8
    zoom_padding_element_lengths: float = 4.0
    arrow_target_element_fraction: float = 1.0 / 3.0
    arrow_length_multiplier: float = 2.0
    # ``None`` keeps every unique node in the zoom.  An integer can still be
    # supplied for deliberately sparse overview figures.
    maximum_arrows: int | None = None
    common_arrow_scale: float | None = None
    common_energy_limit: float | None = None
    common_grid_resolution: int = 400
    figure_size: tuple[float, float] = (10.4, 5.0)
    layout: EventSheetLayout = EventSheetLayout()
    rasterized_dpi: int = 200
    output_format: str = "auto"

    def validate(self) -> None:
        if self.output_format not in {"auto", "pdf", "png"}:
            raise ValueError("output_format must be 'auto', 'pdf', or 'png'.")
        if not 0 < self.activity_fraction <= 1:
            raise ValueError("activity_fraction must lie in (0, 1].")
        if self.maximum_arrows is not None and self.maximum_arrows <= 0:
            raise ValueError("maximum_arrows must be positive.")
        if self.arrow_length_multiplier <= 0:
            raise ValueError("arrow_length_multiplier must be positive.")
        if len(self.figure_size) != 2 or any(size <= 0 for size in self.figure_size):
            raise ValueError("figure_size must contain two positive values.")
        if self.rasterized_dpi <= 0:
            raise ValueError("rasterized_dpi must be positive.")
        self.layout.validate()
