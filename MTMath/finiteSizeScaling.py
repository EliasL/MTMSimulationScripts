"""Small, reusable finite-size-scaling utilities.

The functions in this module operate on already selected positive event
values.  They do not know about a particular simulation format or about
event classification.  For a density obeying

    p(x, L) = x**(-tau) f(x / L**d),

``collapse_variance`` and ``optimize_collapse`` estimate ``tau`` and ``d``
from the overlap of logarithmically binned PDFs.  ``fit_moment_scaling`` is
an independent, deterministic moment-scaling diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LogPDF:
    """A positive log-binned PDF for one system size."""

    size: float
    x: np.ndarray
    density: np.ndarray
    count: int

    def __post_init__(self) -> None:
        size = float(self.size)
        x = np.asarray(self.x, dtype=float)
        density = np.asarray(self.density, dtype=float)
        if np.isfinite(size) and size <= 0:
            raise ValueError("A system size must be positive when supplied.")
        if x.ndim != 1 or density.ndim != 1 or x.shape != density.shape:
            raise ValueError("LogPDF x and density must be matching 1-D arrays.")
        if x.size < 2 or np.any(~np.isfinite(x)) or np.any(x <= 0):
            raise ValueError("LogPDF x must contain at least two positive values.")
        if np.any(~np.isfinite(density)) or np.any(density <= 0):
            raise ValueError("LogPDF density must contain finite positive values.")
        if np.any(np.diff(x) <= 0):
            raise ValueError("LogPDF x values must be strictly increasing.")
        count = int(self.count)
        if count < 3:
            raise ValueError("A LogPDF requires at least three observations.")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "count", count)


def _positive(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 3:
        raise ValueError("Need at least three finite positive observations.")
    return values


def log_pdf(values, *, bins_per_decade: int = 12) -> LogPDF:
    """Return a density-normalized logarithmic histogram."""

    if int(bins_per_decade) != bins_per_decade or bins_per_decade < 1:
        raise ValueError("bins_per_decade must be a positive integer.")
    values = _positive(values)
    low = float(np.log10(values.min()))
    high = float(np.log10(values.max()))
    if not high > low:
        raise ValueError("The observations must span a non-zero logarithmic range.")
    bins = max(2, int(np.ceil((high - low) * bins_per_decade)))
    edges = np.logspace(low, high, bins + 1)
    density, edges = np.histogram(values, bins=edges, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    keep = np.isfinite(density) & (density > 0)
    return LogPDF(float("nan"), centers[keep], density[keep], int(values.size))


def build_log_pdf_curves(
    samples_by_size: Mapping[float, Sequence[float]],
    *,
    bins_per_decade: int = 12,
) -> dict[float, LogPDF]:
    """Build one log-PDF per size, retaining the sample count."""

    if len(samples_by_size) < 2:
        raise ValueError("Finite-size scaling requires at least two system sizes.")
    curves: dict[float, LogPDF] = {}
    for size, values in sorted(samples_by_size.items(), key=lambda item: float(item[0])):
        curve = log_pdf(values, bins_per_decade=bins_per_decade)
        curves[float(size)] = LogPDF(float(size), curve.x, curve.density, curve.count)
    return curves


def filter_by_xmin(
    samples_by_size: Mapping[float, Sequence[float]],
    xmins: Mapping[float, float],
    *,
    min_count: int = 3,
) -> dict[float, np.ndarray]:
    """Keep the fitted tail ``x >= xmin`` independently for each size."""

    if int(min_count) != min_count or min_count < 3:
        raise ValueError("min_count must be an integer of at least three.")
    filtered: dict[float, np.ndarray] = {}
    for size, values in sorted(samples_by_size.items(), key=lambda item: float(item[0])):
        if size not in xmins and float(size) not in xmins:
            raise KeyError(f"Missing xmin for system size {size}.")
        xmin = float(xmins[size] if size in xmins else xmins[float(size)])
        if not np.isfinite(xmin) or xmin <= 0:
            raise ValueError(f"xmin for L={size} must be finite and positive.")
        values = _positive(values)
        tail = values[values >= xmin]
        if tail.size < min_count:
            raise ValueError(
                f"L={size} has only {tail.size} values at or above xmin={xmin:.3e}."
            )
        filtered[float(size)] = tail
    return filtered


def _curve_arrays(curve) -> tuple[float, np.ndarray, np.ndarray]:
    if isinstance(curve, LogPDF):
        return curve.size, curve.x, curve.density
    if len(curve) != 2:
        raise TypeError("A curve must be LogPDF or (x, density).")
    x, density = curve
    x = np.asarray(x, dtype=float)
    density = np.asarray(density, dtype=float)
    if x.shape != density.shape or x.ndim != 1:
        raise ValueError("Curve x and density must be matching 1-D arrays.")
    return np.nan, x, density


def collapse_variance(curves: Mapping[float, object], exponent: float, dimension: float, *, n_points: int = 80) -> float:
    """Return the mean log-density variance over the common scaled domain."""

    exponent = float(exponent)
    dimension = float(dimension)
    if not np.isfinite(exponent) or not np.isfinite(dimension):
        raise ValueError("Collapse parameters must be finite.")
    if int(n_points) != n_points or n_points < 2:
        raise ValueError("n_points must be an integer greater than one.")
    transformed = []
    for size_key, curve in curves.items():
        size, x, density = _curve_arrays(curve)
        if not np.isfinite(size):
            size = float(size_key)
        if not np.isfinite(size) or size <= 0:
            raise ValueError("Every curve needs a finite positive system size.")
        if np.any(~np.isfinite(x)) or np.any(x <= 0) or np.any(np.diff(x) <= 0):
            raise ValueError("Curve x values must be finite, positive and sorted.")
        if np.any(~np.isfinite(density)) or np.any(density <= 0):
            raise ValueError("Curve densities must be finite and positive.")
        transformed.append(
            (
                np.log(x) - dimension * np.log(size),
                np.log(density) + dimension * exponent * np.log(size),
            )
        )
    if len(transformed) < 2:
        raise ValueError("A collapse requires at least two curves.")
    low = max(values[0].min() for values in transformed)
    high = min(values[0].max() for values in transformed)
    if not high > low:
        return float("inf")
    common = np.linspace(low, high, int(n_points))
    interpolated = np.vstack(
        [np.interp(common, log_x, log_y) for log_x, log_y in transformed]
    )
    return float(np.mean(np.var(interpolated, axis=0, ddof=1)))


def _grid(low_high, points: int, name: str) -> np.ndarray:
    low, high = map(float, low_high)
    if not np.isfinite(low) or not np.isfinite(high) or not high > low:
        raise ValueError(f"{name} must be a finite increasing range.")
    if int(points) != points or points < 2:
        raise ValueError(f"{name} points must be an integer greater than one.")
    return np.linspace(low, high, int(points))


def evaluate_collapse_grid(curves, exponents, dimensions, *, n_points: int = 80) -> np.ndarray:
    """Evaluate the collapse objective on a rectangular parameter grid."""

    exponents = np.asarray(exponents, dtype=float)
    dimensions = np.asarray(dimensions, dtype=float)
    quality = np.empty((exponents.size, dimensions.size), dtype=float)
    for i, exponent in enumerate(exponents):
        for j, dimension in enumerate(dimensions):
            quality[i, j] = collapse_variance(
                curves, exponent, dimension, n_points=n_points
            )
    return quality


def optimize_collapse(
    curves,
    *,
    exponent_range=(0.5, 2.5),
    dimension_range=(0.25, 2.25),
    coarse_points: int = 41,
    fine_points: int = 61,
    n_points: int = 80,
) -> dict[str, object]:
    """Find the deterministic minimum of the two-parameter collapse objective."""

    coarse_exponents = _grid(exponent_range, coarse_points, "exponent_range")
    coarse_dimensions = _grid(dimension_range, coarse_points, "dimension_range")
    coarse_quality = evaluate_collapse_grid(
        curves, coarse_exponents, coarse_dimensions, n_points=n_points
    )
    coarse_i, coarse_j = np.unravel_index(np.nanargmin(coarse_quality), coarse_quality.shape)
    exponent_step = coarse_exponents[1] - coarse_exponents[0]
    dimension_step = coarse_dimensions[1] - coarse_dimensions[0]
    fine_exponents = _grid(
        (max(coarse_exponents[0], coarse_exponents[coarse_i] - 2 * exponent_step),
         min(coarse_exponents[-1], coarse_exponents[coarse_i] + 2 * exponent_step)),
        fine_points,
        "fine exponent range",
    )
    fine_dimensions = _grid(
        (max(coarse_dimensions[0], coarse_dimensions[coarse_j] - 2 * dimension_step),
         min(coarse_dimensions[-1], coarse_dimensions[coarse_j] + 2 * dimension_step)),
        fine_points,
        "fine dimension range",
    )
    fine_quality = evaluate_collapse_grid(
        curves, fine_exponents, fine_dimensions, n_points=n_points
    )
    fine_i, fine_j = np.unravel_index(np.nanargmin(fine_quality), fine_quality.shape)
    return {
        "exponent": float(fine_exponents[fine_i]),
        "dimension": float(fine_dimensions[fine_j]),
        "quality": float(fine_quality[fine_i, fine_j]),
        "boundary": bool(
            fine_i in {0, fine_points - 1} or fine_j in {0, fine_points - 1}
        ),
        "coarse_exponents": coarse_exponents,
        "coarse_dimensions": coarse_dimensions,
        "coarse_quality": coarse_quality,
        "fine_exponents": fine_exponents,
        "fine_dimensions": fine_dimensions,
        "fine_quality": fine_quality,
    }


def collapsed_curves(curves, exponent: float, dimension: float) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Return ``(x/L**d, p(x,L)*L**(d*tau))`` for each curve."""

    result = {}
    for size_key, curve in curves.items():
        size, x, density = _curve_arrays(curve)
        if not np.isfinite(size):
            size = float(size_key)
        result[float(size)] = (
            x / size**float(dimension),
            density * size ** (float(dimension) * float(exponent)),
        )
    return result


def _linear_fit(x, y) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or x.shape != y.shape:
        raise ValueError("A scaling fit requires at least three matching points.")
    coefficients, covariance = np.polyfit(x, y, 1, cov=True)
    predicted = np.polyval(coefficients, x)
    residual = y - predicted
    total = np.sum((y - y.mean()) ** 2)
    r_squared = float(1.0 - np.sum(residual**2) / total) if total > 0 else 1.0
    return float(coefficients[0]), float(coefficients[1]), float(np.sqrt(covariance[0, 0])), r_squared


def fit_moment_scaling(
    samples_by_size: Mapping[float, Sequence[float]],
    *,
    orders: Sequence[float] = (1, 2, 3),
) -> dict[str, object]:
    """Fit moment exponents and their linear dependence on moment order.

    For a normalized ``p(x,L)=x**(-tau)f(x/L**d)``, the expected relation is
    ``z_q = d*(q + 1 - tau)``.  Thus the slope of ``z_q`` versus ``q`` is a
    direct estimate of ``d`` and its intercept gives ``tau``.
    """

    orders = np.asarray(orders, dtype=float)
    if orders.ndim != 1 or orders.size < 2 or np.any(~np.isfinite(orders)):
        raise ValueError("orders must contain at least two finite values.")
    if np.any(orders < 0):
        raise ValueError("Moment orders must be non-negative.")
    sizes = np.asarray(sorted(float(size) for size in samples_by_size), dtype=float)
    if sizes.size < 3:
        raise ValueError("Moment scaling requires at least three system sizes.")
    moments = np.empty((sizes.size, orders.size), dtype=float)
    for row, size in enumerate(sizes):
        values = _positive(samples_by_size[size])
        moments[row] = [float(np.mean(values**order)) for order in orders]
    moment_exponents = []
    moment_stderr = []
    moment_r_squared = []
    log_sizes = np.log(sizes)
    for column in range(orders.size):
        slope, intercept, stderr, r_squared = _linear_fit(
            log_sizes, np.log(moments[:, column])
        )
        moment_exponents.append(slope)
        moment_stderr.append(stderr)
        moment_r_squared.append(r_squared)
    dimension, intercept, dimension_stderr, order_r_squared = _linear_fit(
        orders, moment_exponents
    )
    if dimension == 0:
        raise ValueError("Moment-order fit returned zero dimension; cannot infer tau.")
    return {
        "sizes": sizes,
        "orders": orders,
        "moments": moments,
        "moment_exponents": np.asarray(moment_exponents),
        "moment_exponent_stderr": np.asarray(moment_stderr),
        "moment_r_squared": np.asarray(moment_r_squared),
        "dimension": dimension,
        "dimension_stderr": dimension_stderr,
        "moment_order_intercept": intercept,
        "exponent": 1.0 - intercept / dimension,
        "moment_order_r_squared": order_r_squared,
    }
