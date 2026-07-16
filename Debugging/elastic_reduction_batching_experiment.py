"""Minimal scalar experiments for batching elastic-reduction shears.

These implementations are intentionally independent of the production
vectorized routine.  They are for checking the mathematics and operation
counts before considering a C++ implementation.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Iterable

import numpy as np

Matrix = tuple[int, int, int, int]
State = tuple[float, float, float]
Result = tuple[State, Matrix, int, int]

_EPS = math.ulp(1.0)


def _inside(a: float, b: float, c: float) -> bool:
    return abs(c) <= 0.5 * min(a, b)


def _sign_toward_zero(c: float) -> int:
    return -1 if c > 0.0 else 1


def _apply(a: float, b: float, c: float, use_u: bool, m: int) -> State:
    """Apply U_m or V_m, using fused operations where they help."""
    if use_u:
        c_new = math.fma(m, a, c)
        b_new = math.fma(m * m, a, math.fma(2 * m, c, b))
        return a, b_new, c_new
    c_new = math.fma(m, b, c)
    a_new = math.fma(m * m, b, math.fma(2 * m, c, a))
    return a_new, b, c_new


def _accumulate(M: Matrix, use_u: bool, m: int) -> Matrix:
    m00, m01, m10, m11 = M
    if use_u:
        return m00, m00 * m + m01, m10, m10 * m + m11
    return m00 + m01 * m, m01, m10 + m11 * m, m11


def _accurate_det(a: float, b: float, c: float) -> float:
    """Two-product correction for ab-c^2."""
    ab = a * b
    cc = c * c
    return (ab - cc) + (math.fma(a, b, -ab) - math.fma(c, c, -cc))


def _near_integer(x: float, ulps: float = 256.0) -> bool:
    nearest = round(x)
    return abs(x - nearest) <= ulps * _EPS * max(1.0, abs(x))


def _ceil_down(x: float) -> int:
    """Ceiling with a one-ulp downward bias at representational boundaries."""
    return math.ceil(math.nextafter(x, -math.inf))


def reduce_single_step(C: State, max_batches: int = 10000) -> Result:
    """Reference: exactly one unit shear per loop."""
    a, b, c = C
    M: Matrix = (1, 0, 0, 1)
    batches = shears = 0
    while not _inside(a, b, c):
        if batches >= max_batches:
            raise RuntimeError("single-step reduction did not converge")
        use_u = a <= b
        m = _sign_toward_zero(c)
        a, b, c = _apply(a, b, c, use_u, m)
        M = _accumulate(M, use_u, m)
        batches += 1
        shears += 1
    return (a, b, c), M, batches, shears


def _maximal_run(a: float, b: float, c: float, use_u: bool) -> int:
    """Closed-form length of the current identical-generator run."""
    pivot = a if use_u else b
    x = abs(c) / pivot
    eta = _accurate_det(a, b, c) / (pivot * pivot)
    threshold = max(0.5, math.sqrt(max(0.0, 1.0 - eta)))
    distance = x - threshold

    # Exact equality is where stopping and the U/V tie convention matter most.
    # A unit-step fallback keeps the floating implementation deterministic.
    if _near_integer(distance):
        return 1
    return max(1, math.ceil(distance))


def reduce_maximal_sqrt(C: State, max_batches: int = 10000) -> Result:
    """Batch the complete run predicted by the square-root expression."""
    a, b, c = C
    M: Matrix = (1, 0, 0, 1)
    batches = shears = 0
    while not _inside(a, b, c):
        if batches >= max_batches:
            raise RuntimeError("maximal-run reduction did not converge")
        use_u = a <= b

        # n can only exceed one if |c|/pivot > 3/2.  This avoids the
        # determinant and square root for the common one-step case.
        pivot = a if use_u else b
        x = abs(c) / pivot
        n = _maximal_run(a, b, c, use_u) if x > 1.5 else 1
        m = _sign_toward_zero(c) * n

        a, b, c = _apply(a, b, c, use_u, m)
        M = _accumulate(M, use_u, m)
        batches += 1
        shears += n
    return (a, b, c), M, batches, shears


def _conservative_run(a: float, b: float, c: float, use_u: bool) -> int:
    """Safe no-square-root lower bound on the identical-generator run."""
    pivot, other = (a, b) if use_u else (b, a)
    x = abs(c) / pivot

    # All intermediate states stay outside the elastic strip.
    stop_bound = max(1, _ceil_down(x - 0.5))

    # For j repeated steps,
    # other_j-pivot = other-pivot-2*j*|c|+j^2*pivot.
    # Dropping the positive j^2 term gives a conservative lower bound.
    gap = (other - pivot) / (2.0 * abs(c))
    orientation_bound = max(1, _ceil_down(gap))
    return min(stop_bound, orientation_bound)


def reduce_conservative(
    C: State,
    max_batches: int = 10000,
    trigger: float = 3.0,
) -> Result:
    """Batch a cheap, provably safe number of identical unit shears."""
    a, b, c = C
    M: Matrix = (1, 0, 0, 1)
    batches = shears = 0
    while not _inside(a, b, c):
        if batches >= max_batches:
            raise RuntimeError("conservative reduction did not converge")
        use_u = a <= b
        pivot = a if use_u else b
        x = abs(c) / pivot

        # A cheap ratio gate avoids even the bound calculation unless there
        # are likely to be several identical steps to compress.
        n = _conservative_run(a, b, c, use_u) if x > trigger else 1
        m = _sign_toward_zero(c) * n

        a, b, c = _apply(a, b, c, use_u, m)
        M = _accumulate(M, use_u, m)
        batches += 1
        shears += n
    return (a, b, c), M, batches, shears


def poincare_samples(resolution: int = 300) -> list[State]:
    axis = np.linspace(-1.0, 1.0, resolution)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    mask = X * X + Y * Y < 1.0
    x, y = X[mask], Y[mask]
    factor = 2.0 / (1.0 - x * x - y * y)
    return list(zip(factor * (1.0 + x) - 1.0,
                    factor * (1.0 - x) - 1.0,
                    factor * y,
                    strict=True))


def cube_samples(resolution: int = 100) -> list[State]:
    values = np.linspace(-10.0, 10.0, resolution)
    a, b, c = np.meshgrid(values, values, values, indexing="ij")
    mask = (a > 0.0) & (b > 0.0) & (a * b - c * c > 0.0)
    return list(zip(a[mask], b[mask], c[mask], strict=True))


def strong_shear_samples(count: int = 4000) -> list[State]:
    gamma = np.linspace(8.0, 200.0, count)
    return [(1.0, float(g * g + 1.0), float(g)) for g in gamma]


def check(
    samples: Iterable[State],
    candidate: Callable[[State], Result],
) -> tuple[int, int, float, int, int]:
    transform_failures = endpoint_failures = 0
    max_error = 0.0
    reference_batches = candidate_batches = 0
    for C in samples:
        reference = reduce_single_step(C)
        trial = candidate(C)
        reference_batches += reference[2]
        candidate_batches += trial[2]
        if reference[1] != trial[1]:
            transform_failures += 1
        error = max(abs(x - y) for x, y in zip(reference[0], trial[0]))
        scale = max(1.0, *(abs(x) for x in reference[0]))
        max_error = max(max_error, error)
        if error > 1e-10 * scale:
            endpoint_failures += 1
    return (
        transform_failures,
        endpoint_failures,
        max_error,
        reference_batches,
        candidate_batches,
    )


def benchmark(
    samples: list[State],
    reducer: Callable[[State], Result],
    repeats: int = 3,
) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for C in samples:
            reducer(C)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def report_dataset(name: str, samples: list[State], run_benchmark: bool) -> None:
    print(f"\n{name}: {len(samples):,} samples")
    reducers = (
        ("single", reduce_single_step),
        ("maximal sqrt", reduce_maximal_sqrt),
        ("conservative", reduce_conservative),
    )

    for label, reducer in reducers[1:]:
        tf, ef, error, base_batches, trial_batches = check(samples, reducer)
        print(
            f"  {label:12s} transform failures={tf:4d}, "
            f"endpoint failures={ef:4d}, max |dC|={error:.3e}, "
            f"outer loops={trial_batches:,}/{base_batches:,}"
        )

    if run_benchmark:
        times = {label: benchmark(samples, reducer) for label, reducer in reducers}
        base = times["single"]
        for label, _ in reducers:
            print(
                f"  {label:12s} median={times[label]:.4f} s, "
                f"speedup={base / times[label]:.2f}x"
            )


def main() -> None:
    report_dataset("Poincare disk 300x300", poincare_samples(), run_benchmark=True)
    report_dataset("SPD cube from 100^3 candidates", cube_samples(), run_benchmark=True)
    report_dataset("strong simple shears", strong_shear_samples(), run_benchmark=True)


if __name__ == "__main__":
    main()
