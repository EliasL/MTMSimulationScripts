#!/usr/bin/env python3
"""Compare unit-step and batched-integer elastic reduction on dense grids.

The current implementation in :mod:`MTMath.reduction` is the reference.  The
candidate implementation uses the nearest nonzero integer shear, which is the
batched equivalent of applying unit shears repeatedly in the same direction.

Run the Poincare-disk check first and stop on an algorithmic mismatch.  Only
then run the larger component-cube check.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MTMath.reduction import elastic_reduction_components, in_elastic_domain


@dataclass
class Comparison:
    name: str
    sample_count: int
    exact_component_matches: int
    close_component_matches: int
    exact_transform_matches: int
    max_absolute_error: float
    max_relative_error: float
    first_failed_sample: tuple[float, float, float] | None
    first_unit_result: tuple[float, float, float] | None
    first_multi_result: tuple[float, float, float] | None
    first_unit_transform: np.ndarray | None
    first_multi_transform: np.ndarray | None
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return (
            self.close_component_matches == self.sample_count
            and self.exact_transform_matches == self.sample_count
        )


def multistep_elastic_reduction_components(
    C11: np.ndarray,
    C22: np.ndarray,
    C12: np.ndarray,
    *,
    loops: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce using the nearest integer shear instead of a unit shear."""
    a, b, c = np.broadcast_arrays(
        np.asarray(C11, dtype=float),
        np.asarray(C22, dtype=float),
        np.asarray(C12, dtype=float),
    )
    a = a.copy()
    b = b.copy()
    c = c.copy()

    M = np.zeros(a.shape + (2, 2), dtype=float)
    M[..., 0, 0] = 1.0
    M[..., 1, 1] = 1.0

    converged = False
    for _ in range(loops):
        inside = in_elastic_domain(a, b, c)
        active = ~inside
        if not np.any(active):
            converged = True
            break

        # Match the reference implementation's horizontal tie-break.
        use_U = active & (a <= b)
        use_V = active & ~use_U

        if np.any(use_U):
            old_a = a[use_U]
            old_b = b[use_U]
            old_c = c[use_U]
            ratio = -old_c / old_a
            m = np.sign(ratio) * np.floor(np.abs(ratio) + 0.5)

            c[use_U] = old_c + m * old_a
            b[use_U] = old_b + 2.0 * m * old_c + m * m * old_a

            W = np.zeros((m.size, 2, 2), dtype=float)
            W[:, 0, 0] = 1.0
            W[:, 1, 1] = 1.0
            W[:, 0, 1] = m
            M[use_U] = M[use_U] @ W

        if np.any(use_V):
            old_a = a[use_V]
            old_b = b[use_V]
            old_c = c[use_V]
            ratio = -old_c / old_b
            m = np.sign(ratio) * np.floor(np.abs(ratio) + 0.5)

            c[use_V] = old_c + m * old_b
            a[use_V] = old_a + 2.0 * m * old_c + m * m * old_b

            W = np.zeros((m.size, 2, 2), dtype=float)
            W[:, 0, 0] = 1.0
            W[:, 1, 1] = 1.0
            W[:, 1, 0] = m
            M[use_V] = M[use_V] @ W

    if not converged:
        remaining = np.count_nonzero(~in_elastic_domain(a, b, c))
        raise RuntimeError(
            f"Multi-step reduction did not converge for {remaining} samples "
            f"within {loops} iterations"
        )

    return a, b, c, M


def poincare_disk_samples(resolution: int = 300) -> tuple[np.ndarray, ...]:
    """Return the det(C)=1 metrics on an evenly spaced disk-coordinate grid."""
    axis = np.linspace(-1.0, 1.0, resolution)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    inside = X * X + Y * Y < 1.0
    x = X[inside]
    y = Y[inside]

    r = 1.0 - x * x - y * y
    t = 2.0 / r
    C11 = t * (1.0 + x) - 1.0
    C22 = t * (1.0 - x) - 1.0
    C12 = t * y
    return C11, C22, C12


def component_cube_samples(
    resolution: int = 100,
    lower: float = -10.0,
    upper: float = 10.0,
) -> tuple[np.ndarray, ...]:
    """Return the SPD points in an evenly sampled symmetric-component cube."""
    values = np.linspace(lower, upper, resolution)
    C11, C22, C12 = np.meshgrid(values, values, values, indexing="ij")
    valid = (C11 > 0.0) & (C22 > 0.0) & (C11 * C22 - C12 * C12 > 0.0)
    return C11[valid], C22[valid], C12[valid]


def compare(
    name: str,
    samples: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> Comparison:
    C11, C22, C12 = samples
    start = time.perf_counter()

    unit_a, unit_b, unit_c, unit_M = elastic_reduction_components(
        C11, C22, C12, loops=1000, compute_M=True
    )
    multi_a, multi_b, multi_c, multi_M = multistep_elastic_reduction_components(
        C11, C22, C12, loops=1000
    )

    unit = np.column_stack((unit_a, unit_b, unit_c))
    multi = np.column_stack((multi_a, multi_b, multi_c))
    exact_components = np.all(unit == multi, axis=1)
    close_components = np.all(
        np.isclose(unit, multi, rtol=rtol, atol=atol), axis=1
    )
    exact_transforms = np.all(unit_M == multi_M, axis=(1, 2))

    absolute_error = np.abs(unit - multi)
    scale = np.maximum(np.maximum(np.abs(unit), np.abs(multi)), atol)
    relative_error = absolute_error / scale

    failed = ~(close_components & exact_transforms)
    if np.any(failed):
        index = int(np.flatnonzero(failed)[0])
        first_sample = (float(C11[index]), float(C22[index]), float(C12[index]))
        first_unit = tuple(float(value) for value in unit[index])
        first_multi = tuple(float(value) for value in multi[index])
        first_unit_M = unit_M[index].copy()
        first_multi_M = multi_M[index].copy()
    else:
        first_sample = None
        first_unit = None
        first_multi = None
        first_unit_M = None
        first_multi_M = None

    return Comparison(
        name=name,
        sample_count=C11.size,
        exact_component_matches=int(np.count_nonzero(exact_components)),
        close_component_matches=int(np.count_nonzero(close_components)),
        exact_transform_matches=int(np.count_nonzero(exact_transforms)),
        max_absolute_error=float(np.max(absolute_error, initial=0.0)),
        max_relative_error=float(np.max(relative_error, initial=0.0)),
        first_failed_sample=first_sample,
        first_unit_result=first_unit,
        first_multi_result=first_multi,
        first_unit_transform=first_unit_M,
        first_multi_transform=first_multi_M,
        elapsed_seconds=time.perf_counter() - start,
    )


def print_comparison(result: Comparison) -> None:
    print(f"{result.name}: {'PASS' if result.passed else 'FAIL'}")
    print(f"  valid samples:              {result.sample_count:,}")
    print(
        "  bitwise component matches: "
        f"{result.exact_component_matches:,}/{result.sample_count:,}"
    )
    print(
        "  close component matches:   "
        f"{result.close_component_matches:,}/{result.sample_count:,}"
    )
    print(
        "  exact transform matches:   "
        f"{result.exact_transform_matches:,}/{result.sample_count:,}"
    )
    print(f"  maximum absolute error:     {result.max_absolute_error:.17g}")
    print(f"  maximum relative error:     {result.max_relative_error:.17g}")
    print(f"  elapsed:                    {result.elapsed_seconds:.3f} s")
    if result.first_failed_sample is not None:
        print(f"  first failed C components:  {result.first_failed_sample}")
        print(f"  unit-step result:           {result.first_unit_result}")
        print(f"  multi-step result:          {result.first_multi_result}")
        print(f"  unit-step transform:\n{result.first_unit_transform}")
        print(f"  multi-step transform:\n{result.first_multi_transform}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disk-resolution", type=int, default=300)
    parser.add_argument("--cube-resolution", type=int, default=100)
    parser.add_argument("--lower", type=float, default=-10.0)
    parser.add_argument("--upper", type=float, default=10.0)
    args = parser.parse_args()

    disk = compare(
        f"Poincare disk ({args.disk_resolution}x{args.disk_resolution})",
        poincare_disk_samples(args.disk_resolution),
    )
    print_comparison(disk)
    if not disk.passed:
        print("Stopping before the component cube because the disk test failed.")
        return 1

    cube = compare(
        f"SPD component cube ({args.cube_resolution}^3 candidates)",
        component_cube_samples(
            args.cube_resolution,
            lower=args.lower,
            upper=args.upper,
        ),
    )
    print_comparison(cube)
    return 0 if cube.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
