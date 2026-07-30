import numpy as np
import time
from numpy.typing import NDArray


def lagrange_reduction_components(C11, C22, C12, loops=1000):
    """
    Lagrange reduction of (C11, C22, C12) in place.
    M is an array of right-multiplication matrices such that
    the reduced C_R = M.T @ C @ M for each entry.
    """
    assert C11.shape == C22.shape == C12.shape, "Must have same shape"
    C11 = np.asarray(C11, dtype=float).copy()
    C22 = np.asarray(C22, dtype=float).copy()
    C12 = np.asarray(C12, dtype=float).copy()

    # Initialize M as identity matrices everywhere, shape (...,2,2)
    M = np.zeros(C11.shape + (2, 2), dtype=float)
    M[..., 0, 0] = 1.0
    M[..., 1, 1] = 1.0

    # Define the three right-multiplication matrices
    m1 = np.array([[1, 0], [0, -1]], dtype=float)
    m2 = np.array([[0, 1], [1, 0]], dtype=float)
    m3 = np.array([[1, -1], [0, 1]], dtype=float)

    for i in range(loops):
        mask1 = C12 < 0
        # m1 (flip) operation
        C12[mask1] *= -1

        M_mask = M[mask1]
        if M_mask.shape[0] > 0:
            M[mask1] = M_mask @ m1

        mask2 = C22 < C11
        # m2 (swap) operation
        C11[mask2], C22[mask2] = C22[mask2].copy(), C11[mask2].copy()

        M_mask = M[mask2]
        if M_mask.shape[0] > 0:
            M[mask2] = M_mask @ m2

        mask3 = 2 * C12 > C11
        # Stop the loop if no changes are made
        if not np.any(mask1 | mask2 | mask3):
            break

        # m3 operation
        C22[mask3] += C11[mask3] - 2 * C12[mask3]
        C12[mask3] -= C11[mask3]

        M_mask = M[mask3]
        if M_mask.shape[0] > 0:
            M[mask3] = M_mask @ m3

        if i + 1 == loops and loops > 200:
            print("Warning: Not enough lagrange reduction loops")

            # print example of non-reduced C
            index = np.where(mask1 | mask2 | mask3)
            print("Indices of non-reduced C:", index)
            print(
                f"Example of non-reduced C: C11={C11[index][0]}, C22={C22[index][0]}, C12={C12[index][0]}"
            )

    if C11.shape == ():
        return float(C11), float(C22), float(C12), M
    return C11, C22, C12, M


def lagrange_reduction(C, loops=1000):
    """Return a Lagrange-reduced copy of symmetric 2x2 matrices C.

    This function does **not** modify the input `C`.
    Returns (C_reduced, M) where C_reduced = M.T @ C @ M.
    """
    C_in = np.asarray(C, dtype=float)
    assert C_in.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"
    assert np.allclose(C_in[..., 1, 0], C_in[..., 0, 1], equal_nan=True), (
        "C must be symmetric"
    )

    # Work on a copy so the caller's array is never modified.
    C_out = C_in.copy()

    # Extract views (no copy) from the output array
    C11, C22, C12 = C_out[..., 0, 0], C_out[..., 1, 1], C_out[..., 0, 1]

    C11r, C22r, C12r, M = lagrange_reduction_components(C11, C22, C12, loops=loops)

    # Write back (views already point into C_out, but keep explicit for scalar path)
    C_out[..., 0, 0] = C11r
    C_out[..., 1, 1] = C22r
    C_out[..., 0, 1] = C12r
    C_out[..., 1, 0] = C12r

    return C_out, M


def lagrange_reduction_history(C, loops=1000, return_transforms=False):
    """Return every metric visited by the scalar 2D Lagrange reduction.

    The first entry is the input metric and the last entry is the same reduced
    metric returned by :func:`lagrange_reduction`.  Keeping this small,
    non-vectorized trace here avoids coupling static plots to the Qt reduction
    visualizer.  With ``return_transforms=True``, return a second tuple
    containing the right-multiplication matrix used for each successive
    history entry.
    """
    C_current = np.asarray(C, dtype=float).copy()
    if C_current.shape != (2, 2):
        raise ValueError("C must have shape (2, 2)")
    if not np.allclose(C_current, C_current.T):
        raise ValueError("C must be symmetric")
    if not np.all(np.isfinite(C_current)) or np.linalg.det(C_current) <= 0:
        raise ValueError("C must be a finite positive-definite metric")

    history = [C_current.copy()]
    transforms = []

    m1 = np.array([[1, 0], [0, -1]], dtype=float)
    m2 = np.array([[0, 1], [1, 0]], dtype=float)
    m3 = np.array([[1, -1], [0, 1]], dtype=float)

    def result():
        history_array = np.stack(history)
        if return_transforms:
            return history_array, tuple(transforms)
        return history_array

    for _ in range(loops):
        changed = False

        if C_current[1, 1] < C_current[0, 0]:
            C_current[0, 0], C_current[1, 1] = (
                C_current[1, 1],
                C_current[0, 0],
            )
            history.append(C_current.copy())
            transforms.append(m2.copy())
            changed = True

        if C_current[0, 1] < 0:
            C_current[0, 1] *= -1
            C_current[1, 0] = C_current[0, 1]
            history.append(C_current.copy())
            transforms.append(m1.copy())
            changed = True

        if 2 * C_current[0, 1] > C_current[0, 0]:
            C_current[1, 1] += (
                C_current[0, 0] - 2 * C_current[0, 1]
            )
            C_current[0, 1] -= C_current[0, 0]
            C_current[1, 0] = C_current[0, 1]
            history.append(C_current.copy())
            transforms.append(m3.copy())
            changed = True

        if not changed:
            return result()

    raise RuntimeError(f"Lagrange reduction did not converge in {loops} iterations")


def lagrange_reduction_F(F, loops=1000, returnM=False):
    """
    Lagrange reduction acting on F in place.
    We use lagrange reduction on C=F^T F to get the unimodular matrices M,
    then apply F_R <- F M.
    """
    F = np.asarray(F, dtype=float)
    if F.shape[-2:] != (2, 2):
        raise ValueError("F must have shape (..., 2, 2)")

    # Columns of F are e1, e2
    e1 = F[..., :, 0]
    e2 = F[..., :, 1]

    # Components of C = F^T F
    C11 = np.einsum("...i,...i->...", e1, e1)
    C22 = np.einsum("...i,...i->...", e2, e2)
    C12 = np.einsum("...i,...i->...", e1, e2)

    # Run your existing Lagrange reduction on the components, but ask for M
    _, _, _, M = lagrange_reduction_components(C11, C22, C12, loops=loops)

    assert M.shape == F.shape, "M must have shape matching F"

    # Apply the accumulated unimodular matrices to F from the right:
    F_R = F @ M

    return F_R, M


def in_elastic_domain(C11, C22, C12) -> NDArray[np.bool_]:
    """Return True where (C11, C22, C12) lies in the elastic domain."""
    C11a = np.asarray(C11)
    C22a = np.asarray(C22)
    C12a = np.asarray(C12)

    Cmin = np.minimum(C11a, C22a)
    inside = (Cmin > 0) & (np.abs(C12a) <= 0.5 * Cmin)
    return inside


def elastic_domain_quadrant(C) -> NDArray[np.int_]:
    """Return elastic-domain quadrant label (0..3), or nan if outside."""
    assert C.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"
    assert np.allclose(C[..., 1, 0], C[..., 0, 1], equal_nan=True), (
        "C must be symmetric"
    )

    C11a = np.asarray(C[..., 0, 0])
    C22a = np.asarray(C[..., 1, 1])
    C12a = np.asarray(C[..., 0, 1])

    q0 = (C11a > 0) & (C11a <= C22a) & (C12a >= 0) & (C12a <= 0.5 * C11a)
    q1 = (C11a > 0) & (C11a <= C22a) & (C12a < 0) & (C12a >= -0.5 * C11a)
    q2 = (C22a > 0) & (C22a <= C11a) & (C12a >= 0) & (C12a <= 0.5 * C22a)
    q3 = (C22a > 0) & (C22a <= C11a) & (C12a < 0) & (C12a >= -0.5 * C22a)

    labels = np.full(C11a.shape, -1)
    labels[q0] = 0
    labels[q1] = 1
    labels[q2] = 2
    labels[q3] = 3

    return labels


def plastic_reduction_components(C11, C22, C12, loops=1000, compute_M=False):
    """Vectorized plastic reduction of symmetric 2x2 C via component updates.
    Also returns M such that C_reduced = M.T @ C @ M. If compute_M=False,
    M is a small zero placeholder to keep the return signature consistent.
    """
    C11a = np.asarray(C11, dtype=float)
    C22a = np.asarray(C22, dtype=float)
    C12a = np.asarray(C12, dtype=float)

    C11b, C22b, C12b = np.broadcast_arrays(C11a, C22a, C12a)

    # Work on copies to avoid surprising in-place effects for views
    a = C11b.copy()
    b = C22b.copy()
    c = C12b.copy()

    # Accumulate right-multiplication matrices M such that C_reduced = M.T @ C @ M
    if compute_M:
        M = np.zeros(a.shape + (2, 2), dtype=float)
        M[..., 0, 0] = 1.0
        M[..., 1, 1] = 1.0
    else:
        M = None

    def _in_elastic(a_, b_, c_):
        Cmin = np.minimum(a_, b_)
        invalid = ~np.isfinite(a_) | ~np.isfinite(b_) | ~np.isfinite(c_)
        return invalid | ((Cmin >= 0) & (np.abs(c_) <= 0.5 * Cmin))

    done = False
    for _ in range(loops):
        inside = _in_elastic(a, b, c)
        active = ~inside
        if not np.any(active):
            done = True
            break

        # The active-shear implementation tries horizontal shears first, so
        # ties (a == b) follow U rather than V.
        use_U = active & (a <= b)
        use_V = active & ~use_U

        # m = sign(-c/a) or sign(-c/b), but avoid divide-by-zero / NaNs
        mU = np.zeros_like(a)
        mV = np.zeros_like(a)

        if np.any(use_U):
            denom = np.where(a == 0, 1.0, a)
            mU = np.sign(-c / denom)
            mU = np.where(np.isfinite(mU), mU, 0.0)

        if np.any(use_V):
            denom = np.where(b == 0, 1.0, b)
            mV = np.sign(-c / denom)
            mV = np.where(np.isfinite(mV), mV, 0.0)

        # --- Apply U_m where selected: W = [[1,m],[0,1]] ---
        # a' = a
        # c' = c + m*a
        # b' = b + 2*m*c + m^2*a
        if np.any(use_U):
            m = mU
            c_new = c + m * a
            b_new = b + 2.0 * m * c + (m * m) * a

            c = np.where(use_U, c_new, c)
            b = np.where(use_U, b_new, b)
            # a unchanged for U

            if compute_M:
                # Accumulate M <- M @ U_m (per-entry m)
                M_mask = M[use_U]
                W = np.zeros_like(M_mask)
                W[..., 0, 0] = 1.0
                W[..., 1, 1] = 1.0
                W[..., 0, 1] = m[use_U]
                M[use_U] = M_mask @ W

        # --- Apply V_m where selected: W = [[1,0],[m,1]] ---
        # b' = b
        # c' = c + m*b
        # a' = a + 2*m*c + m^2*b
        if np.any(use_V):
            m = mV
            c_new = c + m * b
            a_new = a + 2.0 * m * c + (m * m) * b

            c = np.where(use_V, c_new, c)
            a = np.where(use_V, a_new, a)
            # b unchanged for V

            if compute_M:
                # Accumulate M <- M @ V_m (per-entry m)
                M_mask = M[use_V]
                W = np.zeros_like(M_mask)
                W[..., 0, 0] = 1.0
                W[..., 1, 1] = 1.0
                W[..., 1, 0] = m[use_V]
                M[use_V] = M_mask @ W

    if not done:
        print("Warning! Not enough loops in plastic reduction!")
    if not compute_M:
        # Placeholder to keep return signature consistent without large allocations.
        M = np.zeros((2, 2), dtype=float)
    # Preserve scalar return type when inputs are scalars
    if a.shape == ():
        return float(a), float(b), float(c), M
    return a, b, c, M


def plastic_reduction(C, loops=1000, compute_M=False):
    """Return a plastic-reduced copy of symmetric 2x2 matrices C.

    This function does **not** modify the input `C`.
    Returns (C_reduced, M) where C_reduced = M.T @ C @ M. If compute_M=False,
    M is a small zero placeholder to keep the return signature consistent.
    """
    assert C.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"
    assert np.allclose(C[..., 1, 0], C[..., 0, 1], equal_nan=True), (
        "C must be symmetric"
    )

    # Work on a copy so the caller's array is never modified.
    C_in = np.asarray(C, dtype=float)
    C_out = C_in.copy()

    # Extract views (no copy) from the output array
    C11, C22, C12 = C_out[..., 0, 0], C_out[..., 1, 1], C_out[..., 0, 1]

    C11r, C22r, C12r, M = plastic_reduction_components(
        C11, C22, C12, loops=loops, compute_M=compute_M
    )

    C_out[..., 0, 0] = C11r
    C_out[..., 1, 1] = C22r
    C_out[..., 0, 1] = C12r
    C_out[..., 1, 0] = C12r

    return C_out, M


def plastic_reduction_history(
    C,
    loops=1000,
    return_M=False,
    require_convergence=True,
):
    """Return every metric visited by the scalar plastic reduction.

    This is the reusable metric-space form of
    ``rotation_free_active_shear_reduction`` in
    ``LagrangeReduction/LagrangeReduction.py``.  It follows the same unit
    horizontal/vertical shear choices without depending on the Qt visualizer.
    """
    C_current = np.asarray(C, dtype=float).copy()
    if C_current.shape != (2, 2):
        raise ValueError("C must have shape (2, 2)")
    if not np.allclose(C_current, C_current.T):
        raise ValueError("C must be symmetric")
    if not np.all(np.isfinite(C_current)) or np.linalg.det(C_current) <= 0:
        raise ValueError("C must be a finite positive-definite metric")

    history = [C_current.copy()]
    M_total = np.eye(2, dtype=int)

    def result():
        history_array = np.stack(history)
        if return_M:
            return history_array, M_total
        return history_array

    for _ in range(loops):
        a = C_current[0, 0]
        b = C_current[1, 1]
        c = C_current[0, 1]
        if min(a, b) >= 0 and abs(c) <= 0.5 * min(a, b):
            return result()

        denominator = a if a <= b else b
        if denominator == 0:
            raise RuntimeError("Plastic reduction encountered a zero diagonal")
        m = int(np.sign(-c / denominator))
        if m == 0:
            raise RuntimeError("Plastic reduction made no progress")

        if a <= b:
            shear = np.array([[1, m], [0, 1]], dtype=int)
            C_current[0, 1] = c + m * a
            C_current[1, 0] = C_current[0, 1]
            C_current[1, 1] = b + 2 * m * c + m * m * a
        else:
            shear = np.array([[1, 0], [m, 1]], dtype=int)
            C_current[0, 1] = c + m * b
            C_current[1, 0] = C_current[0, 1]
            C_current[0, 0] = a + 2 * m * c + m * m * b
        M_total = M_total @ shear
        history.append(C_current.copy())

    if require_convergence:
        raise RuntimeError(f"Plastic reduction did not converge in {loops} iterations")
    return result()

    ########


def reduce_components(C11, C22, C12, loops=1000, fullReduction=False):
    """
    Plastic reduction of (C11, C22, C12) in place.
    If fullReduction=True, apply the final m1/m2 operations to map into the
    fundamental domain (C12 >= 0 and C11 <= C22).
    Returns (C11, C22, C12, M) where C_reduced = M.T @ C @ M.
    """
    C11r, C22r, C12r, M = plastic_reduction_components(
        C11, C22, C12, loops=loops, compute_M=True
    )

    if fullReduction:
        m1 = np.array([[1, 0], [0, -1]], dtype=float)
        m2 = np.array([[0, 1], [1, 0]], dtype=float)

        # Scalar path
        if np.asarray(C11r).shape == ():
            if C12r < 0:
                C12r = -C12r
                M = M @ m1
            if C22r < C11r:
                C11r, C22r = C22r, C11r
                M = M @ m2
            return float(C11r), float(C22r), float(C12r), M

        # Vectorized path
        mask1 = C12r < 0
        if np.any(mask1):
            C12r[mask1] *= -1
            M_mask = M[mask1]
            if M_mask.shape[0] > 0:
                M[mask1] = M_mask @ m1

        mask2 = C22r < C11r
        if np.any(mask2):
            C11r[mask2], C22r[mask2] = C22r[mask2].copy(), C11r[mask2].copy()
            M_mask = M[mask2]
            if M_mask.shape[0] > 0:
                M[mask2] = M_mask @ m2

    if np.asarray(C11r).shape == ():
        return float(C11r), float(C22r), float(C12r), M
    return C11r, C22r, C12r, M


def reduce(C, loops=1000, fullReduction=False):
    """Return a plastic-reduced copy of symmetric 2x2 matrices C.

    If fullReduction=True, apply the final m1/m2 operations to map into the
    fundamental domain (C12 >= 0 and C11 <= C22).
    Returns (C_reduced, M) where C_reduced = M.T @ C @ M.
    """
    assert C.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"
    assert np.allclose(C[..., 1, 0], C[..., 0, 1], equal_nan=True), (
        "C must be symmetric"
    )

    C_in = np.asarray(C, dtype=float)
    C_out = C_in.copy()

    C11, C22, C12 = C_out[..., 0, 0], C_out[..., 1, 1], C_out[..., 0, 1]
    C11r, C22r, C12r, M = reduce_components(
        C11, C22, C12, loops=loops, fullReduction=fullReduction
    )

    C_out[..., 0, 0] = C11r
    C_out[..., 1, 1] = C22r
    C_out[..., 0, 1] = C12r
    C_out[..., 1, 0] = C12r

    return C_out, M


def MCheck_reduction(reduction, C, rtol=1e-6, atol_scale=1e-6):
    C_R, M = reduction(C)
    MT = M.swapaxes(-1, -2)
    C_R_test = MT @ C @ M
    scale = np.nanmax(np.abs(C_R))
    atol = max(1.0, scale) * atol_scale
    ok = np.allclose(C_R_test, C_R, rtol=rtol, atol=atol)
    print(C_R)
    print(M)
    if not ok:
        diff = C_R_test - C_R
        abs_diff = np.abs(diff)
        max_abs = np.nanmax(abs_diff)
        denom = np.maximum(np.abs(C_R), 1e-12)
        rel_diff = abs_diff / denom
        max_rel = np.nanmax(rel_diff)
        idx = np.unravel_index(np.nanargmax(abs_diff), abs_diff.shape)
        print(f"Sanity check failed for {reduction}")
        print(f"  shape: {C_R.shape}, rtol={rtol}, atol={atol}")
        print(f"  max_abs_diff={max_abs:.6g}, max_rel_diff={max_rel:.6g}")
        print(f"  worst index: {idx}")
        print(f"  C_R_test{idx}={C_R_test[idx]}, C_R{idx}={C_R[idx]}")
    assert ok, f"{reduction} failed sanity check (rtol={rtol}, atol={atol})"


def MCheck_reductions():
    F = np.array([[1, 1], [0, 1]])
    C = F.T @ F
    # Easy
    MCheck_reduction(lagrange_reduction, C)
    MCheck_reduction(lambda C: plastic_reduction(C, compute_M=True), C)
    MCheck_reduction(reduce, C)
    MCheck_reduction(lambda C: reduce(C, fullReduction=True), C)
    C_lr, _ = lagrange_reduction(C)
    C_red_full, _ = reduce(C, fullReduction=True)
    assert np.allclose(C_lr, C_red_full), "fullReduction mismatch vs lagrange (easy)"
    # Medium
    F2 = F @ F @ F.T @ np.array([[1, -1.4], [0, 1]])
    C2 = F2.T @ F2
    MCheck_reduction(lagrange_reduction, C2)
    MCheck_reduction(lambda C: plastic_reduction(C, compute_M=True), C2)
    MCheck_reduction(reduce, C2)
    MCheck_reduction(lambda C: reduce(C, fullReduction=True), C2)
    C2_lr, _ = lagrange_reduction(C2)
    C2_red_full, _ = reduce(C2, fullReduction=True)
    assert np.allclose(C2_lr, C2_red_full), (
        "fullReduction mismatch vs lagrange (medium)"
    )
    print("Passed M checks!")


def debug_plastic_reduction():
    F = np.array([[1, 1], [0, 1]])
    C = F.T @ F
    C_R = plastic_reduction(C)
    print(C)
    print(C_R)


def createHardC(steps=1000, seed=0):
    """
    Construct a symmetric C with det=1 by repeatedly applying random
    m1/m2/m3 congruence transforms: C <- M.T @ C @ M.
    """
    rng = np.random.default_rng(seed)
    a = 2.0
    C = np.array([[a, 0.3], [0.3, 1.0 / a]], dtype=float)

    m1 = np.array([[1, 0], [0, -1]], dtype=float)
    m2 = np.array([[0, 1], [1, 0]], dtype=float)
    m3 = np.array([[1, -1], [0, 1]], dtype=float)
    mats = [m1, m2, m3]

    for _ in range(steps):
        M = mats[int(rng.integers(0, 3))]
        C = M.T @ C @ M

    return C


def speedTest_reduction(n=1, steps=80, seed=0):
    """
    Compare timing of lagrange_reduction vs reduce(fullReduction=True)
    on a difficult C grid generated by random m1/m2/m3 transforms.
    """
    base_C = createHardC(steps=steps, seed=seed)
    C = np.broadcast_to(base_C, (n, n, 2, 2)).copy()
    print(C)
    print("Lagrange reduction")
    t0 = time.perf_counter()
    MCheck_reduction(lagrange_reduction, C)
    t1 = time.perf_counter()

    print("Plastic reduction")
    t2 = time.perf_counter()

    def r(C):
        return reduce(C, fullReduction=True)

    MCheck_reduction(r, C)
    t3 = time.perf_counter()

    print(f"Grid size: {n}x{n} ({n * n} matrices)")
    print(f"Random transforms: {steps}, seed={seed}")
    print(f"lagrange_reduction: {t1 - t0:.3f} s")
    print(f"reduce(fullReduction=True): {t3 - t2:.3f} s")


def reductionTest():
    F = np.array([[47, 95], [215, 460]])
    F_R, _ = lagrange_reduction_F(F)
    print(F_R)
    a = np.array((3, 4))
    b = np.array((300, 500))
    c = a * 2352 - 12 * b
    d = a * 2322 - 14 * b
    F = np.array((c, d)).T
    print(F)
    F_R, _ = lagrange_reduction_F(F)
    print(F_R)


if __name__ == "__main__":
    MCheck_reductions()
    speedTest_reduction()
    reductionTest()
