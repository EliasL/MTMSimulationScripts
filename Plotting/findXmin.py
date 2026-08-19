"""xmin-selection helpers used by the power-law analysis scripts.

The functions are intentionally general and expose several alternative
selection strategies.  The standard simulation workflow is to split paired
post-yield events with ``Delta E_R`` ``simpleDrop``, fit only irreversible
``Delta E_S`` events, evaluate every observed candidate for the global xmin,
and then perform the maximum-likelihood fit.  Callers using another strategy
or population should state that choice explicitly.
"""

import numpy as np
import warnings
from MTMath.evaluatePowerlawFit import (
    Fit,
    Truncated_Power_Law,
    evaluate_xmin,
    evaluate_xmin_distances,
)
from powerlaw import Distribution
from matplotlib import pyplot as plt
import os
import json
import hashlib
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional


def _smooth_dip_distances(xmins, distances, smoothing):
    if smoothing is None or smoothing == "":
        return distances

    xmins = np.asarray(xmins, dtype=float)
    distances = np.asarray(distances, dtype=float)
    mask = np.isfinite(xmins) & np.isfinite(distances)
    if mask.sum() < 3:
        return distances

    method = str(smoothing).lower()
    x = xmins[mask]
    y = distances[mask]

    if method in {"moving_average", "ma"}:
        window = 10
        if y.size < window:
            return distances
        kernel = np.ones(window, dtype=float) / float(window)
        y_smooth = np.full_like(y, np.nan)
        valid = np.convolve(y, kernel, mode="valid")
        half = (y.size - valid.size) // 2
        y_smooth[half : half + valid.size] = valid
    elif method in {"spline", "cubic_spline"}:
        from scipy.interpolate import LSQUnivariateSpline

        knots = 20
        if y.size <= knots + 3:
            return distances
        logx = np.log10(x)
        knot_positions = np.linspace(logx.min(), logx.max(), knots)[1:-1]
        spline = LSQUnivariateSpline(logx, y, t=knot_positions, k=3)
        y_smooth = spline(logx)
    else:
        raise ValueError(
            "smoothing must be one of: None, 'moving_average'/'ma', 'spline'"
        )

    smoothed = distances.copy()
    smoothed[mask] = y_smooth
    return smoothed


def _save_debug_fig(fig, filename, rect=None):
    debug_path = "Plots/powerLaw/debug/"
    os.makedirs(debug_path, exist_ok=True)
    full_path = f"{debug_path}{filename}"
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(full_path)
    print(f"Saved figure to {full_path}")
    plt.close(fig)
    return full_path


def _smoothing_suffix(smoothing):
    if smoothing is None:
        return "raw"
    method = str(smoothing).lower()
    if method in {"moving_average", "ma"}:
        return "ma"
    if method in {"spline", "cubic_spline"}:
        return "spline"
    return method


def _plot_region_shading(ax, region_start, region_end, min_distance, region_level):
    if region_start is None or region_end is None:
        return
    ax.fill_between(
        [region_start, region_end],
        [min_distance, min_distance],
        [region_level, region_level],
        color="0.85",
        alpha=0.6,
        zorder=0,
        label=r"$D_{{\min}}$ to $D_{{\min}}+0.05$",
    )


def _plot_dip_curve(ax, dip_xmin_values, dip_distances, label="D (dip)"):
    if dip_xmin_values is None or dip_distances is None:
        return
    ax.plot(
        dip_xmin_values,
        dip_distances,
        label=label,
        color="tab:orange",
        alpha=0.9,
    )


def _find_region_bounds(distances, xmins, delta=0.05):
    distances = np.asarray(distances, dtype=float)
    xmins = np.asarray(xmins, dtype=float)
    valid_idx = np.where(np.isfinite(distances) & np.isfinite(xmins))[0]
    if distances.size == 0 or valid_idx.size == 0:
        return None, None, None, None, None

    min_pos = int(np.argmin(distances[valid_idx]))
    min_idx = int(valid_idx[min_pos])
    distances_v = distances[valid_idx]
    xmins_v = xmins[valid_idx]
    min_distance = float(distances_v[min_pos])
    level = min_distance + float(delta)

    def _interp_x_at_level(x1, x2, d1, d2, target):
        if not (np.isfinite(d1) and np.isfinite(d2)) or d1 == d2:
            return float(x2)
        xp = np.array([d1, d2], dtype=float)
        fp = np.array([x1, x2], dtype=float)
        order = np.argsort(xp)
        return float(np.interp(target, xp[order], fp[order]))

    left_cross = np.where(distances_v[: min_pos + 1] >= level)[0]
    if left_cross.size:
        left_hi = int(left_cross[-1])
        left_lo = left_hi + 1
        region_start = _interp_x_at_level(
            xmins_v[left_hi],
            xmins_v[left_lo],
            distances_v[left_hi],
            distances_v[left_lo],
            level,
        )
    else:
        region_start = float(xmins_v[0])

    right_cross = np.where(distances_v[min_pos:] >= level)[0]
    if right_cross.size:
        right_hi = int(min_pos + right_cross[0])
        right_lo = right_hi - 1
        region_end = _interp_x_at_level(
            xmins_v[right_lo],
            xmins_v[right_hi],
            distances_v[right_lo],
            distances_v[right_hi],
            level,
        )
    else:
        region_end = float(xmins_v[-1])

    return min_distance, level, region_start, region_end, min_idx


def _is_uniform_spacing(values, rtol=1e-3, atol=1e-12):
    """
    Return True if a 1D array is approximately uniformly spaced.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return True
    diffs = np.diff(values)
    if not np.isfinite(diffs).all():
        return False
    ref = np.nanmedian(diffs)
    return np.allclose(diffs, ref, rtol=rtol, atol=atol)


def _prepare_uniform_log_grid(x, y, *, resample=True, num=None):
    """
    Prepare a uniform grid in log10(x) and optionally interpolate y onto it.

    Parameters
    ----------
    x, y : array-like
        Positive x values and corresponding y values.
    resample : bool
        If True, interpolate y onto a uniform log grid when needed.
    num : int or None
        Number of points for the uniform grid. Defaults to len(valid data).

    Returns
    -------
    logx_uniform : ndarray
    x_uniform : ndarray
    y_uniform : ndarray
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if mask.sum() < 3:
        return None, None, None

    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    logx = np.log10(x)
    if num is None:
        num = int(logx.size)
    if num < 3:
        return None, None, None

    # Collapse any duplicate logx values by averaging y.
    logx_unique, inv, counts = np.unique(logx, return_inverse=True, return_counts=True)
    if logx_unique.size != logx.size:
        y_accum = np.zeros_like(logx_unique, dtype=float)
        np.add.at(y_accum, inv, y)
        y = y_accum / counts
        logx = logx_unique
        x = 10.0**logx

    if not resample:
        if not _is_uniform_spacing(logx):
            raise ValueError("log10(x) spacing is not uniform; enable resample.")
        return logx, x, y

    logx_uniform = np.linspace(logx.min(), logx.max(), num)
    if logx.size == logx_uniform.size and _is_uniform_spacing(logx):
        if np.allclose(logx, logx_uniform, rtol=1e-6, atol=1e-10):
            return logx, x, y

    from scipy.interpolate import interp1d

    interp = interp1d(
        logx,
        y,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    y_uniform = interp(logx_uniform)
    x_uniform = 10.0**logx_uniform
    return logx_uniform, x_uniform, y_uniform


# Legacy SG-based helpers below are retained for optional/experimental methods.


def _minimum_window_length(polyorder):
    """
    Return the minimum odd SG window length valid for a given polyorder.
    """
    min_len = int(polyorder) + 2
    if min_len % 2 == 0:
        min_len += 1
    return max(3, min_len)


def _sanitize_window_length(n_points, polyorder, window_length):
    """
    Sanitize a window length to be odd, within bounds, and valid for SG.
    """
    if window_length is None:
        return None
    if n_points < 3:
        return None
    W = int(round(window_length))
    if W % 2 == 0:
        W += 1
    W = max(W, _minimum_window_length(polyorder))
    if W > n_points:
        W = n_points if n_points % 2 == 1 else n_points - 1
    if W <= polyorder:
        return None
    return W


def _sanitize_window_candidates(n_points, polyorder, window_candidates):
    """
    Build a sorted list of odd window lengths valid for SG.
    """
    min_len = _minimum_window_length(polyorder)
    if window_candidates is None:
        max_len = min(n_points, 51)
        candidates = list(range(min_len, max_len + 1, 2))
    else:
        candidates = []
        for value in window_candidates:
            W = int(round(value))
            if W % 2 == 0:
                W += 1
            if W < min_len or W > n_points:
                continue
            candidates.append(W)
        candidates = sorted(set(candidates))
    return candidates


def _choose_stable_window_length(
    logx, y, polyorder, window_candidates, stability_tol=0.2
):
    """
    Choose the smallest window where derivatives stabilize across scales.

    The stability score compares successive window derivatives and selects
    the smallest window with score <= stability_tol.
    """
    if not window_candidates:
        return None
    if len(window_candidates) == 1:
        return window_candidates[0]

    from scipy.signal import savgol_filter

    delta = float(np.median(np.diff(logx)))
    derivs = []
    for W in window_candidates:
        d = savgol_filter(y, W, polyorder, deriv=1, delta=delta, mode="interp")
        derivs.append(d)

    eps = 1e-12
    for i in range(len(window_candidates) - 1):
        d0 = derivs[i]
        d1 = derivs[i + 1]
        mask = np.isfinite(d0) & np.isfinite(d1)
        if mask.sum() < 5:
            continue
        scale = np.median(np.abs(d1[mask])) + eps
        score = np.median(np.abs(d0[mask] - d1[mask])) / scale
        if score <= stability_tol:
            return window_candidates[i]

    return window_candidates[-1]


def _estimate_noise_sigma(y, window_length, polyorder, method="mad"):
    """
    Estimate observation noise sigma from high-frequency residuals.
    """
    from scipy.signal import savgol_filter
    from scipy.stats import median_abs_deviation

    y = np.asarray(y, dtype=float)
    smooth = savgol_filter(y, window_length, polyorder, deriv=0, mode="interp")
    resid = y - smooth

    if method == "mad":
        sigma = median_abs_deviation(resid, scale="normal", nan_policy="omit")
    else:
        sigma = np.nanstd(resid, ddof=1)

    if not np.isfinite(sigma) or sigma <= 0:
        diffs = np.diff(y)
        sigma = median_abs_deviation(
            diffs, scale="normal", nan_policy="omit"
        ) / np.sqrt(2.0)
    return float(sigma)


def _sg_derivative_se(window_length, polyorder, delta, sigma):
    """
    Approximate the SG first-derivative standard error from coefficient norm.
    """
    from scipy.signal import savgol_coeffs

    coeffs = savgol_coeffs(window_length, polyorder, deriv=1, delta=delta, use="dot")
    return float(sigma) * float(np.sqrt(np.sum(coeffs**2)))


def _contiguous_true_runs(mask):
    """
    Return list of (start, end) indices for contiguous True runs.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size - 1]
    return list(zip(starts.tolist(), ends.tolist()))


@dataclass
class SizerResult:
    """
    Container for a 1D SiZer map.

    Notes
    -----
    SiZer is a scale-space method: every entry corresponds to a location x
    and a smoothing scale h. The core estimator here is Gaussian local linear
    regression of y on x.
    """

    x: np.ndarray
    x_used: np.ndarray
    h_grid: np.ndarray
    deriv: np.ndarray
    se: np.ndarray
    ess: np.ndarray
    classification: np.ndarray
    sparse_mask: np.ndarray
    q_values: np.ndarray
    alpha: float
    inference: str
    smooth: Optional[np.ndarray] = None
    ess_min: float = 5.0
    bootstrap_mode: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def build_bandwidth_grid(
    x,
    *,
    h_grid=None,
    h_min=None,
    h_max=None,
    n_scales=25,
    spacing="log",
):
    """
    Build a bandwidth grid for SiZer.

    Uses x-range and minimum spacing to select defaults if h_grid is None.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.array([], dtype=float)
    x = np.unique(np.sort(x))
    span = float(x[-1] - x[0])
    diffs = np.diff(x)
    diffs = diffs[diffs > 0]
    min_spacing = float(np.min(diffs)) if diffs.size else span

    if h_grid is not None:
        h = np.asarray(h_grid, dtype=float)
        h = h[np.isfinite(h) & (h > 0)]
        h.sort()
        return h

    if h_min is None:
        h_min = max(min_spacing * 1.5, span * 1e-3)
    if h_max is None:
        h_max = max(h_min * 2.0, span * 0.5)

    if h_min <= 0 or h_max <= 0 or h_min >= h_max:
        raise ValueError("Invalid h_min/h_max for bandwidth grid.")

    spacing = str(spacing).lower()
    if spacing == "log":
        h = np.logspace(np.log10(h_min), np.log10(h_max), int(n_scales))
    elif spacing == "linear":
        h = np.linspace(h_min, h_max, int(n_scales))
    else:
        raise ValueError("spacing must be 'log' or 'linear'.")

    return h


def _apply_x_transform(x, x_transform):
    """
    Apply a monotone transform to x for SiZer regression.

    Returns transformed x and a label describing the transform.
    """
    if x_transform is None:
        return x, "identity"
    if isinstance(x_transform, str):
        key = x_transform.strip().lower()
        if key in {"identity", "none"}:
            return x, "identity"
        if key in {"log", "ln"}:
            if np.any(x <= 0):
                raise ValueError("x must be positive for log transform.")
            return np.log(x), "log"
        if key in {"log10", "log_10"}:
            if np.any(x <= 0):
                raise ValueError("x must be positive for log10 transform.")
            return np.log10(x), "log10"
        raise ValueError("x_transform must be None, 'log', 'log10', or a callable.")
    if callable(x_transform):
        xt = np.asarray(x_transform(x), dtype=float)
        return xt, "callable"
    raise ValueError("x_transform must be None, 'log', 'log10', or a callable.")


def local_linear_gaussian_at_point(
    x,
    y,
    x0,
    h,
    *,
    sigma2=None,
):
    """
    Gaussian local linear regression at a single x0.

    Fits y_i ~= beta0 + beta1*(x_i - x0) with Gaussian weights.
    Returns beta0, beta1, se(beta1), ESS, and sigma2 used for inference.

    Variance model:
    - If sigma2 is None, uses a weighted residual variance with dof ~= ESS-2.
      This assumes locally homoscedastic errors and treats ESS as an effective
      sample size (approximation).
    - If sigma2 is provided, it is treated as a global noise variance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - float(x0)
    if h <= 0:
        return np.nan, np.nan, np.nan, 0.0, np.nan

    w = np.exp(-0.5 * (dx / float(h)) ** 2)
    if not np.isfinite(w).any():
        return np.nan, np.nan, np.nan, 0.0, np.nan

    sum_w = float(np.sum(w))
    sum_w2 = float(np.sum(w**2))
    ess = (sum_w**2 / sum_w2) if sum_w2 > 0 else 0.0

    # Closed-form weighted normal equations for 2x2 system.
    S0 = sum_w
    S1 = float(np.sum(w * dx))
    S2 = float(np.sum(w * dx * dx))
    T0 = float(np.sum(w * y))
    T1 = float(np.sum(w * dx * y))

    det = S0 * S2 - S1 * S1
    det_scale = max(S0 * S2, S1 * S1, 1.0)
    if not np.isfinite(det) or det <= np.finfo(float).eps * det_scale:
        # Ill-conditioned local design: slope not identifiable.
        return np.nan, np.nan, np.nan, ess, np.nan

    beta0 = (S2 * T0 - S1 * T1) / det
    beta1 = (-S1 * T0 + S0 * T1) / det

    if sigma2 is None:
        resid = y - (beta0 + beta1 * dx)
        dof = max(1.0, ess - 2.0)
        sigma2 = float(np.sum(w * resid**2) / dof)

    # Inverse of [[S0, S1], [S1, S2]] is (1/det) * [[S2, -S1], [-S1, S0]]
    var_beta1 = float(sigma2 * (S0 / det))
    se_beta1 = np.sqrt(var_beta1) if var_beta1 > 0 else np.nan
    return beta0, beta1, se_beta1, ess, sigma2


def _local_linear_gaussian_map(
    x,
    y,
    x_eval,
    h,
    *,
    global_sigma=None,
):
    """
    Compute local linear Gaussian estimates for all x_eval at a single h.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    n_eval = int(x_eval.size)
    beta0 = np.full(n_eval, np.nan, dtype=float)
    beta1 = np.full(n_eval, np.nan, dtype=float)
    se = np.full(n_eval, np.nan, dtype=float)
    ess = np.zeros(n_eval, dtype=float)
    sigma2_used = np.full(n_eval, np.nan, dtype=float)

    sigma2 = None
    if global_sigma is not None and np.isfinite(global_sigma) and global_sigma > 0:
        sigma2 = float(global_sigma) ** 2

    for i, x0 in enumerate(x_eval):
        b0, b1, se1, ess_i, s2 = local_linear_gaussian_at_point(
            x, y, x0, h, sigma2=sigma2
        )
        beta0[i] = b0
        beta1[i] = b1
        se[i] = se1
        ess[i] = ess_i
        sigma2_used[i] = s2

    return beta0, beta1, se, ess, sigma2_used


def _classify_derivative(deriv, se, q, sparse_mask):
    """
    Classify derivative significance into +1, -1, 0, 9 (sparse).
    """
    classification = np.full_like(deriv, 9, dtype=int)
    mask = (~sparse_mask) & np.isfinite(deriv) & np.isfinite(se)
    if not np.any(mask):
        return classification

    upper = deriv + q * se
    lower = deriv - q * se

    classification[mask & (lower > 0)] = 1
    classification[mask & (upper < 0)] = -1
    classification[mask & (lower <= 0) & (upper >= 0)] = 0
    return classification


def _bootstrap_q_values_x(
    x,
    y,
    x_eval,
    h_grid,
    *,
    ess_min,
    alpha,
    bootstrap_reps,
    bootstrap_mode,
    random_state,
    baseline_smooth=None,
    global_sigma=None,
    show_progress=False,
    centered=True,
    deriv_ref=None,
):
    """
    Bootstrap simultaneous critical values over x for each h.

    Uses a max-|z| statistic across non-sparse x at fixed h.
    If centered=True, uses z = (deriv_b - deriv_ref) / se_b (approximation).
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(x.size)

    q_values = np.full(len(h_grid), np.nan, dtype=float)

    h_iter = enumerate(h_grid)
    if show_progress:
        h_iter = tqdm(
            h_iter, total=len(h_grid), desc="SiZer bootstrap (x)", leave=False
        )
    for h_idx, h in h_iter:
        max_stats = []
        if bootstrap_mode == "residual":
            if baseline_smooth is None:
                smooth, _, _, _, _ = _local_linear_gaussian_map(
                    x, y, x_eval, h, global_sigma=global_sigma
                )
            else:
                smooth = baseline_smooth[h_idx]
            resid = y - smooth
            resid = resid - np.nanmean(resid)
        rep_iter = range(int(bootstrap_reps))
        if show_progress:
            rep_iter = tqdm(
                rep_iter,
                total=int(bootstrap_reps),
                desc=f"h={h:.2e}",
                leave=False,
            )
        for _ in rep_iter:
            if bootstrap_mode == "pairs":
                idx = rng.integers(0, n, size=n)
                x_b = x[idx]
                y_b = y[idx]
                _, deriv_b, se_b, ess_b, _ = _local_linear_gaussian_map(
                    x_b, y_b, x_eval, h, global_sigma=global_sigma
                )
            else:
                r = rng.choice(resid, size=n, replace=True)
                y_b = smooth + r
                _, deriv_b, se_b, ess_b, _ = _local_linear_gaussian_map(
                    x, y_b, x_eval, h, global_sigma=global_sigma
                )

            if centered and deriv_ref is not None:
                z = (deriv_b - deriv_ref[h_idx]) / se_b
            else:
                z = deriv_b / se_b
            mask = (ess_b >= ess_min) & np.isfinite(z)
            if np.any(mask):
                max_stats.append(float(np.nanmax(np.abs(z[mask]))))

        if max_stats:
            q_values[h_idx] = float(np.quantile(max_stats, 1.0 - alpha))

    return q_values


def _bootstrap_q_value_xh(
    x,
    y,
    x_eval,
    h_grid,
    *,
    ess_min,
    alpha,
    bootstrap_reps,
    bootstrap_mode,
    random_state,
    global_sigma=None,
    show_progress=False,
    centered=True,
    deriv_ref=None,
):
    """
    Bootstrap a single simultaneous critical value over x and h.

    If centered=True, uses z = (deriv_b - deriv_ref) / se_b (approximation).
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(x.size)

    max_stats = []

    if bootstrap_mode == "residual":
        # Approximation: use a single pilot smooth (median h) to generate
        # residuals for all scales in the x-by-h max statistic.
        h0 = float(np.median(h_grid))
        smooth, _, _, _, _ = _local_linear_gaussian_map(
            x, y, x_eval, h0, global_sigma=global_sigma
        )
        resid = y - smooth
        resid = resid - np.nanmean(resid)

    rep_iter = range(int(bootstrap_reps))
    if show_progress:
        rep_iter = tqdm(
            rep_iter,
            total=int(bootstrap_reps),
            desc="SiZer bootstrap (x,h)",
            leave=False,
        )
    for _ in rep_iter:
        if bootstrap_mode == "pairs":
            idx = rng.integers(0, n, size=n)
            x_b = x[idx]
            y_b = y[idx]
        else:
            r = rng.choice(resid, size=n, replace=True)
            y_b = smooth + r
            x_b = x

        max_z = -np.inf
        for i_h, h in enumerate(h_grid):
            _, deriv_b, se_b, ess_b, _ = _local_linear_gaussian_map(
                x_b, y_b, x_eval, h, global_sigma=global_sigma
            )
            if centered and deriv_ref is not None:
                z = (deriv_b - deriv_ref[i_h]) / se_b
            else:
                z = deriv_b / se_b
            mask = (ess_b >= ess_min) & np.isfinite(z)
            if np.any(mask):
                max_z = max(max_z, float(np.nanmax(np.abs(z[mask]))))

        if np.isfinite(max_z):
            max_stats.append(max_z)

    if not max_stats:
        return np.nan
    return float(np.quantile(max_stats, 1.0 - alpha))


def compute_sizer_map(
    x,
    y,
    *,
    h_grid=None,
    h_min=None,
    h_max=None,
    n_scales=25,
    spacing="log",
    alpha=0.1,
    inference="bootstrap_x",
    ess_min=5,
    bootstrap_reps=500,
    bootstrap_mode="pairs",
    random_state=None,
    global_sigma=None,
    show_progress=False,
    x_transform=None,
    bootstrap_centered=True,
):
    """
    Compute a 1D SiZer map using Gaussian local linear regression.

    Notes
    -----
    - Supports irregular x directly (no resampling).
    - Use x_transform="log10" to analyze derivatives w.r.t. log10(x).
    - Standard errors vary with (x, h) via weighted least squares.
    - Sparse regions are masked based on ESS.
    - Bootstrap inference uses a max-|z| statistic; implemented with explicit
      resampling loops for clarity (not the scipy.stats.bootstrap API).
    - Set show_progress=True to enable tqdm progress bars.
    - bootstrap_centered=True recenters bootstrap derivatives (approximation).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        raise ValueError("Need at least 3 finite samples.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_used, x_transform_label = _apply_x_transform(x, x_transform)
    if not np.isfinite(x_used).all():
        raise ValueError("x_transform produced non-finite values.")
    x_eval = x_used.copy()

    h_grid = build_bandwidth_grid(
        x_eval,
        h_grid=h_grid,
        h_min=h_min,
        h_max=h_max,
        n_scales=n_scales,
        spacing=spacing,
    )
    if h_grid.size == 0:
        raise ValueError("Empty bandwidth grid.")

    n_scales = int(h_grid.size)
    n_x = int(x_eval.size)

    deriv = np.full((n_scales, n_x), np.nan, dtype=float)
    se = np.full((n_scales, n_x), np.nan, dtype=float)
    ess = np.zeros((n_scales, n_x), dtype=float)
    smooth = None

    need_smooth = (
        inference in {"bootstrap_x", "bootstrap_xh"} and bootstrap_mode == "residual"
    )
    if need_smooth:
        smooth = np.full((n_scales, n_x), np.nan, dtype=float)

    h_iter = enumerate(h_grid)
    if show_progress:
        h_iter = tqdm(h_iter, total=int(h_grid.size), desc="SiZer scales", leave=False)
    for i, h in h_iter:
        b0, b1, se1, ess1, _ = _local_linear_gaussian_map(
            x_used, y, x_eval, h, global_sigma=global_sigma
        )
        deriv[i, :] = b1
        se[i, :] = se1
        ess[i, :] = ess1
        if need_smooth:
            smooth[i, :] = b0

    sparse_mask = (ess < float(ess_min)) | (~np.isfinite(se)) | (~np.isfinite(deriv))

    inference = str(inference).lower()
    if inference == "pointwise":
        from scipy.stats import norm

        q = float(norm.ppf(1.0 - alpha / 2.0))
        q_values = np.full(n_scales, q, dtype=float)
    elif inference == "bootstrap_x":
        q_values = _bootstrap_q_values_x(
            x_used,
            y,
            x_eval,
            h_grid,
            ess_min=ess_min,
            alpha=alpha,
            bootstrap_reps=bootstrap_reps,
            bootstrap_mode=bootstrap_mode,
            random_state=random_state,
            baseline_smooth=smooth,
            global_sigma=global_sigma,
            show_progress=show_progress,
            centered=bootstrap_centered,
            deriv_ref=deriv,
        )
    elif inference == "bootstrap_xh":
        q = _bootstrap_q_value_xh(
            x_used,
            y,
            x_eval,
            h_grid,
            ess_min=ess_min,
            alpha=alpha,
            bootstrap_reps=bootstrap_reps,
            bootstrap_mode=bootstrap_mode,
            random_state=random_state,
            global_sigma=global_sigma,
            show_progress=show_progress,
            centered=bootstrap_centered,
            deriv_ref=deriv,
        )
        q_values = np.full(n_scales, q, dtype=float)
    else:
        raise ValueError(
            "inference must be 'pointwise', 'bootstrap_x', or 'bootstrap_xh'."
        )

    classification = np.full((n_scales, n_x), 9, dtype=int)
    for i in range(n_scales):
        q = q_values[i]
        if not np.isfinite(q):
            continue
        classification[i, :] = _classify_derivative(
            deriv[i, :], se[i, :], q, sparse_mask[i, :]
        )

    return SizerResult(
        x=x,
        x_used=x_eval,
        h_grid=h_grid,
        deriv=deriv,
        se=se,
        ess=ess,
        classification=classification,
        sparse_mask=sparse_mask,
        q_values=q_values,
        alpha=float(alpha),
        inference=inference,
        smooth=smooth,
        ess_min=float(ess_min),
        bootstrap_mode=bootstrap_mode,
        metadata={
            "bootstrap_reps": int(bootstrap_reps),
            "global_sigma": global_sigma,
            "x_transform": x_transform_label,
        },
    )


def extract_plateaus_from_sizer(
    result,
    *,
    scale_range=None,
    min_scale_fraction=0.6,
    min_run_length=5,
    scale_trim=0.1,
):
    """
    Extract plateau regions based on persistence of class==0 across scales.

    scale_trim (default 0.1) excludes extreme scales when scale_range is None.
    """
    x = result.x
    h = result.h_grid
    cls = result.classification
    sparse = result.sparse_mask

    if scale_range is not None:
        h_min, h_max = scale_range
        scale_mask = (h >= float(h_min)) & (h <= float(h_max))
    else:
        if h.size >= 5 and scale_trim > 0:
            trim = float(scale_trim)
            trim = min(max(trim, 0.0), 0.45)
            lo_idx = int(np.floor(trim * h.size))
            hi_idx = int(np.ceil((1.0 - trim) * h.size)) - 1
            scale_mask = np.zeros_like(h, dtype=bool)
            scale_mask[lo_idx : hi_idx + 1] = True
        else:
            scale_mask = np.ones_like(h, dtype=bool)

    cls = cls[scale_mask]
    sparse = sparse[scale_mask]
    h_used = h[scale_mask]
    if h_used.size == 0:
        return []

    persistence, n_zero, n_eligible = _compute_sizer_persistence(cls, sparse)

    plateau_mask = persistence >= float(min_scale_fraction)
    runs = _contiguous_true_runs(plateau_mask)
    runs = [r for r in runs if (r[1] - r[0] + 1) >= int(min_run_length)]

    plateaus = []
    for start, end in runs:
        segment_p = persistence[start : end + 1]
        segment_zero = n_zero[start : end + 1]
        plateau = {
            "x_start": float(x[start]),
            "x_end": float(x[end]),
            "x_center": float(np.nanmean(x[start : end + 1])),
            "persistence_mean": float(np.nanmean(segment_p)),
            "n_supporting_scales": int(np.round(np.nanmean(segment_zero))),
            "index_start": int(start),
            "index_end": int(end),
        }
        plateaus.append(plateau)

    return plateaus


def _compute_sizer_persistence(classification, sparse_mask):
    """
    Compute persistence of class==0 across scales.
    """
    zero_mask = (classification == 0) & (~sparse_mask)
    eligible = ~sparse_mask
    n_eligible = np.sum(eligible, axis=0)
    n_zero = np.sum(zero_mask, axis=0)

    persistence = np.full_like(n_eligible, np.nan, dtype=float)
    valid = n_eligible > 0
    persistence[valid] = n_zero[valid] / n_eligible[valid]
    return persistence, n_zero, n_eligible


def plot_sizer_persistence(
    result,
    *,
    scale_range=None,
    min_scale_fraction=0.6,
    scale_trim=0.1,
    plateaus=None,
    fmin=None,
    title=None,
):
    """
    Plot persistence of the zero-class across scales.
    """
    x = result.x
    h = result.h_grid
    cls = result.classification
    sparse = result.sparse_mask
    x_transform = result.metadata.get("x_transform", "identity")

    if scale_range is not None:
        h_min, h_max = scale_range
        scale_mask = (h >= float(h_min)) & (h <= float(h_max))
    else:
        if h.size >= 5 and scale_trim > 0:
            trim = float(scale_trim)
            trim = min(max(trim, 0.0), 0.45)
            lo_idx = int(np.floor(trim * h.size))
            hi_idx = int(np.ceil((1.0 - trim) * h.size)) - 1
            scale_mask = np.zeros_like(h, dtype=bool)
            scale_mask[lo_idx : hi_idx + 1] = True
        else:
            scale_mask = np.ones_like(h, dtype=bool)

    cls = cls[scale_mask]
    sparse = sparse[scale_mask]
    if cls.size == 0:
        return None

    persistence, _, _ = _compute_sizer_persistence(cls, sparse)
    plateau_mask = persistence >= float(min_scale_fraction)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, persistence, color="tab:blue", label="Zero-class persistence")
    ax.axhline(
        float(min_scale_fraction),
        linestyle="--",
        color="0.3",
        label="Min scale fraction",
    )

    runs = _contiguous_true_runs(plateau_mask)
    for i, (start, end) in enumerate(runs):
        ax.axvspan(
            x[start],
            x[end],
            color="tab:green",
            alpha=0.2,
            label="Plateau candidates" if i == 0 else None,
        )

    if plateaus:
        for i, p in enumerate(plateaus):
            ax.axvspan(
                p["x_start"],
                p["x_end"],
                color="tab:green",
                alpha=0.35,
                label="Selected plateau" if i == 0 else None,
            )

    if fmin is not None and np.isfinite(fmin):
        ax.axvline(
            fmin,
            color="tab:green",
            linestyle="--",
            linewidth=1.5,
            label=rf"$x_{{\min}}$={fmin:.2e}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("Persistence")
    if title:
        ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    return _save_debug_fig(fig, "find_xmin_sizer_persistence.pdf")


def plot_sizer_scale_slice(result, *, h_index=None, h_value=None, fmin=None):
    """
    Plot derivative and CI band for a single bandwidth slice.
    """
    x = result.x
    h = result.h_grid
    x_transform = result.metadata.get("x_transform", "identity")
    if h_index is None:
        if h_value is None:
            h_index = int(len(h) // 2)
        else:
            h_index = int(np.argmin(np.abs(h - float(h_value))))
    h_index = int(np.clip(h_index, 0, len(h) - 1))

    d1 = result.deriv[h_index]
    se = result.se[h_index]
    cls = result.classification[h_index]
    q = result.q_values[h_index]

    fig, ax = plt.subplots(figsize=(7, 4))
    band = q * se
    if x_transform == "log10":
        dlabel = r"$dD/d\log_{10}(x)$"
    elif x_transform == "log":
        dlabel = r"$dD/d\log(x)$"
    else:
        dlabel = r"$dD/dx$"
    ax.plot(x, d1, color="tab:orange", label=dlabel)
    ax.fill_between(
        x,
        -band,
        band,
        color="tab:gray",
        alpha=0.25,
        label=r"$\pm q \cdot SE$",
    )
    ax.axhline(0.0, color="0.2", linewidth=1.0)

    mask_zero = cls == 0
    if np.any(mask_zero):
        ax.scatter(
            x[mask_zero],
            d1[mask_zero],
            color="tab:green",
            s=10,
            alpha=0.8,
            label="Class 0",
        )

    if fmin is not None and np.isfinite(fmin):
        ax.axvline(
            fmin,
            color="tab:green",
            linestyle="--",
            linewidth=1.5,
            label=rf"$x_{{\min}}$={fmin:.2e}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("Derivative")
    ax.legend(loc="best")
    return _save_debug_fig(fig, f"find_xmin_sizer_slice_{h_index:02d}.pdf")


def plot_sizer_map(result, *, plateaus=None, fmin=None, title=None):
    """
    Plot the SiZer classification map with optional plateau overlays.
    """
    if result is None:
        return None

    import matplotlib.colors as mcolors

    x = result.x
    h = result.h_grid
    cls = result.classification
    x_transform = result.metadata.get("x_transform", "identity")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {
        -1: "#d62728",
        0: "#7f7f7f",
        1: "#1f77b4",
        9: "#ffffff",
    }
    cmap = mcolors.ListedColormap([colors[-1], colors[0], colors[1], colors[9]])
    bounds = [-1.5, -0.5, 0.5, 1.5, 9.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    mesh = ax.pcolormesh(
        x,
        h,
        cls,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    cbar = fig.colorbar(mesh, ax=ax, ticks=[-1, 0, 1, 9])
    cbar.ax.set_yticklabels(["decreasing", "flat", "increasing", "sparse"])
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("x")
    if x_transform == "log10":
        ax.set_ylabel("h (log10 x units)")
    elif x_transform == "log":
        ax.set_ylabel("h (log x units)")
    else:
        ax.set_ylabel("h")
    if title:
        ax.set_title(title)

    if plateaus:
        for i, p in enumerate(plateaus):
            ax.axvspan(
                p["x_start"],
                p["x_end"],
                color="tab:green",
                alpha=0.15,
                label="Plateau" if i == 0 else None,
            )

    if fmin is not None and np.isfinite(fmin):
        ax.axvline(
            fmin,
            color="tab:green",
            linestyle="--",
            linewidth=1.5,
            label=rf"$x_{{\min}}$={fmin:.2e}",
        )

    ax.legend(loc="best")
    return _save_debug_fig(fig, "find_xmin_sizer_map.pdf")


def plot_sizer_signal(x, y, *, plateaus=None, fmin=None, title=None):
    """
    Plot the raw signal with plateau overlays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, color="tab:blue", alpha=0.7, label="Signal")

    if plateaus:
        for i, p in enumerate(plateaus):
            ax.axvspan(
                p["x_start"],
                p["x_end"],
                color="tab:green",
                alpha=0.2,
                label="Plateau candidates" if i == 0 else None,
            )

    if fmin is not None and np.isfinite(fmin):
        ax.axvline(
            fmin,
            color="tab:green",
            linestyle="--",
            linewidth=1.5,
            label=rf"$x_{{\min}}$={fmin:.2e}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("Signal")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    return _save_debug_fig(fig, "find_xmin_signal.pdf")


def _select_plateau_from_runs(plateaus):
    """
    Select a plateau run by persistence, then length, then leftmost.
    """
    if not plateaus:
        return None

    def sort_key(p):
        length = p["index_end"] - p["index_start"]
        return (p["persistence_mean"], length, -p["index_start"])

    return sorted(plateaus, key=sort_key, reverse=True)[0]


def find_fmin(
    x,
    y,
    *,
    sizer_kwargs=None,
    plateau_kwargs=None,
    selection="left",
    debug=False,
    show_progress=False,
    return_details=False,
):
    """
    Find a plateau boundary using a SiZer map and persistence of class==0.
    """
    sizer_kwargs = {} if sizer_kwargs is None else dict(sizer_kwargs)
    plateau_kwargs = {} if plateau_kwargs is None else dict(plateau_kwargs)

    result = compute_sizer_map(x, y, show_progress=show_progress, **sizer_kwargs)
    plateaus = extract_plateaus_from_sizer(result, **plateau_kwargs)
    best = _select_plateau_from_runs(plateaus)

    fmin = np.nan
    if best is not None:
        if selection == "left":
            fmin = best["x_start"]
        elif selection == "right":
            fmin = best["x_end"]
        elif selection == "center":
            fmin = best["x_center"]
        else:
            raise ValueError("selection must be 'left', 'right', or 'center'.")

    if debug:
        scale_range = plateau_kwargs.get("scale_range")
        min_scale_fraction = plateau_kwargs.get("min_scale_fraction", 0.6)
        scale_trim = plateau_kwargs.get("scale_trim", 0.1)
        plot_sizer_signal(x, y, plateaus=plateaus, fmin=fmin, title="Signal")
        plot_sizer_map(result, plateaus=plateaus, fmin=fmin, title="SiZer map")
        plot_sizer_persistence(
            result,
            scale_range=scale_range,
            min_scale_fraction=min_scale_fraction,
            scale_trim=scale_trim,
            plateaus=plateaus,
            fmin=fmin,
            title="SiZer persistence",
        )
        plot_sizer_scale_slice(result, fmin=fmin)

    if not return_details:
        return fmin

    return fmin, {
        "sizer": result,
        "plateaus": plateaus,
        "selected": best,
    }


def _plot_xmin_debug(
    xmin_values,
    distances,
    min_distance=None,
    region_level=None,
    region_start=None,
    region_end=None,
    min_idx=None,
    dip_xmin_values=None,
    dip_distances=None,
    dip_x=None,
    dip_d1=None,
    dip_d2=None,
    smoothing=None,
):
    fig, ax1 = plt.subplots()
    ax1.plot(xmin_values, distances, label="D (coarse)")
    _plot_dip_curve(ax1, dip_xmin_values, dip_distances)

    ax2 = None
    ax3 = None
    if dip_x is not None and dip_d1 is not None:
        ax2 = ax1.twinx()
        ax2.plot(
            dip_x,
            dip_d1,
            color="tab:green",
            label=r"$dD/d\log_{10}(x_{\min})$",
            alpha=0.8,
        )
        ax2.set_ylabel(r"$dD/d\log_{10}(x_{\min})$", color="tab:green")
        ax2.tick_params(axis="y", colors="tab:green")
        ax2.spines["right"].set_color("tab:green")
        ax2.set_zorder(0)

    if dip_x is not None and dip_d2 is not None:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 50))
        ax3.plot(
            dip_x,
            dip_d2,
            color="tab:red",
            label=r"$d^2D/d\log_{10}(x_{\min})^2$",
            alpha=0.8,
        )
        ax3.set_ylabel(r"$d^2D/d\log_{10}(x_{\min})^2$", color="tab:red")
        ax3.tick_params(axis="y", colors="tab:red")
        ax3.spines["right"].set_color("tab:red")
        ax3.set_zorder(0)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$E_{\mathrm{min}}$")
    ax1.set_ylabel("KS distance")
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    if (
        region_start is not None
        and region_end is not None
        and min_distance is not None
        and region_level is not None
    ):
        _plot_region_shading(ax1, region_start, region_end, min_distance, region_level)

    if min_idx is not None:
        min_xmin = float(xmin_values[min_idx])
        ax1.axvline(
            min_xmin,
            linestyle=":",
            linewidth=1.5,
            label=rf"Min D at $x_{{\min}}$ = {min_xmin:.2e}",
            alpha=0.9,
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles = handles1
    labels = labels1
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
    if ax3 is not None:
        handles3, labels3 = ax3.get_legend_handles_labels()
        handles += handles3
        labels += labels3
    legend_ax = ax2 if ax2 is not None else ax1
    legend_ax.legend(
        handles,
        labels,
        loc="upper right",
        ncol=2,
        frameon=True,
    )

    suffix = _smoothing_suffix(smoothing)
    return _save_debug_fig(fig, f"find_xmin_ks_distance_{suffix}.pdf")


def _plot_dip_derivative_extrema_debug(
    dip_x,
    dip_D,
    dip_d1,
    dip_d2=None,
    selected_xmin=None,
    coarse_x=None,
    coarse_D=None,
    smoothing=None,
):
    if dip_x is None or dip_D is None or dip_d1 is None or dip_x.size == 0:
        return None

    fig, ax = plt.subplots()
    if coarse_x is not None and coarse_D is not None:
        ax.plot(
            coarse_x,
            coarse_D,
            label="D (coarse)",
            color="0.6",
            alpha=0.8,
        )
    _plot_dip_curve(ax, dip_x, dip_D)
    ax.set_xscale("log")
    ax.set_xlabel(r"$E_{\mathrm{min}}$")
    ax.set_ylabel("KS distance")

    if np.isfinite(dip_d1).any():
        idx_min_d1 = int(np.nanargmin(dip_d1))
        x_min_d1 = float(dip_x[idx_min_d1])
        y_min_d1 = float(np.interp(x_min_d1, dip_x, dip_D))
        ax.axvline(
            x_min_d1,
            linestyle="--",
            color="tab:green",
            label=r"Min $dD/d\log_{10}(x_{\min})$",
            alpha=0.9,
        )
        ax.plot(x_min_d1, y_min_d1, marker="o", color="tab:green")

    if dip_d2 and np.isfinite(dip_d2).any():
        idx_max_d2 = int(np.nanargmax(dip_d2))
        x_max_d2 = float(dip_x[idx_max_d2])
        y_max_d2 = float(np.interp(x_max_d2, dip_x, dip_D))
        ax.axvline(
            x_max_d2,
            linestyle=":",
            color="tab:red",
            label=r"Max $d^2D/d\log_{10}(x_{\min})^2$",
            alpha=0.9,
        )
        ax.plot(x_max_d2, y_max_d2, marker="s", color="tab:red")

    if selected_xmin is not None and np.isfinite(selected_xmin):
        y_sel = float(np.interp(selected_xmin, dip_x, dip_D))
        ax.axvline(
            selected_xmin,
            linestyle="-.",
            color="tab:purple",
            label=r"Chosen $x_{\min}$",
            alpha=0.9,
        )
        ax.plot(selected_xmin, y_sel, marker="D", color="tab:purple")

    legend = ax.legend(loc="best")
    legend.set_zorder(10)

    suffix = _smoothing_suffix(smoothing)
    return _save_debug_fig(fig, f"find_xmin_dip_deriv_extrema_{suffix}.pdf")


def _plot_fmin_debug(
    x_grid,
    D_grid,
    d1,
    se,
    plateau_mask,
    selected_run,
    fmin,
    q,
    smoothing="savgol",
):
    """
    Debug plot for find_fmin: shows D(x), derivative band, plateau mask, and selection.
    """
    if x_grid is None or D_grid is None or d1 is None or se is None:
        return None

    x_grid = np.asarray(x_grid, dtype=float)
    D_grid = np.asarray(D_grid, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    se = np.asarray(se, dtype=float)
    if plateau_mask is None:
        plateau_mask = np.zeros_like(x_grid, dtype=bool)
    else:
        plateau_mask = np.asarray(plateau_mask, dtype=bool)
        if plateau_mask.shape != x_grid.shape:
            plateau_mask = np.zeros_like(x_grid, dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax_D, ax_d1 = axes

    ax_D.plot(x_grid, D_grid, label="D (grid)", color="tab:blue", alpha=0.7)
    ax_D.set_ylabel("KS distance")

    runs = _contiguous_true_runs(plateau_mask) if plateau_mask.any() else []
    for i, (start, end) in enumerate(runs):
        ax_D.axvspan(
            x_grid[start],
            x_grid[end],
            color="0.85",
            alpha=0.35,
            label="Plateau regions" if i == 0 else None,
        )

    if selected_run is not None:
        start, end = selected_run
        ax_D.axvspan(
            x_grid[start],
            x_grid[end],
            color="tab:green",
            alpha=0.2,
            label="Selected plateau",
        )

    if np.isfinite(fmin):
        ax_D.axvline(
            fmin,
            linestyle="--",
            color="tab:green",
            alpha=0.9,
            label=rf"$x_{{\min}}$={fmin:.2e}",
        )

    ax_D.legend(loc="best")

    band = q * se
    ax_d1.plot(x_grid, d1, color="tab:orange", label=r"$dD/d\log_{10}(x_{\min})$")
    ax_d1.fill_between(
        x_grid,
        -band,
        band,
        color="tab:gray",
        alpha=0.25,
        label=r"$\pm q \cdot SE$",
    )
    if plateau_mask.any():
        ax_d1.scatter(
            x_grid[plateau_mask],
            d1[plateau_mask],
            color="tab:green",
            s=10,
            alpha=0.8,
            label="Plateau mask",
        )
    ax_d1.axhline(0.0, color="0.2", linewidth=1.0)
    ax_d1.set_ylabel(r"$dD/d\log_{10}(x_{\min})$")
    ax_d1.set_xlabel(r"$E_{\mathrm{min}}$")
    ax_d1.legend(loc="best")

    ax_d1.set_xscale("log")
    ax_D.set_xscale("log")

    suffix = _smoothing_suffix(smoothing)
    return _save_debug_fig(fig, f"find_xmin_fmin_{suffix}.pdf")


def _sylvain_cache_dir(cache_dir=None):
    if cache_dir is not None:
        return Path(cache_dir)
    return (
        Path(__file__).resolve().parent.parent / "bootstrapData" / "xmin_sylvain_cache"
    )


def _hash_array(arr):
    arr = np.ascontiguousarray(arr)
    return hashlib.sha1(arr.view(np.uint8)).hexdigest()


def _sylvain_cache_key(
    tail_hash,
    xmin,
    xmax,
    dist_name,
    B,
    fit_method,
    discrete,
    version=1,
):
    key = {
        "v": int(version),
        "tail_hash": str(tail_hash),
        "xmin": float(xmin),
        "xmax": None if xmax is None else float(xmax),
        "dist": str(dist_name),
        "B": int(B),
        "fit_method": str(fit_method),
        "discrete": bool(discrete),
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    return key, digest


def _load_sylvain_cache(path):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if "D_vals" not in data:
                return None
            return np.asarray(data["D_vals"], dtype=float)
    except Exception:
        return None


def _save_sylvain_cache(path, D_vals, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, D_vals=D_vals, meta=json.dumps(meta))
    os.replace(tmp_path, path)


def _plot_sylvain_debug(
    xmin_values,
    ks_obs,
    ks_boot_std,
    delta_obs,
    se_boot_delta,
    p_boot,
    p0,
    c,
    pass_adequacy,
    pass_near_opt,
    pass_both,
    xmin_star,
    coarse=None,
):
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    ax_ks, ax_delta, ax_p = axes

    # KS curve (with bootstrap SD error bars)
    ax_ks.plot(xmin_values, ks_obs, color="tab:blue", alpha=0.6)
    ax_ks.errorbar(
        xmin_values,
        ks_obs,
        yerr=ks_boot_std,
        fmt="o",
        ms=3,
        capsize=2,
        color="tab:blue",
        label="KS distance ±1 SD",
    )
    if coarse is not None:
        ax_ks.scatter(
            coarse["xmins"],
            coarse["KS_obs"],
            marker="s",
            s=30,
            facecolors="none",
            edgecolors="tab:blue",
            alpha=0.8,
            label="Coarse grid",
        )
    if np.isfinite(ks_obs).any():
        ks_min = float(np.nanmin(ks_obs))
        ax_ks.axhline(
            ks_min, linestyle=":", color="0.3", label=rf"$KS_{{\min}}$={ks_min:.2e}"
        )
    ax_ks.scatter(
        xmin_values[pass_both],
        ks_obs[pass_both],
        color="tab:green",
        label="Selected set",
        zorder=5,
    )
    if xmin_star is not None and np.isfinite(xmin_star):
        ax_ks.axvline(
            xmin_star,
            linestyle="--",
            color="tab:green",
            label=rf"$x_{{\min}}^*$={xmin_star:.2e}",
        )
    ax_ks.set_ylabel("KS distance")
    ax_ks.legend(loc="best")

    # Delta vs bootstrap SE
    ax_delta.plot(xmin_values, delta_obs, color="tab:orange", alpha=0.6)
    ax_delta.errorbar(
        xmin_values,
        delta_obs,
        yerr=se_boot_delta,
        fmt="o",
        ms=3,
        capsize=2,
        color="tab:orange",
        label=r"$\Delta(x_{\min}) \pm$ SE",
    )
    if coarse is not None:
        ax_delta.scatter(
            coarse["xmins"],
            coarse["Delta_obs"],
            marker="s",
            s=30,
            facecolors="none",
            edgecolors="tab:orange",
            alpha=0.8,
            label="Coarse grid",
        )
    ax_delta.plot(
        xmin_values,
        c * se_boot_delta,
        linestyle="--",
        color="tab:purple",
        label=rf"{c}*SE$_{{boot}}[\Delta]$",
    )
    ax_delta.scatter(
        xmin_values[pass_both],
        delta_obs[pass_both],
        color="tab:green",
        zorder=5,
    )
    ax_delta.set_ylabel(r"$\Delta(x_{\min})$ and SE")
    ax_delta.legend(loc="best")

    # p_boot
    ax_p.plot(xmin_values, p_boot, color="tab:red", label=r"$p_{boot}$")
    ax_p.axhline(p0, linestyle="--", color="0.3", label=rf"$p_0$={p0}")
    if coarse is not None:
        ax_p.scatter(
            coarse["xmins"],
            coarse["p_boot"],
            marker="s",
            s=30,
            facecolors="none",
            edgecolors="tab:red",
            alpha=0.8,
            label="Coarse grid",
        )
    ax_p.scatter(
        xmin_values[pass_adequacy],
        p_boot[pass_adequacy],
        color="tab:green",
        label="Adequate",
        zorder=5,
    )
    ax_p.set_ylabel(r"$p_{boot}$")
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.legend(loc="best")

    ax_p.set_xscale("log")
    ax_p.set_xlabel(r"$x_{\min}$")

    return _save_debug_fig(fig, "find_xmin_sylvain.pdf")


def find_xmin_derivative(drops, debug=False, smoothing="spline", **kwargs):
    min_xmin = min(drops)
    max_xmin = max(drops)
    nr_first_evaluation = 20
    coarse_xmin_values = np.logspace(
        np.log10(min_xmin), np.log10(max_xmin), nr_first_evaluation
    )
    fits = evaluate_xmin(drops, coarse_xmin_values, **kwargs)
    distances = np.asarray([f.D for f in fits], dtype=float)

    min_distance, region_level, region_start, region_end, min_idx = _find_region_bounds(
        distances, coarse_xmin_values, delta=0.05
    )
    assert region_start is not None

    # Do better search in region of interest (This is where we expect
    # a dip in the D value)
    dip_xmin_values = np.logspace(
        np.log10(region_start / 10),
        np.log10(region_start * 10),
        nr_first_evaluation * 10,
    )
    dip_fits = evaluate_xmin(drops, dip_xmin_values, **kwargs)
    dip_distances = np.asarray([f.D for f in dip_fits], dtype=float)
    dip_distances = _smooth_dip_distances(dip_xmin_values, dip_distances, smoothing)

    dip_mask = (
        np.isfinite(dip_xmin_values)
        & np.isfinite(dip_distances)
        & (dip_xmin_values > 0)
    )
    assert dip_mask.sum() >= 3, "Not enough data"
    x = dip_xmin_values[dip_mask]
    D = dip_distances[dip_mask]
    logx = np.log10(x)
    dip_d1 = np.gradient(D, logx)
    dip_d2 = np.gradient(dip_d1, logx)

    # We now inspect the ks distance of the fits:
    if debug:
        _plot_xmin_debug(
            coarse_xmin_values,
            distances,
            min_distance,
            region_level,
            region_start,
            region_end,
            min_idx,
            dip_xmin_values=dip_xmin_values,
            dip_distances=dip_distances,
            dip_x=x,
            dip_d1=dip_d1,
            dip_d2=dip_d2,
            smoothing=smoothing,
        )
        _plot_dip_derivative_extrema_debug(
            x,
            D,
            dip_d1,
            dip_d2=dip_d2,
            coarse_x=coarse_xmin_values,
            coarse_D=distances,
            smoothing=smoothing,
        )
    return x[np.argmax(dip_d2)]


def _sylvain_evaluate_grid(
    drops,
    xmin_values,
    p0,
    c,
    B,
    distType,
    xmax,
    parallel,
    max_workers,
    min_tail,
    max_synthetic_samples,
    use_cache,
    cache_dir,
    cache_version,
    show_progress,
):
    obs = {}
    for xmin in xmin_values:
        if xmax is None:
            tail_mask = drops >= xmin
        else:
            tail_mask = (drops >= xmin) & (drops <= xmax)
        tail_data = drops[tail_mask]
        n_tail = int(tail_data.size)
        if n_tail < 3:
            continue
        if min_tail is not None and n_tail < min_tail:
            continue
        tail_hash = _hash_array(tail_data)
        fit = Fit(
            data=drops,
            xmin=float(xmin),
            xmax=xmax,
            xmin_distribution=distType.name,
        )
        ks_obs = float(getattr(fit, "D", np.nan))
        dist = getattr(fit, distType.name, None)
        params = {}
        if dist is not None:
            param_names = list(getattr(dist, "parameter_names", []))
            params = {name: getattr(dist, name, np.nan) for name in param_names}
        obs[float(xmin)] = {
            "fit": fit,
            "KS_obs": ks_obs,
            "n_tail": n_tail,
            "params": params,
            "tail_hash": tail_hash,
        }

    if not obs:
        return None

    valid_xmins = np.array(sorted(obs.keys()), dtype=float)
    ks_obs = np.array([obs[x]["KS_obs"] for x in valid_xmins], dtype=float)
    if not np.isfinite(ks_obs).any():
        return None
    ks_min = np.nanmin(ks_obs)
    delta_obs = ks_obs - ks_min

    boot_ks = {}
    cache_root = _sylvain_cache_dir(cache_dir)
    for xmin in valid_xmins:
        fit = obs[xmin]["fit"]
        tail_hash = obs[xmin]["tail_hash"]
        cache_meta, cache_digest = _sylvain_cache_key(
            tail_hash,
            xmin,
            xmax,
            distType.name,
            B,
            fit.fit_method,
            fit.discrete,
            version=cache_version,
        )
        cache_path = cache_root / f"ks_boot_{cache_digest}.npz"
        D_vals = None
        if use_cache:
            D_vals = _load_sylvain_cache(cache_path)
            if D_vals is not None and D_vals.size != int(B):
                D_vals = None
        if D_vals is None:
            D_vals, _ = fit.bootstrap_ks_samples(
                data=drops,
                nr_sets=B,
                parallel=parallel,
                max_workers=max_workers,
                max_synthetic_samples=max_synthetic_samples,
                tqdmDesc=f"bootstrap xmin={xmin:.2e}",
                show_progress=show_progress,
                return_params=False,
            )
            D_vals = np.asarray(D_vals, dtype=float)
            if use_cache and D_vals.size == int(B):
                _save_sylvain_cache(cache_path, D_vals, cache_meta)
        else:
            if show_progress:
                print(f"Loaded cached bootstrap for xmin={xmin:.2e}")
        boot_ks[xmin] = D_vals

    lengths = [len(v) for v in boot_ks.values()]
    min_len = min(lengths) if lengths else 0
    if min_len == 0:
        return None

    ks_mat = np.vstack([boot_ks[x][:min_len] for x in valid_xmins])
    ks_min_b = np.nanmin(ks_mat, axis=0)
    ks_boot_std = np.nanstd(ks_mat, axis=1, ddof=1)

    se_boot_delta = np.full_like(delta_obs, np.nan, dtype=float)
    p_boot = np.full_like(delta_obs, np.nan, dtype=float)
    pass_adequacy = np.zeros_like(delta_obs, dtype=bool)
    pass_near_opt = np.zeros_like(delta_obs, dtype=bool)

    for i, xmin in enumerate(valid_xmins):
        delta_b = ks_mat[i] - ks_min_b
        se = np.nanstd(delta_b, ddof=1)
        se_boot_delta[i] = se
        p = np.mean(ks_mat[i] >= ks_obs[i])
        p_boot[i] = p
        pass_adequacy[i] = np.isfinite(p) and p >= p0
        pass_near_opt[i] = np.isfinite(se) and (delta_obs[i] <= c * se)

    pass_both = pass_adequacy & pass_near_opt
    selected_set = valid_xmins[pass_both].tolist()
    adequate_set = valid_xmins[pass_adequacy].tolist()
    near_opt_set = valid_xmins[pass_near_opt].tolist()
    xmin_star = min(selected_set) if selected_set else np.nan

    results = {
        "xmin_star": xmin_star,
        "selected_set": selected_set,
        "adequate_set": adequate_set,
        "near_optimal_set": near_opt_set,
        "xmins": valid_xmins,
        "KS_obs": ks_obs,
        "KS_boot_std": ks_boot_std,
        "Delta_obs": delta_obs,
        "SE_boot_Delta": se_boot_delta,
        "p_boot": p_boot,
        "params": [obs[x]["params"] for x in valid_xmins],
        "n_tail": [obs[x]["n_tail"] for x in valid_xmins],
        "xmin_at_min_ks": float(valid_xmins[int(np.nanargmin(ks_obs))]),
    }
    return results


def find_xmin_sylvain(
    drops,
    debug=False,
    xmin_values=None,
    nr_evaluation=20,
    p0=0.1,
    c=1.0,
    B=1000,
    distType: type[Distribution] = Truncated_Power_Law,
    xmax=None,
    parallel=False,
    max_workers=None,
    min_tail=None,
    max_synthetic_samples=5e6,
    use_cache=True,
    cache_dir=None,
    cache_version=1,
    return_details=False,
):
    """
    Bootstrap-calibrated KS threshold selection with a coarse-to-fine grid.

    Selects the smallest xmin that:
      (i) passes adequacy: p_boot(xmin) >= p0
      (ii) is near-optimal: Delta(xmin) <= c * SE_boot[Delta(xmin)]
    """
    B = int(B)
    if B <= 0:
        raise ValueError("B must be positive.")
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops)]
    drops = drops[drops > 0]
    if drops.size < 3:
        return (np.nan, {}) if return_details else np.nan

    min_xmin = np.nanmin(drops)
    max_xmin = np.nanmax(drops)
    if not (np.isfinite(min_xmin) and np.isfinite(max_xmin)) or min_xmin <= 0:
        return (np.nan, {}) if return_details else np.nan

    if xmin_values is None:
        if min_xmin == max_xmin:
            xmin_values = np.array([min_xmin], dtype=float)
        else:
            xmin_values = np.logspace(
                np.log10(min_xmin), np.log10(max_xmin), int(nr_evaluation)
            )
    else:
        xmin_values = np.asarray(xmin_values, dtype=float)
        xmin_values = xmin_values[np.isfinite(xmin_values) & (xmin_values > 0)]

    if xmin_values.size == 0:
        return (np.nan, {}) if return_details else np.nan

    xmin_values = np.unique(xmin_values)
    xmin_values.sort()

    # --- coarse evaluation
    coarse_results = _sylvain_evaluate_grid(
        drops,
        xmin_values,
        p0,
        c,
        B,
        distType,
        xmax,
        parallel,
        max_workers,
        min_tail,
        max_synthetic_samples,
        use_cache,
        cache_dir,
        cache_version,
        show_progress=debug,
    )
    if coarse_results is None:
        return (np.nan, {}) if return_details else np.nan

    center_xmin = coarse_results["xmin_star"]
    if not np.isfinite(center_xmin):
        center_xmin = coarse_results["xmin_at_min_ks"]

    # --- fine evaluation: one decade below/above the coarse xmin
    fine_min = max(min_xmin, center_xmin / 10.0)
    fine_max = min(max_xmin, center_xmin * 10.0)
    if fine_min <= 0 or fine_max <= 0 or fine_min >= fine_max:
        fine_xmins = xmin_values
    else:
        fine_xmins = np.logspace(np.log10(fine_min), np.log10(fine_max), 60)

    fine_xmins = np.unique(fine_xmins)
    fine_xmins.sort()

    fine_results = _sylvain_evaluate_grid(
        drops,
        fine_xmins,
        p0,
        c,
        B,
        distType,
        xmax,
        parallel,
        max_workers,
        min_tail,
        max_synthetic_samples,
        use_cache,
        cache_dir,
        cache_version,
        show_progress=debug,
    )
    results = fine_results or coarse_results
    xmin_star = results["xmin_star"]

    # compute masks for debug (using final results)
    if debug:
        coarse_payload = None
        if coarse_results is not None:
            coarse_payload = {
                "xmins": coarse_results["xmins"],
                "KS_obs": coarse_results["KS_obs"],
                "Delta_obs": coarse_results["Delta_obs"],
                "p_boot": coarse_results["p_boot"],
            }
        pass_adequacy = np.asarray(results["p_boot"]) >= p0
        pass_near_opt = np.asarray(results["Delta_obs"]) <= c * np.asarray(
            results["SE_boot_Delta"]
        )
        pass_both = pass_adequacy & pass_near_opt
        _plot_sylvain_debug(
            results["xmins"],
            results["KS_obs"],
            results["KS_boot_std"],
            results["Delta_obs"],
            results["SE_boot_Delta"],
            results["p_boot"],
            p0,
            c,
            pass_adequacy,
            pass_near_opt,
            pass_both,
            xmin_star,
            coarse=coarse_payload,
        )

    if return_details:
        results["coarse"] = coarse_results
        return xmin_star, results
    return xmin_star


def find_xmin_sizer(
    drops,
    samplesPerDecade=5,
    debug=False,
    OPTION="A",
    sizer_kwargs=None,
    plateau_kwargs=None,
    selection="left",
    show_progress=False,
    **kwargs,
):
    """
    Plateau-based xmin selection using a SiZer map on D(xmin).

    Parameters
    ----------
    drops : array-like
        Samples to fit.
    samplesPerDecade : int
        Number of candidate xmin values per decade on the coarse grid.
    debug : bool
        If True, save diagnostic figures.
    OPTION : {"A", "B"}
        "A" returns the SiZer plateau boundary (default, statistically grounded).
        "B" is deprecated and currently behaves like "A".
    sizer_kwargs : dict or None
        Keyword arguments forwarded to compute_sizer_map. Defaults to
        x_transform="log10" for KS-distance vs xmin.
    plateau_kwargs : dict or None
        Keyword arguments forwarded to extract_plateaus_from_sizer.
    selection : {"left", "right", "center"}
        Which boundary to return from the selected plateau.
    show_progress : bool
        If True, show progress bars for SiZer computation/bootstraps.
    **kwargs
        Forwarded to evaluate_xmin (e.g., distType, xmax, parallel).
    """
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops)]
    drops = drops[drops > 0]
    if drops.size < 3:
        return np.nan

    min_xmin = float(np.nanmin(drops))
    max_xmin = float(np.nanmax(drops))
    if not (np.isfinite(min_xmin) and np.isfinite(max_xmin)) or min_xmin <= 0:
        return np.nan
    if min_xmin == max_xmin:
        return min_xmin

    decades = np.log10(max_xmin / min_xmin)
    n_samples = max(5, int(np.ceil(decades * samplesPerDecade)) + 1)
    coarse_xmins = np.logspace(np.log10(min_xmin), np.log10(max_xmin), n_samples)
    coarse_fits = evaluate_xmin(drops, coarse_xmins, **kwargs)
    coarse_distances = np.asarray([f.D for f in coarse_fits], dtype=float)

    sizer_kwargs = {} if sizer_kwargs is None else dict(sizer_kwargs)
    plateau_kwargs = {} if plateau_kwargs is None else dict(plateau_kwargs)
    sizer_kwargs.setdefault("x_transform", "log10")
    fmin = find_fmin(
        coarse_xmins,
        coarse_distances,
        sizer_kwargs=sizer_kwargs,
        plateau_kwargs=plateau_kwargs,
        selection=selection,
        debug=debug,
        show_progress=show_progress,
        return_details=False,
    )

    option = str(OPTION).strip().upper()
    if option == "A":
        return fmin
    if option != "B":
        raise ValueError("OPTION must be 'A' or 'B'.")

    # Deprecated: Option B used to be a second-derivative heuristic.
    # It now returns the same SiZer-based plateau boundary as OPTION "A".
    return fmin


def find_xmin_dks_from_results(xmins, distances, valid_fits=None):
    """Select the steepest KS decrease from an existing candidate grid."""
    xmins = np.asarray(xmins, dtype=float)
    distances = np.asarray(distances, dtype=float)
    mask = np.isfinite(xmins) & np.isfinite(distances) & (xmins > 0)
    if valid_fits is not None:
        mask &= np.asarray(valid_fits, dtype=bool)
    if mask.sum() < 2:
        return np.nan
    xmins = xmins[mask]
    distances = distances[mask]
    order = np.argsort(xmins)
    xmins = xmins[order]
    derivative = np.gradient(distances[order], np.log10(xmins))
    return float(xmins[int(np.nanargmin(derivative))])


def _log_xmin_candidates(drops, samples_per_decade, tail_decades=1.0):
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > 0)]
    if drops.size < 3:
        raise ValueError("Need at least three finite positive drops.")
    if samples_per_decade <= 0:
        raise ValueError("samples_per_decade must be positive.")
    if tail_decades <= 0:
        raise ValueError("tail_decades must be positive.")
    candidate_max = float(drops.max() / 10.0**tail_decades)
    if candidate_max <= drops.min():
        raise ValueError(
            f"Data span fewer than {tail_decades:g} decade(s); no xmin candidates."
        )
    decades = np.log10(candidate_max / drops.min())
    n_samples = max(20, int(np.ceil(decades * samples_per_decade)))
    xmins = np.logspace(np.log10(drops.min()), np.log10(candidate_max), n_samples)
    return drops, xmins


def _xmin_fit_results(fits, xmins):
    distances = np.asarray([fit.D for fit in fits], dtype=float)
    param_vals = [
        [
            getattr(getattr(fit, fit.xmin_distribution.name), parameter, np.nan)
            for parameter in fit.xmin_distribution.parameter_names
        ]
        for fit in fits
    ]
    return {
        "distances": distances,
        "param_vals": param_vals,
        "xmins": np.asarray(xmins, dtype=float),
    }


def find_xmin_dks(drops, samples_per_decade=30, tail_decades=1.0, **kwargs):
    """Select the steepest decrease of KS distance versus log10(xmin)."""
    drops, xmins = _log_xmin_candidates(
        drops, samples_per_decade, tail_decades=tail_decades
    )
    fits = evaluate_xmin(drops, xmins, **kwargs)
    results = _xmin_fit_results(fits, xmins)
    return find_xmin_dks_from_results(xmins, results["distances"]), results


def find_xmin_global_min(
    drops,
    candidate_stride=10,
    tail_decades=1.0,
    **kwargs,
):
    """Select the global KS minimum from strided observed sample values."""
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > 0)]
    if drops.size < 3:
        raise ValueError("Need at least three finite positive drops.")
    if int(candidate_stride) != candidate_stride or candidate_stride < 1:
        raise ValueError("candidate_stride must be a positive integer.")
    if tail_decades <= 0:
        raise ValueError("tail_decades must be positive.")
    candidate_max = float(drops.max() / 10.0**tail_decades)
    candidates = np.unique(drops[drops <= candidate_max])
    candidates = candidates[:: int(candidate_stride)]
    if candidates.size < 2:
        raise ValueError("Fewer than two sampled global-min xmin candidates.")
    distances, param_vals, valid = evaluate_xmin_distances(
        drops, candidates, **kwargs
    )
    results = {
        "distances": distances,
        "param_vals": param_vals,
        "valid_fits": valid,
        "xmins": candidates,
    }
    finite = np.isfinite(distances)
    if not finite.any():
        raise RuntimeError("No finite KS distances in the global-min search.")
    valid_indices = np.flatnonzero(finite)
    best = valid_indices[int(np.argmin(distances[finite]))]
    return float(candidates[best]), results


def find_xmin_ks(drops, distType=Truncated_Power_Law, xmax=None, **kwargs):
    """Use the upstream minimum-KS selector."""
    fit = Fit(
        drops,
        xmax=xmax,
        xmin_distribution=distType.name,
        fast_xmin=False,
        verbose=0,
    )
    return float(fit.xmin)


def find_xmin_dip(
    drops,
    distType=Truncated_Power_Law,
    samples_per_decade=30,
    parallel=False,
    **kwargs,
):
    """Use the post-KS-drop knee selector implemented by the custom Fit."""
    fit = Fit(
        drops,
        xmin_distribution=distType.name,
        fast_xmin=True,
        xmin_samples_per_decade=samples_per_decade,
        parallel_xmin=parallel,
        verbose=0,
    )
    return float(fit.xmin)


def find_xmin_max_p(
    drops,
    distType=Truncated_Power_Law,
    nr_evaluation=20,
    confidence=0.1,
    parallel=False,
    **kwargs,
):
    """Select the candidate with the largest fixed-xmin bootstrap p-value."""
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > 0)]
    if drops.size < 3:
        return np.nan
    xmins = np.logspace(
        np.log10(drops.min()),
        np.log10(drops.max()),
        int(nr_evaluation),
    )
    fits = evaluate_xmin(drops, xmins, distType=distType, parallel=False)
    for fit in fits:
        fit.evaluate_fit(
            drops,
            confidence=confidence,
            parallel=parallel,
        )
    p_values = np.asarray([fit.p for fit in fits], dtype=float)
    if not np.isfinite(p_values).any():
        return np.nan
    return float(xmins[int(np.nanargmax(p_values))])


def _fine_xmin_candidates(drops, min_tail_count):
    """Return every observed xmin that retains the requested tail count."""
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > 0)]
    if drops.size < int(min_tail_count):
        raise ValueError(
            f"Need at least {int(min_tail_count)} positive drops for the fine "
            f"xmin search; got {drops.size}."
        )

    sorted_drops = np.sort(drops)
    candidate_hi = float(sorted_drops[-int(min_tail_count)])
    candidates = np.unique(sorted_drops[sorted_drops <= candidate_hi])
    if candidates.size < 2:
        raise ValueError(
            "Need at least two distinct observed xmin candidates for the fine "
            "local-minimum search."
        )
    return candidates


def _nearest_sorted_xmin_index(xmins, target):
    """Find the observed xmin nearest to ``target`` in log-space."""
    xmins = np.asarray(xmins, dtype=float)
    if xmins.ndim != 1 or xmins.size == 0:
        raise ValueError("xmins must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(xmins) & (xmins > 0)):
        raise ValueError("xmins must contain only finite positive values.")
    if np.any(np.diff(xmins) <= 0):
        raise ValueError("xmins must be strictly increasing.")
    target = float(target)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target must be finite and positive.")

    insertion = int(np.searchsorted(xmins, target, side="left"))
    candidate_indices = np.unique(
        np.clip([insertion - 1, insertion], 0, xmins.size - 1)
    )
    log_target = np.log10(target)
    return min(
        (int(index) for index in candidate_indices),
        key=lambda index: (abs(np.log10(xmins[index]) - log_target), index),
    )


def _evaluate_fine_xmin_indices(
    drops,
    fine_xmins,
    indices,
    distance_cache,
    *,
    min_tail_count=3,
    parameter_cache=None,
    initial_params=None,
    **fit_kwargs,
):
    """Evaluate selected observed-xmin indices and update the shared caches."""
    fine_xmins = np.asarray(fine_xmins, dtype=float)
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional.")
    if np.any(indices < 0) or np.any(indices >= fine_xmins.size):
        raise IndexError("indices contain a value outside fine_xmins.")
    if parameter_cache is None:
        parameter_cache = {}

    missing_indices = []
    missing_xmins = []
    for index in indices:
        xmin = float(fine_xmins[index])
        key = float(np.float64(xmin))
        if key in distance_cache:
            continue
        if np.count_nonzero(drops >= xmin) < int(min_tail_count):
            distance_cache[key] = np.inf
        else:
            missing_indices.append(int(index))
            missing_xmins.append(xmin)

    if missing_xmins:
        distances, _, params = evaluate_xmin_distances(
            drops,
            missing_xmins,
            initial_params=initial_params,
            **fit_kwargs,
        )
        for index, distance, params_for_xmin in zip(
            missing_indices,
            distances,
            params,
        ):
            key = float(np.float64(fine_xmins[index]))
            distance_cache[key] = (
                float(distance) if np.isfinite(distance) else np.inf
            )
            parameter_cache[key] = params_for_xmin

    return np.asarray(
        [distance_cache[float(np.float64(fine_xmins[index]))] for index in indices],
        dtype=float,
    )


def _simple_drop_local_search(
    drops,
    fine_xmins,
    start_index,
    distance_cache,
    *,
    min_tail_count=3,
    parameter_cache=None,
    search_bounds=None,
    **fit_kwargs,
):
    """Minimize KS distance by moving to direct neighbors in ``fine_xmins``.

    ``fine_xmins`` is the sorted unique set of observed drop values. Each
    iteration evaluates only the current candidate and its direct array
    neighbors. The old continuous log-space pattern search is deliberately
    not used here: an xmin is refined at the full fidelity of the observed
    data, one candidate index at a time.
    """
    drops = np.asarray(drops, dtype=float)
    fine_xmins = np.asarray(fine_xmins, dtype=float)
    if fine_xmins.ndim != 1 or fine_xmins.size < 2:
        raise ValueError("fine_xmins must contain at least two candidates.")
    if not np.all(np.isfinite(fine_xmins) & (fine_xmins > 0)):
        raise ValueError("fine_xmins must contain only finite positive values.")
    if np.any(np.diff(fine_xmins) <= 0):
        raise ValueError("fine_xmins must be strictly increasing.")
    start_index = int(start_index)
    if start_index < 0 or start_index >= fine_xmins.size:
        raise IndexError("start_index is outside fine_xmins.")
    if search_bounds is None:
        lower_index, upper_index = 0, fine_xmins.size - 1
    else:
        if len(search_bounds) != 2:
            raise ValueError("search_bounds must contain two candidate indices.")
        lower_index, upper_index = (int(value) for value in search_bounds)
        if not 0 <= lower_index <= upper_index < fine_xmins.size:
            raise ValueError("search_bounds must lie within fine_xmins.")
        if not lower_index <= start_index <= upper_index:
            raise ValueError("start_index must lie within search_bounds.")
    if parameter_cache is None:
        parameter_cache = {}

    def evaluate_many(indices, initial_params=None):
        return _evaluate_fine_xmin_indices(
            drops,
            fine_xmins,
            indices,
            distance_cache,
            min_tail_count=min_tail_count,
            parameter_cache=parameter_cache,
            initial_params=initial_params,
            **fit_kwargs,
        )

    current_index = start_index
    current_distance = float(evaluate_many([current_index])[0])
    current_params = parameter_cache.get(
        float(np.float64(fine_xmins[current_index]))
    )
    iterations = 0
    while True:
        neighbor_indices = np.arange(
            max(lower_index, current_index - 1),
            min(upper_index + 1, current_index + 2),
            dtype=int,
        )
        trial_distances = evaluate_many(neighbor_indices, current_params)
        best_position = int(np.argmin(trial_distances))
        best_index = int(neighbor_indices[best_position])
        best_distance = float(trial_distances[best_position])
        if best_distance < current_distance:
            current_index = best_index
            current_distance = best_distance
            current_params = parameter_cache.get(
                float(np.float64(fine_xmins[current_index]))
            )
            iterations += 1
            continue
        break

    return {
        "xmin": float(fine_xmins[current_index]),
        "distance": current_distance,
        "start_xmin": float(fine_xmins[start_index]),
        "start_candidate_index": int(start_index),
        "final_candidate_index": int(current_index),
        "search_bounds": (int(lower_index), int(upper_index)),
        "iterations": iterations,
    }


SIMPLE_DROP_START_LABELS = ("left", "middle", "right")


def summarize_simple_drop_starts(local_minima, *, rtol=1e-6):
    """Summarize agreement between the three simpleDrop local searches."""
    local_minima = list(local_minima)
    if len(local_minima) != len(SIMPLE_DROP_START_LABELS):
        raise ValueError(
            "simpleDrop must provide exactly three local-search results."
        )
    if rtol < 0:
        raise ValueError("rtol must be non-negative.")

    for index, (label, result) in enumerate(
        zip(SIMPLE_DROP_START_LABELS, local_minima)
    ):
        result["start_label"] = label
        result["start_label_index"] = index

    finite_results = [
        result for result in local_minima if np.isfinite(result["distance"])
    ]
    if not finite_results:
        raise RuntimeError("Cannot summarize three failed simpleDrop searches.")

    unique_xmins = []
    for result in finite_results:
        if not any(
            np.isclose(result["xmin"], other, rtol=rtol, atol=0.0)
            for other in unique_xmins
        ):
            unique_xmins.append(float(result["xmin"]))

    selected_index = min(
        range(len(local_minima)),
        key=lambda index: (local_minima[index]["distance"], local_minima[index]["xmin"]),
    )
    middle = local_minima[1]
    lowest_distance = min(result["distance"] for result in finite_results)
    middle_is_lowest = np.isfinite(middle["distance"]) and np.isclose(
        middle["distance"],
        lowest_distance,
        rtol=rtol,
        atol=0.0,
    )

    def starts_differ(first, second):
        return not np.isclose(
            local_minima[first]["xmin"],
            local_minima[second]["xmin"],
            rtol=rtol,
            atol=0.0,
        )

    return {
        "unique_local_minimum_count": len(unique_xmins),
        "finds_different_local_minima": len(unique_xmins) > 1,
        "middle_search_is_lowest": bool(middle_is_lowest),
        "selected_start": SIMPLE_DROP_START_LABELS[selected_index],
        "selected_start_index": int(selected_index),
        "pairwise_different": {
            "left_middle": starts_differ(0, 1),
            "middle_right": starts_differ(1, 2),
            "left_right": starts_differ(0, 2),
        },
        "unique_xmins": unique_xmins,
        "comparison_rtol": float(rtol),
    }


def find_xmin_simple_drop_from_results(
    drops,
    xmins,
    distances,
    valid_fits=None,
    *,
    min_tail_count=100,
    distance_cache=None,
    parameter_cache=None,
    max_xmin=None,
    refine=True,
    **fit_kwargs,
):
    """Select the largest raw eligible KS drop, then descend to a local minimum.

    The coarse scan is kept unsmoothed. Only adjacent coarse candidates that
    both retain at least ``min_tail_count`` drops are considered. The largest
    decrease in KS distance between those adjacent candidates defines the
    fine-search interval. Every observed xmin in that interval is evaluated,
    then the best candidate is used as the starting point for a direct-neighbor
    search over the complete observed candidate array.  The latter continuation
    is deliberately independent of the coarse xmin grid.

    Coarse candidates with fewer than ``min_tail_count`` retained drops remain
    available in the supplied scan for plotting, but cannot define or win the
    simpleDrop selection.
    """
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > 0)]
    xmins = np.asarray(xmins, dtype=float)
    distances = np.asarray(distances, dtype=float)
    if xmins.ndim != 1 or distances.ndim != 1:
        raise ValueError("xmins and distances must be one-dimensional.")
    if xmins.shape != distances.shape:
        raise ValueError("xmins and distances must have the same shape.")
    if not np.all(np.isfinite(xmins) & (xmins > 0)):
        raise ValueError("xmins must contain only finite positive values.")
    if int(min_tail_count) != min_tail_count or min_tail_count < 3:
        raise ValueError("min_tail_count must be an integer of at least three.")
    min_tail_count = int(min_tail_count)
    if max_xmin is not None:
        max_xmin = float(max_xmin)
        if not np.isfinite(max_xmin) or max_xmin <= 0:
            raise ValueError("max_xmin must be finite and positive.")

    finite_mask = np.isfinite(xmins) & np.isfinite(distances) & (xmins > 0)
    fit_validity = (
        np.ones(xmins.size, dtype=bool)
        if valid_fits is None
        else np.asarray(valid_fits, dtype=bool)
    )
    if fit_validity.shape != xmins.shape:
        raise ValueError("valid_fits must have the same shape as xmins.")
    tail_counts = np.asarray(
        [np.count_nonzero(drops >= xmin) for xmin in xmins],
        dtype=int,
    )
    search_mask = finite_mask & (tail_counts >= min_tail_count)
    if max_xmin is not None:
        search_mask &= xmins <= max_xmin
    if search_mask.sum() < 2:
        raise RuntimeError(
            "Need at least two finite initial KS measurements retaining the "
            "requested tail count."
        )

    order = np.argsort(xmins)
    xmins = xmins[order]
    if np.any(np.diff(xmins) <= 0):
        raise ValueError("xmins must be distinct.")
    distances = distances[order]
    search_mask = search_mask[order]
    fit_validity = fit_validity[order]
    tail_counts = tail_counts[order]

    eligible_adjacent = search_mask[:-1] & search_mask[1:]
    adjacent_drops = np.full(xmins.size - 1, np.nan, dtype=float)
    adjacent_drops[eligible_adjacent] = (
        distances[:-1][eligible_adjacent] - distances[1:][eligible_adjacent]
    )
    if not np.any(np.isfinite(adjacent_drops)):
        raise RuntimeError(
            "No adjacent coarse KS measurements retain the requested tail count."
        )
    largest_adjacent_drop_index = int(np.nanargmax(adjacent_drops))
    largest_adjacent_drop = float(adjacent_drops[largest_adjacent_drop_index])
    if largest_adjacent_drop <= 0:
        warnings.warn(
            "The eligible KS scan contains no decrease; simpleDrop will use "
            "the adjacent pair with the smallest increase."
        )

    interval_coarse_indices = np.asarray(
        [largest_adjacent_drop_index, largest_adjacent_drop_index + 1],
        dtype=int,
    )
    left = float(xmins[interval_coarse_indices[0]])
    right = float(xmins[interval_coarse_indices[1]])
    if distance_cache is None:
        distance_cache = {}
    if parameter_cache is None:
        parameter_cache = {}
    distance_cache.update(
        {
            float(np.float64(xmin)): float(distance)
            for xmin, distance, keep in zip(xmins, distances, search_mask)
            if keep
        }
    )

    if refine:
        fine_xmins = _fine_xmin_candidates(drops, min_tail_count)
        if max_xmin is not None:
            fine_xmins = fine_xmins[fine_xmins <= max_xmin]
            if fine_xmins.size < 2:
                raise RuntimeError(
                    "Fewer than two observed xmin candidates satisfy max_xmin."
                )
        region_indices = np.flatnonzero(
            (fine_xmins >= left) & (fine_xmins <= right)
        )
        if region_indices.size == 0:
            region_indices = np.unique(
                [
                    _nearest_sorted_xmin_index(fine_xmins, left),
                    _nearest_sorted_xmin_index(fine_xmins, right),
                ]
            )
        region_distances = _evaluate_fine_xmin_indices(
            drops,
            fine_xmins,
            region_indices,
            distance_cache,
            min_tail_count=min_tail_count,
            parameter_cache=parameter_cache,
            **fit_kwargs,
        )
    else:
        fine_xmins = xmins
        region_indices = interval_coarse_indices
        region_distances = distances[region_indices]
    finite_region = np.isfinite(region_distances)
    if not np.any(finite_region):
        raise RuntimeError("No finite KS distances were found in the simpleDrop region.")
    region_finite_positions = np.flatnonzero(finite_region)
    region_best_position = int(
        region_finite_positions[
            np.argmin(region_distances[region_finite_positions])
        ]
    )
    region_best_index = int(region_indices[region_best_position])
    region_best_xmin = float(fine_xmins[region_best_index])
    local_minimum = _simple_drop_local_search(
        drops,
        fine_xmins,
        region_best_index,
        distance_cache,
        min_tail_count=min_tail_count,
        parameter_cache=parameter_cache,
        search_bounds=None,
        **fit_kwargs,
    )
    if not np.isfinite(local_minimum["distance"]):
        raise RuntimeError("The simpleDrop fine local search failed.")

    evaluated = sorted(
        (
            float(xmin),
            float(distance),
        )
        for xmin, distance in distance_cache.items()
        if np.isfinite(distance)
    )
    details = {
        "distances": distances,
        "xmins": xmins,
        "valid_fits": fit_validity,
        "search_mask": search_mask,
        "tail_counts": tail_counts,
        "largest_drop_interval": (left, right),
        "largest_distance_drop": largest_adjacent_drop,
        "adjacent_drop_values": adjacent_drops,
        "eligible_adjacent_mask": eligible_adjacent,
        "largest_adjacent_drop_index": int(largest_adjacent_drop_index),
        "interval_coarse_indices": interval_coarse_indices,
        "interval_coarse_xmins": xmins[interval_coarse_indices],
        "interval_coarse_distances": distances[interval_coarse_indices],
        "region_xmins": fine_xmins[region_indices],
        "region_distances": region_distances,
        "region_candidate_indices": region_indices,
        "region_best_xmin": region_best_xmin,
        "region_best_distance": float(region_distances[region_best_position]),
        "local_minimum": local_minimum,
        "selected_distance": float(local_minimum["distance"]),
        "initial_measurement_count": int(xmins.size),
        "fine_candidate_source": (
            "sorted_unique_observed_drops" if refine else "coarse_scan"
        ),
        "fine_candidate_count": int(fine_xmins.size),
        "fine_step": (
            "selected_interval_then_direct_neighbor_descent_over_all_observed_xmins"
            if refine
            else "none"
        ),
        "fine_candidate_min": float(fine_xmins[0]),
        "fine_candidate_max": float(fine_xmins[-1]),
        "fine_search_bounds": local_minimum["search_bounds"],
        "local_search_scope": (
            "all_observed_xmins" if refine else "coarse_scan"
        ),
        "local_search_start_xmin": region_best_xmin,
        "evaluated_xmins": [xmin for xmin, _ in evaluated],
        "evaluated_distances": [distance for _, distance in evaluated],
        "eligible_initial_measurement_count": int(search_mask.sum()),
        "selection_min_tail_count": min_tail_count,
        "selection_max_xmin": max_xmin,
        "refinement": "refined" if refine else "coarse_scan",
    }
    return float(local_minimum["xmin"]), details


def find_xmin_refined_global_min_from_results(
    drops,
    xmins,
    distances,
    valid_fits=None,
    *,
    min_tail_count=3,
    distance_cache=None,
    parameter_cache=None,
    max_xmin=None,
    **fit_kwargs,
):
    """Refine every rough local minimum in a coarse KS scan.

    The supplied coarse scan is first searched over its full valid range for
    rough local minima, including minima at either boundary. A fine neighbor
    search starts from each rough minimum after mapping it to the nearest
    observed xmin. The fine search uses sorted unique observed drops and moves
    only to direct array neighbors. Only after all local searches have
    finished is the smallest evaluated KS distance chosen.

    Fine refinement continues until neither direct neighbor improves the KS
    distance.
    """
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > 0)]
    xmins = np.asarray(xmins, dtype=float)
    distances = np.asarray(distances, dtype=float)
    if xmins.shape != distances.shape:
        raise ValueError("xmins and distances must have the same shape.")
    if max_xmin is not None:
        max_xmin = float(max_xmin)
        if not np.isfinite(max_xmin) or max_xmin <= 0:
            raise ValueError("max_xmin must be finite and positive.")

    fit_validity = (
        np.ones(xmins.size, dtype=bool)
        if valid_fits is None
        else np.asarray(valid_fits, dtype=bool)
    )
    if fit_validity.shape != xmins.shape:
        raise ValueError("valid_fits must have the same shape as xmins.")

    order = np.argsort(xmins)
    xmins = xmins[order]
    distances = distances[order]
    fit_validity = fit_validity[order]
    tail_counts = np.asarray(
        [np.count_nonzero(drops >= xmin) for xmin in xmins],
        dtype=int,
    )
    search_mask = (
        np.isfinite(xmins)
        & (xmins > 0)
        & np.isfinite(distances)
        & (tail_counts >= int(min_tail_count))
    )
    if max_xmin is not None:
        search_mask &= xmins <= max_xmin
    valid_indices = np.flatnonzero(search_mask)
    if valid_indices.size < 2:
        raise RuntimeError("Need at least two valid initial KS measurements.")

    rough_indices = []
    for position, index in enumerate(valid_indices):
        distance = float(distances[index])
        left_distance = (
            float(distances[valid_indices[position - 1]])
            if position > 0
            else np.inf
        )
        right_distance = (
            float(distances[valid_indices[position + 1]])
            if position + 1 < valid_indices.size
            else np.inf
        )
        if (
            distance <= left_distance
            and distance <= right_distance
            and (distance < left_distance or distance < right_distance)
        ):
            rough_indices.append(int(index))

    if not rough_indices:
        rough_indices = [
            int(valid_indices[np.argmin(distances[valid_indices])])
        ]

    fine_xmins = _fine_xmin_candidates(drops, min_tail_count)
    if max_xmin is not None:
        fine_xmins = fine_xmins[fine_xmins <= max_xmin]
        if fine_xmins.size < 2:
            raise RuntimeError(
                "Fewer than two observed xmin candidates satisfy max_xmin."
            )
    if distance_cache is None:
        distance_cache = {}
    distance_cache.update(
        {
            float(np.float64(xmin)): float(distance)
            for xmin, distance, keep in zip(xmins, distances, search_mask)
            if keep
        }
    )

    local_minima = []
    for index in rough_indices:
        start_index = _nearest_sorted_xmin_index(fine_xmins, xmins[index])
        position = int(np.flatnonzero(valid_indices == index)[0])
        lower_target = (
            xmins[valid_indices[position - 1]]
            if position > 0
            else fine_xmins[0]
        )
        upper_target = (
            xmins[valid_indices[position + 1]]
            if position + 1 < valid_indices.size
            else fine_xmins[-1]
        )
        search_bounds = (
            _nearest_sorted_xmin_index(fine_xmins, lower_target),
            _nearest_sorted_xmin_index(fine_xmins, upper_target),
        )
        local_minima.append(
            _simple_drop_local_search(
                drops,
                fine_xmins,
                start_index,
                distance_cache,
                min_tail_count=min_tail_count,
                parameter_cache=parameter_cache,
                search_bounds=search_bounds,
                **fit_kwargs,
            )
        )

    evaluated = sorted(
        (
            float(xmin),
            float(distance),
        )
        for xmin, distance in distance_cache.items()
        if np.isfinite(distance)
    )
    if not evaluated:
        raise RuntimeError("No finite KS distances were evaluated.")
    selected_xmin, selected_distance = min(
        evaluated,
        key=lambda item: (item[1], item[0]),
    )
    rough_local_minima = [
        {
            "index": int(index),
            "xmin": float(xmins[index]),
            "distance": float(distances[index]),
        }
        for index in rough_indices
    ]
    details = {
        "distances": distances,
        "xmins": xmins,
        "valid_fits": fit_validity,
        "search_mask": search_mask,
        "tail_counts": tail_counts,
        "rough_local_minima": rough_local_minima,
        "local_minima": local_minima,
        "selected_distance": float(selected_distance),
        "initial_measurement_count": int(xmins.size),
        "fine_candidate_source": "sorted_unique_observed_drops",
        "fine_candidate_count": int(fine_xmins.size),
        "fine_step": "direct_neighbor_index",
        "fine_candidate_min": float(fine_xmins[0]),
        "fine_candidate_max": float(fine_xmins[-1]),
        "fine_search_bounds": [
            result["search_bounds"] for result in local_minima
        ],
        "evaluated_xmins": [xmin for xmin, _ in evaluated],
        "evaluated_distances": [distance for _, distance in evaluated],
        "selection_max_xmin": max_xmin,
    }
    return float(selected_xmin), details


def select_global_min_from_search_details(*search_details):
    """Choose the smallest KS result after all supplied searches have run."""
    all_evaluations = {}
    for details in search_details:
        evaluated_xmins = details.get("evaluated_xmins", ())
        evaluated_distances = details.get("evaluated_distances", ())
        if len(evaluated_xmins) != len(evaluated_distances):
            raise ValueError(
                "Each search must provide equally sized evaluated_xmins and "
                "evaluated_distances."
            )
        for xmin, distance in zip(evaluated_xmins, evaluated_distances):
            xmin = float(xmin)
            distance = float(distance)
            if np.isfinite(xmin) and xmin > 0 and np.isfinite(distance):
                all_evaluations[xmin] = min(
                    distance,
                    all_evaluations.get(xmin, np.inf),
                )
    if not all_evaluations:
        raise RuntimeError("No finite KS evaluations are available.")
    selected_xmin, selected_distance = min(
        all_evaluations.items(),
        key=lambda item: (item[1], item[0]),
    )
    return float(selected_xmin), float(selected_distance), all_evaluations


def analyze_xmin(
    drops,
    *,
    nr_initial=100,
    min_tail_count=100,
    distType=Truncated_Power_Law,
    parallel=False,
    max_xmin=None,
    refine=True,
    progress=False,
    progress_label="xmin",
):
    """Run the canonical raw-adjacent simpleDrop and global-min analysis."""
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > 0)]
    if int(nr_initial) != nr_initial or nr_initial < 2:
        raise ValueError("nr_initial must be an integer of at least two.")
    if int(min_tail_count) != min_tail_count or min_tail_count < 3:
        raise ValueError("min_tail_count must be an integer of at least three.")
    if drops.size < min_tail_count:
        raise ValueError(
            f"Need at least {min_tail_count} finite positive drops; got {drops.size}."
        )
    if max_xmin is not None:
        max_xmin = float(max_xmin)
        if not np.isfinite(max_xmin) or max_xmin <= 0:
            raise ValueError("max_xmin must be finite and positive.")

    sorted_drops = np.sort(drops)
    candidate_lo = float(sorted_drops[0])
    tail_valid_hi = float(sorted_drops[-int(min_tail_count)])
    # Scan far enough to display the low-count tail, but never evaluate a
    # candidate with fewer than three observations in its fitted tail.
    display_candidate_hi = float(sorted_drops[-3])
    if not display_candidate_hi > candidate_lo:
        raise ValueError(
            "Could not form an xmin interval with at least three distinct-tail "
            "observations."
        )

    xmins = np.geomspace(candidate_lo, display_candidate_hi, int(nr_initial))
    distances, param_vals, valid_fits = evaluate_xmin_distances(
        drops,
        xmins,
        distType=distType,
        parallel=parallel,
        progress=progress,
        progress_label=progress_label,
    )
    distances = np.asarray(distances, dtype=float)
    valid_fits = np.asarray(valid_fits, dtype=bool)
    alphas = np.asarray(
        [
            values[0] if values is not None and len(values) else np.nan
            for values in param_vals
        ],
        dtype=float,
    )
    fit_search_kwargs = dict(
        min_tail_count=int(min_tail_count),
        distType=distType,
        parallel=parallel,
        progress=progress,
        progress_label=progress_label,
    )
    distance_cache = {}
    parameter_cache = {}
    simple_drop_xmin, simple_drop_details = find_xmin_simple_drop_from_results(
        drops,
        xmins,
        distances,
        valid_fits,
        distance_cache=distance_cache,
        parameter_cache=parameter_cache,
        max_xmin=max_xmin,
        refine=refine,
        **fit_search_kwargs,
    )
    if refine:
        _, global_search_details = find_xmin_refined_global_min_from_results(
            drops,
            xmins,
            distances,
            valid_fits,
            distance_cache=distance_cache,
            parameter_cache=parameter_cache,
            max_xmin=max_xmin,
            **fit_search_kwargs,
        )
    else:
        tail_counts = np.asarray(
            [np.count_nonzero(drops >= xmin) for xmin in xmins], dtype=int
        )
        global_valid = (
            np.isfinite(distances)
            & (tail_counts >= int(min_tail_count))
        )
        if max_xmin is not None:
            global_valid &= xmins <= max_xmin
        global_search_details = {
            "xmins": xmins,
            "distances": distances,
            "evaluated_xmins": xmins[global_valid].tolist(),
            "evaluated_distances": distances[global_valid].tolist(),
            "refinement": "coarse_scan",
            "selection_max_xmin": max_xmin,
        }
    global_xmin, global_distance, all_evaluations = (
        select_global_min_from_search_details(
            simple_drop_details,
            global_search_details,
        )
    )
    return {
        "method": "simpleDrop",
        "selection_mode": "largest_raw_adjacent_drop",
        "xmins": xmins,
        "distances": distances,
        "param_vals": param_vals,
        "alphas": alphas,
        "sigmas": np.full(xmins.shape, np.nan),
        "valid_fits": valid_fits,
        "tail_counts": np.asarray(
            [np.count_nonzero(drops >= xmin) for xmin in xmins],
            dtype=int,
        ),
        "simple_drop_xmin": float(simple_drop_xmin),
        "simple_drop_distance": float(simple_drop_details["selected_distance"]),
        "simple_drop_details": simple_drop_details,
        "global_min_xmin": global_xmin,
        "global_min_distance": global_distance,
        "global_search_details": global_search_details,
        "all_evaluations": all_evaluations,
        "nr_initial": int(nr_initial),
        "min_tail_count": int(min_tail_count),
        "display_candidate_min_tail_count": 3,
        "data_max": float(sorted_drops[-1]),
        "tail_valid_max": tail_valid_hi,
        "xmin_search_max": min(
            tail_valid_hi,
            float(max_xmin) if max_xmin is not None else tail_valid_hi,
        ),
        "max_xmin": max_xmin,
        "refinement": "refined" if refine else "coarse_scan",
    }


def xmin_global_differs(analysis, *, rtol=1e-6):
    return not np.isclose(
        analysis["simple_drop_xmin"],
        analysis["global_min_xmin"],
        rtol=rtol,
        atol=0.0,
    )


def annotate_xmin_choices(ax, analysis, *, add_labels=True, xmin_scale=1.0):
    """Mark the global minimum, and simpleDrop only when it differs."""
    xmin_scale = float(xmin_scale)
    if not np.isfinite(xmin_scale) or xmin_scale <= 0.0:
        raise ValueError(f"xmin_scale must be positive and finite, got {xmin_scale!r}")
    axis_label = analysis.get("xmin_axis_label", r"$x_{\min}$")
    simple_xmin = analysis["simple_drop_xmin"] / xmin_scale
    if xmin_global_differs(analysis):
        ax.axvline(
            simple_xmin,
            color="tab:blue",
            linestyle="--",
            linewidth=1.0,
            label=(
                rf"simpleDrop: {axis_label}={simple_xmin:.1e}"
                if add_labels
                else "_nolegend_"
            ),
        )
        global_xmin = analysis["global_min_xmin"] / xmin_scale
        ax.axvline(
            global_xmin,
            color="0.35",
            linestyle=":",
            linewidth=1.0,
            label=(
                rf"Global min.: {axis_label}={global_xmin:.1e}"
                if add_labels
                else "_nolegend_"
            ),
        )
    else:
        global_xmin = analysis["global_min_xmin"] / xmin_scale
        ax.axvline(
            global_xmin,
            color="0.25",
            linestyle=":",
            linewidth=1.0,
            label=(
                rf"Global min.: {axis_label}={global_xmin:.1e}"
                if add_labels
                else "_nolegend_"
            ),
        )


def plot_xmin_analysis(analysis, ax=None):
    """Plot the raw coarse scan and final xmin choices.

    Candidates with too few retained tail observations are shown in gray for
    diagnostics, but are not part of the simpleDrop search.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    xmin_scale = float(analysis.get("xmin_scale", 1.0))
    if not np.isfinite(xmin_scale) or xmin_scale <= 0.0:
        raise ValueError(f"xmin_scale must be positive and finite, got {xmin_scale!r}")
    axis_label = analysis.get("xmin_axis_label", r"$x_{\min}$")
    simple_details = analysis["simple_drop_details"]
    xmins = np.asarray(analysis["xmins"], dtype=float)
    distances = np.asarray(analysis["distances"], dtype=float)
    search_mask = np.asarray(simple_details["search_mask"], dtype=bool)
    tail_counts = np.asarray(analysis["tail_counts"], dtype=int)
    finite_mask = np.isfinite(xmins) & np.isfinite(distances)
    eligible_mask = finite_mask & search_mask
    display_only_mask = finite_mask & (
        tail_counts < int(analysis["min_tail_count"])
    )
    configured_out_mask = finite_mask & ~search_mask & ~display_only_mask
    ax.plot(
        xmins[finite_mask] / xmin_scale,
        distances[finite_mask],
        color="0.65",
        linewidth=0.8,
        alpha=0.65,
        zorder=1,
        label="Raw coarse scan",
    )
    ax.scatter(
        xmins[display_only_mask] / xmin_scale,
        distances[display_only_mask],
        s=12,
        color="0.65",
        alpha=0.8,
        zorder=2,
        label=(
            rf"Displayed only: $n_{{tail}}<{analysis['min_tail_count']}$ "
            rf"({display_only_mask.sum()} points)"
        ),
    )
    if np.any(configured_out_mask):
        ax.scatter(
            xmins[configured_out_mask] / xmin_scale,
            distances[configured_out_mask],
            s=12,
            facecolor="white",
            edgecolor="0.55",
            linewidth=0.6,
            zorder=2,
            label="Outside configured xmin search range",
        )
    ax.scatter(
        xmins[eligible_mask] / xmin_scale,
        distances[eligible_mask],
        s=14,
        color="tab:red",
        zorder=3,
        label=(
            rf"Eligible raw $D(x_{{\min}})$ "
            rf"($n_{{tail}}\geq{analysis['min_tail_count']}$; "
            rf"{eligible_mask.sum()} points)"
        ),
    )
    adjacent_coarse_indices = np.asarray(
        simple_details["interval_coarse_indices"], dtype=int
    )
    candidate_left = (
        float(simple_details["xmins"][adjacent_coarse_indices[0]]) / xmin_scale
    )
    candidate_right = (
        float(simple_details["xmins"][adjacent_coarse_indices[-1]]) / xmin_scale
    )
    ax.axvspan(
        candidate_left,
        candidate_right,
        color="0.7",
        alpha=0.12,
        zorder=0,
        label="Largest eligible raw adjacent drop",
    )
    search_max = float(analysis.get("xmin_search_max", np.nan))
    if np.isfinite(search_max):
        ax.axvline(
            search_max / xmin_scale,
            color="0.5",
            linestyle="--",
            linewidth=0.8,
            zorder=1,
            label=rf"Selection limit ($n_{{tail}}\geq{analysis['min_tail_count']}$)",
        )
    simple_xmin = analysis["simple_drop_xmin"] / xmin_scale
    simple_distance = analysis["simple_drop_distance"]
    if xmin_global_differs(analysis):
        ax.scatter(
            [simple_xmin],
            [simple_distance],
            marker="D",
            s=30,
            color="tab:blue",
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
            label=rf"simpleDrop: {axis_label}={simple_xmin:.1e}",
        )
        global_xmin = analysis["global_min_xmin"] / xmin_scale
        ax.scatter(
            [global_xmin],
            [analysis["global_min_distance"]],
            marker="X",
            s=35,
            facecolor="white",
            edgecolor="0.25",
            linewidth=0.8,
            zorder=6,
            label=rf"Global min.: {axis_label}={global_xmin:.1e}",
        )
    else:
        global_xmin = analysis["global_min_xmin"] / xmin_scale
        ax.scatter(
            [global_xmin],
            [analysis["global_min_distance"]],
            marker="X",
            s=38,
            facecolor="white",
            edgecolor="0.25",
            linewidth=0.8,
            zorder=6,
            label=rf"Global min.: {axis_label}={global_xmin:.1e}",
        )
    annotate_xmin_choices(
        ax, analysis, add_labels=False, xmin_scale=xmin_scale
    )
    ax.set_xscale("log")
    data_max = float(analysis.get("data_max", np.nan))
    tail_valid_max = float(
        analysis.get("tail_valid_max", np.nanmax(np.asarray(analysis["xmins"])))
    )
    if np.isfinite(data_max) and np.isfinite(tail_valid_max) and data_max > tail_valid_max:
        ax.set_xlim(right=data_max / xmin_scale * 1.05)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(r"$D$")
    ax.set_ylim(0.0, 0.5)
    ax.legend(loc="best")
    return fig, ax


def find_xmin_simple_drop(
    drops,
    debug=False,
    nr_initial=100,
    min_tail_count=100,
    **kwargs,
):
    """Select xmin with the canonical simpleDrop/global-min analysis."""
    analysis = analyze_xmin(
        drops,
        nr_initial=nr_initial,
        min_tail_count=min_tail_count,
        **kwargs,
    )
    if debug:
        plot_xmin_analysis(analysis)
        plt.show()
    return analysis["simple_drop_xmin"], analysis


# Backward-compatible Python alias. The public strategy name is ``simpleDrop``.
find_xmin = find_xmin_simple_drop


def find_xmin_rising_level(drops, debug=False, **kwargs):
    min_xmin = min(drops)
    max_xmin = max(drops)
    nr_first_evaluation = 100
    coarse_xmin_values = np.logspace(
        np.log10(min_xmin), np.log10(max_xmin), nr_first_evaluation
    )
    fits = evaluate_xmin(drops, coarse_xmin_values, **kwargs)
    distances = np.asarray([f.D for f in fits], dtype=float)
    mask = np.isfinite(distances) & np.isfinite(coarse_xmin_values)
    if mask.sum() == 0:
        warnings.warn("No finite KS distances found; cannot compute plateau level.")
        return np.nan
    distances_valid = distances[mask]
    x_valid = coarse_xmin_values[mask]

    # We now have a rough outline of the ks distance plot.
    # We now imagine a horizontal line at some height h,
    # and we measure the distance between the line and the ks distance.

    h = np.linspace(0, 1, nr_first_evaluation)
    pd = [np.nansum(np.abs(distances_valid - h_)) for h_ in h]
    if debug:
        fig, ax1 = plt.subplots()
        ax1.plot(h, pd)

    plateau_h = h[np.argmin(pd)]

    if debug:
        fig, ax2 = plt.subplots()
        ax2.plot(x_valid, distances_valid, label="D (coarse)")
        ax2.axhline(plateau_h)
        ax2.set_xscale("log")
        plt.show()

    # Assuming that there is a roughly flat plateau, the minimum of this
    # measurement should give the height of the plateu.
    # Select the first point after the KS minimum that rises back to that level.
    min_idx = int(np.nanargmin(distances_valid))
    right = np.arange(min_idx + 1, len(distances_valid))
    crossings = right[distances_valid[right] >= plateau_h]
    if crossings.size:
        return float(x_valid[int(crossings[0])])
    if right.size:
        closest = right[int(np.nanargmin(np.abs(distances_valid[right] - plateau_h)))]
        return float(x_valid[int(closest)])
    return float(x_valid[min_idx])


@dataclass(frozen=True)
class XminStrategyResult:
    strategy: str
    xmin: float
    n_tail: int


XMIN_STRATEGIES = {
    "min_ks": find_xmin_ks,
    "ks": find_xmin_ks,
    "global_min": find_xmin_global_min,
    "dip": find_xmin_dip,
    "max_p": find_xmin_max_p,
    "simpleDrop": find_xmin_simple_drop,
    "simple_drop": find_xmin_simple_drop,
    "derivative": find_xmin_derivative,
    "dks": find_xmin_dks,
    "slope": find_xmin_dks,
    "rising_level": find_xmin_rising_level,
    "sizer": find_xmin_sizer,
    "sylvain": find_xmin_sylvain,
}
DEFAULT_XMIN_COMPARISON_STRATEGIES = (
    "simpleDrop",
    "slope",
    "global_min",
)


def select_xmin_with_details(drops, strategy="simpleDrop", **kwargs):
    """Run one named xmin strategy and retain any diagnostic search results."""
    try:
        selector = XMIN_STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError(
            f"Unknown xmin strategy {strategy!r}; choose from {tuple(XMIN_STRATEGIES)}."
        ) from exc
    result = selector(drops, **kwargs)
    if isinstance(result, tuple):
        xmin, details = result
        return float(xmin), details if isinstance(details, dict) else None
    return float(result), None


def select_xmin(drops, strategy="simpleDrop", **kwargs):
    """Run one named xmin strategy while preserving its public API."""
    return select_xmin_with_details(drops, strategy=strategy, **kwargs)[0]


def compare_xmin_strategies(drops, strategies=None, strategy_kwargs=None):
    """Return comparable xmin and tail-size results for multiple strategies."""
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > 0)]
    if drops.size < 3:
        raise ValueError("Need at least three finite positive drops.")
    strategies = tuple(strategies or DEFAULT_XMIN_COMPARISON_STRATEGIES)
    strategy_kwargs = strategy_kwargs or {}
    results = {}
    for strategy in strategies:
        xmin = select_xmin(
            drops,
            strategy=strategy,
            **strategy_kwargs.get(strategy, {}),
        )
        results[strategy] = XminStrategyResult(
            strategy=strategy,
            xmin=xmin,
            n_tail=int(np.count_nonzero(drops >= xmin)) if np.isfinite(xmin) else 0,
        )
    return results


def plot_xmin_strategy_comparison(
    drops,
    strategies=None,
    strategy_kwargs=None,
    *,
    true_xmin=None,
    ax=None,
    save_path=None,
):
    """Plot selected xmin and retained tail size for named strategies."""
    results = compare_xmin_strategies(
        drops,
        strategies=strategies,
        strategy_kwargs=strategy_kwargs,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    names = list(results)
    xmins = [results[name].xmin for name in names]
    tails = [results[name].n_tail for name in names]
    ax.plot(names, xmins, marker="o", linestyle="none", label=r"Selected $x_{min}$")
    ax.set_yscale("log")
    ax.set_ylabel(r"$x_{min}$")
    ax.tick_params(axis="x", rotation=30)
    if true_xmin is not None:
        ax.axhline(true_xmin, linestyle="--", label=r"True $x_{min}$")
    ax2 = ax.twinx()
    ax2.plot(names, tails, marker="s", linestyle=":", label="Tail size")
    ax2.set_ylabel("Drops retained")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, (ax, ax2), results
