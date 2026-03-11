import numpy as np
from MTMath.evaluatePowerlawFit import Fit, Truncated_Power_Law
from powerlaw import Distribution
from matplotlib import pyplot as plt
import os
from tqdm import tqdm


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


def _save_debug_fig(fig, filename):
    debug_path = "Plots/powerLaw/debug/"
    os.makedirs(debug_path, exist_ok=True)
    full_path = f"{debug_path}{filename}"
    fig.tight_layout()
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


def _plot_xmin_debug(
    xmin_values,
    distances,
    min_distance,
    region_level,
    region_start,
    region_end,
    min_idx,
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
    legend = ax1.legend(handles, labels, loc="upper right")
    legend.set_zorder(10)

    suffix = _smoothing_suffix(smoothing)
    return _save_debug_fig(fig, f"find_xmin_ks_distance_{suffix}.pdf")


def _plot_dip_derivative_extrema_debug(
    dip_x,
    dip_D,
    dip_d1,
    dip_d2,
    coarse_x=None,
    coarse_D=None,
    smoothing=None,
):
    if (
        dip_x is None
        or dip_D is None
        or dip_d1 is None
        or dip_d2 is None
        or dip_x.size == 0
    ):
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

    if np.isfinite(dip_d2).any():
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

    legend = ax.legend(loc="best")
    legend.set_zorder(10)

    suffix = _smoothing_suffix(smoothing)
    return _save_debug_fig(fig, f"find_xmin_dip_deriv_extrema_{suffix}.pdf")


def _fit_single_xmin_task(args):
    drops, trial_xmin, xmax, dist_name = args
    fit = Fit(
        data=drops,
        xmin=trial_xmin,
        xmax=xmax,
        xmin_distribution=dist_name,
    )
    # Avoid nested multiprocessing inside each worker.
    # fit.evaluate_fit(drops, parallel=False)
    return fit


def evaluate_xmin(
    drops,
    xmin_values,
    distType: type[Distribution] = Truncated_Power_Law,
    xmax=None,
    parallel=False,
    max_workers=None,
):
    tasks = [(drops, trial_xmin, xmax, distType.name) for trial_xmin in xmin_values]

    if parallel:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fits_iter = ex.map(_fit_single_xmin_task, tasks)
            test_fits = list(
                tqdm(
                    fits_iter,
                    total=len(tasks),
                    desc="Fitting xmins",
                    disable=False,
                )
            )
    else:
        test_fits = []
        for i, trial_xmin in enumerate(xmin_values):
            desc = f"xmin:{trial_xmin:.2e}: {i + 1}/{len(xmin_values)}:"
            fit = Fit(
                data=drops,
                xmin=trial_xmin,
                xmax=xmax,
                xmin_distribution=distType.name,
            )
            # fit.evaluate_fit(drops, parallel=False, tqdmDesc=desc)
            test_fits.append(fit)

    return test_fits


def find_xmin(drops, debug=False, smoothing="spline", **kwargs):
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
            dip_d2,
            coarse_x=coarse_xmin_values,
            coarse_D=distances,
            smoothing=smoothing,
        )
    return x[np.argmax(dip_d2)]
