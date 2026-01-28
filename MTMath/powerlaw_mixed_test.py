from .plotPowerLaw import (
    make_fit,
    plot_data_and_fit,
    PLOTPATH,
    make_title_from_fit,
    plot_xmin_fitting,
    find_best_xmin,
    plot_ks_distance,
    dist_from_fit,
)
from .evaluatePowerlawFit import Truncated_Power_Law
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from scipy.special import gamma, gammaincc


# Generate synthetic power-law distributed data
def generate_truncated_powerlaw_data(n, alpha, Lambda, xmin):
    dist = Truncated_Power_Law(xmin=xmin, alpha=alpha, Lambda=Lambda)
    return dist.generate_random(size=n)


# Generate power-law avalanche data
def generate_powerlaw_avalanche_data(alpha, size=5000, xmin=1e-8):
    increments = np.random.normal(xmin, xmin, size=size)  # Small incremental increases

    # Randomly select drop points
    drop_mask = np.random.uniform(size=size) > 0.7  # 40% chance of a drop
    Lambda = 1e4
    drops = generate_truncated_powerlaw_data(drop_mask.sum(), alpha, Lambda, xmin)
    # Apply drops
    assert drops is not None
    increments[drop_mask] = -drops
    return increments


def get_only_drops(data):
    drop_mask = data < 0
    drops = -data[drop_mask]
    return drops


def _sample_trunc_powerlaw(n, beta, xlow, xmin, rng):
    u = rng.random(n)
    if np.isclose(beta, 1.0):
        return xlow * (xmin / xlow) ** u
    a = 1.0 - beta
    return (u * (xmin**a - xlow**a) + xlow**a) ** (1.0 / a)


def _sample_cutoff_powerlaw_fast(n, alpha, lam, xmin, rng):
    dist = Truncated_Power_Law(xmin=xmin, alpha=alpha, Lambda=lam)
    return dist.generate_random(size=n, rng=rng)


def _Z_tail(alpha, lam, xmin):
    s = 1.0 - alpha
    x = xmin / lam
    z = (lam ** (1.0 - alpha)) * gamma(s) * gammaincc(s, x)
    return abs(z)


def sample_piecewise(n, A, B, beta, alpha, lam, xmin, xlow=None, rng=None):
    """
    Sample from:
      0 < x < xmin:  f(x) ∝ A*x^{-beta} + B*xmin^{-alpha}*exp(-xmin/lam)
      x >= xmin:     f(x) ∝ B*x^{-alpha}*exp(-x/lam)
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (np.isfinite(xmin) and xmin > 0):
        raise ValueError("xmin must be finite and > 0.")
    if lam <= 0:
        raise ValueError("lam must be > 0.")
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1.")
    if A < 0 or B < 0:
        raise ValueError("A and B must be nonnegative.")
    if not (-alpha * 2 < beta < 3.0 * alpha):
        raise ValueError("Require -alpha < beta < 3*alpha.")

    if xlow is None:
        xlow = xmin * 1e-12
    if not (np.isfinite(xlow) and 0 < xlow < xmin):
        raise ValueError("Need 0 < xlow < xmin.")

    if np.isclose(beta, 1.0):
        I_beta = np.log(xmin / xlow)
    else:
        I_beta = (xmin ** (1.0 - beta) - xlow ** (1.0 - beta)) / (1.0 - beta)
    M_P = A * I_beta

    c = (xmin ** (-alpha)) * np.exp(-xmin / lam)
    M_U = B * c * (xmin - xlow)

    Zt = _Z_tail(alpha, lam, xmin)
    M_T = B * Zt

    M_total = M_P + M_U + M_T
    if M_total <= 0:
        raise ValueError("Total mass is zero; check A,B and parameters.")

    w_left = (M_P + M_U) / M_total
    w_tail = 1.0 - w_left
    w_pow_left = (M_P / (M_P + M_U)) if (M_P + M_U) > 0 else 0.0

    x = np.empty(n, dtype=float)
    u = rng.random(n)
    is_left = u < w_left
    n_left = int(is_left.sum())
    n_tail = n - n_left

    if n_left > 0:
        u2 = rng.random(n_left)
        is_pow = u2 < w_pow_left
        n_pow = int(is_pow.sum())
        n_uni = n_left - n_pow
        left = np.empty(n_left, dtype=float)
        if n_pow > 0:
            left[is_pow] = _sample_trunc_powerlaw(n_pow, beta, xlow, xmin, rng)
        if n_uni > 0:
            left[~is_pow] = xlow + (xmin - xlow) * rng.random(n_uni)
        x[is_left] = left

    if n_tail > 0:
        x[~is_left] = _sample_cutoff_powerlaw_fast(n_tail, alpha, lam, xmin, rng)

    info = {
        "xlow": xlow,
        "M_P": M_P,
        "M_U": M_U,
        "M_T": M_T,
        "M_total": M_total,
        "w_left": w_left,
        "w_tail": w_tail,
        "w_pow_left": w_pow_left,
        "w_uni_left": 1.0 - w_pow_left,
    }
    return x, info


def grid_compare_xmin(
    alphas=None,
    xmins=None,
    n=4000,
    Lambda=1e4,
    A=1.0,
    B=1.0,
    beta=0.5,
    xlow=None,
    rng=None,
    xmin_range=None,
    fast_xmin=True,
):
    """
    Compare xmin estimates on a grid of (alpha, true xmin) using a single distribution.

    xmin1: make_fit(...).xmin
    xmin2: find_best_xmin(...).xmin
    """
    if alphas is None:
        alphas = np.linspace(1.05, 1.8, 6)
    if xmins is None:
        xmins = np.logspace(-9, -6, 6)
    if rng is None:
        rng = np.random.default_rng()

    xmin1_grid = np.full((len(alphas), len(xmins)), np.nan, dtype=float)
    xmin2_grid = np.full_like(xmin1_grid, np.nan)
    alpha1_grid = np.full_like(xmin1_grid, np.nan)
    alpha2_grid = np.full_like(xmin1_grid, np.nan)
    lambda1_grid = np.full_like(xmin1_grid, np.nan)
    lambda2_grid = np.full_like(xmin1_grid, np.nan)

    for i, alpha in enumerate(alphas):
        for j, xmin_true in enumerate(xmins):
            data_info = {
                "customTitle": rf"Synthetic: $\alpha={alpha}, \lambda={Lambda}, E_{{\mathrm{{min}}}}={xmin_true}$"
            }

            drops, _ = sample_piecewise(
                n=n,
                A=A,
                B=B,
                beta=beta,
                alpha=float(alpha),
                lam=Lambda,
                xmin=float(xmin_true),
                xlow=xlow,
                rng=rng,
            )
            drops = np.asarray(drops, dtype=float)
            drops = drops[np.isfinite(drops)]
            if drops.size < 10:
                continue

            fit1 = make_fit(drops, xmin_range=xmin_range, fast_xmin=fast_xmin)
            xmin1_grid[i, j] = float(fit1.xmin)
            alpha1_grid[i, j] = getattr(dist_from_fit(fit1), "alpha", np.nan)
            lambda1_grid[i, j] = getattr(dist_from_fit(fit1), "Lambda", np.nan)

            fit2 = find_best_xmin(
                drops,
                min_xmin=float(drops.min()),
                max_xmin=float(drops.max()),
                xmin_results=fit1.xmin_fitting_results,
                data_info=data_info,
            )
            # plot_ks_distance(drops, max(fit1.xmin, fit2.xmin), data_info=data_info)
            plot_data_and_fit(fit2, data_info=data_info)
            xmin2_grid[i, j] = float(fit2.xmin)
            alpha2_grid[i, j] = getattr(dist_from_fit(fit2), "alpha", np.nan)
            lambda2_grid[i, j] = getattr(dist_from_fit(fit2), "Lambda", np.nan)

    xmin_min = np.nanmin([xmin1_grid.min(), xmin2_grid.min()])
    xmin_max = np.nanmax([xmin1_grid.max(), xmin2_grid.max()])

    dx1_grid = xmin1_grid - np.array(xmins)[None, :]
    dx2_grid = xmin2_grid - np.array(xmins)[None, :]
    da1_grid = alpha1_grid - np.array(alphas)[:, None]
    da2_grid = alpha2_grid - np.array(alphas)[:, None]
    dl1_grid = lambda1_grid - Lambda
    dl2_grid = lambda2_grid - Lambda

    dx_min = np.nanmin([dx1_grid.min(), dx2_grid.min()])
    dx_max = np.nanmax([dx1_grid.max(), dx2_grid.max()])
    da_min = np.nanmin([da1_grid.min(), da2_grid.min()])
    da_max = np.nanmax([da1_grid.max(), da2_grid.max()])
    dl_min = np.nanmin([dl1_grid.min(), dl2_grid.min()])
    dl_max = np.nanmax([dl1_grid.max(), dl2_grid.max()])
    dl_abs_max = np.nanmax(np.abs([dl_min, dl_max]))

    fig, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)

    im1 = axes[0, 0].imshow(
        xmin1_grid, aspect="auto", origin="lower", vmin=xmin_min, vmax=xmin_max
    )
    axes[0, 0].set_title("xmin (min KS)")
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 1].imshow(
        xmin2_grid, aspect="auto", origin="lower", vmin=xmin_min, vmax=xmin_max
    )
    axes[0, 1].set_title("xmin (max $p$)")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im3 = axes[1, 0].imshow(
        dx1_grid, aspect="auto", origin="lower", vmin=dx_min, vmax=dx_max
    )
    axes[1, 0].set_title("xmin (min KS) - true xmin")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(
        dx2_grid, aspect="auto", origin="lower", vmin=dx_min, vmax=dx_max
    )
    axes[1, 1].set_title("xmin (max $p$) - true xmin")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[2, 0].imshow(
        da1_grid, aspect="auto", origin="lower", vmin=da_min, vmax=da_max
    )
    axes[2, 0].set_title("alpha (min KS) - true alpha")
    fig.colorbar(im5, ax=axes[2, 0], fraction=0.046, pad=0.04)

    im6 = axes[2, 1].imshow(
        da2_grid, aspect="auto", origin="lower", vmin=da_min, vmax=da_max
    )
    axes[2, 1].set_title("alpha (max $p$) - true alpha")
    fig.colorbar(im6, ax=axes[2, 1], fraction=0.046, pad=0.04)

    dl1_abs = np.abs(dl1_grid)
    dl2_abs = np.abs(dl2_grid)
    dl_abs_min = np.nanmin([dl1_abs.min(), dl2_abs.min()])
    dl_abs_max = np.nanmax([dl1_abs.max(), dl2_abs.max()])
    if not np.isfinite(dl_abs_min) or dl_abs_min <= 0:
        dl_abs_min = max(dl_abs_max * 1e-6, 1e-12)
    lambda_norm = (
        mcolors.LogNorm(vmin=dl_abs_min, vmax=dl_abs_max)
        if np.isfinite(dl_abs_max) and dl_abs_max > 0
        else None
    )
    im7 = axes[3, 0].imshow(dl1_abs, aspect="auto", origin="lower", norm=lambda_norm)
    axes[3, 0].set_title("|lambda (min KS) - true lambda|")
    fig.colorbar(im7, ax=axes[3, 0], fraction=0.046, pad=0.04)

    im8 = axes[3, 1].imshow(dl2_abs, aspect="auto", origin="lower", norm=lambda_norm)
    axes[3, 1].set_title("|lambda (max $p$) - true lambda|")
    fig.colorbar(im8, ax=axes[3, 1], fraction=0.046, pad=0.04)

    for row in range(4):
        for col in range(2):
            axes[row, col].set_xticks(range(len(xmins)))
            axes[row, col].set_xticklabels(
                [f"{x:.1e}" for x in xmins], rotation=45, ha="right"
            )
            axes[row, col].set_yticks(range(len(alphas)))
            axes[row, col].set_yticklabels([f"{a:.2f}" for a in alphas])
            axes[row, col].set_xlabel("true xmin")
            axes[row, col].set_ylabel("true alpha")

    fig.suptitle("Grid comparison of xmin estimates")
    filename = f"{PLOTPATH}grid_compare_xmin.pdf"
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {filename}")
    plt.close(fig)

    return xmin1_grid, xmin2_grid


def testCombinedDists(alpha1=1.2, alpha2=1.4):
    # drops = generate_truncated_powerlaw_data(
    #     n=1000, alpha=alpha1, Lambda=1e4, xmin=1e-8
    # )
    # # It is different when i use the cache!
    # fit = make_fit(drops, xmin_range=(9.9e-9, 1.1e-8))

    # print(fit.xmin)
    # fit.evaluate_fit()

    drops = get_only_drops(
        generate_powerlaw_avalanche_data(alpha1)
        + generate_powerlaw_avalanche_data(alpha2)
    )

    fit = make_fit(drops, fast_xmin=True)
    fit.evaluate_fit()
    find_best_xmin(drops, debug=True, xmin_results=fit.xmin_fitting_results)
    plot_xmin_fitting(fit, save=True, show=True)

    filename = f"testing/{alpha1}_{alpha2}_lamb=1e4"
    title = make_title_from_fit(fit)
    plot_data_and_fit(fit, title=title, extraPath=filename)
