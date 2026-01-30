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
from scipy.special import gamma, gammaincc, expn


# Generate synthetic power-law distributed data
def generate_truncated_powerlaw_data(n, alpha, Lambda, xmin):
    dist = Truncated_Power_Law(xmin=xmin, alpha=alpha, Lambda=Lambda)
    return np.array(dist.generate_random(size=n))


# Generate power-law avalanche data
def generate_powerlaw_avalanche_data(
    alpha, size=5000, xmin=1e-8, lognormal_sigma=1.0, Lambda=1e4
):
    # Positive background increments with a softer transition than a uniform draw.
    logNormalDrops = np.random.lognormal(
        mean=np.log(xmin * 1e-1),
        sigma=lognormal_sigma,
        size=int(size / 2),
    )

    drops = generate_truncated_powerlaw_data(size - int(size / 2), alpha, Lambda, xmin)
    return np.concatenate((logNormalDrops, drops))


def _sample_trunc_powerlaw(n, beta, xlow, xmin, rng):
    u = rng.random(n)
    if np.isclose(beta, 1.0):
        return xlow * (xmin / xlow) ** u
    a = 1.0 - beta
    return (u * (xmin**a - xlow**a) + xlow**a) ** (1.0 / a)


def _sample_cutoff_powerlaw_fast(n, alpha, lam, xmin, rng):
    dist = Truncated_Power_Law(xmin=xmin, alpha=alpha, Lambda=lam)
    return dist.generate_random(size=n, rng=rng)


def _upper_incomplete_gamma(a, x):
    """
    Compute Γ(a, x) for real a (including a<=0) using recurrence.
    Requires x > 0.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("Need x>0 for Γ(a,x).")

    if np.isclose(a, np.round(a), atol=1e-12) and (np.round(a) <= 0):
        n = int(round(1.0 - a))
        return (x**a) * expn(n, x)

    if a > 0.0:
        return gamma(a) * gammaincc(a, x)

    k = int(np.floor(-a)) + 1
    ap = a + k
    G = gamma(ap) * gammaincc(ap, x)

    t = ap
    logx = np.log(x)
    for _ in range(k):
        term = np.exp((t - 1.0) * logx - x)
        G = (G - term) / (t - 1.0)
        t -= 1.0
    return G


def _Z_tail(alpha, lam, xmin):
    s = 1.0 - alpha
    z = np.asarray(xmin, dtype=float) / lam
    G = _upper_incomplete_gamma(s, z)
    return (lam ** (1.0 - alpha)) * G


def sample_piecewise(
    n,
    A,
    B,
    beta,
    alpha,
    lam,
    xmin,
    xlow=None,
    rng=None,
):
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
        xlow = 1e-10
    if not (np.isfinite(xlow) and 0 < xlow < xmin):
        raise ValueError("Need 0 < xlow < xmin.")

    # I_beta is the (unnormalized) integral of x^{-beta} over [xlow, xmin]
    # for the left-side truncated power-law component.
    if np.isclose(beta, 1.0):
        I_beta = np.log(xmin / xlow)
    else:
        I_beta = (xmin ** (1.0 - beta) - xlow ** (1.0 - beta)) / (1.0 - beta)
    # Enforce continuity at xmin by setting the left-side scale A
    # so that A * xmin^{-beta} = B * xmin^{-alpha} * exp(-xmin/lam)
    A = B * (xmin ** (beta - alpha)) * np.exp(-xmin / lam)

    # M_P: total mass of the left power-law component (scaled by A)
    M_P = A * I_beta

    # No uniform plateau on the left; only a pure truncated power law.
    M_U = 0.0

    # Zt is the tail normalization integral over [xmin, ∞) for x^{-alpha} exp(-x/lam)
    Zt = _Z_tail(alpha, lam, xmin)
    # M_T: total mass of the right cutoff-power-law tail (scaled by B)
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
        x[is_left] = _sample_trunc_powerlaw(n_left, beta, xlow, xmin, rng)

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
    n=5000,
    Lambda=1e4,
    A=1.0,
    B=1.0,
    beta=1.5,
    xlow=None,
    seed=0,
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
        alphas = np.linspace(1.05, 3, 6)
    if xmins is None:
        xmins = np.logspace(-9, -3, 6)
    if rng is None:
        rng = np.random.default_rng(seed)

    xmin1_grid = np.full((len(alphas), len(xmins)), np.nan, dtype=float)
    xmin2_grid = np.full_like(xmin1_grid, np.nan)
    alpha1_grid = np.full_like(xmin1_grid, np.nan)
    alpha2_grid = np.full_like(xmin1_grid, np.nan)
    alpha2std_grid = np.full_like(xmin1_grid, np.nan)
    lambda1_grid = np.full_like(xmin1_grid, np.nan)
    lambda2_grid = np.full_like(xmin1_grid, np.nan)

    for i, alpha in enumerate(alphas):
        for j, xmin_true in enumerate(xmins):
            data_info = {
                "customTitle": rf"Synthetic: $\alpha={alpha:.2}, \lambda={Lambda:.0e}, E_{{\mathrm{{min}}}}={xmin_true:.2e}$"
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
            # drops = generate_powerlaw_avalanche_data(alpha, xmin=xmin_true, size=n)

            drops = np.asarray(drops, dtype=float)
            drops = drops[np.isfinite(drops)]
            if drops.size < 10:
                continue

            fit1 = make_fit(
                drops, xmin_range=xmin_range, fast_xmin=fast_xmin, xmin_accuracy=0.1
            )
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
            alpha2std_grid[i, j] = getattr(fit2, "alpha_std", np.nan)
            lambda2_grid[i, j] = getattr(dist_from_fit(fit2), "Lambda", np.nan)

    xmins_arr = np.array(xmins, dtype=float)[None, :]
    xmin1_factor = xmin1_grid / xmins_arr
    xmin2_factor = xmin2_grid / xmins_arr
    da1_grid = alpha1_grid - np.array(alphas)[:, None]
    da2_grid = alpha2_grid - np.array(alphas)[:, None]
    lambda1_factor = lambda1_grid / Lambda
    lambda2_factor = lambda2_grid / Lambda

    dx1_log = np.log10(xmin1_factor)
    dx2_log = np.log10(xmin2_factor)
    dx_min = np.nanmin([dx1_log.min(), dx2_log.min()])
    dx_max = np.nanmax([dx1_log.max(), dx2_log.max()])

    da_min = np.nanmin([da1_grid.min(), da2_grid.min()])
    da_max = np.nanmax([da1_grid.max(), da2_grid.max()])
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)

    im1 = axes[0, 0].imshow(
        dx1_log, aspect="auto", origin="lower", vmin=dx_min, vmax=dx_max
    )
    axes[0, 0].set_title(r"$\log_{10}(x_{\min}/x_{\min,true})$ (min KS)")
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 1].imshow(
        dx2_log, aspect="auto", origin="lower", vmin=dx_min, vmax=dx_max
    )
    axes[0, 1].set_title(r"$\log_{10}(x_{\min}/x_{\min,true})$ (max $p$)")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im3 = axes[1, 0].imshow(
        da1_grid, aspect="auto", origin="lower", vmin=da_min, vmax=da_max
    )
    axes[1, 0].set_title("alpha (min KS) - true alpha")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(
        da2_grid, aspect="auto", origin="lower", vmin=da_min, vmax=da_max
    )
    axes[1, 1].set_title("alpha (max $p$) - true alpha")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    dl1_log = np.log10(lambda1_factor)
    dl2_log = np.log10(lambda2_factor)
    dl_min = np.nanmin([dl1_log.min(), dl2_log.min()])
    dl_max = np.nanmax([dl1_log.max(), dl2_log.max()])
    im5 = axes[2, 0].imshow(
        dl1_log, aspect="auto", origin="lower", vmin=dl_min, vmax=dl_max
    )
    axes[2, 0].set_title(r"$\log_{10}(\lambda/\lambda_{true})$ (min KS)")
    fig.colorbar(im5, ax=axes[2, 0], fraction=0.046, pad=0.04)

    im6 = axes[2, 1].imshow(
        dl2_log, aspect="auto", origin="lower", vmin=dl_min, vmax=dl_max
    )
    axes[2, 1].set_title(r"$\log_{10}(\lambda/\lambda_{true})$ (max $p$)")
    fig.colorbar(im6, ax=axes[2, 1], fraction=0.046, pad=0.04)

    for row in range(3):
        for col in range(2):
            axes[row, col].set_xticks(range(len(xmins)))
            axes[row, col].set_xticklabels(
                [f"{x:.1e}" for x in xmins], rotation=45, ha="right"
            )
            axes[row, col].set_yticks(range(len(alphas)))
            axes[row, col].set_yticklabels([f"{a:.2f}" for a in alphas])
            if row == 2:
                axes[row, col].set_xlabel("true xmin")
            else:
                axes[row, col].set_xlabel("")
            if col == 0:
                axes[row, col].set_ylabel("true alpha")
            else:
                axes[row, col].set_ylabel("")

    from datetime import datetime

    fig.suptitle("Grid comparison of xmin estimates")
    timestamp = datetime.now().strftime("%H%M")
    filename = f"{PLOTPATH}grid_compare_xmin_{timestamp}.pdf"
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {filename}")
    plt.close(fig)

    # --- Standalone alpha z-score plot for method 2 ---
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = np.abs(da2_grid) / alpha2std_grid
    z2 = np.where(np.isfinite(z2), z2, np.nan)

    fig2, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    z_vmax = np.nanmax(z2)
    if not np.isfinite(z_vmax) or z_vmax <= 0:
        z_vmax = 1.0
    imz = ax.imshow(z2, aspect="auto", origin="lower", vmin=0, vmax=z_vmax)
    ax.set_title(
        r"Alpha z-score (max $p$): $\vert \Delta\alpha\vert / \alpha_{\mathrm{std}}$"
    )
    ax.set_xticks(range(len(xmins)))
    ax.set_xticklabels([f"{x:.1e}" for x in xmins], rotation=45, ha="right")
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlabel("true xmin")
    ax.set_ylabel("true alpha")
    fig2.colorbar(imz, ax=ax, fraction=0.046, pad=0.04)

    filename2 = f"{PLOTPATH}grid_compare_alpha_zscore.pdf"
    fig2.savefig(filename2, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {filename2}")
    plt.close(fig2)

    return xmin1_grid, xmin2_grid


def testDist(alpha1=1.05):
    Lambda = 1e4
    drops = generate_powerlaw_avalanche_data(alpha1, xmin=1e-6, Lambda=Lambda)
    fit = make_fit(drops, fast_xmin=True)
    fit.evaluate_fit()
    find_best_xmin(drops, debug=True, xmin_results=fit.xmin_fitting_results)
    plot_xmin_fitting(fit, save=True)

    filename = f"testing/{alpha1}_lamb={Lambda:.0e}"
    title = make_title_from_fit(fit)
    plot_data_and_fit(fit, title=title, extraPath=filename)


def testSamplePiecewise(
    n=5000,
    A=1.0,
    B=1.0,
    beta=2.0,
    alpha=1.2,
    lam=1e3,
    xmin=1e-6,
    xlow=None,
    seed=0,
):
    rng = np.random.default_rng(seed)
    drops, info = sample_piecewise(
        n=n,
        A=A,
        B=B,
        beta=beta,
        alpha=alpha,
        lam=lam,
        xmin=xmin,
        xlow=xlow,
        rng=rng,
    )
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops)]
    if drops.size < 10:
        print("Not enough samples to fit.")
        return None

    fit = make_fit(drops, fast_xmin=True, xmin_accuracy=0.1)
    # fit.evaluate_fit()
    # find_best_xmin(drops, debug=True, xmin_results=fit.xmin_fitting_results)
    # plot_xmin_fitting(fit, save=True)

    filename = f"testing/piecewise_a{alpha:.2f}_lam{lam:.0e}_xmin{xmin:.1e}"
    title = make_title_from_fit(fit)
    plot_data_and_fit(fit, title=title, extraPath=filename)
    return fit, info
