from Plotting.plotPowerLaw import (
    make_fit,
    plot_data_and_fit,
    PLOTPATH,
    make_title_from_fit,
    plot_xmin_fitting,
    find_best_xmin,
    plot_ks_distance,
    dist_from_fit,
)
from comparePlots import combine_pdfs_grid
from .evaluatePowerlawFit import Truncated_Power_Law
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from scipy.special import gamma, gammaincc, expn
from math import isfinite


def _upper_incomplete_gamma(a, x):
    """
    Compute Γ(a, x) (upper incomplete gamma) for real a (including a<=0),
    using recurrence from a' = a + k > 0, and a special-case for integer poles.

    Requires x>0.
    """
    if gammaincc is None or gamma is None:
        raise RuntimeError("SciPy is required (scipy.special.gammaincc, gamma, expn).")
    if not (x > 0.0):
        raise ValueError("Need x>0 for Γ(a,x).")

    # If a is a non-positive integer, gamma(a) has a pole; use identity with expn:
    # For n = 1-a (integer >= 1):  Γ(1-n, x) = x^{1-n} E_n(x) = x^a * expn(n, x)
    if np.isclose(a, np.round(a), atol=1e-12) and (np.round(a) <= 0):
        n = int(round(1.0 - a))
        return (x**a) * expn(n, x)

    # If a>0, direct formula is fine
    if a > 0.0:
        return gamma(a) * gammaincc(a, x)

    # Otherwise, shift up to positive a+k, compute Γ(a+k, x), then recur down
    k = int(np.floor(-a)) + 1  # ensures a+k > 0
    ap = a + k
    G = gamma(ap) * gammaincc(ap, x)  # Γ(ap, x) with ap>0

    # Downward recurrence: Γ(t-1,x) = (Γ(t,x) - x^{t-1} e^{-x}) / (t-1)
    t = ap
    logx = np.log(x)
    for _ in range(k):
        term = np.exp((t - 1.0) * logx - x)  # x^{t-1} e^{-x}
        G = (G - term) / (t - 1.0)
        t -= 1.0

    return G


def _Z_tail(alpha, lam, xmin):
    """
    Z_tail = ∫_{xmin}^∞ x^{-alpha} exp(-x/lam) dx
           = lam^{1-alpha} Γ(1-alpha, xmin/lam)
    Works for alpha>1 (i.e., 1-alpha<0).
    """
    if lam <= 0 or xmin <= 0:
        raise ValueError("Need lam>0 and xmin>0.")
    a = 1.0 - alpha
    z = xmin / lam
    G = _upper_incomplete_gamma(a, z)
    Zt = (lam ** (1.0 - alpha)) * G
    if not np.isfinite(Zt) or Zt <= 0:
        raise RuntimeError(
            f"Z_tail is invalid: {Zt}. Check alpha={alpha}, lam={lam}, xmin={xmin}."
        )
    return Zt


def _sample_trunc_powerlaw(n, beta, xlow, xhigh, rng):
    """
    Sample from pdf ∝ x^{-beta} on [xlow, xhigh], with xlow>0.

    Inverse CDF:
      - if beta != 1:
          F(x) = (x^{1-beta} - xlow^{1-beta}) / (xhigh^{1-beta} - xlow^{1-beta})
      - if beta == 1:
          F(x) = log(x/xlow) / log(xhigh/xlow)
    """
    if not (xlow > 0 and xhigh > xlow):
        raise ValueError("Need 0 < xlow < xhigh for truncated power law.")

    u = rng.random(n)

    if np.isclose(beta, 1.0):
        # x = xlow * (xhigh/xlow)^u
        return xlow * (xhigh / xlow) ** u

    p = 1.0 - beta
    a = xlow**p
    b = xhigh**p
    # x = (a + u*(b-a))^(1/p)
    return (a + u * (b - a)) ** (1.0 / p)


def _sample_cutoff_powerlaw_fast_sylvain(n, alpha, lam, xmin, rng):
    """
    Sample from density proportional to x^{-alpha} exp(-x/lam) on [xmin, ∞),
    using Pareto proposal and accept with exp(-x/lam). Requires alpha > 1.
    """
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1 for this sampler.")

    out = np.empty(n, dtype=float)
    filled = 0
    finfo = np.finfo(float)

    while filled < n:
        m = max(4096, int(1.2 * (n - filled)))

        u = rng.random(m)
        u = np.minimum(u, 1.0 - finfo.eps)  # avoid u=1

        # Proposal g(x) ∝ x^{-alpha} on [xmin,∞):
        y = xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))

        # Accept with probability exp(-y/lam)
        accept = rng.random(m) < np.exp(-y / lam)
        k = int(accept.sum())
        if k:
            take = min(k, n - filled)
            out[filled : filled + take] = y[accept][:take]
            filled += take

    return out


def sample_piecewise(n, beta, alpha, lam, xmin, A=None, B=None, xlow=None, rng=None):
    """
    Sample from:
      0 < x < xmin:  f(x) ∝ A*x^{-beta} + B*xmin^{-alpha}*exp(-xmin/lam)
      x >= xmin:     f(x) ∝ B*x^{-alpha}*exp(-x/lam)

    Notes:
      - alpha > 1 required.
      - Left power-law needs a lower cutoff xlow>0 if beta>=1 (otherwise not normalizable).
        If xlow is None, we set xlow = xmin*1e-3 as a practical default.
      - If A and/or B are None, they are chosen to enforce continuity at xmin:
        A * xmin^{-beta} = B * xmin^{-alpha} * exp(-xmin/lam).

    Returns:
      x: samples (shape n,)
      info: dict of mixture weights and component masses.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(n)
    # Basic checks
    if not (isfinite(xmin) and xmin > 0):
        raise ValueError("xmin must be finite and > 0.")
    if lam <= 0:
        raise ValueError("lam must be > 0.")
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1.")
    if A is not None and A < 0:
        raise ValueError("A must be nonnegative.")
    if B is not None and B < 0:
        raise ValueError("B must be nonnegative.")
    if not (-alpha * 2 < beta < 3.0 * alpha):
        raise ValueError("Require -alpha < beta < 3*alpha (as requested).")

    # Set / validate xlow
    if xlow is None:
        xlow = xmin * 1e-3  # edit as needed
    if not (isfinite(xlow) and 0 < xlow < xmin):
        raise ValueError(
            "Need 0 < xlow < xmin (xlow is the lower cutoff for the left power-law)."
        )

    # Choose A/B to enforce continuity at xmin if needed.
    if A is None and B is None:
        B = 1.0
        A = B * (xmin ** (beta - alpha)) * np.exp(-xmin / lam)
    elif A is None:
        A = B * (xmin ** (beta - alpha)) * np.exp(-xmin / lam)
    elif B is None:
        B = A * (xmin ** (alpha - beta)) * np.exp(xmin / lam)

    # Component masses
    # Left: power-law mass M_P = A * ∫_{xlow}^{xmin} x^{-beta} dx
    if np.isclose(beta, 1.0):
        I_beta = np.log(xmin / xlow)
    else:
        I_beta = (xmin ** (1.0 - beta) - xlow ** (1.0 - beta)) / (1.0 - beta)
    M_P = A * I_beta

    # Left: plateau constant density value at xmin (times B)
    c = (xmin ** (-alpha)) * np.exp(-xmin / lam)
    # M_U = B * c * (xmin - 0.0)  # uniform on (0,xmin)
    M_U = B * c * (xmin - xlow)

    # Tail: cutoff power-law mass
    Zt = _Z_tail(alpha, lam, xmin)
    M_T = B * Zt

    M_total = M_P + M_U + M_T
    if M_total <= 0:
        raise ValueError("Total mass is zero; check A,B and parameters.")

    # Mixture weights:
    w_left = (M_P + M_U) / M_total
    w_tail = 1.0 - w_left
    w_pow_left = (M_P / (M_P + M_U)) if (M_P + M_U) > 0 else 0.0
    w_uni_left = 1.0 - w_pow_left

    # Allocate and sample
    x = np.empty(n, dtype=float)

    u = rng.random(n)
    is_left = u < w_left
    n_left = int(is_left.sum())
    n_tail = n - n_left

    # Left region sampling: choose power-law vs uniform plateau
    if n_left > 0:
        u2 = rng.random(n_left)
        is_pow = u2 < w_pow_left
        n_pow = int(is_pow.sum())
        n_uni = n_left - n_pow

        left = np.empty(n_left, dtype=float)
        if n_pow > 0:
            left[is_pow] = _sample_trunc_powerlaw(n_pow, beta, xlow, xmin, rng)
        if n_uni > 0:
            # left[~is_pow] = xmin * rng.random(n_uni)  # uniform on (0,xmin)
            left[~is_pow] = xlow + (xmin - xlow) * rng.random(
                n_uni
            )  # uniform on (xlow,xmin)

        x[is_left] = left

    # Tail sampling
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
        "w_uni_left": w_uni_left,
    }
    return x, info


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


def _sample_cutoff_powerlaw_fast(n, alpha, lam, xmin, rng):
    dist = Truncated_Power_Law(xmin=xmin, alpha=alpha, Lambda=lam)
    return dist.generate_random(size=n, rng=rng)


def _symmetric_limits(arrays, default=1.0):
    values = []
    for arr in arrays:
        if arr is None:
            continue
        finite_vals = np.asarray(arr)[np.isfinite(arr)]
        if finite_vals.size:
            values.append(finite_vals)
    if not values:
        return -default, default
    max_abs = np.nanmax(np.abs(np.concatenate(values)))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = default
    return -max_abs, max_abs


def _format_grid_axes(axes, xmins, alphas):
    x_tick_labels = [f"{x:.1e}" for x in xmins]
    y_tick_labels = [f"{a:.2f}" for a in alphas]
    for row in range(3):
        for col in range(2):
            ax = axes[row, col]
            ax.set_xticks(range(len(xmins)))
            ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
            ax.set_yticks(range(len(alphas)))
            ax.set_yticklabels(y_tick_labels)
            ax.set_xlabel("true xmin" if row == 2 else "")
            ax.set_ylabel("true alpha" if col == 0 else "")


def _plot_grid_compare(
    dx1_log,
    dx2_log,
    da1_grid,
    da2_grid,
    dl1_log,
    dl2_log,
    xmins,
    alphas,
    filename,
    dx_limits,
    da_limits,
    dl_limits,
    cmap="coolwarm",
):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)

    im1 = axes[0, 0].imshow(
        dx1_log,
        aspect="auto",
        origin="lower",
        vmin=dx_limits[0],
        vmax=dx_limits[1],
        cmap=cmap,
    )
    axes[0, 0].set_title(r"$\log_{10}(x_{\min}/x_{\min,true})$ (min KS)")
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 1].imshow(
        dx2_log,
        aspect="auto",
        origin="lower",
        vmin=dx_limits[0],
        vmax=dx_limits[1],
        cmap=cmap,
    )
    axes[0, 1].set_title(r"$\log_{10}(x_{\min}/x_{\min,true})$ (max $p$)")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im3 = axes[1, 0].imshow(
        da1_grid,
        aspect="auto",
        origin="lower",
        vmin=da_limits[0],
        vmax=da_limits[1],
        cmap=cmap,
    )
    axes[1, 0].set_title("alpha (min KS) - true alpha")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(
        da2_grid,
        aspect="auto",
        origin="lower",
        vmin=da_limits[0],
        vmax=da_limits[1],
        cmap=cmap,
    )
    axes[1, 1].set_title("alpha (max $p$) - true alpha")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[2, 0].imshow(
        dl1_log,
        aspect="auto",
        origin="lower",
        vmin=dl_limits[0],
        vmax=dl_limits[1],
        cmap=cmap,
    )
    axes[2, 0].set_title(r"$\log_{10}(\lambda/\lambda_{true})$ (min KS)")
    fig.colorbar(im5, ax=axes[2, 0], fraction=0.046, pad=0.04)

    im6 = axes[2, 1].imshow(
        dl2_log,
        aspect="auto",
        origin="lower",
        vmin=dl_limits[0],
        vmax=dl_limits[1],
        cmap=cmap,
    )
    axes[2, 1].set_title(r"$\log_{10}(\lambda/\lambda_{true})$ (max $p$)")
    fig.colorbar(im6, ax=axes[2, 1], fraction=0.046, pad=0.04)

    _format_grid_axes(axes, xmins, alphas)
    fig.suptitle("Grid comparison of xmin estimates")
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {filename}")
    plt.close(fig)


def _plot_alpha_zscore(z2, xmins, alphas, filename, vmin, vmax, cmap="coolwarm"):
    fig2, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    imz = ax.imshow(z2, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
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
    fig2.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {filename}")
    plt.close(fig2)


def _ensure_placeholder_pdf(path, text="Missing"):
    if os.path.exists(path):
        return path
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.axis("off")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def _save_fit_plot_paths(drops, ks_fit, p_fit, data_info, extra_path=""):
    ax1 = plot_ks_distance(
        drops,
        ks_fit.xmin,
        data_info=data_info,
        ax=None,
        save=True,
        close=True,
        extraPath=extra_path,
    )
    ks_min_path = getattr(ax1.figure, "path", None)

    ax2 = plot_ks_distance(
        drops,
        p_fit.xmin,
        data_info=data_info,
        ax=None,
        save=True,
        close=True,
        extraPath=extra_path,
    )
    ks_max_path = getattr(ax2.figure, "path", None)

    ax3 = plot_data_and_fit(
        p_fit,
        data_info=data_info,
        ax=None,
        save=True,
        close=True,
        extraPath=extra_path,
    )
    fit_path = getattr(ax3.figure, "path", None)

    return ks_min_path, ks_max_path, fit_path


def grid_compare_xmin(
    alphas=None,
    xmins=None,
    n=5e4,
    Lambda=1,
    beta=0.0,
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
        xmins = np.logspace(-5, 0, 6)
    if rng is None:
        rng = np.random.default_rng(seed)

    xmin1_grid = np.full((len(alphas), len(xmins)), np.nan, dtype=float)
    xmin2_grid = np.full_like(xmin1_grid, np.nan)
    alpha1_grid = np.full_like(xmin1_grid, np.nan)
    alpha2_grid = np.full_like(xmin1_grid, np.nan)
    alpha2std_grid = np.full_like(xmin1_grid, np.nan)
    lambda1_grid = np.full_like(xmin1_grid, np.nan)
    lambda2_grid = np.full_like(xmin1_grid, np.nan)
    ks_min_paths = []
    ks_max_paths = []
    fit_paths = []
    output_subdir = "grid_compare_xmin_cells/"
    os.makedirs(f"{PLOTPATH}{output_subdir}", exist_ok=True)
    placeholder_path = _ensure_placeholder_pdf(
        f"{PLOTPATH}{output_subdir}placeholder.pdf", text="No data"
    )

    for i, alpha in enumerate(alphas):
        for j, xmin_true in enumerate(xmins):
            data_info = {
                "customTitle": rf"Synthetic: $\alpha={alpha:.2}, \lambda={Lambda:.0e}, E_{{\mathrm{{min}}}}={xmin_true:.2e}$"
            }

            drops, _ = sample_piecewise(
                n=n,
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
                ks_min_paths.append(placeholder_path)
                ks_max_paths.append(placeholder_path)
                fit_paths.append(placeholder_path)
                continue

            KS_fit = make_fit(
                drops, xmin_range=xmin_range, fast_xmin=fast_xmin, xmin_accuracy=0.1
            )
            xmin1_grid[i, j] = float(KS_fit.xmin)
            alpha1_grid[i, j] = getattr(dist_from_fit(KS_fit), "alpha", np.nan)
            lambda1_grid[i, j] = getattr(dist_from_fit(KS_fit), "Lambda", np.nan)

            p_fit = find_best_xmin(
                drops, xmin_results=KS_fit.xmin_fitting_results, data_info=data_info
            )
            ks_min_path, ks_max_path, fit_path = _save_fit_plot_paths(
                drops, KS_fit, p_fit, data_info, extra_path=output_subdir
            )
            ks_min_paths.append(ks_min_path or placeholder_path)
            ks_max_paths.append(ks_max_path or placeholder_path)
            fit_paths.append(fit_path or placeholder_path)

            xmin2_grid[i, j] = float(p_fit.xmin)
            alpha2_grid[i, j] = getattr(dist_from_fit(p_fit), "alpha", np.nan)
            alpha2std_grid[i, j] = getattr(p_fit, "alpha_std", np.nan)
            lambda2_grid[i, j] = getattr(dist_from_fit(p_fit), "Lambda", np.nan)

    xmins_arr = np.array(xmins, dtype=float)[None, :]
    xmin1_factor = xmin1_grid / xmins_arr
    xmin2_factor = xmin2_grid / xmins_arr
    alphas_arr = np.array(alphas, dtype=float)[:, None]
    da1_grid = alpha1_grid - alphas_arr
    da2_grid = alpha2_grid - alphas_arr
    lambda1_factor = lambda1_grid / Lambda
    lambda2_factor = lambda2_grid / Lambda

    dx1_log = np.log10(xmin1_factor)
    dx2_log = np.log10(xmin2_factor)
    dl1_log = np.log10(lambda1_factor)
    dl2_log = np.log10(lambda2_factor)

    from datetime import datetime

    timestamp = datetime.now().strftime("%H%M")
    fixed_limits = (-1.0, 1.0)
    filename = f"{PLOTPATH}grid_compare_xmin_{timestamp}.pdf"
    _plot_grid_compare(
        dx1_log,
        dx2_log,
        da1_grid,
        da2_grid,
        dl1_log,
        dl2_log,
        xmins,
        alphas,
        filename,
        fixed_limits,
        fixed_limits,
        fixed_limits,
        cmap="coolwarm",
    )

    dx_limits = _symmetric_limits([dx1_log, dx2_log])
    da_limits = _symmetric_limits([da1_grid, da2_grid])
    dl_limits = _symmetric_limits([dl1_log, dl2_log])
    filename_centered = f"{PLOTPATH}grid_compare_xmin_centered_{timestamp}.pdf"
    _plot_grid_compare(
        dx1_log,
        dx2_log,
        da1_grid,
        da2_grid,
        dl1_log,
        dl2_log,
        xmins,
        alphas,
        filename_centered,
        dx_limits,
        da_limits,
        dl_limits,
        cmap="coolwarm",
    )

    rows = len(alphas)
    cols = len(xmins)
    filename1 = f"{PLOTPATH}ks_distance_grid_minKS_{timestamp}.pdf"
    filename2 = f"{PLOTPATH}ks_distance_grid_maxP_{timestamp}.pdf"
    filename3 = f"{PLOTPATH}fit_grid_{timestamp}.pdf"
    combine_pdfs_grid(ks_min_paths, rows, cols, filename1)
    print(f"Saved figure to {filename1}")
    combine_pdfs_grid(ks_max_paths, rows, cols, filename2)
    print(f"Saved figure to {filename2}")
    combine_pdfs_grid(fit_paths, rows, cols, filename3)
    print(f"Saved figure to {filename3}")

    # --- Standalone alpha z-score plot for method 2 ---
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = np.abs(da2_grid) / alpha2std_grid
    z2 = np.where(np.isfinite(z2), z2, np.nan)

    z_vmax = np.nanmax(z2)
    if not np.isfinite(z_vmax) or z_vmax <= 0:
        z_vmax = 1.0
    filename2 = f"{PLOTPATH}grid_compare_alpha_zscore.pdf"
    _plot_alpha_zscore(
        z2, xmins, alphas, filename2, vmin=0, vmax=z_vmax, cmap="coolwarm"
    )

    filename2_fixed = f"{PLOTPATH}grid_compare_alpha_zscore_fixed_0_2.pdf"
    _plot_alpha_zscore(
        z2, xmins, alphas, filename2_fixed, vmin=0, vmax=2, cmap="coolwarm"
    )

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
    n=5e4,
    beta=0,
    alpha=3,
    lam=1,
    xmin=1e-1,
    xlow=None,
    seed=0,
):
    rng = np.random.default_rng(seed)
    drops, info = sample_piecewise(
        n=n,
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

    KS_fit = make_fit(drops, xmin_range=None, fast_xmin=True, xmin_accuracy=0.01)
    # KS_fit.evaluate_fit()
    p_fit = find_best_xmin(drops, debug=True, xmin_results=KS_fit.xmin_fitting_results)
    plot_xmin_fitting(p_fit, save=True)

    filename = f"testing/piecewise_a{alpha:.2f}_lam{lam:.0e}_xmin{xmin:.1e}"
    title = make_title_from_fit(p_fit)
    plot_data_and_fit(p_fit, title=title, extraPath=filename)
    return p_fit, info
