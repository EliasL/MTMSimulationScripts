from Plotting.plotPowerLaw import (
    make_fit,
    plot_data_and_fit,
    PLOTPATH,
    make_title_from_fit,
    plot_xmin_fitting,
    plot_KS_fitting,
    find_best_xmin,
    plot_ks_distance,
    dist_from_fit,
)
from Plotting.makePlots import (
    create_color_matrix,
    plot_color_matrix,
    find_best_color_matrix_corner,
)

from Plotting.findXmin import (
    find_xmin,
    find_xmin_sylvain,
    find_xmin_rising_level,
    find_xmin_derivative,
    find_xmin_dks_from_results,
)
from comparePlots import combine_pdfs_grid
from .evaluatePowerlawFit import Truncated_Power_Law
import os
import json
import gzip
import hashlib
import numpy as np
from matplotlib import pyplot as plt
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
    ks_label="KS",
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
    axes[0, 0].set_title(
        rf"$\log_{{10}}(x_{{\min}}/x_{{\min,true}})$ (min {ks_label})"
    )
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
    axes[1, 0].set_title(f"alpha (min {ks_label}) - true alpha")
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
    axes[2, 0].set_title(
        rf"$\log_{{10}}(\lambda/\lambda_{{true}})$ (min {ks_label})"
    )
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


def _grid_compare_xmin_params(
    alphas,
    xmins,
    n,
    Lambda,
    beta,
    xlow,
    seed,
    xmin_range,
):
    return {
        "alphas": [float(a) for a in alphas],
        "xmins": [float(x) for x in xmins],
        "n": int(n),
        "Lambda": float(Lambda),
        "beta": float(beta),
        "xlow": None if xlow is None else float(xlow),
        "seed": int(seed),
        "fixed_xmin": None if xmin_range is None else float(xmin_range),
        "xmin_method": "simpleDrop",
        "dks_method": "min_dks",
    }


def _grid_compare_xmin_cache_path(params, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    key = json.dumps(params, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(key).hexdigest()
    return os.path.join(cache_dir, f"grid_compare_xmin_{h}.json.gz")


def _save_grid_compare_xmin_cache(data, cache_path):
    with gzip.open(cache_path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def _load_grid_compare_xmin_cache(cache_path):
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        return json.load(f)


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


def grid_compare_xmin_plot(data=None, cache_path=None):
    if data is None:
        if cache_path is None:
            raise ValueError("Provide data or cache_path.")
        data = _load_grid_compare_xmin_cache(cache_path)

    seed = None
    if isinstance(data, dict):
        if "seed" in data:
            seed = data["seed"]
        elif isinstance(data.get("params"), dict):
            seed = data["params"].get("seed")
    seed_tag = f"_s{seed}" if seed is not None else ""

    alphas = np.array(data["alphas"], dtype=float)
    xmins = np.array(data["xmins"], dtype=float)
    Lambda = float(data["Lambda"])
    n_samples = int(data["n_samples"])
    ks_label = "simpleDrop"
    ks_tag_lower = "simpledrop"

    xmin1_grid = np.array(data["xmin1_grid"], dtype=float)
    xmin2_grid = np.array(data["xmin2_grid"], dtype=float)
    alpha1_grid = np.array(data["alpha1_grid"], dtype=float)
    alpha2_grid = np.array(data["alpha2_grid"], dtype=float)
    alpha2std_grid = np.array(data["alpha2std_grid"], dtype=float)
    lambda1_grid = np.array(data["lambda1_grid"], dtype=float)
    lambda2_grid = np.array(data["lambda2_grid"], dtype=float)

    xmins_arr = xmins[None, :]
    xmin1_factor = xmin1_grid / xmins_arr
    xmin2_factor = xmin2_grid / xmins_arr
    alphas_arr = alphas[:, None]
    da1_grid = alpha1_grid - alphas_arr
    da2_grid = alpha2_grid - alphas_arr
    lambda1_factor = lambda1_grid / Lambda
    lambda2_factor = lambda2_grid / Lambda

    with np.errstate(divide="ignore", invalid="ignore"):
        dx1_log = np.log10(xmin1_factor)
        dx2_log = np.log10(xmin2_factor)
        dl1_log = np.log10(lambda1_factor)
        dl2_log = np.log10(lambda2_factor)

    fixed_limits = (-1.0, 1.0)
    filename = f"{PLOTPATH}grid_compare_xmin_n{n_samples}{seed_tag}.pdf"
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
        ks_label=ks_label,
        cmap="coolwarm",
    )

    dx_limits = _symmetric_limits([dx1_log, dx2_log])
    da_limits = _symmetric_limits([da1_grid, da2_grid])
    dl_limits = _symmetric_limits([dl1_log, dl2_log])
    filename_centered = (
        f"{PLOTPATH}grid_compare_xmin_centered_n{n_samples}{seed_tag}.pdf"
    )
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
        ks_label=ks_label,
        cmap="coolwarm",
    )

    rows = len(alphas)
    cols = len(xmins)
    ks_min_paths = data["ks_min_paths"]
    ks_max_paths = data["ks_max_paths"]
    fit_paths = data["fit_paths"]
    xmin_plot_paths = data["xmin_plot_paths"]
    placeholder_path = data["placeholder_path"]

    def _sanitize_paths(paths):
        return [p if p and os.path.exists(p) else placeholder_path for p in paths]

    filename1 = (
        f"{PLOTPATH}{ks_tag_lower}_distance_grid_min{ks_label}_n{n_samples}{seed_tag}.pdf"
    )
    filename2 = (
        f"{PLOTPATH}{ks_tag_lower}_distance_grid_maxP_n{n_samples}{seed_tag}.pdf"
    )
    filename3 = f"{PLOTPATH}fit_grid_n{n_samples}{seed_tag}.pdf"
    filename4 = f"{PLOTPATH}xmin_fits_grid_n{n_samples}{seed_tag}.pdf"
    combine_pdfs_grid(_sanitize_paths(ks_min_paths), rows, cols, filename1)
    print(f"Saved figure to {filename1}")
    combine_pdfs_grid(_sanitize_paths(ks_max_paths), rows, cols, filename2)
    print(f"Saved figure to {filename2}")
    combine_pdfs_grid(_sanitize_paths(fit_paths), rows, cols, filename3)
    print(f"Saved figure to {filename3}")
    combine_pdfs_grid(_sanitize_paths(xmin_plot_paths), rows, cols, filename4)
    print(f"Saved figure to {filename4}")

    # --- Standalone alpha z-score plot for method 2 ---
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = np.abs(da2_grid) / alpha2std_grid
    z2 = np.where(np.isfinite(z2), z2, np.nan)

    z_vmax = np.nanmax(z2)
    if not np.isfinite(z_vmax) or z_vmax <= 0:
        z_vmax = 1.0
    filename2 = f"{PLOTPATH}grid_compare_alpha_zscore_n{n_samples}{seed_tag}.pdf"
    _plot_alpha_zscore(
        z2, xmins, alphas, filename2, vmin=0, vmax=z_vmax, cmap="coolwarm"
    )

    filename2_fixed = (
        f"{PLOTPATH}grid_compare_alpha_zscore_fixed_0_2_n{n_samples}{seed_tag}.pdf"
    )
    _plot_alpha_zscore(
        z2, xmins, alphas, filename2_fixed, vmin=0, vmax=2, cmap="coolwarm"
    )

    return data


def _coerce_grid_compare_inputs(data=None, cache_path=None):
    if data is None and cache_path is None:
        raise ValueError("Provide data or cache_path.")
    if data is None:
        if isinstance(cache_path, (list, tuple)):
            return [_load_grid_compare_xmin_cache(p) for p in cache_path]
        return [_load_grid_compare_xmin_cache(cache_path)]
    if isinstance(data, (list, tuple)):
        return list(data)
    return [data]


def _parse_subgrid(subgrid):
    if subgrid is None:
        return None
    if isinstance(subgrid, (int, np.integer)):
        n = int(subgrid)
        if n <= 0:
            raise ValueError("subgrid size must be positive.")
        return (n, n)
    try:
        n_alpha, n_xmin = subgrid
    except Exception as e:
        raise ValueError("subgrid must be an int or a (n_alpha, n_xmin) tuple.") from e
    n_alpha = int(n_alpha)
    n_xmin = int(n_xmin)
    if n_alpha <= 0 or n_xmin <= 0:
        raise ValueError("subgrid sizes must be positive.")
    return (n_alpha, n_xmin)


def _slice_grid(grid, n_alpha, n_xmin):
    arr = np.asarray(grid, dtype=float)
    if arr.ndim < 2:
        return np.empty((0, 0), dtype=float)
    return arr[:n_alpha, :n_xmin]


def _apply_subgrid_to_data(data, subgrid):
    subgrid = _parse_subgrid(subgrid)
    if subgrid is None or data is None:
        return data

    alphas = np.array(data.get("alphas", []), dtype=float)
    xmins = np.array(data.get("xmins", []), dtype=float)
    if alphas.size == 0 or xmins.size == 0:
        return data

    n_alpha, n_xmin = subgrid
    n_alpha = min(n_alpha, len(alphas))
    n_xmin = min(n_xmin, len(xmins))

    out = dict(data)
    out["alphas"] = [float(a) for a in alphas[:n_alpha]]
    out["xmins"] = [float(x) for x in xmins[:n_xmin]]

    grid_keys = [
        "xmin1_grid",
        "xmin2_grid",
        "xmin3_grid",
        "alpha1_grid",
        "alpha2_grid",
        "alpha2std_grid",
        "alpha3_grid",
        "alpha3std_grid",
        "lambda1_grid",
        "lambda2_grid",
        "lambda2std_grid",
        "lambda3_grid",
        "lambda3std_grid",
        "plateau_xmin_grid",
        "n_tail1_grid",
        "n_tail2_grid",
        "n_tail3_grid",
        "n_tail_plateau_grid",
    ]
    for key in grid_keys:
        if key in out:
            out[key] = _slice_grid(out[key], n_alpha, n_xmin).tolist()

    full_cols = len(xmins)

    def _slice_paths(paths):
        if not isinstance(paths, (list, tuple)):
            return paths
        sliced = []
        for i in range(n_alpha):
            row_start = i * full_cols
            for j in range(n_xmin):
                idx = row_start + j
                if idx < len(paths):
                    sliced.append(paths[idx])
        return sliced

    for key in ("ks_min_paths", "ks_max_paths", "fit_paths", "xmin_plot_paths"):
        if key in out:
            out[key] = _slice_paths(out[key])

    if isinstance(out.get("params"), dict):
        params = dict(out["params"])
        params["alphas"] = [float(a) for a in out["alphas"]]
        params["xmins"] = [float(x) for x in out["xmins"]]
        out["params"] = params

    return out


def _format_sample_size(value):
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return value
        iv = int(value)
        if float(value) == iv:
            return str(iv)
    except (TypeError, ValueError):
        return str(value)
    return f"{value}"


def _get_sample_size_label(data, override=None):
    if override is not None:
        return _format_sample_size(override)
    if "n_samples" in data:
        return _format_sample_size(data["n_samples"])
    if "params" in data and "n" in data["params"]:
        return _format_sample_size(data["params"]["n"])
    return None


def _scatter_points(
    ax, xvals, yvals, colors, marker, logx=False, logy=False, size=36, alpha=0.8
):
    xs = []
    ys = []
    for x, y, c in zip(xvals, yvals, colors):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if logx and x <= 0:
            continue
        if logy and y <= 0:
            continue
        ax.scatter(
            x,
            y,
            s=size,
            facecolors="none",
            edgecolors=c,
            marker=marker,
            alpha=alpha,
        )
        xs.append(x)
        ys.append(y)
    return xs, ys


def plot_compare_xmin(
    data=None,
    cache_path=None,
    sample_sizes=None,
    subgrid=None,
    method="all",
    legend=True,
):
    """
    Scatter-plot estimated vs true values using grid_compare_xmin_generate output.

    Parameters
    ----------
    data : dict or list[dict]
        Output from grid_compare_xmin_generate (or list of such dicts).
    cache_path : str or list[str]
        Cache path(s) produced by grid_compare_xmin_generate (ignored if data given).
    sample_sizes : list or None
        Optional labels for number of drops per dataset; falls back to data["n_samples"]
        or data["params"]["n"] if available.
    subgrid : int or tuple or None
        If set, selects the first (n_alpha, n_xmin) parameters starting at (0,0).
    method : {"max_p", "min_ks", "dks", "plateau", "all"}
        Which estimator to plot for xmin/alpha/lambda.
    legend : bool
        If True, add a legend entry per sample size (marker shape).
    Notes
    -----
    The alpha error vs alpha std plot is produced for max_p and dks when
    alpha_std is available.
    """
    datasets = _coerce_grid_compare_inputs(data=data, cache_path=cache_path)
    if sample_sizes is not None and len(sample_sizes) != len(datasets):
        raise ValueError("sample_sizes must match number of datasets.")
    subgrid = _parse_subgrid(subgrid)

    method = method.lower()
    if method == "both":
        method = "all"
    if method not in ("max_p", "min_ks", "dks", "plateau", "all"):
        raise ValueError(
            "method must be 'max_p', 'min_ks', 'dks', 'plateau', or 'all'."
        )

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]

    all_xmins_list = []
    all_alphas_list = []
    for d in datasets:
        alphas = np.array(d["alphas"], dtype=float)
        xmins = np.array(d["xmins"], dtype=float)
        if subgrid is not None:
            n_alpha, n_xmin = subgrid
            alphas = alphas[: min(n_alpha, len(alphas))]
            xmins = xmins[: min(n_xmin, len(xmins))]
        all_alphas_list.append(alphas)
        all_xmins_list.append(xmins)
    all_xmins = (
        np.unique(np.concatenate(all_xmins_list)) if all_xmins_list else np.array([])
    )
    all_alphas = (
        np.unique(np.concatenate(all_alphas_list)) if all_alphas_list else np.array([])
    )
    if all_xmins.size == 0 or all_alphas.size == 0:
        raise ValueError("Subgrid selection resulted in empty parameter set.")

    if all_xmins.size == 0 or all_alphas.size == 0:
        raise ValueError("Subgrid selection resulted in empty parameter set.")

    xmin_grid, alpha_grid = np.meshgrid(all_xmins, all_alphas)
    color_matrix, unique_xmins, unique_alphas = create_color_matrix(
        xmin_grid.ravel(),
        alpha_grid.ravel(),
        log_p1=True,
        log_p2=False,
    )
    color_lookup = {
        (float(alpha), float(xmin)): color_matrix[row, col]
        for row, alpha in enumerate(unique_alphas)
        for col, xmin in enumerate(unique_xmins)
    }
    color_keys = ["xmin", "alpha"]
    color_xlabel = r"$\log_{10}(x_{\min})$"

    sample_labels = []
    for idx, data_i in enumerate(datasets):
        sample_label = _get_sample_size_label(
            data_i, None if sample_sizes is None else sample_sizes[idx]
        )
        sample_labels.append(sample_label)

    def _fmt_xmin_log(val):
        if not np.isfinite(val) or val <= 0:
            return "?"
        return f"{int(np.round(np.log10(val)))}"

    def _fmt_alpha(val):
        if not np.isfinite(val):
            return "?"
        return f"{val:.1f}"

    def _plot_scatter(
        points,
        title,
        xlabel,
        ylabel,
        filename,
        logx=False,
        logy=False,
        add_diagonal=False,
        color_matrix_kwargs=None,
    ):
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        all_x = []
        all_y = []
        for xvals, yvals, colors, marker, _sample_label in points:
            xs, ys = _scatter_points(
                ax, xvals, yvals, colors, marker, logx=logx, logy=logy
            )
            all_x.extend(xs)
            all_y.extend(ys)

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

        if add_diagonal and all_x and all_y:
            min_val = min(np.min(all_x), np.min(all_y))
            max_val = max(np.max(all_x), np.max(all_y))
            if (logx or logy) and min_val <= 0:
                min_val = min(v for v in all_x + all_y if v > 0)
            ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cm_loc, cm_bbox = find_best_color_matrix_corner(
            all_x, all_y, logx=logx, logy=logy
        )
        if color_matrix_kwargs is None:
            color_matrix_kwargs = {}
        color_matrix_kwargs = {
            "loc": cm_loc,
            "bbox_to_anchor": cm_bbox,
            **color_matrix_kwargs,
        }
        plot_color_matrix(
            ax,
            color_matrix,
            unique_xmins,
            unique_alphas,
            color_keys,
            fmt_p1=_fmt_xmin_log,
            fmt_p2=_fmt_alpha,
            xlabel=color_xlabel,
            **color_matrix_kwargs,
        )
        if legend:
            used = set()
            for _, _, _, marker, sample_label in points:
                label = f"n={sample_label}" if sample_label is not None else "n=?"
                if label in used:
                    continue
                ax.scatter(
                    [],
                    [],
                    s=36,
                    facecolors="none",
                    edgecolors="k",
                    marker=marker,
                    alpha=0.8,
                    label=label,
                )
                used.add(label)
            ax.legend(
                fontsize="x-small",
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
            )
        fig.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"Saved figure to {filename}")
        plt.close(fig)

    def _plot_alpha_err_vs_std(points, title, xlabel, ylabel, filename):
        if not points:
            print("No alpha std data available; skipping alpha error vs std plot.")
            return
        _plot_scatter(
            points,
            title,
            xlabel,
            ylabel,
            filename,
            logx=False,
            logy=False,
            add_diagonal=True,
        )

    def _collect_points(method_name):
        xmin_points = []
        lambda_xmin_points = []
        lambda_alpha_points = []
        alpha_points = []
        alpha_err_points = []

        for idx, data_i in enumerate(datasets):
            alphas = np.array(data_i["alphas"], dtype=float)
            xmins = np.array(data_i["xmins"], dtype=float)
            if subgrid is not None:
                n_alpha, n_xmin = subgrid
                alphas = alphas[: min(n_alpha, len(alphas))]
                xmins = xmins[: min(n_xmin, len(xmins))]
            alpha_true = np.repeat(alphas, len(xmins))
            xmin_true = np.tile(xmins, len(alphas))

            if method_name == "min_ks":
                xmin_est = _slice_grid(
                    data_i["xmin1_grid"], len(alphas), len(xmins)
                ).ravel()
                alpha_est = _slice_grid(
                    data_i["alpha1_grid"], len(alphas), len(xmins)
                ).ravel()
                lambda_est = _slice_grid(
                    data_i["lambda1_grid"], len(alphas), len(xmins)
                ).ravel()
            elif method_name == "max_p":
                xmin_est = _slice_grid(
                    data_i["xmin2_grid"], len(alphas), len(xmins)
                ).ravel()
                alpha_est = _slice_grid(
                    data_i["alpha2_grid"], len(alphas), len(xmins)
                ).ravel()
                lambda_est = _slice_grid(
                    data_i["lambda2_grid"], len(alphas), len(xmins)
                ).ravel()
            elif method_name == "dks":
                xmin_est = _slice_grid(
                    data_i["xmin3_grid"], len(alphas), len(xmins)
                ).ravel()
                alpha_est = _slice_grid(
                    data_i["alpha3_grid"], len(alphas), len(xmins)
                ).ravel()
                lambda_est = _slice_grid(
                    data_i["lambda3_grid"], len(alphas), len(xmins)
                ).ravel()
            else:
                if "plateau_xmin_grid" not in data_i:
                    raise ValueError(
                        "plateau_xmin_grid not found. Regenerate cache with "
                        "the updated grid_compare_xmin_generate."
                    )
                xmin_est = _slice_grid(
                    data_i["plateau_xmin_grid"], len(alphas), len(xmins)
                ).ravel()
                alpha_est = np.full_like(xmin_est, np.nan)
                lambda_est = np.full_like(xmin_est, np.nan)

            colors = [
                color_lookup.get((float(a), float(x)), (0.0, 0.0, 0.0, 1.0))
                for a, x in zip(alpha_true, xmin_true)
            ]
            marker = marker_cycle[idx % len(marker_cycle)]
            sample_label = sample_labels[idx]

            xmin_points.append((xmin_true, xmin_est, colors, marker, sample_label))
            lambda_xmin_points.append(
                (xmin_true, lambda_est, colors, marker, sample_label)
            )
            lambda_alpha_points.append(
                (alpha_true, lambda_est, colors, marker, sample_label)
            )
            alpha_points.append((alpha_true, alpha_est, colors, marker, sample_label))

            if method_name == "max_p":
                alpha_std = _slice_grid(
                    data_i.get("alpha2std_grid", []),
                    len(alphas),
                    len(xmins),
                ).ravel()
                if alpha_std.size:
                    alpha2_est = _slice_grid(
                        data_i["alpha2_grid"], len(alphas), len(xmins)
                    ).ravel()
                    if alpha2_est.shape == alpha_std.shape:
                        alpha_err = np.abs(alpha2_est - alpha_true)
                        alpha_err_points.append(
                            (alpha_std, alpha_err, colors, marker, sample_label)
                        )
            elif method_name == "dks":
                alpha_std = _slice_grid(
                    data_i.get("alpha3std_grid", []),
                    len(alphas),
                    len(xmins),
                ).ravel()
                if alpha_std.size:
                    alpha3_est = _slice_grid(
                        data_i["alpha3_grid"], len(alphas), len(xmins)
                    ).ravel()
                    if alpha3_est.shape == alpha_std.shape:
                        alpha_err = np.abs(alpha3_est - alpha_true)
                        alpha_err_points.append(
                            (alpha_std, alpha_err, colors, marker, sample_label)
                        )

        return (
            xmin_points,
            lambda_xmin_points,
            lambda_alpha_points,
            alpha_points,
            alpha_err_points,
        )

    if len(datasets) == 1:
        n_samples = int(datasets[0].get("n_samples", 0))
        suffix = f"n{n_samples}" if n_samples > 0 else "single"
    else:
        suffix = "multi"
    if subgrid is not None:
        suffix = f"{suffix}_subgrid{int(subgrid[0])}x{int(subgrid[1])}"

    output_dir = f"{PLOTPATH}plot_compare/"
    os.makedirs(output_dir, exist_ok=True)

    methods = ["min_ks", "max_p", "dks", "plateau"] if method == "all" else [method]

    for method_name in methods:
        if method_name == "min_ks":
            method_tag = "minKS"
        elif method_name == "max_p":
            method_tag = "maxP"
        elif method_name == "dks":
            method_tag = "DKS"
        else:
            method_tag = "Plateau"
        (
            xmin_points,
            lambda_xmin_points,
            lambda_alpha_points,
            alpha_points,
            alpha_err_points,
        ) = _collect_points(method_name)

        _plot_scatter(
            xmin_points,
            f"xmin estimate vs true ({method_tag})",
            "true xmin",
            "estimated xmin",
            f"{output_dir}plot_compare_xmin_scatter_xmin_{method_tag}_{suffix}.pdf",
            logx=True,
            logy=True,
            add_diagonal=True,
        )
        if method_name != "plateau":
            _plot_scatter(
                lambda_xmin_points,
                f"lambda estimate vs true xmin ({method_tag})",
                "true xmin",
                "estimated lambda",
                f"{output_dir}plot_compare_xmin_scatter_lambda_vs_xmin_{method_tag}_{suffix}.pdf",
                logx=True,
                logy=True,
                add_diagonal=False,
            )
            _plot_scatter(
                lambda_alpha_points,
                f"lambda estimate vs true alpha ({method_tag})",
                "true alpha",
                "estimated lambda",
                f"{output_dir}plot_compare_xmin_scatter_lambda_vs_alpha_{method_tag}_{suffix}.pdf",
                logx=False,
                logy=True,
                add_diagonal=False,
            )
            _plot_scatter(
                alpha_points,
                f"alpha estimate vs true ({method_tag})",
                "true alpha",
                "estimated alpha",
                f"{output_dir}plot_compare_xmin_scatter_alpha_{method_tag}_{suffix}.pdf",
                logx=False,
                logy=False,
                add_diagonal=True,
            )
            if method_name == "max_p":
                _plot_alpha_err_vs_std(
                    alpha_err_points,
                    "alpha error vs alpha std (max p)",
                    r"$\alpha_{\mathrm{std}}$",
                    r"$|\hat{\alpha} - \alpha_{\mathrm{true}}|$",
                    f"{output_dir}plot_compare_xmin_scatter_alpha_err_std_maxP_{suffix}.pdf",
                )
            if method_name == "dks":
                _plot_alpha_err_vs_std(
                    alpha_err_points,
                    "alpha error vs alpha std (DKS)",
                    r"$\alpha_{\mathrm{std}}$",
                    r"$|\hat{\alpha} - \alpha_{\mathrm{true}}|$",
                    f"{output_dir}plot_compare_xmin_scatter_alpha_err_std_DKS_{suffix}.pdf",
                )


def grid_compare_xmin_generate(
    alphas=None,
    xmins=None,
    n=5e3,
    Lambda=1,
    beta=0.0,
    xlow=1e-7,
    seed=0,
    rng=None,
    xmin_range=None,
    use_cache=True,
    cache_dir=None,
    force=False,
):
    """
    Generate data/fits for the xmin grid and save individual plots.
    Returns a dict with grids and plot paths (memoizable).
    """
    if alphas is None:
        alphas = np.linspace(1.05, 3, 6)
    if xmins is None:
        xmins = np.logspace(-5, 0, 6)
    if rng is None:
        rng = np.random.default_rng(seed)

    params = _grid_compare_xmin_params(
        alphas, xmins, n, Lambda, beta, xlow, seed, xmin_range
    )
    cache_dir = cache_dir or f"{PLOTPATH}grid_compare_xmin_cache/"
    cache_path = _grid_compare_xmin_cache_path(params, cache_dir)
    if use_cache and not force and os.path.exists(cache_path):
        return _load_grid_compare_xmin_cache(cache_path)

    xmin1_grid = np.full((len(alphas), len(xmins)), np.nan, dtype=float)
    xmin2_grid = np.full_like(xmin1_grid, np.nan)
    xmin3_grid = np.full_like(xmin1_grid, np.nan)
    alpha1_grid = np.full_like(xmin1_grid, np.nan)
    alpha2_grid = np.full_like(xmin1_grid, np.nan)
    alpha2std_grid = np.full_like(xmin1_grid, np.nan)
    alpha3_grid = np.full_like(xmin1_grid, np.nan)
    alpha3std_grid = np.full_like(xmin1_grid, np.nan)
    lambda1_grid = np.full_like(xmin1_grid, np.nan)
    lambda2_grid = np.full_like(xmin1_grid, np.nan)
    lambda2std_grid = np.full_like(xmin1_grid, np.nan)
    lambda3_grid = np.full_like(xmin1_grid, np.nan)
    lambda3std_grid = np.full_like(xmin1_grid, np.nan)
    plateau_xmin_grid = np.full_like(xmin1_grid, np.nan)
    n_tail1_grid = np.full_like(xmin1_grid, np.nan)
    n_tail2_grid = np.full_like(xmin1_grid, np.nan)
    n_tail3_grid = np.full_like(xmin1_grid, np.nan)
    n_tail_plateau_grid = np.full_like(xmin1_grid, np.nan)
    ks_min_paths = []
    ks_max_paths = []
    fit_paths = []
    xmin_plot_paths = []
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
                xmin_plot_paths.append(placeholder_path)
                continue

            KS_fit = make_fit(
                drops,
                xmin_range=xmin_range,
                parallel_xmin=True,
            )
            # Cheaky messy spaghetti, we sneak in our new xmin result
            plateau_xmin = find_xmin(drops)
            KS_fit.xmin_fitting_results["plateau_xmin"] = plateau_xmin
            plateau_xmin_grid[i, j] = float(plateau_xmin)
            n_tail_plateau_grid[i, j] = float(np.sum(drops >= plateau_xmin))

            xmin1_grid[i, j] = float(KS_fit.xmin)
            n_tail1_grid[i, j] = float(np.sum(drops >= KS_fit.xmin))
            alpha1_grid[i, j] = getattr(dist_from_fit(KS_fit), "alpha", np.nan)
            lambda1_grid[i, j] = getattr(dist_from_fit(KS_fit), "Lambda", np.nan)

            p_fit = find_best_xmin(
                drops,
                xmin_results=KS_fit.xmin_fitting_results,
                data_info=data_info,
                extraPath=output_subdir,
                parallel=True,
            )
            ks_min_path, ks_max_path, fit_path = _save_fit_plot_paths(
                drops, KS_fit, p_fit, data_info, extra_path=output_subdir
            )
            ks_min_paths.append(ks_min_path or placeholder_path)
            ks_max_paths.append(ks_max_path or placeholder_path)
            fit_paths.append(fit_path or placeholder_path)
            xmin_plot_path = getattr(p_fit, "xmin_plot_path", None)
            xmin_plot_paths.append(xmin_plot_path or placeholder_path)

            xmin2_grid[i, j] = float(p_fit.xmin)
            n_tail2_grid[i, j] = float(np.sum(drops >= p_fit.xmin))
            alpha2_grid[i, j] = getattr(dist_from_fit(p_fit), "alpha", np.nan)
            alpha2std_grid[i, j] = getattr(p_fit, "alpha_std", np.nan)
            lambda2_grid[i, j] = getattr(dist_from_fit(p_fit), "Lambda", np.nan)
            lambda2std_grid[i, j] = getattr(
                p_fit, "Lambda_std", getattr(p_fit, "lambda_std", np.nan)
            )

            xmin_results = getattr(KS_fit, "xmin_fitting_results", {}) or {}
            dks_xmin = find_xmin_dks_from_results(
                xmin_results.get("xmins", []),
                xmin_results.get("distances", []),
                xmin_results.get("valid_fits"),
            )
            if np.isfinite(dks_xmin):
                dks_fit = make_fit(
                    drops,
                    xmin_range=float(dks_xmin),
                    parallel_xmin=False,
                )
                dks_fit.evaluate_fit(drops, confidence=0.05, parallel=False)
                xmin3_grid[i, j] = float(dks_fit.xmin)
                n_tail3_grid[i, j] = float(np.sum(drops >= dks_fit.xmin))
                alpha3_grid[i, j] = getattr(dist_from_fit(dks_fit), "alpha", np.nan)
                alpha3std_grid[i, j] = getattr(dks_fit, "alpha_std", np.nan)
                lambda3_grid[i, j] = getattr(dist_from_fit(dks_fit), "Lambda", np.nan)
                lambda3std_grid[i, j] = getattr(
                    dks_fit, "Lambda_std", getattr(dks_fit, "lambda_std", np.nan)
                )

    data = {
        "params": params,
        "n_samples": int(n),
        "alphas": [float(a) for a in alphas],
        "xmins": [float(x) for x in xmins],
        "Lambda": float(Lambda),
        "xmin1_grid": xmin1_grid.tolist(),
        "xmin2_grid": xmin2_grid.tolist(),
        "xmin3_grid": xmin3_grid.tolist(),
        "alpha1_grid": alpha1_grid.tolist(),
        "alpha2_grid": alpha2_grid.tolist(),
        "alpha2std_grid": alpha2std_grid.tolist(),
        "alpha3_grid": alpha3_grid.tolist(),
        "alpha3std_grid": alpha3std_grid.tolist(),
        "lambda1_grid": lambda1_grid.tolist(),
        "lambda2_grid": lambda2_grid.tolist(),
        "lambda2std_grid": lambda2std_grid.tolist(),
        "lambda3_grid": lambda3_grid.tolist(),
        "lambda3std_grid": lambda3std_grid.tolist(),
        "plateau_xmin_grid": plateau_xmin_grid.tolist(),
        "n_tail1_grid": n_tail1_grid.tolist(),
        "n_tail2_grid": n_tail2_grid.tolist(),
        "n_tail3_grid": n_tail3_grid.tolist(),
        "n_tail_plateau_grid": n_tail_plateau_grid.tolist(),
        "ks_min_paths": ks_min_paths,
        "ks_max_paths": ks_max_paths,
        "fit_paths": fit_paths,
        "xmin_plot_paths": xmin_plot_paths,
        "output_subdir": output_subdir,
        "placeholder_path": placeholder_path,
    }
    _save_grid_compare_xmin_cache(data, cache_path)
    return data


def plot_convergence_xmin(
    data=None,
    cache_path=None,
    subgrid=None,
    method="all",
):
    """
    Plot convergence of estimates vs sample size n.

    Produces raw and rescaled plots for xmin, alpha, and lambda.
    Raw xmin/lambda are log-log. Rescaled plots show estimate/true.

    Parameters
    ----------
    data : dict or list[dict]
        Output from grid_compare_xmin_generate (or list of such dicts).
    cache_path : str or list[str]
        Cache path(s) produced by grid_compare_xmin_generate (ignored if data given).
    subgrid : int or tuple or None
        If set, selects the first (n_alpha, n_xmin) parameters starting at (0,0).
    method : {"max_p", "min_ks", "dks", "all"}
        Which estimator to plot for xmin/alpha/lambda.
    """
    datasets = _coerce_grid_compare_inputs(data=data, cache_path=cache_path)
    if not datasets:
        raise ValueError("No datasets provided.")
    subgrid = _parse_subgrid(subgrid)

    method = method.lower()
    if method == "both":
        method = "all"
    if method not in ("max_p", "min_ks", "dks", "all"):
        raise ValueError("method must be 'max_p', 'min_ks', 'dks', or 'all'.")

    ns = np.array([float(d.get("n_samples", np.nan)) for d in datasets], dtype=float)
    order = np.argsort(ns)
    datasets = [datasets[i] for i in order]
    ns = ns[order]

    all_xmins_list = []
    all_alphas_list = []
    for d in datasets:
        alphas = np.array(d["alphas"], dtype=float)
        xmins = np.array(d["xmins"], dtype=float)
        if subgrid is not None:
            n_alpha, n_xmin = subgrid
            alphas = alphas[: min(n_alpha, len(alphas))]
            xmins = xmins[: min(n_xmin, len(xmins))]
        all_alphas_list.append(alphas)
        all_xmins_list.append(xmins)
    all_xmins = (
        np.unique(np.concatenate(all_xmins_list)) if all_xmins_list else np.array([])
    )
    all_alphas = (
        np.unique(np.concatenate(all_alphas_list)) if all_alphas_list else np.array([])
    )
    xmin_grid, alpha_grid = np.meshgrid(all_xmins, all_alphas)
    color_matrix, unique_xmins, unique_alphas = create_color_matrix(
        xmin_grid.ravel(),
        alpha_grid.ravel(),
        log_p1=True,
        log_p2=False,
    )
    color_lookup = {
        (float(alpha), float(xmin)): color_matrix[row, col]
        for row, alpha in enumerate(unique_alphas)
        for col, xmin in enumerate(unique_xmins)
    }
    color_keys = ["xmin", "alpha"]
    color_xlabel = r"$\log_{10}(x_{\min})$"

    def _fmt_xmin_log(val):
        if not np.isfinite(val) or val <= 0:
            return "?"
        return f"{int(np.round(np.log10(val)))}"

    def _fmt_alpha(val):
        if not np.isfinite(val):
            return "?"
        return f"{val:.1f}"

    def _collect_series(method_name):
        xmin_series = {}
        alpha_series = {}
        lambda_series = {}

        for d in datasets:
            alphas = np.array(d["alphas"], dtype=float)
            xmins = np.array(d["xmins"], dtype=float)
            if subgrid is not None:
                n_alpha, n_xmin = subgrid
                alphas = alphas[: min(n_alpha, len(alphas))]
                xmins = xmins[: min(n_xmin, len(xmins))]
            n_val = float(d.get("n_samples", np.nan))
            lambda_true = float(d.get("Lambda", np.nan))
            if method_name == "min_ks":
                xmin_grid = _slice_grid(d["xmin1_grid"], len(alphas), len(xmins))
                alpha_grid = _slice_grid(d["alpha1_grid"], len(alphas), len(xmins))
                lambda_grid = _slice_grid(d["lambda1_grid"], len(alphas), len(xmins))
                n_tail_grid = (
                    _slice_grid(d.get("n_tail1_grid"), len(alphas), len(xmins))
                    if "n_tail1_grid" in d
                    else None
                )
            elif method_name == "max_p":
                xmin_grid = _slice_grid(d["xmin2_grid"], len(alphas), len(xmins))
                alpha_grid = _slice_grid(d["alpha2_grid"], len(alphas), len(xmins))
                lambda_grid = _slice_grid(d["lambda2_grid"], len(alphas), len(xmins))
                n_tail_grid = (
                    _slice_grid(d.get("n_tail2_grid"), len(alphas), len(xmins))
                    if "n_tail2_grid" in d
                    else None
                )
            elif method_name == "dks":
                xmin_grid = _slice_grid(d["xmin3_grid"], len(alphas), len(xmins))
                alpha_grid = _slice_grid(d["alpha3_grid"], len(alphas), len(xmins))
                lambda_grid = _slice_grid(d["lambda3_grid"], len(alphas), len(xmins))
                n_tail_grid = (
                    _slice_grid(d.get("n_tail3_grid"), len(alphas), len(xmins))
                    if "n_tail3_grid" in d
                    else None
                )
            else:
                if "plateau_xmin_grid" not in d:
                    raise ValueError(
                        "plateau_xmin_grid not found. Regenerate cache with "
                        "the updated grid_compare_xmin_generate."
                    )
                xmin_grid = _slice_grid(d["plateau_xmin_grid"], len(alphas), len(xmins))
                alpha_grid = np.full_like(xmin_grid, np.nan)
                lambda_grid = np.full_like(xmin_grid, np.nan)
                n_tail_grid = (
                    _slice_grid(d.get("n_tail_plateau_grid"), len(alphas), len(xmins))
                    if "n_tail_plateau_grid" in d
                    else None
                )

            for i, alpha_true in enumerate(alphas):
                for j, xmin_true in enumerate(xmins):
                    key = (float(alpha_true), float(xmin_true))
                    if n_tail_grid is not None and n_tail_grid.size:
                        n_used = float(n_tail_grid[i, j])
                        if not np.isfinite(n_used) or n_used <= 0:
                            n_used = n_val
                    else:
                        n_used = n_val
                    xmin_series.setdefault(key, []).append((n_used, xmin_grid[i, j]))
                    alpha_series.setdefault(key, []).append((n_used, alpha_grid[i, j]))
                    lambda_series.setdefault(key, []).append(
                        (n_used, lambda_grid[i, j], lambda_true)
                    )

        return xmin_series, alpha_series, lambda_series

    def _plot_series(
        series,
        title,
        xlabel,
        ylabel,
        filename,
        logx=False,
        logy=False,
        rescale=None,
        hline=None,
        y_clip=None,
    ):
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        all_x = []
        all_y = []
        for (alpha_true, xmin_true), values in series.items():
            values = sorted(values, key=lambda v: v[0])
            xs = np.array([v[0] for v in values], dtype=float)
            ys = np.array([v[1] for v in values], dtype=float)
            if len(values[0]) >= 3:
                ts = np.array([v[2] for v in values], dtype=float)
            else:
                ts = None
            if rescale is not None:
                ys = rescale(ys, alpha_true, xmin_true, ts)

            if logx:
                mask = np.isfinite(xs) & (xs > 0)
                xs = xs[mask]
                ys = ys[mask]
            if logy:
                mask = np.isfinite(ys) & (ys > 0)
                xs = xs[mask]
                ys = ys[mask]

            if xs.size < 2:
                continue

            color = color_lookup.get((alpha_true, xmin_true), (0.0, 0.0, 0.0, 1.0))
            ax.plot(xs, ys, color=color, lw=1.2, alpha=0.8)
            ax.scatter(
                xs,
                ys,
                s=20,
                facecolors="none",
                edgecolors=color,
                alpha=0.9,
            )
            all_x.extend(xs.tolist())
            all_y.extend(ys.tolist())

        if logx and not any((np.isfinite(v) and v > 0) for v in all_x):
            print(f"Skipping plot {filename}: no positive x-values for log scale.")
            plt.close(fig)
            return
        if logy and not any((np.isfinite(v) and v > 0) for v in all_y):
            print(f"Skipping plot {filename}: no positive y-values for log scale.")
            plt.close(fig)
            return

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

        if y_clip is not None and all_y:
            y_vals = np.array(all_y, dtype=float)
            y_mask = np.isfinite(y_vals)
            if logy:
                y_mask &= y_vals > 0
            y_vals = y_vals[y_mask]
            if y_vals.size:
                y_vals = y_vals[np.abs(y_vals) <= y_clip]
                if y_vals.size:
                    y_min = float(np.min(y_vals))
                    y_max = float(np.max(y_vals))
                    if np.isclose(y_min, y_max):
                        pad = 0.1 * (abs(y_min) if y_min != 0 else 1.0)
                    else:
                        pad = 0.05 * (y_max - y_min)
                    ax.set_ylim(y_min - pad, y_max + pad)

        if hline is not None and np.isfinite(hline):
            ax.axhline(hline, color="k", lw=1, ls="--", zorder=0)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        cm_loc, cm_bbox = find_best_color_matrix_corner(
            all_x, all_y, logx=logx, logy=logy
        )
        plot_color_matrix(
            ax,
            color_matrix,
            unique_xmins,
            unique_alphas,
            color_keys,
            fmt_p1=_fmt_xmin_log,
            fmt_p2=_fmt_alpha,
            xlabel=color_xlabel,
            loc=cm_loc,
            bbox_to_anchor=cm_bbox,
        )

        fig.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"Saved figure to {filename}")
        plt.close(fig)

    output_dir = f"{PLOTPATH}plot_convergence/"
    os.makedirs(output_dir, exist_ok=True)
    methods = ["min_ks", "max_p", "dks", "plateau"] if method == "all" else [method]

    if len(datasets) == 1:
        n_samples = int(datasets[0].get("n_samples", 0))
        suffix = f"_n{n_samples}" if n_samples > 0 else "_single"
    else:
        suffix = "_multi"
    if subgrid is not None:
        suffix += f"_subgrid{int(subgrid[0])}x{int(subgrid[1])}"

    x_label = r"$n_{\mathrm{tail}}$ (samples in fit)"

    for method_name in methods:
        if method_name == "min_ks":
            method_tag = "minKS"
        elif method_name == "max_p":
            method_tag = "maxP"
        elif method_name == "dks":
            method_tag = "DKS"
        else:
            method_tag = "Plateau"
        xmin_series, alpha_series, lambda_series = _collect_series(method_name)

        _plot_series(
            xmin_series,
            f"xmin estimate vs n ({method_tag})",
            x_label,
            "estimated xmin",
            f"{output_dir}convergence_xmin_raw_{method_tag}{suffix}.pdf",
            logx=True,
            logy=True,
        )
        _plot_series(
            xmin_series,
            f"xmin estimate / true vs n ({method_tag})",
            x_label,
            "estimated xmin / true xmin",
            f"{output_dir}convergence_xmin_rescaled_{method_tag}{suffix}.pdf",
            logx=True,
            logy=False,
            rescale=lambda y, a, x, t: y / x,
            hline=1.0,
            y_clip=5.0,
        )
        if method_name != "plateau":
            _plot_series(
                alpha_series,
                f"alpha estimate vs n ({method_tag})",
                x_label,
                "estimated alpha",
                f"{output_dir}convergence_alpha_raw_{method_tag}{suffix}.pdf",
                logx=True,
                logy=False,
            )
            _plot_series(
                alpha_series,
                f"alpha estimate / true vs n ({method_tag})",
                x_label,
                "estimated alpha / true alpha",
                f"{output_dir}convergence_alpha_rescaled_{method_tag}{suffix}.pdf",
                logx=True,
                logy=False,
                rescale=lambda y, a, x, t: y / a,
                hline=1.0,
                y_clip=5.0,
            )
            _plot_series(
                lambda_series,
                f"lambda estimate vs n ({method_tag})",
                x_label,
                "estimated lambda",
                f"{output_dir}convergence_lambda_raw_{method_tag}{suffix}.pdf",
                logx=True,
                logy=True,
            )
            _plot_series(
                lambda_series,
                f"lambda estimate / true vs n ({method_tag})",
                x_label,
                "estimated lambda / true lambda",
                f"{output_dir}convergence_lambda_rescaled_{method_tag}{suffix}.pdf",
                logx=True,
                logy=False,
                rescale=lambda y, a, x, t: y / t,
                hline=1.0,
                y_clip=5.0,
            )


def grid_compare_xmin(
    alphas=None,
    xmins=None,
    n=5e3,
    Lambda=1,
    beta=0.0,
    xlow=1e-7,
    seed=0,
    rng=None,
    xmin_range=None,
    use_cache=True,
    cache_dir=None,
    force=False,
    subgrid=None,
    plot=True,
):
    data = grid_compare_xmin_generate(
        alphas=alphas,
        xmins=xmins,
        n=n,
        Lambda=Lambda,
        beta=beta,
        xlow=xlow,
        seed=seed,
        rng=rng,
        xmin_range=xmin_range,
        use_cache=use_cache,
        cache_dir=cache_dir,
        force=force,
    )
    data = _apply_subgrid_to_data(data, subgrid)
    if plot:
        grid_compare_xmin_plot(data=data)
    return data


def testDist(alpha1=1.05):
    Lambda = 1e4
    drops = generate_powerlaw_avalanche_data(alpha1, xmin=1e-6, Lambda=Lambda)
    fit = make_fit(drops)
    fit.evaluate_fit()
    find_best_xmin(
        drops, debug=True, xmin_results=fit.xmin_fitting_results
    )
    plot_xmin_fitting(fit, save=True)

    filename = f"testing/{alpha1}_lamb={Lambda:.0e}"
    title = make_title_from_fit(fit)
    plot_data_and_fit(fit, title=title, extraPath=filename)


def testSamplePiecewise(
    n=5e3,
    beta=0.0,
    alpha=3.0,
    lam=1.0,
    xmin=1e0,
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

    KS_fit = make_fit(drops)
    # KS_fit.evaluate_fit()
    plateau_xmin = find_xmin(drops, debug=True)

    # xmin = find_xmin_sylvain(drops, debug=True, B=1000, p0=0.1, c=1.0, nr_evaluation=20)
    # find_xmin_rising_level(drops, True)
    KS_fit.xmin_fitting_results["plateau_xmin"] = plateau_xmin

    p_fit = find_best_xmin(
        drops, debug=True, xmin_results=KS_fit.xmin_fitting_results
    )
    plot_xmin_fitting(KS_fit, save=True)
    plot_KS_fitting(KS_fit, save=True)

    filename = f"testing/piecewise_a{alpha:.2f}_lam{lam:.0e}_xmin{xmin:.1e}"
    title = make_title_from_fit(p_fit)
    plot_data_and_fit(p_fit, title=title, extraPath=filename)
    return p_fit, info
