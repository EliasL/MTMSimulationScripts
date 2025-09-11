# KS vs AD power comparison for truncated power-law tail mis-specification
# - Null model: truncated power law on [xmin, xmax] with exponent alpha (fitted by MLE)
# - Alternative: small tail contamination: with prob eps, draw from heavier tail (alpha - delta);
#   otherwise from base tail (alpha). This creates subtle tail mis-specification.
#
# We test with:
#   - KS (Kolmogorov–Smirnov), bootstrap p-values under composite null
#   - AD (Anderson–Darling), bootstrap p-values under composite null
#
# We report empirical power at 5% level over repetitions, across contamination levels.

import numpy as np
import pandas as pd
from math import log, exp
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import kstest
from numpy.random import default_rng
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import json
import hashlib
from pathlib import Path
import threading
import fcntl


rng = default_rng(12345)

# paths
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
bootstrap_dir = repo_root / "bootstrapData"
os.makedirs(bootstrap_dir, exist_ok=True)

# ---------- Bootstrap caching utilities ----------


def _stat_func_name(stat_func):
    return getattr(stat_func, "__name__", repr(stat_func))


# Single-file JSON cache (thread/process safe on POSIX via fcntl)
CACHE_FILE = bootstrap_dir / "bootstrap_cache.json"
LOCK_FILE = bootstrap_dir / "bootstrap_cache.lock"
_CACHE_MEM = None  # in-memory dict keyed by digest -> payload
_CACHE_MEM_MTIME = None
_CACHE_MEM_LOCK = threading.Lock()


def _cache_digest_for_key(key: dict) -> str:
    key_str = json.dumps(key, sort_keys=True)
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]


def _bootstrap_cache_key(x, xmin, xmax, stat_func, B, alpha_level):
    """Create a deterministic dict key for the cache for these args."""
    sum_x = float(np.round(np.sum(x), 12))
    return {
        "sum_x": sum_x,
        "n": int(len(x)),
        "xmin": float(xmin),
        "xmax": float(xmax),
        "B": int(B),
        "alpha_level": float(alpha_level),
        "stat": _stat_func_name(stat_func),
        # room for future versioning if schema changes
        "v": 1,
    }


def _cache_mem_ensure_loaded():
    """Load cache into memory once per process; refresh if file mtime changes when needed."""
    global _CACHE_MEM, _CACHE_MEM_MTIME
    with _CACHE_MEM_LOCK:
        # Already loaded
        if _CACHE_MEM is not None:
            return
        # Initial load (non-blocking read; writer uses atomic replace)
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    _CACHE_MEM = json.load(f)
                    if not isinstance(_CACHE_MEM, dict):
                        _CACHE_MEM = {}
                _CACHE_MEM_MTIME = CACHE_FILE.stat().st_mtime
            except Exception:
                _CACHE_MEM = {}
                _CACHE_MEM_MTIME = None
        else:
            _CACHE_MEM = {}
            _CACHE_MEM_MTIME = None


def _cache_mem_maybe_reload_if_missing(digest: str):
    """If a key is missing, refresh from disk only if the file changed since last load."""
    global _CACHE_MEM, _CACHE_MEM_MTIME
    with _CACHE_MEM_LOCK:
        if _CACHE_MEM is None:
            _cache_mem_ensure_loaded()
            return
        if digest in _CACHE_MEM:
            return
        try:
            mtime = CACHE_FILE.stat().st_mtime if CACHE_FILE.exists() else None
        except Exception:
            mtime = None
        if (
            mtime is not None
            and _CACHE_MEM_MTIME is not None
            and mtime <= _CACHE_MEM_MTIME
        ):
            return  # no changes on disk
        # Reload
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        _CACHE_MEM = data
                        _CACHE_MEM_MTIME = mtime
        except Exception:
            pass


def _bootstrap_cache_lookup(x, xmin, xmax, stat_func, B, alpha_level):
    """Return payload dict if present, else None."""
    _cache_mem_ensure_loaded()
    key = _bootstrap_cache_key(x, xmin, xmax, stat_func, B, alpha_level)
    digest = _cache_digest_for_key(key)
    _cache_mem_maybe_reload_if_missing(digest)
    payload = _CACHE_MEM.get(digest) if _CACHE_MEM is not None else None
    # Minimal validation
    if isinstance(payload, dict) and "pval" in payload and "T_obs" in payload:
        return payload
    return None


def _bootstrap_cache_store(x, xmin, xmax, stat_func, B, alpha_level, payload: dict):
    """Atomically merge `payload` into the single JSON cache file and update memory copy.
    Uses an external lock file with fcntl (POSIX) to guard concurrent writers.
    """
    key = _bootstrap_cache_key(x, xmin, xmax, stat_func, B, alpha_level)
    digest = _cache_digest_for_key(key)

    # Update in-memory first
    _cache_mem_ensure_loaded()
    with _CACHE_MEM_LOCK:
        if _CACHE_MEM is None:
            local_cache = {}
        else:
            local_cache = dict(_CACHE_MEM)
        local_cache[digest] = payload

    # Exclusive lock around read-modify-write of on-disk JSON
    # Lock a separate lockfile so atomic os.replace doesn't drop the lock.
    os.makedirs(bootstrap_dir, exist_ok=True)
    with open(LOCK_FILE, "a+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            disk_cache = {}
            if CACHE_FILE.exists():
                try:
                    with open(CACHE_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            disk_cache = data
                except Exception:
                    disk_cache = {}
            disk_cache[digest] = payload
            tmp_path = str(CACHE_FILE) + f".tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as tf:
                json.dump(disk_cache, tf, ensure_ascii=False, separators=(",", ":"))
                tf.flush()
                os.fsync(tf.fileno())
            os.replace(tmp_path, CACHE_FILE)  # atomic replace
            # refresh in-memory cache and mtime
            with _CACHE_MEM_LOCK:
                _CACHE_MEM.update({digest: payload})
                try:
                    _CACHE_MEM_MTIME = CACHE_FILE.stat().st_mtime
                except Exception:
                    _CACHE_MEM_MTIME = None
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


# ---------- Truncated power-law utilities ----------
def tpl_cdf(x, alpha, xmin, xmax):
    x = np.asarray(x)
    # Handle alpha near 1 separately
    if np.isclose(alpha, 1.0):
        Z = np.log(xmax / xmin)
        Fx = np.clip(np.log(x / xmin) / Z, 0.0, 1.0)
        return Fx
    a = 1.0 - alpha
    num = np.power(x, a) - xmin**a
    den = xmax**a - xmin**a
    Fx = np.clip(num / den, 0.0, 1.0)
    return Fx


def tpl_ppf(u, alpha, xmin, xmax):
    u = np.asarray(u)
    if np.isclose(alpha, 1.0):
        return xmin * np.power((xmax / xmin), u)
    a = 1.0 - alpha
    val = u * (xmax**a - xmin**a) + xmin**a
    return np.power(val, 1.0 / a)


def sample_tpl(n, alpha, xmin, xmax, rng):
    u = rng.random(n)
    return tpl_ppf(u, alpha, xmin, xmax)


# ---------- MLE for truncated power-law exponent ----------
def tpl_negloglik(alpha, x, xmin, xmax):
    # Guard domain
    if alpha <= 0.5 or alpha >= 10.0:
        return np.inf
    n = x.size
    s = np.sum(np.log(x))
    if np.isclose(alpha, 1.0):
        # log-likelihood = -n*log(log(xmax/xmin)) - alpha*sum(log x)
        return n * np.log(np.log(xmax / xmin)) + alpha * s
    a = 1.0 - alpha
    # Normalizing constant: (1 - alpha) / (xmax^(1-alpha) - xmin^(1-alpha))
    logC = np.log(abs(1.0 - alpha)) - np.log(abs(xmax**a - xmin**a))
    return -(n * logC) + alpha * s  # negative log-likelihood


def fit_alpha_mle(x, xmin, xmax, bounds=(0.1, 6.0), xatol=1e-4, maxiter=500):
    """
    Scalar MLE for alpha with configurable bounds (default [0.1, 6.0]).
    """
    res = minimize_scalar(
        tpl_negloglik,
        args=(x, xmin, xmax),
        bounds=bounds,
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )
    return float(res.x)


# ---------- KS statistic (one-sample) ----------
def ks_statistic(x, cdf_func):
    """
    One-sample Kolmogorov–Smirnov statistic using SciPy's kstest.
    We only return the test statistic because p-values under a composite null
    are handled via bootstrap elsewhere in this script.
    `cdf_func` must accept a NumPy array and return an array of CDF values in [0,1].
    """
    res = kstest(x, cdf_func, alternative="two-sided", mode="auto")
    return float(res.statistic)


# ---------- Anderson–Darling statistic for a specified CDF ----------
def ad_statistic(x, cdf_func):
    x_sorted = np.sort(x)
    n = x_sorted.size
    F = np.clip(cdf_func(x_sorted), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) * (np.log(F) + np.log(1.0 - F[::-1])))
    A2 = -n - s / n
    return A2


# ---------- Bootstrap p-values under composite null (refit each bootstrap) ----------
# ---------- Bootstrap p-values under composite null (refit each bootstrap) ----------
def bootstrap_pvalue(x, xmin, xmax, stat_func, B=100, rng=None, alpha_level=0.05):
    """
    Compute bootstrap p-value for a statistic under a composite null by refitting on each bootstrap sample.
    Adds on-disk caching keyed by sum(x) and other parameters. If a matching entry exists, it is returned.
    The cache stores: p_value, T_observed, alpha_hat, timestamp, and the key used.
    """
    # Ensure numpy array for consistent sum and length
    x = np.asarray(x)

    # 1. Check if results are already cached on disk
    cached = _bootstrap_cache_lookup(x, xmin, xmax, stat_func, B, alpha_level)
    if cached is not None:
        return float(cached["pval"]), float(cached["T_obs"])  # fast path

    # 2. Fit power-law exponent (alpha) on observed data
    alpha_hat = fit_alpha_mle(x, xmin, xmax)

    # 3. Compute the observed test statistic
    def cdf_hat(t):
        return tpl_cdf(t, alpha_hat, xmin, xmax)

    T_observed = stat_func(x, cdf_hat)

    # 4. Generate bootstrap samples, refit, and compute test statistic
    n = x.size
    local_rng = rng if rng is not None else default_rng()
    test_statistics_bootstrap = np.empty(B)
    for b in range(B):
        xb = sample_tpl(n, alpha_hat, xmin, xmax, local_rng)
        alpha_b = fit_alpha_mle(xb, xmin, xmax)

        def cdf_b(t, a=alpha_b):
            return tpl_cdf(t, a, xmin, xmax)

        test_statistics_bootstrap[b] = stat_func(xb, cdf_b)


# ---------- Joint bootstrap for KS and AD: reuse samples and warm-start refits ----------
def bootstrap_pvalues_ks_ad(
    x, xmin, xmax, B=100, rng=None, alpha_level=0.05, warm_delta=0.2
):
    """
    Joint bootstrap for KS and AD under the composite null.
    Reuses the same bootstrap samples for both statistics and warm-starts alpha refits
    using a narrow bound around the previous fit: [alpha_prev - warm_delta, alpha_prev + warm_delta],
    clipped to [0.1, 6.0].
    Returns: (pval_ks, pval_ad, T_obs_ks, T_obs_ad)
    """
    x = np.asarray(x)
    n = int(x.size)
    local_rng = rng if rng is not None else default_rng()

    # Fit on observed data
    alpha_hat = fit_alpha_mle(x, xmin, xmax, bounds=(0.1, 6.0))

    # Observed stats use the same fitted CDF
    def cdf_hat(t):
        return tpl_cdf(t, alpha_hat, xmin, xmax)

    T_obs_ks = ks_statistic(x, cdf_hat)
    T_obs_ad = ad_statistic(x, cdf_hat)

    # Bootstrap arrays
    ts_ks = np.empty(int(B))
    ts_ad = np.empty(int(B))

    # Warm-start with observed fit
    alpha_prev = alpha_hat

    for b in range(int(B)):
        xb = sample_tpl(n, alpha_hat, xmin, xmax, local_rng)
        # Warm-start bounds around previous alpha
        lo = max(0.1, alpha_prev - warm_delta)
        hi = min(6.0, alpha_prev + warm_delta)
        if hi <= lo:
            lo, hi = 0.1, 6.0  # fallback to full range if degenerate
        alpha_b = fit_alpha_mle(xb, xmin, xmax, bounds=(lo, hi))

        def cdf_b(t, a=alpha_b):
            return tpl_cdf(t, a, xmin, xmax)

        ts_ks[b] = ks_statistic(xb, cdf_b)
        ts_ad[b] = ad_statistic(xb, cdf_b)

        alpha_prev = alpha_b  # update warm start

    # p-values (upper-tail)
    p_ks = (1 + np.sum(ts_ks >= T_obs_ks)) / (int(B) + 1)
    p_ad = (1 + np.sum(ts_ad >= T_obs_ad)) / (int(B) + 1)
    return p_ks, p_ad, T_obs_ks, T_obs_ad


# ---------- Generate alternative data: tail contamination ----------
def sample_tail_contaminated(n, alpha_base, alpha_tail, eps, xmin, xmax, rng):
    # With prob eps, sample from heavier tail (alpha_tail), else from base (alpha_base)
    mask = rng.random(n) < eps
    x = np.empty(n)
    k = np.sum(mask)
    if k > 0:
        x[mask] = sample_tpl(k, alpha_tail, xmin, xmax, rng)
    if n - k > 0:
        x[~mask] = sample_tpl(n - k, alpha_base, xmin, xmax, rng)
    return x


# ---------- Worker for parallel execution ----------


def _single_run(args):
    (
        eps,
        n,
        xmin,
        xmax,
        alpha_true,
        alpha_tail,
        B,
        alpha_level,
        seed,
    ) = args
    local_rng = default_rng(seed)

    x = sample_tail_contaminated(n, alpha_true, alpha_tail, eps, xmin, xmax, local_rng)
    p_ks, p_ad, _, _ = bootstrap_pvalues_ks_ad(
        x, xmin, xmax, B=B, rng=local_rng, alpha_level=alpha_level, warm_delta=0.2
    )
    return eps, int(p_ks < alpha_level), int(p_ad < alpha_level)


# ---------- Power study ----------
def power_study(
    n=300,
    xmin=1.0,
    xmax=200.0,
    alpha_true=2.2,
    alpha_tail=1.8,  # alpha_tail < alpha_true -> heavier tail
    eps_list=(0.0, 0.05, 0.10),
    n_rep=40,
    B=100,
    alpha_level=0.05,
    rng=None,
    n_jobs=None,
):
    results = []
    total_iters = len(eps_list) * n_rep
    rej_counts = {eps: {"KS": 0, "AD": 0} for eps in eps_list}

    # Prepare task list with deterministic seeding per task
    base_seed = (
        12345 if rng is None else 12345
    )  # keep reproducible; you can wire rng later if desired
    tasks = []
    t = 0
    for eps in eps_list:
        for _ in range(n_rep):
            tasks.append(
                (
                    eps,
                    n,
                    xmin,
                    xmax,
                    alpha_true,
                    alpha_tail,
                    B,
                    alpha_level,
                    base_seed + t,
                )
            )
            t += 1

    if n_jobs is None:
        n_jobs = cpu_count()

    if n_jobs == 1:
        # Serial fallback with tqdm
        with tqdm(total=total_iters, desc="Power study") as pbar:
            for args in tasks:
                eps, rks, rad = _single_run(args)
                rej_counts[eps]["KS"] += rks
                rej_counts[eps]["AD"] += rad
                pbar.update(1)
    else:
        # Parallel execution with a single tqdm over all tasks
        with Pool(processes=n_jobs) as pool:
            for eps, rks, rad in tqdm(
                pool.imap_unordered(_single_run, tasks),
                total=total_iters,
            ):
                rej_counts[eps]["KS"] += rks
                rej_counts[eps]["AD"] += rad

    for eps in eps_list:
        results.append(
            {
                "epsilon_tail_contam": eps,
                "power_KS": rej_counts[eps]["KS"] / n_rep,
                "power_AD": rej_counts[eps]["AD"] / n_rep,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Parameter grids
    n_list = [500, 1000, 5000, 10000, 15000]
    n_rep_list = [500]
    B_list = [2500]

    # Common study settings
    eps_list = np.linspace(0.0, 0.15, 5)  # Tail contamination levels
    xmin = 1e-8
    xmax = 20.0
    alpha_true = 1.0
    alpha_tail = 1.2
    alpha_level = 0.05

    # Ensure output dirs
    plot_path = repo_root / "Plots"
    data_path = repo_root / "Plots" / "Data"
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    # Run all combinations and collect results
    results = []  # list of (label, df)

    # We avoid importing itertools at top; quick local product
    for n in n_list:
        for n_rep in n_rep_list:
            for B in B_list:
                label = f"n={n}, rep={n_rep}, B={B}"
                print(f"\n=== Running study: {label} ===")
                df_power = power_study(
                    n=n,
                    xmin=xmin,
                    xmax=xmax,
                    alpha_true=alpha_true,
                    alpha_tail=alpha_tail,
                    eps_list=eps_list,
                    n_rep=n_rep,
                    B=B,
                    alpha_level=alpha_level,
                    rng=rng,
                    n_jobs=cpu_count(),
                )

                # Save per-combination CSV for later reuse
                safe_label = label.replace(", ", "_").replace("=", "")
                csv_file = data_path / f"power_grid_{safe_label}.csv"
                df_power.to_csv(csv_file, index=False)
                print(f"Saved data to {csv_file}")

                results.append((label + " | KS", "KS", df_power))
                results.append((label + " | AD", "AD", df_power))

    # Plot all lines on a single figure
    fig, ax = plt.subplots()
    for label, stat_name, df_power in results:
        y = df_power["power_KS"] if stat_name == "KS" else df_power["power_AD"]
        marker = "o" if stat_name == "KS" else "s"
        ax.plot(df_power["epsilon_tail_contam"], y, marker=marker, label=label)

    ax.set_xlabel("Tail contamination epsilon")
    ax.set_ylabel(f"Empirical power (alpha = {alpha_level})")
    ax.set_title(
        "Power to detect tail mis-specification (truncated power law)\nAll combinations of n, n_rep, B (both KS and AD)"
    )
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()

    # Save figure once with a descriptive name
    figfile = plot_path / "powerlaw_ks_ad_power_all_combinations.png"
    fig.savefig(figfile, dpi=300)
    print(f"Saved combined figure to {figfile}")

    # Optionally show after saving
    plt.show()
    plt.close(fig)
