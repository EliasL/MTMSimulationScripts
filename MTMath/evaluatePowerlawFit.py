"""Power-law distributions, fitting, and xmin-search helpers.

The fitting classes in this module are intentionally generic: they can fit
any positive event population and can evaluate several xmin-selection
strategies.  For the simulation energy-drop analysis, the standard workflow
is more specific; see the ``Standard power-law energy-drop workflow`` section
in the repository ``ReadMe.md`` and make any departure from it explicit in
the calling script.
"""

import powerlaw
from powerlaw import Distribution, trim_to_range, bisect_map, SUPPORTED_DISTRIBUTIONS
from powerlaw import Truncated_Power_Law as Original_Truncated_Power_Law
import numpy as np
from scipy import special
from numpy import nan
from matplotlib import pyplot as plt
# For checking how many processes are available
import os
import tempfile
import time
import uuid

# For parallelization
import multiprocessing

# So we can ignore this warning while fitting xmin
from scipy.optimize import OptimizeWarning
import warnings
from tqdm import tqdm


POWERLAW_STANDARD_WORKFLOW = r"""
Recommended simulation energy-drop workflow
--------------------------------------------
1. Restrict the event extraction to the post-yield region.
2. Extract Delta E_R and Delta E_S as paired values from the same event
   transitions, in the same order and with the same length.
3. Apply the ``simpleDrop`` rule to Delta E_R.  Call the resulting
   reversible/irreversible classification threshold ``er_det`` and label it
   Delta E_{R,\det}; it is not the final fit cutoff.
4. Transfer that classification to the paired Delta E_S events, then keep
   only finite positive Delta E_S values from the irreversible events.  Fit
   that population, ``es_irrev``, only.
5. Select ``es_xmin_ks`` by evaluating the KS distance at every observed
   candidate in ``es_irrev``, then choose the true global minimum.  Label it
   Delta E_{S,\min}^{KS}.  A coarse/local search is an approximation and
   should be selected deliberately (and marked as such); caching the
   exhaustive search is fine.
6. With ``es_xmin_ks`` fixed, perform the maximum-likelihood fit for
   alpha and lambda.

The low-level ``Fit`` and xmin functions remain deliberately flexible and do
not enforce this sequence.  If a script fits Delta E_R, includes reversible
events, uses a different split (for example Otsu or a slope criterion), or
uses a coarse xmin search, that is an alternative analysis and should be
stated explicitly rather than treated as the standard result.
""".strip()


def get_powerlaw_standard_workflow() -> str:
    """Return the recommended simulation energy-drop analysis recipe."""
    return POWERLAW_STANDARD_WORKFLOW

"""
Currently just templated; doesn't work yet.

Whether to enable parallelization for certain heavy calculations, eg. 
fitting the xmin value.
"""
PARALLEL_ENABLE = False
"""
Currently just templated; doesn't work yet.

This is the number of cores that the library should leave free when doing
certain heavy calculations. For example, if you have 8 cores, and this is
set to 2, then the processing would use (up to) 6 cores.
"""
PARALLEL_UNUSED_CORES = 2


# Let me know if there is a simpler way to get the xmin_distribution with fit values
_EDGE_WARN_RE = r"Fitted parameters are very close to the edge of parameter ranges.*"
_INIT_GUESS_WARN_RE = r"Initial guess is not within the specified bounds"


def _suppress_powerlaw_warnings():
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings(
        "ignore",
        message=_EDGE_WARN_RE,
        category=UserWarning,
        module=r"powerlaw\.distributions",
    )
    warnings.filterwarnings(
        "ignore",
        message=_INIT_GUESS_WARN_RE,
        category=OptimizeWarning,
        module=r"powerlaw\.distributions",
    )


_MEMMAP_CACHE: dict[tuple[str, str, tuple[int, ...]], np.memmap] = {}


def _load_memmap(spec):
    key = (
        spec["memmap_path"],
        spec["dtype"],
        tuple(spec["shape"]),
    )
    if key not in _MEMMAP_CACHE:
        _MEMMAP_CACHE[key] = np.memmap(
            spec["memmap_path"],
            mode="r",
            dtype=np.dtype(spec["dtype"]),
            shape=tuple(spec["shape"]),
        )
    return _MEMMAP_CACHE[key]


def _coerce_param_list(params, parameter_names):
    if not parameter_names:
        return None
    if params is None:
        return None
    if isinstance(params, dict):
        return [params.get(p, nan) for p in parameter_names]
    try:
        seq = list(params)
    except TypeError:
        return None
    if len(seq) == len(parameter_names):
        return seq
    return None


def _fit_xmin_batch_worker(args):
    (
        data_spec,
        xmin_values,
        xmin_distance,
        dist_cls,
        xmax,
        discrete,
        fit_method,
        parameter_ranges,
        parameter_constraints,
        estimate_discrete,
        parameter_names,
        initial_params,
    ) = args

    if isinstance(data_spec, dict) and "memmap_path" in data_spec:
        data = _load_memmap(data_spec)
    else:
        data = data_spec

    initial_params = _coerce_param_list(initial_params, parameter_names)
    distances = []
    params_out = []
    valid_out = []

    with warnings.catch_warnings():
        _suppress_powerlaw_warnings()
        for xmin in xmin_values:
            pl = dist_cls(
                xmin=xmin,
                xmax=xmax,
                discrete=discrete,
                fit_method=fit_method,
                data=data,
                parameters=initial_params,
                parameter_ranges=parameter_ranges,
                parameter_constraints=parameter_constraints,
                parent_Fit=None,
                estimate_discrete=estimate_discrete,
                verbose=0,
            )
            param_vals = [getattr(pl, p, nan) for p in parameter_names]
            distances.append(getattr(pl, xmin_distance))
            params_out.append(param_vals)
            valid_out.append(pl.in_range() and not pl.noise_flag)

    return distances, params_out, valid_out


def _fit_single_xmin_task(args):
    drops, trial_xmin, xmax, dist_name = args
    return Fit(
        data=drops,
        xmin=trial_xmin,
        xmax=xmax,
        xmin_distribution=dist_name,
        verbose=0,
    )


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
        return (x**a) * special.expn(n, x)

    if a > 0.0:
        return special.gamma(a) * special.gammaincc(a, x)

    k = int(np.floor(-a)) + 1
    ap = a + k
    G = special.gamma(ap) * special.gammaincc(ap, x)

    t = ap
    logx = np.log(x)
    for _ in range(k):
        term = np.exp((t - 1.0) * logx - x)
        G = (G - term) / (t - 1.0)
        t -= 1.0

    return G


def dist_from_fit(fit: powerlaw.Fit) -> Distribution:
    dist = getattr(fit, fit.xmin_distribution.name)
    return dist


def _extract_fit_param_vals(fit: powerlaw.Fit):
    parameter_names = list(
        getattr(getattr(fit, "xmin_distribution", None), "parameter_names", [])
    )

    data = getattr(fit, "data", None)
    if data is None or not hasattr(data, "__len__") or len(data) == 0:
        return [np.nan] * len(parameter_names)

    try:
        dist = dist_from_fit(fit)
    except (AttributeError, ValueError):
        return [np.nan] * len(parameter_names)

    return [getattr(dist, p, np.nan) for p in parameter_names]


class Truncated_Power_Law(Original_Truncated_Power_Law):
    # The default generator is very slow. This is faster
    # To do that, we also add a rng object to the distribution.
    # This helps ensure control and repeatability which is important
    # for hashing.

    def generate_random(self, size=1, estimate_discrete=None, rng=None, seed=0):
        """
        Generate random numbers from the theoretical probability distribution.

        This will follow the theoretical distribution, including upper
        and lower limits defined by ``xmin`` or ``xmax``. For example, if
        this function is called from a distribution with a finite value of
        ``xmax``, the generated values will be less than that value. If
        no value is given for ``xmax``, random values will have no upper
        limit.

        For discrete distributions without an approximation method, we
        use numerical inverse transform sampling.

        Parameters
        ----------
        size : tuple or int, optional
            The number of random numbers to generate.

            If a tuple, will be taken as the shape of the array to generate
            where each value is randomly generated according to the theoretical
            distribution.

        estimate_discrete : bool, optional
            For discrete distributions, whether to use a faster approximation of
            the random number generator.

            If ``None``, attempts to inherit the estimate_discrete behavior used
            for fitting from the ``Distribution`` object or the parent ``Fit``
            object, if present. Approximations only exist for some
            distributions (namely the power law). If an approximation does
            not exist, an ``estimate_discrete=True`` setting will not be inherited.

        Returns
        -------
        r : array
            Random numbers drawn from the distribution with shape equal
            to ``size``.
        """

        if rng is not None:
            self.rng = rng
        elif not hasattr(self, "rng"):
            from numpy.random import default_rng

            self.rng = default_rng(seed)

        # For generating random numbers from an arbitrary distribution, we
        # use inverse transform sampling, which involves finding the inverse
        # of the cumulative distribution function, and then
        # evaluating that function for random uniform values in the domain
        # of the CDF, ie. [0, 1].

        # For continuous random numbers we usually don't need to do any
        # approximations, and can just transform according to the specific
        # distribution.
        if not self.discrete:
            # Note that assuming the full range [0, 1] will give unbounded
            # random numbers on the upper side; the distribution functions
            # (_generate_random_continuous) are derived assuming an xmin
            # value, but no xmax value.

            # So for the lower bound we can safely use zero when we don't have
            # an xmax value.
            lower_bound = 0
            upper_bound = 1

            # Truncated power laws are weird because we can't implement
            # the proper inverse CDF (see _generate_random_continuous for
            # more information) so we just set the bounds the 0 and 1 and
            # the rest will be handled by that function.
            # Similarly, power laws with alpha < 1 will handle the bounds
            # in their method as well, so we define a bool for each
            # distribution if it handles the bounds internally or if we
            # need to set them externally.
            if self.xmax and not self.internally_bounded_rng:
                # When we have an xmax value, we need to change the upper
                # bound. This is because the whole cumulative distribution
                # shifts when we have an xmax.

                # You might be tempted to use the self.cdf() function for that,
                # but this function already accounts for xmin and xmax, so it will
                # just give you 0 and 1, which we don't want. What we need is the
                # unadjusted cdf value, since the inverse transform sampling
                # function is derived without being adjusted for xmax.

                # upper_bound = self._cdf_base_function(self.xmax)

                # TODO: There is an issue with lognormal and exponential random
                # number generation, possibly because that inverse cdf is derived
                # differently than the others, but I don't understand why. For
                # lognormal or exponential generation to have proper bounds, you
                # need to somehow change the upper bound to some other value, but
                # I have no idea what that value is; it doesn't seem to be any
                # of the obvious combinations of self._cdf_base_function(self.xmin)
                # and self._cdf_base_function(self.xmax). As such, we just perform
                # a bisect search to find this value for ALL distributions;
                # for everything except exponential and lognormal, the value
                # of this should be identical to self._cdf_base_function(self.xmax).

                # Since this bisect search only needs to perform once per
                # random generation, this doesn't increase computation
                # that much, so long as you aren't generate one number at
                # a time. But of course we should still try to fix this issue.

                # We also have to make sure that our xmax isn't too large;
                # if we have defined a huge xmax when our distribution
                # goes to (nearly) zero much before it reaches this value,
                # we can't actually perform this search (because of
                # numerical precision). In that case, we can just use
                # 1 as an upper bound and call it a day :)

                # 1e-10 is arbitrary
                if 1 - self._cdf_base_function(self.xmax) <= 1e-10:
                    upper_bound = 1

                else:
                    upper_bound = bisect_map(
                        mn=0,
                        mx=1 - 1e-10,
                        function=self._generate_random_continuous,
                        target=self.xmax,
                        tol=1e-8,
                    )

                    # This search might fail for some other mysterious
                    # reason, most likely that the xmax is too large but
                    # the previous if statement didn't quite catch it. So
                    # we should check if it worked, and if it didn't, we
                    # just use 1.
                    if not upper_bound:
                        upper_bound = 1

                    else:
                        # Minus some epsilon since the bisect search isn't perfect
                        upper_bound *= 1 - 1e-5

            uniform_r = self.rng.uniform(lower_bound, upper_bound, size=size)
            r = self._generate_random_continuous(uniform_r)

        # For discrete distributions, we usually have to make some
        # approximation.
        else:
            # Make sure that this distribution supports approximating the
            # continuous distribution with some discrete scheme.
            if estimate_discrete and not hasattr(
                self, "_generate_random_discrete_estimate"
            ):
                raise AttributeError(
                    "This distribution does not have an estimation of the discrete form for generating simulated data. Try the exact form with estimate_discrete=False."
                )

            # If no value for estimate discrete is given, we should decide
            # based on whether the distribution is first able to do this
            # at all, then whether the class has already been passed a
            # value on creation.
            if estimate_discrete is None:
                # We can't estimate discrete is there isn't a function
                # for it.
                if not hasattr(self, "_generate_random_discrete_estimate"):
                    estimate_discrete = False

                # Check the value of self.estimate_discrete.
                elif hasattr(self, "estimate_discrete"):
                    estimate_discrete = self.estimate_discrete

                # Check the value of estimate_discrete for the parent object.
                elif self.parent_Fit:
                    estimate_discrete = self.parent_Fit.estimate_discrete

                # If none of those worked, don't estimate.
                else:
                    estimate_discrete = False

            # Use the approximation method if it's available and
            # desired.
            if estimate_discrete:
                # Note that if we use the approximate discrete method, the
                # upper bound is different from a continuous one, since we
                # are using a different function than _generate_random_continuous.
                # So we first do a search to find the maximum value, ie.
                # the r value such that _generate_random_discrete_estimate(r) = xmax
                if self.xmax:
                    # For the upper limit mx here, we can't use exactly 1
                    # since that would lead to infinity for most distributions.
                    upper_bound = bisect_map(
                        mn=0,
                        mx=1 - 1e-15,
                        function=self._generate_random_discrete_estimate,
                        target=self.xmax
                        - 1,  # -1 to make sure we always generate values under xmax
                        tol=1e-8,
                    )

                else:
                    upper_bound = 1

                # This function takes xmin into account, so we can just
                # use 0 as the lower bound.
                uniform_r = np.random.uniform(0, upper_bound, size=size)

                r = np.array(
                    self._generate_random_discrete_estimate(uniform_r), dtype=np.int64
                )

            else:
                # For each of the uniform values (r), we do the
                # inverse search problem to find the specific value of x
                # where the ccdf is equal to that value of r. The x value
                # is then the random value we return.

                # This does the search on the function ccdf which will
                # automatically account for xmin and xmax, so we can just
                # use plain 0 and 1 for our bounds.
                uniform_r = np.random.uniform(0, 1, size=size)
                r = np.array(
                    [self._double_search_discrete(R) for R in uniform_r.flatten()],
                    dtype=np.int64,
                )

                # Now reshape
                r = r.reshape(size)

        return r

    def _generate_random_continuous(self, r, max_size=1e8, forceGeneration=False):
        """
        Generate samples from the cutoff power-law:

            f(x) ∝ x^(-alpha) * exp(-Lambda * x),    x >= xmin
        where Lambda = 1/lambda in your earlier notation.

        This implementation is a hybrid rejection sampler that dynamically chooses
        between two *mathematically equivalent* proposal–acceptance factorizations
        of the same target density, using a small pilot run to estimate efficiency.

        The target density factorizes as
            f(x) ∝ [x^(-alpha)] * [exp(-Lambda * x)].
        Each branch samples one factor exactly and applies the other as a rejection
        probability.

        (A) "Weak-cutoff" proposal (Pareto):
            Propose g(x) ∝ x^(-alpha) on [xmin, ∞), accept with exp(-Lambda*x).
            - Efficient when the exponential cutoff is weak (xmin*Lambda << 1,
            equivalently lambda >> xmin), because exp(-Lambda*x) ≈ 1 over most
            of the probability mass.

        (B) "Strong-cutoff" proposal (Shifted exponential):
            Propose q(x) = xmin + Exp(scale=1/Lambda), accept with (xmin/x)^alpha.
            - Efficient when the exponential cutoff is strong (xmin*Lambda ≳ 1,
            equivalently lambda ≲ xmin), because proposals concentrate near xmin,
            where (xmin/x)^alpha ≈ 1.

        Crucially, both branches produce *exact samples from the same density*
        f(x) ∝ x^(-alpha) exp(-Lambda x). The "weak" vs "strong" cutoff distinction
        affects only rejection efficiency, not the sampled distribution.

        Notes on correctness:
        - Both branches are standard rejection samplers.
        - The density of accepted samples is proportional to
            proposal(x) × acceptance(x) ∝ x^(-alpha) exp(-Lambda x)
        in either branch.
        - Filling the output array in proposal arrival order is unbiased because
        proposals and accept/reject decisions are i.i.d.; we simply keep the
        first N accepted samples.

        Performance:
        - Preallocates output and fills in vectorized batches.
        - Avoids Python lists, which are a major bottleneck at large N.
        """

        import numpy as np

        size = len(r)
        if size == 0:
            return np.asarray([], dtype=float)

        alpha = float(self.alpha)
        xmin = float(self.xmin)
        Lam = float(self.Lambda)

        if not (xmin > 0.0 and Lam > 0.0):
            raise ValueError("Require xmin > 0 and Lambda > 0.")

        rng = self.rng
        finfo = np.finfo(float)

        # --- Case 1: alpha < 1 -> exact inverse-CDF route (your original logic) ---
        # This path avoids rejection issues and is typically fastest/stablest for alpha < 1.
        if alpha < 1.0:
            from scipy.special import gammainc, gammaincinv

            k = 1.0 - alpha
            theta = 1.0 / Lam
            # If U ~ Uniform(0,1), conditional on X>=xmin:
            #   Fmin = P(Gamma(k,theta) <= xmin) = gammainc(k, xmin/theta)
            #   then sample U' = Fmin + (1-Fmin)*U and invert.
            Fmin = gammainc(k, xmin / theta)
            u = Fmin + (1.0 - Fmin) * np.asarray(r, dtype=float)
            y = gammaincinv(k, u)
            return theta * y

        # --- Rejection-sampling utilities (vectorized, preallocated fill) ---

        def _fill_from_pareto(out, filled, need, batch):
            """
            Propose from Pareto tail g(x) ∝ x^(-alpha), x>=xmin, and accept with exp(-Lam*x).

            Inverse-CDF for Pareto with exponent alpha:
                X = xmin * (1-U)^(-1/(alpha-1)), U ~ Uniform(0,1)
            """
            # Draw U in (0,1); clamp away from exactly 1.0 to avoid inf
            u = rng.random(batch)
            u = np.minimum(u, 1.0 - finfo.eps)

            # Pareto proposal
            x = xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))

            # Accept with probability exp(-Lam*x)
            # Use log test for stability: log(V) < -Lam*x
            logv = np.log(rng.random(batch))
            mask = logv < (-Lam * x)

            k = int(mask.sum())
            if k:
                take = min(k, need)
                out[filled : filled + take] = x[mask][:take]
                filled += take
                need -= take
            return filled, need

        def _fill_from_shifted_exp(out, filled, need, batch):
            """
            Propose from shifted exponential q(x) = xmin + Exp(rate=Lam),
            and accept with (xmin/x)^alpha.

            Since (xmin/x)^alpha ∈ (0,1], a log-test is stable:
                log(V) < alpha*(log(xmin) - log(x))
            """
            # Shifted exponential proposal
            x = xmin + rng.exponential(scale=1.0 / Lam, size=batch)

            # Accept with probability (xmin/x)^alpha using logs
            logv = np.log(rng.random(batch))
            loga = alpha * (np.log(xmin) - np.log(x))
            mask = logv < loga

            k = int(mask.sum())
            if k:
                take = min(k, need)
                out[filled : filled + take] = x[mask][:take]
                filled += take
                need -= take
            return filled, need

        # --- Dynamic choice: estimate acceptance of both proposals via a small pilot ---
        # We do a cheap pilot to estimate acceptance rates r1, r2 for THIS (alpha,xmin,Lam).
        # This is more robust than a fixed threshold on z=xmin*Lam because acceptance depends on alpha too.
        max_size = int(max_size)
        pilot = 2048 if size >= 2048 else max(256, size)
        pilot = min(pilot, max_size)

        # Estimate acceptance for Pareto proposal
        u = rng.random(pilot)
        u = np.minimum(u, 1.0 - finfo.eps)
        x_p = xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))
        # acceptance prob is exp(-Lam*x); average gives acceptance rate estimate
        r_pareto = float(np.mean(np.exp(-Lam * x_p)))

        # Estimate acceptance for shifted exponential proposal
        x_e = xmin + rng.exponential(scale=1.0 / Lam, size=pilot)
        r_exp = float(np.mean((xmin / x_e) ** alpha))

        # Pick the better proposal (higher acceptance => fewer proposals per accepted sample)
        use_pareto = r_pareto >= r_exp

        # --- Main fill loop (vectorized, preallocated) ---
        out = np.empty(size, dtype=float)
        filled = 0
        need = size

        # Batch sizing:
        # For rejection sampling, expected proposals ~ need / acc_rate.
        # We overshoot slightly to reduce loop iterations.
        acc = r_pareto if use_pareto else r_exp
        # Avoid division by zero if acc is extremely tiny
        acc = max(acc, 1e-12)
        overshoot = 1.25

        while need > 0:
            # Propose enough to likely fill most of what's left
            batch = int(np.ceil(overshoot * need / acc))
            if batch > 1000 * max_size and not forceGeneration:
                # If batch is much larger than the largest size we can
                # work with, we give up.
                print("Warning! Cannot generate distribution!")
                return None
            batch = max(1024, batch)  # keep batches large for vectorization
            batch = min(batch, max_size)  # cap memory usage

            if use_pareto:
                filled, need = _fill_from_pareto(out, filled, need, batch)
            else:
                filled, need = _fill_from_shifted_exp(out, filled, need, batch)

        return out

    def _cdf_base_function(self, x):
        x = np.asarray(x, dtype=np.float64)
        s = 1.0 - float(self.alpha)
        lam = float(self.Lambda)
        z = lam * x
        if np.any(z <= 0):
            raise ValueError("Lambda*x must be positive for real-valued Γ(a,x).")

        G = _upper_incomplete_gamma(s, z)
        cdf = 1.0 - G / (lam**s)
        return cdf


def evaluate_xmin(
    drops,
    xmin_values,
    distType: type[Distribution] = Truncated_Power_Law,
    xmax=None,
    parallel=False,
    max_workers=None,
):
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > 0)]
    if drops.size < 3:
        raise ValueError("Need at least three finite positive drops.")
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
        for trial_xmin in xmin_values:
            fit = Fit(
                data=drops,
                xmin=trial_xmin,
                xmax=xmax,
                xmin_distribution=distType.name,
                verbose=0,
            )
            test_fits.append(fit)

    return test_fits


def evaluate_xmin_distances(
    drops,
    xmin_values,
    distType: type[Distribution] = Truncated_Power_Law,
    xmax=None,
    parallel=False,
    max_workers=None,
    initial_params=None,
    progress=False,
    progress_label="xmin",
):
    """Fit a candidate grid in batches and return lightweight diagnostics."""
    drops = np.asarray(drops, dtype=float)
    drops = drops[np.isfinite(drops) & (drops > 0)]
    xmin_values = np.asarray(xmin_values, dtype=float)
    if drops.size < 3:
        raise ValueError("Need at least three finite positive drops.")
    if xmin_values.size == 0:
        raise ValueError("Need at least one xmin candidate.")

    parameter_names = list(distType.parameter_names)
    if parallel and xmin_values.size > 1:
        from concurrent.futures import ProcessPoolExecutor

        max_workers = max_workers or max(
            1, (os.cpu_count() or 1) - PARALLEL_UNUSED_CORES
        )
        batches = np.array_split(
            xmin_values,
            min(xmin_values.size, 4 * max_workers),
        )
    else:
        max_workers = 1
        batches = [xmin_values]

    tasks = [
        (
            drops,
            batch,
            "D",
            distType,
            xmax,
            False,
            "likelihood",
            None,
            None,
            None,
            parameter_names,
            initial_params,
        )
        for batch in batches
    ]
    if parallel and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(
                tqdm(
                    executor.map(_fit_xmin_batch_worker, tasks),
                    total=len(tasks),
                    desc=f"{progress_label}: xmin batches",
                )
            )
    else:
        if not progress:
            batch_results = [_fit_xmin_batch_worker(tasks[0])]
        else:
            batch_results = []
            total = len(xmin_values)
            started = time.perf_counter()
            for index, xmin in enumerate(xmin_values, start=1):
                elapsed = time.perf_counter() - started
                rate = (index - 1) / elapsed if index > 1 else 0.0
                eta = (total - index + 1) / rate if rate > 0 else float("nan")
                print(
                    f"{progress_label}: starting candidate {index}/{total}, "
                    f"xmin={xmin:.6g}, elapsed={elapsed:.1f}s, "
                    f"ETA={eta:.1f}s",
                    flush=True,
                )
                task = list(tasks[0])
                task[1] = np.asarray([xmin], dtype=float)
                batch_results.append(_fit_xmin_batch_worker(tuple(task)))
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed > 0 else 0.0
                eta = (total - index) / rate if rate > 0 else float("nan")
                print(
                    f"{progress_label}: finished candidate {index}/{total}, "
                    f"elapsed={elapsed:.1f}s, ETA={eta:.1f}s",
                    flush=True,
                )

    distances = np.concatenate(
        [np.asarray(result[0], dtype=float) for result in batch_results]
    )
    param_vals = [values for result in batch_results for values in result[1]]
    valid = np.concatenate(
        [np.asarray(result[2], dtype=bool) for result in batch_results]
    )
    return distances, param_vals, valid


def _find_post_drop_knee(x, D, recovery_frac=0.25, use_recovery_frac=False):
    logx = np.log10(x)
    dip_d1 = np.gradient(D, logx)
    dip_d2 = np.gradient(dip_d1, logx)

    if not np.isfinite(dip_d1).any():
        return float(x[0]), dip_d1, dip_d2

    idx_drop = int(np.nanargmin(dip_d1))
    if idx_drop >= len(x) - 1:
        return float(x[idx_drop]), dip_d1, dip_d2

    recovery_threshold = None
    if use_recovery_frac:
        steepest_slope = dip_d1[idx_drop]
        recovery_threshold = recovery_frac * steepest_slope

    for i in range(max(idx_drop + 1, 1), len(x) - 1):
        if not (
            np.isfinite(dip_d2[i - 1])
            and np.isfinite(dip_d2[i])
            and np.isfinite(dip_d2[i + 1])
        ):
            continue
        is_local_max_d2 = dip_d2[i] >= dip_d2[i - 1] and dip_d2[i] >= dip_d2[i + 1]
        if not is_local_max_d2:
            continue
        if use_recovery_frac:
            if not np.isfinite(dip_d1[i]) or dip_d1[i] < recovery_threshold:
                continue
        return float(x[i]), dip_d1, dip_d2

    if use_recovery_frac:
        recovered = np.flatnonzero(
            (np.arange(len(x)) > idx_drop)
            & np.isfinite(dip_d1)
            & (dip_d1 >= recovery_threshold)
        )
        if recovered.size:
            return float(x[int(recovered[0])]), dip_d1, dip_d2

    right_side = np.arange(idx_drop + 1, len(x))
    finite_d2_right = right_side[np.isfinite(dip_d2[right_side])]
    if finite_d2_right.size:
        best_i = int(finite_d2_right[np.nanargmax(dip_d2[finite_d2_right])])
        return float(x[best_i]), dip_d1, dip_d2

    return float(x[idx_drop]), dip_d1, dip_d2


def my_dip_find_xmin(
    drops,
    debug=False,
    samples_per_decade=30,
    recovery_frac=0.25,
    use_recovery_frac=False,
    **kwargs,
):
    log_min_xmin = np.log10(min(drops))
    log_max_xmin = np.log10(max(drops))

    decades = max(log_max_xmin - log_min_xmin, 0.0)
    nr_first_evaluation = int(max(20, np.ceil(decades * samples_per_decade)))
    # Coarse grid (downsampled) to find dip_d1
    coarse_xmin_values = np.logspace(log_min_xmin, log_max_xmin, nr_first_evaluation)
    fits = evaluate_xmin(drops, coarse_xmin_values, **kwargs)
    distances = np.asarray([f.D for f in fits], dtype=float)

    x = coarse_xmin_values
    D = distances
    mask = np.isfinite(x) & np.isfinite(D) & (x > 0)
    if mask.sum() < 3:
        warnings.warn("Not enough finite KS distances to find a local minimum.")
        return np.nan
    x = x[mask]
    D = D[mask]
    xmin, dip_d1, dip_d2 = _find_post_drop_knee(
        x,
        D,
        recovery_frac=recovery_frac,
        use_recovery_frac=use_recovery_frac,
    )
    param_vals = [_extract_fit_param_vals(fit) for fit in fits]
    param_vals = [vals for vals, keep in zip(param_vals, mask) if keep]
    xmin_fitting_results = {
        "distances": D,
        "param_vals": param_vals,
        "xmins": x,
    }
    if debug:
        plt.plot(x, D)
        plt.xscale("log")
        plt.vlines([x[np.nanargmin(dip_d1)]], min(D), max(D), colors="tab:green", linestyles="--")
        plt.vlines([xmin], min(D), max(D), colors="tab:red", linestyles=":")
        plt.show()

    return xmin, xmin_fitting_results


def my_coarse_find_xmin(drops, debug=False, samples_per_decade=30, **kwargs):
    log_min_xmin = np.log10(min(drops))
    log_max_xmin = np.log10(max(drops))

    decades = log_max_xmin - log_min_xmin
    nr_first_evaluation = int(max(20, np.ceil(decades * samples_per_decade)))
    coarse_xmin_values = np.logspace(log_min_xmin, log_max_xmin, nr_first_evaluation)
    fits = evaluate_xmin(drops, coarse_xmin_values, **kwargs)
    distances = np.asarray([f.D for f in fits], dtype=float)

    x = coarse_xmin_values
    D = distances
    mask = np.isfinite(x) & np.isfinite(D) & (x > 0)
    if mask.sum() < 3:
        warnings.warn("Not enough finite KS distances to find a local minimum.")
        return np.nan
    x = x[mask]
    D = D[mask]
    
    xmin_local_min = float(x[int(np.nanargmin(D))])

    param_vals = [_extract_fit_param_vals(fit) for fit in fits]
    param_vals = [vals for vals, keep in zip(param_vals, mask) if keep]
    xmin_fitting_results = {
        "distances": D,
        "param_vals": param_vals,
        "xmins": x,
    }
    if debug:
        plt.plot(x, D)
        plt.xscale("log")
        plt.vlines([xmin_local_min], min(D), max(D))
        plt.show()

    return xmin_local_min, xmin_fitting_results

class Fit(powerlaw.Fit):
    def __init__(
        self,
        data,
        discrete=False,
        xmin=None,
        xmax=None,
        fit_method="likelihood",
        fast_xmin=False,
        estimate_discrete=None,
        discrete_normalization="round",
        sigma_threshold=None,
        initial_parameters=None,
        parameter_ranges=None,
        parameter_constraints=None,
        xmin_distance="D",
        xmin_distribution="power_law",
        verbose=1,
        xmin_samples_per_decade=30,
        parallel_xmin=False,
    ):
        # The upstream powerlaw fit can emit OptimizeWarning/UserWarning during init.
        # Suppress them here so callers don't need to wrap every Fit construction.
        with warnings.catch_warnings():
            _suppress_powerlaw_warnings()

            # We need to replace the old truncated power law with our new one so we use
            # the faster generator
            SUPPORTED_DISTRIBUTIONS["truncated_power_law"] = Truncated_Power_Law
            self.fast_xmin = fast_xmin
            self.xmin_samples_per_decade = float(xmin_samples_per_decade)
            if self.xmin_samples_per_decade <= 0:
                raise ValueError("xmin_samples_per_decade must be positive.")
            self.parallel_xmin = bool(parallel_xmin)
            super().__init__(
                data,
                discrete,
                xmin,
                xmax,
                fit_method,
                estimate_discrete,
                discrete_normalization,
                sigma_threshold,
                initial_parameters,
                parameter_ranges,
                parameter_constraints,
                xmin_distance,
                xmin_distribution,
                verbose,
            )
            data_in_range = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
            if len(data_in_range) > 2:
                dist = getattr(self, xmin_distribution)
                setattr(self, xmin_distance, getattr(dist, xmin_distance))
            else:
                setattr(self, xmin_distance, np.nan)

    def find_xmin(self):
        if not self.fast_xmin:

            return super().find_xmin()

        xmin, xmin_fitting_results = my_dip_find_xmin(
            self.data,
            samples_per_decade=self.xmin_samples_per_decade,
            parallel=self.parallel_xmin,
        )

        # Set the Fit's xmin to the optimal xmin
        self.xmin = xmin

        self.xmin_fitting_results = xmin_fitting_results

        # Update the fitting CDF given the new xmin, in case other objects, like
        # Distributions, want to use it for fitting (like if they do KS fitting)
        self.fitting_cdf_bins, self.fitting_cdf = self.cdf()

        return self.xmin

    def _get_cache_path(self, cache_dir, data, nr_sets):
        import hashlib
        from numpy import asarray

        # build a stable cache key from the *pre-fit* state + confidence
        # include a hash of the data to invalidate if data changes
        data_bytes = asarray(data).tobytes()
        h = hashlib.sha1()
        h.update(data_bytes)
        data_sig = h.hexdigest()

        cache_key = (
            f"{self.xmin_distribution.name}"
            f"_len={len(data)}_data={data_sig}"
            f"_nr_sets={nr_sets}_xmin={self.xmin}_xmax={self.xmax}"
            f"_discrete={self.discrete}_fit_method={self.fit_method}"
        )
        cache_name = hashlib.sha1(cache_key.encode("utf-8")).hexdigest() + ".json"

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, cache_name)
        return cache_path

    @staticmethod
    def _fit_on_sample(
        sample,
        dist: type[Distribution],
        xmin,
        xmax,
        discrete,
        fit_method,
        parameter_names=None,
    ):
        with warnings.catch_warnings():
            _suppress_powerlaw_warnings()
            m = dist(
                data=sample,
                xmin=xmin,
                xmax=xmax,
                discrete=discrete,
                fit_method=fit_method,
            )
        if parameter_names:
            param_vals = [getattr(m, p, nan) for p in parameter_names]
        else:
            param_vals = [getattr(m, m.parameter1_name)]
        return m.D, param_vals

    def bootstrap_ks_samples(
        self,
        data=None,
        nr_sets=None,
        confidence=0.01,
        parallel=True,
        max_synthetic_samples=5e6,
        tqdmDesc="",
        show_progress=True,
        max_workers=None,
        return_params=False,
    ):
        """
        Generate bootstrap KS distances for the current fit.

        Returns
        -------
        D_vals : np.ndarray
            Bootstrap KS distances.
        param_vals : list | None
            Bootstrap parameter estimates if return_params=True.
        """
        from numpy import asarray
        from functools import partial
        from tqdm import tqdm

        if data is None:
            data = self.data
        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
        if len(data) <= 2:
            return asarray([]), ([] if return_params else None)

        if nr_sets is None:
            nr_sets = max(1, int(1 / (4 * confidence**2)))  # At least one set
        nr_sets = int(nr_sets)

        # Get distribution (Let me know if there is a better way to do this)
        dist = dist_from_fit(self)

        samples_per_set = len(data)
        if max_synthetic_samples is None:
            max_synthetic_samples = samples_per_set * nr_sets
        max_synthetic_samples = int(max_synthetic_samples)
        sets_per_batch = max(1, max_synthetic_samples // max(1, samples_per_set))

        parameter_names = list(getattr(self.xmin_distribution, "parameter_names", []))
        worker = partial(
            self._fit_on_sample,
            dist=self.xmin_distribution,
            xmin=self.xmin,
            xmax=self.xmax,
            discrete=self.discrete,
            fit_method=self.fit_method,
            parameter_names=parameter_names if return_params else None,
        )

        D_vals = []
        param_vals = [] if return_params else None
        remaining = nr_sets

        with warnings.catch_warnings():
            _suppress_powerlaw_warnings()
            if parallel:
                from concurrent.futures import ProcessPoolExecutor

                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    progress = tqdm(
                        total=nr_sets, desc=tqdmDesc, disable=not show_progress
                    )
                    while remaining > 0:
                        batch_sets = min(sets_per_batch, remaining)
                        synthetic_data = dist.generate_random(
                            samples_per_set * batch_sets
                        )
                        if synthetic_data is None:
                            progress.close()
                            break
                        if batch_sets == 1:
                            synthetic_sets = [synthetic_data]
                        else:
                            synthetic_sets = synthetic_data.reshape(
                                batch_sets, samples_per_set
                            )
                        results = list(ex.map(worker, synthetic_sets))
                        progress.update(len(results))
                        batch_D, batch_params = zip(*results)
                        D_vals.extend(batch_D)
                        if return_params:
                            param_vals.extend(batch_params)
                        remaining -= batch_sets
                    progress.close()
            else:
                progress = tqdm(total=nr_sets, desc=tqdmDesc, disable=not show_progress)
                while remaining > 0:
                    batch_sets = min(sets_per_batch, remaining)
                    synthetic_data = dist.generate_random(samples_per_set * batch_sets)
                    if synthetic_data is None:
                        progress.close()
                        break
                    if batch_sets == 1:
                        synthetic_sets = [synthetic_data]
                    else:
                        synthetic_sets = synthetic_data.reshape(
                            batch_sets, samples_per_set
                        )
                    results = [worker(s) for s in synthetic_sets]
                    progress.update(len(results))
                    batch_D, batch_params = zip(*results)
                    D_vals.extend(batch_D)
                    if return_params:
                        param_vals.extend(batch_params)
                    remaining -= batch_sets
                progress.close()

        return asarray(D_vals), param_vals

    def evaluate_fit(
        self,
        data=None,
        confidence=0.01,
        parallel=True,
        max_workers=None,
        use_cache=True,
        cache_dir=".eval_cache",
        tqdmDesc="",
        max_synthetic_samples=5e6,
    ):
        """
        Evaluate fit, optionally parallel, and cache computed scalars on disk via JSON.

        Notes
        -----
        - If `use_cache` is True and a cache hit occurs, this loads cached values and
          updates *only* the computed attributes in-place:
          `p`, `p_std`, `alpha_mean`, `alpha_std`, plus any cached parameter means/stds.
        - The cache key depends on key params and a SHA-1 hash of the (trimmed) data.
        """
        from numpy import mean, std, asarray

        if data is None:
            data = self.data
        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
        if len(data) <= 2:
            print(f"Not enough data to evaluate fit ({len(data)} point(s))")
            self.p = -0.01
            self.p_std = 0
            self.alpha_mean = 0
            self.alpha_std = 0
            return self.p, self.alpha_mean, self.alpha_std

        # --- compute number of synthetic sets
        nr_sets = max(1, int(1 / (4 * confidence**2)))  # At least one set

        # --- try cache (JSON of computed scalars only)
        cache_path = None
        if use_cache:
            import os
            import json

            cache_path = self._get_cache_path(cache_dir, data, nr_sets)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)

                    # update only the computed attributes
                    self.p = float(payload["p"])
                    self.p_std = float(payload["p_std"])
                    self.alpha_mean = float(payload["alpha_mean"])
                    self.alpha_std = float(payload["alpha_std"])

                    parameter_names = list(
                        getattr(self.xmin_distribution, "parameter_names", [])
                    )
                    param_means = payload.get("param_means")
                    param_stds = payload.get("param_stds")
                    if not parameter_names:
                        cache_complete = True
                    else:
                        cache_complete = bool(param_means) and bool(param_stds)
                        if cache_complete:
                            cache_complete = all(
                                (name in param_means and name in param_stds)
                                for name in parameter_names
                            )

                    if cache_complete and parameter_names:
                        for name, mean_val in param_means.items():
                            setattr(self, f"{name}_mean", float(mean_val))
                        for name, std_val in param_stds.items():
                            setattr(self, f"{name}_std", float(std_val))
                        if "alpha" in param_means:
                            self.alpha_mean = float(param_means["alpha"])
                        if "alpha" in param_stds:
                            self.alpha_std = float(param_stds["alpha"])

                        return self.p, self.alpha_mean, self.alpha_std
                except Exception:
                    # fall through to recompute if loading fails
                    pass

        parameter_names = list(getattr(self.xmin_distribution, "parameter_names", []))

        D_vals, param_vals = self.bootstrap_ks_samples(
            data=data,
            nr_sets=nr_sets,
            parallel=parallel,
            max_workers=max_workers,
            max_synthetic_samples=max_synthetic_samples,
            tqdmDesc=tqdmDesc,
            show_progress=True,
            return_params=True,
        )

        if len(D_vals):
            # Get distribution (Let me know if there is a better way to do this)
            dist = dist_from_fit(self)
            # --- aggregate results
            self.p = mean(asarray(D_vals) >= dist.D)
            self.p_std = confidence

            param_means = {}
            param_stds = {}
            if parameter_names and param_vals:
                by_name = {name: [] for name in parameter_names}
                for params in param_vals:
                    if not params:
                        continue
                    for name, val in zip(parameter_names, params):
                        if np.isfinite(val):
                            by_name[name].append(val)
                for name, vals in by_name.items():
                    if vals:
                        param_means[name] = float(mean(vals))
                        param_stds[name] = float(std(vals))
                        setattr(self, f"{name}_mean", param_means[name])
                        setattr(self, f"{name}_std", param_stds[name])

            if "alpha" in param_means:
                self.alpha_mean = param_means["alpha"]
                self.alpha_std = param_stds.get("alpha", 0.0)
            else:
                # fallback if alpha wasn't part of parameter_names
                alpha_vals = [p[0] for p in param_vals if p]
                self.alpha_mean = mean(alpha_vals) if alpha_vals else 0.0
                self.alpha_std = std(alpha_vals) if alpha_vals else 0.0
        else:
            print("Fit not evaluated.")
            self.p = -0.01
            self.p_std = 0
            self.alpha_mean = 0
            self.alpha_std = 0

        # --- save computed scalars if requested (atomic JSON write)
        if use_cache and cache_path is not None:
            try:
                import json
                import os
                import tempfile

                payload = {
                    "p": float(self.p),
                    "p_std": float(self.p_std),
                    "alpha_mean": float(self.alpha_mean),
                    "alpha_std": float(self.alpha_std),
                }
                if "param_means" not in locals():
                    param_means = {}
                if "param_stds" not in locals():
                    param_stds = {}
                if param_means or param_stds:
                    payload["param_means"] = param_means
                    payload["param_stds"] = param_stds

                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

                # atomic write: write to temp file then replace
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    delete=False,
                    dir=os.path.dirname(cache_path) or ".",
                    prefix=".tmp_",
                    suffix=".json",
                ) as tf:
                    json.dump(payload, tf, ensure_ascii=False)
                    tf.flush()
                    os.fsync(tf.fileno())
                    tmp_path = tf.name

                os.replace(tmp_path, cache_path)
            except Exception as e:
                # don't fail the computation if persistence fails
                print(e)
                pass

        return self.p, self.alpha_mean, self.alpha_std
