import powerlaw
from powerlaw import Distribution, trim_to_range, bisect_map
from powerlaw import Truncated_Power_Law as Original_Truncated_Power_Law
import numpy as np


# Let me know if there is a simpler way to get the xmin_distribution with fit values
def dist_from_fit(fit: powerlaw.Fit) -> Distribution:
    dist = getattr(fit, fit.xmin_distribution.name)
    return dist


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

    def _generate_random_continuous(self, r, max_size=1e8):
        """
        Unbiased batched rejection sampler for
            f(x) ∝ x^(-alpha) * exp(-Lambda * x), x >= xmin.

        Strategy
        --------
        - If alpha < 1: use the exact inverse-CDF sampler (same as before).
        - Else: propose in batches from xmin + Exp(scale=1/Lambda), accept with
        a(x) = (xmin / x)^alpha. We *adaptively* size each batch using an
        online estimate of the acceptance rate, and finally take the first
        size accepted samples in (proposal) arrival order.

        Notes
        -----
        - Taking the *first N accepted* is distribution-preserving (no bias),
        because acceptance decisions are i.i.d. Bernoulli given the proposals,
        and selection is independent of sample values.
        - Do **not** sort or otherwise pick accepted samples by value.
        """
        # Heavy tail ⇒ inverse-CDF path is faster/numerically stable.
        if self.alpha < 1:
            from scipy.special import gammainc, gammaincinv

            k = 1.0 - self.alpha
            theta = 1.0 / self.Lambda

            Fmin = gammainc(k, self.xmin / theta)
            u = Fmin + (1.0 - Fmin) * r
            y = gammaincinv(k, u)
            x = theta * y
            return x

        accepted = []
        size = len(r)
        need = size

        # Start with a conservative acceptance-rate guess to avoid too-small batches.
        # We'll update this estimate on the fly.
        # Heuristic: acceptance ≈ (xmin*Lambda) / (xmin*Lambda + alpha)
        r_est = (self.xmin * self.Lambda) / (self.xmin * self.Lambda + self.alpha)
        from numpy import ceil, log, asarray

        while need > 0:
            # Over-propose by 1/r_est with a overshoot factor
            overshoot = 1.5
            n_prop = max(32, int(ceil(overshoot * need / r_est)))
            if n_prop / max_size > 10000:
                # If n_prop is much larger than the largest size we can
                # work with, we give up.
                print("Warning! Cannot generate distribution!")
                return None
            n_prop = min(n_prop, int(max_size))

            # Proposals from the exponential tail anchored at xmin
            prop = self.xmin + self.rng.exponential(
                scale=1.0 / self.Lambda, size=n_prop
            )

            # Accept with probability (xmin / x)^alpha
            # Using log-space for numerical stability
            log_u = log(self.rng.random(n_prop))
            log_prop = self.alpha * (log(self.xmin) - log(prop))
            mask = log_u < log_prop

            # Append in *arrival order* to preserve unbiasedness
            if mask.any():
                accepted.extend(prop[mask])

            # Update remaining count
            got = len(accepted)
            need = size - got

        # Take the first N accepted in arrival order (clip any extra)
        if len(accepted) > size:
            accepted = accepted[:size]
        return asarray(accepted, dtype=float)


class Fit(powerlaw.Fit):
    def __init__(
        self,
        data,
        discrete=False,
        xmin=None,
        xmax=None,
        fit_method="likelihood",
        estimate_discrete=None,
        discrete_normalization="round",
        sigma_threshold=None,
        initial_parameters=None,
        parameter_ranges=None,
        parameter_constraints=None,
        xmin_distance="D",
        xmin_distribution="power_law",
        verbose=1,
    ):
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
        # We need to replace the old truncated power law with our new one so we use
        # the faster generator
        self.supported_distributions["truncated_power_law"] = Truncated_Power_Law

    def _get_cache_path(self, cache_dir, data, nr_sets):
        import os
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
    def _fit_on_sample(sample, dist: Distribution, xmin, xmax, discrete, fit_method):
        m = dist(
            data=sample, xmin=xmin, xmax=xmax, discrete=discrete, fit_method=fit_method
        )
        return m.D, getattr(m, m.parameter1_name)

    def evaluate_fit(
        self,
        data=None,
        confidence=0.01,
        parallel=True,
        use_cache=True,
        cache_dir=".eval_cache",
    ):
        """
        Evaluate fit, optionally parallel, and cache computed scalars on disk via JSON.

        Notes
        -----
        - If `use_cache` is True and a cache hit occurs, this loads cached values and
          updates *only* the computed attributes in-place:
          `p`, `p_std`, `alpha_mean`, `alpha_std`.
        - The cache key depends on key params and a SHA-1 hash of the (trimmed) data.
        """
        from numpy import array_split, mean, std, asarray
        from functools import partial
        from tqdm import tqdm

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

                    return self.p, self.alpha_mean, self.alpha_std
                except Exception:
                    # fall through to recompute if loading fails
                    pass

        # Get distribution (Let me know if there is a better way to do this)
        dist = dist_from_fit(self)
        # --- no (usable) cache: compute
        # We add a print statement here since this generation can be slow
        print("Generating synthetic data...")
        synthetic_data = dist.generate_random(len(data) * nr_sets)
        if synthetic_data is None:
            print("Fit not evaluated.")
            self.p = -0.01
            self.p_std = 0
            # keep both mean and std; fix the original overwrite bug
            self.alpha_mean = 0
            self.alpha_std = 0
            return self.p, self.alpha_mean, self.alpha_std

        synthetic_sets = array_split(synthetic_data, nr_sets)

        worker = partial(
            self._fit_on_sample,
            dist=self.xmin_distribution,
            xmin=self.xmin,
            xmax=self.xmax,
            discrete=self.discrete,
            fit_method=self.fit_method,
        )

        if parallel:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor() as ex:
                results = list(
                    tqdm(ex.map(worker, synthetic_sets), total=len(synthetic_sets))
                )

        else:
            results = [worker(s) for s in tqdm(synthetic_sets)]

        # --- aggregate results
        D_vals, alpha_vals = zip(*results)
        # ensure vectorized compare
        dist = dist_from_fit(self)
        self.p = mean(asarray(D_vals) >= dist.D)

        # keep both mean and std; fix the original overwrite bug
        self.alpha_mean = mean(alpha_vals)
        self.alpha_std = std(alpha_vals)
        # a conservative bound for p-uncertainty (optional; keep if you use it elsewhere)
        self.p_std = confidence

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
