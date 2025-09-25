# The MIT License (MIT)
#
# Copyright (c) 2013-2021 Jeff Alstott
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# as described in https://docs.python.org/2/library/functions.html#print
from __future__ import print_function
import sys

__version__ = "1.5"


class Distribution(object):
    """
    An abstract class for theoretical probability distributions. Can be created
    with particular parameter values, or fitted to a dataset. Fitting is
    by maximum likelihood estimation by default.

    Parameters
    ----------
    xmin : int or float, optional
        The data value beyond which distributions should be fitted. If
        None an optimal one will be calculated.
    xmax : int or float, optional
        The maximum value of the fitted distributions.
    discrete : boolean, optional
        Whether the distribution is discrete (integers).

    data : list or array, optional
        The data to which to fit the distribution. If provided, the fit will
        be created at initialization.
    fit_method : "Likelihood" or "KS", optional
        Method for fitting the distribution. "Likelihood" is maximum Likelihood
        estimation. "KS" is minimial distance estimation using The
        Kolmogorov-Smirnov test.

    parameters : tuple or list, optional
        The parameters of the distribution. Will be overridden if data is
        given or the fit method is called.
    parameter_range : dict, optional
        Dictionary of valid parameter ranges for fitting. Formatted as a
        dictionary of parameter names ('alpha' and/or 'sigma') and tuples
        of their lower and upper limits (ex. (1.5, 2.5), (None, .1)
    initial_parameters : tuple or list, optional
        Initial values for the parameter in the fitting search.

    discrete_approximation : "round", "xmax" or int, optional
        If the discrete form of the theoeretical distribution is not known,
        it can be estimated. One estimation method is "round", which sums
        the probability mass from x-.5 to x+.5 for each data point. The other
        option is to calculate the probability for each x from 1 to N and
        normalize by their sum. N can be "xmax" or an integer.

    """

    def __init__(
        self,
        data=None,
        xmin=1,
        xmax=None,
        discrete=False,
        fit_method="Likelihood",
        parameters=None,
        parameter_range=None,
        initial_parameters=None,
        discrete_approximation="round",
        **kwargs,
    ):
        self.xmin = xmin
        self.xmax = xmax
        self.discrete = discrete
        self.fit_method = fit_method
        self.discrete_approximation = discrete_approximation

        self.parameter1 = None
        self.parameter2 = None
        self.parameter3 = None
        self.parameter1_name = None
        self.parameter2_name = None
        self.parameter3_name = None

        if parameters is not None:
            self.parameters(parameters)

        if parameter_range:
            self.parameter_range(parameter_range)

        if initial_parameters:
            self._given_initial_parameters(initial_parameters)

        if data is None:
            raise ValueError("Data must be provided.")
        self.fit(data)

    def trim_to_range(self, data):
        return trim_to_range(data, xmin=self.xmin, xmax=self.xmax)

    def fit(self, data, suppress_output=False):
        """
        Fits the parameters of the distribution to the data. Uses options set
        at initialization.
        """
        data = self.trim_to_range(data)

        if self.fit_method == "Likelihood":

            def fit_function(params):
                self.parameters(params)
                return -sum(self.loglikelihoods(data))
        elif self.fit_method == "KS":

            def fit_function(params):
                self.parameters(params)
                self.KS()
                return self.D

        from scipy.optimize import fmin

        (
            parameters,
            negative_loglikelihood,
            iter,
            funcalls,
            warnflag,
        ) = fmin(
            lambda params: fit_function(params),
            self.initial_parameters(data),
            full_output=1,
            disp=False,
        )
        self.parameters(parameters)
        if not self.in_range():
            self.noise_flag = True
        else:
            self.noise_flag = False
        if self.noise_flag and not suppress_output:
            print("No valid fits found.", file=sys.stderr)
        self.loglikelihood = -negative_loglikelihood
        self.KS(data)

    def KS(self, data):
        """
        Returns the Kolmogorov-Smirnov distance D between the distribution and
        the data. Also sets the properties D+, D-, V (the Kuiper testing
        statistic), and Kappa (1 + the average difference between the
        theoretical and empirical distributions).

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.
        """
        if len(data) < 2:
            print("Not enough data. Returning nan", file=sys.stderr)
            from numpy import nan

            self.D = nan
            self.D_plus = nan
            self.D_minus = nan
            self.Kappa = nan
            self.V = nan
            self.Asquare = nan
            return self.D

        bins, Actual_CDF = cdf(data)

        Theoretical_CDF = self.cdf(bins)

        CDF_diff = Theoretical_CDF - Actual_CDF

        self.D_plus = CDF_diff.max()
        self.D_minus = -1.0 * CDF_diff.min()

        from numpy import mean, argmax, argmin

        # Indices of maxima/minima
        self.D_plus_index = int(argmax(CDF_diff))
        self.D_minus_index = int(argmin(CDF_diff))

        # Overall D is whichever is larger, so pick matching index
        if self.D_plus >= self.D_minus:
            self.D = self.D_plus
            self.D_index = self.D_plus_index
            self.D_x = data[self.D_index]
        else:
            self.D = self.D_minus
            self.D_index = self.D_minus_index
            self.D_x = data[self.D_index]

        self.Kappa = 1 + mean(CDF_diff)

        self.V = self.D_plus + self.D_minus
        self.Asquare = sum(
            ((CDF_diff**2) / (Theoretical_CDF * (1 - Theoretical_CDF) + 1e-12))[1:]
        )
        return self.D

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
            f"{self.__class__.__name__}"
            f"_len={len(data)}_data={data_sig}"
            f"_nr_sets={nr_sets}_xmin={self.xmin}_xmax={self.xmax}"
            f"_discrete={self.discrete}_fit_method={self.fit_method}"
        )
        cache_name = hashlib.sha1(cache_key.encode("utf-8")).hexdigest() + ".joblib"

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, cache_name)
        return cache_path

    @staticmethod
    def _fit_on_sample(sample, cls, xmin, xmax, discrete, fit_method):
        m = cls(
            data=sample, xmin=xmin, xmax=xmax, discrete=discrete, fit_method=fit_method
        )
        return m.D, getattr(m, m.parameter1_name)

    def evaluate_fit(
        self,
        data,
        confidence: float = 0.01,
        parallel: bool = False,
        use_cache: bool = True,
        cache_dir: str = ".joblib_cache",
        n_jobs: int = -1,
    ):
        """
        Evaluate fit, optionally parallel, and persist *self* on disk via joblib.

        Notes
        -----
        - If `use_cache` is True and a cache hit occurs, this loads the saved object
        and updates `self` in-place (self.__dict__.update(...)).
        - The cache key depends on class name, key params, and a SHA-1 hash of data.
        """
        # --- minimal imports used in both paths
        from numpy import array_split, mean, std, asarray
        from functools import partial

        data = self.trim_to_range(data)

        # --- compute number of synthetic sets
        nr_sets = max(1, int(1 / (4 * confidence**2)))  # At least one set

        # --- try cache
        cache_path = None
        if use_cache:
            import os

            cache_path = self._get_cache_path(cache_dir, data, nr_sets)
            if os.path.exists(cache_path):
                import joblib

                try:
                    cached_self = joblib.load(cache_path)
                    # update our current instance to the cached state
                    self.__dict__.update(cached_self.__dict__)
                    # return cached outputs (assumes these attributes were set previously)
                    return self.p, self.alpha_mean, self.alpha_std
                except Exception:
                    # fall through to recompute if loading fails
                    pass

        # --- no (usable) cache: compute
        synthetic_data = self.generate_random(len(data) * nr_sets)
        synthetic_sets = array_split(synthetic_data, nr_sets)

        worker = partial(
            self._fit_on_sample,
            cls=self.__class__,
            xmin=self.xmin,
            xmax=self.xmax,
            discrete=self.discrete,
            fit_method=self.fit_method,
        )

        if parallel:
            # Some users might not have access to joblib
            from concurrent.futures import ProcessPoolExecutor

            try:
                from tqdm import tqdm
            except ImportError:

                def tqdm(doNothing, total=None):
                    return doNothing

            with ProcessPoolExecutor() as ex:
                results = list(
                    tqdm(ex.map(worker, synthetic_sets), total=len(synthetic_sets))
                )

        else:
            results = []
            total = len(synthetic_sets)
            for i, s in enumerate(synthetic_sets, 1):
                results.append(worker(s))
                print(f"{i}/{total}", end="\r")
            print("         ", end="\r")

        # --- aggregate results
        D_vals, alpha_vals = zip(*results)
        # ensure vectorized compare

        self.p = mean(asarray(D_vals) >= self.D)

        # keep both mean and std; fix the original overwrite bug
        self.alpha_mean = mean(alpha_vals)
        self.alpha_std = std(alpha_vals)
        # a conservative bound for p-uncertainty (optional; keep if you use it elsewhere)
        self.p_std = confidence

        # --- persist *self* if requested
        if use_cache and cache_path is not None:
            try:
                import joblib

                joblib.dump(self, cache_path)
            except Exception as e:
                # don't fail the computation if persistence fails
                print(e)
                pass

        return self.p, self.alpha_mean, self.alpha_std

    def ccdf(self, data=None, survival=True):
        """
        The complementary cumulative distribution function (CCDF) of the
        theoretical distribution. Calculated for the values given in data
        within xmin and xmax, if present.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.
        survival : bool, optional
            Whether to calculate a CDF (False) or CCDF (True).
            True by default.

        Returns
        -------
        X : array
            The sorted, unique values in the data.
        probabilities : array
            The portion of the data that is less than or equal to X.
        """
        return self.cdf(data=data, survival=survival)

    def cdf(self, data=None, survival=False):
        """
        The cumulative distribution function (CDF) of the theoretical
        distribution. Calculated for the values given in data within xmin and
        xmax, if present.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.
        survival : bool, optional
            Whether to calculate a CDF (False) or CCDF (True).
            False by default.

        Returns
        -------
        X : array
            The sorted, unique values in the data.
        probabilities : array
            The portion of the data that is less than or equal to X.
        """
        data = self.trim_to_range(data)
        n = len(data)
        from sys import float_info

        if not self.in_range():
            from numpy import tile

            return tile(10**float_info.min_10_exp, n)

        if self._cdf_xmin == 1:
            # If cdf_xmin is 1, it means we don't have the numerical accuracy to
            # calculate this tail. So we make everything 1, indicating
            # we're at the end of the tail. Such an xmin should be thrown
            # out by the KS test.
            from numpy import ones

            CDF = ones(n)
            return CDF

        CDF = self._cdf_base_function(data) - self._cdf_xmin

        norm = 1 - self._cdf_xmin
        if self.xmax:
            norm = norm - (1 - self._cdf_base_function(self.xmax))

        CDF = CDF / norm

        if survival:
            CDF = 1 - CDF

        possible_numerical_error = False
        from numpy import isnan, min

        if isnan(min(CDF)):
            print("'nan' in fit cumulative distribution values.", file=sys.stderr)
            possible_numerical_error = True
        # if 0 in CDF or 1 in CDF:
        #    print("0 or 1 in fit cumulative distribution values.", file=sys.stderr)
        #    possible_numerical_error = True
        if possible_numerical_error:
            print(
                "Likely underflow or overflow error: the optimal fit for this distribution gives values that are so extreme that we lack the numerical precision to calculate them.",
                file=sys.stderr,
            )
        return CDF

    @property
    def _cdf_xmin(self):
        return self._cdf_base_function(self.xmin)

    def pdf(self, data=None):
        """
        Returns the probability density function (normalized histogram) of the
        theoretical distribution for the values in data within xmin and xmax,
        if present.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.

        Returns
        -------
        probabilities : array
        """

        n = len(data)
        from sys import float_info

        if not self.in_range():
            from numpy import tile

            return tile(10**float_info.min_10_exp, n)

        if not self.discrete:
            f = self._pdf_base_function(data)
            C = self._pdf_continuous_normalizer
            likelihoods = f * C
        else:
            if self._pdf_discrete_normalizer:
                f = self._pdf_base_function(data)
                C = self._pdf_discrete_normalizer
                likelihoods = f * C
            elif self.discrete_approximation == "round":
                lower_data = data - 0.5
                upper_data = data + 0.5
                # Temporarily expand xmin and xmax to be able to grab the extra bit of
                # probability mass beyond the (integer) values of xmin and xmax
                # Note this is a design decision. One could also say this extra
                # probability "off the edge" of the distribution shouldn't be included,
                # and that implementation is retained below, commented out. Note, however,
                # that such a cliff means values right at xmin and xmax have half the width to
                # grab probability from, and thus are lower probability than they would otherwise
                # be. This is particularly concerning for values at xmin, which are typically
                # the most likely and greatly influence the distribution's fit.
                self.xmin -= 0.5
                if self.xmax:
                    self.xmax += 0.5
                # Clean data for invalid values before handing to cdf, which will purge them
                # lower_data[lower_data<self.xmin] +=.5
                # if self.xmax:
                #    upper_data[upper_data>self.xmax] -=.5
                likelihoods = self.cdf(upper_data) - self.cdf(lower_data)
                self.xmin += 0.5
                if self.xmax:
                    self.xmax -= 0.5
            else:
                if self.discrete_approximation == "xmax":
                    upper_limit = self.xmax
                else:
                    upper_limit = self.discrete_approximation
                #            from mpmath import exp
                from numpy import arange

                X = arange(self.xmin, upper_limit + 1)
                PDF = self._pdf_base_function(X)
                PDF = (PDF / sum(PDF)).astype(float)
                likelihoods = PDF[(data - self.xmin).astype(int)]
        likelihoods[likelihoods == 0] = 10**float_info.min_10_exp
        return likelihoods

    @property
    def _pdf_continuous_normalizer(self):
        C = 1 - self._cdf_xmin
        if self.xmax:
            C -= 1 - self._cdf_base_function(self.xmax + 1)
        C = 1.0 / C
        return C

    @property
    def _pdf_discrete_normalizer(self):
        return False

    def parameter_range(self, r, initial_parameters=None):
        """
        Set the limits on the range of valid parameters to be considered while
        fitting.

        Parameters
        ----------
        r : dict
            A dictionary of the parameter range. Restricted parameter
            names are keys, and with tuples of the form (lower_bound,
            upper_bound) as values.
        initial_parameters : tuple or list, optional
            Initial parameter values to start the fitting search from.
        """
        from types import FunctionType

        if isinstance(r, FunctionType):
            self._in_given_parameter_range = r
        else:
            self._range_dict = r

        if initial_parameters:
            self._given_initial_parameters = initial_parameters

    def in_range(self):
        """
        Whether the current parameters of the distribution are within the range
        of valid parameters.
        """
        try:
            r = self._range_dict
            result = True
            for k in r.keys():
                # For any attributes we've specificed, make sure we're above the lower bound
                # and below the lower bound (if they exist). This must be true of all of them.
                lower_bound, upper_bound = r[k]
                if upper_bound is not None:
                    result *= getattr(self, k) < upper_bound
                if lower_bound is not None:
                    result *= getattr(self, k) > lower_bound
            return result
        except AttributeError:
            try:
                in_range = self._in_given_parameter_range(self)
            except AttributeError:
                in_range = self._in_standard_parameter_range()
        return bool(in_range)

    def initial_parameters(self, data):
        """
        Return previously user-provided initial parameters or, if never
        provided,  calculate new ones. Default initial parameter estimates are
        unique to each theoretical distribution.
        """
        try:
            return self._given_initial_parameters
        except AttributeError:
            return self._initial_parameters(data)

    def likelihoods(self, data):
        """
        The likelihoods of the observed data from the theoretical distribution.
        Another name for the probabilities or probability density function.
        """
        return self.pdf(data)

    def loglikelihoods(self, data):
        """
        The logarithm of the likelihoods of the observed data from the
        theoretical distribution.
        """
        from numpy import log

        return log(self.likelihoods(data))

    def plot_ccdf(self, data=None, ax=None, survival=True, **kwargs):
        """
        Plots the complementary cumulative distribution function (CDF) of the
        theoretical distribution for the values given in data within xmin and
        xmax, if present. Plots to a new figure or to axis ax if provided.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.
        ax : matplotlib axis, optional
            The axis to which to plot. If None, a new figure is created.
        survival : bool, optional
            Whether to plot a CDF (False) or CCDF (True). True by default.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """
        return self.plot_cdf(data, ax=ax, survival=survival, **kwargs)

    def plot_cdf(self, data=None, ax=None, survival=False, **kwargs):
        """
        Plots the cumulative distribution function (CDF) of the
        theoretical distribution for the values given in data within xmin and
        xmax, if present. Plots to a new figure or to axis ax if provided.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.
        ax : matplotlib axis, optional
            The axis to which to plot. If None, a new figure is created.
        survival : bool, optional
            Whether to plot a CDF (False) or CCDF (True). False by default.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """

        from numpy import unique

        bins = unique(self.trim_to_range(data))
        CDF = self.cdf(bins, survival=survival)
        if not ax:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        ax.plot(bins, CDF, **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        return ax

    def plot_pdf(self, data=None, ax=None, trim_data=True, **kwargs):
        """
        Plots the probability density function (PDF) of the
        theoretical distribution for the values given in data within xmin and
        xmax, if present. Plots to a new figure or to axis ax if provided.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.
        ax : matplotlib axis, optional
            The axis to which to plot. If None, a new figure is created.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """

        from numpy import unique

        if trim_data:
            bins = unique(self.trim_to_range(data))
        else:
            bins = unique(data)
        PDF = self.pdf(bins)
        from numpy import nan

        PDF[PDF == 0] = nan
        if not ax:
            import matplotlib.pyplot as plt

            plt.plot(bins, PDF, **kwargs)
            ax = plt.gca()
        else:
            ax.plot(bins, PDF, **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        return ax

    def generate_random(self, n=1, estimate_discrete=None, rng=None, seed=0):
        """
        Generates random numbers from the theoretical probability distribution.
        If xmax is present, it is currently ignored.

        Parameters
        ----------
        n : int or float
            The number of random numbers to generate
        estimate_discrete : boolean
            For discrete distributions, whether to use a faster approximation of
            the random number generator. Approximations only
            exist for some distributions (namely the power law). If an
            approximation does not exist an estimate_discrete setting of True
            will not be inherited.

        Returns
        -------
        r : array
            Random numbers drawn from the distribution
        """
        from numpy import array

        if rng is not None:
            self.rng = rng
        elif not hasattr(self, "rng"):
            from numpy.random import default_rng

            self.rng = default_rng(seed)

        r = self.rng.uniform(0, 1, n)
        if not self.discrete:
            x = self._generate_random_continuous(r)
        else:
            if estimate_discrete and not hasattr(
                self, "_generate_random_discrete_estimate"
            ):
                raise AttributeError(
                    "This distribution does not have an "
                    "estimation of the discrete form for generating simulated "
                    "data. Try the exact form with estimate_discrete=False."
                )
            if estimate_discrete is None:
                if not hasattr(self, "_generate_random_discrete_estimate"):
                    estimate_discrete = False
                elif hasattr(self, "estimate_discrete"):
                    estimate_discrete = self.estimate_discrete
                else:
                    estimate_discrete = False
            if estimate_discrete:
                x = self._generate_random_discrete_estimate(r)
            else:
                x = array([self._double_search_discrete(R) for R in r], dtype="float")
        return x

    def _double_search_discrete(self, r):
        # Find a range from x1 to x2 that our random probability fits between
        x2 = int(self.xmin)
        while self.ccdf(data=[x2]) >= (1 - r):
            x1 = x2
            x2 = 2 * x1
        # Use binary search within that range to find the exact answer, up to
        # the limit of being between two integers.
        x = bisect_map(x1, x2, self.ccdf, 1 - r)
        return x


class Power_Law(Distribution):
    def __init__(self, estimate_discrete=True, **kwargs):
        self.estimate_discrete = estimate_discrete
        Distribution.__init__(self, **kwargs)

    def parameters(self, params):
        self.alpha = params[0]
        self.parameter1 = self.alpha
        self.parameter1_name = "alpha"

    @property
    def name(self):
        return "power_law"

    @property
    def sigma(self):
        # Only is calculable after self.fit is started, when the number of data points is
        # established
        from numpy import sqrt

        return (self.alpha - 1) / sqrt(self.n)

    def _in_standard_parameter_range(self):
        return self.alpha > 0

    def fit(self, data=None):
        data = self.trim_to_range(data)
        self.n = len(data)
        from numpy import log, sum

        if not self.discrete and not self.xmax:
            self.alpha = 1 + (self.n / sum(log(data / self.xmin)))
            if not self.in_range():
                Distribution.fit(self, data, suppress_output=True)
            self.KS(data)
        elif self.discrete and self.estimate_discrete and not self.xmax:
            self.alpha = 1 + (self.n / sum(log(data / (self.xmin - 0.5))))
            if not self.in_range():
                Distribution.fit(self, data, suppress_output=True)
            self.KS(data)
        else:
            Distribution.fit(self, data, suppress_output=True)

        # parameters is not run unless we use the Distribution.fit function
        self.parameters([self.alpha])

        if not self.in_range():
            self.noise_flag = True
        else:
            self.noise_flag = False

    def _initial_parameters(self, data):
        from numpy import log, sum

        return 1 + len(data) / sum(log(data / (self.xmin)))

    def _cdf_base_function(self, x):
        if self.discrete:
            from scipy.special import zeta

            CDF = 1 - zeta(self.alpha, x)
        else:
            # Can this be reformulated to not reference xmin? Removal of the probability
            # before xmin and after xmax is handled in Distribution.cdf(), so we don't
            # strictly need this element. It doesn't hurt, for the moment.
            CDF = 1 - (x / self.xmin) ** (-self.alpha + 1)
        return CDF

    def _pdf_base_function(self, x):
        return x**-self.alpha

    @property
    def _pdf_continuous_normalizer(self):
        return (self.alpha - 1) * self.xmin ** (self.alpha - 1)

    @property
    def _pdf_discrete_normalizer(self):
        C = 1.0 - self._cdf_xmin
        if self.xmax:
            C -= 1 - self._cdf_base_function(self.xmax + 1)
        C = 1.0 / C
        return C

    def _generate_random_continuous(self, r):
        return self.xmin * (1 - r) ** (-1 / (self.alpha - 1))

    def _generate_random_discrete_estimate(self, r):
        x = (self.xmin - 0.5) * (1 - r) ** (-1 / (self.alpha - 1)) + 0.5
        from numpy import around

        return around(x)


class Exponential(Distribution):
    def parameters(self, params):
        self.Lambda = params[0]
        self.parameter1 = self.Lambda
        self.parameter1_name = "lambda"

    @property
    def name(self):
        return "exponential"

    def _initial_parameters(self, data):
        from numpy import mean

        return 1 / mean(data)

    def _in_standard_parameter_range(self):
        return self.Lambda > 0

    def _cdf_base_function(self, x):
        from numpy import exp

        CDF = 1 - exp(-self.Lambda * x)
        return CDF

    def _pdf_base_function(self, x):
        from numpy import exp

        return exp(-self.Lambda * x)

    @property
    def _pdf_continuous_normalizer(self):
        from numpy import exp

        return self.Lambda * exp(self.Lambda * self.xmin)

    @property
    def _pdf_discrete_normalizer(self):
        from numpy import exp

        C = (1 - exp(-self.Lambda)) * exp(self.Lambda * self.xmin)
        if self.xmax:
            Cxmax = (1 - exp(-self.Lambda)) * exp(self.Lambda * self.xmax)
            C = 1.0 / C - 1.0 / Cxmax
            C = 1.0 / C
        return C

    def pdf(self, data=None):
        if not self.discrete and self.in_range() and not self.xmax:
            data = self.trim_to_range(data)
            from numpy import exp

            #        likelihoods = exp(-Lambda*data)*\
            #                Lambda*exp(Lambda*xmin)
            likelihoods = self.Lambda * exp(self.Lambda * (self.xmin - data))
            # Simplified so as not to throw a nan from infs being divided by each other
            from sys import float_info

            likelihoods[likelihoods == 0] = 10**float_info.min_10_exp
        else:
            likelihoods = Distribution.pdf(self, data)
        return likelihoods

    def loglikelihoods(self, data=None):
        if not self.discrete and self.in_range() and not self.xmax:
            data = self.trim_to_range(data)
            from numpy import log

            #        likelihoods = exp(-Lambda*data)*\
            #                Lambda*exp(Lambda*xmin)
            loglikelihoods = log(self.Lambda) + (self.Lambda * (self.xmin - data))
            # Simplified so as not to throw a nan from infs being divided by each other
            from sys import float_info

            loglikelihoods[loglikelihoods == 0] = log(10**float_info.min_10_exp)
        else:
            loglikelihoods = Distribution.loglikelihoods(self, data)
        return loglikelihoods

    def _generate_random_continuous(self, r):
        from numpy import log

        return self.xmin - (1 / self.Lambda) * log(1 - r)


class Stretched_Exponential(Distribution):
    def parameters(self, params):
        self.Lambda = params[0]
        self.parameter1 = self.Lambda
        self.parameter1_name = "lambda"
        self.beta = params[1]
        self.parameter2 = self.beta
        self.parameter2_name = "beta"

    @property
    def name(self):
        return "stretched_exponential"

    def _initial_parameters(self, data):
        from numpy import mean

        return (1 / mean(data), 1)

    def _in_standard_parameter_range(self):
        return self.Lambda > 0 and self.beta > 0

    def _cdf_base_function(self, x):
        from numpy import exp

        CDF = 1 - exp(-((self.Lambda * x) ** self.beta))
        return CDF

    def _pdf_base_function(self, x):
        from numpy import exp

        return ((x * self.Lambda) ** (self.beta - 1)) * exp(
            -((self.Lambda * x) ** self.beta)
        )

    @property
    def _pdf_continuous_normalizer(self):
        from numpy import exp

        C = self.beta * self.Lambda * exp((self.Lambda * self.xmin) ** self.beta)
        return C

    @property
    def _pdf_discrete_normalizer(self):
        return False

    def pdf(self, data=None):
        if not self.discrete and self.in_range() and not self.xmax:
            data = self.trim_to_range(data)
            from numpy import exp

            likelihoods = (
                (data * self.Lambda) ** (self.beta - 1)
                * self.beta
                * self.Lambda
                * exp(
                    (self.Lambda * self.xmin) ** self.beta
                    - (self.Lambda * data) ** self.beta
                )
            )
            # Simplified so as not to throw a nan from infs being divided by each other
            from sys import float_info

            likelihoods[likelihoods == 0] = 10**float_info.min_10_exp
        else:
            likelihoods = Distribution.pdf(self, data)
        return likelihoods

    def loglikelihoods(self, data=None):
        if not self.discrete and self.in_range() and not self.xmax:
            data = self.trim_to_range(data)
            from numpy import log

            loglikelihoods = (
                log((data * self.Lambda) ** (self.beta - 1) * self.beta * self.Lambda)
                + (self.Lambda * self.xmin) ** self.beta
                - (self.Lambda * data) ** self.beta
            )
            # Simplified so as not to throw a nan from infs being divided by each other
            from sys import float_info
            from numpy import inf

            loglikelihoods[loglikelihoods == -inf] = log(10**float_info.min_10_exp)
        else:
            loglikelihoods = Distribution.loglikelihoods(self, data)
        return loglikelihoods

    def _generate_random_continuous(self, r):
        from numpy import log

        #        return ( (self.xmin**self.beta) -
        #            (1/self.Lambda) * log(1-r) )**(1/self.beta)
        return (1 / self.Lambda) * (
            (self.Lambda * self.xmin) ** self.beta - log(1 - r)
        ) ** (1 / self.beta)


class Truncated_Power_Law(Distribution):
    def parameters(self, params):
        self.alpha = params[0]
        self.parameter1 = self.alpha
        self.parameter1_name = "alpha"
        self.Lambda = params[1]
        self.parameter2 = self.Lambda
        self.parameter2_name = "lambda"

    @property
    def name(self):
        return "truncated_power_law"

    def _initial_parameters(self, data):
        from numpy import log, sum, mean

        alpha = 1 + len(data) / sum(log(data / (self.xmin)))
        Lambda = 1 / mean(data)
        return (alpha, Lambda)

    def _in_standard_parameter_range(self):
        return self.Lambda > 0 and self.alpha > 0

    def _cdf_base_function(self, x):
        from mpmath import gammainc
        from numpy import vectorize

        gammainc = vectorize(gammainc)

        CDF = (gammainc(1 - self.alpha, self.Lambda * x)).astype(
            "float"
        ) / self.Lambda ** (1 - self.alpha)
        CDF = 1 - CDF
        return CDF

    def _pdf_base_function(self, x):
        from numpy import exp

        return x ** (-self.alpha) * exp(-self.Lambda * x)

    @property
    def _pdf_continuous_normalizer(self):
        from mpmath import gammainc

        C = self.Lambda ** (1 - self.alpha) / float(
            gammainc(1 - self.alpha, self.Lambda * self.xmin)
        )
        return C

    @property
    def _pdf_discrete_normalizer(self):
        if 0:
            return False
        from mpmath import lerchphi
        from mpmath import exp  # faster /here/ than numpy.exp

        C = float(
            exp(self.xmin * self.Lambda)
            / lerchphi(exp(-self.Lambda), self.alpha, self.xmin)
        )
        if self.xmax:
            Cxmax = float(
                exp(self.xmax * self.Lambda)
                / lerchphi(exp(-self.Lambda), self.alpha, self.xmax)
            )
            C = 1.0 / C - 1.0 / Cxmax
            C = 1.0 / C
        return C

    def pdf(self, data=None):
        if not self.discrete and self.in_range() and False:
            data = self.trim_to_range(data)
            from numpy import exp
            from mpmath import gammainc

            #        likelihoods = (data**-alpha)*exp(-Lambda*data)*\
            #                (Lambda**(1-alpha))/\
            #                float(gammainc(1-alpha,Lambda*xmin))
            likelihoods = self.Lambda ** (1 - self.alpha) / (
                data**self.alpha
                * exp(self.Lambda * data)
                * gammainc(1 - self.alpha, self.Lambda * self.xmin)
            ).astype(float)
            # Simplified so as not to throw a nan from infs being divided by each other
            from sys import float_info

            likelihoods[likelihoods == 0] = 10**float_info.min_10_exp
        else:
            likelihoods = Distribution.pdf(self, data)
        return likelihoods

    def _old_generate_random_continuous(self, r):
        def helper(r):
            from numpy import log
            from numpy.random import rand

            while 1:
                x = self.xmin - (1 / self.Lambda) * log(1 - r)
                p = (x / self.xmin) ** -self.alpha
                if rand() < p:
                    return x
                r = rand()

        from numpy import array

        return array([helper(r_) for r_ in r])

    def _generate_random_continuous(self, r):
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
        # Heuristic: acceptance ≈ (xmin*Lambda) / (xmin*Lambda + alpha), clipped.
        r_est = min(
            0.8,
            max(
                0.05, (self.xmin * self.Lambda) / (self.xmin * self.Lambda + self.alpha)
            ),
        )
        from numpy import ceil, log, asarray

        while need > 0:
            # Over-propose by 1/r_est with a small safety factor.
            overshoot = 1.15
            n_prop = max(32, int(ceil(overshoot * need / r_est)))

            # Proposals from the exponential tail anchored at xmin
            prop = self.xmin + self.rng.exponential(
                scale=1.0 / self.Lambda, size=n_prop
            )

            # Accept with probability (xmin / x)^alpha
            # Using log-space for numerical stability
            log_u = log(self.rng.random(n_prop))
            mask = log_u < self.alpha * (log(self.xmin) - log(prop))

            # Append in *arrival order* to preserve unbiasedness
            if mask.any():
                accepted.extend(prop[mask])

            # Update remaining count
            got = min(len(accepted), size)  # guard against overflow
            need = size - got

            # Update acceptance rate estimate (EMA to stabilize)
            # instantaneous rate:
            inst_rate = mask.mean()
            # avoid divide-by-zero; keep r_est within sensible bounds
            if inst_rate > 0:
                r_est = 0.7 * r_est + 0.3 * float(inst_rate)
                r_est = min(0.95, max(0.02, r_est))

        # Take the first N accepted in arrival order (clip any extra)
        if len(accepted) > size:
            accepted = accepted[:size]
        return asarray(accepted, dtype=float)


class Lognormal(Distribution):
    def parameters(self, params):
        self.mu = params[0]
        self.parameter1 = self.mu
        self.parameter1_name = "mu"

        self.sigma = params[1]
        self.parameter2 = self.sigma
        self.parameter2_name = "sigma"

    @property
    def name(self):
        return "lognormal"

    def pdf(self, data=None):
        """
        Returns the probability density function (normalized histogram) of the
        theoretical distribution for the values in data within xmin and xmax,
        if present.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.

        Returns
        -------
        probabilities : array
        """

        data = self.trim_to_range(data)
        n = len(data)
        from sys import float_info
        from numpy import tile

        if not self.in_range():
            return tile(10**float_info.min_10_exp, n)

        if not self.discrete:
            f = self._pdf_base_function(data)
            C = self._pdf_continuous_normalizer
            if C > 0:
                likelihoods = f / C
            else:
                likelihoods = tile(10**float_info.min_10_exp, n)
        else:
            if self._pdf_discrete_normalizer:
                f = self._pdf_base_function(data)
                C = self._pdf_discrete_normalizer
                likelihoods = f * C
            elif self.discrete_approximation == "round":
                likelihoods = self._round_discrete_approx(data)
            else:
                if self.discrete_approximation == "xmax":
                    upper_limit = self.xmax
                else:
                    upper_limit = self.discrete_approximation
                #            from mpmath import exp
                from numpy import arange

                X = arange(self.xmin, upper_limit + 1)
                PDF = self._pdf_base_function(X)
                PDF = (PDF / sum(PDF)).astype(float)
                likelihoods = PDF[(data - self.xmin).astype(int)]
        likelihoods[likelihoods == 0] = 10**float_info.min_10_exp
        return likelihoods

    def _round_discrete_approx(self, data):
        """
        This function reformulates the calculation to avoid underflow errors
        with the erf function. As implemented, erf(x) quickly approaches 1
        while erfc(x) is more accurate. Since erfc(x) = 1 - erf(x),
        calculations can be written using erfc(x)
        """
        import numpy as np
        import scipy.special as ss

        """ Temporarily expand xmin and xmax to be able to grab the extra bit of
        probability mass beyond the (integer) values of xmin and xmax
        Note this is a design decision. One could also say this extra
        probability "off the edge" of the distribution shouldn't be included,
        and that implementation is retained below, commented out. Note, however,
        that such a cliff means values right at xmin and xmax have half the width to
        grab probability from, and thus are lower probability than they would otherwise
        be. This is particularly concerning for values at xmin, which are typically
        the most likely and greatly influence the distribution's fit.
        """
        lower_data = data - 0.5
        upper_data = data + 0.5
        self.xmin -= 0.5
        if self.xmax:
            self.xmax += 0.5

        # revised calculation written to avoid underflow errors
        arg1 = (np.log(lower_data) - self.mu) / (np.sqrt(2) * self.sigma)
        arg2 = (np.log(upper_data) - self.mu) / (np.sqrt(2) * self.sigma)
        likelihoods = 0.5 * (ss.erfc(arg1) - ss.erfc(arg2))
        if not self.xmax:
            norm = 0.5 * ss.erfc(
                (np.log(self.xmin) - self.mu) / (np.sqrt(2) * self.sigma)
            )
        else:
            # may still need to be fixed
            norm = -self._cdf_xmin + self._cdf_base_function(self.xmax)
        self.xmin += 0.5
        if self.xmax:
            self.xmax -= 0.5

        return likelihoods / norm

    def cdf(self, data=None, survival=False):
        """
        The cumulative distribution function (CDF) of the lognormal
        distribution. Calculated for the values given in data within xmin and
        xmax, if present. Calculation was reformulated to avoid underflow
        errors

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.
        survival : bool, optional
            Whether to calculate a CDF (False) or CCDF (True).
            False by default.

        Returns
        -------
        X : array
            The sorted, unique values in the data.
        probabilities : array
            The portion of the data that is less than or equal to X.
        """
        from numpy import log, sqrt
        import scipy.special as ss

        data = self.trim_to_range(data)
        n = len(data)
        from sys import float_info

        if not self.in_range():
            from numpy import tile

            return tile(10**float_info.min_10_exp, n)

        val_data = (log(data) - self.mu) / (sqrt(2) * self.sigma)
        val_xmin = (log(self.xmin) - self.mu) / (sqrt(2) * self.sigma)
        CDF = 0.5 * (ss.erfc(val_xmin) - ss.erfc(val_data))

        norm = 0.5 * ss.erfc(val_xmin)
        if self.xmax:
            # TO DO: Improve this line further for better numerical accuracy?
            norm = norm - (1 - self._cdf_base_function(self.xmax))

        CDF = CDF / norm

        if survival:
            CDF = 1 - CDF

        possible_numerical_error = False
        from numpy import isnan, min

        if isnan(min(CDF)):
            print("'nan' in fit cumulative distribution values.", file=sys.stderr)
            possible_numerical_error = True
        # if 0 in CDF or 1 in CDF:
        #    print("0 or 1 in fit cumulative distribution values.", file=sys.stderr)
        #    possible_numerical_error = True
        if possible_numerical_error:
            print(
                "Likely underflow or overflow error: the optimal fit for this distribution gives values that are so extreme that we lack the numerical precision to calculate them.",
                file=sys.stderr,
            )
        return CDF

    def _initial_parameters(self, data):
        from numpy import mean, std, log

        logdata = log(data)
        return (mean(logdata), std(logdata))

    def _in_standard_parameter_range(self):
        # The standard deviation can't be negative
        return self.sigma > 0

    def _cdf_base_function(self, x):
        from numpy import sqrt, log
        from scipy.special import erf

        return 0.5 + (0.5 * erf((log(x) - self.mu) / (sqrt(2) * self.sigma)))

    def _pdf_base_function(self, x):
        from numpy import exp, log

        return (1.0 / x) * exp(-((log(x) - self.mu) ** 2) / (2 * self.sigma**2))

    @property
    def _pdf_continuous_normalizer(self):
        from mpmath import erfc

        #        from scipy.special import erfc
        from scipy.constants import pi
        from numpy import sqrt, log

        C = erfc((log(self.xmin) - self.mu) / (sqrt(2) * self.sigma)) / sqrt(
            2 / (pi * self.sigma**2)
        )
        return float(C)

    @property
    def _pdf_discrete_normalizer(self):
        return False

    def _generate_random_continuous(self, r):
        from numpy import exp, sqrt, log, frompyfunc
        from mpmath import erf, erfinv

        # This is a long, complicated function broken into parts.
        # We use mpmath to maintain numerical accuracy as we run through
        # erf and erfinv, until we get to more sane numbers. Thanks to
        # Wolfram Alpha for producing the appropriate inverse of the CCDF
        # for me, which is what we need to calculate these things.
        erfinv = frompyfunc(erfinv, 1, 1)
        Q = erf((log(self.xmin) - self.mu) / (sqrt(2) * self.sigma))
        Q = Q * r - r + 1.0
        Q = erfinv(Q).astype("float")
        return exp(self.mu + sqrt(2) * self.sigma * Q)


#    def _generate_random_continuous(self, r1, r2=None):
#        from numpy import log, sqrt, exp, sin, cos
#        from scipy.constants import pi
#        if r2==None:
#            from numpy.random import rand
#            r2 = rand(len(r1))
#            r2_provided = False
#        else:
#            r2_provided = True
#
#        rho = sqrt(-2.0 * self.sigma**2.0 * log(1-r1))
#        theta = 2.0 * pi * r2
#        x1 = exp(rho * sin(theta))
#        x2 = exp(rho * cos(theta))
#
#        if r2_provided:
#            return x1, x2
#        else:
#            return x1


class Lognormal_Positive(Lognormal):
    @property
    def name(self):
        return "lognormal_positive"

    def _in_standard_parameter_range(self):
        # The standard deviation and mean can't be negative
        return self.sigma > 0 and self.mu > 0


def nested_loglikelihood_ratio(loglikelihoods1, loglikelihoods2, **kwargs):
    """
    Calculates a loglikelihood ratio and the p-value for testing which of two
    probability distributions is more likely to have created a set of
    observations. Assumes one of the probability distributions is a nested
    version of the other.

    Parameters
    ----------
    loglikelihoods1 : list or array
        The logarithms of the likelihoods of each observation, calculated from
        a particular probability distribution.
    loglikelihoods2 : list or array
        The logarithms of the likelihoods of each observation, calculated from
        a particular probability distribution.
    nested : bool, optional
        Whether one of the two probability distributions that generated the
        likelihoods is a nested version of the other. True by default.
    normalized_ratio : bool, optional
        Whether to return the loglikelihood ratio, R, or the normalized
        ratio R/sqrt(n*variance)

    Returns
    -------
    R : float
        The loglikelihood ratio of the two sets of likelihoods. If positive,
        the first set of likelihoods is more likely (and so the probability
        distribution that produced them is a better fit to the data). If
        negative, the reverse is true.
    p : float
        The significance of the sign of R. If below a critical value
        (typically .05) the sign of R is taken to be significant. If above the
        critical value the sign of R is taken to be due to statistical
        fluctuations.
    """
    return loglikelihood_ratio(loglikelihoods1, loglikelihoods2, nested=True, **kwargs)


def loglikelihood_ratio(
    loglikelihoods1, loglikelihoods2, nested=False, normalized_ratio=False
):
    """
    Calculates a loglikelihood ratio and the p-value for testing which of two
    probability distributions is more likely to have created a set of
    observations.

    Parameters
    ----------
    loglikelihoods1 : list or array
        The logarithms of the likelihoods of each observation, calculated from
        a particular probability distribution.
    loglikelihoods2 : list or array
        The logarithms of the likelihoods of each observation, calculated from
        a particular probability distribution.
    nested : bool, optional
        Whether one of the two probability distributions that generated the
        likelihoods is a nested version of the other. False by default.
    normalized_ratio : bool, optional
        Whether to return the loglikelihood ratio, R, or the normalized
        ratio R/sqrt(n*variance)

    Returns
    -------
    R : float
        The loglikelihood ratio of the two sets of likelihoods. If positive,
        the first set of likelihoods is more likely (and so the probability
        distribution that produced them is a better fit to the data). If
        negative, the reverse is true.
    p : float
        The significance of the sign of R. If below a critical value
        (typically .05) the sign of R is taken to be significant. If above the
        critical value the sign of R is taken to be due to statistical
        fluctuations.
    """
    from numpy import sqrt
    from scipy.special import erfc

    n = float(len(loglikelihoods1))

    if n == 0:
        R = 0
        p = 1
        return R, p
    from numpy import asarray

    loglikelihoods1 = asarray(loglikelihoods1)
    loglikelihoods2 = asarray(loglikelihoods2)

    # Clean for extreme values, if any
    from numpy import inf, log
    from sys import float_info

    min_val = log(10**float_info.min_10_exp)
    loglikelihoods1[loglikelihoods1 == -inf] = min_val
    loglikelihoods2[loglikelihoods2 == -inf] = min_val

    R = sum(loglikelihoods1 - loglikelihoods2)

    from numpy import mean

    mean_diff = mean(loglikelihoods1) - mean(loglikelihoods2)
    variance = sum(((loglikelihoods1 - loglikelihoods2) - mean_diff) ** 2) / n

    if nested:
        from scipy.stats import chi2

        p = 1 - chi2.cdf(abs(2 * R), 1)
    else:
        p = erfc(abs(R) / sqrt(2 * n * variance))

    if normalized_ratio:
        R = R / sqrt(n * variance)

    return R, p


def cdf(data, survival=False, **kwargs):
    """
    The cumulative distribution function (CDF) of the data.

    Parameters
    ----------
    data : list or array, optional
    survival : bool, optional
        Whether to calculate a CDF (False) or CCDF (True). False by default.

    Returns
    -------
    X : array
        The sorted, unique values in the data.
    probabilities : array
        The portion of the data that is less than or equal to X.
    """
    return cumulative_distribution_function(data, survival=survival, **kwargs)


def ccdf(data, survival=True, **kwargs):
    """
    The complementary cumulative distribution function (CCDF) of the data.

    Parameters
    ----------
    data : list or array, optional
    survival : bool, optional
        Whether to calculate a CDF (False) or CCDF (True). True by default.

    Returns
    -------
    X : array
        The sorted, unique values in the data.
    probabilities : array
        The portion of the data that is less than or equal to X.
    """
    return cumulative_distribution_function(data, survival=survival, **kwargs)


def cumulative_distribution_function(
    data, xmin=None, xmax=None, survival=False, **kwargs
):
    """
    The cumulative distribution function (CDF) of the data.

    Parameters
    ----------
    data : list or array, optional
    survival : bool, optional
        Whether to calculate a CDF (False) or CCDF (True). False by default.
    xmin : int or float, optional
        The minimum data size to include. Values less than xmin are excluded.
    xmax : int or float, optional
        The maximum data size to include. Values greater than xmin are
        excluded.

    Returns
    -------
    X : array
        The sorted, unique values in the data.
    probabilities : array
        The portion of the data that is less than or equal to X.
    """

    from numpy import array

    data = array(data)
    if not data.any():
        from numpy import nan

        return array([nan]), array([nan])

    data = trim_to_range(data, xmin=xmin, xmax=xmax)

    n = float(len(data))
    from numpy import sort

    data = sort(data)
    all_unique = not (any(data[:-1] == data[1:]))

    if all_unique:
        from numpy import arange

        CDF = arange(1, n + 1) / n
    else:
        # This clever bit is a way of using searchsorted to rapidly calculate the
        # CDF of data with repeated values comes from Adam Ginsburg's plfit code,
        # specifically https://github.com/keflavich/plfit/commit/453edc36e4eb35f35a34b6c792a6d8c7e848d3b5#plfit/plfit.py
        from numpy import searchsorted, unique

        CDF = searchsorted(data, data, side="right") / n
        unique_data, unique_indices = unique(data, return_index=True)
        data = unique_data
        CDF = CDF[unique_indices]

    if survival:
        CDF = 1 - CDF
    return data, CDF


def is_discrete(data):
    """Checks if every element of the array is an integer."""
    from numpy import floor

    return (floor(data) == data.astype(float)).all()


def trim_to_range(data, xmin=None, xmax=None, **kwargs):
    """
    Removes elements of the data that are above xmin or below xmax (if present)
    """
    from numpy import asarray

    data = asarray(data)
    if xmin:
        data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]
    return data


def pdf(data, xmin=None, xmax=None, linear_bins=False, **kwargs):
    """
    Returns the probability density function (normalized histogram) of the
    data.

    Parameters
    ----------
    data : list or array
    xmin : float, optional
        Minimum value of the PDF. If None, uses the smallest value in the data.
    xmax : float, optional
        Maximum value of the PDF. If None, uses the largest value in the data.
    linear_bins : float, optional
        Whether to use linearly spaced bins, as opposed to logarithmically
        spaced bins (recommended for log-log plots).

    Returns
    -------
    bin_edges : array
        The edges of the bins of the probability density function.
    probabilities : array
        The portion of the data that is within the bin. Length 1 less than
        bin_edges, as it corresponds to the spaces between them.
    """

    from numpy import logspace, histogram, floor, unique, asarray
    from math import ceil, log10

    data = asarray(data)
    if not xmax:
        xmax = max(data)
    if not xmin:
        xmin = min(data)

    if (
        xmin < 1
    ):  # To compute the pdf also from the data below x=1, the data, xmax and xmin are rescaled dividing them by xmin.
        xmax2 = xmax / xmin
        xmin2 = 1
    else:
        xmax2 = xmax
        xmin2 = xmin

    if "bins" in kwargs.keys():
        bins = kwargs.pop("bins")
    elif linear_bins:
        bins = range(int(xmin2), ceil(xmax2) + 1)
    else:
        log_min_size = log10(xmin2)
        log_max_size = log10(xmax2)
        number_of_bins = ceil((log_max_size - log_min_size) * 10)
        bins = logspace(log_min_size, log_max_size, num=number_of_bins)
        bins[:-1] = floor(bins[:-1])
        bins[-1] = ceil(bins[-1])
        bins = unique(bins)

    if xmin < 1:  # Needed to include also data x<1 in pdf.
        hist, edges = histogram(data / xmin, bins, density=True)
        edges = edges * xmin  # transform result back to original
        hist = hist / xmin  # rescale hist, so that np.sum(hist*edges)==1
    else:
        hist, edges = histogram(data, bins, density=True)

    return edges, hist


def checkunique(data):
    """Quickly checks if a sorted array is all unique elements."""
    for i in range(len(data) - 1):
        if data[i] == data[i + 1]:
            return False
    return True


def plot_ccdf(data, ax=None, survival=False, **kwargs):
    return plot_cdf(data, ax=ax, survival=True, **kwargs)
    """
    Plots the complementary cumulative distribution function (CDF) of the data
    to a new figure or to axis ax if provided.

    Parameters
    ----------
    data : list or array
    ax : matplotlib axis, optional
        The axis to which to plot. If None, a new figure is created.
    survival : bool, optional
        Whether to plot a CDF (False) or CCDF (True). True by default.

    Returns
    -------
    ax : matplotlib axis
        The axis to which the plot was made.
    """


def plot_cdf(data, ax=None, survival=False, **kwargs):
    """
    Plots the cumulative distribution function (CDF) of the data to a new
    figure or to axis ax if provided.

    Parameters
    ----------
    data : list or array
    ax : matplotlib axis, optional
        The axis to which to plot. If None, a new figure is created.
    survival : bool, optional
        Whether to plot a CDF (False) or CCDF (True). False by default.

    Returns
    -------
    ax : matplotlib axis
        The axis to which the plot was made.
    """
    bins, CDF = cdf(data, survival=survival, **kwargs)
    if not ax:
        import matplotlib.pyplot as plt

        plt.plot(bins, CDF, **kwargs)
        ax = plt.gca()
    else:
        ax.scatter(bins, CDF, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax


def plot_pdf(data, ax=None, linear_bins=False, **kwargs):
    """
    Plots the probability density function (PDF) to a new figure or to axis ax
    if provided.

    Parameters
    ----------
    data : list or array
    ax : matplotlib axis, optional
        The axis to which to plot. If None, a new figure is created.
    linear_bins : bool, optional
        Whether to use linearly spaced bins (True) or logarithmically
        spaced bins (False). False by default.

    Returns
    -------
    ax : matplotlib axis
        The axis to which the plot was made.
    """
    edges, hist = pdf(data, linear_bins=linear_bins, **kwargs)
    bin_centers = (edges[1:] + edges[:-1]) / 2.0
    # for h, bin_center in zip(hist, bin_centers):
    # print(f"hist:{h}, c: {bin_center}")
    from numpy import nan

    hist[hist == 0] = nan
    if not ax:
        import matplotlib.pyplot as plt

        plt.plot(bin_centers, hist, **kwargs)
        ax = plt.gca()
    else:
        ax.plot(bin_centers, hist, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax


def bisect_map(mn, mx, function, target):
    """
    Uses binary search to find the target solution to a function, searching in
    a given ordered sequence of integer values.

    Parameters
    ----------
    seq : list or array, monotonically increasing integers
    function : a function that takes a single integer input, which monotonically
        decreases over the range of seq.
    target : the target value of the function

    Returns
    -------
    value : the input value that yields the target solution. If there is no
    exact solution in the input sequence, finds the nearest value k such that
    function(k) <= target < function(k+1). This is similar to the behavior of
    bisect_left in the bisect package. If even the first, leftmost value of seq
    does not satisfy this condition, -1 is returned.
    """
    if function([mn]) < target or function([mx]) > target:
        return -1
    while 1:
        if mx == mn + 1:
            return mn
        m = (mn + mx) / 2
        value = function([m])[0]
        if value > target:
            mn = m
        elif value < target:
            mx = m
        else:
            return m


def distribution_compare(
    data,
    distribution1,
    parameters1,
    distribution2,
    parameters2,
    discrete,
    xmin,
    xmax,
    nested=None,
    **kwargs,
):
    no_data = False
    if xmax and all((data > xmax) + (data < xmin)):
        # Everything is beyond the bounds of the xmax and xmin
        no_data = True
    if all(data < xmin):
        no_data = True

    if no_data:
        R = 0
        p = 1
        return R, p

    likelihood_function1 = likelihood_function_generator(
        distribution1, discrete, xmin, xmax
    )
    likelihood_function2 = likelihood_function_generator(
        distribution2, discrete, xmin, xmax
    )

    likelihoods1 = likelihood_function1(parameters1, data)
    likelihoods2 = likelihood_function2(parameters2, data)

    if (
        (distribution1 in distribution2)
        or (distribution2 in distribution1)
        and nested is None
    ):
        print("Assuming nested distributions", file=sys.stderr)
        nested = True

    from numpy import log

    R, p = loglikelihood_ratio(
        log(likelihoods1), log(likelihoods2), nested=nested, **kwargs
    )

    return R, p


def likelihood_function_generator(distribution_name, discrete=False, xmin=1, xmax=None):
    if distribution_name == "power_law":

        def likelihood_function(parameters, data):
            return power_law_likelihoods(data, parameters[0], xmin, xmax, discrete)

    elif distribution_name == "exponential":

        def likelihood_function(parameters, data):
            return exponential_likelihoods(data, parameters[0], xmin, xmax, discrete)

    elif distribution_name == "stretched_exponential":

        def likelihood_function(parameters, data):
            return stretched_exponential_likelihoods(
                data, parameters[0], parameters[1], xmin, xmax, discrete
            )

    elif distribution_name == "truncated_power_law":

        def likelihood_function(parameters, data):
            return truncated_power_law_likelihoods(
                data, parameters[0], parameters[1], xmin, xmax, discrete
            )

    elif distribution_name == "lognormal":

        def likelihood_function(parameters, data):
            return lognormal_likelihoods(
                data, parameters[0], parameters[1], xmin, xmax, discrete
            )

    elif distribution_name == "negative_binomial":

        def likelihood_function(parameters, data):
            return negative_binomial_likelihoods(
                data, parameters[0], parameters[1], xmin, xmax
            )

    elif distribution_name == "gamma":

        def likelihood_function(parameters, data):
            return gamma_likelihoods(data, parameters[0], parameters[1], xmin, xmax)

    return likelihood_function


def power_law_ks_distance(data, alpha, xmin, xmax=None, discrete=False, kuiper=False):
    from numpy import arange, sort, mean

    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]
    n = len(data)
    if n < 2:
        if kuiper:
            return 1, 1, 2
        return 1

    if not all(data[i] <= data[i + 1] for i in arange(n - 1)):
        data = sort(data)

    if not discrete:
        Actual_CDF = arange(n) / float(n)
        Theoretical_CDF = 1 - (data / xmin) ** (-alpha + 1)

    if discrete:
        from scipy.special import zeta

        if xmax:
            bins, Actual_CDF = cumulative_distribution_function(
                data, xmin=xmin, xmax=xmax
            )
            Theoretical_CDF = 1 - (
                (zeta(alpha, bins) - zeta(alpha, xmax + 1))
                / (zeta(alpha, xmin) - zeta(alpha, xmax + 1))
            )
        if not xmax:
            bins, Actual_CDF = cumulative_distribution_function(data, xmin=xmin)
            Theoretical_CDF = 1 - (zeta(alpha, bins) / zeta(alpha, xmin))

    D_plus = max(Theoretical_CDF - Actual_CDF)
    D_minus = max(Actual_CDF - Theoretical_CDF)
    Kappa = 1 + mean(Theoretical_CDF - Actual_CDF)

    if kuiper:
        return D_plus, D_minus, Kappa

    D = max(D_plus, D_minus)

    return D


def power_law_likelihoods(data, alpha, xmin, xmax=False, discrete=False):
    if alpha < 0:
        from numpy import tile
        from sys import float_info

        return tile(10**float_info.min_10_exp, len(data))

    xmin = float(xmin)
    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    if not discrete:
        likelihoods = (data**-alpha) * ((alpha - 1) * xmin ** (alpha - 1))
    if discrete:
        if alpha < 1:
            from numpy import tile
            from sys import float_info

            return tile(10**float_info.min_10_exp, len(data))
        if not xmax:
            from scipy.special import zeta

            likelihoods = (data**-alpha) / zeta(alpha, xmin)
        if xmax:
            from scipy.special import zeta

            likelihoods = (data**-alpha) / (zeta(alpha, xmin) - zeta(alpha, xmax + 1))
    from sys import float_info

    likelihoods[likelihoods == 0] = 10**float_info.min_10_exp
    return likelihoods


def negative_binomial_likelihoods(data, r, p, xmin=0, xmax=False):
    # Better to make this correction earlier on in distribution_fit, so as to not recheck for discreteness and reround every time fmin is used.
    # if not is_discrete(data):
    #    print("Rounding to nearest integer values for negative binomial fit.", file=sys.stderr)
    #    from numpy import around
    #    data = around(data)

    xmin = float(xmin)
    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    from numpy import asarray
    from scipy.special import comb

    def pmf(k):
        return comb(k + r - 1, k) * (1 - p) ** r * p**k

    likelihoods = asarray(list(map(pmf, data))).flatten()

    if xmin != 0 or xmax:
        xmax = max(data)
        from numpy import arange

        normalization_constant = sum(list(map(pmf, arange(xmin, xmax + 1))))
        likelihoods = likelihoods / normalization_constant

    from sys import float_info

    likelihoods[likelihoods == 0] = 10**float_info.min_10_exp
    return likelihoods


def exponential_likelihoods(data, Lambda, xmin, xmax=False, discrete=False):
    if Lambda < 0:
        from numpy import tile
        from sys import float_info

        return tile(10**float_info.min_10_exp, len(data))

    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    from numpy import exp

    if not discrete:
        #        likelihoods = exp(-Lambda*data)*\
        #                Lambda*exp(Lambda*xmin)
        likelihoods = Lambda * exp(
            Lambda * (xmin - data)
        )  # Simplified so as not to throw a nan from infs being divided by each other
    if discrete:
        if not xmax:
            likelihoods = exp(-Lambda * data) * (1 - exp(-Lambda)) * exp(Lambda * xmin)
        if xmax:
            likelihoods = (
                exp(-Lambda * data)
                * (1 - exp(-Lambda))
                / (exp(-Lambda * xmin) - exp(-Lambda * (xmax + 1)))
            )
    from sys import float_info

    likelihoods[likelihoods == 0] = 10**float_info.min_10_exp
    return likelihoods


def stretched_exponential_likelihoods(
    data, Lambda, beta, xmin, xmax=False, discrete=False
):
    if Lambda < 0:
        from numpy import tile
        from sys import float_info

        return tile(10**float_info.min_10_exp, len(data))

    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    from numpy import exp

    if not discrete:
        likelihoods = (
            (data * Lambda) ** (beta - 1)
            * beta
            * Lambda
            * exp((Lambda * (xmin - data)) ** beta)
        )
        # Simplified so as not to throw a nan from infs being divided by each other
    if discrete:
        if not xmax:
            xmax = max(data)
        if xmax:
            from numpy import arange

            X = arange(xmin, xmax + 1)
            PDF = (
                X ** (beta - 1) * beta * Lambda * exp(Lambda * (xmin**beta - X**beta))
            )  # Simplified so as not to throw a nan from infs being divided by each other
            PDF = PDF / sum(PDF)
            likelihoods = PDF[(data - xmin).astype(int)]
    from sys import float_info

    likelihoods[likelihoods == 0] = 10**float_info.min_10_exp
    return likelihoods


def gamma_likelihoods(data, k, theta, xmin, xmax=False, discrete=False):
    if k <= 0 or theta <= 0:
        from numpy import tile
        from sys import float_info

        return tile(10**float_info.min_10_exp, len(data))

    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    from numpy import exp
    from mpmath import gammainc

    #    from scipy.special import gamma, gammainc #Not NEARLY numerically accurate enough for the job
    if not discrete:
        likelihoods = (data ** (k - 1)) / (
            exp(data / theta) * (theta**k) * float(gammainc(k))
        )
        # Calculate how much probability mass is beyond xmin, and normalize by it
        normalization_constant = 1 - float(
            gammainc(k, 0, xmin / theta, regularized=True)
        )  # Mpmath's regularized option divides by gamma(k)
        likelihoods = likelihoods / normalization_constant
    if discrete:
        if not xmax:
            xmax = max(data)
        if xmax:
            from numpy import arange

            X = arange(xmin, xmax + 1)
            PDF = (X ** (k - 1)) / (exp(X / theta) * (theta**k) * float(gammainc(k)))
            PDF = PDF / sum(PDF)
            likelihoods = PDF[(data - xmin).astype(int)]
    from sys import float_info

    likelihoods[likelihoods == 0] = 10**float_info.min_10_exp
    return likelihoods


def truncated_power_law_likelihoods(
    data, alpha, Lambda, xmin, xmax=False, discrete=False
):
    if alpha < 0 or Lambda < 0:
        from numpy import tile
        from sys import float_info

        return tile(10**float_info.min_10_exp, len(data))

    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    from numpy import exp

    if not discrete:
        from mpmath import gammainc

        #        from scipy.special import gamma, gammaincc #Not NEARLY accurate enough to do the job
        #        likelihoods = (data**-alpha)*exp(-Lambda*data)*\
        #                (Lambda**(1-alpha))/\
        #                float(gammaincc(1-alpha,Lambda*xmin))
        # Simplified so as not to throw a nan from infs being divided by each other
        likelihoods = (Lambda ** (1 - alpha)) / (
            (data**alpha) * exp(Lambda * data) * gammainc(1 - alpha, Lambda * xmin)
        ).astype(float)
    if discrete:
        if not xmax:
            xmax = max(data)
        if xmax:
            from numpy import arange

            X = arange(xmin, xmax + 1)
            PDF = (X**-alpha) * exp(-Lambda * X)
            PDF = PDF / sum(PDF)
            likelihoods = PDF[(data - xmin).astype(int)]
    from sys import float_info

    likelihoods[likelihoods == 0] = 10**float_info.min_10_exp
    return likelihoods


def lognormal_likelihoods(data, mu, sigma, xmin, xmax=False, discrete=False):
    from numpy import log

    if sigma <= 0 or mu < log(xmin):
        # The standard deviation can't be negative, and the mean of the logarithm of the distribution can't be smaller than the log of the smallest member of the distribution!
        from numpy import tile
        from sys import float_info

        return tile(10**float_info.min_10_exp, len(data))

    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    if not discrete:
        from numpy import sqrt, exp

        #        from mpmath import erfc
        from scipy.special import erfc
        from scipy.constants import pi

        likelihoods = (
            (1.0 / data)
            * exp(-((log(data) - mu) ** 2) / (2 * sigma**2))
            * sqrt(2 / (pi * sigma**2))
            / erfc((log(xmin) - mu) / (sqrt(2) * sigma))
        )
    #        likelihoods = likelihoods.astype(float)
    if discrete:
        if not xmax:
            xmax = max(data)
        if xmax:
            from numpy import arange, exp

            #            from mpmath import exp
            X = arange(xmin, xmax + 1)
            #            PDF_function = lambda x: (1.0/x)*exp(-( (log(x) - mu)**2 ) / 2*sigma**2)
            #            PDF = asarray(list(map(PDF_function,X)))
            PDF = (1.0 / X) * exp(-((log(X) - mu) ** 2) / (2 * (sigma**2)))
            PDF = (PDF / sum(PDF)).astype(float)
            likelihoods = PDF[(data - xmin).astype(int)]
    from sys import float_info

    likelihoods[likelihoods == 0] = 10**float_info.min_10_exp
    return likelihoods
