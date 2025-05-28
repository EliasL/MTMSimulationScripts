#!/usr/bin/env python3
import numpy as np
from scipy.special import gammaincc, gammainccinv
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from tqdm import tqdm


def truncatedPowerlawGenerator(xmin, alpha, Lambda, size):
    """
    Generate `size` samples from the PDF
        f(x) ∝ x**(-alpha) * exp(-Lambda*x),  x >= xmin
    using vectorized rejection sampling with an exponential proposal.
    """
    # container for accepted draws
    result = np.empty(size, dtype=float)
    n_done = 0
    pbar = tqdm(total=size, desc="Sampling truncated powerlaw")

    while n_done < size:
        # how many more we need
        n_remain = size - n_done
        # 1) propose from Exp(Lambda) shifted to start at xmin
        proposals = xmin + np.random.exponential(scale=1.0 / Lambda, size=n_remain)
        # 2) compute acceptance probabilities = (xmin / x) ** alpha
        accept_probs = (xmin / proposals) ** alpha
        # 3) draw uniforms and accept
        u = np.random.rand(n_remain)
        mask = u < accept_probs
        n_accept = mask.sum()
        if n_accept:
            result[n_done : n_done + n_accept] = proposals[mask]
            n_done += n_accept
            pbar.update(n_accept)
    pbar.close()
    return result


def scipy_truncated_powerlaw(xmin, alpha, Lambda, size):
    import numpy as np
    from scipy.special import gammainc, gammaincinv

    k = 1.0 - alpha
    theta = 1.0 / Lambda

    Fmin = gammainc(k, xmin / theta)  # lower‐gamma regularized at x_min
    u = Fmin + (1.0 - Fmin) * np.random.rand(size)  # a uniform in [Fmin,1)
    y = gammaincinv(k, u)  # invert P(k,y)=u
    x = theta * y  # your draw
    return x


def truncated_powerlaw_rejection(xmin, alpha, Lambda, size):
    """
    The original per‐sample rejection helper, lifted out of the class.
    Very slow for large `size`, but exact.
    """

    def draw_one(r0):
        r = r0
        while True:
            x = xmin - (1.0 / Lambda) * np.log(1 - r)
            p = (x / xmin) ** (-alpha)
            if np.random.rand() < p:
                return x
            r = np.random.rand()

    # initial uniforms
    r = np.random.rand(size)
    # map each r through helper
    return np.array([draw_one(rr) for rr in r])


def main():
    # parameters
    xmin = 1e-8
    alpha = 0.9
    Lambda = 1.0
    alpha = 0.975
    Lambda = 2204.838
    size = 110004  # demo size; increase if you like

    # draw samples
    print("Sampling rejection…")
    # samples_rej = truncated_powerlaw_rejection(xmin, alpha, Lambda, size)
    samples_rej = truncatedPowerlawGenerator(xmin, alpha, Lambda, size)
    print("Sampling ICDF…")
    samples_icdf = scipy_truncated_powerlaw(xmin, alpha, Lambda, size)

    # summary stats
    print("\nSummary statistics:")
    for name, data in [("ICDF", samples_icdf), ("Rejection", samples_rej)]:
        print(f"  {name}: mean={data.mean():.4f}, var={data.var():.4f}")

    # KS test
    ks_stat, p_value = ks_2samp(samples_icdf, samples_rej)
    print(f"\nTwo‐sample KS test: D={ks_stat:.4f}, p‐value={p_value:.4g}")

    # plot histograms
    plt.figure(figsize=(8, 5))
    bins = np.logspace(
        np.log10(xmin), np.log10(max(samples_icdf.max(), samples_rej.max())), 50
    )
    plt.hist(samples_icdf, bins=bins, density=True, alpha=0.6, label="ICDF")
    plt.hist(samples_rej, bins=bins, density=True, alpha=0.6, label="Rejection")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("pdf")
    plt.legend()
    plt.title("Truncated Power‐Law: ICDF vs. Rejection")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
