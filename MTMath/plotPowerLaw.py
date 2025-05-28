import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm, colors
import powerlaw
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import functools
import os
import glob

np.random.seed(0)
# Create directories for saving plots
os.makedirs("Plots/powerLaw", exist_ok=True)


def get_energy_drops(csvPath, df=None, strainLim=[-np.inf, np.inf], debug=False):
    """
    Strain energy drop data from CSV, filter by strain limits, and return drops.
    If debug=True, plot intermediate energy and drop traces.
    """
    if df is None:
        df = pd.read_csv(csvPath)
    diffs = df["avg_energy_change"]
    strain = df["load"]
    lim_mask = (strain > strainLim[0]) & (strain < strainLim[1])
    drop_mask = diffs < 0
    mask = drop_mask & lim_mask
    drops = -diffs[mask]
    if debug:
        e = df["avg_energy"]
        debug_fig, ax1 = plt.subplots()
        ax1.plot(strain, e, label=r"$\langle E \rangle$")
        ax1.set_ylabel(r"$\langle E \rangle$")
        ax1.set_xlabel(r"$\gamma$")
        ax2 = ax1.twinx()
        ax2.plot([])  # advance color cycle
        ax2.plot(strain[mask], drops, label=r"$-\Delta \langle E \rangle$")
        ax2.set_ylabel(r"$-\Delta \langle E \rangle$ (Energy Drop)")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2)
        ax2.set_ylim(0, drops.max() * 1.5)

        # ——— Compute 0.1%‐wide central slice ———
        mid = 0.5 * (strainLim[0] + strainLim[1])
        total_width = strainLim[1] - strainLim[0]
        slice_width = total_width * 0.05  # 1% of window
        x1, x2 = mid - slice_width / 2, mid + slice_width / 2
        zoom_mask = (strain >= x1) & (strain <= x2)

        # find energy‐axis extents in that slice
        y1, y2 = (-diffs[zoom_mask]).min(), (-diffs[zoom_mask]).max()
        zoomWidth = np.clip(x2 - x1, 0.01, None)
        zoomHeight = y2 - y1

        # draw red dashed box on main axes
        rect = Rectangle(
            (x1 - zoomWidth * 0.5, y1),  # lower‐left corner
            zoomWidth * 1.5,  # width
            zoomHeight,  # height
            linewidth=2,
            edgecolor="black",
            linestyle="--",
            facecolor="none",
            zorder=10,
        )
        ax2.add_patch(rect)

        # ——— Inset axes at top middle-left ———
        axins = inset_axes(
            ax1,
            width=1.5,
            height=0.7,
            loc="center",
            bbox_to_anchor=(0.45, 0.7, 0.0, 0.30),
            bbox_transform=ax1.transAxes,
        )
        # plot energy in inset
        axins.plot(strain[zoom_mask], e[zoom_mask], lw=0.8)
        axins.set_xlim(x1, x2)
        axins.set_title("Zoom", fontsize=8)

        # twin‐axis for drops in the inset
        axins2 = axins.twinx()
        drops_zoom = -diffs[zoom_mask]
        axins2.plot(strain[zoom_mask], drops_zoom)
        axins2.set_ylim(0, drops_zoom.max() * 1.5)

        debug_fig.tight_layout()
        # Save debug energy plot
        filename = f"{plotPath}energy_drops_strain_{strainLim[0]:.2f}_{strainLim[1]:.2f}{outputType}"
        debug_fig.savefig(filename, dpi=300)
        # to save memory, close the figure
        plt.close(debug_fig)
    return drops


def plot_data(ax, fit=None, data=None, xmin=None):
    if data is None and fit is not None:
        data = fit.data_original
    elif fit is None and data is not None:
        fit = powerlaw.Fit(data, xmin=xmin)
    else:
        raise ValueError("Either data or fit must be provided.")

    # full-data empirical
    fit.plot_ccdf(
        ax=ax,
        marker="o",
        linestyle="None",
        label="All drops",
        original_data=True,
        facecolor="none",
        edgecolor="black",
    )
    return fit


def plot_fit(ax, fit, dist_name=None, title=None, color=None):
    # compute weight and x-grid
    data = fit.data_original
    xmin = fit.xmin
    x_vals = np.logspace(
        np.log10(data.min()),  # start at xmin
        np.log10(data.max()),
        num=200,
    )

    dist = getattr(fit, dist_name)
    CDF = dist._cdf_base_function(x_vals)
    CCDF = 1 - CDF

    # Area under  CCDF in the fit region
    # try to use data from ax
    if len(ax.collections) > 0:
        x = ax.collections[0].get_offsets()[:, 0]  # CCDF x values
        mask = x > xmin
        empirical_area = np.trapezoid(
            ax.collections[0].get_offsets()[:, 1][mask],  # CCDF area
            x=x[mask],  # CCDF x values
        )  # CCDF area

        mask = x_vals > xmin
        # Area under fitted CCDF in the fit region
        fitted_area = np.trapezoid(CCDF[mask], x=x_vals[mask])  # fitted CCDF area
        # Scale the fitted CCDF to match the CCDF area
        CCDF = CCDF * empirical_area / fitted_area

    label = f"{dist_name}: "
    params = zip(
        [
            dist.parameter1_name,
            dist.parameter2_name,
            dist.parameter3_name,
        ],
        [
            dist.parameter1,
            dist.parameter2,
            dist.parameter3,
        ],
    )

    for name, p in params:
        if name is not None:
            if name == "lambda":
                label += f"1/{name}={(1 / p):.2e}, "
            else:
                label += f"{name}={p:.3f}, "
            # print(f"{name}={p:.3f}")
        # For some reason, power_law does not have any parameters
        if dist_name == "power_law":
            label += f"alpha={dist.alpha:.3f}, "
            break
    # remove last comma
    label = label[:-2]
    ax.plot(x_vals, CCDF, linestyle="-", label=pretty_label(label), color=color)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$-\Delta \langle E \rangle$ (Energy Drop)")
    plt.ylabel(r"$P(X > x)$")
    plt.title(title)
    ax.legend(loc="lower left")


def compare_dists(fit):
    mainDist = "truncated_power_law"
    compareDists = ["lognormal", "power_law"]
    # Compare distributions
    for dist_name in compareDists:
        # Likelihood ratio test comparing mainDist to dist_name
        R, p = fit.distribution_compare(mainDist, dist_name, normalized_ratio=True)
        print(
            f"Likelihood ratio test ({mainDist} vs {dist_name}): R={R:.3f}, p={p:.3e}"
        )


def pretty_label(label):
    r"""
    Make the lable nicer.
    truncated_power_law: alpha=1.02, lambda=0.5
    ->
    Truncated Power Law: \alpha=1.02, \lambda=0.5
    """
    label = label.replace("_", " ")
    label = label.replace("alpha", r"$\alpha$")
    label = label.replace("lambda", r"$\lambda$")
    label = label.replace("mu", r"$\mu$")
    label = label.replace("sigma", r"$\sigma$")
    # Captialize first letter of the first word
    label = label.capitalize()
    return label


def get_drops_in_windows(
    csvPath=None, strainLim=None, df=None, steps=1, window_width=np.inf, debug=False
):
    if df is None:
        df = pd.read_csv(csvPath)
    strain = df["load"]
    if strainLim is not None:
        lim_mask = (strain > strainLim[0]) & (strain < strainLim[1])
        df = df[lim_mask]
        strain = df["load"]

    global_max_strain = strain.max()
    global_min_strain = strain.min()
    if window_width == np.inf:
        window_width = global_max_strain - global_min_strain

    if global_max_strain - global_min_strain < window_width:
        centers = [global_min_strain + window_width / 2]
    else:
        # get list of window centers
        centers = np.linspace(
            global_min_strain + window_width / 2,
            global_max_strain - window_width / 2,
            steps,
        )

    drops_in_windows = []
    windows = []
    for center in centers:
        # get the window
        min_strain = center - window_width / 2
        max_strain = center + window_width / 2
        # get the data in the window
        drops = get_energy_drops(
            csvPath, df=df, strainLim=[min_strain, max_strain], debug=debug
        )
        windows.append((min_strain, max_strain))
        drops_in_windows.append(drops)
    return drops_in_windows, windows, centers


def plot_data_and_fit(
    fit,
    ax=None,
    xmin=None,
    title="",
    dist_names=[
        "truncated_power_law",
        # "lognormal",
        # "power_law",
        # "exponential",
        # "stretched_exponential",
        # "lognormal_positive",
    ],
):
    if ax is None:
        fig, ax = plt.subplots()
    # plot the data
    plot_data(ax, fit=fit)
    # plot the fit

    cmap_colors = ["green", "red", "yellow", "orange", "blue", "cyan"]

    for dist_name, color in zip(dist_names, cmap_colors):
        plot_fit(
            ax,
            fit,
            dist_name=dist_name,
            title=title,
            color=color,
        )

    # Add shaded fit region
    ax.axvspan(xmin, fit.data.max(), color="gray", alpha=0.2, label="Fit region")
    ax.legend()
    return ax


def get_window_power_law_exponents(
    xmin=-np.inf,
    dist="truncated_power_law",
    syntheticData=False,
    syntheticExponent=1.0,
    **kwargs,
):
    """
    We slide this window over the data and plot the power law fit for each window.
    """
    drops_in_windows, windows, centers = get_drops_in_windows(**kwargs)
    fits = []
    ps = []
    debug = kwargs.get("debug", False)
    for drops, strainLim in zip(drops_in_windows, windows):
        if syntheticData:
            # We generate synthetic data instead of using the real data
            drops = create_synthetic_data(
                drops,
                xmin=xmin,
                nrSets=1,
                dist_name=dist,
                params={"alpha": syntheticExponent},
            )[0]
        # fit the data
        fit = powerlaw.Fit(drops, xmin=xmin)
        fits.append(fit)
        p = evaluate_fit(
            drops,
            xmin=xmin,
            dist_name=dist,
            parallel=False,
            verbose=True,  # debug,
        )
        ps.append(p)

        if debug:
            debug_fig, debug_ax = plt.subplots()
            title = rf"$\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f},  $E_{{\mathrm{{min}}}}$={xmin:.2e}"

            plot_data_and_fit(fit, debug_ax, xmin, title)
            debug_fig.tight_layout()
            debug_fig.show()
            # Save debug window power law plot
            filename = f"{plotPath}window_strain_{strainLim[0]:.2f}_{strainLim[1]:.2f}_xmin_{xmin:.2e}{outputType}"
            debug_fig.savefig(filename)
            # to save memory, close the figure
            plt.close(debug_fig)

    # plot the exponents against the window centers
    exponents = [fit.truncated_power_law.alpha for fit in fits]

    return centers, exponents, ps


def worker_get_exponents(xmin, kwargs):
    import numpy as np

    seed = int((np.log10(xmin) * 1e6) % (2**32))  # Stable and unique
    np.random.seed(seed)
    return get_window_power_law_exponents(xmin=xmin, **kwargs)


def get_power_law_surface(xmins=None, **kwargs):
    # If we debug, we don't use multiprocessing
    if kwargs.get("debug", False):
        exponent_xmin_surface, p = [], []
        for xmin in tqdm(xmins):
            centers, exponents, p_ = get_window_power_law_exponents(xmin=xmin, **kwargs)
            exponent_xmin_surface.append(exponents)
            p.append(p_)
        return centers, np.array(exponent_xmin_surface), np.array(p)
    else:
        # Do the first call without the pool
        # centers, exponents, p = get_window_power_law_exponents(xmin=xmins[0], **kwargs)
        # Pre-bind kwargs using functools.partial
        with ProcessPoolExecutor() as executor:
            bound_worker = functools.partial(worker_get_exponents, kwargs=kwargs)
            results = list(tqdm(executor.map(bound_worker, xmins), total=len(xmins)))

        centers = results[0][0]  # All share same centers
        exponent_xmin_surface = np.array([r[1] for r in results])
        p = np.array([r[2] for r in results])
        return centers, exponent_xmin_surface, p


def plot_power_law_map(
    csvPath=None,
    xmins=None,
    df=None,
    strainLim=[-np.inf, np.inf],
    window_steps=20,
    window_width=0.4,
    debug=False,
    use_confidence_color=False,
    syntheticData=False,
    syntheticExponent=1.0,
):
    """
    Takes a csvPath or an already loaded file as a pandas dataframe (df)
    """

    # convert exponents to numpy array
    centers, exponent_xmin_surface, p = get_power_law_surface(
        csvPath=csvPath,
        xmins=xmins,
        df=df,
        strainLim=strainLim,
        steps=window_steps,
        window_width=window_width,
        debug=debug,
        syntheticData=syntheticData,
        syntheticExponent=syntheticExponent,
    )

    # Now we can plot a surface of the exponents on the z axis, centers on the x axis, and xmins on the y axis
    fig = plt.figure()
    if use_confidence_color:
        ax = fig.add_subplot(projection="3d")
        ax.set_zlabel(r"$\alpha$ (Exponent)")
    else:
        ax = fig.add_subplot()

    ax.set_xlabel("Strain window center")
    ax.set_ylabel(r"$\log_{10}(\Delta E_{\mathrm{min}})$")  # Changed label

    figType = "p" if use_confidence_color else "exp"

    # Choose plotting logic based on confidence‐color flag
    if use_confidence_color:
        # use p to color the surface
        facecolors = cm.viridis(p)  # Use colormap for p values

        X, Y = np.meshgrid(centers, np.log10(xmins))
        ax.plot_surface(
            X,
            Y,
            exponent_xmin_surface,
            facecolors=facecolors,
            shade=False,
            antialiased=False,
        )
        # Add colorbar for p values
        norm = colors.Normalize(vmin=np.nanmin(p), vmax=np.nanmax(p))
        sm = cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.1, aspect=10)
        cbar.set_label(r"$p$")

    else:
        X, Y = np.meshgrid(centers, np.log10(xmins))
        pcm = ax.pcolormesh(
            X,
            Y,
            exponent_xmin_surface,
            shading="auto",
            cmap="viridis",
            norm=colors.Normalize(
                vmin=np.nanmin(exponent_xmin_surface),
                vmax=np.nanmax(exponent_xmin_surface),
            ),
        )
        fig.colorbar(pcm, ax=ax, pad=0.1, aspect=10, label=r"$\alpha$")

    fig.tight_layout()
    # Save final power law surface plot
    syntheticTag = "synthetic_" if syntheticData else ""
    filename = (
        f"{plotPath}{syntheticTag}power_law_surface_"
        f"strain_{strainLim[0]:.2f}_{strainLim[1]:.2f}_"
        f"xmin_{xmins[0]:.2e}_{xmins[-1]:.2e}_"
        f"steps_{window_steps}_width_{window_width:.2f}_"
        f"{figType}{outputType}"
    )
    fig.savefig(filename, dpi=300)
    plt.show()


def make_debug_plot(xmins, strainLim=None):
    # Create debug plot grids for each xmin
    for xmin in xmins:
        # Find all saved fit plots for this xmin
        pattern = f"{plotPath}window_strain_*_xmin_{xmin:.2e}{outputType}"
        fit_files = sorted(glob.glob(pattern))
        if not fit_files:
            print(f"No fit debug files found for xmin {xmin:.2e}")
            continue

        n = len(fit_files)
        fig, axes = plt.subplots(2, n, figsize=(n * 6, 10))

        for i, fit_file in enumerate(fit_files):
            base = os.path.basename(fit_file)
            # Extract the strain range string between "window_strain_" and "_xmin"
            strain_range = base[len("window_strain_") : base.find("_xmin")]

            # Strain and display the energy drops image
            energy_file = f"{plotPath}energy_drops_strain_{strain_range}{outputType}"
            if os.path.exists(energy_file):
                img_energy = plt.imread(energy_file)
                axes[0, i].imshow(img_energy)
            else:
                axes[0, i].text(0.5, 0.5, "Missing image", ha="center", va="center")
            axes[0, i].axis("off")
            # axes[0, i].set_title(f"Energy drops\nstrain {strain_range}")

            # Strain and display the fit plot
            img_fit = plt.imread(fit_file)
            axes[1, i].imshow(img_fit)
            axes[1, i].axis("off")
            # axes[1, i].set_title(f"x_{\mathrm{min}}={xmin:.2e}")

        # fig.suptitle(f"Debug plots for x_{\mathrm{min}}={xmin:.2e}")
        fig.tight_layout()
        # Save the debug plot
        debug_filename = f"{plotPath}debug_fit_plots_xmin_{xmin:.2e}{outputType}"
        fig.savefig(debug_filename)
        # plt.show()


def scipy_truncated_powerlaw(xmin, alpha, Lambda, size, rng):
    from scipy.special import gammainc, gammaincinv

    k = 1.0 - alpha
    theta = 1.0 / Lambda

    Fmin = gammainc(k, xmin / theta)
    u = Fmin + (1.0 - Fmin) * rng.random(size)
    y = gammaincinv(k, u)
    x = theta * y
    return x


def truncatedPowerlawGenerator(xmin, alpha, Lambda, size, rng):
    """
    Generate `size` samples from the PDF
        f(x) ∝ x**(-alpha) * exp(-Lambda*x),  x >= xmin
    using vectorized rejection sampling with an exponential proposal.
    """
    result = np.empty(size, dtype=float)
    n_done = 0
    if alpha < 1:
        return scipy_truncated_powerlaw(xmin, alpha, Lambda, size, rng)
    else:
        while n_done < size:
            n_remain = size - n_done
            proposals = xmin + rng.exponential(scale=1.0 / Lambda, size=n_remain)
            accept_probs = (xmin / proposals) ** alpha
            u = rng.random(n_remain)
            mask = u < accept_probs
            n_accept = mask.sum()
            if n_accept:
                result[n_done : n_done + n_accept] = proposals[mask]
                n_done += n_accept
        return result


def create_synthetic_data(
    drops,
    xmin=-np.inf,
    nrSets=2500,
    dist_name="truncated_power_law",
    params={},
):
    """
    Create synthetic data for testing the power law fitting.
    If not all parameters are given, it will use the fitted parameters from the
    original data.
    """
    fit = powerlaw.Fit(drops, xmin=xmin)
    dist_name = getattr(fit, dist_name)

    tailDrops = drops[drops >= xmin]
    nonTailDrops = drops[drops < xmin]
    # Create a local RNG for reproducibility, seed based on xmin
    rng = np.random.default_rng(int((np.log10(xmin) * 1e6) % (2**32)))

    if len(nonTailDrops) == 0:
        total_samples = nrSets * len(drops)
        samples = truncatedPowerlawGenerator(
            xmin=dist_name.xmin,
            alpha=dist_name.alpha,
            Lambda=dist_name.Lambda,
            size=total_samples,
            rng=rng,
        )
        return samples.reshape((nrSets, len(drops)))

    nrTailObservations = len(tailDrops)
    nrObservations = len(drops)
    p_tail = nrTailObservations / nrObservations
    tail_counts = rng.binomial(nrObservations, p_tail, size=nrSets)

    for key, value in params.items():
        if hasattr(dist_name, key) and value is not None:
            setattr(dist_name, key, value)

    non_tail = rng.choice(nonTailDrops, size=(nrSets, nrObservations), replace=True)
    total_tails = tail_counts.sum()
    all_tails = truncatedPowerlawGenerator(
        xmin=dist_name.xmin,
        alpha=dist_name.alpha,
        Lambda=dist_name.Lambda,
        size=total_tails,
        rng=rng,
    )
    offsets = np.concatenate([[0], np.cumsum(tail_counts)])
    syntheticSets = non_tail.copy()
    for i in range(nrSets):
        k = tail_counts[i]
        if k:
            start = offsets[i]
            end = offsets[i + 1]
            syntheticSets[i, :k] = all_tails[start:end]
    return syntheticSets


def _compute_D_for_set(args):
    """
    Worker for goodnessOfFit parallelization:
      args = (synthetic_dataset, xmin, dist_name)
    Returns the KS‐distance D for that dataset.
    Note that we fit the synthetic data and evaluate
    it's KS distance with new parameters (alpha and lambda)

    Here is an analogy which helps justify why we should fit
    with new parameters:
    Say you and your friend are trying to determine who is better at darts.
    To answer the question, you agree to place a dot on the dart board
    and you both try to hit the dot. Fitting a distribution to data is a bit
    like trying to find out where the dot was, after the darts have been trown.
    If you first find the center of your throws and assume the dot must have
    been in the middle of your throws (giving you the best score), and then
    judge your friend based on the assumption that the dot was where it would be
    most advantageous to you, there will be a significant bias in your favour.


    """
    s, xmin, dist_name = args
    fit_s = powerlaw.Fit(s, xmin=xmin)
    dist_s = getattr(fit_s, dist_name)
    if False:
        ax = plot_data_and_fit(fit_s, xmin=xmin, title="Synthetic Set Fit")
        ax.figure.show()
    return dist_s.D


# --- KS p-value computation function ---
def goodnessOfFit(
    drops, synthetic_sets, xmin=-np.inf, dist_name="truncated_power_law", parallel=False
):
    # Fit the original data
    fit_orig = powerlaw.Fit(drops, xmin=xmin)
    dist_orig = getattr(fit_orig, dist_name)
    D_orig = dist_orig.D

    # print("Computing KS distances for synthetic sets...")
    if parallel:
        # build arg‐tuples so each worker knows xmin & dist_name
        args_list = [(s, xmin, dist_name) for s in synthetic_sets]
        with ProcessPoolExecutor() as executor:
            # tqdm can wrap the map if you like progress bars
            D_synth = list(
                tqdm(
                    executor.map(_compute_D_for_set, args_list),
                    total=len(synthetic_sets),
                )
            )
    else:
        D_synth = []
        for s in synthetic_sets:
            D_synth.append(_compute_D_for_set((s, xmin, dist_name)))
    D_synth = np.array(D_synth)
    # p-value: proportion of synthetic distances >= original distance
    p_value = np.mean(D_synth >= D_orig)

    return p_value


def evaluate_fit(
    drops,
    xmin,
    dist_name="truncated_power_law",
    parallel=True,
    verbose=False,
):
    if verbose:
        print("nr of drops:", len(drops))
    if len(drops) < 200:
        print("Warning: this is not a lot of data, the p-value might not be reliable.")
    drop_sum = np.sum(drops)

    # Check if p has already been calculated for these drops
    # by assuming that the mean is unique enough
    # The files should be stored in the bootstrapData folder and saved as a json file
    # with the name "p_{mean}_{xmin}.json"
    import os
    import json

    nr_sets = 2500
    p_file = f"bootstrapData/p_{drop_sum}_{xmin}_{nr_sets}.json"
    if os.path.exists(p_file):
        with open(p_file, "r") as f:
            p = json.load(f)
        if verbose:
            print(f"Strained p-value from {p_file}")
    else:
        if verbose:
            print("Generating synthetic data...")
        sets = create_synthetic_data(
            drops,
            xmin=xmin,
            nrSets=nr_sets,
            dist_name=dist_name,
        )
        p = goodnessOfFit(drops, sets, xmin, dist_name=dist_name, parallel=parallel)
        # Save the p-value to a file
        with open(p_file, "w") as f:
            json.dump(p, f)
    # if verbose:
    #     print(
    #         f"p-value for fit: {p:.3f}, ie. {p * 100:.1f}% of synthetic sets had a worse fit"
    #     )
    #     print(
    #         "If p > 0.1, the fit is likely a good fit. (This also depends on the number of drops.)"
    #     )
    return p


if __name__ == "__main__":
    # User parameters
    res = 10
    debug = False
    if debug:
        res = 3
    csvPath = "/Volumes/data/MTS2D_output/unfixed_simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    csvPath = "/Volumes/data/MTS2D_output/simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    plotPath = "Plots/powerLaw/"
    outputType = ".png"
    df = pd.read_csv(csvPath)
    xmins = np.logspace(-6.5, -5, num=res, base=10)
    strainLim = [0.3, 3]
    window_steps = res
    window_width = 0.5
    plot_power_law_map(
        df=df,
        strainLim=strainLim,
        xmins=xmins,
        window_steps=window_steps,
        window_width=window_width,
        debug=debug,
        use_confidence_color=True,
        syntheticData=False,
        syntheticExponent=1,
    )

    if debug:
        make_debug_plot(xmins, strainLim=strainLim)

    xmin = 1e-6
    strainLim = [1, 3]
    fig, ax = plt.subplots()
    drops, _, _ = get_drops_in_windows(csvPath, strainLim)
    drops = drops[0]  # we only have one window, so we take the first one
    fit = powerlaw.Fit(drops, xmin=xmin)
    title = rf"$\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f},  $E_{{\mathrm{{min}}}}$={xmin:.2e}"
    plot_data_and_fit(fit, ax, xmin, title)
    p = evaluate_fit(drops, xmin, parallel=True)
    print(p)
    plt.show()
