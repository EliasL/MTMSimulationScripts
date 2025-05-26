import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm, colors
from matplotlib.ticker import LogFormatter
import powerlaw
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import functools
import os
import glob

# Create directories for saving plots
os.makedirs("Plots/powerLaw", exist_ok=True)


def get_energy_drops(csvPath, df=None, loadLim=[-np.inf, np.inf], debug=False):
    """
    Load energy drop data from CSV, filter by load limits, and return drops.
    If debug=True, plot intermediate energy and drop traces.
    """
    if df is None:
        df = pd.read_csv(csvPath)
    diffs = df["avg_energy_change"]
    load = df["load"]
    lim_mask = (load > loadLim[0]) & (load < loadLim[1])
    drop_mask = diffs < 0
    mask = drop_mask & lim_mask
    drops = -diffs[mask]
    if debug:
        e = df["avg_energy"]
        debug_fig, ax1 = plt.subplots()
        ax1.plot(load, e, label="Avg Energy")
        ax1.set_ylabel("Avg Energy")
        ax2 = ax1.twinx()
        ax2.plot([])  # advance color cycle
        ax2.plot(load[mask], drops, label="Energy Drops")
        ax2.set_ylabel("Drops")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2)
        ax2.set_ylim(0, drops.max() * 1.5)

        # ——— Compute 0.1%‐wide central slice ———
        mid = 0.5 * (loadLim[0] + loadLim[1])
        total_width = loadLim[1] - loadLim[0]
        slice_width = total_width * 0.05  # 1% of window
        x1, x2 = mid - slice_width / 2, mid + slice_width / 2
        zoom_mask = (load >= x1) & (load <= x2)

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
        axins.plot(load[zoom_mask], e[zoom_mask], lw=0.8)
        axins.set_xlim(x1, x2)
        axins.set_title("Zoom", fontsize=8)

        # twin‐axis for drops in the inset
        axins2 = axins.twinx()
        drops_zoom = -diffs[zoom_mask]
        axins2.plot(load[zoom_mask], drops_zoom)
        axins2.set_ylim(0, drops_zoom.max() * 1.5)

        # Save debug energy plot
        filename = (
            f"{plotPath}energy_drops_load_{loadLim[0]:.2f}_{loadLim[1]:.2f}{outputType}"
        )
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
            label += f"{name}={p:.3f}, {dist.D} "
            print(f"{name}={p:.3f}, {dist.D}")
        # For some reason, power_law does not have any parameters
        if dist_name == "power_law":
            label += f"alpha={dist.alpha:.3f}, "
            break
    # remove last comma
    label = label[:-2]
    ax.plot(x_vals, CCDF, linestyle="-", label=pretty_label(label), color=color)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$-\Delta E$ (Energy Drop)")
    plt.ylabel("Complementary CDF")
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
    csvPath=None, loadLim=None, df=None, steps=1, window_width=np.inf, debug=False
):
    if df is None:
        df = pd.read_csv(csvPath)
    load = df["load"]
    if loadLim is not None:
        lim_mask = (load > loadLim[0]) & (load < loadLim[1])
        df = df[lim_mask]
        load = df["load"]

    global_max_load = load.max()
    global_min_load = load.min()
    if window_width == np.inf:
        window_width = global_max_load - global_min_load

    if global_max_load - global_min_load < window_width:
        centers = [global_min_load + window_width / 2]
    else:
        # get list of window centers
        centers = np.linspace(
            global_min_load + window_width / 2,
            global_max_load - window_width / 2,
            steps,
        )

    drops_in_windows = []
    windows = []
    for center in centers:
        # get the window
        min_load = center - window_width / 2
        max_load = center + window_width / 2
        # get the data in the window
        drops = get_energy_drops(
            csvPath, df=df, loadLim=[min_load, max_load], debug=debug
        )
        windows.append((min_load, max_load))
        drops_in_windows.append(drops)
    return drops_in_windows, windows, centers


def plot_data_and_fit(
    ax,
    fit,
    xmin=None,
    title="",
    dist_names=[
        "truncated_power_law",
        "lognormal",
        "power_law",
        "exponential",
        "stretched_exponential",
        "lognormal_positive",
    ],
):
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


def get_window_power_law_exponents(
    xmin=-np.inf,
    debug=False,
    **kwargs,
):
    """
    We slide this window over the data and plot the power law fit for each window.
    """
    drops_in_windows, windows, centers = get_drops_in_windows(**kwargs)
    fits = []
    for drops, (min_load, max_load) in zip(drops_in_windows, windows):
        # fit the data
        fit = powerlaw.Fit(drops, xmin=xmin)
        fits.append(fit)
        if debug:
            debug_fig, debug_ax = plt.subplots()
            title = f"Window: {min_load:.2f} - {max_load:.2f}, xmin={xmin:.2e}"
            plot_data_and_fit(debug_ax, fit, xmin, title)
            debug_fig.tight_layout()
            debug_fig.show()
            # Save debug window power law plot
            filename = f"{plotPath}window_load_{min_load:.2f}_{max_load:.2f}_xmin_{xmin:.2e}{outputType}"
            debug_fig.savefig(filename)
            # to save memory, close the figure
            plt.close(debug_fig)

    # plot the exponents against the window centers
    exponents = [fit.truncated_power_law.alpha for fit in fits]

    # Compare truncated_power_law to lognormal
    R, p = [], []
    for fit in fits:
        # Likelihood ratio test comparing truncated_power_law to lognormal
        R_, p_ = fit.distribution_compare("truncated_power_law", "lognormal")
        R.append(R_)
        p.append(p_)

    return centers, exponents, R, p


def worker_get_exponents(xmin, kwargs):
    return get_window_power_law_exponents(xmin=xmin, **kwargs)


def get_power_law_surface(xmins=None, **kwargs):
    # If we debug, we don't use multiprocessing
    if kwargs.get("debug", False):
        exponent_xmin_surface, R, p = [], [], []
        for xmin in tqdm(xmins):
            centers, exponents, R_, p_ = get_window_power_law_exponents(
                xmin=xmin, **kwargs
            )
            exponent_xmin_surface.append(exponents)
            R.append(R_)
            p.append(p_)
        return centers, np.array(exponent_xmin_surface), np.array(R), np.array(p)
    else:
        # Pre-bind kwargs using functools.partial
        with ProcessPoolExecutor() as executor:
            bound_worker = functools.partial(worker_get_exponents, kwargs=kwargs)
            results = list(tqdm(executor.map(bound_worker, xmins), total=len(xmins)))

        centers = results[0][0]  # All share same centers
        exponent_xmin_surface = np.array([r[1] for r in results])
        R = np.array([r[2] for r in results])
        p = np.array([r[3] for r in results])
        return centers, exponent_xmin_surface, R, p


def plot_power_law_map(
    csvPath=None,
    xmins=None,
    df=None,
    loadLim=[-np.inf, np.inf],
    window_steps=20,
    window_width=0.3,
    debug=False,
    use_confidence_color=False,
):
    # convert exponents to numpy array
    centers, exponent_xmin_surface, R, p = get_power_law_surface(
        csvPath=csvPath,
        xmins=xmins,
        df=df,
        loadLim=loadLim,
        steps=window_steps,
        window_width=window_width,
        debug=debug,
    )

    # Now we can plot a surface of the exponents on the z axis, centers on the x axis, and xmins on the y axis
    fig = plt.figure()
    if use_confidence_color:
        ax = fig.add_subplot(projection="3d")
        ax.set_zlabel("Exponent")
    else:
        ax = fig.add_subplot()

    ax.set_title("Power law exponents")
    ax.set_xlabel("Load")
    ax.set_ylabel(r"$\log_{10}(\Delta E_{\mathrm{min}})$")  # Changed label

    figType = "p" if use_confidence_color else "exp"

    # Choose plotting logic based on confidence‐color flag
    if use_confidence_color:
        # Two log‐scaled colorbars for R > 0 and R < 0
        eps = 1e-19
        p_safe = np.clip(p, eps, None)
        vmin, vmax = p_safe.min(), p_safe.max()
        norm_pos = colors.LogNorm(vmin=vmin, vmax=vmax)
        norm_neg = colors.LogNorm(vmin=vmin, vmax=vmax)
        cmap_pos = plt.get_cmap("Greens_r")
        cmap_neg = plt.get_cmap("Reds_r")
        sm_pos = cm.ScalarMappable(norm=norm_pos, cmap=cmap_pos)
        sm_neg = cm.ScalarMappable(norm=norm_neg, cmap=cmap_neg)

        facecolors = np.zeros(R.shape + (4,))
        mask_pos = R > 0
        mask_neg = R < 0
        facecolors[mask_pos] = sm_pos.to_rgba(p_safe[mask_pos])
        facecolors[mask_neg] = sm_neg.to_rgba(p_safe[mask_neg])

        X, Y = np.meshgrid(centers, np.log10(xmins))
        ax.plot_surface(
            X,
            Y,
            exponent_xmin_surface,
            facecolors=facecolors,
            shade=False,
            antialiased=False,
        )

        cbar_pos = fig.colorbar(sm_pos, ax=ax, pad=0.02, shrink=0.5, aspect=10)
        cbar_pos.set_label("p-value Truncated Power Law")
        cbar_pos.ax.yaxis.set_major_formatter(LogFormatter(10.0))

        cbar_neg = fig.colorbar(sm_neg, ax=ax, pad=0.14, shrink=0.5, aspect=10)
        cbar_neg.set_label("p-value Lognormal")
        cbar_neg.ax.yaxis.set_major_formatter(LogFormatter(10.0))

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
        fig.colorbar(pcm, ax=ax, pad=0.1, aspect=10, label="Exponent")

    fig.tight_layout()
    plt.show()
    # Save final power law surface plot
    filename = (
        f"{plotPath}power_law_surface_"
        f"load_{loadLim[0]:.2f}_{loadLim[1]:.2f}_"
        f"xmin_{xmins[0]:.2e}_{xmins[-1]:.2e}_"
        f"steps_{window_steps}_width_{window_width:.2f}_"
        f"{figType}{outputType}"
    )
    fig.savefig(filename, dpi=300)


def make_debug_plot(xmins, loadLim=None):
    # Create debug plot grids for each xmin
    for xmin in xmins:
        # Find all saved fit plots for this xmin
        pattern = f"{plotPath}window_load_*_xmin_{xmin:.2e}{outputType}"
        fit_files = sorted(glob.glob(pattern))
        if not fit_files:
            print(f"No fit debug files found for xmin {xmin:.2e}")
            continue

        n = len(fit_files)
        fig, axes = plt.subplots(2, n, figsize=(n * 6, 10))

        for i, fit_file in enumerate(fit_files):
            base = os.path.basename(fit_file)
            # Extract the load range string between "window_load_" and "_xmin"
            load_range = base[len("window_load_") : base.find("_xmin")]

            # Load and display the energy drops image
            energy_file = f"{plotPath}energy_drops_load_{load_range}{outputType}"
            if os.path.exists(energy_file):
                img_energy = plt.imread(energy_file)
                axes[0, i].imshow(img_energy)
            else:
                axes[0, i].text(0.5, 0.5, "Missing image", ha="center", va="center")
            axes[0, i].axis("off")
            # axes[0, i].set_title(f"Energy drops\nload {load_range}")

            # Load and display the fit plot
            img_fit = plt.imread(fit_file)
            axes[1, i].imshow(img_fit)
            axes[1, i].axis("off")
            # axes[1, i].set_title(f"xmin={xmin:.2e}")

        # fig.suptitle(f"Debug plots for xmin={xmin:.2e}")
        fig.tight_layout()
        # Save the debug plot
        debug_filename = f"{plotPath}debug_fit_plots_xmin_{xmin:.2e}{outputType}"
        fig.savefig(debug_filename)
        # plt.show()


if __name__ == "__main__":
    # User parameters
    debug = False
    csvPath = "/Volumes/data/MTS2D_output/unfixed_simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    csvPath = "/Volumes/data/MTS2D_output/simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    plotPath = "Plots/powerLaw/"
    outputType = ".png"
    df = pd.read_csv(csvPath)
    res = 30
    xmins = np.logspace(-8, -5, num=res, base=10)
    loadLim = [0.3, 3]
    window_steps = res
    window_width = 0.5
    plot_power_law_map(
        df=df,
        loadLim=loadLim,
        xmins=xmins,
        window_steps=window_steps,
        window_width=window_width,
        debug=debug,
        use_confidence_color=True,
    )
    # if debug:
    #     make_debug_plot(xmins, loadLim=loadLim)

    # xmin = 1e-6
    # loadLim = [1, 3]
    # fig, ax = plt.subplots()
    # drops, _, _ = get_drops_in_windows(csvPath, loadLim)
    # fit = powerlaw.Fit(drops[0], xmin=xmin)
    # title = f"Window: {loadLim[0]:.2f} - {loadLim[1]:.2f}, xmin={xmin:.2e}"
    # plot_data_and_fit(ax, fit, xmin, title)
    # plt.show()
