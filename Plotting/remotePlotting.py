import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import threading
import pandas as pd
from .makePlots import (
    makePlot,
    makeAverageComparisonPlot,
    makeLogPlotComparison,
    add_power_law_line,
    duration_to_seconds,
)
import matplotlib.pyplot as plt
from .fixLineNumbers import fix_csv_files_in_data_folder, fix_csv_files
from Management.connectToCluster import getServerUserName
from tqdm import tqdm
import numpy as np

# Add Management to sys.path (used to import files)
sys.path.append(str(Path(__file__).resolve().parent.parent / "Management"))
# Now we can import from Management
from Management.connectToCluster import connectToCluster, Servers, download_folders
from Management.configGenerator import ConfigGenerator, SimulationConfig

FOLDER_PATH = "/Users/elias/Work/PhD/Code/remoteData"
FOLDER_PATH = "/Users/eliaslundheim/work/PhD/remoteData"
MACRO_PATH = os.path.join(FOLDER_PATH, "macro")
PLOTS_PATH = os.path.join(FOLDER_PATH, "plots")
RAW_DATA_PATH = os.path.join(FOLDER_PATH, "data")


def handleLocalPath(dataPath, configs, returnCsv=True):
    local_data_folder_name = "MTS2D_output"
    names = [config.generate_name(False) for config in configs]

    existing_paths = []  # This will store the paths to existing data files
    base_path = os.path.join(dataPath, local_data_folder_name)

    for name in names:
        # Construct the path to the specific data folder for this configuration
        folder_path = os.path.join(base_path, name)
        if returnCsv:
            # Construct the path to the macroData.csv file within the data folder
            file_path = os.path.join(folder_path, "macroData.csv")
        else:
            file_path = folder_path

        # Check if the file exists
        if os.path.exists(file_path):
            # If it exists, add its path to the list of existing paths
            existing_paths.append(file_path)

    # fix_csv_files(existing_paths, use_tqdm=False)
    return existing_paths


# Shared variables
completed_servers = 0
nr_files = 0
lock = threading.Lock()  # Create a lock for thread-safe operations


def update_progress(total_files):
    with lock:  # Acquire lock before modifying shared variables
        global completed_servers, nr_files
        sys.stdout.write(
            f"\r{completed_servers}/{len(Servers.servers)} servers, {nr_files}/{total_files} files"
        )
        sys.stdout.flush()


def get_csv_from_server(server, configs):
    global nr_files
    if "espci.fr" not in server:
        # server is actually not a ssh address, but a local path
        return handleLocalPath(server, configs)

    # Connect to the server
    ssh = connectToCluster(server, False)

    # Check if /data2 exists, otherwise use /data
    stdin, stdout, stderr = ssh.exec_command(
        "if [ -d /data2 ]; then echo '/data2'; else echo '/data'; fi"
    )
    base_dir = stdout.read().strip().decode()

    user = getServerUserName(server)
    data_path = os.path.join(base_dir, user)

    remote_folder_name = "MTS2D_output"

    # List all folders within the output folder
    command = f"cd /{data_path}/{remote_folder_name}; ls -d */"
    stdin, stdout, stderr = ssh.exec_command(command)
    folders = stdout.read().strip().decode().split("\n")
    folders = [folder.rstrip("/") for folder in folders]  # Clean up folder names
    names = [config.generate_name(False) for config in configs]
    newPaths = []
    # This line ensures the MTS2D folder is created
    os.makedirs(MACRO_PATH, exist_ok=True)

    # Using ThreadPoolExecutor to download files in parallel
    with ThreadPoolExecutor(max_workers=7) as executor:
        future_to_name = {
            executor.submit(
                download_file,
                name,
                folders,
                data_path,
                remote_folder_name,
                MACRO_PATH,
                ssh,
            ): name
            for name in names
        }
        for future in as_completed(future_to_name):
            result = future.result()
            if result:
                newPaths.append(result)
                with lock:  # Safe update
                    nr_files += 1
                update_progress(len(names))

    # This fix is needed due to an old bug in the C++ program (fixed now)
    # so when downloading some data from the server, we need a fix
    # fix_csv_files(newPaths, use_tqdm=False)
    return newPaths


def download_file(name, folders, data_path, remote_folder_name, folder_path, ssh):
    if name in folders:
        attempts = 0
        max_attempts = 3
        while attempts < max_attempts:
            try:
                sftp = ssh.open_sftp()
                remote_file_path = (
                    f"{data_path}/{remote_folder_name}/{name}/macroData.csv"
                )
                local_file_path = os.path.join(folder_path, f"{name}.csv")
                sftp.get(remote_file_path, local_file_path)
                sftp.close()
                return local_file_path
            except Exception as e:
                attempts += 1
                time.sleep(
                    random.uniform(1, 3)
                )  # Random delay to prevent synchronized reconnection attempts
                # print(f"Attempt {attempts} failed for {name}: {e}")
                if attempts >= max_attempts:
                    print(f"Error downloading {name}: {e}")
    return None


import os
import time
import pandas as pd


def search_for_cvs_files(configs, useOldFiles=False, forceUpdate=False):
    """
    Searches for CSV files corresponding to given configurations in predefined folders.

    - If `forceUpdate` is True, returns immediately with no files.
    - Only includes files that are less than 12 hours old unless `useOldFiles` is True.
    - Ensures all search directories exist.
    - Files are considered valid if their "Est_time_remaining" column is 0 or missing.

    Returns:
        paths (list): List of valid file paths.
        remaining_configs (list): List of configurations still needing files.
    """

    # If forced update, return no files.
    if forceUpdate:
        return [], configs

    paths, remaining_configs = [], []
    # If an incomplete file is older than x hours, we update it
    updateAfterHours = 12
    search_folders = ["/tmp/MTS2D", MACRO_PATH]  # Directories to search in

    for i, folder in enumerate(search_folders):
        os.makedirs(folder, exist_ok=True)  # Ensure folder exists
        # Check if it's the last search folder
        last_search_folder = i == len(search_folders) - 1

        # Get existing CSV file names (without extensions) for quick lookup
        existing_files = {
            os.path.splitext(f)[0]
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        }

        for config in configs:
            file_path = os.path.join(folder, f"{config.name}.csv")

            if config.name in existing_files:
                # Read estimated time remaining from CSV file
                est_time_remaining = pd.read_csv(file_path)["Est_time_remaining"]
                time_remaining = (
                    duration_to_seconds(est_time_remaining.iloc[-1])
                    if not est_time_remaining.empty
                    else None
                )

                if time_remaining is None or time_remaining > 0:
                    # File might still be processing; check age
                    file_mod_time = os.path.getmtime(file_path)
                    if time_remaining is not None and (
                        time.time() - file_mod_time < updateAfterHours * 3600
                        or useOldFiles
                    ):
                        # Include if recent enough
                        paths.append(file_path)
                    else:
                        # Still needs processing
                        remaining_configs.append(config)
                else:
                    # File is done processing
                    paths.append(file_path)
            elif last_search_folder:
                # If no file found, add to remaining configs
                remaining_configs.append(config)

    return paths, remaining_configs


# Converts config to a path, but if given paths, it matches the given
# config with the path is is most likely to corespond to.
# If a config could match with two paths, the first path found is chosen
def configToPath(config, paths=None):
    if paths:
        # Search for the coresponding path and config
        matches = [path for path in paths if config.name in path]
        if matches:
            return matches[0]
        else:
            return None
    else:
        return f"{MACRO_PATH}/{config.name}.csv"


def flatToStructure(config_groups, label_groups, found_paths=None):
    # This function searches for where the file WOULD be if it was
    # successfully downloaded, therefore preserving the structure of the groups
    paths = []
    labels = []
    for config_group, label_group in zip(config_groups, label_groups):
        matchingPaths, matchingLabels = rematchPathsAndLabels(
            config_group, label_group, found_paths
        )
        if matchingPaths:
            paths.append(matchingPaths)
            labels.append(matchingLabels)
    return paths, labels


# Given two lists of matched configs and labels and and unstructured list of paths,
# this function returns the same lists of labels and paths, but such that the
# order they have correspond to eachother and match the order of the configs.
def rematchPathsAndLabels(configs, labels, paths):
    matched_paths = []
    matched_labels = []
    assert len(configs) == len(labels)
    for config, label in zip(configs, labels):
        path = configToPath(config, paths)
        if path and os.path.isfile(path):
            matched_paths.append(path)
            matched_labels.append(label)
        else:
            print(f"Warning: missing file:\n {path}")
    return matched_paths, matched_labels


def flattenConfigList(listOfListsOfConfigs):
    # Check if the first element is a list and contains instances of SimulationConfig
    if isinstance(listOfListsOfConfigs[0], SimulationConfig):
        # Here we don't need to flaten at all
        return listOfListsOfConfigs
    elif isinstance(listOfListsOfConfigs[0][0], SimulationConfig):
        # Use list comprehension to flatten the list of lists
        return [config for sublist in listOfListsOfConfigs for config in sublist]
    else:
        raise ValueError(
            "The input must be a list or a list of lists of SimulationConfig instances."
        )


# This function searches all the servers for the given config file,
# downloads the csv file associated with the config file to a temp file,
# and returns the new local path to the csv
def get_csv_files(all_configs, labels=[], useOldFiles=False, forceUpdate=False):
    nested = False
    config_groups = all_configs
    if not isinstance(all_configs[0], SimulationConfig):
        nested = True
        all_configs = [config for sublist in config_groups for config in sublist]

    global completed_servers, nr_files

    completed_servers, nr_files = 0, 0
    # First check if the files have already been downloaded
    paths, remaining_configs = search_for_cvs_files(
        all_configs, useOldFiles, forceUpdate
    )
    if len(remaining_configs) == 0:
        print("All files already downloaded.")
        if nested:
            paths, labels = flatToStructure(config_groups, labels, paths)
        return paths, labels
    elif len(paths) != 0:
        print(
            f"{len(paths)} files found, searching for the remaining {len(remaining_configs)}."
        )
    if len(paths) == 0 and useOldFiles:
        raise Exception("No files found!")

    # Second check local path to see if we can avoid checking the servers
    localPaths = get_csv_from_server(Servers.local_path_mac, remaining_configs)
    if len(localPaths) == len(remaining_configs):
        # We have found all the requested files, so we don't need to search more.
        print(f"{len(localPaths)} files found. Not searching servers.")
        paths = paths + localPaths
        if nested:
            paths, labels = flatToStructure(config_groups, labels, paths)
        return paths, labels

    print("Searching servers for files...")
    # Use ThreadPoolExecutor to execute find_data_on_server in parallel across all servers
    # get_csv_from_server(Servers.poincare, configs)
    with ThreadPoolExecutor(max_workers=len(Servers.servers)) as executor:
        future_to_server = {
            executor.submit(get_csv_from_server, server, remaining_configs): server
            for server in Servers.servers
        }
        for future in as_completed(future_to_server):
            server = future_to_server[future]
            with lock:
                completed_servers += 1  # Increment completed count
            update_progress(len(remaining_configs))
            try:
                server_paths = future.result()
                if server_paths:
                    # We extend, not append
                    paths += server_paths
            except Exception as exc:
                print(f"\n{server} generated an exception: {exc}")
                print("Trying to use old files... ")
                if useOldFiles is False:
                    return get_csv_files(config_groups, useOldFiles=True, labels=labels)
    print("")  # New line from progress indicator
    if nested:
        paths, labels = flatToStructure(config_groups, labels)
    else:
        # The paths are returned in psedu random order, so we need to
        # match them with their correct label again
        paths, labels = rematchPathsAndLabels(all_configs, labels, paths)
    return paths, labels


def get_csv_from_folder(folderPath):
    return [
        os.path.join(folderPath, f)
        for f in os.listdir(folderPath)
        if f.endswith(".csv")
    ]


def get_folders_from_servers(configs, fix=True):
    configs = flattenConfigList(configs)
    print("Searching servers for folders...")
    # Use ThreadPoolExecutor to execute find_data_on_server in parallel across all servers
    pathsAndConfig = []
    with ThreadPoolExecutor(max_workers=len(Servers.servers)) as executor:
        future_to_server = {
            executor.submit(download_folders, server, configs, RAW_DATA_PATH): server
            for server in Servers.servers
        }
        for future in as_completed(future_to_server):
            pAndC = future.result()
            pathsAndConfig.extend(pAndC)

    new_paths = [None] * len(configs)  # old order
    for i in range(len(configs)):
        for p, c in pathsAndConfig:
            if c.name == configs[i].name:
                new_paths[i] = p
                continue
    # Remove none objects not found
    new_paths = [c for c in new_paths if c is not None]

    if fix:
        fix_csv_files_in_data_folder(Path(new_paths[0]).parent)

    # We also check local files
    localPaths = handleLocalPath(Servers.local_path_mac, configs, returnCsv=False)
    return localPaths + new_paths


def set_font_size(ax, axis_size=17, legend_size=17, tick_size=17, extra_size=0):
    # Add extra_size to the main font sizes
    axis_size += extra_size
    legend_size += extra_size
    tick_size += extra_size

    # Set axis labels font size
    ax.set_xlabel(ax.get_xlabel(), fontsize=axis_size)
    ax.set_ylabel(ax.get_ylabel(), fontsize=axis_size)

    # Adjust the font size for the legend, if it exists
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_size)

    # Set tick labels font size for both x and y axes
    ax.tick_params(axis="both", which="major", labelsize=tick_size)


def synchronize_y_limits(ax_list):
    """
    Synchronize the y-limits of a list of Axes objects based on the overall min and max y-values.

    Parameters:
    ax_list (list): List of Matplotlib Axes objects.
    """
    min_y = float("inf")
    max_y = float("-inf")
    ax_list = np.array(ax_list).flatten()

    # Iterate over each Axes object to find the overall min and max y-values
    for ax in ax_list:
        # Get data from lines (e.g., plot, plot_date)
        for line in ax.get_lines():
            y_data = line.get_ydata()
            if len(y_data) > 0:
                min_y = min(min_y, np.nanmin(y_data))
                max_y = max(max_y, np.nanmax(y_data))

        # Get data from scatter plots
        for collection in ax.collections:
            offsets = collection.get_offsets()
            if offsets.size > 0:
                y_data = offsets[:, 1]  # Extract y-values
                min_y = min(min_y, np.nanmin(y_data))
                max_y = max(max_y, np.nanmax(y_data))

    # Set the y-limits for all Axes objects
    for ax in ax_list:
        if ax.get_yscale() == "log":
            ax.set_ylim(min_y * 0.5, max_y * 2)
        else:
            ax.set_ylim(min_y, max_y)


def createVideoes(configs, paths=None, **kwargs):
    from .makeAnimations import makeAnimations

    if not paths:
        # Download the folders associated with the configs from the server
        paths = get_folders_from_servers(configs, fix=False)
    for path in paths:
        makeAnimations(path, **kwargs)


def createPlotsWithImages(configs, paths, metric, **kwargs):
    if not paths:
        # Download the folders associated with the configs from the server
        paths = get_folders_from_servers(configs)

    base = 5 if len(configs) == 3 else 7
    # Create a figure with subplots, one for each configuration
    fig, axes = plt.subplots(1, len(configs), figsize=(base * len(configs), base))

    # If there's only one configuration, axes won't be a list, so convert it into one
    if len(configs) == 1:
        axes = [axes]

    colors = {"FIRE": "#d24646", "LBFGS": "#008743", "CG": "#ffa701"}
    colors = {"LBFGS": "#56BD94", "CG": "#9456BD", "FIRE": "#BD9456"}
    sp = len(configs) == 1  # Single plot
    # Loop over the configurations, paths, and axes
    for ax, path, config, mark in zip(axes, paths, configs, "abc"):
        # Call the provided plot function (either makeStressPlot or makeEnergyPlot)
        fig, ax = makePlot(
            path + "/macroData.csv",
            name=config.name + f"_{metric}+.pdf",
            add_images=True,
            metric=metric,
            ax=ax,
            fig=fig,
            save=False,
            xlim=(0.15, 1),
            colors=[colors[config.minimizer]],
            use_y_axis_name=config.minimizer == "LBFGS" if not sp else True,
            add_cbar=config.minimizer == "FIRE" if not sp else True,
            mark=mark if not sp else None,
            legend=config.minimizer,
            legend_loc="upper left",
            mark_fontsize=20 + 2 * len(configs),
            **kwargs,
        )
        set_font_size(ax, extra_size=2 * len(configs))
        fig.tight_layout()

    if sp:
        method = configs[0].minimizer
    else:
        method = "combined"

    # Save the combined figure
    plt.savefig(f"Plots/{method}_{metric}_plots.pdf")


def stressPlotWithImages(configs, paths=None):
    createPlotsWithImages(
        configs=configs,
        paths=paths,
        ylim=(0, 0.27),
        mark_pos=(0.85, 0.15),
        image_pos=[
            [0.3, 0.01],  # first image, bottom middle
            [0.03, 0.5],  # second image, upper left
            [0.6, 0.55],  # upper right
        ],
        image_size=[0.37, 0.4, 0.4],
        Y="Avg_RSS",
        metric="stress",
    )


def energyPlotWithImages(configs, paths=None):
    createPlotsWithImages(
        configs=configs,
        paths=paths,
        ylim=(0, 0.047),
        mark_pos=(0.7, 0.95),
        image_pos=[
            [0.02, 0.5],  # first image, upper left
            [0.29, 0.02],  # second image, lower center
            [0.6, 0.1],  # upper right
        ],
        image_size=[0.4, 0.4, 0.4],
        Y="Avg_energy",
        metric="energy",
    )


def plotWholeRangePowerLaw(paths, Y, **kwargs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Define limits
    if Y == "Avg_energy":
        ylim = [8e-3, 2e7]
    elif Y == "Avg_RSS":
        ylim = [1e-5, 2e5]
    for ax, group, method, mark in zip(axes, paths, ["L-BFGS", "CG", "FIRE"], "abc"):
        kwargs["labels"] = [[method]]
        makeLogPlotComparison(
            [group],
            outerStrainLims=(0.31, 1),
            innerStrainLims=(1, np.inf),
            plot_post_yield=False,
            save=False,
            use_y_axis_name=method == "L-BFGS",
            Y=Y,
            ax=ax,
            fig=fig,
            legend_loc="lower left",
            show=False,
            add_fit=Y == "Avg_RSS",
            mark=mark,
            mark_pos=(0.85, 0.9),
            **kwargs,
        )
        if Y == "Avg_energy":
            add_power_law_line(ax, -0.85, [5e-7, 3e-4], 7e-1)
            add_power_law_line(ax, -2.5, [3e-4, 9e-3], 1e-6, linestyle="-.")
        if Y == "Avg_RSS":
            add_power_law_line(ax, -2.8, [3e-5, 5e-4], 5e-9, linestyle="-.")
        set_font_size(ax)

    synchronize_y_limits(axes)

    fig.tight_layout()
    name = "energy" if Y == "Avg_energy" else "stress"
    # Display all plots in a row
    plt.savefig(f"Plots/combined_{name}_powerlaw_full_range.pdf")


def plotPreYieldPowerLaw(paths, Y, **kwargs):
    # Define preyield range
    preYield = (0.15, 0.45)
    # Define limits

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, group, method, mark in zip(range(3), paths, ["L-BFGS", "CG", "FIRE"], "abc"):
        kwargs["labels"] = [[method]]

        makeAverageComparisonPlot(
            [group],
            Y=Y,
            xlim=preYield,
            ax=axes[0, i],
            use_y_axis_name=method == "L-BFGS",
            fig=fig,
            save=False,
            mark=mark.upper(),
            mark_pos=(0.85, 0.1),
            **kwargs,
        )

        makeLogPlotComparison(
            [group],
            outerStrainLims=(preYield[0], 1),
            innerStrainLims=(preYield[1], 1),
            plot_post_yield=False,
            save=False,
            use_y_axis_name=method == "L-BFGS",
            Y=Y,
            ax=axes[1, i],
            fig=fig,
            legend_loc="lower left",
            mark=mark,
            mark_pos=(0.85, 0.9),
            **kwargs,
        )
        set_font_size(axes[0, i])
        set_font_size(axes[1, i])

    synchronize_y_limits(axes[0])
    synchronize_y_limits(axes[1])

    fig.tight_layout()
    name = "energy" if Y == "Avg_energy" else "stress"
    # Display all plots in a row
    plt.savefig(f"Plots/combined_{name}_powerlaw_preYield.pdf")


def plotPostYieldPowerLaw(paths, Y, **kwargs):
    # Define preyield range
    postYield = (0.7, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, group, method, mark in zip(range(3), paths, ["L-BFGS", "CG", "FIRE"], "abc"):
        kwargs["labels"] = [[method]]

        makeAverageComparisonPlot(
            [group],
            Y=Y,
            xlim=postYield,
            ax=axes[0, i],
            use_y_axis_name=method == "L-BFGS",
            fig=fig,
            save=False,
            mark=mark,
            mark_pos=(0.85, 0.1),
            **kwargs,
        )

        makeLogPlotComparison(
            [group],
            outerStrainLims=(0.31, postYield[1]),
            innerStrainLims=(0.31, postYield[0]),
            plot_pre_yield=False,
            save=False,
            use_y_axis_name=method == "L-BFGS",
            Y=Y,
            ax=axes[1, i],
            fig=fig,
            legend_loc="lower left",
            # ylim=log_ylim,
            mark=mark,
            mark_pos=(0.85, 0.9),
            **kwargs,
        )
        set_font_size(axes[0, i])
        set_font_size(axes[1, i])

    synchronize_y_limits(axes[0])
    synchronize_y_limits(axes[1])
    fig.tight_layout()
    name = "energy" if Y == "Avg_energy" else "stress"
    # Display all plots in a row
    plt.savefig(f"Plots/combined_{name}_powerlaw_postYield.pdf")


def plotWindowPowerLaw(paths, Y, show_lambda=False, **kwargs):
    # Define limits
    if Y == "Avg_energy":
        ylim = [0.62, 0.83]
    elif Y == "Avg_RSS":
        ylim = [0.95, 1.34]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, group, method, mark in zip(axes, paths, ["L-BFGS", "CG", "FIRE"], "abc"):
        kwargs["labels"] = [[method]]

        makeLogPlotComparison(
            [group],
            plot_post_yield=False,
            save=False,
            use_y_axis_name=method == "L-BFGS",
            Y=Y,
            ax=ax,
            fig=fig,
            legend_loc="lower right",
            window=True,
            ylim=ylim,
            show_lambda=show_lambda,
            mark=mark,
            mark_pos=(0.85, 0.9),
            **kwargs,
        )
        set_font_size(ax)

    fig.tight_layout()
    name = "energy" if Y == "Avg_energy" else "stress"
    name = name + "_withLambda" if show_lambda else name
    # Display all plots in a row
    plt.savefig(f"Plots/combined_window_{name}_powerlaw.pdf")


def plotAverage(config_groups, labels, useStress=False, **kwargs):
    paths, labels = get_csv_files(
        config_groups, labels=labels, useOldFiles=False, forceUpdate=False
    )
    kwargs["labels"] = labels
    yColumns = ["Avg_energy"]
    if useStress:
        yColumns.append("Avg_RSS")
    print("Plotting...")
    for Y in yColumns:
        makeAverageComparisonPlot(paths, Y=Y, **kwargs)


def plotTime(config_groups, labels, **kwargs):
    paths, labels = get_csv_files(
        config_groups, labels=labels, useOldFiles=False, forceUpdate=False
    )
    print("Plotting...")
    for Y in [
        "Minimization_time",
        "Nr_LBFGS_iterations",
        "Nr_LBFGS_func_evals",
    ]:  # "Write_time", "Run_time", "Est_time_remaining"]:
        fig, ax = makePlot(
            paths,
            Y=Y,
            name=f"{Y.replace(' ', '_')}.pdf",
            labels=labels,
            legend=True,
            use_title=True,
            **kwargs,
        )


def plotEnergy(configs, labels, name="Energy", **kwargs):
    paths, labels = get_csv_files(
        configs, labels=labels, useOldFiles=False, forceUpdate=False
    )
    fig, ax = makePlot(
        paths,
        name=f"{name}.pdf",
        labels=labels,
        legend=True,
        **kwargs,
    )


def plotLog(config_groups, labels, **kwargs):
    paths, labels = get_csv_files(
        config_groups, labels=labels, useOldFiles=False, forceUpdate=False
    )
    kwargs["labels"] = labels

    print("Plotting...")
    # Iterate over the groups and methods, and plot each one in a separate subplot
    for Y, dropLim in zip(
        ["Avg_energy", "Avg_RSS"],
        [[5e-7, None], [5e-4, None]],
    ):
        kwargs["dropLim"] = dropLim
        # makeAverageComparisonPlot(paths, Y=Y, **kwargs)
        ## makeLogPlotComparison(paths, Y=Y, **kwargs)
        plotWholeRangePowerLaw(paths, Y, **kwargs)
        plotPreYieldPowerLaw(paths, Y, **kwargs)
        # plotPostYieldPowerLaw(paths, Y, **kwargs)
        # plotWindowPowerLaw(paths, Y, **kwargs)

    # makeLogPlotComparison(paths, f"{name} - EnergyPowerLawWindow", window=True, **kwargs)
    # makeEnergyAvalancheComparison(paths, f"{name} - Histogram", **kwargs)
    # makeItterationsPlot(paths, f"{name}Itterations.pdf", **kwargs)


if __name__ == "__main__":
    seeds = range(0, 60)
    configs = ConfigGenerator.generate_over_seeds(
        seeds,
        rows=60,
        cols=60,
        startLoad=0.15,
        nrThreads=1,
        loadIncrement=1e-5,
        maxLoad=1.0,
        LBFGSEpsx=1e-6,
        minimizer="LBFGS",
        scenario="simpleShear",
    )
    # paths = get_csv_files(configs)
    paths = get_csv_from_folder(
        "/Volumes/data/MTS2D_output/FailedStrangeFireSimulatinos"
    )
    if paths:
        makePlot(paths, "ParamExploration.pdf", show=True, legend=False, ylim=(-100, 2))

        # makeTimePlot(paths, "Run time.pdf", show=True, legend=True)
        # makeItterationsPlot(paths, "ParamExploration.pdf", show=True)
        # makePowerLawPlot(paths, "ParamExplorationPowerLaw.pdf", show=True)
    else:
        print("No files found")
