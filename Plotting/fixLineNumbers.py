import os
import pandas as pd
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor


def fix_csv_files(paths, use_tqdm=True):
    if len(paths) == 0:
        return
    fix_missing_column(paths, use_tqdm=use_tqdm)

    if isinstance(paths[0], list):
        paths = [item for sublist in paths for item in sublist]
    # Iterate over all files in the given folder
    for path in tqdm(paths, disable=not use_tqdm):
        if path.endswith(".csv"):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(path)

            # Create a Series that tracks the maximum value encountered so far
            cummax_series = df["Load"].cummax()

            # Create a boolean mask where the current value is less than the maximum encountered
            overlap_mask = df["Load"] < cummax_series

            # Drop the rows where overlap_mask is True
            df_cleaned = df[~overlap_mask].reset_index(drop=True)

            # Save the modified DataFrame back to the same CSV file
            df_cleaned.to_csv(path, index=False)


def process_file(path):
    # This function processes a single file
    temp_path = f"{path[:-3]}.temp.csv"

    # Open the original file for reading and a temporary file for writing
    with open(path, "r") as file, open(temp_path, "w") as temp_file:
        firstLine = True
        for line in file:
            if not line[:7] == "Line nr" and firstLine:
                # If the first line doesn't start with "Line nr"
                # then we break. We skip processing because original
                # format doesn't match what we expect.
                os.remove(temp_path)
                return
            else:
                firstLine = False

                # Get the first element
                n, rest = line.split(sep=",", maxsplit=1)
                # We want to remove the line numbers, but keep the floats
                # that might come if the lines numbers have been removed in
                # the middle of the file
                if "." not in n:
                    # Remove it
                    temp_file.write(rest)
                else:
                    temp_file.write(line)

    # After processing, replace the original file with the modified one
    os.replace(temp_path, path)


def fix_missing_column(paths, use_tqdm=True):
    # This function processes each file in the given paths using multithreading.
    # Each file will be processed in a separate thread.
    # A progress bar will be displayed if use_tqdm is True.

    # Create a ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        # Submit each file processing task to the executor
        if use_tqdm:
            # Wrap the executor with tqdm to show the progress bar
            list(tqdm(executor.map(process_file, paths), total=len(paths)))
        else:
            list(executor.map(process_file, paths))


def fix_csv_files_in_folder(folder_path, use_tqdm=True):
    paths = [
        os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
    ]
    fix_csv_files(paths, use_tqdm=use_tqdm)


def fix_csv_files_in_data_folder(folder_path, max_workers=10):
    # Get the list of all folders in the given directory
    folders = [
        f
        for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ]

    # Define a helper function to process a single folder
    def process_folder(folder):
        path = os.path.join(folder_path, folder)
        fix_csv_files_in_folder(path, use_tqdm=False)

    # Use ThreadPoolExecutor to process folders in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Wrap the executor's map call with tqdm for progress tracking
        list(tqdm(executor.map(process_folder, folders), total=len(folders)))


if __name__ == "__main__":
    # Example usage
    folder_path = "/Users/elias/Work/PhD/Code/remoteData/"
    folder_path = "/Users/eliaslundheim/work/PhD/remoteData/"

    fix_missing_column(
        [
            "/Users/eliaslundheim/work/PhD/remoteData/macro/simpleShear,s100x100l0.15,1e-05,1.0PBCt3minimizerFIRELBFGSEpsg1e-05CGEpsg1e-05eps1e-05s18.csv"
        ]
    )
    # fix_csv_files_in_folder(folder_path)
