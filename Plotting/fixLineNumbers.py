import os
import pandas as pd
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor


def fix_csv_files(paths, use_tqdm=True):
    # Disable: Don't do it automatically.
    return
    if len(paths) == 0:
        return
    # fix_missing_column(paths, use_tqdm=use_tqdm)

    if isinstance(paths[0], list):
        paths = [item for sublist in paths for item in sublist]
    # Iterate over all files in the given folder
    for path in tqdm(paths, disable=not use_tqdm):
        if path.endswith(".csv"):
            if path.endswith("temp.csv"):
                # I don't understand why these files are being created...
                os.remove(path)
                return
            # Read the CSV file into a DataFrame
            df = pd.read_csv(path)

            # Replace spaces in column names with underscores
            df.columns = df.columns.str.replace(" ", "_", regex=False)

            # Create a Series that tracks the maximum value encountered so far
            cummax_series = df["Load"].cummax()

            # Create a boolean mask where the current value is less than the maximum encountered
            overlap_mask = df["Load"] < cummax_series

            # Drop the rows where overlap_mask is True
            df_cleaned = df[~overlap_mask].reset_index(drop=True)

            # Save the modified DataFrame back to the same CSV file
            df_cleaned.to_csv(path, index=False)


def process_file(path):
    # Ensure only CSV files are processed
    if not path.endswith(".csv"):
        return

    # Temporary file path
    temp_path = f"{path[:-4]}_temp.csv"

    # Open the original file for reading and the temporary file for writing
    with open(path, "r") as oFile:
        header = oFile.readline()
        # We don't need to modify this file
        if header.startswith("Load_step"):
            return
        with open(temp_path, "w") as temp_file:
            # Read the first line (header)
            if header.startswith("Line nr"):
                # If the header starts with "Line nr", write the rest of the header (ignore "Line nr" column)
                _, rest = header.split(sep=",", maxsplit=1)
                # I also decided half way to use "_" instead of " "
                rest = rest.replace(" ", "_")
                temp_file.write(rest)
            else:
                # Otherwise, write the header as-is
                temp_file.write(header)

            # Process the rest of the file
            for line in oFile:
                # Split the line into the first element and the rest
                n, rest = line.split(sep=",", maxsplit=1)
                # If the first element is not a float, it's a line number; remove it
                if "." in n:
                    temp_file.write(line)  # If it's a float, keep the full line
                else:
                    # Otherwise, remove the line number and write the rest
                    temp_file.write(rest)

    # Replace the original file with the modified one
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
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.endswith(".csv")
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
