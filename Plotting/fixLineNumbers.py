import os
import pandas as pd
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor


def fix_csv_files(paths, use_tqdm=True):
    if isinstance(paths[0], list):
        paths = [item for sublist in paths for item in sublist]
    # Iterate over all files in the given folder
    for path in tqdm(paths, disable=not use_tqdm):
        if path.endswith(".csv"):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(path, low_memory=False)

            # Create a Series that tracks the maximum value encountered so far
            cummax_series = df["Load"].cummax()

            # Create a boolean mask where the current value is less than the maximum encountered
            overlap_mask = df["Load"] < cummax_series

            # Drop the rows where overlap_mask is True
            df_cleaned = df[~overlap_mask].reset_index(drop=True)

            # Save the modified DataFrame back to the same CSV file
            df_cleaned.to_csv(path, index=False)


def fix_missing_column(paths, use_tqdm=True):
    # This function processes each file in the given paths
    # It removes the first column if the first header is "Line nr"
    # and continues removing rows if the values in the first column
    # are floats equal to 1, until a non-1, non-float is found.

    for path in tqdm(paths, disable=not use_tqdm):
        # Read the file into a dataframe (assuming CSV format)
        df = pd.read_csv(path)

        # Check if the first column header is "Line nr"
        if df.columns[0] == "Line nr":
            # Iterate over the rows in the first column
            for i, val in enumerate(df.iloc[:, 0]):
                try:
                    # Try to convert the value to a float
                    float_val = float(val)

                    # If it's a float and equals 1, continue checking
                    if float_val == 1.0:
                        break
                    else:
                        # Stop once we find a value that isn't 1
                        break
                except ValueError:
                    # If we hit a value that can't be converted to float, stop
                    break

            # Remove the first column up to the valid row
            df = df.iloc[i:, 1:]  # Skip rows and drop the first column

            # Optionally, save the cleaned dataframe back to a file
            df.to_csv(path + "test.txt", index=False)


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
