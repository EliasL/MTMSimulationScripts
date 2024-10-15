import os
import pandas as pd
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor


def fix_csv_files_in_folder(folder_path, use_tqdm=True):
    # Iterate over all files in the given folder
    for filename in tqdm(os.listdir(folder_path), disable=not use_tqdm):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path, low_memory=False)

            # Create a Series that tracks the maximum value encountered so far
            cummax_series = df["Load"].cummax()

            # Create a boolean mask where the current value is less than the maximum encountered
            overlap_mask = df["Load"] < cummax_series

            # Drop the rows where overlap_mask is True
            df_cleaned = df[~overlap_mask].reset_index(drop=True)

            # Save the modified DataFrame back to the same CSV file
            df_cleaned.to_csv(file_path, index=False)


def fix_lines_in_data_folder(folder_path, max_workers=10):
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
    fix_csv_files_in_folder(folder_path)
