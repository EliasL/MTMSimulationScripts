import os
import pandas as pd
from tqdm import tqdm


def fix_csv_files_in_folder(folder_path):
    # Iterate over all files in the given folder
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Create a Series that tracks the maximum value encountered so far
            cummax_series = df["Load"].cummax()

            # Create a boolean mask where the current value is less than the maximum encountered
            overlap_mask = df["Load"] < cummax_series

            # Drop the rows where overlap_mask is True
            df_cleaned = df[~overlap_mask].reset_index(drop=True)

            # Save the modified DataFrame back to the same CSV file
            df_cleaned.to_csv(file_path, index=False)


# Example usage
folder_path = "/Users/elias/Work/PhD/Code/remoteData/"
fix_csv_files_in_folder(folder_path)
