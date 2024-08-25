import os
import pandas as pd
from tqdm import tqdm


def fix_csv_files_in_folder(folder_path):
    # Iterate over all files in the given folder
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path, header=None)

            # Replace the first column with accurate line numbers (starting from 1)
            df[0] = ["Line nr"] + list(range(1, len(df)))

            # Save the modified DataFrame back to the same CSV file
            df.to_csv(file_path, index=False, header=False)


# Example usage
folder_path = "/Users/elias/Work/PhD/Code/remoteData/"
fix_csv_files_in_folder(folder_path)
