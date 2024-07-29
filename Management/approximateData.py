import os


"""
Instead of calculating the full size of a folder, we can make a very good
approximation since all the files in the subfolders (data, dumps, frames) will
have a very similar size. Therefore, we can multiply the size of the first file
with the number of files in the folder.
"""


def find_first_folder(base_dir):
    """Find the first folder in the specified directory."""
    try:
        with os.scandir(base_dir) as entries:
            folder = next((entry for entry in entries if entry.is_dir()), None)
            if folder:
                return folder.name
            else:
                return None
    except StopIteration:
        return None


def first_existing_directory(directories):
    """Find and return the first existing directory from a list."""
    for directory in directories:
        if os.path.exists(directory):
            return directory
    return None


def approximate_size(path):
    """Estimate the total size of files in a directory by approximating subdirectory sizes."""
    total_size = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                total_size += entry.stat().st_size
            elif entry.is_dir():
                subfolder_size, num_files = approximate_subfolder(entry.path)
                if num_files > 0:
                    total_size += subfolder_size * num_files
    return total_size


def approximate_subfolder(path):
    """Calculate the size of the first file in a subfolder and count the number of files."""
    file_count = 0
    first_file_size = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                file_count += 1
                if first_file_size == 0:
                    first_file_size = entry.stat().st_size
    return (first_file_size, file_count)


def find_folders(directory):
    """Return a list of folders in the specified directory using os.scandir() with a list comprehension."""
    with os.scandir(directory) as entries:
        folders = [entry for entry in entries if entry.is_dir()]
    return folders


def find_data():
    """Find a specific directory and estimate the size of its contents."""
    preferred_directories = [
        "/data2/elundheim",
        "/data/elundheim",
        "/Volumes/data/",
        "/Users/elias/Work/PhD/Code/localData",
    ]
    # We print 0 to tell the data manager script that there are 0 folders
    base_dir = first_existing_directory(preferred_directories)
    if base_dir is None:
        print("0")
        # print("No valid data directory found.")
        return None

    folder_name = find_first_folder(base_dir)

    if folder_name is None:
        print("0")
        # print("No valid data directory found.")
        return None

    if folder_name != "MTS2D_output":
        print("0")
        print(
            f"Warning: The folder in {base_dir} is not called 'MTS2D_output'. Found: {folder_name}"
        )

    base_dir = os.path.join(base_dir, folder_name)
    simulation_folders = find_folders(base_dir)
    print(len(simulation_folders))
    for folder in simulation_folders:
        full_path = os.path.join(base_dir, folder)
        size = approximate_size(full_path)
        print(f"{full_path}\t{size}")


if __name__ == "__main__":
    folder = find_data()
