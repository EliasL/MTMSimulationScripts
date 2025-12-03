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


def existing_directories(directories):
    """Return a list of directories from the input list that exist on disk."""
    return [directory for directory in directories if os.path.exists(directory)]


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
    """Find all existing preferred directories and estimate the size of their contents.

    For each base directory that exists, this will:
      * Look for the `MTS2D_output` subdirectory.
      * List all simulation folders and approximate their sizes (one line per folder as
        `<full_path>\t<size>`).
      * Track the free space (in GB) for each filesystem.

    At the very end, it prints a single line with the smallest free space value among
    all processed directories, so that the parser can treat the last line as the
    global minimum free space.
    """
    preferred_directories = [
        "/data2/elundheim",
        "/data/elundheim",
        "/Volumes/data/",
        "/Users/elias/Work/PhD/Code/localData",
        "/lustre/fswork/projects/rech/bph/uog82gz/",
    ]

    existing_dirs = existing_directories(preferred_directories)

    if not existing_dirs:
        # Keep a similar behaviour as before when nothing is found
        print("0")
        print("Warning: None of the preferred base directories exist.")
        return

    min_free_space_gb = None

    for base_dir in existing_dirs:
        out_dir = os.path.join(base_dir, "MTS2D_output")
        if not os.path.exists(out_dir):
            if find_first_folder(base_dir) is not None:
                print(
                    f"Warning: The folder {out_dir} does not exsist! Found: {find_first_folder(base_dir)}"
                )
            continue

        simulation_folders = find_folders(out_dir)
        print(len(simulation_folders))
        for folder in simulation_folders:
            full_path = os.path.join(out_dir, folder)
            size = approximate_size(full_path)
            print(f"{full_path}\t{size}")

        # Filesystem statistics for this output directory
        stats = os.statvfs(out_dir)
        free_space_bytes = stats.f_bavail * stats.f_frsize
        free_space_gb = free_space_bytes / (1024**3)

        # Track the smallest free-space value seen so far
        if min_free_space_gb is None or free_space_gb < min_free_space_gb:
            min_free_space_gb = free_space_gb

    # At the very end, print the minimum free-space value so the parser can
    # treat the last line as the free-space line
    if min_free_space_gb is not None:
        print(min_free_space_gb)
    else:
        # No valid `MTS2D_output` directories were found; print a sentinel value
        print(-1.0)


if __name__ == "__main__":
    folder = find_data()
