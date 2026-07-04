import os
import re
from pathlib import Path


def create_collection(
    folder_path,
    destination: (str | Path) = "..",
    collection_name="collection",
    extension=".vtu",
    skipBadStop=True,
    splitKey="reference",
):
    split_label = splitKey.strip("_")
    if not split_label:
        raise ValueError("splitKey must contain at least one non-'_' character.")
    split_token = f"_{split_label}"

    files_with_numbers = []
    split_files_with_numbers = []

    # Regular expression to match file numbers in the filename
    regex_pattern = r".*\.([0-9]+)\.vtu"

    min_regex_pattern = r"^.*_minStep=[0-9]+\.([0-9]+)(?:_[^.]+)?\.[0-9]+\.vtu$"

    # Iterate over files in the directory
    for entry in Path(folder_path).iterdir():
        if skipBadStop and "badStop" in str(entry):
            continue
        if entry.suffix == extension:
            filename = entry.name
            if "minStep" in filename:
                match = re.match(min_regex_pattern, filename)
            else:
                match = re.match(regex_pattern, filename)
            if match and len(match.groups()) == 1:
                number = int(match.group(1))
                if split_token in filename:
                    split_files_with_numbers.append((number, entry))
                else:
                    files_with_numbers.append((number, entry))

    # Sort files based on the extracted number
    files_with_numbers.sort()
    split_files_with_numbers.sort()

    if destination == "..":
        destination = Path(folder_path).parent
    destination = Path(destination).absolute()

    def write_collection_file(name, entries):
        with open(os.path.join(destination, f"{name}.pvd"), "w") as out_file:
            out_file.write('<?xml version="1.0"?>\n')
            out_file.write('<VTKFile type="Collection" version="0.1">\n')
            out_file.write("<Collection>\n")

            for i, (_, file) in enumerate(entries):
                relative_path = file.relative_to(destination)
                out_file.write(
                    f'<DataSet timestep="{i}" group="" part="0" file="{relative_path}"/>\n'
                )

            out_file.write("</Collection>\n")
            out_file.write("</VTKFile>\n")

    write_collection_file(collection_name, files_with_numbers)
    write_collection_file(f"{collection_name}_{split_label}", split_files_with_numbers)


if __name__ == "__main__":
    create_collection(os.path.expanduser("~/Downloads/data"))
