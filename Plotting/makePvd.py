import os
import re
from pathlib import Path

def create_collection(folder_path, destination="..", collection_name="collection", extension=".vtu"):
    files_with_numbers = []

    # Regular expression to match file numbers in the filename
    regex_pattern = r".*\.([0-9]+)\.vtu"

    # Iterate over files in the directory
    for entry in Path(folder_path).iterdir():
        if entry.suffix == extension:
            filename = entry.name
            match = re.match(regex_pattern, filename)
            if match and len(match.groups()) == 1:
                number = int(match.group(1))
                files_with_numbers.append((number, entry))

    # Sort files based on the extracted number
    files_with_numbers.sort()

    # Create and write to the .pvd file
    if destination == "..":
        destination = Path(folder_path).parent
    destination = Path(destination).absolute()
    with open(os.path.join(destination, f"{collection_name}.pvd"), 'w') as out_file:
        out_file.write('<?xml version="1.0"?>\n')
        out_file.write('<VTKFile type="Collection" version="0.1">\n')
        out_file.write('<Collection>\n')

        for i, (num, file) in enumerate(files_with_numbers):
            out_file.write(f'<DataSet timestep="{i}" group="" part="0" file="{folder_path}/{file.name}"/>\n')

        out_file.write('</Collection>\n')
        out_file.write('</VTKFile>\n')

if __name__ == "__main__":
    create_collection('/Users/eliaslundheim/Downloads/data')