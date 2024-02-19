import os
from pathlib import Path
def parse_cpp_header(header_path):
    settings = {}
    
    with open(header_path, 'r') as file:
        for line in file:
            if line.startswith('#define'):
                parts = line.split()
                if len(parts) == 3:
                    settings[parts[1]] = parts[2].strip('\"')
    return settings

# Add Management to sys.path (used to import files)
settings = parse_cpp_header(str(Path(__file__).resolve().parent.parent.parent) + '/MTS2D/src/settings.h')