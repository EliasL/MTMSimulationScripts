import xml.etree.ElementTree as ET
import os


def parse_pvd_file(path, pvd_file):
    tree = ET.parse(pvd_file)
    root = tree.getroot()
    vtu_files = []

    for dataset in root.iter("DataSet"):
        vtu_files.append(os.path.join(path, dataset.attrib["file"]))

    return vtu_files


def get_data_from_name(nameOrPath):
    # Split the filename by underscores
    fileName = nameOrPath.split("/")[-1]
    parts = fileName.split("_")

    # Initialize an empty dictionary
    result = {}

    # We skipp the first and last part. The first part is the 'name', the last
    # part is the type, ie .N.vtu:
    # resettingSimpleShearPeriodicBoundary,s60x60l0.15,1e-05,10PBCt4s0_load=3.79001_nrM=0_.364001.vtu

    result["name"] = parts[0]
    for part in parts[1:-1]:
        key, value = part.split("=")
        # Add the key-value pair to the dictionary
        result[key] = value

    # We can now extract some extra stuff from the name
    # It will for example have the form:
    # resettingSimpleShearPeriodicBoundary,s60x60l0.15,1e-05,10PBCt4s0
    result["dims"] = tuple(
        map(int, result["name"].split(",")[1].split("s")[1].split("l")[0].split("x"))
    )

    # Extract start load, load increment, and max load
    load_parts = result["name"].split(",")[1:]
    result["startLoad"] = load_parts[0].split("l")[1]
    result["loadIncrement"] = load_parts[1]
    if "NPBC" in load_parts:
        result["maxLoad"] = load_parts[2].split("NPBC")[0]
        result["BC"] = "NPBC"
    else:
        result["maxLoad"] = load_parts[2].split("PBC")[0]
        result["BC"] = "PBC"

    return result
