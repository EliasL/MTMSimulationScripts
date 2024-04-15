import xml.etree.ElementTree as ET

def parse_pvd_file(path, pvd_file):
    tree = ET.parse(path+pvd_file)
    root = tree.getroot()
    vtu_files = []

    for dataset in root.iter('DataSet'):
        vtu_files.append(dataset.attrib['file'])

    return vtu_files

def get_data_from_name(nameOrPath):
    # Split the filename by underscores
    fileName = nameOrPath.split('/')[-1]
    parts = fileName.split('_')

    # Initialize an empty dictionary
    result = {}

    # We skipp the first and last part. The first part is the 'name', the last
    # part is the type, ie .N.vtu
    result['name'] = parts[0]
    for part in parts[1:-1]:
        key, value = part.split('=')
        # Add the key-value pair to the dictionary
        result[key] = value

    return result

