# Management Overview

This folder contains the API layer for defining simulations, running them, and
finding the resulting data locally or on cluster servers.

## Core Concepts

`SimulationConfig` in `configGenerator.py` is the central object. It stores all
simulation parameters and generates the canonical simulation name with
`generate_name(withExtension=False)`. That name is also the output folder name
under `MTS2D_output`, so most data lookup is name based.

`ConfigGenerator.generate(...)` builds lists of `SimulationConfig` objects from
single values or lists. Lists are expanded as a Cartesian product, and labels are
generated from varying parameters. `group_by_settings(...)` can regroup configs
after generation, usually to compare seeds within the same parameter setting.

`jobs.py` is a library of named config factories. Functions like
`sylvainBatches(...)`, `reversibilityJob(...)`, `umutJob(...)`, and others return
`(configs, labels)`. Plotting scripts usually import these job factories rather
than writing raw config dictionaries.

## Running Simulations

`simulationManager.py` wraps local execution. `SimulationManager(config)` writes
the config file into the MTS2D build folder, builds if needed, and runs:

```text
MTS2D -c <config-file> -o <output-root>
```

`findOutputPath()` chooses the first available output root from known local and
cluster paths, then appends `MTS2D_output/`.

Cluster helpers are split across:

- `connectToCluster.py`: SSH setup, server lists, folder download helpers.
- `runOnCluster.py`: remote command/script execution and queue submission.
- `multiServerJob.py`, `jobManager.py`: higher-level queue/status tools.

`Servers` in `connectToCluster.py` defines the important server groups:

- `Servers.local_path_mac`: local mounted data path, usually `/Volumes/data/`.
- `Servers.search_servers`: servers used when looking for existing data.
- `Servers.run_servers`: servers used for running new jobs.
- `Servers.serversAndLocal`: search servers plus local path.

## Finding Existing Data

Most plotting code uses helpers in `Plotting/remotePlotting.py`.

Use `get_csv_files(configs, labels=...)` when you only need `macroData.csv`.
It checks cached CSVs, local data under `Servers.local_path_mac/MTS2D_output`,
then remote servers. Remote CSVs are downloaded into the `remoteData/macro`
cache, using `/Volumes/data/remoteData` when the external data drive is mounted
and `~/Work/PhD/remoteData` otherwise.

Use `get_folders_from_servers(configs)` when you need full simulation folders,
for example VTU mesh files. It checks local folders and can download remote
folders into `remoteData/data` in the same external-drive cache when available.
This is heavier than CSV lookup because it uses `rsync` on full folders.

Local-only folder lookup can be done with:

```python
from Management.connectToCluster import Servers
from Plotting.remotePlotting import handleLocalPath

folders = handleLocalPath(Servers.local_path_mac, configs, returnCsv=False)
```

Folder and CSV matching are based on `config.name`, which should match the
simulation folder name.

## Reading Mesh Data

Mesh loading helpers live in `Plotting/dataFunctions.py`.

`resolve_vtu_files(source)` accepts a simulation folder, `.pvd`, `.vtu`, or CSV
path and returns ordered VTU files. The usual convention for the final mesh is:

```python
from Plotting.dataFunctions import VTUData, resolve_vtu_files

vtu_files = resolve_vtu_files(simulation_folder)
final_data = VTUData(vtu_files[-1])
nodes = final_data.get_nodes()
connectivity = final_data.get_connectivity()
```

`VTUData` also exposes common fields such as `get_reference_nodes()`, `get_F()`,
`get_C()`, `get_P()`, `get_energy_field()`, and `get_stress_field()`.

## Typical Plotting Flow

A common analysis script follows this pattern:

```python
from Management.jobs import sylvainBatches
from Plotting.remotePlotting import get_csv_files, get_folders_from_servers
from Plotting.dataFunctions import VTUData, resolve_vtu_files

configs, labels = sylvainBatches(-1)

# For macro plots:
csv_paths, labels = get_csv_files(configs, labels=labels)

# For mesh plots:
folders = get_folders_from_servers(configs)
for folder in folders:
    final_vtu = resolve_vtu_files(folder)[-1]
    data = VTUData(final_vtu)
```

## Gotchas

- `get_csv_files(...)` only gives CSVs. It is not enough if you need VTU files.
- `get_folders_from_servers(...)` may download large folders.
- `plot.py` imports `from Management.jobs import *`; there is no top-level
  `jobs.py`.
- Most path lookup assumes outputs are stored in a folder named exactly like
  `config.generate_name(False)`.
- `VTUData` expects real MTS2D VTU files with the usual cell and point fields.
