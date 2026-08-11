# MTS2D quick tests for the supervisor

These examples use the `reversibilityProtocolTest` experiment in MTS2D. The
protocol applies one affine strain step, relaxes the mesh, applies the reverse
step, and records whether the state returns to the starting state.

## Install MTS2D

```sh
git clone https://github.com/EliasL/MTS2D.git
cd MTS2D
mkdir -p build-release
cd build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

The MTS2D `ReadMe.md` has the platform-specific prerequisites. On macOS, the
main extra issue is usually OpenMP (`libomp`); on Linux, install the usual C++
compiler, CMake, OpenMP, zlib, and optional CGAL development packages.

## 1. Read one configuration and run a strain range

From the MTS2D directory:

```sh
./build-release/MTS2D \
  -c /path/to/SimulationScripts/examples/supervisor_reversibility/reversibility_no_reconnection.conf \
  -o /path/to/MTS2D_output/supervisor_quick_no_reconnection
```

Change `startLoad`, `loadIncrement`, `maxLoad`, `rows`, `cols`, or `seed` in the
configuration to change the test. The edge-flip version is the same command
with `reversibility_edge_flip.conf`.

## 2. Load one dump

To load a dump and explicitly replace its settings with a new configuration:

```sh
./build-release/MTS2D \
  -d /path/to/existing_job/dumps/dump_l0.20.xml.gz \
  -c /path/to/SimulationScripts/examples/supervisor_reversibility/reversibility_edge_flip.conf \
  -o /path/to/MTS2D_output/supervisor_from_dump_edge_flip
```

When the dump is kept in its original MTS2D job folder, the program can find
that folder's `config.conf` automatically:

```sh
./build-release/MTS2D \
  -d /path/to/existing_job/dumps/dump_l0.20.xml.gz \
  -o /path/to/MTS2D_output/supervisor_from_dump_auto_config
```

The automatic form requires `config.conf` beside the `dumps/` directory. If a
different reconnection method, strain range, or minimizer is wanted, pass
`-c` explicitly. The dump stores the physical state; the supplied config
controls the settings used while continuing the run.

## 3. Prepared dump dataset

The supplied dataset contains the completed non-reconnecting `simpleShear`
size-scaling dump files. It has 80 regime folders in total: 4 sizes × 10
seeds × 2 regimes. The split is strict: a dump belongs to `pre_yield` when its
load is at or below the stress-maximum load, and to `post_yield` otherwise.
The current raw files use the legacy `avg_sigmaxy` column as that proxy. Each
regime has a matching `config.conf`; the number of dump files depends on the
completed simulation. If a job's available dump range does not cross its
yield load, the corresponding regime directory is kept with its config but
contains no dump files.

The folder is arranged as:

```text
supervisor_dumps_no_reconnection/
├── L50x50/seed000/pre_yield/config.conf
├── L50x50/seed000/pre_yield/dumps/dump_l*.xml.gz
├── L50x50/seed000/post_yield/config.conf
├── L50x50/seed000/post_yield/dumps/dump_l*.xml.gz
├── ...
├── L200x200/seed009/post_yield/dumps/dump_l*.xml.gz
├── manifest.csv
└── manifest.json
```

Because `config.conf` is copied beside each regime's `dumps/` folder, this
works without an explicit `-c`:

```sh
./build-release/MTS2D \
  -d /Volumes/data/supervisor_dumps_no_reconnection/L50x50/seed000/pre_yield/dumps/dump_l0.20.xml.gz \
  -o /Volumes/data/MTS2D_output/supervisor_from_pre_yield_dump
```
