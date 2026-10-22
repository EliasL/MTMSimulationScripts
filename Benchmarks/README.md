# Running from SimulationScripts

This suite lives in SimulationScripts. Run the commands below from its root.
The default simulator source is the sibling `../MTS2D` checkout; both runners
accept `--mts2d-source /path/to/MTS2D` for another checkout. The controlled suite
configures `Benchmarks/CMakeLists.txt`, which reuses the simulator objects and
libraries without maintaining a second list of simulation sources. Builds go
to `SimulationScripts/build-benchmark`, and results to `output/benchmarks`.
The reconnecting runner writes configurations, profiles and simulation output
to `output/benchmark_reconnect`; `--build-release` builds the simulator under
`SimulationScripts/build-release`. Source commit and dirty status in suite
manifests refer to the selected MTS2D checkout.

```bash
.venv/bin/python Benchmarks/run_benchmarks.py --preset smoke
.venv/bin/python Benchmarks/run_benchmarks.py --preset full --dry-run
.venv/bin/python Benchmarks/benchmark_reconnect.py --build-release
```

Historical outputs moved from MTS2D are kept in `output/animations`,
`output/pdf`, and `output/benchmarks/legacy`. Historical Slurm script names
mentioned below are examples from the original campaign; those scripts were
not present in the migrated checkout.

# MTS2D benchmark suite

The benchmark suite measures controlled force evaluations and output-free full
minimizations. It is portable across macOS development machines and Linux
compute nodes and has no required Python packages beyond the standard library.

The suite has two layers:

- `benchmark_MTS2D` is the native timed executable. It initializes one mesh,
  performs untimed warm-up calls for force modes, or loads one native dump for
  a minimization replay, times with `steady_clock`, verifies the requested
  OpenMP team size, consumes the numerical result, and emits one JSON record.
- `Benchmarks/run_benchmarks.py` builds the native target, generates system/strong/
  weak-scaling and dump-replay cases, prepares missing load-0.15 states,
  calibrates safe call counts, runs one child process at a time, randomizes
  repetitions, enforces budgets, and writes statistics, plots, and
  hardware-specific configuration candidates.

## Benchmark modes

- `force-kernel` marks an unchanged mesh dirty and recomputes element forces,
  energy, per-thread scratch, and nodal force accumulation. This is the most
  controlled kernel measurement.
- `force-evaluation` additionally copies a deterministic displacement field
  into the nodes and copies nodal forces into an optimizer-style gradient.
  This is the preferred approximation to one minimizer function evaluation.

Both modes repeatedly use one deterministic mesh state after warm-up. They do
not include mesh construction, random-state generation, logging, VTU output,
relaxation, or reconnection in the timed region.

The `minimization` mode measures an entire next-step relaxation, including
solver overhead and optional edge-flip reconnection. Loading/decompression and
the affine next-step setup are outside the timed region. Each timed sample runs
one independent minimization in a fresh process.

Curvature preconditioning is enabled by default (`exactTangent`). Pass
`--preconditioner diagonal` or `--preconditioner none` to either
`Benchmarks/run_benchmarks.py` or the native benchmark to compare native diagonal
preconditioning or ordinary minimization. Use the same initial-condition dump
for paired replay comparisons. Preconditioner construction and application are
included in minimization timing. The diagonal mode assembles only the diagonal;
it does not build or factorize a sparse matrix. Native JSON records include the
selected mode, build/reuse/fallback counts, and total build time in milliseconds.
Force-only modes do not use preconditioning.

## History-dependent minimization workloads

Workload cases come from `Benchmarks/initialConditions/`. The hierarchy is:

```text
initialConditions/<noReconnecting|edgeFlipping>/load_<load>/
  size_<rows>x<cols>/seed_<seed>.xml.gz
```

A dump must contain a relaxed post-step simulation state. For every sample the
native benchmark:

1. loads and validates the dump's size, seed, nominal-load tolerance, and
   reconnection method;
2. initializes the deserialized solvers without opening normal output files;
3. applies one affine step using the dump's serialized `loadIncrement`; and
4. times the resulting minimization.

This preserves the accumulated mesh and edge-flip history. In particular, the
suite never creates a load-0.7 case by setting or shearing a fresh mesh directly
to 0.7. Missing load-0.7 dumps are reported as skipped cases.
Production dump names are rounded to two decimals, so `load_0.7` is a nominal
bin: the stored load must be within 0.005 by default and is reported exactly.

Load 0.15 is special. If a requested dump is missing, the runner can generate
it with the normal first-step recipe: start at load 0.15, add deterministic
Gaussian initial-guess noise (default standard deviation 0.05), minimize with
the production reconnection choice, and save the relaxed state. Generation
uses one thread, is limited by `--initial-condition-budget-minutes` (20 by
default), and never replaces an existing dump. Its timing and minimizer counts
are stored in `initial_condition_preparation.json`; they are not mixed into the
subsequent-step replay statistics. Use `--no-generate-load-015` to require a
fully pre-populated catalog.

Both catalog branches are benchmarked independently:

- `noReconnecting` accepts dumps serialized with reconnection method `none`;
- `edgeFlipping` accepts dumps serialized with `edgeFlip` and runs the next
  minimization with edge flipping enabled.

Delaunay states are never accepted. An `edgeFlipping` replay remains valid even
when its particular next step does not flip an edge: the branch describes the
history and allowed production algorithm, not a promise that every step flips.

By default, workload repetitions reuse one deterministic fixture, matched across
every thread count and binding policy. Their standard deviation therefore
measures timing variation without mixing in a change of physical problem.
`--fixture-seed-stride 1` enables consecutive matched fixtures for a separate
robustness study, but those seeds should first be screened for finite relaxed
states. Controlled force repetitions also reuse the same data.

For sparse imported catalogs, `--allow-available-fixture-fallback` keeps that
fixed-fixture behavior while allowing each workload to use the lowest valid
catalog seed when the requested seed is absent. The chosen serialized seed is
validated normally and recorded in every sample; dumps are never relabeled.
This option requires the default fixed seed stride of zero.

This is a deliberately sparse interaction design, not a formal Taguchi array.
It keeps size × threads and load/history × threads interactions while letting
the dump catalog select a small number of scientifically representative states.
An optional affinity study adds size × threads × affinity interactions. Once
real load-0.7 dumps are available, multiple seeded dump sets can represent
ordinary short steps and known long events without sweeping every simulation
parameter combination.

## Local use

Run the tiny validation suite while developing:

```sh
python3 Benchmarks/run_benchmarks.py --preset smoke
```

Run the default suite, designed to finish within one hour:

```sh
python3 Benchmarks/run_benchmarks.py --preset quick
```

Run the topology-aware `shortBenchmark` suite. It normally finishes well below
its deliberately generous 54-minute safety ceiling.
It keeps the full force-scaling thread grid but screens expensive minimization
workloads at a sparse power-of-two grid containing 1, 2, 4, 8, 16, and 32
threads, plus the full allocation. The general thread grid continues with 64,
128, and so on where available; it does not add a nearly redundant
machine-specific socket count such as 60:

```sh
python3 Benchmarks/run_benchmarks.py --preset shortBenchmark
```

On Cauchy, submit `Benchmarks/slurm_cauchy_short_benchmark.sh`; its 70-minute
Slurm limit is a safety ceiling that includes clean-build and report-generation
headroom.

Run the higher-accuracy suite, with a twelve-hour hard total budget and a
20-minute budget for each benchmark case:

```sh
python3 Benchmarks/run_benchmarks.py --preset full
```

On Cauchy, `Benchmarks/slurm_cauchy_full_benchmark.sh` reserves twelve hours and
uses an 11.75-hour internal benchmark ceiling, leaving clean-build and final
report-generation headroom before Slurm ends the job.

The full preset includes controlled force sizes through `1024`; these calls are
cheap enough that the large force-only points remain useful. Its minimization sizes are
50, 100, 150, 200, and 250 to match the scientific size-scaling simulations,
plus 500 for a reduced large-system screen. The 500 workload uses threads
1, 8, 32, and 64 plus the full allocation, with at most three replay
repetitions per case. It calibrates force-call counts independently and never
exceeds 10,000 calls per sample. Load-0.15 minimization states may take several
minutes to prepare at the largest sizes. Slow or
memory-heavy cases are timed out or skipped rather than breaking the global
budget. Load-0.7 cases at 500 are simply absent until matching dumps are
added.

Every workload suite also measures the first minimization after random noise
over its minimization thread grid. At L=500 this uses the reduced thread grid.
If the one-thread run generated a missing catalog dump, that timing is reused
rather than running the same expensive minimization twice.

Useful customizations include:

```sh
# Exact thread sweep and selected mesh sizes
python3 Benchmarks/run_benchmarks.py --preset custom \
  --threads 1-16 --sizes 64,128,256,512 \
  --strong-size 512 --repetitions 8

# Request 10,000 calls when the time budget permits
python3 Benchmarks/run_benchmarks.py --preset custom --calls 10000

# Inspect the generated matrix without building or running it
python3 Benchmarks/run_benchmarks.py --preset full --dry-run

# Workload-only replay with matched seeds and selected sizes
python3 Benchmarks/run_benchmarks.py --preset custom --experiments workload \
  --workload-sizes 32,64 --workload-loads 0.15,0.7 \
  --threads 1,2,4,8 --repetitions 8

# Keep a dense force-scaling sweep but screen costly minimizations sparsely
python3 Benchmarks/run_benchmarks.py --preset custom \
  --threads auto --workload-threads screen

# Use a separately synchronized dump catalog
python3 Benchmarks/run_benchmarks.py --preset custom --experiments workload \
  --initial-conditions-dir /scratch/MTS2D_initialConditions
```

By default the benchmark gets a clean optimized build in `build-benchmark`
with `ENABLE_NOINLINE=OFF`. Use `--keep-noinline` only when matching a
profiler-oriented build is intentional. Use `--no-build --exe PATH` to test a
specific existing executable.

## Linux compute nodes and affinity

Resource allocation and thread placement solve different parts of the problem:

1. The batch scheduler grants one task an exclusive set of CPUs.
2. The runner places OpenMP threads inside that granted CPU set.

Force-scaling cases use the selected policy. The default is `--affinity close`,
so ordinary benchmark presets do not double their minimization workload with a
core-placement comparison. The portable constraints are:

```text
OMP_PLACES=cores
OMP_PROC_BIND=close
OMP_DYNAMIC=FALSE
```

On Linux, the runner builds an explicit OpenMP place list containing one
hardware thread per physical core and orders it by NUMA node, package, die, and
core. `close` therefore fills a local NUMA domain first even on machines whose
CPU numbers alternate sockets; `spread` samples across the full place list.
This is not universally optimal: memory-bandwidth-limited or multi-socket cases
may prefer `--affinity spread`. Use `--affinity auto` for a dedicated comparison
of `close` and `spread`. The suite also supports `--affinity none` for an
explicit unbound comparison.

For Slurm, request a whole node (or an exclusive CPU allocation), start one
task, and give that task all physical cores that the suite may use. For a
128-core node, a typical job script is:

```sh
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --time=06:15:00

srun --cpu-bind=cores --mem-bind=local \
  python3 Benchmarks/run_benchmarks.py --preset full --threads auto
```

Cluster policies differ, so confirm the site's meaning of `--exclusive`, CPU
counts, simultaneous multithreading, and memory binding. Do not make the runner
start multiple benchmark processes concurrently.

On Linux, each native result records the logical CPU ID and OpenMP place of
every thread. The run manifest also records the allowed CPU set, package, die,
core, and NUMA-node topology read from sysfs. These records make it possible to
verify that `close` actually produced the intended placement. On runtimes that
do not expose OpenMP places, the report warns that affinity is unverified.

The runner uses physical cores by default. Pass an explicit `--threads` list if
simultaneous multithreading is itself part of the experiment.

## Comparative minimization plots

After a run, use the uv-managed plotting environment to create cross-workload
thread, linear-size, speedup, minimizer-path, and recommended-thread plots as
QA PNGs and publication-quality vector PDFs. Plots use the linear size $L$;
the redundant $N=L^2$ version is not generated. Early and late replay events
are shown without reconnection qualifiers. First minimization keeps separate
default and edge-flipping series. Consolidated thread plots use color for $L$
and hollow marker shape for event type; 64 replaces the nearby 60-thread point:

```sh
uv venv .venv
uv pip sync --python .venv/bin/python Benchmarks/plot-requirements.lock
.venv/bin/python Benchmarks/plot_benchmark_minimizations.py RESULT_DIRECTORY \
  --first-minimization PREPARATION_RECORD.json
```

Log-log plots against $L$ include unweighted least-squares fits of
$y=C L^\alpha$. The fitted exponent is printed in each series label and the
smooth dashed line shows the fit. Thus doubling $L$ multiplies the fitted cost
by $2^\alpha$. These fits summarize the measured size range; they are not
extrapolation guarantees.

The plotting command automatically reads first-minimization records from the
result directory. Pass `--first-minimization` for additional preparation files
from older or split runs. Replay timing error bars are sample standard
deviations. Each first-minimization thread point is currently one independent
run, so it has no standard-deviation estimate.

## Time and memory budgets

`quick` defaults to a one-hour global budget. `full` defaults to twelve hours.
The total budget includes load-0.15 initial-condition preparation. Every
ordinary case has a 20-minute total budget covering calibration, process
startup, warm-up, and all repetitions; each generated initial condition also
has a separate 20-minute ceiling. First minimizations and missing initial
conditions at L=500 instead have a 40-minute ceiling. These are hard subprocess
timeouts, not estimates.

Before running a case, the runner estimates the current dense force scratch as

```text
threads × nodes × sizeof(Vector2d)
```

and skips it if that estimate alone exceeds half of currently available
memory. Change the guard with `--max-memory-fraction` when an allocation makes
a different limit appropriate.

## Results

Results are written below `output/benchmarks/<preset>_<timestamp>/`:

- `samples.jsonl` is appended after every successful independent process, so
  partial results survive interruption.
- `samples.csv` contains all raw observations and thread-placement records.
- `summary.csv` and `summary.json` contain the arithmetic mean, sample standard
  deviation, median, range, coefficient of variation, standard error,
  throughput, strong/weak efficiency, and checksum validation.
- `manifest.json` records the executable hash, git state, effective build
  flags, system and scheduler metadata, topology, budgets, skips, and failures.
- `initial_condition_plan.json` records every requested catalog path and whether
  it existed or was unavailable after preparation.
- `initial_condition_preparation.json` records first-noisy generation timings,
  minimizer work, paths, sizes, and content hashes. It is empty when every
  requested load-0.15 dump already exists.
- `first_minimization_benchmarks.json` records the first-noisy thread sweep.
- `report.md` provides a compact human-readable table.
- `plots/` shows mean time per force evaluation or full minimization with
  standard-deviation error bars. Matplotlib produces PNGs when available;
  otherwise the runner writes dependency-free SVG plots.
- `recommendations.json` records conservative candidates separately for every
  available load, history branch, and tested size.
- `recommended_configs/` contains one OpenMP environment file and one
  `nrThreads` config fragment per load/history profile, selected from the
  largest measured size. Config fragments must be merged into an existing
  simulation config; they are not standalone configurations.

Recommendations prefer fewer threads when their mean is within 10% of the
fastest and the difference is no larger than 1.96 combined standard errors.
The JSON records edge-flip counts, means, standard deviations, fixture seeds,
selection rule, and confidence. A zero-flip next step does not invalidate an
`edgeFlipping` history replay. Affinity confidence is capped at
`exploratory` when the runtime cannot expose enough placement information to
verify binding.

The existing `Benchmarks/benchmark_reconnect.py` remains the end-to-end benchmark
for relaxation/reconnection, output, wall time, memory, and profiling. It is
complementary to this controlled benchmark suite.
