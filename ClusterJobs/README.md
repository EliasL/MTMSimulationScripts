# Sylvain reversibility postprocessing

These jobs turn the large per-event VTUs into small compressed event tables.
The worker processes one simulation folder at a time and reads one affine/
relaxed VTU pair at a time, so it does not accumulate the mesh data in memory.

The output columns include:

- `delta_E_S`: second-order stress-corrected energy drop divided by mesh volume.
- `delta_sigma_S`: first-order elasticity-corrected stress drop.
- `delta_E_I`, `delta_E_R`, `delta_sigma_I`, and `delta_sigma_R` for comparison.
- `delta_rev_E`, `delta_rev_sigma`, `delta_rev_u`, and `delta_u_R`.
- batch, setting, seed, event type, yield regime, load, and correction metadata.

The existing `calculate_energy_step_data` and `calculate_stress_step_data`
implement the correction conventions used by the plotting code. The worker
raises on missing or inconsistent data rather than silently omitting events.

The cluster’s existing synced virtual environments may point to a local macOS
interpreter, so create a separate Linux environment once on each server:

```bash
bash ~/simulation/SimulationScripts/ClusterJobs/setup_reversibility_postprocessing_env.sh
```

Upload the worker sources with `upload_reversibility_postprocessing.sh`; it
copies only the required helpers and four small `MTMath` dependencies.

After this directory has been uploaded to the cluster, prepare and submit jobs
from a cluster login node with, for example:

```bash
bash ~/simulation/SimulationScripts/ClusterJobs/submit_reversibility_postprocessing.sh \
  /data2/elundheim/MTS2D_output \
  /data2/elundheim/MTS2D_postprocessed/sylvain_reversibility \
  -2
```

For `duchemin`, use `/data/elundheim/MTS2D_output` as the input root. The
submission script discovers only the requested batch folders on that server,
reports missing expected folders, submits a Slurm array, and submits a merge
job after the array succeeds. It does not delete or modify the raw simulation
data.

## Targeted elastic-event replay

`replay_elastic_from_dumps.py` selects the first one or two rows with zero
forward m3 changes after each supplied dump, runs the patched
`reversibilityProtocolTest` in a private output directory, and saves the five
states under `elastic_replay_l_*`.  The C++ option
`saveElasticReversibilityStates` is disabled by default; when enabled, the
replay performs the backward minimization and records the measured closure
distance.  `maximumSavedElasticReversibilityStates` is a hard in-process cap;
enabling elastic saves without a positive cap raises an error.  The driver
validates both the source and replay macro rows,
including zero forward m3 change and `is_reversible=1` when that column is
available.

For example, on a cluster login node:

```bash
PYTHONPATH=~/simulation/SimulationScripts \
  python3 ~/simulation/SimulationScripts/ClusterJobs/replay_elastic_from_dumps.py \
  --source-job /data2/elundheim/MTS2D_output/JOB \
  --output-root /data2/elundheim/MTS2D_elastic_replays \
  --mts2d-binary ~/simulation/MTS2D/build-release/MTS2D \
  --maximum-events-per-dump 2
```

The raw source job and its dumps are read only.  The output manifest records
the dump, target load, event directory, and verified forward m3 count.

`replay_selected_real_space_event.py` can instead target one known plastic
event with `--expected-event-kind plastic`.  Its private configuration enables
`saveFinalReversibilityState`, disables dumps and ordinary mesh VTUs, and
suppresses the normal sparse reversibility-snapshot cadence.  Thus a run with
`--maximum-elastic-events 2` can write at most fifteen VTUs: five for the final
plastic target and five for each of two elastic examples.

## Post-yield xmin collection scaffold

`collect_post_yield_xmin_events.py` separates the near-xmin collection into
seven reviewable stages and includes a `run` command that executes them in
order.  It waits for new source checkpoints when necessary and never permits
more than two concurrent two-thread replays.

Run the stages from the repository root with `PYTHONPATH=.`:

1. `select` fits the post-yield irreversible xmin and freezes ten targets plus
   five backups on each side using only energy ratio and seed.
2. `inventory` lists existing Pascal dumps into
   `checkpoint_inventory.csv` without downloading them.
3. `plan` attaches the nearest preceding dump and groups ready events into
   waves of two two-thread replays.
4. `fetch` downloads only the planned configs, macro files, and dumps.
5. `replay` executes one target per private output directory.
6. `validate` rejects any target that does not reproduce the source event
   or lacks exactly five states.
7. `render` produces the marked PDF followed by the 3x2 event sheets.

The `run` command uses the 100 GB free-space guard by default.  It terminates
active replay children immediately if the data volume falls below that limit.

To run the complete campaign from the repository root:

```bash
MPLCONFIGDIR=/tmp/mpl-cache PYTHONPATH=. ./.venv/bin/python -u \
  ClusterJobs/collect_post_yield_xmin_events.py run \
  --mts2d-binary /Users/eliaslundheim/work/PhD/MTS2D/build-release/MTS2D \
  --poll-seconds 1800
```

If the terminal is interrupted after the campaign has created its manifest,
rerun the same command with `--resume`.  The default output is on
`/Volumes/data/MTS2D_xmin_collection/post_yield_irreversible`.

The primary matched comparison uses `0.5 <= Delta E_S/(V0*xmin) < 1` below
and `1 <= Delta E_S/(V0*xmin) <= 2` above.  Pass
`--above-max-ratio 10` to preselect a separate one-decade context cohort.
