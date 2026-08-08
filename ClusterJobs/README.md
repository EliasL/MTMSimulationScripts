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
