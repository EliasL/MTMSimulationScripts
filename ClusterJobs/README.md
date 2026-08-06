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
