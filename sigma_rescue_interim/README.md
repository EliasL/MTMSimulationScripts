# Interim sigma-rescue drop snapshots

This directory is the local, read-only analysis layer for using completed
sigma-rescue segments before the full rescue campaign finishes. Cluster source
simulations and rescue outputs remain immutable. Every acquisition creates a
new timestamped snapshot below `snapshots/`; an existing snapshot is never
updated in place.

## Safe data contract

For each completed non-reconnecting run, download only:

- the original `macroData.csv`, which is authoritative for load, energy and
  event variables;
- the original `config.conf`;
- each accepted rescue task's `result.json`;
- each accepted rescue task's `validated_sigma.csv`.

Do not download replay `macroData.csv` files or dumps for this analysis. Do not
use old-schema source stress, and never substitute `avg_P12` for `avg_sigma12`.

The three aligned event quantities are:

- `delta_E_I = -total_energy_change` at the ending row;
- `delta_E_R = -total_e_change_from_init` at the ending row;
- `delta_E_S`, calculated by
  `Plotting.energyDropCalculations.calculate_energy_step_data` using the
  corrected `avg_sigma12` and the current-strain second-order tangent.

An event is usable only if:

1. its two source rows have consecutive `load_step` values and the configured
   `loadIncrement`;
2. both stress rows are available, finite and not the rescue sentinel `-1`;
3. at least one accepted rescue task covers both rows, or both rows belong to
   the native correct-new schema;
4. all three drop values are finite.

Condition 3 deliberately removes transitions across independently replayed
segment boundaries. It costs at most a small number of events and avoids
inventing stress-corrected drops from an artificial boundary. Duplicate source
keys, overlapping rescue values that disagree, missing columns and unsafe file
paths are fatal errors.

## Proposed snapshot layout

```text
snapshots/<UTC timestamp>/
  inventory.json
  raw/L050/seed_000/<run name>/
    macroData.csv
    config.conf
    rescue/<task id>/
      result.json
      validated_sigma.csv
  tables/by_run/L050/seed_000/drops.csv.gz
  tables/drops_all.csv.gz
  tables/coverage.csv
  tables/exclusions.csv.gz
```

Raw snapshot files should be recorded with remote path, byte size and SHA256.
The table should retain `size`, `seed`, `run_name`, source step/load keys,
stress provenance, all three drops, a `usable` flag and an explicit exclusion
reason. Seed—not replay segment—remains the statistical sample unit.

## Implementation sequence

1. Implement `discover_remote_campaign` in
   `Management/sigmaRescueDropSnapshot.py`. Reuse the campaign manifests and
   accept only `validated` or `validated_with_sentinels` results. Initially
   restrict this to source runs whose original simulation reached `maxLoad`.
2. Write the immutable `inventory.json`, review its exact file list and sizes,
   then call `download_artifacts(..., dry_run=False)`.
3. For each run, call `merge_available_sigma`, then
   `build_interim_drop_table`. Write both all audited transitions and the
   `usable == True` subset.
4. Build `coverage.csv` by size, seed and strain bin. Current rescue completion
   is ordered by cluster task scheduling and is not automatically an unbiased
   sample of strain.
5. Before scaling fits, select a common post-yield strain window and report the
   number of seeds/events per size. Prefer a balanced seed subset as a
   sensitivity check.
6. Feed paired `delta_E_R`/`delta_E_S` rows into `EventDrops`. Apply the normal
   `simpleDrop` classification to `delta_E_R`, then fit the positive
   irreversible `delta_E_S` population using the global KS minimum.

The implementation should also report how many events were excluded for each
reason. Do not derive a yield point from a partial reconstructed stress curve;
use a fixed, documented strain window until complete curves are available.

