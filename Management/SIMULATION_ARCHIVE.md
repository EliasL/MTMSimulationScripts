# Simulation archive: local practice workflow

`simulationArchive.py` is a standalone Python 3.9+ program using only the
standard library. The repository is the maintained source; a versioned copy and
this guide can live under the backup folder's `tools/` directory. It never
deletes or relocates a source simulation.

This first version supports local inventory, planning, copying/compressing one
unit at a time, and offline archive verification. SSH/rsync transport, a batch
download queue, and the scientific Git-history audit are later stages; these
are not silently approximated here. The same inventory command can run on a
cluster without importing this project or its Python dependencies.

## Inventory first

From the repository root:

```sh
./.venv/bin/python Management/simulationArchive.py inventory \
  /Volumes/data/MTS2D_output output/archive_practice_local/catalogue \
  --source-label local-data \
  --active-run 'reversibilityProtocolTest,s200x200l0.15,1e-05,2.0PBCt3epsR0.0LBFGSEpsg0.0LBFGSEpsx1e-06s0'
```

Inventory reads sources and writes only to the specified catalogue. It finds
nested runs as well as loose ZIPs, videos, and unclassified directories. Exact
logical sizes come from every file, rather than sampling. Raw PNGs in frame
directories are explicitly excluded; movies, frame ZIPs and other contents are
retained. Symlinks and unsupported file types are reported as errors rather
than followed. Empty directories are represented.

The catalogue contains:

- `catalogue.sqlite`: source-instance IDs and current metadata.
- `simulations.csv`: wide overview, including every recorded config parameter
  and presence/absence of every observed macro column.
- `assets.csv`: loose archives, videos and non-simulation assets.
- `manifests/*.json.gz`: file paths, byte sizes, modification times and
  directory lists. Large payloads are not hashed during inventory.

A source-instance ID distinguishes server/root/path combinations. Matching
names or configurations never merge two executions. Rescans retain archive
UUIDs for the same source instance. Keep using the same source label. Moving
the catalogue is supported, but changing the *source* root/label creates new
source instances until an explicit identity mapping is added.

Config `# Version:` comments are recorded as evidence about the config-writing
checkout, not certified executable versions. Stress/energy conventions and
scientific quality default to unknown/not audited. Macro-column presence does
not certify its definition, normalization, or reconstructibility. In
particular, `column.delta_E_S` is not evidence of the gamma-zero convention.

Legacy missing-index rows are counted explicitly. Malformed rows, missing
configs/macros, header-only macros and unterminated CSV lines block import.
Completion is based on `(last_load-startLoad)/(maxLoad-startLoad)` only for
monotonic increasing load ranges; other protocols require interpretation.
Reaching the target does not certify numerical quality or file completeness.

Explicitly declare known running sources using `--active-run`; patterns are
relative to the inventory root. Absence from this list is not process-based
proof that a simulation has stopped. File changes during scanning/import also
block publication. These checks do not replace filesystem snapshots.

## Assign campaigns and make a plan

Years are not inferred from copied file timestamps. Use a separate annotations
JSON keyed by the exact source-relative path, for example:

```json
{
  "sizeScalingJobs/example-run": {
    "year": "2026",
    "campaign": "size_scaling_reconnecting",
    "comments": "Year established from the campaign submission record"
  }
}
```

Pass `--annotations annotations.json` on a rescan. Annotations may also set
`scientific_quality`, but should cite supporting evidence. `campaign_suggestion`
is only a hint based on the existing grouping/experiment; it is not an approved
assignment. The CSV is a generated view: edit annotations rather than editing
CSV cells that a rescan will replace.

```sh
./.venv/bin/python Management/simulationArchive.py plan \
  output/archive_practice_local/catalogue \
  /Volumes/data/Lundheim_MTM_PhD_backup_2023_2026
```

`plan.json` distinguishes ready, needs-assignment and blocked units. Planning
does not create destination simulation folders. Unknown years can remain
explicitly `unknown` during a pilot rather than inventing a production date.
Use `export CATALOGUE OUTPUT_DIRECTORY` to publish the generated overview CSVs
beside the years at the backup root; the SQLite catalogue remains the source
for those generated views.

## Import a single practice simulation

Select a small, stopped simulation with no inventory errors. Use its full
`archive_id` from the CSV/database:

```sh
./.venv/bin/python Management/simulationArchive.py import \
  output/archive_practice_local/catalogue \
  /Volumes/data/Lundheim_MTM_PhD_backup_2023_2026 \
  --archive-id UUID --campaign archive_practice
```

The default free-space reserve is 100 decimal GB, and the importer additionally
budgets the entire source logical size without assuming compression savings.
Work goes into `staging/UUID` and becomes visible in its final campaign only
after verification. `data/` and `dumps/` become separate `data.tar.gz` and
`dumps.tar.gz` files inside that simulation's folder. CSVs, configuration,
videos, logs and other directories remain accessible. Existing gzip dumps may
compress little further. The source is retained, including omitted frames.

Every copied file and every compressed archive member is checked with SHA256.
`archive_manifest.json` records the source snapshot, metadata and checksums.
An interrupted invocation can be repeated: matching copied files and complete
verified payload archives are reused; incomplete tar streams restart. This
does not yet provide byte-level remote transfer resumption. A completed import
is idempotent and is verified on a repeated invocation.

```sh
python3 simulationArchive.py verify /path/to/one/imported/simulation
```

Verification reads only the imported directory, so it also works after moving
the backup to another disk or disconnecting the original source. Missing,
changed and unexpected output files cause an error. This first version does
not execute code, extract archive members, infer historical corrections,
perform power-law analysis, or delete server data.

Before full migration, add the SSH/rsync backend, benchmark compression on
representative large runs, audit provenance/conventions, and settle campaign
assignments. Budget disk migration by logical size: APFS shared copies may
expand when copied to another filesystem.
