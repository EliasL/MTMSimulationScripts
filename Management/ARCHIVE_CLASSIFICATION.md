# Archive classification rules, version 1

`archiveClassification.py` classifies metadata collected by `archiveRuleSamples.py`.
Both are standalone stdlib Python 3.9+ programs. Neither imports or executes MTS2D,
changes source simulations, copies payloads, or chooses data for scientific fits.
The twenty real cases are a regression sample, not a complete archive inventory.

## Run

From the repository, using the project environment:

```bash
.venv/bin/python Management/archiveClassification.py \
  tests/fixtures/archive_rule_samples.json \
  --git-repo ../MTS2D --output output/archive_rule_classification
```

Default input is the frozen probe snapshot. `--live` validates the pinned
metadata against the original directory, ZIP, or SSH source first. A changed
source fails explicitly; do not silently refresh a fixture to make it pass.
An archive moved to another disk can edit the manifest's root locators.

A standalone bundle includes an exported Git evidence index, avoiding a Git
checkout at read time:

```bash
python3 archiveClassification.py archive_rule_samples.json \
  --code-index archive_rule_code_index.json --output results
```

Outputs: `classification.json` with structured evidence/reasons; `catalogue.csv`
with one row per sample and 200+ columns; `REVIEW.md` with per-case review items;
and `code_index.json` with full commit IDs, ancestry and historical stress code.
CSV columns include config settings, observed schema features, loading progress,
and separate scientific issue assessments. JSON retains duplicate header positions.

## Evidence and scope

- Config `# Version` markers resolve against Git, using ancestry rather than dates.
  The year is the commit's year. Era names identify the latest included audited
  milestone; they do not imply that every later commit shares its implementation.
- Version markers describe the checkout that wrote the config, sometimes using
  a `GIT_VERSION` fallback. They do not establish the executable, its uncommitted
  changes, or the code of resumed segments. Dirty markers retain this warning.
  Even a clean marker never certifies a simulation's scientific correctness.
- Unversioned/unknown markers retain a schema family without a guessed year.
  ZIP names, filesystem dates and old archive names cannot prove code provenance.
  No 2023 run is required. Current-main format and a positive gamma0 build manifest
  remain gaps in this real sample.
- Campaign signatures mirror settings in `Management/jobs.py`: legacy `propperJob`,
  `allPlasticEventsJob`, size scaling and experiment families. They are candidates,
  because `basicJob` and historical factories overlap. Diagonal/logging variants
  are excluded from the basic size-scaling signature. Paths supply separate context,
  including stress repair and delayed reconnection. Original job manifests would
  allow confirmed membership; no such confirmation is invented here.
- Legacy `reconnectingEnabled=true` does not automatically mean edgeFlip. Missing
  settings remain unknown. The explicit transport flag is recorded independently
  of code revision; 0 and 1 at the same revision are different configurations.

## Audited scientific milestones

| Rule | Revision | Precise interpretation |
| --- | --- | --- |
| Element reference area | `1f93290` | Element energy changed from density to `(density - ground_state_density) * initArea`. Both normalization and baseline changed. |
| Cauchy calculation defined | `cfb09cf` | Source contains the initial Cauchy function; this does not establish that it was called. |
| Cauchy update called | `f8af5e0` | Adds the missing update call to element updates. |
| Extensive snapshot trigger | `672288c` | Removes multiplication of the threshold by the element count and the positive-plastic-change requirement in snapshot selection. Does not fix all energy columns. |
| March stress formula | `cf71595` | Changes Cauchy from `F_fixed_ref` to `F`; also changes reversibility outputs. Not a universal correctness certificate. |
| April stress formula | `e5f8c03` | Changes Cauchy back to `F_fixed_ref`; historical formula needs frame/rotation review. |
| Stress area weights | `54898f7` | Reference-area weighted P and current-area weighted sigma replace element-count means. Average energy stays total energy per element. Earlier results need actual area-distribution review, not blanket rejection. |
| Transport setting | `deb9688` | Exposes and uses minimum_transport_distance; explicit config value still matters. |

Scientific issue statuses report whether an audited change is present in the
config checkout. An absent change is a risk flag with quantity-specific scope,
not proof of a numerical error in every run. Dirty base revisions receive no
stronger claim. Cauchy formulas are exported as historical source evidence;
reconnecting stress reconstruction needs total `T = F_e T_p`, not plastic `T_p`
or a rotation-losing reconstruction from the metric.

`Delta E_R`, `Delta E_I`, and `Delta E_S` semantics remain `not_established` until
an explicit transition/extraction audit. Column names alone cannot establish
these definitions or the gamma0 tangent and current first-Piola peak convention.
Repair names/paths also cannot establish them. The historical analysis audit
flags its associated saved analysis as historical; it does not disqualify the
raw macro CSV from a fresh extraction. Full audit-source hash association still
needs verification before associating its scientific result with an export.

## CSV and preservation behavior

Ordinary probes read only a config, the first three macro lines and an 8 KiB tail.
If tail row widths disagree with the initial header, the reader streams that
CSV to find embedded `#HEADER:` records and records its SHA-256 and header byte
offsets. This reads metadata CSV bytes, never meshes/checkpoint payloads. The
schema-changing partial run in this sample exercises that fallback. This is
not a universal full-file integrity or scientific validation: same-width schema
changes, internal malformed rows, and non-finite values outside probes may remain
undetected. ZIP probes remain prefix-only to avoid decompressing a large member.

Duplicate column positions are preserved, with ambiguous selection flagged.
Malformed/truncated tails are flagged and incomplete rows excluded. An unreadable
tail never causes a prefix load to be reported as the final load. Progress is
`(last_complete_sampled_load - startLoad) / (maxLoad - startLoad)`, clipped to 0–100,
only when target and increment directions agree. Negative-increment/positive-target
cases are left for review. A target reached with NaN energy is explicitly flagged;
load progress is not successful scientific completion. The last sampled finite
energy load is not necessarily the last finite load in the entire file.

Storage integrity, metadata confidence and scientific suitability are separate.
All twenty cases remain preservation candidates, including partial, header-only
and scientifically questionable runs. The classifier cannot authorize deletion
or claim a verified backup. This policy is advisory output here: the existing
`simulationArchive.py` importer still has its earlier metadata error gate; integrating
these classifications and relaxing that gate requires a separate preservation change.

## Tests

`tests/test_archive_classification.py` tests all twenty expected code-year/progress
cases, campaign candidates, dirty/unknown versions, source-scoped evidence, transport
pairs, fix boundaries, duplicate headers, resumed schemas, malformed and truncated
rows, NaN energies, historical analysis scope and CSV export. Classification only
receives raw records, source path and the Git index. Labels, case IDs, coverage tags,
precomputed code hints and expected observations are never rule inputs.
