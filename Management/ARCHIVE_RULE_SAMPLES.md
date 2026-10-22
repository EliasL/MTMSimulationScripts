# Archive-rule real-data sample

The machine-readable list is `tests/fixtures/archive_rule_samples.json`.
It selects 20 real sources for developing code-era, campaign, schema and
scientific-issue rules. This is a targeted edge-case sample, not a statistical
sample of simulations. There is no requirement for a run in every calendar
year. The available older archive evidence is from 2024; resolved config code
revisions span 2025–2026.

Each entry has an exact root, source-relative directory or ZIP prefix, file
locators, coverage tags, observations, and config-version evidence. Config
markers identify a candidate code era rather than a certified executable or
all resumed segments. Fixed-commit ancestry is recorded mechanically; it is
not a conclusion that a numerical issue affected the run. Paired temporal
cases may have different physical settings and are not controlled numerical
comparisons. The minimum-transport pair is specifically selected to contrast
an explicit setting at the same code revision.

## Selected cases

| Case | Config-code date, if known | Selection purpose |
|---|---|---|
| `01_galois_unversioned` | Unresolved | Unversioned Galois legacy CSV without a row index |
| `02_early_archive_no_config` | Unresolved | Early archived run with no configuration |
| `03_legacy_binary_checkpoint` | Unresolved | Legacy config with inline defaults and binary checkpoints |
| `04_dirty_2025` | 2025-01-20 | Dirty 2025 checkout, repeated headers and serialization transition |
| `05_decreasing_loading` | 2025-02-28 | Early remeshing with a negative loading increment |
| `06_fire_2025` | 2025-07-04 | 2025 FIRE minimizer campaign and initial-energy fields |
| `07_boolean_reconnection` | 2025-10-13 | Early reconnecting run using a boolean instead of an enum |
| `08_initial_cauchy_export` | 2026-01-19 | First Cauchy-stress export era |
| `09_before_extensive_trigger` | 2026-03-11 | Before the extensive energy-drop output-trigger change |
| `10_extensive_trigger` | 2026-03-17 | Energy-drop trigger change with a misleading historical folder name |
| `11_cauchy_fix_truncated` | 2026-03-24 | Cauchy/reversibility export changes with a malformed trailing CSV fragment |
| `12_metric_cauchy` | 2026-04-24 | Metric-representative Cauchy stress era |
| `13_before_area_weighting` | 2026-08-19 | Reconnecting reversibility run before the area-weighting fix |
| `14_after_area_weighting` | 2026-09-03 | Later code era after the area-weighting fix |
| `15_transport_disabled` | 2026-09-25 | Minimum transport explicitly disabled despite a late code revision |
| `16_transport_enabled` | 2026-09-25 | Matched configuration with minimum transport enabled |
| `17_partial_size_scaling` | Unresolved | Partial historical size-scaling source with unknown revision |
| `18_sigma_rescue_replay` | Unresolved | Short stress-repair replay with an unknown config revision |
| `19_failed_header_only` | 2026-09-28 | Failed retry: header-only CSV and an individually archived crash checkpoint |
| `20_historical_derived_analysis` | Unresolved | Historical corrected macro and audited old analysis conventions |

## Fast offline development

`tests/fixtures/archive_rule_probes.json` contains approximately 188 KB of
actual metadata content: complete small configs, three-line CSV previews,
local/SSH CSV tails, and the historical audit. Payloads are not copied.
Checksums pin those observations. Use it to develop initial rules without
repeated server access:

```sh
./.venv/bin/python Management/archiveRuleSamples.py \
  tests/fixtures/archive_rule_samples.json --snapshot \
  --output output/archive_rule_samples/snapshot_validation.json

./.venv/bin/python -m unittest tests.test_archive_rule_samples
```

The output explicitly marks frozen-snapshot mode. These probes do not replace
full CSV readers or mesh/checkpoint validation: tails can begin mid-row, and
ZIP macros have only a prefix preview to avoid scanning a large compressed
member. Every entry retains exact full-source locators for later tests.

## Check live access

```sh
./.venv/bin/python Management/archiveRuleSamples.py \
  tests/fixtures/archive_rule_samples.json \
  --output output/archive_rule_samples/live_validation.json
```

The Galois case requires existing SSH access and Python 3 on that host. The
reader runs only a bounded read-only metadata probe. It never starts MTS2D,
extracts a complete ZIP, downloads full simulation payloads, changes source
files, or substitutes another similarly named run.

Use `--case CASE_ID` to select one or more entries. `--offline` skips live SSH
access and labels it `not_checked_offline`, rather than reporting success.
`--root NAME=/new/path` overrides a root after a disk/repository move. Roots
are explicit, including the individual ZIP paths, so no filesystem search or
filename guessing is needed. Missing files, changed frozen metadata or an
unexpected version marker fail the check.

For a standalone copy, put `archiveRuleSamples.py`, `archive_rule_samples.json`
and `archive_rule_probes.json` together and run:

```sh
python3 archiveRuleSamples.py archive_rule_samples.json --snapshot \
  --output snapshot_validation.json
```

## What the assertions mean

The current tests establish accessibility, stable metadata identities and
known format/config facts. They do not yet certify historical scientific
corrections. In particular, the historical-analysis case distinguishes the
saved analysis from reusable raw macro data; it does not forbid rebuilding a
new analysis from that macro data.

Future additions should target remaining capabilities rather than arbitrary
calendar years: a positively identified gamma-zero build-manifest/drop-table
pair, a current main-format simulation if that format enters the archive, and
cyclic-loading or additional layouts that the baseline rules cannot resolve.
The current main simulator is not backwards compatible with these legacy PhD
jobs. All sample operations are metadata-only; choosing an executable and
resuming a run is a separate, provenance-checked operation.
