# Common-event minimizer comparison CSV

This README documents:

`common_event_minimizer_comparison_all_events.csv`

## Scope

The CSV contains one row per completed matched event from:

`size_scaling_L100_post_yield_events_epsg1e-5_epsx0`

It contains 246 events across seeds 0–9. Each event starts FIRE, CG, and LBFGS from the same saved pre-event configuration. The older 11-event legacy comparison is not included.

The CSV is a processed summary. It does not contain every minimization iteration. The full iteration histories remain in the compressed raw output folders.

## Naming convention

Most columns follow this pattern:

`<minimizer>_<section>_<quantity>_<statistic>`

where `<minimizer>` is `FIRE`, `CG`, or `LBFGS`.

### Event and source metadata

These columns identify the observation and its provenance:

- `seed`, `event_index`, `event_id`: stable event identifiers.
- `source_load`: load of the source dump used to search for an event.
- `pre_event_load`: load of the saved configuration immediately before the event.
- `event_relative_path`, `*_relative_path`: paths relative to the comparison-data root and are preferred after archiving.
- `*_path`: original absolute paths at extraction time; these may become invalid after moving or unpacking data elsewhere.
- `*_present_at_extraction`, `*_present`: whether the referenced raw file or plot existed when this CSV was created.

### Configuration columns

`source_config_*` describes the original source simulation. `collector_config_*` describes the forward search used to find the event. The values are copied from the corresponding `.conf` files.

Important settings include:

- `rows`, `cols`: lattice dimensions.
- `startLoad`, `loadIncrement`, `maxLoad`: loading protocol.
- `LBFGSEpsg`, `LBFGSEpsx`, `eps`: minimization tolerances.
- `energyDropThreshold`: threshold used to identify an energy-drop event.
- `reconnectionMethod`, `usingPBC`: topology and boundary-condition settings.

### `<MINIMIZER>_first_drop_*`

These are the fields from the stored `first_drop` record for that minimizer. The record is the simulation row where the wrapper detected the first negative energy change. It is not necessarily the final converged state, and different minimizers can reach the first drop at slightly different load steps.

The fields include:

- energy: `total_energy`, `avg_energy`, their changes, and energies relative to initialization;
- mechanics: `avg_sigma11`, `avg_sigma12`, `avg_sigma22`, and `avg_P11`–`avg_P22`;
- convergence: `nr_iterations`, `nr_func_evals`, `max_force`, and termination reasons;
- plasticity/event structure: `participationFraction`, `m3_participationFraction`, `sum_m3`, `nr_elements_with_m3_change`, and plastic-jump counts;
- geometry: center of mass, coordinate extrema, and mesh-reduction counts;
- runtime: `run_time`, `minimization_time`, `write_time`, plus parsed numeric `*_seconds` columns.

The tensor labels such as `P11` and `sigma12` are component names emitted by MTS2D. Their physical normalization should be interpreted using the MTS2D model/code definitions; the CSV does not convert units.

### `<MINIMIZER>_trajectory_*`

These are compact summaries of the retained minimization `macroData.csv` used for the convergence plot. For most quantities, the suffix gives the statistic:

- `_initial`: first recorded value;
- `_final`: last recorded value;
- `_minimum`: lowest recorded value;
- `_maximum`: highest recorded value.

Other useful fields are:

- `trajectory_rows`: number of retained minimization records;
- `trajectory_nonfinite_count`: count of non-finite numeric values encountered;
- `trajectory_nonmonotone_total_energy_steps`: count of energy increases larger than the comparison tolerance.

The summarized quantities include total/average energy, energy changes, force, function evaluations, iterations, participation fractions, plastic-state counts, and maximum energy.

Important distinction: `trajectory_total_energy_change_*` and
`trajectory_avg_energy_change_*` summarize the MTS2D `total_energy_change` and
`avg_energy_change` columns. Those are changes relative to the previous load
step, not the change between the final two optimizer evaluations. The latter is
stored only in the raw `min_iter_total_energy_change` and
`min_iter_avg_energy_change` columns inside each retained `macroData.csv`.

### Derived comparison columns

- `*_winner`: minimizer with the lowest value for that metric. Lower is treated as better.
- `*_order_low_to_high`: ranking such as `LBFGS<FIRE<CG`; `=` indicates a tie.
- `*_difference_FIRE_minus_CG`: arithmetic difference `FIRE - CG`. A negative value means FIRE is lower for that quantity.
- `first_drop_load_span`: difference between the largest and smallest minimizer-reported first-drop load.
- `first_drop_loads_match_within_5e_minus_9`: whether all three first-drop loads agree within `5e-9`.

Energy winner/tie fields in this export use the following absolute tolerance:

`comparison_energy_tie_absolute_tolerance = 1e-10`

However, `total_energy` is written with 11 significant digits. Around total
energies of a few hundred, that gives a stored-value resolution of roughly
`1e-8`, so the `1e-10` comparison is finer than the precision of that field.
Those tie columns should therefore be treated as historical diagnostics, not
as evidence that two physical minima agree to `1e-10`. A reliable strict test
requires higher-precision energies and synchronized loads.

Function-call and force comparisons use exact numeric ordering unless the values are identical.

## Interpretation cautions

1. Function-call counts are directly useful for comparing minimizer effort.
2. First-drop energies can be evaluated at slightly different load steps, so energy-winner counts should be treated as a diagnostic rather than a perfectly synchronized final-energy comparison.
3. The trajectory minimum/final fields summarize the retained minimization directory, not every raw file in the run directory.
4. Empty cells mean that the source record did not provide that field or that the value was not numeric; they are not automatically zeros.
5. In the 246-event collection, 200 events have identical recorded first-drop
   loads for FIRE, CG, and LBFGS. The previously reported number 169 refers to
   the historical union of energy-tie flags, not to a common-load event count.
