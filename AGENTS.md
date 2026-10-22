# MTS2D version compatibility

MTS2D has two actively relevant branches. `main` is **not backwards
compatible** with the older simulation configuration and checkpoint formats.
Use `main` for new work that explicitly targets the current simulator, including
the MAGI curvature benchmark. The legacy PhD branch is named `MTS2D_PhD` in
Git; use it for older jobs and their existing configurations, dumps, and
continuations.

Before running, resuming, or diagnosing a job, identify its required MTS2D
branch and verify the source revision and executable actually used. A legacy
job failing on `main` may simply need `MTS2D_PhD`; do not "fix" its settings,
readers, or scientific behavior to accommodate the wrong simulator version.
Keep branch-specific source/build directories and record the exact revision.
The current upload/build helpers target a shared `~/simulation/MTS2D`
directory, so check which version existing jobs need before replacing it.

# Energy-drop power-law analysis

## Delta E_S convention

Compute the second-order stress-corrected `Delta E_S` with the fixed elastic
shear tangent at `gamma=0`, using the starting-step `avg_sigma12` for the
linear term. The former strain-dependent tangent convention is archived as
`old_dynamic_mu`; never use its drop tables, caches, or fitted plots as input
to a new analysis. Sigma-rescue snapshot readers must require a build manifest
with `delta_E_S_tangent: gamma0` and the current first-Piola peak post-yield rule.

The archived `Plots/powerLaw/truncated_powerlaw_reversibility_flowchart/panels/mle_fit.pdf`
predates the gamma=0 convention: it used the stress-peak +0.01 window and a
strain-dependent tangent. Reading today's plotting code does not establish
the convention of that saved figure. The isolated historical diagnostic in
`Plots/powerLaw/panel_f_yield_definition_comparison/historical_audit.json`
reproduces its full irreversible-drop population hash and fit exactly.

## Standard protocol

When a request asks for the standard energy-drop power-law result, use this
sequence unless the request explicitly specifies another analysis:

1. Find each run's global maximum of the native first Piola-Kirchhoff
   component `avg_P12`, using the first occurrence if maxima tie. Post-yield
   transitions satisfy `load_ip1 > peak_load + 0.01`; pre-yield transitions
   satisfy `load_ip1 < peak_load - 1e-4`. Omit the intervening buffer and
   retain the existing strict upper end of the available post-yield window.
   Use `yield_peak_load`, `yield_regimes`, and `yield_region_mask` in
   `Plotting.standardPowerlaw`; `findPrePostSplit` returns the unbuffered peak.
   Missing or non-finite `avg_P12` is an error: never substitute `avg_sigma12`
   or a fixed strain. Reject selections and figures made using fixed 0.7
   or the former Cauchy-stress peak. `avg_P12` defines the window only;
   the energy correction still uses `avg_sigma12` and fixed gamma=0 modulus.
2. Extract `Delta E_R` and `Delta E_S` as paired values from the same event
   transitions, in the same order and with the same length.
3. Compute `kappa = Delta E_R / (rho V_0 Delta gamma^2)` with
   `rho=N/V_0=2` and use the fixed classification threshold
   `kappa_det = mu / (2 rho)`. Label it in
   plots as `kappa_det`. It is a detection/classification threshold, not an
   energy minimum used for the final power-law fit.
4. Apply the resulting reversible/irreversible labels to the paired
   `Delta E_S` values. Filter for finite positive `Delta E_S` only after the
   labels have been assigned.
5. Fit only `es_irrev`, the finite positive `Delta E_S` values belonging to
   irreversible events.
6. Evaluate every observed candidate cutoff in `es_irrev`, select the true
   global KS minimum, and call it `es_xmin_ks`. Label it in plots as
   `Delta E_{S,\min}^{KS}`.
7. Keep that cutoff fixed while estimating the truncated-power-law
   parameters, including `alpha` and `lambda` (or `1/lambda` where that is
   the established plotting convention).
8. For the reference Clauset p-value, re-select `xmin` with the full global
   search in every bootstrap sample. A 100-point coarse scan plus local
   refinement may be used only as an explicitly labeled bootstrap
   approximation; it must never replace the empirical full-global search.
   When comparing the approximation, use identical bootstrap samples and
   report how often `xmin` matches and how far `xmin` and KS distance differ.

Always report the p-value for the selected `es_xmin_ks` fit (not a scan of
p-values over candidate cutoffs). The same refit bootstrap should also report
the bootstrap mean and one standard deviation for the reselected cutoff and
fitted `alpha`. Each bootstrap dataset has the original event count. Distinguish
the empirical `alpha(xmin)` cutoff-sensitivity curve from the model-bootstrap
`alpha` spread; the latter is not obtained by reading the empirical curve at
the cutoff interval's endpoints.
Use `confidence=0.05` as the suggested bootstrap precision default for the
selected ("best") fit parameters and their p-value: this requests 100 replicates
per population using `ceil(1 / (4 confidence^2))`. This is a Monte Carlo
precision target, not a confidence-interval level, and is not a strict
requirement. Explicit cutoff-scan diagnostics may use fewer replicates for
surrounding candidates, for example `confidence=0.1` (25 replicates), while
using 0.05 for the selected fit. Such scans do not replace reporting the
p-value and refit-bootstrap statistics for the selected `es_xmin_ks` fit.

Omit this expensive bootstrap only when explicitly requested, and clearly
mark the resulting plot as lacking a p-value and uncertainty intervals.

The pairing contract is represented by `EventDrops` in
`Plotting/standardPowerlaw.py`. Do not independently sort, filter, or
concatenate `Delta E_R` and `Delta E_S` before classification. A generic
all-event fit, an `Delta E_R` fit, an Otsu/slope split, or a coarse/local
cutoff search is an alternative analysis and should be named and reported
as such.

## Reuse existing power-law figures first

Before writing any energy-drop plotting or fitting code, look for an existing
function that produces the requested figure and use it with priority. For a
single system size, the canonical entry point is
`Plotting.truncated_powerlaw_reversibility_flowchart.generate_single_size_panels`.
It makes the existing flowchart's `reversibility_er` (classification),
`reversibility_es` (paired reversible/irreversible split), `xmin_scan`
(KS cutoff diagnostic), and `mle_fit` (panel-f-style power-law fit) plots.
Its CLI accepts `--single-size L`,
`--data-root ROOT`, and `--skip-pvalue`; skipping the bootstrap does **not**
skip the exhaustive empirical KS cutoff or the fitted PDF. The flowchart's
automatic post-yield window starts strictly after each run's `avg_P12` peak +0.01;
state any different window explicitly when comparing results.

Do not reimplement these individual-size plots in a `sizeScaling` script or
temporary helper. Size-scaling scripts should focus on cross-size comparisons,
collapse, or analyses that the existing functions cannot produce. Reuse
`Plotting.plotPowerLaw` and `Plotting.standardPowerlaw` for any genuinely new
analysis, and document why the existing plot entry point cannot be used.

`Plotting.sizeScalingCollapse` defaults to the same positive irreversible
`Delta E_S` population for cross-size plots. Share the classification cache
through `--standard-cache-root`; classify paired events before filtering E_S.
Its collapse uses normalized tails above the per-size global KS cutoffs.
For a cutoff-sensitivity comparison, use `--stage shared --common-cutoff-size 100
--skip-pvalue` to write `*_common_cutoff.pdf` with the L=100 cutoff for all sizes.
This changes both the plotted tail range and its normalization; it does not
replace the per-size KS fits. Collapse quality scores use their respective
overlap ranges and are not direct like-for-like comparisons across cutoffs.
Sizes below L=100 are excluded by default. For an explicitly requested L=50
comparison, pass `--minimum-size 50` to the collapse/classification entry points
and use a separate output folder.
For the alternative cutoff-derived presentation, `--stage lambda-scaling
--yield-regime post --skip-pvalue` writes lambda scaling, a fixed-exponent
collapse, raw PDFs with per-size cutoff markers, and the KS scan as four
individual PDFs. It fits
`log(lambda) = log(A) - D_lambda log(L)` across all included sizes and uses
the L=250 truncated-power-law `alpha` directly; it never optimizes collapse
parameters. This collapse defaults to the common L=100 cutoff
(`--lambda-cutoff common --common-cutoff-size 100`). Use
`--lambda-cutoff per-size` for an explicitly requested individual-cutoff comparison.
Lambda values always come from the per-size KS fits; this cutoff option
only controls which normalized tails are shown in the collapsed PDF.
For a full-PDF normalization comparison, use `--lambda-normalization full
--lambda-cutoff per-size` in a separate folder. This bins and normalizes the
entire positive irreversible E_S population, then masks complete bins below
each KS cutoff without renormalizing the retained tail. The default `tail`
normalization instead divides by the fraction above the applied cutoff;
scaling exponents do not automatically remove this threshold-dependent factor.
`--population-mode all` is an explicit alternative with separate output paths.

Use all available samples extending beyond their own `avg_P12` peak +0.01, including partial
runs, unless an explicit sample limit is requested. Do not impose equal seed
counts or require completion to gamma=1. Use `size_scaling_paths` for this
selection; `completed_size_scaling_paths` is only for explicitly requested
completion-limited comparisons. Record exact source CSVs and sample counts.
Extraction cache keys include the yield-rule string, buffered windows, and
cache version. Current outputs use `_piola_peak` directories. Old fixed-0.7
outputs remain historical artifacts and must not be reused as current fits.
Unfiltered gamma=0 sigma-rescue drop tables can be reused through
`build_piola_yield_view`, which reads native P12 from the frozen source macros
and writes a fresh `yield_peaks.csv` and manifest before selecting events.
Use `Plotting.plotPowerLaw.getHist` for the histogram bins shared by raw PDFs
and collapse curves, with five bins per decade by default. Collapse tail
normalization is a separate, explicit choice from full-PDF normalization.

## Default plotting and animation entry point

Use `plotAll.py` as the standard entry point for default plotting and
animation generation. It routes animation jobs through the project’s
standard `Plotting.makeAnimations` pipeline and should produce the usual
MP4 outputs. A mesh video produced by `plotAll.py` is a standalone standard
mesh animation.

Element-tracking animations are a separate specialized pipeline in
`Plotting/element_tracking_animation.py`. Its final Poincare-disk composition
uses the transparent local neighbourhood from `render_mesh_animation` and the
tracking-specific periodic overview from `render_periodic_mesh_animation`.
Do not substitute the standalone `plotAll.py` mesh video for either of those
composition inputs. For full-deformation tracking, use the total matrix from
`VTUData.get_T()` (including the elastic component), not the plastic matrix
alone.

For Cartesian periodic mesh plots, determine and freeze the plot limits from
the original untiled mesh (or an explicit requested viewport) before drawing
periodic copies. Tiling exists only to cover that fixed viewport and eliminate
white space; tiled copies must never expand or otherwise determine the limits.

Keep animation rendering conservative with memory. The standard default is at
most two rendering workers, and meshes with `L >= 200` are restricted to one
worker. Do not increase this automatically from the CPU count, do not render
multiple animation jobs concurrently, and only raise the worker count when
available memory has been checked explicitly.

## Reconnecting deformation and stress

For reconnecting edge-flip simulations, the total matrix
`T = F_e T_p` is the reconnection-invariant replacement for `F`. The VTU
fields named `T11`, `T12`, `T21`, and `T22` are legacy notation for `T_p`;
obtain the total matrix through `VTUData.get_T()`. Use the full total `T`
directly wherever a deformation gradient is needed, including Cauchy-stress
calculations. Do not reconstruct `F` from `C = T.T @ T` with `F_from_C` for
reconnecting data: that drops the polar rotation and changes fixed-coordinate
Cauchy-stress components. `F` itself is affected by reconnection and must not
be treated as reliable in those simulations. A rotation-fixed metric
representative is appropriate only for a deliberately theoretical curve, such
as the loss-of-ellipticity boundary, where no reconstructed mesh is involved.

## Finding simulation data

Use system sizes L >= 100 for size-scaling analyses and figures. Omit L=50
from fits, plotted curves, legends, and captions; do not create separate
"without L=50" variants. Historical job factories and source files may still
contain that size so that old campaign configurations remain discoverable.

Unless explicitly requested otherwise, save each plot in its own PDF.
Collapse curves and collapse-quality maps must be separate figures; do not
combine them, fitted PDFs, cutoff scans, or other diagnostics into panel grids.

Prefer size-scaling jobs for figures and analyses unless the request specifies
another dataset; data from other jobs are also sometimes appropriate. Match
the exact configuration, size, and seed before selecting a similarly named
run. Check the last valid `load`, since a local copy may be incomplete.

For plots that only need `macroData.csv`, use
`Plotting.remotePlotting.get_csv_files(configs)` to search cached data, the
local drive, and configured clusters and fetch the CSV. Use
`Management.jobs.nonreconnecting_size_scaling_job()` and
`reconnecting_size_scaling_job()` to identify campaign configurations, but
verify each generated `config.name`: the historical L=100, seed-0 edge-flip
run uses `t3`, while the current reconnecting factory generates `t2` for
that case. `Management/README.md` documents the search helpers, including
`get_folders_from_servers` when a full output folder is actually needed.

For example, the L=100, seed-0 non-reconnecting run
`simpleShear,s100x100l0.15,1e-05,1.0PBCt3LBFGSEpsx1e-06s0` is on Schwartz
under `/data2/elundheim/MTS2D_output`; the corresponding reconnecting run
`simpleShear,s100x100l0.15,1e-05,1.0PBCedgeFlipt3LBFGSEpsx1e-06s0` is on
Pascal under `/data/elundheim/MTS2D_output`. Do not infer that the first run
is absent because it is missing from a local `sizeScalingJobs` directory.


## Python environment and headless plotting

Use the project virtual environment rather than the system Python. Its
interpreter is:

```text
/Users/elias/Work/PhD/Code/SimulationScripts/.venv/bin/python
```

From the repository root, set `MPLBACKEND=Agg` for non-interactive plot
generation and `MPLCONFIGDIR=.matplotlib-cache` for a writable Matplotlib
cache. Scripts under `Plotting/` that import top-level project packages also
need `PYTHONPATH=.`. Working commands include:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache \
  ./.venv/bin/python -c \
  'from plot import plotReductionHistory; plotReductionHistory(show=False)'

MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache PYTHONPATH=. \
  ./.venv/bin/python Plotting/plasticReductionDeterminantQuadrantsIllustration.py

MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache PYTHONPATH=. \
  ./.venv/bin/python -m unittest \
  tests.test_reduction_history_plot \
  tests.test_plastic_reduction_determinant_quadrants_illustration
```
