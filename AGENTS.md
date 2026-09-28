# Energy-drop power-law analysis

## Delta E_S convention

Compute the second-order stress-corrected `Delta E_S` with the fixed elastic
shear tangent at `gamma=0`, using the starting-step `avg_sigma12` for the
linear term. The former strain-dependent tangent convention is archived as
`old_dynamic_mu`; never use its drop tables, caches, or fitted plots as input
to a new analysis. Sigma-rescue snapshot readers must require a build manifest
with `delta_E_S_tangent: gamma0`.

## Standard protocol

When a request asks for the standard energy-drop power-law result, use this
sequence unless the request explicitly specifies another analysis:

1. Restrict extraction to the post-yield region by default. Determine yield
   separately for each run from the maximum valid Cauchy shear stress
   `avg_sigma12`, and start at `gamma_peak + 0.01`. If the stress peak cannot
   be established, exclude that run and report it. Never use `0.7` as an
   automatic post-yield cutoff when stress data are available; that value was
   only a temporary proxy for missing stress data.
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
95% percentile intervals for the reselected cutoff and fitted `alpha`.
Each bootstrap dataset has the original event count; the 95% refers to the
2.5th–97.5th percentiles of refitted estimates, not to using 95% of events.
Distinguish the empirical `alpha(xmin)` cutoff-sensitivity curve from the
model-bootstrap `alpha` interval: the latter is not obtained by reading the
empirical curve at the cutoff interval's endpoints.
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
automatic post-yield window starts 0.01 load after each run's stress peak;
state any different window explicitly when comparing results.

Do not reimplement these individual-size plots in a `sizeScaling` script or
temporary helper. Size-scaling scripts should focus on cross-size comparisons,
collapse, or analyses that the existing functions cannot produce. Reuse
`Plotting.plotPowerLaw` and `Plotting.standardPowerlaw` for any genuinely new
analysis, and document why the existing plot entry point cannot be used.

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
