# Energy-drop power-law analysis

## Standard protocol

When a request asks for the standard energy-drop power-law result, use this
sequence unless the request explicitly specifies another analysis:

1. Restrict extraction to the post-yield region by default.
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

The pairing contract is represented by `EventDrops` in
`Plotting/standardPowerlaw.py`. Do not independently sort, filter, or
concatenate `Delta E_R` and `Delta E_S` before classification. A generic
all-event fit, an `Delta E_R` fit, an Otsu/slope split, or a coarse/local
cutoff search is an alternative analysis and should be named and reported
as such.

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
