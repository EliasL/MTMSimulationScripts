These scripts are tools to manage the Mesoscopic Tensorial Simulation (MTS)

## Standard power-law energy-drop workflow

When a request says “power-law fit”, “power-law plot”, or asks for the
standard energy-drop result without specifying another population, use this
recipe:

The second-order correction in $\Delta E_S$ uses the fixed reference shear
tangent $\mu_0=a_{1212}(0)$. Outputs made with $a_{1212}(\gamma)$ are archived
as `old_dynamic_mu` and must not be mixed with current results.

1. Define $\gamma_{\mathrm{peak}}$ separately for each run as the load at
   the global maximum of native `avg_P12` (first Piola-Kirchhoff component 12).
   Post-yield means $\gamma_{i+1}>\gamma_{\mathrm{peak}}+0.01$;
   pre-yield means $\gamma_{i+1}<\gamma_{\mathrm{peak}}-10^{-4}$.
   Omit the intervening buffer. Select post-yield transitions strictly below
   the available final load, as in the existing flowchart. Use the shared
   `yield_regimes`/`yield_region_mask` helpers in `Plotting.standardPowerlaw`.
   Missing/non-finite P12 raises an error; no Cauchy-stress or fixed-strain
   fallback is allowed. P12 only selects the window: the energy prediction
   still uses starting-step `avg_sigma12` and fixed $\mu_0$.
2. Extract $\Delta E_R$ and $\Delta E_S$ as paired values from the same
   event transitions, in the same order and with the same length.
3. Compute $\kappa=\Delta E_R/(\rho V_0\Delta\gamma^2)$ with
   $\rho=N/V_0=2$ and classify with the fixed detector
   $\kappa_{\mathrm{det}}=\mu/(2\rho)$.
4. Transfer that event classification to the paired $\Delta E_S$ values and
   keep only finite positive $\Delta E_S$ values from irreversible events.
5. Evaluate the KS distance at every observed candidate in that irreversible
   $\Delta E_S$ population and use the true global minimum
   $\Delta E_{S,\min}^{KS}$.  A coarse/local search is an approximation and
   should be selected deliberately.
6. With $\Delta E_{S,\min}^{KS}$ fixed, fit $\alpha$ and $\lambda$
   by maximum likelihood.
7. Evaluate the reported goodness-of-fit p-value with the semiparametric
   Clauset bootstrap, repeating the same global $x_{\min}$ search for every
   bootstrap sample.  A fixed-$x_{\min}$ bootstrap p-value is not valid for
   the selected fit and is not permitted on this path.  A 100-point coarse
   scan plus local refinement is allowed only as an explicitly labeled
   bootstrap approximation; it must not select the empirical $x_{\min}$.
   Compare that approximation against the full-global bootstrap using
   identical samples and report the $x_{\min}$ match rate and the deviations
   in $x_{\min}$ and KS distance.

The event-pairing contract is represented by ``EventDrops`` in
``Plotting/standardPowerlaw.py``.  Do not independently sort, filter, or
concatenate the two energy-drop arrays before the event-level $\kappa$
classification.  The fitting and plotting entry points default to the full
observed-candidate global xmin search; `simpleDrop`, strided searches, and
other alternatives must be requested explicitly.

For a single size, **reuse the existing flowchart panels first** instead of
writing another size-scaling PDF, reversible/irreversible split, or fit plot.
Call `Plotting.truncated_powerlaw_reversibility_flowchart.generate_single_size_panels`
or run, for example:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache PYTHONPATH=. \
  ./.venv/bin/python -m Plotting.truncated_powerlaw_reversibility_flowchart \
  --single-size 200 --data-root /path/to/MTS2D_output --skip-pvalue
```

This produces the established classification, paired split, KS scan, and
panel-(f) fit figures. `--skip-pvalue` avoids only the bootstrap; the empirical KS
cutoff and fitted PDF remain. The automatic post-yield start is each run's
`avg_P12` peak +0.01. The reconnecting size-scaling batch wrapper
`tmp_generate_reconnecting_size_scaling_powerlaw_pdfs.py` now only discovers
runs and calls this entry point. Multi-size comparison/collapse plots remain
separate.

Size-scaling analyses and plots use L=100, 150, 200, 250 by default. Historical
campaign inventories can retain smaller runs for lookup, but those runs are
not analysis inputs. Save each plot as an individual PDF: raw distributions,
cutoff scans, collapse curves, collapse-quality maps, and parameter curves
must not share a panel grid unless explicitly requested.

For provisional second-order size-scaling plots from cached empirical fits:

`Plotting.sizeScalingCollapse` defaults to the standard positive irreversible
`Delta E_S` population: classify paired events with `kappa_det` before filtering
`Delta E_S`, then use the exhaustive global KS cutoff. Its collapse shows
separately normalized tail PDFs above each size's cutoff, with both axes
rescaled by system size. Histogram bins come from the same `getHist` helper
as flowchart panel (f), using five bins per decade by default.
For collapse tails, bin the full population first and omit any bin straddling
the cutoff; trimming before binning can artificially depress that first bin.
The explicit `--population-mode all` is an alternative analysis and writes to
a separate `all_events` directory; its results cannot replace the standard fit.

Use all locally available size-scaling samples that extend past their own `avg_P12` peak +0.01,
including runs that stop early. Pool events over each sample's available
post-yield window; neither require completion to gamma=1 nor enforce equal
sample counts between sizes. `--seeds-per-size N` is an explicit optional cap.
Output summaries record the exact source CSVs and sample counts.
`sizeScalingERClassification --split-only` redraws the raw reversible and
irreversible distributions without repeating fits or bootstraps.

To redraw these distributions, collapse and provisional parameters first,
then run second-order uncertainties with the same complete sample selection:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache PYTHONPATH=. \
  ./.venv/bin/python -u regenerate_gamma_c_powerlaws.py --size-scaling-redraw
```

Append `--skip-pvalue` to run only the splits, collapse, and parameter plots
without uncertainties; the uncertainty stage is then omitted entirely.

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache PYTHONPATH=. \
  ./.venv/bin/python -m Plotting.sizeScalingCollapse \
  --output Plots/powerLaw/size_collapse_piola_peak_post_only \
  --standard-cache-root Plots/reversible_event_analysis/size_scaling_E_R_classification_piola_peak_post_only/cache \
  --yield-regime post --largest-first --protocols second_order \
  --parameter-xmin-methods global_min --skip-pvalue
```

The parameter plots without uncertainties are saved under
`xmin_global/parameters_per_size/without_uncertainties`; the completed
bootstrap results use `xmin_global/parameters_per_size`.

The redraw runner retains its historical filename `regenerate_gamma_c_powerlaws.py`,
but now uses the first-Piola peak rule and writes to `_piola_peak` directories.
The fixed-0.7 runner was stopped; its figures and status remain historical.
Extraction hashes include the current rule, buffered windows, and cache version;
cached summaries with another yield rule are rejected. Raw gamma=0 sigma-rescue
tables are reusable only with a new native-P12 peak audit and manifest, prepared
by `Management.sigmaRescueDropSnapshot.build_piola_yield_view`.

By default, report the Clauset p-value for the selected KS cutoff and the
bootstrap mean ± one standard deviation for cutoff and alpha. These are
obtained from the same refit bootstrap; there is no need to scan p-values
across candidate cutoffs. Each bootstrap dataset has the original number of
events. The plotted original-data alpha-versus-cutoff curve is a separate
sensitivity analysis; it does not define the bootstrap alpha spread. The
size-scaling xmin, alpha, and Lambda panels show 95% percentile intervals from
the refit bootstrap; the summary also retains bootstrap means and SDs. The
percentile intervals do not assume a log-normal distribution. They assume the
fitted tail model, so interpret them cautiously if its p-value is small.
The full bootstrap can be slow for large sizes. Use `--skip-pvalue` only for
an explicitly provisional plot, and do not interpret its missing intervals
as zero uncertainty.
