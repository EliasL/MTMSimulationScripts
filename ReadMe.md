These scripts are tools to manage the Mesoscopic Tensorial Simulation (MTS)

## Standard power-law energy-drop workflow

When a request says “power-law fit”, “power-law plot”, or asks for the
standard energy-drop result without specifying another population, use this
recipe:

The second-order correction in $\Delta E_S$ uses the fixed reference shear
tangent $\mu_0=a_{1212}(0)$. Outputs made with $a_{1212}(\gamma)$ are archived
as `old_dynamic_mu` and must not be mixed with current results.

1. Restrict the data to the post-yield region.
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
stress-peak load plus 0.01. The reconnecting size-scaling batch wrapper
`tmp_generate_reconnecting_size_scaling_powerlaw_pdfs.py` now only discovers
runs and calls this entry point. Multi-size comparison/collapse plots remain
separate.

By default, report the Clauset p-value for the selected KS cutoff and 95%
bootstrap intervals for cutoff and alpha. These are obtained from the same
refit bootstrap; there is no need to scan p-values across candidate cutoffs.
Each bootstrap dataset has the original number of events. The 95% interval
uses the central 95% of the refitted estimates, not 95% of the original events.
The plotted original-data alpha-versus-cutoff curve is a separate sensitivity
analysis; its values at the cutoff interval endpoints do not define the
bootstrap alpha interval. These intervals assume the fitted tail model, so
interpret them cautiously if its p-value is small.
The full bootstrap can be slow for large sizes. Use `--skip-pvalue` only for
an explicitly provisional plot, and do not interpret its missing intervals
as zero uncertainty.
