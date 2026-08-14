# Energy-drop power-law analysis

## Standard protocol

When a request asks for the standard energy-drop power-law result, use this
sequence unless the request explicitly specifies another analysis:

1. Restrict extraction to the post-yield region by default.
2. Extract `Delta E_R` and `Delta E_S` as paired values from the same event
   transitions, in the same order and with the same length.
3. Apply `simpleDrop` to the positive `Delta E_R` population. Call its
   classification threshold `er_det` and label it in plots as
   `Delta E_{R,\det}`. It is a detection/classification threshold, not an
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



