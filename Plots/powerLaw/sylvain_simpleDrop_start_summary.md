# Sylvain simpleDrop start comparison

This report describes the final 18 analyses regenerated on 2026-07-31. The
fine search uses adjacent observed `xmin` values, with direct-neighbour steps
of one. The three searches start at the left, middle, and right simpleDrop
coarse-local-minimum locations. Two `xmin` values are considered the same local
minimum when their relative difference is at most `1e-6`.

## Where the data came from

All analyses use four-seed groups from the same experiment family:

`reversibilityProtocolTest`, size `100x100`, start load `0.14`, maximum load
`1.0`, `PBCt3`, `LBFGS`, and `energyDropThreshold=1e-5`. The CSV data were
loaded from the local `remoteData/macro` cache, normally located at
`/Volumes/data/remoteData/macro`. Each parameter group contains seeds `s0` to
`s3`; each is split into pre-peak and post-peak regimes by the power-law
analysis.

The 9 parameter groups are:

- batch `-2`: `LBFGSEpsx = 1e-4, 1e-5, 1e-6, 1e-7`, with `loadIncrement=1e-5`.
- batch `-1`: `loadIncrement = 1e-4, 5e-5, 1e-5, 5e-6, 1e-6`, with
  `LBFGSEpsx=1e-6`.

The batch `-2`, `LBFGSEpsx=1e-6` data and batch `-1`, `loadIncrement=1e-5`
data are identical, so the 18 output analyses contain 16 unique data/regime
series.

The accompanying overview plot is
[`sylvain_simpleDrop_quality_overview.png`](sylvain_simpleDrop_quality_overview.png)
and the exact values are in
[`sylvain_simpleDrop_quality_overview.csv`](sylvain_simpleDrop_quality_overview.csv).

## Results

- All three starts agreed in 2/18 analyses.
- Two distinct local minima were found in 10/18 analyses.
- Three distinct local minima were found in 6/18 analyses.
- Thus, at least two starts disagreed in 16/18 analyses.
- The middle start found the lowest minimum, including ties, in 10/18
  analyses; it was not lowest in 8/18.
- The selected lowest start was left in 5 analyses, middle in 5, and right in
  8.

The overview uses color for the number of distinct minima, marker shape for
pre/post peak, and a star outline when the middle start was not lowest. The
horizontal panel labels each source group and shows the number of observations
retained by simpleDrop together with the simpleDrop KS distance.

The disagreement cases are not confined to obviously small datasets: the
number of retained tail observations ranges from 3,236 to 1,489,039 across the
18 analyses, and three-minimum cases occur at 94,618, 139,601, 689,040, and
1,489,039 observations. In particular, the 94,618-observation three-minimum
case has a relatively small KS distance (`D≈0.0137`). This does not support a
simple explanation based only on too few tail observations or a poor KS fit;
the source distributions themselves should be inspected next.
