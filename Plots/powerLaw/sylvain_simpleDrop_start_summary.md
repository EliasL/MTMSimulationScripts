# Sylvain simpleDrop start comparison

Regenerated on 2026-07-31 for the 18 Sylvain pre/post power-law fits in batches
`-2` and `-1`. Two `xmin` values are considered the same local minimum when
their relative difference is at most `1e-6`.

- 7/18 fits found different local minima from the three starts.
- 11/18 fits found the same local minimum from all three starts.
- 14/18 fits had the middle start at the lowest KS distance (including ties).
- 4/18 fits had the middle start at a higher KS distance than the left/right
  result.
- In all 7 disagreement cases, the left and right starts agreed and only the
  middle start found a different minimum.

The disagreement cases were:

| batch | regime | parameter | left/right xmin | middle xmin | middle lowest |
| --- | --- | --- | ---: | ---: | :---: |
| -2 | pre | `LBFGSEpsx=1e-6` | `3.655e-3` | `5.382e-3` | yes |
| -2 | post | `LBFGSEpsx=1e-7` | `5.394e-3` | `7.133e-3` | no |
| -1 | post | `loadIncrement=1e-4` | `7.911e-2` | `4.772e-2` | no |
| -1 | post | `loadIncrement=5e-5` | `2.119e-2` | `1.439e-2` | yes |
| -1 | pre | `loadIncrement=1e-5` | `5.394e-3` | `7.133e-3` | no |
| -1 | pre | `loadIncrement=5e-6` | `3.655e-3` | `5.382e-3` | yes |
| -1 | post | `loadIncrement=1e-6` | `1.050e-3` | `5.651e-4` | no |

The detailed comparison is also retained in each `xmin_analysis` cache entry,
under `simple_drop_start_summary`.
