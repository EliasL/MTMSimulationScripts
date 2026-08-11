# Non-reconnecting real-space event atlas

This folder contains minimal, nominal and extreme examples from the selected
non-reconnecting setting
`epsilon_x=1e-6`, `Delta_gamma=1e-5`.  Most examples use seed 2; the new
minimal reversible-plastic example uses seed 3.  Reversibility is the pooled
setting-specific Otsu split in `Delta_rev u`; plasticity is a forward m3 change.
Only positive second-order stress-corrected energy drops are included.

The available saved event data contain reversible-plastic and
irreversible-plastic events.  No positive `Delta E_S` event without a forward
m3 change was found, so reversible-elastic and irreversible-elastic events are
not placed on this positive-drop scatter.  Two targeted elastic five-state
sets were nevertheless replayed and retained under
`output/targeted_replay_sub1e8`; their manifest records the selected loads.

The seed-3 minimal event has `Delta E_S/V_0 = 8.31e-11` in the original macro
data.  Its replay reproduces the target load, total energy, forward m3 change,
and reversible population.  The current executable reports a shifted
`avg_sigma12` diagnostic, so the plotted corrected-drop value is deliberately
taken from the original macro row rather than recomputed from the replay CSV.

`selection.csv` records the chosen events, `atlas_manifest.csv` records the
generated PDFs, and `acquisition_manifest.json` records the targeted remote
state-file requests.

For each sheet, the full-mesh panel is colored by the same `Delta E^(e)` field
as the lower energy panel, with each element boundary drawn in that element's
face color.  The two lower panels use a shared 20-unit-wide viewport centered
on the strongest energy-difference density after a 10-by-10-unit convolution
selection.
