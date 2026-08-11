# Real-space event visualization scaffold

This package produces PNG real-space views by default of selected reversible and
irreversible events.  PDF output remains available when explicitly requested. It initially targets the chosen numerical setting
`LBFGSEpsx=1e-6`, `loadIncrement=1e-5`, but the catalogue supports both
non-reconnecting and reconnecting data.

## Scientific definitions

Reversibility and plasticity are independent:

- closing/non-closing is determined by the setting-specific unbinned log-Otsu
  cut in `rev_u_diff`;
- plasticity means a forward-step m3 change, even if the backward step undoes
  it; and
- `is_reversible` and `rev_drop`/`irrev_drop` folder names are never used for
  classification.

The C++ protocol currently returns early when the forward step does not have
both an m3 event and an affine-to-relaxed energy drop.  Such rows did not
receive a backward test.  They must be labelled `reversibility_unmeasured`, not
reversible elastic, until one or two selected examples have been replayed.

## Five states

The expected VTUs are:

1. `state0_min_gamma`: relaxed state at gamma;
2. `state1_affine_gamma_plus`: affine state at gamma + Delta gamma;
3. `state2_relaxed_gamma_plus`: forward-relaxed state;
4. `state3_affine_gamma_minus`: affine return to gamma; and
5. `state4_relaxed_gamma`: backward-relaxed state.

The arrow fields are `x2-x1` for forward relaxation, `x4-x3` for backward
relaxation, and `x4-x0` for closure.  Remove only mean translation after
periodic-image alignment; do not remove a best-fit affine field from `x2-x1`,
because state 1 already supplies that affine reference.

## Figure policy

- Output PNG by default; PDF can be requested explicitly, and animations are
  not generated.
- Use a symmetric `coolwarm` color scale centered at zero for energy changes.
- Show both `E(state0)-E(state2)` and `E(state2)-E(state4)` with the same
  symmetric `coolwarm` limits, each with a full-mesh locator and a zoomed
  panel.
- Convey displacement magnitude with arrow length.
- Include a boxed in-panel quiver key using `|->| = ...` notation; the key
  arrow is the amplified display length corresponding to the stated physical
  displacement magnitude.
- Outline forward m3-changed elements without using them to choose
  reversibility.
- Show load and pre/post-yield status, but do not duplicate every selection into
  separate pre/post groups.  The CLI may filter to `all`, `pre`, or `post`.
- Select the zoom from a 10-by-10-unit smoothed density of the absolute local
  energy difference, with a displayed width capped at 20 units.  The density
  is only a selection device and is not plotted.

## Reconnecting meshes

Node displacements should be matched by `refIndex`.  Cell arrays may be
subtracted directly only when connectivity is identical.  When reconnection
changes topology, project both cell-energy fields to one common periodic grid
before subtraction.  Never subtract reconnecting cell arrays by storage index.

## Data acquisition policy

Full simulation folders must not be downloaded.  The workflow is:

1. build and inspect the catalogue locally from cached macro CSVs;
2. select typical, large-inter-strain-drop, and high-participation examples;
3. resolve exact remote event directories and five state filenames;
4. write an acquisition manifest;
5. download only those individual VTUs to `remoteData/real_space_events`; and
6. replay at most one or two unmeasured no-m3 examples from an earlier dump.

Saved event folders are biased because the original code stores every 300th
reversible event and every 10th irreversible event using its old fixed
threshold.  Use them for available examples, but use macro-data selection and
targeted replay whenever an unbiased representative is required.

## Suggested implementation order

1. Implement and test `catalog.classify_event`, catalogue construction, and
   representative selection using existing numerical-parameter helpers.
2. Implement remote discovery and manifest writing without downloading.
3. Implement strict five-state loading in `meshEventPlotting.py` for one
   non-reconnecting event.
4. Implement periodic alignment, displacement fields, zooming, and one event
   image.
5. Add common-grid energy projection for reconnecting topology.
6. Implement targeted downloads after the manifest has been reviewed.
7. Implement the one-off replay job and validate replayed macro quantities
   against the original row before accepting its VTUs.
8. Add the shared-scale four-class comparison image.

Every unexpected field shape, duplicate remote match, incomplete state set,
invalid node correspondence, or unsupported topology must raise an error.
