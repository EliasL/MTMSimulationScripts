# Simulation configuration reference

This reference describes the `key = value` entries accepted by the current
MTS2D simulation source and the additional options emitted or handled by this
repository's Python runner.  The C++ parser accepts `true`/`false` and `1`/`0`
for Boolean values, strips whitespace, and treats `#` as a comment delimiter.

`Management/configGenerator.py` supplies convenient Python defaults, while
MTS2D has its own defaults when a key is absent, so generated configuration
files should normally include every setting needed for a reproducible run.

## Simulation and mesh settings

- `name` — Names the simulation and therefore its output directory and files.
- `rows` — Sets the number of lattice-node rows in the mesh.
- `cols` — Sets the number of lattice-node columns in the mesh.
- `usingPBC` — Selects periodic boundary conditions when `true` and a finite mesh when `false`.
- `reconnectionMethod` — Selects topology handling: `none` disables it, `edgeFlip` performs energy-based edge flips, and `delaunay` performs Delaunay reconnection.
- `reconnectRevert` — When `true`, restores the best topology and stops the reconnection cycle as soon as a reconnection does not lower the minimized energy.
- `reconnectEdgeLocking` — When `true`, prevents an edge already considered in an `edgeFlip` reconnection cycle from being reused.
- `experiment` — Selects the loading protocol, as described in the [experiment reference](#experiment-reference).
- `nrThreads` — Sets the OpenMP thread count, with `0` requesting the runtime's suggested count and oversized values reduced to the available capacity.
- `seed` — Sets the random seed used to make disorder and initial-guess noise reproducible.
- `QDSD` — Sets the standard deviation of the quenched per-element disorder used when the mesh is created.
- `initialGuessNoise` — Sets the amplitude of random displacement noise added before the standard initial stabilization minimization.
- `meshDiagonal` — Chooses the initial square-cell triangulation as `major`, `minor`, or `alternate`.
- `energyFunction` — Selects the element energy model, currently `contiSquare` or `contiTriangular`.
- `bulkModulus` — Sets the bulk-modulus parameter supplied to every triangular element.

## Loading and experiment parameters

- `startLoad` — Sets the load value assigned before the first loading step.
- `loadIncrement` — Sets the signed simple-shear or prescribed-displacement increment applied at each step.
- `maxLoad` — Sets the load limit at which a standard monotonic experiment stops.
- `GP1` — Provides experiment-specific parameter 1, most notably the reference shear in reference-state tests and the boundary choice in `doubleDislocationTest`.
- `GP2` — Provides experiment-specific parameter 2, most notably the loading order in `doubleDislocationTest` and the centre-node offset in `reconnectSSTest`.
- `GP3` — Provides experiment-specific parameter 3, most notably the direction-switch interval in `doubleDislocationTest` and the vertical-shear flag in `reconnectSSTest`.

## Minimizer selection and shared stopping criterion

- `minimizer` — Selects `LBFGS`, `CG`, or `FIRE` for mechanical relaxation.
- `epsR` — Sets the maximum residual force that requests early termination of the LBFGS and CG minimizers, while the current FIRE implementation receives but does not consume this value.

### L-BFGS settings

- `LBFGSNrCorrections` — Sets the number of correction-vector pairs retained by the limited-memory BFGS solver.
- `LBFGSScale` — Is retained in the configuration format but is not passed to the current L-BFGS implementation and therefore has no runtime effect.
- `LBFGSEpsg` — Sets the L-BFGS gradient-norm convergence tolerance.
- `LBFGSEpsf` — Sets the L-BFGS relative objective-improvement convergence tolerance.
- `LBFGSEpsx` — Sets the L-BFGS step-size convergence tolerance.
- `LBFGSMaxIterations` — Caps L-BFGS iterations when positive, with `0` leaving the iteration count uncapped.

### Conjugate-gradient settings

- `CGScale` — Is retained in the configuration format but is not passed to the current conjugate-gradient implementation and therefore has no runtime effect.
- `CGEpsg` — Sets the conjugate-gradient gradient-norm convergence tolerance.
- `CGEpsf` — Sets the conjugate-gradient relative objective-improvement convergence tolerance.
- `CGEpsx` — Sets the conjugate-gradient step-size convergence tolerance.
- `CGMaxIterations` — Caps conjugate-gradient iterations when positive, with `0` leaving the iteration count uncapped.

### FIRE settings

- `finc` — Sets the factor by which FIRE increases its time step after sustained downhill motion.
- `fdec` — Sets the factor by which FIRE decreases its time step after an uphill step.
- `alphaStart` — Sets FIRE's initial velocity–force mixing coefficient.
- `falpha` — Sets the factor by which FIRE reduces its mixing coefficient during downhill motion.
- `dtStart` — Sets FIRE's initial time step.
- `dtMax` — Sets FIRE's largest permitted time step.
- `dtMin` — Sets FIRE's smallest permitted time step after an unstable step.
- `maxCompS` — Is passed as FIRE's maximum component-step setting, but the corresponding constraint is disabled in the current FIRE implementation.
- `eps` — Sets FIRE's absolute gradient-norm convergence tolerance.
- `epsRel` — Sets FIRE's relative gradient-norm convergence tolerance, scaled by the displacement-vector norm.
- `delta` — Is passed as FIRE's objective-stagnation tolerance, but has no effect because the current integration leaves FIRE's comparison history disabled.
- `maxIt` — Caps FIRE iterations when positive, with `0` leaving the iteration count uncapped.

## Output and logging settings

- `logDuringMinimization` — Enables per-minimization CSV and VTU diagnostics, retaining detailed folders only for sufficiently plastic, sufficiently energetic, forced, or periodic steps.
- `fullMinimizationLogging` — Selects all available VTU fields instead of the reduced field set when minimization logging is enabled.
- `writeDumps` — Enables serialized restart dumps at the midpoint, approximately hourly, final, forced, and explicitly targeted saves.
- `nrVTUFrames` — Sets the minimum requested number of regular mesh-output intervals across the configured load range and must be positive.
- `plasticityEventThreshold` — Sets the fraction of elements with plastic changes required to force mesh output and retain detailed minimization logging.
- `energyDropThreshold` — Sets the energy-drop magnitude required to force mesh output and retain detailed minimization logging.
- `showProgress` — Uses `-1` to suppress normal console progress, while other values are treated as normal display by the current source.

## Python-runner and compatibility options

- `makeDumpAt` — Requests a dump near the specified load through the Python runner's `--makeDumpAt` command-line flag rather than through a line written to the configuration file.
- `writeDebugVTUs` — Is still emitted by `SimulationConfig`, but the current MTS2D parser does not read it and it has no runtime effect.
- `scenario` — Is a legacy alias for `experiment` and should be replaced with `experiment` in new files.
- `FullMinimizationLogging` — Is a legacy capitalized alias for `fullMinimizationLogging` and should be replaced with the lowercase spelling in new files.
- `forceReRun` — Has no effect as a configuration-file line in the current executable, so reruns must be requested with the executable's `-r` command-line option.
- `writeMeshVTUs` — Appears in older replay configurations but is not read by the current MTS2D parser and therefore cannot disable ordinary mesh output.
- `dumpPreEventAfterReversibility` — Appears in older replay configurations but is not read by the current MTS2D parser and therefore has no runtime effect.
- `saveElasticReversibilityStates` — Appears in older replay configurations but is not read by the current MTS2D parser and therefore has no runtime effect.
- `maximumSavedElasticReversibilityStates` — Appears in older replay configurations but is not read by the current MTS2D parser and therefore has no runtime effect.
- `saveFinalReversibilityState` — Appears in older replay configurations but is not read by the current MTS2D parser and therefore has no runtime effect.

`configPath` and `forceReRun` are serialized inside restart dumps, but neither
is a normal configuration-file input setting.  Likewise, the `large`,
`singleDislocation`, and `longSim` names in `get_custom_configs()` are Python
preset names rather than experiment values accepted by the current executable.

## Experiment reference

These are the experiment values registered by MTS2D's current
`runSimulationExperiment()` dispatch table.

- `simpleShear` — Applies an affine simple-shear increment, minimizes the mesh, and records every loading step.
- `noMinimizationSS` — Affinely shears an all-free periodic mesh and performs reconnection without an energy minimization.
- `noMinimizationSSReferenceTest` — Sets a reference state sheared by `GP1` and then runs the no-minimization periodic simple-shear protocol.
- `simpleShearFixedBoundary` — Applies affine simple shear with the border nodes fixed and minimizes after every step.
- `simpleShearWithNoise` — Runs simple shear while adding a fixed `8e-7` displacement noise to each step's initial guess before minimization.
- `periodicBoundaryTest` — Compares periodic geometry with selected fixed rows and a fixed column that are sheared while the free nodes relax.
- `periodicBoundaryFixedComparisonTest` — Fixes every border node and the middle row before carrying out affine-shear relaxation for a boundary-condition comparison.
- `cyclicSimpleShear` — Reverses the shear direction as the deformation crosses approximately `0.30` and `0.16` to create a cyclic loading path.
- `createDumpBeforeEnergyDrop` — Runs the periodic-boundary comparison setup while rotating diagnostic dump checkpoints until its energy-change trigger fires.
- `doubleDislocationTest` — Loads a partly fixed mesh along horizontal and vertical displacement paths controlled by `GP1`, `GP2`, and `GP3` to study dislocation behavior and reconnection.
- `singleDislocationFixedBoundaryTest` — Loads a fixed-boundary dislocation setup and then explicitly flips middle-row element pairs while writing diagnostic VTUs.
- `reconnectTest` — Moves the centre node of a fixed `3 x 3` mesh along a scripted displacement path without minimization to test reconnection behavior.
- `reconnectSSTest` — Uses a fixed-column periodic `3 x 3` mesh, a `GP1` reference shear, and a `GP2` centre-node displacement to test edge flips under simple shear.
- `reversibilityProtocolTest` — Runs a forward step and its reversibility check with a fixed closure tolerance of `1e-4`, adding reversibility columns to the output.
- `simpleShearReferenceTest` — Sets a reference configuration sheared by `GP1` and then performs ordinary affine simple-shear relaxation.
