# Benchmark initial conditions

This directory is a catalog of native MTS2D simulation dumps used to replay
history-dependent minimization problems. The directory name is part of the
benchmark contract:

```text
initialConditions/
├── noReconnecting/
│   ├── load_0.15/
│   │   └── size_128x128/
│   │       ├── seed_0.xml.gz
│   │       └── seed_0.metadata.json
│   └── load_0.7/
│       └── size_128x128/
│           └── seed_0.xml.gz
└── edgeFlipping/
    ├── load_0.15/
    └── load_0.7/
```

- `noReconnecting` dumps must come from simulations whose reconnection method
  was `none`.
- `edgeFlipping` dumps must come from simulations whose reconnection method was
  `edgeFlip`. Delaunay states are not accepted.
- `load_*` is a nominal load bin for the relaxed state stored in the dump.
- `size_<rows>x<cols>` is checked against the mesh stored in the dump.
- `seed_<seed>.xml.gz` is the native compressed XML dump. Seeds start at zero.

## Replay semantics

A catalog dump represents a relaxed, post-step state. A replay benchmark loads
that state without creating normal simulation output, reads `loadIncrement`
from the serialized config, applies one ordinary affine loading step, and then
times the following minimization. Thus the timed target load is normally the
catalog load plus one `loadIncrement`.

For a known avalanche, retain the relaxed dump from immediately before the
step that triggers it. A post-event dump cannot reproduce the event that has
already happened.

## Adding existing dumps

Copy or download a native `.xml.gz` simulation dump into the nearest matching
nominal load bin. For example:

```text
initialConditions/edgeFlipping/load_0.7/size_128x128/seed_0.xml.gz
```

The native replay validates the mesh dimensions, seed, stored load, and
`reconnectionMethod`. By default the actual stored load may differ from its
nominal folder by at most 0.005; this accommodates production dump names rounded
to two decimals, such as `dump_l0.70.xml.gz` storing load 0.70251. The exact load
is retained in raw results. Change the limit with `--replay-load-tolerance`.
An optional `seed_0.metadata.json` may record the source simulation, original
path, capture time, and file hash, but the dump itself remains the source of
truth. The catalog root can also live outside the repository and be selected
with `--initial-conditions-dir`, which is useful for large files on a compute
node.

## Generation and missing data

The benchmark may generate a missing `load_0.15` dump by running the normal
first noisy minimization at `startLoad=0.15`. Generation is atomic and never
replaces an existing dump. Its preparation time is recorded separately from
the replay timings. The defaults are seed 0, initial-guess noise 0.05,
quenched-disorder SD 0.0, and load increment 1e-5; the preparation metadata
records the actual values used.

The benchmark never fabricates a `load_0.7` state. Missing replay problems are
reported as skipped cases; there is no affine jump fallback. A present dump
whose stored size, load, or reconnection method contradicts its catalog path is
rejected as a failed sample so a mislabeled scientific state cannot pass
silently.

Runtime-generated dumps and metadata are ignored by Git because large initial
conditions belong in external storage. They can still be uploaded to compute
nodes by the project synchronization scripts.
