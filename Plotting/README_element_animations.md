# Element animations and historical deformation comparisons

Run from the SimulationScripts root using its virtual environment:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache PYTHONPATH=. \
  .venv/bin/python -m Plotting.animate_element \
  --data-dir /path/to/simulation --element 389 --box-size 40
```

`--output-directory` defaults to `output/animations`. The driver uses the
existing `element_tracking_animation` renderers for the Poincare disk, local
transparent neighbourhood, periodic overview and final composition. It does
not substitute the standalone standard mesh movie. Rendering is sequential.
Omit `--data-dir` to select the original cell-389 detailed replay; this requires
the original main and minimization VTUs at the paths defined in the driver.

The default tracks total deformation `T = F_E @ T_p`. The explicit
`--matrix T_p` option reproduces the legacy driver, which followed plastic
history and used total deformation for the camera. Legacy `T..` fields in VTUs
store `T_p`. Cached matrices retain those raw fields and validate source file
sizes, modification times, paths, and element index before reuse.
Use distinct `--output-prefix` values when retaining both scientific variants.
The migrated movies and caches are historical outputs and are not regenerated
or relabeled by this move.

For mesh component movies, reuse `plotAll.py` and
`Plotting.makeAnimations.makeAnimations` with `matrix_name` and `videoNames`
(e.g. `matrix_name="T", videoNames=["T_component_grid"]`). This already uses
`pyplotFunctions.plot_matrix_component_grid`; there is no second grid renderer.

The two former `compare_ftotal_*_vtu.py` scripts are replaced by:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache PYTHONPATH=. \
  .venv/bin/python -m Plotting.compare_deformation_means \
  /path/to/data output/animations/deformation_means.png --comparison both
```

`--comparison T` (default) compares Ftotal with total `F_E @ T_p`;
`--comparison T_p` compares it with the raw plastic history. Outputs are PNG,
PDF and CSV. These comparisons require archived VTUs containing `Ftotal..`;
missing fields raise errors rather than substituting another deformation.
