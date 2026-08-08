"""Small orchestration helpers for catalogue, selection and image rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .acquisition import state_paths_from_directory
from .catalog import (
    build_catalog_from_job,
    build_standard_scatter_catalog,
    select_representatives,
    write_catalog,
)
from .models import RepresentativeKind, RenderOptions
from .render import event_output_name, render_event_pdf


def render_standard_downloaded_event(
    *,
    job_name: str,
    event_load: float,
    state_directory: Path,
    output_path: Path,
    batch: int = -2,
    setting: float = 1e-6,
    figure_size: tuple[float, float] = (7.4, 5.2),
) -> pd.Series:
    """Render one downloaded event against the standard scatter population.

    The event row and the background scatter are both taken from the same
    ``plot_epsilon_scatter`` data path.  Matching by job name and target load
    prevents a local event directory from silently being paired with a row
    from another seed or setting.
    """

    scatter_catalog = build_standard_scatter_catalog(batch=batch, setting=setting)
    matches = scatter_catalog[
        scatter_catalog["job_name"].eq(job_name)
        & np.isclose(scatter_catalog["load"], event_load)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one standard-scatter row for {job_name!r} at load "
            f"{event_load:.12g}; found {len(matches)}."
        )
    event = matches.iloc[0].copy()
    if event["population"] != "closing" or event["forward_m3_changes"] <= 0:
        raise ValueError(
            "The selected event is not a reversible-plastic event under the "
            "standard Otsu/m3 definitions."
        )
    event["reconnection_mode"] = "none"
    event["reversibility_measured"] = True
    event["saved_event_directory"] = str(state_directory)
    render_event_pdf(
        event,
        state_paths_from_directory(Path(state_directory)),
        Path(output_path),
        RenderOptions(output_root=Path(output_path).parent, figure_size=figure_size),
        setting_catalog=scatter_catalog,
    )
    return event


def render_local_job_examples(
    job_directory: Path,
    *,
    output_root: Path = Path("Plots/real_space_events/prototype"),
    examples_per_strategy: int = 1,
) -> pd.DataFrame:
    """Render a few available saved events from one local complete job."""

    catalog = build_catalog_from_job(Path(job_directory))
    output_root = Path(output_root)
    write_catalog(catalog, output_root / "catalog.csv")
    selected = select_representatives(
        catalog,
        examples_per_strategy=examples_per_strategy,
        strategies=(
            RepresentativeKind.TYPICAL,
            RepresentativeKind.LARGE_INTERSTRAIN_DROP,
            RepresentativeKind.HIGH_PARTICIPATION,
        ),
    )
    selected.to_csv(output_root / "selected_events.csv", index=False)
    rendered_rows = []
    for _, event in selected.iterrows():
        saved_directory = str(event["saved_event_directory"])
        if not saved_directory:
            continue
        event_directory = Path(saved_directory)
        if not event_directory.is_dir():
            continue
        paths = state_paths_from_directory(event_directory)
        output_path = output_root / event_output_name(event)
        render_event_pdf(
            event,
            paths,
            output_path,
            RenderOptions(output_root=output_root),
            setting_catalog=catalog,
        )
        rendered_rows.append({"event_id": event["event_id"], "output": str(output_path)})
    pd.DataFrame(rendered_rows).to_csv(output_root / "rendered_events.csv", index=False)
    if not rendered_rows:
        raise RuntimeError("No selected events had complete local VTU state sets.")
    return selected
