"""Select, acquire and render a small real-space event atlas."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .acquisition import (
    download_event,
    locate_remote_event_directory,
    management_sources_for_job,
    write_acquisition_manifest,
)
from .catalog import build_standard_scatter_catalog
from .models import DownloadRequest, RenderOptions
from .render import render_event_pdf


EVENT_CLASSES = ("reversible_plastic", "reversible_elastic", "irreversible_plastic", "irreversible_elastic")
_EVENT_DIRECTORY = re.compile(r"(?:rev|irrev)_drop_l_(?P<load>[0-9.eE+-]+)$")


def _nominal_row(rows: pd.DataFrame) -> pd.Series:
    """Select a robust medoid in the event-feature space."""

    features = [
        "delta_rev_u",
        "delta_E_S_over_V0",
        "participation_fraction",
        "forward_m3_changes",
    ]
    values = np.log10(
        np.maximum(rows[features].to_numpy(dtype=float), np.finfo(float).tiny)
    )
    scale = np.nanstd(values, axis=0)
    normalized = (values - np.nanmedian(values, axis=0)) / (scale + 1e-12)
    return rows.iloc[int(np.nanargmin(np.nansum(normalized**2, axis=1)))]


def select_saved_standard_atlas(
    *,
    job_name: str,
    candidate_root: Path,
    batch: int = -2,
    setting: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select nominal and extreme saved events from one standard setting."""

    catalog = build_standard_scatter_catalog(batch=batch, setting=setting)
    catalog = catalog[catalog["job_name"].eq(job_name)].copy()
    if catalog.empty:
        raise ValueError(f"No standard-scatter rows found for {job_name!r}.")

    available = []
    for directory in sorted(Path(candidate_root).iterdir()):
        match = _EVENT_DIRECTORY.fullmatch(directory.name)
        if not directory.is_dir() or match is None:
            continue
        start_load = float(match.group("load"))
        rows = catalog[np.isclose(catalog["event_start_load"], start_load)]
        if len(rows) != 1:
            continue
        row = rows.iloc[0].copy()
        row["saved_event_directory"] = str(directory)
        available.append(row)
    available = pd.DataFrame(available)
    if available.empty:
        raise RuntimeError(f"No saved standard events found in {candidate_root}.")

    selected = []
    for event_class in ("reversible_plastic", "reversible_elastic", "irreversible_plastic", "irreversible_elastic"):
        rows = available[available["event_class"].eq(event_class)]
        if rows.empty:
            continue
        nominal = _nominal_row(rows).copy()
        nominal["atlas_label"] = f"{event_class}_nominal"
        selected.append(nominal)
        minimal = rows.loc[rows["delta_E_S_over_V0"].idxmin()].copy()
        minimal["atlas_label"] = f"{event_class}_minimal"
        if str(minimal["event_id"]) not in {
            str(nominal["event_id"]),
        }:
            selected.append(minimal)
        extreme = rows.loc[rows["delta_E_S_over_V0"].idxmax()].copy()
        extreme["atlas_label"] = f"{event_class}_extreme"
        if str(extreme["event_id"]) not in {
            str(nominal["event_id"]), str(minimal["event_id"]),
        }:
            selected.append(extreme)
    if not selected:
        raise RuntimeError("No atlas classes were available.")
    return pd.DataFrame(selected).reset_index(drop=True), catalog.reset_index(drop=True)


def acquire_and_render_atlas(
    selection: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    output_root: Path,
    local_event_root: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    """Download selected state sets and render one PNG per selected event."""

    output_root = Path(output_root)
    local_event_root = Path(local_event_root)
    sources_by_job = {
        job: management_sources_for_job(job)
        for job in selection["job_name"].drop_duplicates()
    }
    requests = []
    local_directories = {}
    for _, row in selection.iterrows():
        sources = sources_by_job[str(row["job_name"])]
        if not sources:
            raise RuntimeError(f"No management source found for {row['job_name']}.")
        found = locate_remote_event_directory(row, sources)
        if found is None:
            raise RuntimeError(
                f"No remote event directory found for {row['event_id']} at load {row['load']}."
            )
        source, remote_directory = found
        label = str(row["atlas_label"])
        local_directory = local_event_root / label
        requests.append(
            DownloadRequest(
                event_id=str(row["event_id"]),
                source=source,
                remote_event_directory=remote_directory,
                local_event_directory=local_directory,
            )
        )
        local_directories[str(row["event_id"])] = local_directory
    write_acquisition_manifest(requests, [], Path(manifest_path))

    rendered = []
    for request, (_, row) in zip(requests, selection.iterrows()):
        paths = download_event(request)
        event = row.copy()
        event["reconnection_mode"] = "none"
        event["reversibility_measured"] = True
        event["saved_event_directory"] = str(request.local_event_directory)
        output_path = output_root / f"{event['atlas_label']}.png"
        render_event_pdf(
            event,
            paths,
            output_path,
            RenderOptions(output_root=output_root),
            setting_catalog=catalog,
        )
        rendered.append({
            "atlas_label": event["atlas_label"],
            "event_class": event["event_class"],
            "job_name": event["job_name"],
            "load": event["load"],
            "delta_E_S_over_V0": event["delta_E_S_over_V0"],
            "delta_rev_u": event["delta_rev_u"],
            "forward_m3_changes": event["forward_m3_changes"],
            "output": str(output_path),
        })
    result = pd.DataFrame(rendered)
    result.to_csv(output_root / "atlas_manifest.csv", index=False)
    return result
