"""Render irreversible events immediately below and above the global xmin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter

from Management.connectToCluster import Servers, getServerUserName
from Plotting.findXmin import analyze_xmin
from Plotting.plotPowerLaw import (
    Truncated_Power_Law,
    dist_from_fit,
    getHist,
    make_fit,
    plot_data_pdf,
    plot_fit_pdf,
)

from .acquisition import (
    download_event,
    locate_remote_event_directory,
    management_sources_for_job,
    state_paths_from_directory,
)
from .catalog import build_standard_scatter_catalog
from .models import DownloadRequest, RemoteSource, RenderOptions
from .render import render_event_pdf


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output/pdf/post_yield_irreversible_xmin_event_atlas"
DEFAULT_BATCH = -2
DEFAULT_SETTING = 1e-6
DEFAULT_YIELD_REGIME = "post"
EVENTS_PER_SIDE = 10
MIN_TAIL_COUNT = 100
NR_INITIAL = 100
SEARCH_MAX_XMIN = 1e-4
PREFERRED_MAX_ABOVE_FACTOR = 10.0
MAX_ABOVE_FACTOR = 100.0


def _standard_sources(job_name: str) -> tuple[RemoteSource, ...]:
    """Resolve standard-job locations, including jobs absent from data.json."""

    indexed = management_sources_for_job(job_name)
    if indexed:
        return indexed

    # The standard jobs are on the active Pascal storage, but are not present
    # in the management index.  Try it first with a bounded SSH handshake;
    # this avoids a long series of unbounded DNS/SSH probes for stale hosts.
    sources = []
    preferred_servers = (Servers.pascal,) + tuple(
        server for server in Servers.search_servers if server != Servers.pascal
    )
    for server in preferred_servers:
        ssh_host = server.split(".", 1)[0]
        username = getServerUserName(server)
        candidate_roots = [
            Path(base) / username / "MTS2D_output" for base in ("/data", "/data2")
        ]
        event_roots = [
            root / job_name / "data" / "reversibilityData"
            for root in candidate_roots
        ]
        test_script = "for p in " + " ".join(
            f"{str(path)!r}" for path in event_roots
        ) + "; do if [ -d \"$p\" ]; then printf '%s\\n' \"$p\"; fi; done"
        result = subprocess.run(
            ["ssh", "-T", "-o", "ConnectTimeout=5", ssh_host, test_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            for root in candidate_roots:
                if str(root / job_name / "data" / "reversibilityData") in result.stdout.decode().splitlines():
                    sources.append(RemoteSource(host=ssh_host, data_root=root))
            if sources:
                return tuple(sources)
    if not sources:
        raise RuntimeError(f"Could not locate standard job {job_name!r} on any source.")
    return tuple(sources)


def _fit_default_irreversible(catalog: pd.DataFrame, yield_regime: str = "all"):
    if yield_regime not in {"all", "pre", "post"}:
        raise ValueError(f"Unknown yield regime: {yield_regime!r}")
    irreversible = catalog[catalog["population"].eq("nonclosing")].copy()
    if yield_regime != "all":
        irreversible = irreversible[irreversible["yield_regime"].eq(yield_regime)].copy()
    if irreversible.empty:
        raise RuntimeError(
            f"The default scatter catalogue has no irreversible {yield_regime}-yield events."
        )
    values = irreversible["delta_E_S_over_V0"].to_numpy(dtype=float)
    analysis = analyze_xmin(
        values,
        nr_initial=NR_INITIAL,
        min_tail_count=MIN_TAIL_COUNT,
        distType=Truncated_Power_Law,
        max_xmin=SEARCH_MAX_XMIN,
        refine=False,
    )
    xmin = float(analysis["global_min_xmin"])
    if not np.isfinite(xmin) or xmin <= 0:
        raise RuntimeError(f"Invalid global-minimum xmin: {xmin}")
    fit = make_fit(
        values,
        xmin_range=xmin,
        distType=Truncated_Power_Law,
        use_cache=False,
    )
    fit.xmin_analysis = analysis
    fit.xmin_fitting_results = analysis
    return irreversible, values, analysis, fit


def _remote_event_directories(source: RemoteSource, job_name: str) -> set[str]:
    """List one job's saved event directories with one bounded SSH call."""

    root = source.data_root / job_name / "data" / "reversibilityData"
    if source.host in {"local", "localhost"}:
        return {path.name for path in root.iterdir() if path.is_dir()}
    result = subprocess.run(
        [
            "ssh", "-T", "-o", "ConnectTimeout=5", source.host,
            "find", str(root), "-maxdepth", "1", "-mindepth", "1",
            "-type", "d", "-print",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not list saved event directories for {job_name!r} on {source.host}."
        )
    return {
        Path(line.strip()).name
        for line in result.stdout.splitlines()
        if line.strip()
    }


def _discover_local_replayed_rows(
    catalog: pd.DataFrame, replay_root: Path | None
) -> pd.DataFrame:
    """Return catalogue rows whose completed private replays include five states."""

    if replay_root is None:
        return catalog.iloc[0:0].copy()
    manifests = sorted(Path(replay_root).glob("replay_*/replay_manifest.csv"))
    rows = []
    seen_event_ids = set()
    for manifest in manifests:
        recorded = pd.read_csv(manifest)
        required = {"event_kind", "target_load", "event_directory"}
        missing = required.difference(recorded.columns)
        if missing:
            raise KeyError(f"Replay manifest {manifest} is missing columns: {sorted(missing)}")
        plastic = recorded[recorded["event_kind"].eq("plastic")]
        if len(plastic) != 1:
            raise RuntimeError(f"Expected one plastic event in replay manifest {manifest}.")
        target = plastic.iloc[0]
        event_directory = Path(str(target["event_directory"]))
        state_paths_from_directory(event_directory)
        target_load = float(target["target_load"])
        matches = catalog[
            catalog["load"].sub(target_load).abs() <= 1e-10
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one catalogue event at replay target {target_load:.12g}; "
                f"found {len(matches)}."
            )
        selected = matches.iloc[0].copy()
        event_id = str(selected["event_id"])
        if event_id in seen_event_ids:
            raise RuntimeError(f"Duplicate replayed event {event_id!r}.")
        seen_event_ids.add(event_id)
        selected["local_event_directory"] = str(event_directory)
        rows.append(selected)
    return pd.DataFrame(rows) if rows else catalog.iloc[0:0].copy()


def _select_available_side(
    rows: pd.DataFrame,
    xmin: float,
    side: str,
    count: int,
    sources_by_job: dict[str, tuple[RemoteSource, ...]],
    local_rows: pd.DataFrame | None = None,
    *,
    max_above_factor: float | None = None,
    strict: bool = True,
) -> list[pd.Series]:
    if local_rows is not None and not local_rows.empty:
        remote_rows = rows[~rows["event_id"].isin(local_rows["event_id"])]
        rows = pd.concat([remote_rows, local_rows], ignore_index=True)
    if side == "below":
        candidates = rows[rows["delta_E_S_over_V0"] < xmin].sort_values(
            "delta_E_S_over_V0", ascending=False
        )
    elif side == "above":
        candidates = rows[rows["delta_E_S_over_V0"] >= xmin].sort_values(
            "delta_E_S_over_V0", ascending=True
        )
        if max_above_factor is not None:
            if max_above_factor < 1:
                raise ValueError("max_above_factor must be at least one.")
            candidates = candidates[
                candidates["delta_E_S_over_V0"] <= max_above_factor * xmin
            ]
    else:
        raise ValueError(f"Unknown xmin side: {side!r}")

    selected = []
    directory_cache: dict[tuple[str, str, str], set[str]] = {}
    for _, row in candidates.iterrows():
        local_event_directory = row.get("local_event_directory", "")
        if isinstance(local_event_directory, str) and local_event_directory:
            chosen = row.copy()
            chosen["xmin_side"] = side
            selected.append(chosen)
            if len(selected) == count:
                break
            continue
        job_name = str(row["job_name"])
        if job_name not in sources_by_job:
            sources_by_job[job_name] = _standard_sources(job_name)
        for source in sources_by_job[job_name]:
            cache_key = (source.host, str(source.data_root), job_name)
            if cache_key not in directory_cache:
                directory_cache[cache_key] = _remote_event_directories(source, job_name)
            names = directory_cache[cache_key]
            start_load = float(row["event_start_load"])
            expected = {
                f"rev_drop_l_{start_load:.5f}",
                f"irrev_drop_l_{start_load:.5f}",
            }
            matches = sorted(names & expected)
            if len(matches) > 1:
                raise RuntimeError(
                    f"Both reversible and irreversible saved directories exist for "
                    f"{row['event_id']}: {matches}"
                )
            if not matches:
                continue
            remote_directory = source.data_root / job_name / "data" / "reversibilityData" / matches[0]
            break
        else:
            continue
        chosen = row.copy()
        chosen["xmin_side"] = side
        chosen["remote_source"] = source.host
        chosen["remote_data_root"] = str(source.data_root)
        chosen["remote_event_directory"] = str(remote_directory)
        selected.append(chosen)
        if len(selected) == count:
            break

    if strict and len(selected) != count:
        available = int(len(candidates))
        raise RuntimeError(
            f"Only {len(selected)} complete-looking {side} candidates were located; "
            f"needed {count} from {available} catalogue candidates."
        )
    return selected


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _pdf_marker_y(values: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    bins, density = getHist(values)
    valid = np.isfinite(bins) & np.isfinite(density) & (bins > 0) & (density > 0)
    if valid.sum() < 2:
        raise RuntimeError("Could not construct a positive empirical PDF for markers.")
    return 10 ** np.interp(
        np.log10(x_values),
        np.log10(bins[valid]),
        np.log10(density[valid]),
    )


def _render_fit_pdf(
    output_path: Path,
    values: np.ndarray,
    selected: pd.DataFrame,
    analysis: dict,
    fit,
    yield_regime: str = "all",
    selected_kind: str = "event",
) -> Path:
    xmin = float(analysis["global_min_xmin"])
    distribution = dist_from_fit(fit)
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    plot_data_pdf(
        ax,
        values,
        color="0.35",
        alpha=0.7,
        label=f"All irreversible events (n={values.size})",
        drop_label=r"E_S/V_0",
        drop_sign="positive",
        show_legend=False,
    )
    plot_fit_pdf(
        ax,
        fit,
        color="tab:red",
        label="Truncated power-law fit above xmin",
        drop_label=r"E_S/V_0",
        drop_sign="positive",
        show_legend=False,
        set_title=False,
        x_grid_mode="smooth",
        xmin_only=True,
        linewidth=2.0,
    )
    ymin, ymax = ax.get_ylim()
    colors = {"below": "tab:blue", "above": "tab:orange"}
    markers = {"below": "v", "above": "^"}
    for side in ("below", "above"):
        rows = selected[selected["xmin_side"].eq(side)]
        x_values = rows["delta_E_S_over_V0"].to_numpy(dtype=float)
        marker_y = _pdf_marker_y(values, x_values)
        for x_value, y_value in zip(x_values, marker_y):
            ax.vlines(
                x_value,
                ymin,
                y_value,
                color=colors[side],
                alpha=0.22,
                linewidth=0.8,
                zorder=3,
            )
        ax.scatter(
            x_values,
            marker_y,
            marker=markers[side],
            s=42,
            facecolors="white",
            edgecolors=colors[side],
            linewidths=1.0,
            zorder=5,
        )
    ax.axvline(
        xmin,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=rf"Global KS minimum $x_{{\min}}={xmin:.3e}$",
        zorder=4,
    )
    ax.set_title(
        rf"Default {yield_regime}-yield irreversible population: fitted $\Delta E_S/V_0$ PDF"
        rf" ($\alpha={distribution.alpha:.3f}$, "
        rf"$\lambda={distribution.Lambda:.3e}$)"
    )
    below_count = int((selected["xmin_side"] == "below").sum())
    above_count = int((selected["xmin_side"] == "above").sum())
    if not selected_kind:
        raise ValueError("selected_kind must not be empty.")
    below_noun = selected_kind if below_count == 1 else f"{selected_kind}s"
    above_noun = selected_kind if above_count == 1 else f"{selected_kind}s"
    handles = [
        Line2D([], [], color="0.35", marker="o", linestyle="None", label="Empirical PDF"),
        Line2D([], [], color="tab:red", label="Truncated power-law fit"),
        Line2D(
            [], [], marker="v", markerfacecolor="white", markeredgecolor="tab:blue",
            linestyle="None",
            label=f"{below_count} {below_noun} below xmin",
        ),
        Line2D(
            [], [], marker="^", markerfacecolor="white", markeredgecolor="tab:orange",
            linestyle="None",
            label=f"{above_count} {above_noun} at/above xmin",
        ),
        Line2D([], [], color="black", linestyle="--", label=rf"Global KS xmin = {xmin:.3e}"),
    ]
    ax.legend(handles=handles, loc="best", fontsize="small", frameon=True)
    ax.text(
        0.03,
        0.04,
        rf"$D_{{\mathrm{{global}}}}={analysis['global_min_distance']:.4f}$; "
        rf"tail count $n_{{\geq x_{{\min}}}}={(values >= xmin).sum()}$",
        transform=ax.transAxes,
        fontsize="small",
        va="bottom",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _combine_pdfs(paths: list[Path], output_path: Path) -> Path:
    writer = PdfWriter()
    for path in paths:
        reader = PdfReader(path)
        if len(reader.pages) != 1:
            raise RuntimeError(f"Expected one page in {path}; found {len(reader.pages)}.")
        writer.add_page(reader.pages[0])
    if len(writer.pages) != len(paths):
        raise RuntimeError("Combined PDF page count does not match input page count.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        writer.write(stream)
    return output_path


def build_atlas(
    output_dir: Path = DEFAULT_OUTPUT,
    count: int = EVENTS_PER_SIDE,
    yield_regime: str = DEFAULT_YIELD_REGIME,
    replay_root: Path | None = None,
) -> Path:
    if count <= 0:
        raise ValueError("count must be positive.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root = ROOT / "tmp/pdfs/irreversible_xmin_event_atlas"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_events = temp_root / "events"
    temp_events.mkdir(parents=True)

    catalog = build_standard_scatter_catalog(
        batch=DEFAULT_BATCH,
        setting=DEFAULT_SETTING,
    )
    irreversible, values, analysis, fit = _fit_default_irreversible(catalog, yield_regime)
    sources_by_job: dict[str, tuple[RemoteSource, ...]] = {}
    rows = irreversible.copy()
    local_rows = _discover_local_replayed_rows(rows, replay_root)
    selected_rows = _select_available_side(
        rows,
        float(analysis["global_min_xmin"]),
        "below",
        count,
        sources_by_job,
        local_rows,
        strict=False,
    )
    xmin = float(analysis["global_min_xmin"])
    above = _select_available_side(
        rows,
        xmin,
        "above",
        count,
        sources_by_job,
        local_rows,
        max_above_factor=PREFERRED_MAX_ABOVE_FACTOR,
        strict=False,
    )
    if len(above) < count:
        above = _select_available_side(
            rows,
            xmin,
            "above",
            count,
            sources_by_job,
            local_rows,
            max_above_factor=MAX_ABOVE_FACTOR,
            strict=False,
        )
    selected_rows += above
    if not selected_rows:
        raise RuntimeError("No renderable saved irreversible events were located.")
    selected = pd.DataFrame(selected_rows)
    selected["selection_rank"] = selected.groupby("xmin_side").cumcount() + 1
    selected["atlas_label"] = selected.apply(
        lambda row: f"{row['xmin_side']}_xmin_{int(row['selection_rank']):02d}",
        axis=1,
    )
    selected.to_csv(output_dir / "selected_events.csv", index=False)

    distribution = dist_from_fit(fit)
    summary = {
        "batch": DEFAULT_BATCH,
        "setting": DEFAULT_SETTING,
        "population": f"{yield_regime}-yield nonclosing / irreversible",
        "yield_regime": yield_regime,
        "catalogue_count": int(len(irreversible)),
        "global_min_xmin": float(analysis["global_min_xmin"]),
        "global_min_distance": float(analysis["global_min_distance"]),
        "tail_count": int(np.count_nonzero(values >= analysis["global_min_xmin"])),
        "alpha": float(distribution.alpha),
        "lambda": float(distribution.Lambda),
        "requested_event_count_each_side": count,
        "completed_local_replays": int(len(local_rows)),
        "selected_event_count_by_side": {
            side: int((selected["xmin_side"] == side).sum())
            for side in ("below", "above")
        },
        "selection_rule": (
            "nearest saved irreversible events below xmin and nearest above xmin; "
            "above-xmin events are restricted to 10*xmin when possible and 100*xmin "
            "otherwise; fewer than requested are retained when the saved archive is sparse"
        ),
    }
    (output_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    page_paths = [
        _render_fit_pdf(
            output_dir / f"{yield_regime}_yield_irreversible_pdf_with_selected_events.pdf",
            values,
            selected,
            analysis,
            fit,
            yield_regime,
        )
    ]
    for _, row in selected.iterrows():
        label = str(row["atlas_label"])
        replayed_directory = row.get("local_event_directory", "")
        if isinstance(replayed_directory, str) and replayed_directory:
            paths = state_paths_from_directory(Path(replayed_directory))
        else:
            local_directory = temp_events / label
            local_directory.mkdir(parents=True, exist_ok=True)
            source = next(
                source
                for source in sources_by_job[str(row["job_name"])]
                if source.host == row["remote_source"]
                and str(source.data_root) == row["remote_data_root"]
            )
            paths = download_event(
                DownloadRequest(
                    event_id=str(row["event_id"]),
                    source=source,
                    remote_event_directory=Path(row["remote_event_directory"]),
                    local_event_directory=local_directory,
                )
            )
        event = row.copy()
        event["reconnection_mode"] = "none"
        event["reversibility_measured"] = True
        page_path = output_dir / f"{label}.pdf"
        render_event_pdf(
            event,
            paths,
            page_path,
            RenderOptions(output_root=output_dir, output_format="pdf"),
            setting_catalog=catalog,
        )
        page_paths.append(page_path)

    final_path = output_dir / f"{yield_regime}_yield_irreversible_xmin_event_atlas.pdf"
    _combine_pdfs(page_paths, final_path)
    shutil.rmtree(temp_root)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=EVENTS_PER_SIDE)
    parser.add_argument("--yield-regime", choices=("pre", "post"), default=DEFAULT_YIELD_REGIME)
    parser.add_argument("--replay-root", type=Path, default=None)
    args = parser.parse_args()
    print(
        build_atlas(
            args.output_dir,
            count=args.count,
            yield_regime=args.yield_regime,
            replay_root=args.replay_root,
        )
    )


if __name__ == "__main__":
    main()
