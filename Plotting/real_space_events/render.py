"""Render selected event sheets and class comparisons as images."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from Plotting import meshEventPlotting as mesh_plot

from .models import EventSheetLayout, EventStatePaths, RenderOptions


def _scientific_latex(value):
    """Return explicit coefficient-times-ten notation without math delimiters."""

    if not np.isfinite(value):
        return ""
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10**exponent
    coefficient = round(coefficient, 1)
    if abs(coefficient) >= 10:
        coefficient /= 10
        exponent += 1
    return rf"{coefficient:.1f}\times 10^{{{exponent}}}"


def _scientific_tick(value, _position):
    """Format colourbar ticks as explicit coefficient-times-ten notation."""

    formatted = _scientific_latex(value)
    return rf"${formatted}$" if formatted else ""


def _load_states(paths: EventStatePaths):
    short_states = paths.as_dict()
    return mesh_plot.align_periodic_states(
        {
            "state0_min_gamma": mesh_plot.load_mesh_state(short_states["state0"]),
            "state1_affine_gamma_plus": mesh_plot.load_mesh_state(short_states["state1"]),
            "state2_relaxed_gamma_plus": mesh_plot.load_mesh_state(short_states["state2"]),
            "state3_affine_gamma_minus": mesh_plot.load_mesh_state(short_states["state3"]),
            "state4_relaxed_gamma": mesh_plot.load_mesh_state(short_states["state4"]),
        },
        load_by_state={},
        box_size=1.0,
    )


def _unique_point_indices(state):
    _, indices = np.unique(state.reference_indices, return_index=True)
    return np.sort(indices)


def _arrow_indices(state, displacement, zoom, maximum):
    unique = _unique_point_indices(state)
    points = state.points[unique]
    inside = (
        (points[:, 0] >= zoom.xlim[0])
        & (points[:, 0] <= zoom.xlim[1])
        & (points[:, 1] >= zoom.ylim[0])
        & (points[:, 1] <= zoom.ylim[1])
    )
    candidates = unique[inside]
    if maximum is None or candidates.size <= maximum:
        return candidates
    magnitudes = np.linalg.norm(displacement[candidates], axis=1)
    order = np.argsort(magnitudes)[::-1]
    return candidates[order[:maximum]]


def _plot_m3_outline(ax, state, changed):
    changed = np.asarray(changed, dtype=bool)
    if changed.shape != (len(state.triangles),) or not np.any(changed):
        return
    for triangle in state.triangles[changed]:
        polygon = state.points[np.r_[triangle, triangle[0]]]
        line, = ax.plot(
            polygon[:, 0], polygon[:, 1], color="black", linewidth=0.7, alpha=0.8
        )
        line.set_rasterized(True)


def _display_arrow_scale(scale, options):
    """Apply the common display amplification and optional override."""

    scale = type(scale)(
        scale.amplification * options.arrow_length_multiplier,
        scale.physical_key_length,
        scale.target_element_fraction,
    )
    if options.common_arrow_scale is not None:
        scale = type(scale)(
            float(options.common_arrow_scale),
            scale.physical_key_length,
            scale.target_element_fraction,
        )
    return scale


def _shift_axes_horizontally(axes, shift: float) -> None:
    """Move axes horizontally by a fraction of the full figure width."""

    for axis in axes:
        position = axis.get_position()
        axis.set_position(
            [position.x0 + shift, position.y0, position.width, position.height]
        )


def _center_colorbar_on_axis(colorbar, axis) -> None:
    """Center an existing horizontal colourbar under its mesh axis."""

    axis_position = axis.get_position()
    bar_position = colorbar.ax.get_position()
    centered_x = axis_position.x0 + (axis_position.width - bar_position.width) / 2
    colorbar.ax.set_position(
        [centered_x, bar_position.y0, bar_position.width, bar_position.height]
    )


def _plot_reversibility_scatter(
    ax,
    catalog: pd.DataFrame,
    event: pd.Series,
    layout: EventSheetLayout,
) -> None:
    """Draw the chosen-setting scatter using ``plot_epsilon_scatter`` semantics."""

    if catalog is None or catalog.empty:
        catalog = pd.DataFrame([event])
    x_name = "delta_rev_u"
    y_name = "delta_E_S_over_V0"
    required = {x_name, y_name, "yield_regime", "rev_u_cut"}
    missing = required.difference(catalog.columns)
    if missing:
        raise KeyError(f"Scatter catalogue is missing columns: {sorted(missing)}")
    rows = catalog.copy()
    if "delta_gamma" in rows and np.isfinite(float(event.get("delta_gamma", np.nan))):
        rows = rows[np.isclose(rows["delta_gamma"], float(event["delta_gamma"]))]
    if rows.empty:
        raise ValueError("No rows remain for the selected setting scatter.")

    cut_values = rows["rev_u_cut"].to_numpy(dtype=float)
    cut_values = cut_values[np.isfinite(cut_values) & (cut_values > 0)]
    if cut_values.size == 0:
        raise ValueError("The selected setting has no finite Delta_rev u cut.")
    cut = float(event.get("rev_u_cut", np.median(cut_values)))
    if not np.isfinite(cut) or cut <= 0:
        raise ValueError(f"Invalid selected-setting Delta_rev u cut: {cut}")

    # Marker shape identifies yield regime, while color identifies the
    # reversible/irreversible population and yield regime.
    colors = {
        ("pre", "closing"): "#2166ac",
        ("post", "closing"): "#67a9cf",
        ("pre", "nonclosing"): "#b2182b",
        ("post", "nonclosing"): "#ef8a62",
    }
    cut_color = "0.25"
    plotted_x = []
    plotted_y = []
    has_discarded = False
    for post_yield, closing_marker, nonclosing_marker in (
        ("pre", "^", "s"),
        ("post", "o", "D"),
    ):
        regime_rows = rows[rows["yield_regime"].astype(str) == post_yield]
        x = regime_rows[x_name].to_numpy(dtype=float)
        y = regime_rows[y_name].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if "population" in regime_rows:
            population = regime_rows["population"].astype(str).to_numpy()
            closing = valid & (population == "closing")
            nonclosing = valid & (population == "nonclosing")
            discarded = valid & (population == "discarded")
        else:
            closing = valid & (x <= cut)
            nonclosing = valid & (x > cut)
            discarded = np.zeros_like(valid)
        for mask, marker in (
            (closing, closing_marker),
            (nonclosing, nonclosing_marker),
            (discarded, "x"),
        ):
            if np.any(mask):
                if marker == "x":
                    color = "0.45"
                    has_discarded = True
                else:
                    population_name = "closing" if marker == closing_marker else "nonclosing"
                    color = colors[(post_yield, population_name)]
                plotted_x.append(x[mask])
                plotted_y.append(y[mask])
                marker_colors = (
                    {"color": color}
                    if marker == "x"
                    else {"facecolors": "none", "edgecolors": color}
                )
                ax.scatter(
                    x[mask], y[mask], marker=marker, s=18,
                    alpha=0.5,
                    linewidths=0.8, rasterized=True, zorder=2,
                    **marker_colors,
                )

    event_x = float(event.get(x_name, np.nan))
    event_y = float(event.get(y_name, np.nan))
    if not (np.isfinite(event_x) and np.isfinite(event_y) and event_x > 0 and event_y > 0):
        raise ValueError("Selected event is not a positive stress-corrected energy drop.")
    plotted_x.append(np.array([event_x]))
    plotted_y.append(np.array([event_y]))
    ax.axvline(cut, color=cut_color, linestyle="--", linewidth=1.0, alpha=0.8, zorder=3)
    ax.axvline(event_x, color="black", linewidth=0.75, alpha=0.8)
    ax.axhline(event_y, color="black", linewidth=0.75, alpha=0.8)
    ax.scatter(
        [event_x], [event_y], facecolors="none", edgecolors="black",
        s=55, linewidths=0.9, zorder=8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    all_x = np.concatenate(plotted_x)
    all_y = np.concatenate(plotted_y)
    ax.set_xlim(all_x.min() / 1.25, all_x.max() * 1.25)
    ax.set_ylim(all_y.min() / 1.5, all_y.max() * 1.5)
    ax.set_xlabel(r"$\Delta_{\mathrm{rev}}\mathbf{u}$")
    ax.set_ylabel(r"$\Delta E_S/V_0$", labelpad=layout.scatter_ylabel_pad)
    ax.tick_params(axis="y", pad=layout.scatter_ytick_pad)
    ax.set_title("(e) chosen-setting reversibility scatter")
    legend_markers = (
        ("^", colors[("pre", "closing")], "pre-yield reversible"),
        ("o", colors[("post", "closing")], "post-yield reversible"),
        ("s", colors[("pre", "nonclosing")], "pre-yield irreversible"),
        ("D", colors[("post", "nonclosing")], "post-yield irreversible"),
    )
    handles = [
        Line2D(
            [], [], marker=marker, linestyle="None", markerfacecolor="none",
            markeredgecolor=color, label=label,
        )
        for marker, color, label in legend_markers
    ]
    if has_discarded:
        handles.append(
            Line2D([], [], marker="x", linestyle="None", color="0.45", label="discarded island")
        )
    handles.append(
        Line2D([], [], color=cut_color, linestyle="--", label=r"$\Delta_{\mathrm{rev}}\mathbf{u}$ cut")
    )
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=6,
        frameon=True,
        ncol=1,
    )


def render_event_pdf(
    event: pd.Series,
    state_paths: EventStatePaths,
    output_path: Path,
    options: RenderOptions,
    *,
    setting_catalog: pd.DataFrame | None = None,
) -> Path:
    """Render one publication-oriented event sheet as PDF or PNG.

    Panels are arranged in three columns.  The left and middle columns show
    forward and backward energy-difference fields, respectively; the right
    column retains the chosen-setting scatter and closure field.  The two
    full-mesh panels show the same energy colours as their corresponding
    zoomed panels.

    The energy field uses a symmetric coolwarm norm centered at zero.  Forward
    m3-changed elements are outlined.  Arrow amplification is recorded in the
    caption.  The title/caption records load, yield regime,
    class, Delta E_inter/V0, Delta E_S/V0, Delta_rev u and participation.
    """

    options.validate()
    layout = options.layout
    suffix = output_path.suffix.lower()
    if suffix not in {".pdf", ".png"}:
        raise ValueError("Event figures must be written as PDF or PNG files.")
    if options.output_format not in {"auto", suffix[1:]}:
        raise ValueError(
            f"Output suffix {suffix!r} does not match output_format={options.output_format!r}."
        )
    states = _load_states(state_paths)
    state0 = states["state0_min_gamma"]
    state1 = states["state1_affine_gamma_plus"]
    state2 = states["state2_relaxed_gamma_plus"]
    state3 = states["state3_affine_gamma_minus"]
    state4 = states["state4_relaxed_gamma"]
    relation = mesh_plot.determine_topology_relation(state0, state2)
    displacements = mesh_plot.calculate_event_displacements(states)
    m3_changed = mesh_plot.calculate_forward_m3_change(states)
    energy_change, geometry = mesh_plot.calculate_energy_change_field(
        state0,
        state2,
        relation=relation,
        common_grid_resolution=options.common_grid_resolution,
    )
    backward_relation = mesh_plot.determine_topology_relation(state2, state4)
    backward_energy_change, backward_geometry = mesh_plot.calculate_energy_change_field(
        state2,
        state4,
        relation=backward_relation,
        common_grid_resolution=options.common_grid_resolution,
    )
    zoom = mesh_plot.choose_energy_density_zoom(
        state2,
        energy_change,
        geometry,
        convolution_width=10.0,
        maximum_width=20.0,
    )
    full_points = state2.points
    full_zoom = mesh_plot.ZoomRegion(
        xlim=(float(np.min(full_points[:, 0])), float(np.max(full_points[:, 0]))),
        ylim=(float(np.min(full_points[:, 1])), float(np.max(full_points[:, 1]))),
        activity_fraction=1.0,
        center=(float(np.mean(full_points[:, 0])), float(np.mean(full_points[:, 1]))),
    )
    field_values = (
        energy_change
        if geometry.kind == "triangles"
        else np.asarray(geometry.values, dtype=float)
    )
    backward_field_values = (
        backward_energy_change
        if backward_geometry.kind == "triangles"
        else np.asarray(backward_geometry.values, dtype=float)
    )
    energy_limit = options.common_energy_limit
    if energy_limit is None:
        energy_limit = max(
            float(np.nanmax(np.abs(field_values))),
            float(np.nanmax(np.abs(backward_field_values))),
        )
    if not np.isfinite(energy_limit) or energy_limit <= 0:
        energy_limit = 1.0
    element_edges = state2.points[state2.triangles[:, [0, 1, 1, 2]].reshape(-1)]
    element_length = float(
        np.median(np.linalg.norm(element_edges[0::2] - element_edges[1::2], axis=1))
    )
    arrow_scale = _display_arrow_scale(
        mesh_plot.choose_arrow_scale(
            displacements.forward_relaxation,
            element_length=element_length,
            target_element_fraction=options.arrow_target_element_fraction,
        ),
        options,
    )
    backward_scale = _display_arrow_scale(
        mesh_plot.choose_arrow_scale(
            displacements.backward_relaxation,
            element_length=element_length,
            target_element_fraction=options.arrow_target_element_fraction,
        ),
        options,
    )
    closure_scale = _display_arrow_scale(
        mesh_plot.choose_arrow_scale(
            displacements.closure_residual,
            element_length=element_length,
            target_element_fraction=options.arrow_target_element_fraction,
        ),
        options,
    )

    fig = plt.figure(figsize=options.figure_size)
    grid = fig.add_gridspec(2, 3, width_ratios=layout.column_width_ratios)
    locator_ax = fig.add_subplot(grid[0, 0])
    backward_locator_ax = fig.add_subplot(grid[0, 1])
    scatter_ax = fig.add_subplot(grid[0, 2])
    event_ax = fig.add_subplot(grid[1, 0])
    backward_ax = fig.add_subplot(grid[1, 1])
    closure_ax = fig.add_subplot(grid[1, 2])

    mesh_plot.plot_energy_change_background(
        locator_ax,
        state2,
        energy_change,
        geometry,
        zoom=full_zoom,
        symmetric_limit=energy_limit,
        show_mesh_edges=True,
        mesh_edge_color="face",
        rasterized=True,
    )
    locator_ax.add_patch(
        Rectangle(
            (zoom.xlim[0], zoom.ylim[0]),
            zoom.xlim[1] - zoom.xlim[0],
            zoom.ylim[1] - zoom.ylim[0],
            fill=False,
            color="C3",
            linewidth=1.2,
        )
    )
    locator_ax.set_title(r"(a) full mesh: $E_0^{(e)}-E_2^{(e)}$")

    mesh_plot.plot_energy_change_background(
        backward_locator_ax,
        state2,
        backward_energy_change,
        backward_geometry,
        zoom=full_zoom,
        symmetric_limit=energy_limit,
        show_mesh_edges=True,
        mesh_edge_color="face",
        rasterized=True,
    )
    backward_locator_ax.add_patch(
        Rectangle(
            (zoom.xlim[0], zoom.ylim[0]),
            zoom.xlim[1] - zoom.xlim[0],
            zoom.ylim[1] - zoom.ylim[0],
            fill=False,
            color="C3",
            linewidth=1.2,
        )
    )
    backward_locator_ax.set_title(r"(c) full mesh: $E_2^{(e)}-E_4^{(e)}$")
    _plot_reversibility_scatter(scatter_ax, setting_catalog, event, layout)
    scatter_ax.set_title("(e) chosen-setting reversibility scatter")
    mappable = mesh_plot.plot_energy_change_background(
        event_ax,
        state2,
        energy_change,
        geometry,
        zoom=zoom,
        symmetric_limit=energy_limit,
        rasterized=True,
    )
    arrow_indices = _arrow_indices(
        state1, displacements.forward_relaxation, zoom, options.maximum_arrows
    )
    forward_quiver = mesh_plot.plot_displacement_arrows(
        event_ax,
        state1.points[arrow_indices],
        displacements.forward_relaxation[arrow_indices],
        arrow_scale=arrow_scale,
        zoom=zoom,
        show_key=True,
        key_label=rf"$|\rightarrow| = {_scientific_latex(arrow_scale.physical_key_length)}$",
        rasterized=True,
    )
    if relation is mesh_plot.TopologyRelation.IDENTICAL:
        _plot_m3_outline(event_ax, state2, m3_changed)
    event_ax.set_title(r"(b) forward relaxation, $E_0^{(e)}-E_2^{(e)}$")
    colorbar = fig.colorbar(
        mappable,
        ax=event_ax,
        orientation="horizontal",
        shrink=0.65,
        fraction=0.05,
        pad=0.13,
        aspect=50,
    )
    colorbar.set_ticks((-0.75 * energy_limit, 0.0, 0.75 * energy_limit))
    colorbar.formatter = FuncFormatter(_scientific_tick)
    colorbar.update_ticks()

    backward_mappable = mesh_plot.plot_energy_change_background(
        backward_ax,
        state2,
        backward_energy_change,
        backward_geometry,
        zoom=zoom,
        symmetric_limit=energy_limit,
        rasterized=True,
    )
    backward_indices = _arrow_indices(
        state3, displacements.backward_relaxation, zoom, options.maximum_arrows
    )
    mesh_plot.plot_displacement_arrows(
        backward_ax,
        state3.points[backward_indices],
        displacements.backward_relaxation[backward_indices],
        arrow_scale=backward_scale,
        zoom=zoom,
        show_key=True,
        key_label=rf"$|\rightarrow| = {_scientific_latex(backward_scale.physical_key_length)}$",
        rasterized=True,
    )
    backward_ax.set_title(r"(d) backward relaxation, $E_2^{(e)}-E_4^{(e)}$")
    backward_colorbar = fig.colorbar(
        backward_mappable,
        ax=backward_ax,
        orientation="horizontal",
        shrink=0.65,
        fraction=0.05,
        pad=0.13,
        aspect=50,
    )
    backward_colorbar.set_ticks(
        (-0.75 * energy_limit, 0.0, 0.75 * energy_limit)
    )
    backward_colorbar.formatter = FuncFormatter(_scientific_tick)
    backward_colorbar.update_ticks()

    closure_triangulation = mtri.Triangulation(
        state0.points[:, 0], state0.points[:, 1], state0.triangles
    )
    closure_edges = closure_ax.triplot(
        closure_triangulation, color="0.55", linewidth=0.08, alpha=0.4
    )
    for artist in closure_edges:
        artist.set_rasterized(True)
    closure_indices = _arrow_indices(
        state0, displacements.closure_residual, zoom, options.maximum_arrows
    )
    closure_quiver = mesh_plot.plot_displacement_arrows(
        closure_ax,
        state0.points[closure_indices],
        displacements.closure_residual[closure_indices],
        arrow_scale=closure_scale,
        zoom=zoom,
        show_key=True,
        key_label=rf"$|\rightarrow| = {_scientific_latex(closure_scale.physical_key_length)}$",
        rasterized=True,
    )
    closure_ax.set_xlim(*zoom.xlim)
    closure_ax.set_ylim(*zoom.ylim)
    closure_ax.set_aspect("equal", adjustable="box")
    closure_ax.set_title(r"(f) closure residual $\mathbf{x}_4-\mathbf{x}_0$")

    # Equal-aspect mesh panels otherwise sit centered inside wider grid cells.
    # Anchor the first two columns toward one another to remove that invisible
    # in-cell padding.  Keep the final column left-anchored but separated enough
    # to accommodate its y-axis labels.
    locator_ax.set_anchor("E")
    event_ax.set_anchor("E")
    backward_locator_ax.set_anchor("W")
    backward_ax.set_anchor("W")
    closure_ax.set_anchor("W")

    event_class = str(event.get("event_class", "unknown")).replace("_", " ")
    atlas_label = str(event.get("atlas_label", "")).replace("_", " ")
    load = float(event.get("load", np.nan))
    yield_regime = {"pre": "pre-yield", "post": "post-yield"}.get(
        str(event.get("yield_regime", "unknown")),
        str(event.get("yield_regime", "unknown")),
    )
    fig.suptitle(
        f"{atlas_label + ' | ' if atlas_label else ''}{event_class} | "
        f"$\\gamma$ = {load:.5g} | {yield_regime}",
        fontsize=11,
    )
    fig.subplots_adjust(
        top=layout.top,
        bottom=layout.bottom,
        left=layout.left,
        right=layout.right,
        wspace=layout.column_spacing,
        hspace=layout.row_spacing,
    )
    # Leave just enough room for the middle column's y tick labels while using
    # the recovered space to tighten the gap before the final column.
    _shift_axes_horizontally(
        (backward_locator_ax, backward_ax), layout.middle_column_shift
    )
    for mesh_axis, colorbar in (
        (event_ax, colorbar),
        (backward_ax, backward_colorbar),
    ):
        _center_colorbar_on_axis(colorbar, mesh_axis)
    # Keep the final column's left-side tick labels and y label clear of the
    # middle mesh panels; axes-box separation alone is insufficient here.
    _shift_axes_horizontally((scatter_ax, closure_ax), layout.final_column_shift)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".pdf":
        with plt.rc_context({"pdf.compression": 9}):
            fig.savefig(output_path, dpi=options.rasterized_dpi)
    else:
        fig.savefig(output_path, dpi=options.rasterized_dpi)
    plt.close(fig)
    return output_path


def render_comparison_pdf(
    selected_events: pd.DataFrame,
    state_paths_by_event: dict[str, EventStatePaths],
    output_path: Path,
    options: RenderOptions,
) -> Path:
    """Render one row per nonempty event class using shared visual scales."""

    options.validate()
    if output_path.suffix.lower() != ".pdf":
        raise ValueError("Comparison figures must be written as PDF files.")
    if selected_events.empty:
        raise ValueError("Cannot render an empty event comparison.")
    raise NotImplementedError


def event_output_name(event: pd.Series, extension: str = ".png") -> str:
    """Return a deterministic filesystem-safe image name."""

    if extension not in {".pdf", ".png"}:
        raise ValueError("Event output extension must be '.pdf' or '.png'.")

    job = str(event.get("job_name", "event")).replace("/", "_")
    load = float(event.get("load", 0.0))
    kind = str(event.get("representative_kind", "event"))
    event_class = str(event.get("event_class", "unknown")).replace("_", "-")
    return f"{event_class}_{kind}_{job}_load{load:.8g}{extension}"


def event_pdf_name(event: pd.Series) -> str:
    """Return the legacy PDF name for callers that explicitly need PDF."""

    return event_output_name(event, ".pdf")
