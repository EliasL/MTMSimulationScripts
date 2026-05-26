from __future__ import annotations

from dataclasses import dataclass
import re

import matplotlib.pyplot as plt
import numpy as np


DEBUG_BLOCK = """
Standard exception: Reduction exploded in Mesh::updateElementsForces: eIndex=62120, m3Nr=299, load=1.4152200000007946, loadSteps=126523.
[LBFGS] Mesh context:
[LBFGS]   load=1.4152200000007946, loadSteps=126523, nrMinFunctionCalls=1660, nrMinItterations=982
[LBFGS]   totalEnergy=5844.1222245966246, maxForce=1018867607.0794692, rows=200, cols=200, nrElements=80000, nrNodes=40000, usingPBC=1, updateState=Dirty
[LBFGS] Element:
[LBFGS]   eIndex=62120, m3Nr=299, pastM3Nr=0, pastStepM3Nr=0, redQuadrant=3, angleNode=-1, angleEdge=invalid, noise=1
[LBFGS]   ghost[0]: referenceId=31459, id=(59, 157), periodicShift=(0, 0), pos=(265.4841234876086, 152.54303000517862), ref_pos=(-0.5, -0.5)
[LBFGS]   ghost[1]: referenceId=31061, id=(61, 155), periodicShift=(0, 0), pos=(265.50705462034171, 152.24625652527953), ref_pos=(0.5, -0.5)
[LBFGS]   ghost[2]: referenceId=31659, id=(59, 158), periodicShift=(0, 0), pos=(265.41517052702295, 153.4343414524017), ref_pos=(-0.5, 0.5)
[LBFGS]   currentArea=1.2314479765563273e-05, referenceArea=0.5
[LBFGS]   F:
[LBFGS]  0.022931132733106097 -0.068952960585647816
[LBFGS]  -0.29677347989908753   0.89131144722307454
[LBFGS]   C:
[LBFGS] 0.088600335219837442 -0.26609876935781385
[LBFGS] -0.26609876935781385  0.79919060672441755
[LBFGS]   C_R:
[LBFGS] 1.0075560714084553e-06 4.7289883538248478e-07
[LBFGS] 4.7289883538248478e-07 0.00060225857786065651
[LBFGS]   G:
[LBFGS]  1.1251080414268595 0.27935277794138919
[LBFGS] 0.27935277794138919  1.3690734575873376
[LBFGS]   M_l:
[LBFGS]   3 889
[LBFGS]   1 296

"""


GROUP_ORDER = (
    "preFlipSelfGhost",
    "preFlipPartnerGhost",
    "postFlipSelfGhost",
    "postFlipPartnerGhost",
)

GROUP_META = {
    "preFlipSelfGhost": {"stage": "pre", "role": "self", "prefix": "S"},
    "preFlipPartnerGhost": {"stage": "pre", "role": "partner", "prefix": "P"},
    "postFlipSelfGhost": {"stage": "post", "role": "self", "prefix": "S"},
    "postFlipPartnerGhost": {"stage": "post", "role": "partner", "prefix": "P"},
}

PAIR_SECTION_ORDER = ("element1", "element2")

PAIR_SECTION_META = {
    "element1": {"stage": "pair", "role": "self", "prefix": "S"},
    "element2": {"stage": "pair", "role": "partner", "prefix": "P"},
}

ROLE_STYLE = {
    "self": {"color": "tab:blue", "marker": "o", "label_sign": 1.0},
    "partner": {"color": "tab:orange", "marker": "s", "label_sign": -1.0},
}

NODE_RE = re.compile(
    r"""
    ^
    (?P<group>
        preFlipSelfGhost
        |preFlipPartnerGhost
        |postFlipSelfGhost
        |postFlipPartnerGhost
        |ghost
        |realNodeRef
    )
    \[(?P<index>\d+)\]:
    \s+refId=(?P<ref_id>-?\d+)
    (?:\s+id=\((?P<id_x>-?\d+),\s*(?P<id_y>-?\d+)\))?
    (?:\s+pShift=\((?P<shift_x>-?\d+),\s*(?P<shift_y>-?\d+)\))?
    \s+pos=\((?P<pos_x>[-+0-9.eE]+),\s*(?P<pos_y>[-+0-9.eE]+)\)
    \s+ref=\((?P<ref_x>[-+0-9.eE]+),\s*(?P<ref_y>[-+0-9.eE]+)\)
    \s+u=\((?P<u_x>[-+0-9.eE]+),\s*(?P<u_y>[-+0-9.eE]+)\)
    $
    """,
    re.VERBOSE,
)

SHARED_EDGE_RE = re.compile(
    r"""
    ^
    sharedEdge\[(?P<index>\d+)\]_(?P<section>element1|element2):
    \s+refId=(?P<ref_id>-?\d+)
    (?:\s+id=\((?P<id_x>-?\d+),\s*(?P<id_y>-?\d+)\))?
    (?:\s+pShift=\((?P<shift_x>-?\d+),\s*(?P<shift_y>-?\d+)\))?
    \s+pos=\((?P<pos_x>[-+0-9.eE]+),\s*(?P<pos_y>[-+0-9.eE]+)\)
    \s+ref=\((?P<ref_x>[-+0-9.eE]+),\s*(?P<ref_y>[-+0-9.eE]+)\)
    \s+u=\((?P<u_x>[-+0-9.eE]+),\s*(?P<u_y>[-+0-9.eE]+)\)
    $
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class GhostNode:
    group_name: str
    index: int
    ref_id: int
    grid_id: tuple[int, int]
    p_shift: tuple[int, int]
    pos: np.ndarray
    ref: np.ndarray
    u: np.ndarray
    stage: str
    role: str
    prefix: str

    @property
    def short_label(self) -> str:
        return f"{self.prefix}{self.index}({self.ref_id})"


@dataclass(frozen=True)
class TriangleGroup:
    name: str
    stage: str
    role: str
    nodes: tuple[GhostNode, ...]

    def coords(self, field: str) -> np.ndarray:
        if field not in {"ref", "pos"}:
            raise ValueError(f"Unsupported coordinate field: {field}")
        return np.array([getattr(node, field) for node in self.nodes], dtype=float)


def parse_vec2(match: re.Match[str], x_name: str, y_name: str) -> np.ndarray:
    return np.array([float(match.group(x_name)), float(match.group(y_name))], dtype=float)


def parse_optional_int(match: re.Match[str], name: str, default: int = 0) -> int:
    value = match.group(name)
    return default if value is None else int(value)


def node_from_match(
    match: re.Match[str], group_name: str, *, stage: str, role: str, prefix: str
) -> GhostNode:
    return GhostNode(
        group_name=group_name,
        index=int(match.group("index")),
        ref_id=int(match.group("ref_id")),
        grid_id=(
            parse_optional_int(match, "id_x"),
            parse_optional_int(match, "id_y"),
        ),
        p_shift=(
            parse_optional_int(match, "shift_x"),
            parse_optional_int(match, "shift_y"),
        ),
        pos=parse_vec2(match, "pos_x", "pos_y"),
        ref=parse_vec2(match, "ref_x", "ref_y"),
        u=parse_vec2(match, "u_x", "u_y"),
        stage=stage,
        role=role,
        prefix=prefix,
    )


def clone_node_for_group(node: GhostNode, group_name: str, index: int) -> GhostNode:
    meta = GROUP_META[group_name]
    return GhostNode(
        group_name=group_name,
        index=index,
        ref_id=node.ref_id,
        grid_id=node.grid_id,
        p_shift=node.p_shift,
        pos=node.pos.copy(),
        ref=node.ref.copy(),
        u=node.u.copy(),
        stage=meta["stage"],
        role=meta["role"],
        prefix=meta["prefix"],
    )


def enrich_node_metadata(
    node: GhostNode,
    parsed_by_group: dict[str, dict[int, GhostNode]],
    generic_ghosts: dict[int, GhostNode],
) -> GhostNode:
    if node.grid_id != (0, 0):
        return node

    for group_nodes in list(parsed_by_group.values()) + [generic_ghosts]:
        for candidate in group_nodes.values():
            if candidate.ref_id != node.ref_id:
                continue
            if candidate.grid_id == (0, 0):
                continue
            return GhostNode(
                group_name=node.group_name,
                index=node.index,
                ref_id=node.ref_id,
                grid_id=candidate.grid_id,
                p_shift=candidate.p_shift,
                pos=node.pos.copy(),
                ref=node.ref.copy(),
                u=node.u.copy(),
                stage=node.stage,
                role=node.role,
                prefix=node.prefix,
            )

    return node


def fill_missing_groups(
    parsed_by_group: dict[str, dict[int, GhostNode]],
    generic_ghosts: dict[int, GhostNode],
    real_node_refs: dict[int, GhostNode],
) -> None:
    post_self_group = parsed_by_group["postFlipSelfGhost"]
    for index, node in generic_ghosts.items():
        if index not in post_self_group:
            post_self_group[index] = clone_node_for_group(node, "postFlipSelfGhost", index)

    complete_ref_sets = [
        frozenset(node.ref_id for node in group_nodes.values())
        for group_nodes in parsed_by_group.values()
        if len(group_nodes) == 3
    ]

    for group_name in GROUP_ORDER:
        group_nodes = parsed_by_group[group_name]
        missing = [index for index in range(3) if index not in group_nodes]
        if not missing:
            continue
        if len(missing) != 1:
            raise ValueError(
                f"Group {group_name} is incomplete and cannot be recovered: missing {missing}"
            )

        known_ref_ids = {node.ref_id for node in group_nodes.values()}
        candidates: list[GhostNode] = []
        for node in real_node_refs.values():
            if node.ref_id in known_ref_ids:
                continue
            candidate_set = frozenset(known_ref_ids | {node.ref_id})
            if len(candidate_set) != 3:
                continue
            if candidate_set in complete_ref_sets:
                continue
            candidates.append(node)

        if len(candidates) != 1:
            raise ValueError(
                f"Could not uniquely recover {group_name}[{missing[0]}]; "
                f"candidate refIds={[node.ref_id for node in candidates]}"
            )

        recovered = clone_node_for_group(
            enrich_node_metadata(candidates[0], parsed_by_group, generic_ghosts),
            group_name,
            missing[0],
        )
        group_nodes[missing[0]] = recovered
        complete_ref_sets.append(frozenset(node.ref_id for node in group_nodes.values()))
        print(
            f"Recovered {group_name}[{missing[0]}] from realNodeRef "
            f"refId={recovered.ref_id}"
        )


def build_indexed_groups(
    parsed_nodes: dict[str, dict[int, GhostNode]],
    order: tuple[str, ...],
    meta_by_name: dict[str, dict[str, str]],
    expected_indices: set[int],
) -> dict[str, TriangleGroup]:
    groups: dict[str, TriangleGroup] = {}
    for group_name in order:
        group_nodes = parsed_nodes[group_name]
        if set(group_nodes) != expected_indices:
            raise ValueError(
                f"Group {group_name} must contain indices {sorted(expected_indices)}, "
                f"got {sorted(group_nodes)}"
            )
        meta = meta_by_name[group_name]
        groups[group_name] = TriangleGroup(
            name=group_name,
            stage=meta["stage"],
            role=meta["role"],
            nodes=tuple(group_nodes[i] for i in sorted(expected_indices)),
        )
    return groups


def parse_debug_block(text: str) -> tuple[str, dict[str, TriangleGroup]]:
    parsed_by_group: dict[str, dict[int, GhostNode]] = {name: {} for name in GROUP_ORDER}
    generic_ghosts: dict[int, GhostNode] = {}
    real_node_refs: dict[int, GhostNode] = {}
    pair_section_ghosts: dict[str, dict[int, GhostNode]] = {
        name: {} for name in PAIR_SECTION_ORDER
    }
    shared_edge_nodes: dict[str, dict[int, GhostNode]] = {
        name: {} for name in PAIR_SECTION_ORDER
    }
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line in {f"{name}:" for name in PAIR_SECTION_ORDER}:
            current_section = line[:-1]
            continue

        shared_edge_match = SHARED_EDGE_RE.fullmatch(line)
        if shared_edge_match is not None:
            section = shared_edge_match.group("section")
            index = int(shared_edge_match.group("index"))
            if index in shared_edge_nodes[section]:
                raise ValueError(f"Duplicate node index {index} in shared edge {section}")
            meta = PAIR_SECTION_META[section]
            shared_edge_nodes[section][index] = node_from_match(
                shared_edge_match,
                section,
                stage="shared_edge",
                role=meta["role"],
                prefix=meta["prefix"],
            )
            continue

        match = NODE_RE.fullmatch(line)
        if match is None:
            continue

        group_name = match.group("group")
        index = int(match.group("index"))
        if group_name in GROUP_META:
            meta = GROUP_META[group_name]
            if index in parsed_by_group[group_name]:
                raise ValueError(f"Duplicate node index {index} in group {group_name}")
            parsed_by_group[group_name][index] = node_from_match(
                match,
                group_name,
                stage=meta["stage"],
                role=meta["role"],
                prefix=meta["prefix"],
            )
            continue

        if group_name == "ghost":
            if current_section in PAIR_SECTION_META:
                if index in pair_section_ghosts[current_section]:
                    raise ValueError(
                        f"Duplicate node index {index} in section {current_section}"
                    )
                meta = PAIR_SECTION_META[current_section]
                pair_section_ghosts[current_section][index] = node_from_match(
                    match,
                    current_section,
                    stage=meta["stage"],
                    role=meta["role"],
                    prefix=meta["prefix"],
                )
                continue

            if index in generic_ghosts:
                raise ValueError(f"Duplicate node index {index} in group ghost")
            generic_ghosts[index] = node_from_match(
                match,
                "ghost",
                stage="unknown",
                role="self",
                prefix="G",
            )
            continue

        if group_name == "realNodeRef":
            if index in real_node_refs:
                raise ValueError(f"Duplicate node index {index} in group realNodeRef")
            real_node_refs[index] = node_from_match(
                match,
                "realNodeRef",
                stage="reference",
                role="self",
                prefix="R",
            )
            continue

        raise ValueError(f"Unsupported parsed group: {group_name}")

    if any(parsed_by_group[group_name] for group_name in GROUP_ORDER):
        fill_missing_groups(parsed_by_group, generic_ghosts, real_node_refs)
        return "flip", build_indexed_groups(
            parsed_by_group, GROUP_ORDER, GROUP_META, {0, 1, 2}
        )

    if any(pair_section_ghosts[group_name] for group_name in PAIR_SECTION_ORDER):
        return "pair", build_indexed_groups(
            pair_section_ghosts, PAIR_SECTION_ORDER, PAIR_SECTION_META, {0, 1, 2}
        )

    if any(shared_edge_nodes[group_name] for group_name in PAIR_SECTION_ORDER):
        return "shared_edge", build_indexed_groups(
            shared_edge_nodes, PAIR_SECTION_ORDER, PAIR_SECTION_META, {0, 1}
        )

    raise ValueError("Did not find any supported ghost-node triangle groups in debug block")


def format_vec2(vec: np.ndarray) -> str:
    return f"({vec[0]:8.4f}, {vec[1]:8.4f})"


def print_triangle_summary(kind: str, groups: dict[str, TriangleGroup]) -> None:
    header = "Parsed ghost triangles:" if kind != "shared_edge" else "Parsed shared edges:"
    print(header)
    group_order = GROUP_ORDER if kind == "flip" else PAIR_SECTION_ORDER
    for group_name in group_order:
        group = groups[group_name]
        print(f"\n{group.name}:")
        for node in group.nodes:
            print(
                f"  {node.short_label}: "
                f"id={node.grid_id} "
                f"refId={node.ref_id} "
                f"ref={format_vec2(node.ref)} "
                f"pos={format_vec2(node.pos)} "
                f"u={format_vec2(node.u)}"
            )


def panel_limits(groups: list[TriangleGroup], field: str, margin_fraction: float = 0.18):
    points = np.vstack([group.coords(field) for group in groups])
    xy_min = points.min(axis=0)
    xy_max = points.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1.0)
    center = 0.5 * (xy_min + xy_max)
    half_extent = 0.5 * span.max() * (1.0 + margin_fraction)
    return (
        center[0] - half_extent,
        center[0] + half_extent,
        center[1] - half_extent,
        center[1] + half_extent,
    )


def plot_triangle_group(ax, group: TriangleGroup, field: str, text_scale: float) -> None:
    style = ROLE_STYLE[group.role]
    color = style["color"]
    marker = style["marker"]
    coords = group.coords(field)
    if coords.shape[0] < 2:
        raise ValueError(f"Need at least two nodes to plot group {group.name}")
    if coords.shape[0] == 2:
        path = coords
    else:
        path = np.vstack([coords, coords[0]])

    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0, zorder=2)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=70,
        marker=marker,
        facecolors="white",
        edgecolors=color,
        linewidths=1.8,
        zorder=3,
    )

    for node in group.nodes:
        point = getattr(node, field)
        dx = text_scale * (1.0 + 0.20 * node.index)
        dy = text_scale * style["label_sign"] * (1.0 - 0.12 * node.index)
        ax.text(
            point[0] + dx,
            point[1] + dy,
            node.short_label,
            color=color,
            fontsize=10,
            family="monospace",
            ha="left",
            va="center",
            zorder=4,
        )


def plot_panel(ax, groups: list[TriangleGroup], field: str, title: str) -> None:
    xmin, xmax, ymin, ymax = panel_limits(groups, field)
    text_scale = 0.03 * max(xmax - xmin, ymax - ymin)

    for group in groups:
        plot_triangle_group(ax, group, field, text_scale)

    ax.set_title(title)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def build_figure(kind: str, groups: dict[str, TriangleGroup]) -> plt.Figure:
    if kind in {"pair", "shared_edge"}:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        pair_groups = [groups["element1"], groups["element2"]]
        if kind == "pair":
            left_title = "reference triangles"
            right_title = "current triangles"
        else:
            left_title = "reference shared edge"
            right_title = "current shared edge"
        plot_panel(axes[0], pair_groups, "ref", left_title)
        plot_panel(axes[1], pair_groups, "pos", right_title)
        fig.suptitle("Element-pair ghost-node geometry", fontsize=16)
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    pre_groups = [groups["preFlipSelfGhost"], groups["preFlipPartnerGhost"]]
    post_groups = [groups["postFlipSelfGhost"], groups["postFlipPartnerGhost"]]

    plot_panel(axes[0, 0], pre_groups, "ref", "pre-flip reference triangles")
    plot_panel(axes[0, 1], pre_groups, "pos", "pre-flip current triangles")
    plot_panel(axes[1, 0], post_groups, "ref", "post-flip reference triangles")
    plot_panel(axes[1, 1], post_groups, "pos", "post-flip current triangles")

    fig.suptitle("Element-pair ghost-node geometry", fontsize=16)
    fig.tight_layout()
    return fig


def plot_debug_block(debug_block: str = DEBUG_BLOCK, show: bool = True) -> plt.Figure:
    kind, groups = parse_debug_block(debug_block)
    print_triangle_summary(kind, groups)
    fig = build_figure(kind, groups)
    if show:
        plt.show()
    return fig


def main() -> None:
    plot_debug_block(DEBUG_BLOCK, show=True)


if __name__ == "__main__":
    main()
