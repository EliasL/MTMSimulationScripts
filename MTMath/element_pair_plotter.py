from __future__ import annotations

from dataclasses import dataclass
import re

import matplotlib.pyplot as plt
import numpy as np


DEBUG_BLOCK = """
 libc++abi: terminating due to uncaught exception of type std::runtime_error: Reduction exploded in Mesh::updateElementsForces.
 
 minimization:
 nrMinItterations: 168
 nrMinFunctionCalls: 333
 load: 0.65275
 loadSteps: 50276
 
 element:
 eIndex: 10
 m3Nr: 149 red_quadrant: 4
 thetaElastic: 1.26566 referenceTheta: -1.53514 thetaTotal: -0.269483
 F:
   -0.301787   0.0034085
     1.22581 -0.00759409
 F_P:
  1 -0
  1  1
 F_E:
  0.469548 -0.890178
   1.16227  0.176925
 C:
    1.59368 -0.0103375
 -0.0103375 6.9288e-05
 C_R:
  6.9288e-05 1.36059e-05
 1.36059e-05   0.0513589
 G:
  0.82372 0.212346
 0.212346  1.57135
 M_e:
   1   0
 149   1
 M_l:
  -0   1
  -1 149
 P:
      268617 3.67937e+07
      120557 2.53016e+07
 sigma:
     1.43 0.421727
 0.421727  2.13815
 ghost[0]: refId=55 id=(5,1) pShift=(0,0) pos=(5.99003, 1.13878) ref=(-0.5, 0.5) u=(6.49003, 0.638779)
 ghost[1]: refId=6 id=(6,0) pShift=(0,0) pos=(5.98662, 1.14637) ref=(-0.5, -0.5) u=(6.48662, 1.64637)
 ghost[2]: refId=105 id=(5,2) pShift=(0,0) pos=(5.68484, 2.37218) ref=(0.5, -0.5) u=(5.18484, 2.87218)
 realNodeRefs:
 realNodeRef[0]: refId=55 pos=(5.99003, 1.13878) ref=(5, 1) u=(0.990033, 0.138779)
 realNodeRef[1]: refId=6 pos=(5.98662, 1.14637) ref=(6, 0) u=(-0.0133754, 1.14637)
 realNodeRef[2]: refId=105 pos=(5.68484, 2.37218) ref=(5, 2) u=(0.684838, 0.37218)
 
 lastFlipDebug:
 element=10 with partner 111
 minIterationsAtFlip: 168
 minFunctionCallsAtFlip: 332
 deltaMinIterationsSinceFlip: 0
 deltaMinFunctionCallsSinceFlip: 1
 applied_F_P:
  1 -1
 -0  1
 thetaElasticBefore: 1.29587
 thetaElasticAfter: 1.26566
 thetaElasticDelta: -0.0302178
 oldAnchor: (0, 0)
 newAnchor: (0, 0)
 postFlipSelfGhost:
 postFlipSelfGhost[0]: refId=55 id=(5,1) pShift=(0,0) pos=(5.99003, 1.13878) ref=(-0.5, 0.5) u=(6.49003, 0.638779)
 postFlipSelfGhost[1]: refId=6 id=(6,0) pShift=(0,0) pos=(5.98662, 1.14637) ref=(-0.5, -0.5) u=(6.48662, 1.64637)
 postFlipSelfGhost[2]: refId=105 id=(5,2) pShift=(0,0) pos=(5.68484, 2.37218) ref=(0.5, -0.5) u=(5.18484, 2.87218)
 
 partnerElement:
 eIndex: 111
 m3Nr: 1 red_quadrant: 2
 thetaElastic: 1.21118 referenceTheta: -1.60063 thetaTotal: -0.389449
 F:
 -0.301787  -1.67028
   1.22581  0.330307
 F_P:
  1 -0
  1  1
 F_E:
  0.362664 -0.783295
   1.02283   0.31637
 C:
  1.59368 0.908962
 0.908962  2.89894
 C_R:
  1.59368 0.684716
 0.684716  2.67469
 G:
     1.1777 -0.0395182
 -0.0395182   0.713641
 M_e:
  1 -1
  0  1
 M_l:
  1  1
  0 -1
 P:
  4.01041 -14.2365
  19.0149  -3.5635
 sigma:
  -1.93092    0.1995
    0.1995 -0.883487
 postFlipPartnerGhost:
 postFlipPartnerGhost[0]: refId=56 id=(6,1) pShift=(0,0) pos=(7.35512, 2.04187) ref=(0.5, -0.5) u=(6.85512, 2.54187)
 postFlipPartnerGhost[1]: refId=6 id=(6,0) pShift=(0,0) pos=(5.98662, 1.14637) ref=(-0.5, 0.5) u=(6.48662, 0.646373)
 postFlipPartnerGhost[2]: refId=105 id=(5,2) pShift=(0,0) pos=(5.68484, 2.37218) ref=(0.5, 0.5) u=(5.18484, 1.87218)
 preFlipSelfGhost:
 preFlipSelfGhost[0]: refId=6 id=(6,0) pShift=(0,0) pos=(6.49119, 1.04006) ref=(-0.5, -0.5) u=(6.99119, 1.54006)
 preFlipSelfGhost[1]: refId=55 id=(5,1) pShift=(0,0) pos=(5.60101, 1.21699) ref=(-0.5, 0.5) u=(6.10101, 0.71699)
 preFlipSelfGhost[2]: refId=56 id=(6,1) pShift=(0,0) pos=(6.85385, 2.06289) ref=(0.5, -0.5) u=(6.35385, 2.56289)
 preFlipPartnerGhost:
 preFlipPartnerGhost[0]: refId=105 id=(5,2) pShift=(0,0) pos=(6.07056, 2.37926) ref=(0.5, 0.5) u=(5.57056, 1.87926)
 preFlipPartnerGhost[1]: refId=55 id=(5,1) pShift=(0,0) pos=(5.60101, 1.21699) ref=(-0.5, 0.5) u=(6.10101, 0.71699)
 preFlipPartnerGhost[2]: refId=56 id=(6,1) pShift=(0,0) pos=(6.85385, 2.06289) ref=(0.5, -0.5) u=(6.35385, 2.56289)
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
