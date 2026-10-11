#!/usr/bin/env python3
"""Seed a non-reconnecting MTS2D dump from a full VTU snapshot.

This is intentionally restricted to an unreconnected periodic square mesh.
The template supplies serialization-only state and reference geometry; the VTU
supplies the real-node positions and stable plastic-well indices.
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import meshio
import numpy as np


NODE_START = re.compile(r"^\t{5}<value(?P<index>\d+)>\s*$")
ELEMENT_START = re.compile(r"^\t{4}<value(?P<index>\d+)>\s*$")
VALUE_LINE = re.compile(
    r"^(?P<indent>\s*)<value(?P<index>\d+)>.*</value(?P=index)>\s*$"
)


def scalar_cell_field(mesh: meshio.Mesh, name: str) -> np.ndarray:
    values = np.asarray(mesh.cell_data[name][0], dtype=float).reshape(-1)
    if len(values) != 80000:
        raise ValueError(f"Expected 80000 values for {name}, found {len(values)}.")
    return values


def real_node_positions(mesh: meshio.Mesh) -> np.ndarray:
    references = np.asarray(mesh.point_data["refIndex"], dtype=float).reshape(-1)
    rounded = np.rint(references).astype(int)
    if not np.allclose(references, rounded, rtol=0.0, atol=1e-10):
        raise ValueError("refIndex contains non-integer values.")
    if set(rounded) != set(range(40000)):
        raise ValueError("VTU does not contain exactly the expected real-node IDs.")

    positions = np.full((40000, 2), np.nan)
    for point_index, node_index in enumerate(rounded):
        if np.isnan(positions[node_index, 0]):
            positions[node_index] = mesh.points[point_index, :2]
    if not np.all(np.isfinite(positions)):
        raise ValueError("Could not recover every real-node position.")
    return positions


def replacement_value(line: str, value: float) -> str:
    match = VALUE_LINE.match(line.rstrip("\n"))
    if match is None:
        raise ValueError(f"Unexpected matrix value line: {line!r}")
    index = match.group("index")
    return f"{match.group('indent')}<value{index}>{value:.17g}</value{index}>\n"


def synthesize(template: Path, snapshot: Path, output: Path, load: float) -> None:
    mesh = meshio.read(snapshot)
    if [(block.type, block.data.shape) for block in mesh.cells] != [
        ("triangle", (80000, 3))
    ]:
        raise ValueError("Expected one 80000-triangle cell block.")
    positions = real_node_positions(mesh)
    m3_values = scalar_cell_field(mesh, "nrm3")
    m3_rounded = np.rint(m3_values).astype(int)
    if not np.allclose(m3_values, m3_rounded, rtol=0.0, atol=1e-10):
        raise ValueError("nrm3 contains non-integer values.")

    load_steps_float = (load - 0.15) / 1e-5 + 1.0
    load_steps = int(round(load_steps_float))
    if abs(load_steps_float - load_steps) > 1e-8:
        raise ValueError("Load does not lie on the configured load grid.")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    in_nodes = False
    in_elements = False
    node_index: int | None = None
    element_index: int | None = None
    node_vector: str | None = None
    in_current_deformation = False
    counts = {"nodes": 0, "elements": 0, "load": 0, "load_steps": 0}

    with gzip.open(template, "rt") as source, gzip.open(output, "wt") as target:
        for line in source:
            stripped = line.strip()

            if line.startswith("\t\t\t<nodes>"):
                in_nodes = True
            elif line.startswith("\t\t\t</nodes>"):
                in_nodes = False
                node_index = None
            elif line.startswith("\t\t\t<elements"):
                in_elements = True
            elif line.startswith("\t\t\t</elements>"):
                in_elements = False
                element_index = None

            if in_nodes:
                match = NODE_START.match(line.rstrip("\n"))
                if match is not None:
                    node_index = int(match.group("index"))
                if stripped in ("<m_pos>", "<m_u>"):
                    node_vector = stripped[1:-1]
                elif stripped in ("</m_pos>", "</m_u>"):
                    node_vector = None
                elif node_vector is not None and VALUE_LINE.match(line.rstrip("\n")):
                    if node_index is None or node_index >= len(positions):
                        raise ValueError("Invalid node index in template dump.")
                    component = int(VALUE_LINE.match(line.rstrip("\n")).group("index"))
                    if node_vector == "m_pos":
                        value = positions[node_index, component]
                    else:
                        reference = (node_index % 200, node_index // 200)
                        value = positions[node_index, component] - reference[component]
                    line = replacement_value(line, value)
                    counts["nodes"] += 1

            if in_elements:
                match = ELEMENT_START.match(line.rstrip("\n"))
                if match is not None:
                    element_index = int(match.group("index"))
                if stripped.startswith(("<m3Nr>", "<pastM3Nr>", "<pastStepM3Nr>")):
                    if element_index is None or element_index >= len(m3_rounded):
                        raise ValueError("Invalid element index in template dump.")
                    tag = stripped.split(">", 1)[0][1:]
                    indent = line[: len(line) - len(line.lstrip())]
                    value = int(m3_rounded[element_index])
                    line = f"{indent}<{tag}>{value}</{tag}>\n"
                    counts["elements"] += 1

            if line.startswith("\t\t\t<load>"):
                line = f"\t\t\t<load>{load:.17g}</load>\n"
                counts["load"] += 1
            elif line.startswith("\t\t\t<loadSteps>"):
                line = f"\t\t\t<loadSteps>{load_steps}</loadSteps>\n"
                counts["load_steps"] += 1
            elif line.startswith("\t\t\t<currentDeformation>"):
                in_current_deformation = True
            elif line.startswith("\t\t\t</currentDeformation>"):
                in_current_deformation = False
            elif in_current_deformation and VALUE_LINE.match(line.rstrip("\n")):
                component = int(VALUE_LINE.match(line.rstrip("\n")).group("index"))
                values = (1.0, 0.0, load, 1.0)
                if component >= len(values):
                    raise ValueError("Unexpected current-deformation component.")
                line = replacement_value(line, values[component])
            elif line.startswith("\t\t\t<nr_elements_with_m3_change>"):
                line = "\t\t\t<nr_elements_with_m3_change>0</nr_elements_with_m3_change>\n"
            elif line.startswith("\t\t\t<nr_elements_with_m3_changeInStep>"):
                line = (
                    "\t\t\t<nr_elements_with_m3_changeInStep>0"
                    "</nr_elements_with_m3_changeInStep>\n"
                )

            target.write(line)

    expected = {
        "nodes": 40000 * 4,
        "elements": 80000 * 3,
        "load": 1,
        "load_steps": 1,
    }
    if counts != expected:
        output.unlink()
        raise RuntimeError(f"Replacement count mismatch: {counts}, expected {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--load", type=float, required=True)
    args = parser.parse_args()
    synthesize(args.template, args.snapshot, args.output, args.load)
    print(args.output)


if __name__ == "__main__":
    main()
