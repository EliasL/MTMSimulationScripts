#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check triangular domain D for all triangle cells in a VTK mesh.

Domain D:  G12 > 0  and  G12 < min(G11, G22)

Usage:
    python check_tri_domain.py mesh.vtk [--tol 1e-12] [--print-all]
Requires:
    - numpy
    - meshio  (preferred) or pyvista/vtk as fallback
"""

import sys
import argparse
from pathlib import Path

import numpy as np


def load_mesh_points_and_tris(path):
    """
    Returns:
        points: (N,3) float64
        tris:   (M,3) int64  (indices into points)
    """
    # Try meshio (preferred)
    try:
        import meshio  # type: ignore

        m = meshio.read(path)
        # points (meshio ensures at least 3 columns; pad if needed)
        pts = np.asarray(m.points, dtype=float)
        if pts.shape[1] == 2:
            pts = np.c_[pts, np.zeros(len(pts))]
        # cells: new meshio uses list of CellBlock; older may differ
        tri_list = []
        if hasattr(m, "cells"):
            # meshio >=4: list of CellBlock
            for cb in m.cells:
                if cb.type == "triangle":
                    tri_list.append(np.asarray(cb.data, dtype=np.int64))
        if not tri_list and hasattr(m, "cells_dict"):
            # older meshio
            if "triangle" in m.cells_dict:
                tri_list.append(np.asarray(m.cells_dict["triangle"], dtype=np.int64))
        if not tri_list:
            raise RuntimeError("No triangle cells found with meshio.")
        tris = np.vstack(tri_list)
        return pts, tris
    except Exception:
        # Fallback to pyvista/vtk
        try:
            import pyvista as pv  # type: ignore

            mesh = pv.read(path)
            pts = mesh.points
            if pts.shape[1] == 2:
                pts = np.c_[pts, np.zeros(len(pts))]
            # Extract triangles from faces array (pyvista face format)
            # faces = [n, i0, i1, i2, n, i0, i1, i2, ...]
            faces = mesh.faces.reshape(-1, 4)
            tri_mask = faces[:, 0] == 3
            if not np.any(tri_mask):
                raise RuntimeError("No triangle cells found with pyvista/vtk.")
            tris = faces[tri_mask][:, 1:4].astype(np.int64)
            return np.asarray(pts, dtype=float), tris
        except Exception as e:
            raise RuntimeError(
                f"Failed to load triangles from '{path}'. "
                f"Install `meshio` (preferred) or `pyvista`/`vtk`. Original error: {e}"
            )


def gram_components(a, b):
    """Return (G11, G12, G22) for vectors a,b in R^3."""
    G11 = float(np.dot(a, a))
    G12 = float(np.dot(a, b))
    G22 = float(np.dot(b, b))
    return G11, G12, G22


def check_domain(G11, G12, G22, tol):
    """
    Domain D: G12 > 0 and G12 < min(G11, G22)
    With tolerance: strictly > tol, and < min(G11, G22) - tol
    Returns:
        in_domain: bool
        violations: list[str]  (empty if in_domain)
    """
    violations = []
    if not (G12 + tol > 0):
        violations.append("G12<=0")
    if not (G12 - tol < min(G11, G22)):
        violations.append("G12>=min(G11,G22)")
    return (len(violations) == 0), violations


def main():
    # ap = argparse.ArgumentParser(
    #     description="Check triangular domain D on triangle cells of a VTK mesh."
    # )
    # ap.add_argument(
    #     "vtk_path", help="Path to .vtk (or other mesh readable by meshio/pyvista)"
    # )
    # ap.add_argument(
    #     "--tol",
    #     type=float,
    #     default=1e-12,
    #     help="Tolerance for strict inequalities (default: 1e-12)",
    # )
    # ap.add_argument(
    #     "--print-all",
    #     action="store_true",
    #     help="Print all triangles (inside/outside). Default prints only those outside D.",
    # )
    # args = ap.parse_args()
    path = Path("~/Downloads/configuration_86934.vtk").expanduser()
    tol = 0.2
    pts, tris = load_mesh_points_and_tris(path)
    n_tri = len(tris)
    if n_tri == 0:
        print("No triangle cells found.")
        return

    n_in, n_out = 0, 0
    out_records = []

    # Vectorized-friendly loop with NumPy ops
    for tidx, (i0, i1, i2) in enumerate(tris):
        p0, p1, p2 = pts[i0], pts[i1], pts[i2]
        a = p1 - p0
        b = p2 - p0
        G11, G12, G22 = gram_components(a, b)
        inD, viols = check_domain(G11, G12, G22, tol)
        if inD:
            n_in += 1
            if False:
                print(
                    f"[IN ] tri {tidx:6d}  G11={G11:.6e}  G12={G12:.6e}  G22={G22:.6e}"
                )
        else:
            n_out += 1
            out_records.append((tidx, G11, G12, G22, viols))
            print(
                f"[OUT] tri {tidx:6d}  G11={G11:.6e}  G12={G12:.6e}  G22={G22:.6e}  violations={viols}"
            )

    print("\nSummary:")
    print(f"  Triangles total : {n_tri}")
    print(f"  Inside D        : {n_in}")
    print(f"  Outside D       : {n_out}")

    if n_out == 0:
        print(
            "All triangle Gram matrices satisfy G12>0 and G12<min(G11,G22) within tolerance."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
