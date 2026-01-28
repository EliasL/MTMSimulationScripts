#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DIRECT PSL/BFS with word-length and multiplicity plots
--------------------------------------------------------------------
Pedagogical notes:
- We work on the isochoric strain manifold det C = 1.
- Points in the Poincaré disk (x,y) are mapped to the upper half-plane w = u + i v
  via a Cayley transform, and then to a metric C(u,v) s.t. det C = 1.
- Lattice symmetries act by congruence: C -> W^T C W, with W in PSL(2,Z).
- A BFS over generators finds the minimal word length (depth) that lands a given C
  in one of four central quadrants (an expanded fundamental region). The multiplicity
  at that minimal depth counts how many distinct quadrants are reached by words
  of the same minimal length.
"""

import argparse
from collections import deque
from typing import Tuple, List, Set
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

# ---------- Small linear algebra helpers ----------

def matmul_int(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Integer-safe matrix product for group words (avoid dtype float drift)."""
    C = A @ B
    return np.array(C, dtype=int)

def canonicalize_psl(M: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Canonical representative in PSL(2,Z): M and -M are identified.
    We enforce c>0 or (c==0 and d>0) on the lower row (c,d).
    """
    a, b, c, d = int(M[0,0]), int(M[0,1]), int(M[1,0]), int(M[1,1])
    if c < 0 or (c == 0 and d < 0):
        a, b, c, d = -a, -b, -c, -d
    return (a, b, c, d)

# ---------- Generators of PSL(2,Z) and relatives ----------

# Classic modular generators
T    = np.array([[1,  1],[0,  1]], dtype=int)
Tinv = np.array([[1, -1],[0,  1]], dtype=int)
S    = np.array([[0, -1],[1,  0]], dtype=int)

# Simple shear set
U, Uinv = T, Tinv
V    = np.array([[1, 0],[1, 1]], dtype=int)
Vinv = np.array([[1, 0],[-1, 1]], dtype=int)


def pick_generators(kind: str) -> List[np.ndarray]:
    """Choose the generating set used by the BFS."""
    if kind == "st":
        return [T, Tinv, S]
    else:
        return [U, Uinv, V, Vinv]

# ---------- Word enumeration (BFS) ----------

def psl_bfs_words_with_depth(depth_max: int, gens: List[np.ndarray]) -> List[Tuple[np.ndarray,int]]:
    """
    Breadth-first search on the Cayley graph up to depth_max.
    Returns list of (word_matrix, word_length).
    """
    I = np.eye(2, dtype=int)
    words: List[Tuple[np.ndarray,int]] = [(I, 0)]
    seen = {canonicalize_psl(I): 0}
    Q = deque([(I, 0)])
    while Q:
        g, d = Q.popleft()
        if d == depth_max:
            continue
        for G in gens:
            h = matmul_int(g, G)
            key = canonicalize_psl(h)
            if key in seen:
                continue
            seen[key] = d + 1
            words.append((h, d + 1))
            Q.append((h, d + 1))
    return words

# ---------- Geometry of the four central quadrants ----------


def label_and_depth_direct_psl(C: np.ndarray, words_with_depth: List[Tuple[np.ndarray,int]]) -> Tuple[int, int]:
    """
    Minimal reduction: scan words in BFS order; the first hit gives the quadrant label
    and minimal word length (depth).
    """
    for W, d in words_with_depth:
        Cw = W.T @ C @ W
        for lab in (0,1,2,3):
            if in_quadrant_geom(Cw, lab):
                return lab, d
    return -1, -1

def U_m(m:int):
    return np.array([[1,  m],[0,  1]], dtype=int)
def V_m(m:int):
    return np.array([[1,  0],[m,  1]], dtype=int)

def in_quadrant_geom(C: np.ndarray, label: int, tol: float = 1e-12) -> bool:
    """
    Test if C lies in one of four central 'quadrants' defined by linear constraints
    in (C11, C22, C12). These boundaries project to geodesics (diameters/orthogonal arcs)
    on the Poincaré disk.
    """
    C11, C22, C12 = C[0,0], C[1,1], C[0,1]
    if label == 0:
        return (C11 > tol) and (C11 <= C22 + tol) and (C12 >= -tol) and (C12 <= 0.5*C11 + tol)
    if label == 1:
        return (C11 > tol) and (C11 <= C22 + tol) and (C12 <=  tol) and (C12 >= -0.5*C11 - tol)
    if label == 2:
        return (C22 > tol) and (C22 <= C11 + tol) and (C12 >= -tol) and (C12 <= 0.5*C22 + tol)
    if label == 3:
        return (C22 > tol) and (C22 <= C11 + tol) and (C12 <=  tol) and (C12 >= -0.5*C22 - tol)
    return False
def elasticReduction(C: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Minimal reduction: scan words in BFS order; the first hit gives the quadrant label
    and minimal word length (depth).
    """
    max_depth=100
    C = C.copy()
    for d in range(0,max_depth+1):
        for lab in (0,1,2,3):
            # Is C in elastic domain?
            if in_quadrant_geom(C, lab):
                return C, lab, d
        if C[0,0]<C[1,1]:
            W = U_m(np.sign(-C[0,1]/C[0,0]))
        else:
            W = V_m(np.sign(-C[0,1]/C[1,1]))
        C = W.T@C@W
    return C, -1, -1

def multiplicity_at_min_depth(C: np.ndarray, words_with_depth: List[Tuple[np.ndarray,int]]) -> Tuple[int, int]:
    """
    Among words achieving the minimal depth, count how many distinct quadrants are reached.
    Returns (minimal_depth, multiplicity).
    """
    min_depth = -1
    labels_at_min = []
    for W, d in words_with_depth:
        Cw = W.T @ C @ W
        found = any(in_quadrant_geom(Cw, lab) for lab in (0,1,2,3))
        if not found:
            continue
        if min_depth == -1:
            min_depth = d
        if d > min_depth:
            break
        for lab in (0,1,2,3):
            if in_quadrant_geom(Cw, lab):
                labels_at_min.append(lab)
    if min_depth == -1:
        return -1, 0
    return min_depth, len(labels_at_min)

# ---------- Disk / upper half-plane mapping and strain construction ----------

def disk_to_upper(x: float, y: float) -> Tuple[float, float]:
    """
    Cayley transform: unit disk z=x+iy → upper half-plane w=u+iv, v>0:
        w = i * (1+z)/(1-z)
    """
    z = x + 1j*y
    w = 1j*(1.0 + z)/(1.0 - z)
    return float(np.real(w)), float(np.imag(w))

def upper_to_C(u: float, v: float) -> np.ndarray:
    """
    Map (u,v) in the upper half-plane to a metric C with det C = 1:
        C22 = 1/v,  C12 = u*C22,  C11 = (1 + C12^2)/C22.
    This ensures C11*C22 - C12^2 = 1 exactly (up to float tolerance).
    """
    if v < 1e-12:
        v = 1e-12
    C22 = 1.0 / v
    C12 = u * C22
    C11 = (1.0 + C12*C12) / C22
    return np.array([[C11, C12],[C12, C22]], dtype=float)

def F_simple(gamma: float, theta: float) -> np.ndarray:
    """Rotated simple shear: F = R^T [[1, gamma],[0,1]] R."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[ c, -s],[ s,  c]], dtype=float)
    F0 = np.array([[1.0, gamma],[0.0, 1.0]], dtype=float)
    return R.T @ F0 @ R

def F_pure(gamma: float, theta: float) -> np.ndarray:
    """Rotated isochoric pure stretch: λ1=1+γ, λ2=1/λ1, then rotate by θ."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[ c, -s],[ s,  c]], dtype=float)
    lam1 = 1.0 + gamma
    lam2 = 1.0 / lam1
    F0 = np.array([[lam1, 0.0],[0.0, lam2]], dtype=float)
    return R.T @ F0 @ R

# ---------- Plotting utilities ----------

def render_disk_map(XY: np.ndarray, A: np.ndarray, nx: int, out: Path, title: str,
                    add_colorbar: bool=False, cbar_label: str=""):
    """
    Render a field A defined on a regular grid inside the unit disk (mask outside).
    """
    X = XY[:,0].reshape(nx, nx)
    Y = XY[:,1].reshape(nx, nx)
    Z = A.reshape(nx, nx).astype(float)
    mask = (X*X + Y*Y) > 1.0
    Z[mask] = np.nan
    if add_colorbar:
        Z[Z < 0] = np.nan

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(Z, origin="lower", extent=(-1,1,-1,1), interpolation="nearest")
    ax.add_artist(Circle((0,0), 1.0, fill=False))
    ax.set_title(title); ax.set_xlabel("x (disk)"); ax.set_ylabel("y (disk)")
    ax.set_aspect("equal", adjustable="box")

    if add_colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if cbar_label:
            cb.set_label(cbar_label)

    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved figure to {out}")

def render_scatter_map(XY: np.ndarray, A: np.ndarray, out: Path, title: str,
                       add_colorbar: bool=False, cbar_label: str=""):
    """
    Scatter-plot version for parametric samples (simple shear / pure stretch).
    """
    fig, ax = plt.subplots(figsize=(6,6))
    if add_colorbar:
        mask = (A >= 0)
        sc = ax.scatter(XY[mask,0], XY[mask,1], c=A[mask], s=4)
    else:
        sc = ax.scatter(XY[:,0], XY[:,1], c=A, s=4)
    ax.add_artist(Circle((0,0), 1.0, fill=False))
    ax.set_title(title); ax.set_xlabel("x (disk)"); ax.set_ylabel("y (disk)")
    ax.set_aspect("equal", adjustable="box")

    if add_colorbar:
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if cbar_label:
            cb.set_label(cbar_label)

    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved figure to {out}")

# ---------- Driver ----------

def main():
    ap = argparse.ArgumentParser(description="DIRECT PSL/BFS with depth & multiplicity plots")
    ap.add_argument("--gens", type=str, default="shear", choices=["st","shear"],
                    help="generator set: 'st' uses {T,Tinv,S}, 'shear' uses {U,Uinv,V,Vinv}")
    ap.add_argument("--mode", type=str, default="disk", choices=["disk","simple","pure"],
                    help="sampling: fill the disk grid, or sample simple shear / pure stretch")
    ap.add_argument("--nx", type=int, default=501, help="disk grid size (per axis)")
    ap.add_argument("--theta-n", type=int, default=181, help="number of rotation angles in [0,pi]")
    ap.add_argument("--gamma-n", type=int, default=361, help="number of shear values in [gmin,gmax]")
    ap.add_argument("--gamma-min", type=float, default=-6.0, help="min shear gamma")
    ap.add_argument("--gamma-max", type=float, default= 6.0, help="max shear gamma")
    ap.add_argument("--class-depth", type=int, default=6, help="BFS max word length")
    ap.add_argument("--out", type=str, default="Plots/poincare_direct_psl.png",
                    help="base output file; depth map gets *_depth.png, multiplicity gets *_mult.png")
    args = ap.parse_args()

    gens = pick_generators(args.gens)
    words_with_depth = psl_bfs_words_with_depth(args.class_depth, gens)

    out = Path(args.out)
    out_depth = out.with_name(out.stem + "_depth.png")
    out_mult  = out.with_name(out.stem + "_mult.png")

    if args.mode == "disk":
        # Uniform grid on [-1,1]^2 (mask outside the unit disk)
        xs = np.linspace(-1.0, 1.0, args.nx)
        ys = np.linspace(-1.0, 1.0, args.nx)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        XY = np.vstack([X.ravel(), Y.ravel()]).T

        labels = np.full(XY.shape[0], -1, dtype=int)
        depths = np.full(XY.shape[0], -1, dtype=int)
        mults  = np.full(XY.shape[0], -1, dtype=int)

        for k, (x, y) in enumerate(XY):
            if x*x + y*y >= 1.0 - 1e-12:
                continue
            # Disk → upper half-plane → metric C with det C = 1
            u, v = disk_to_upper(x, y)
            C = upper_to_C(u, v)

            # Minimal-depth classification and multiplicity
            lab, d = label_and_depth_direct_psl(C, words_with_depth)
            #lab, d = label_and_depth_elias(C)
            labels[k] = lab; depths[k] = d

            md, m = multiplicity_at_min_depth(C, words_with_depth)
            mults[k] = m

        ttl_lbl = f"DIRECT PSL (depth={args.class_depth}, gens={args.gens}) — disk"
        ttl_dep = f"Word length (BFS depth) — gens={args.gens}"
        ttl_mul = f"Multiplicity at minimal depth — gens={args.gens}"
        render_disk_map(XY, labels, args.nx, out, ttl_lbl, add_colorbar=False)
        render_disk_map(XY, depths, args.nx, out_depth, ttl_dep, add_colorbar=True, cbar_label="BFS word length")
        render_disk_map(XY, mults, args.nx, out_mult, ttl_mul, add_colorbar=True, cbar_label="Multiplicity")


if __name__ == "__main__":
    main()
