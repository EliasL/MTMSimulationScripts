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
  in the union of four central quadrants (an expanded fundamental region). The
  multiplicity at that depth counts distinct shortest generator words, independent
  of how many quadrant labels contain their endpoints.
"""

import argparse
from collections import deque
from functools import lru_cache
from typing import Iterator, Tuple, List, Set
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import MaxNLocator

from MTMath.poincareEnergy import drawPoincareGrid

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


def iter_shear_paths_by_depth(
    depth_max: int,
    gens: List[np.ndarray],
) -> Iterator[np.ndarray]:
    """Yield every potentially shortest generator word, one depth at a time.

    Immediate generator/inverse pairs are omitted because a word containing
    one cannot be shortest.  Matrices are deliberately not deduplicated:
    different words are the distinct paths whose multiplicity we want.
    """
    identity_key = canonicalize_psl(np.eye(2, dtype=int))
    inverse_indices = []
    for generator in gens:
        matches = [
            index
            for index, candidate in enumerate(gens)
            if canonicalize_psl(matmul_int(generator, candidate)) == identity_key
        ]
        if len(matches) != 1:
            raise ValueError("Generators must have exactly one inverse in the set.")
        inverse_indices.append(matches[0])

    generators = np.asarray(gens, dtype=int)
    matrices = np.eye(2, dtype=int)[None, ...]
    previous_indices = np.full(1, -1, dtype=int)
    generator_indices = np.arange(len(generators))
    for depth in range(depth_max + 1):
        yield matrices
        if depth == depth_max:
            break
        products = matrices[:, None, ...] @ generators[None, ...]
        allowed = generator_indices[None, :] != np.asarray(inverse_indices)[
            previous_indices, None
        ]
        if depth == 0:
            allowed[:] = True
        matrices = products[allowed]
        previous_indices = np.broadcast_to(
            generator_indices,
            allowed.shape,
        )[allowed]


def shortest_path_fields(
    XY: np.ndarray,
    depth_max: int,
    gens: List[np.ndarray],
    *,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify pixels and count distinct shortest unit-generator words.

    Each word is counted once if its transformed metric lies anywhere in the
    union of the four central quadrants at the first successful BFS depth.
    Quadrant overlaps therefore do not inflate the path multiplicity.
    """
    n_points = len(XY)
    labels = np.full(n_points, -1, dtype=int)
    depths = np.full(n_points, -1, dtype=int)
    multiplicities = np.full(n_points, -1, dtype=int)
    valid = np.sum(XY * XY, axis=1) < 1.0 - 1e-12
    multiplicities[valid] = 0

    z = XY[valid, 0] + 1j * XY[valid, 1]
    w = 1j * (1.0 + z) / (1.0 - z)
    c22 = 1.0 / w.imag
    c12 = w.real * c22
    c11 = (1.0 + c12 * c12) / c22
    valid_indices = np.flatnonzero(valid)

    for depth, matrices in enumerate(iter_shear_paths_by_depth(depth_max, gens)):
        unresolved = np.flatnonzero(depths[valid_indices] < 0)
        if unresolved.size == 0:
            break
        matrices = matrices.astype(float)
        depth_batch_size = min(
            batch_size,
            max(1, 2_000_000 // len(matrices)),
        )
        p, q = matrices[:, 0, 0], matrices[:, 0, 1]
        r, s = matrices[:, 1, 0], matrices[:, 1, 1]

        for start in range(0, unresolved.size, depth_batch_size):
            local = unresolved[start : start + depth_batch_size]
            a = c11[local, None]
            b = c12[local, None]
            c = c22[local, None]
            transformed_11 = a * p * p + 2.0 * b * p * r + c * r * r
            transformed_22 = a * q * q + 2.0 * b * q * s + c * s * s
            transformed_12 = a * p * q + b * (p * s + q * r) + c * r * s
            hits = (
                np.abs(transformed_12)
                <= 0.5 * np.minimum(transformed_11, transformed_22) + 1e-12
            )
            reached = np.any(hits, axis=1)
            if not np.any(reached):
                continue

            reached_local = local[reached]
            reached_global = valid_indices[reached_local]
            first_columns = np.argmax(hits[reached], axis=1)
            rows = np.flatnonzero(reached)
            first_11 = transformed_11[rows, first_columns]
            first_22 = transformed_22[rows, first_columns]
            first_12 = transformed_12[rows, first_columns]
            labels[reached_global] = np.where(
                first_11 <= first_22,
                np.where(first_12 >= 0.0, 0, 1),
                np.where(first_12 >= 0.0, 2, 3),
            )
            depths[reached_global] = depth
            multiplicities[reached_global] = np.sum(hits[reached], axis=1)

    return labels, depths, multiplicities

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
def plasticReduction(C: np.ndarray) -> Tuple[np.ndarray, int, int]:
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

@lru_cache(maxsize=None)
def poincare_grid_segments(depth: int):
    """Build a Poincare grid once and return it as compact line segments."""
    fig, ax = plt.subplots()
    drawPoincareGrid(ax=ax, grid_size=2, depth=depth)
    segments = tuple(
        np.column_stack((line.get_xdata() - 1, line.get_ydata() - 1))
        for line in ax.lines
    )
    linestyles = tuple(line.get_linestyle() for line in ax.lines)
    plt.close(fig)
    return segments, linestyles

def render_disk_map(XY: np.ndarray, A: np.ndarray, nx: int, out: Path,
                    add_colorbar: bool=False, cbar_label: str="",
                    unresolved_label: str | None=None,
                    cap_at: int | None=None,
                    grid_depth: int=7):
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
    unresolved = (Z == 0) if unresolved_label is not None else np.zeros_like(Z, dtype=bool)
    if unresolved_label is not None:
        Z[unresolved] = np.nan
    if cap_at is not None:
        Z = np.minimum(Z, cap_at)

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(
        Z,
        origin="lower",
        extent=(-1,1,-1,1),
        interpolation="nearest",
        vmin=1 if unresolved_label is not None else None,
        vmax=cap_at,
    )
    if unresolved_label is not None:
        unresolved_overlay = np.zeros((*Z.shape, 4))
        unresolved_overlay[unresolved] = to_rgba("#e6e6e6")
        ax.imshow(
            unresolved_overlay,
            origin="lower",
            extent=(-1,1,-1,1),
            interpolation="nearest",
        )
        ax.legend(
            handles=[Patch(facecolor="#e6e6e6", label=unresolved_label)],
            loc="lower right",
            fontsize=7,
            framealpha=0.95,
        )
    grid_segments, grid_linestyles = poincare_grid_segments(grid_depth)
    ax.add_collection(
        LineCollection(
            grid_segments,
            colors="#d0d0d0",
            alpha=0.32,
            linewidths=0.18,
            linestyles=grid_linestyles,
            zorder=2,
        )
    )
    ax.add_artist(Circle((0,0), 1.0, fill=False))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel(r"$x_p$"); ax.set_ylabel(r"$y_p$")
    ax.set_aspect("equal", adjustable="box")

    if add_colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cap_at is None:
            cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            first_tick = 1 if unresolved_label is not None else 0
            ticks = list(range(first_tick, cap_at + 1))
            cb.set_ticks(ticks)
            cb.set_ticklabels([str(tick) for tick in ticks])
        if cbar_label:
            cb.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
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

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
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
    ap.add_argument("--class-depth", type=int, default=12, help="BFS max word length")
    ap.add_argument("--out", type=str, default="Plots/poincare_direct_psl.png",
                    help="base output file; depth map gets *_depth.png, multiplicity gets *_mult.png")
    args = ap.parse_args()

    gens = pick_generators(args.gens)
    out = Path(args.out)
    out_depth = out.with_name(out.stem + "_depth.png")
    out_mult  = out.with_name(out.stem + "_mult.png")

    if args.mode == "disk":
        # Uniform grid on [-1,1]^2 (mask outside the unit disk)
        xs = np.linspace(-1.0, 1.0, args.nx)
        ys = np.linspace(-1.0, 1.0, args.nx)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        XY = np.vstack([X.ravel(), Y.ravel()]).T

        labels, depths, mults = shortest_path_fields(
            XY,
            args.class_depth,
            gens,
        )

        render_disk_map(
            XY, labels, args.nx, out,
            add_colorbar=False,
            grid_depth=args.class_depth,
        )
        render_disk_map(
            XY, depths, args.nx, out_depth,
            add_colorbar=True,
            cbar_label="Depth",
            grid_depth=args.class_depth,
        )
        render_disk_map(
            XY,
            mults,
            args.nx,
            out_mult,
            add_colorbar=True,
            cbar_label="Shortest-path multiplicity",
            cap_at=2,
            grid_depth=args.class_depth,
        )


if __name__ == "__main__":
    main()
