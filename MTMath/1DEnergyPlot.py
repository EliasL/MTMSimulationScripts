from contiPotential import ContiEnergy
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import os
from multiprocessing import Pool, cpu_count


# ---------- utilities ----------
def _writer_from_path(save_path, fps):
    """
    Choose the right Matplotlib writer based on file extension.
    Returns (writer_name, writer_kwargs)
    """
    ext = os.path.splitext(save_path)[1].lower()
    if ext in {".mp4", ".m4v"}:
        # Requires ffmpeg installed (macOS: `brew install ffmpeg`)
        return "ffmpeg", dict(
            fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
        )
    if ext in {".webm"}:
        return "ffmpeg", dict(fps=fps, codec="libvpx-vp9")
    if ext in {".gif"}:
        return "pillow", dict(fps=fps)
    if ext in {".mov"}:
        return "ffmpeg", dict(
            fps=fps, codec="prores_ks"
        )  # large files, editing-friendly
    # Fallback
    return "ffmpeg", dict(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])


def _chunk_indices(n, n_chunks):
    # even-ish splits of range(n)
    edges = np.linspace(0, n, n_chunks + 1, dtype=int)
    return [
        slice(edges[i], edges[i + 1])
        for i in range(n_chunks)
        if edges[i] < edges[i + 1]
    ]


def _forces_worker(args):
    """Worker for multiprocessing."""
    strain_chunk, dN_dX_chunk, X = args
    # Build x_series for this chunk (simple shear mapping)
    x_chunk = np.empty((len(strain_chunk), 3, 2), dtype=float)
    x_chunk[..., 0] = X[None, :, 0] + strain_chunk[:, None] * X[None, :, 1]
    x_chunk[..., 1] = X[None, :, 1]
    # Compute dN/dx from current coordinates for Eulerian forces
    dN_dx_chunk = dN_dx_from_coords(x_chunk)
    f_lag = ContiEnergy.lagrangian_forces_from_simpleShear(strain_chunk, dN_dX_chunk)
    f_eul = ContiEnergy.eulerian_forces_from_simpleShear(strain_chunk, dN_dx_chunk)
    return f_lag, f_eul, x_chunk


def plot_energy():
    strain = np.linspace(0.0, 1, 100)
    e = ContiEnergy.energy_from_simpleShear(strain)

    fig, ax = plt.subplots()
    ax.plot(strain, e, label="Energy")
    ax.set_xlabel(r"$\gamma$ (Strain)")
    ax.set_ylabel("Energy")
    ax.set_title("Energy in Simple Shear")
    ax.legend()
    fig.tight_layout()


def plot_forces():
    strain = np.linspace(0.0, 5, 1000)
    dN_dX = np.array([[-1, -1], [1, 0], [0, 1]])
    dN_dX = np.tile(dN_dX, (len(strain), 1, 1))
    forces = ContiEnergy.lagrangian_forces_from_simpleShear(strain, dN_dX)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        ax.plot(strain, forces[:, i, 0], label="$f_x$")
        ax.plot(strain, forces[:, i, 1], label="$f_y$", linestyle="--")
        ax.set_xlabel(r"$\gamma$ (Strain)")
        if i == 0:
            ax.set_ylabel("Force")
        ax.set_title(f"Node {i + 1}")
        ax.legend()
        ax.grid()
    fig.suptitle("Lagrangian Forces in Simple Shear")
    fig.tight_layout()


def dN_dx_from_coords(coords):
    """
    Compute dN/dx and area for a linear T3 element using only matrix multiplications.

    Parameters
    ----------
    coords : array_like, shape (..., 3, 2)
        Nodal coordinates [[x1, y1], [x2, y2], [x3, y3]].

    Returns
    -------
    dN_dx : ndarray, shape (..., 3, 2)
        Gradient of shape functions wrt x,y (row i = [∂N_i/∂x, ∂N_i/∂y]).
    area : ndarray, shape (...)
        Element area (positive scalar).
    """
    coords = np.asarray(coords)
    assert coords.shape[-2:] == (3, 2), "coords must have shape (..., 3, 2)"

    # Reference (natural) shape function derivatives wrt (ξ, η)
    dN_dxi = np.array(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=coords.dtype,
    )  # (3, 2)

    # Jacobian: J = dX/dξ = coords^T @ dN_dxi
    J = coords.swapaxes(-1, -2) @ dN_dxi  # (..., 2, 2)

    # Inverse Jacobian
    J_inv = np.linalg.inv(J)  # (..., 2, 2)

    # Transform shape function gradients: dN/dx = dN/dξ @ J^{-1}
    dN_dx = dN_dxi @ J_inv  # (..., 3, 2)

    return dN_dx


def plot_eulerian_forces(coords=None):
    strain = np.linspace(0.0, 5, 1000)
    if coords is None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (3, 2):
        raise ValueError("coords must have shape (3, 2)")
    # Compute dN_dx and area from coords
    dN_dx = dN_dx_from_coords(np.tile(coords, (len(strain), 1, 1)))
    forces = ContiEnergy.eulerian_forces_from_simpleShear(strain, dN_dx)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        ax.plot(strain, forces[:, i, 0], label=r"$f_x$")
        ax.plot(strain, forces[:, i, 1], label=r"$f_y$", linestyle="--")
        ax.set_xlabel(r"$\gamma$ (Strain)")
        if i == 0:
            ax.set_ylabel("Eulerian Force")
        ax.set_title(f"Node {i + 1}")
        ax.legend()
        ax.grid()
    fig.suptitle("Eulerian Forces in Simple Shear")
    fig.tight_layout()


# ---------- animation with auto writer + optional multiprocessing ----------
def animate_nodes_and_forces(
    coords_ref=None,
    strain=None,
    interval=20,
    arrow_scale=1.0,
    save_path=None,
    writer=None,  # "ffmpeg" | "pillow" | None (auto from save_path)
    dpi=150,
    n_procs=1,  # >1 enables multiprocessing over strain
):
    """
    Create animation of node motion + internal forces (PK1 on left, Cauchy on right).

    - MP4/WebM/GIF saving: decide by the extension of `save_path`.
    - Set n_procs>1 to parallelize force computations over the strain samples.
    """
    if coords_ref is None:
        coords_ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    if strain is None:
        strain = np.linspace(0.0, 5.0, 200)

    coords_ref = np.asarray(coords_ref, dtype=float)
    strain = np.asarray(strain, dtype=float)
    if coords_ref.shape != (3, 2):
        raise ValueError("coords_ref must have shape (3, 2)")

    # Reference gradients
    dN_dX = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    dN_dX_batched = np.tile(dN_dX, (len(strain), 1, 1))
    X = coords_ref

    # Compute forces + current coords (optionally parallel)
    if n_procs is None or n_procs < 1:
        n_procs = 1
    if n_procs > 1:
        n_procs = min(n_procs, cpu_count())
        slices = _chunk_indices(len(strain), n_procs)
        args = [(strain[s], dN_dX_batched[s], X) for s in slices]
        with Pool(processes=n_procs) as pool:
            results = pool.map(_forces_worker, args)
        f_lag = np.concatenate([r[0] for r in results], axis=0)
        f_eul = np.concatenate([r[1] for r in results], axis=0)
        x_series = np.concatenate([r[2] for r in results], axis=0)
    else:
        # Vectorized path (fast if ContiEnergy is vectorized)
        x_series = np.empty((len(strain), 3, 2), dtype=float)
        x_series[..., 0] = X[None, :, 0] + strain[:, None] * X[None, :, 1]
        x_series[..., 1] = X[None, :, 1]
        # Compute dN/dx for each frame from current coordinates
        dN_dx_series = dN_dx_from_coords(x_series)
        f_lag = ContiEnergy.lagrangian_forces_from_simpleShear(strain, dN_dX_batched)
        f_eul = ContiEnergy.eulerian_forces_from_simpleShear(strain, dN_dx_series)

    # Axis limits
    all_pts = np.vstack([x_series.reshape(-1, 2), X])
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    pad = 0.1 * max(
        xmax - xmin if xmax > xmin else 1.0, ymax - ymin if ymax > ymin else 1.0
    )
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    # Quiver scaling
    mag_ref = np.maximum(
        1e-12, np.percentile(np.linalg.norm(f_eul.reshape(-1, 2), axis=1), 95)
    )

    def scaled(F):
        return (arrow_scale / mag_ref) * F

    # Figure + artists
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Node Motion and Internal Forces under Simple Shear")

    for ax in (axL, axR):
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3)

    axL.set_title("Reference (Lagrangian)")
    axR.set_title("Current (Eulerian)")

    (triL,) = axL.plot(
        [X[0, 0], X[1, 0], X[2, 0], X[0, 0]],
        [X[0, 1], X[1, 1], X[2, 1], X[0, 1]],
        lw=1.0,
    )
    ptsL = axL.scatter(X[:, 0], X[:, 1], s=30)
    qL = axL.quiver(
        X[:, 0],
        X[:, 1],
        np.zeros(3),
        np.zeros(3),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.005,
    )

    (triR,) = axR.plot(
        [x_series[0, 0, 0], x_series[0, 1, 0], x_series[0, 2, 0], x_series[0, 0, 0]],
        [x_series[0, 0, 1], x_series[0, 1, 1], x_series[0, 2, 1], x_series[0, 0, 1]],
        lw=1.0,
    )
    ptsR = axR.scatter(x_series[0, :, 0], x_series[0, :, 1], s=30)
    qR = axR.quiver(
        x_series[0, :, 0],
        x_series[0, :, 1],
        np.zeros(3),
        np.zeros(3),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.005,
    )

    txt = fig.text(0.5, 0.02, "", ha="center", va="center")

    def update(k):
        FxL = scaled(f_lag[k, :, 0])
        FyL = scaled(f_lag[k, :, 1])
        qL.set_UVC(FxL, FyL)

        xk = x_series[k]
        triR.set_data(
            [xk[0, 0], xk[1, 0], xk[2, 0], xk[0, 0]],
            [xk[0, 1], xk[1, 1], xk[2, 1], xk[0, 1]],
        )
        ptsR.set_offsets(xk)
        FxR = scaled(f_eul[k, :, 0])
        FyR = scaled(f_eul[k, :, 1])
        qR.set_offsets(xk)
        qR.set_UVC(FxR, FyR)

        txt.set_text(rf"$\gamma$ = {strain[k]:.3f}")
        return triL, ptsL, qL, triR, ptsR, qR, txt

    fps = max(1, int(1000 / interval))
    anim = animation.FuncAnimation(
        fig, update, frames=len(strain), interval=interval, blit=False
    )

    if save_path is not None:
        if writer is None:
            writer, writer_kwargs = _writer_from_path(save_path, fps=fps)
        else:
            # respect explicit writer; sensible defaults
            writer_kwargs = dict(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi, **writer_kwargs)

    return anim


if __name__ == "__main__":
    plot_eulerian_forces()
    plot_forces()
    # animate_nodes_and_forces(
    #     save_path="simple_shear_nodes_forces.mp4",
    #     interval=30,
    #     n_procs=1,  # try >1 if ContiEnergy calls are expensive
    # )
    plt.show()
