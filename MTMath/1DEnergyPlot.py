import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MTMath.energyFunction import ContiEnergy
from MTMath.meshUtils import triangle_shape_grads_and_area
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
    # Compute dN/dx and area from current coordinates for Eulerian forces
    dN_dx_chunk, area_chunk = dN_dx_from_coords(x_chunk)
    f_lag = ContiEnergy.lagrangian_forces_from_simpleShear(strain_chunk, dN_dX_chunk)
    f_eul = ContiEnergy.eulerian_forces_from_simpleShear(
        strain_chunk, dN_dx_chunk, area=area_chunk
    )
    return f_lag, f_eul, x_chunk


def plot_energy():
    strain = np.linspace(-3, 3, 1000)
    e = ContiEnergy.energy_from_simpleShear(strain)

    fig, ax = plt.subplots()
    ax.plot(strain, e, label="Energy")
    ax.set_xlabel(r"$\gamma$ (Strain)")
    ax.set_ylabel("Energy")
    ax.set_title("Energy in Simple Shear")
    ax.legend()
    fig.tight_layout()


def plot_Lagrangian_forces():
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
    return triangle_shape_grads_and_area(coords)


def plot_eulerian_forces(coords=None):
    strain = np.linspace(0.0, 5, 1000)
    if coords is None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (3, 2):
        raise ValueError("coords must have shape (3, 2)")
    # Compute dN_dx and area from current coordinates under simple shear
    X = coords
    x_series = np.empty((len(strain), 3, 2), dtype=float)
    x_series[..., 0] = X[None, :, 0] + strain[:, None] * X[None, :, 1]
    x_series[..., 1] = X[None, :, 1]
    dN_dx, area = dN_dx_from_coords(x_series)
    forces = ContiEnergy.eulerian_forces_from_simpleShear(strain, dN_dx, area=area)

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


def plot_stress_tensor_components(strain=None, beta=-1 / 4, K=4, noise=1):
    """
    Plot all 2x2 components of PK1 (P) and Cauchy (sigma) stresses
    for simple shear in a single figure.
    """
    if strain is None:
        strain = np.linspace(-1.0, 5.0, 1000)
    strain = np.asarray(strain, dtype=float)
    if strain.ndim != 1:
        raise ValueError("strain must be a 1D array.")

    F = np.tile(np.eye(2), (len(strain), 1, 1)).astype(float)
    F[:, 0, 1] = strain

    P = ContiEnergy.P_from_F(F, beta=beta, K=K, noise=noise)
    sigma = ContiEnergy.cauchy_from_F(F, beta=beta, K=K, noise=noise)

    fig, ax = plt.subplots(figsize=(9, 5))
    comp_ids = [(0, 0), (0, 1), (1, 0), (1, 1)]
    piola_levels = np.linspace(0.35, 0.85, len(comp_ids))
    cauchy_levels = np.linspace(0.35, 0.85, len(comp_ids))

    for (i, j), level in zip(comp_ids, piola_levels):
        ax.plot(
            strain,
            P[:, i, j],
            color=plt.cm.Blues(level),
            linestyle="-",
            linewidth=1.6,
            zorder=2,
            label=rf"$P_{{{i+1}{j+1}}}$ (Piola)",
        )

    for (i, j), level in zip(comp_ids, cauchy_levels):
        ax.plot(
            strain,
            sigma[:, i, j],
            color=plt.cm.Reds(level),
            linestyle="--",
            linewidth=1.8,
            zorder=4,
            label=rf"$\sigma_{{{i+1}{j+1}}}$ (Cauchy)",
        )

    ax.set_xlabel(r"$\gamma$ (Strain)")
    ax.set_ylabel("Stress")
    ax.set_title("PK1 and Cauchy Stress Components in Simple Shear")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig, ax


def _simple_shear_F(gamma):
    gamma = np.asarray(gamma, dtype=float)
    F = np.tile(np.eye(2), (*gamma.shape, 1, 1)).astype(float)
    F[..., 0, 1] = gamma
    return F


def _pure_shear_F(gamma):
    gamma = np.asarray(gamma, dtype=float)
    diagonal = np.sqrt(1.0 + 0.25 * gamma**2)
    F = np.tile(np.eye(2), (*gamma.shape, 1, 1)).astype(float)
    F[..., 0, 0] = diagonal
    F[..., 0, 1] = 0.5 * gamma
    F[..., 1, 0] = 0.5 * gamma
    F[..., 1, 1] = diagonal
    return F


def plot_cauchy_stress_components_simple_and_pure_shear(
    gamma=None,
    beta=-1 / 4,
    K=4,
    noise=1,
    save_path=os.path.join(
        "Plots", "cauchy_stress_components_simple_and_pure_shear.pdf"
    ),
):
    """
    Plot the independent 2D Cauchy stress components for determinant-one
    simple shear and symmetric pure shear over gamma in [0, 1].
    """
    if gamma is None:
        gamma = np.linspace(0.0, 1.0, 1000)
    gamma = np.asarray(gamma, dtype=float)
    if gamma.ndim != 1:
        raise ValueError("gamma must be a 1D array.")

    plt.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})

    paths = (
        ("Simple shear", _simple_shear_F(gamma)),
        ("Pure shear", _pure_shear_F(gamma)),
    )
    components = (
        ((0, 0), r"$\sigma_{11}$", "-"),
        ((0, 1), r"$\sigma_{12}$", "--"),
        ((1, 1), r"$\sigma_{22}$", ":"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, (title, F) in zip(axes, paths):
        sigma = ContiEnergy.cauchy_from_F(F, beta=beta, K=K, noise=noise)
        for (i, j), label, linestyle in components:
            ax.plot(
                gamma,
                sigma[:, i, j],
                linestyle=linestyle,
                linewidth=1.8,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("Cauchy stress")
        ax.grid(True, alpha=0.25)

    axes[1].legend()
    fig.suptitle("Square Conti Energy: Cauchy Stress Components")
    fig.tight_layout()

    if save_path is not None:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    return fig, axes


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
        # Compute dN/dx and area for each frame from current coordinates
        dN_dx_series, area_series = dN_dx_from_coords(x_series)
        f_lag = ContiEnergy.lagrangian_forces_from_simpleShear(strain, dN_dX_batched)
        f_eul = ContiEnergy.eulerian_forces_from_simpleShear(
            strain, dN_dx_series, area=area_series
        )

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
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 2))
    #fig.suptitle("Node Motion and Internal Forces under Simple Shear")

    for ax in (axL, axR):
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_aspect("equal")
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

    txt = None

    n_frames = len(strain)

    def update(k):
        percent = 100.0 * (k + 1) / n_frames
        print(f"{percent:.1f}%",end="\r")
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

        return triL, ptsL, qL, triR, ptsR, qR

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
        print(f"Animation saved to: {save_path}")

    return anim


if __name__ == "__main__":
    #plot_energy()
    #plot_eulerian_forces()
    #plot_Lagrangian_forces()
    plot_cauchy_stress_components_simple_and_pure_shear()
    # animate_nodes_and_forces(
    #     save_path="simple_shear_nodes_forces.mp4",
    #     interval=30,
    #     n_procs=1,  # try >1 if ContiEnergy calls are expensive
    # )
    if plt.get_backend().lower() != "agg":
        plt.show()
