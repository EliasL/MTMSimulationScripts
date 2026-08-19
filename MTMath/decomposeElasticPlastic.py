
from .reduction import plastic_reduction
from .energyFunction import SShear
import matplotlib.pyplot as plt
from .poincareEnergy import (
    prepPoincareFig,
    drawC,
)
import numpy as np

def showDecomposition():
    grid_size = 200
    fig, ax = prepPoincareFig(
        grid_size=grid_size,
        grid_depth=4,
        withYieldSurface=False,
    )
    fig.set_size_inches(4.2, 4.2)
    F0 = SShear(1.3) @ SShear(0.9, s_conponent=(1, 0))@SShear(-0.2) 
    C0 = F0.T @ F0
    label_box = {
        "facecolor": "white",
        "edgecolor": "none",
        "pad": 1.5,
        "alpha": 0.5,
    }

    # Mark start
    drawC(
        ax,
        np.array([C0]),
        grid_size=grid_size,
        scatter=True,
        s=40,
        label=r"$\mathbf{F}^{\mathsf{T}}\mathbf{F}$",
        label_ha="right",
        label_x=-8,
        label_bbox=label_box,
    )

    _, M = plastic_reduction(C0, compute_M=True)
    M_inv = np.linalg.inv(M)
    F_p = M_inv
    F_e = F0@M

    drawC(
        ax,
        [np.eye(2), F_p.T @ F_p],
        arrow=True,
        label=r"$\mathbf{F}_p^{\mathsf{T}}\mathbf{F}_p$",
        label_va="top",
        label_x=8,
        label_bbox=label_box,
    )
    drawC(
        ax,
        [np.eye(2), F_e.T @ F_e],
        arrow=True,
        label=r"$\mathbf{F}_e^{\mathsf{T}}\mathbf{F}_e$",
        label_va="top",
        label_x=-8,
        label_y=-12,
        label_bbox=label_box,
    )

    plt.show()
