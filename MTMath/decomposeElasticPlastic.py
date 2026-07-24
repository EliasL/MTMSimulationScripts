
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
    fig, ax = prepPoincareFig(grid_size=grid_size)
    F0 = SShear(1.3) @ SShear(0.9, s_conponent=(1, 0))@SShear(-0.2) 
    C0 = F0.T @ F0

    # Mark start
    drawC(
        ax,
        np.array([C0]),
        grid_size=grid_size,
        scatter=True,
        s=40,
        label=r"$\mathbf{C}$"
    )

    CE, M = plastic_reduction(C0, compute_M=True)
    M_inv = np.linalg.inv(M)
    F_p = M_inv
    F_e = F0@M
    F_test = F_e@F_p

    drawC(ax, [np.eye(2), F_p.T@F_p],arrow=True, label=r"$\mathbf{F}_p^T\mathbf{F}_p$",label_va="top")
    drawC(ax, [np.eye(2), F_e.T@F_e],arrow=True, label=r"$\mathbf{F}_e^T\mathbf{F}_e$",label_va="top")
    #drawC(ax, [np.eye(2), F_test.T@F_test],arrow=True, label=r"$F^TF$")

    drawC(
        ax,
        np.array([CE]),
        grid_size=grid_size,
        scatter=True,
        s=40,
        label=r"$\mathbf{C}_E$",
        label_ha="right",
        label_x=-5,
    )

    plt.show()
