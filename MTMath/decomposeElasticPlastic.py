
from .reduction import elastic_reduction
from .energyFunction import SShear
import matplotlib.pyplot as plt
from .plotEnergy import (
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

    CE, M = elastic_reduction(C0)
    M_inv = np.linalg.inv(M)
    F_P = M_inv
    F_E = F0@M
    F_test = F_E@F_P

    drawC(ax, [np.eye(2), F_P.T@F_P],arrow=True, label=r"$\mathbf{F}_P^T\mathbf{F}_P$",label_va="top")
    drawC(ax, [np.eye(2), F_E.T@F_E],arrow=True, label=r"$\mathbf{F}_E^T\mathbf{F}_E$",label_va="top")
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