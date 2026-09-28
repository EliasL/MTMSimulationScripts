"""Render the laboratory-frame PK2 stress-sign map over the full Poincare disk."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from MTMath.energyFunction import ContiEnergy
from MTMath.poincareEnergy import drawPoincareGrid, generate_poincare_disk, prepPoincareFig


OUTPUT = Path("Plots/RealFquadrantStress/quadrant/poincare_PK2_sign_map_q1000.pdf")
GRID_SIZE = 1000


def plot_pk2_sign_map(output: Path = OUTPUT, grid_size: int = GRID_SIZE) -> Path:
    """Plot signs of ``S12`` and ``S11 - S22`` directly as functions of ``C``."""
    C, outside = generate_poincare_disk(grid_size, returnMask=True)
    S = ContiEnergy.S_from_C(C)
    labels = 2 * (S[..., 0, 0] - S[..., 1, 1] < 0) + (S[..., 0, 1] < 0)
    labels = np.ma.array(labels, mask=outside)

    _, ax = prepPoincareFig(
        grid_size=grid_size,
        withGrid=False,
        withYieldSurface=False,
    )
    drawPoincareGrid(ax=ax, grid_size=grid_size, depth=4, alpha=0.5)
    cmap = ListedColormap(plt.colormaps["tab10"](np.arange(4)))
    image = ax.imshow(
        labels,
        origin="lower",
        extent=(0, grid_size, 0, grid_size),
        interpolation="nearest",
        cmap=cmap,
        norm=BoundaryNorm(np.arange(-0.5, 4.5), cmap.N),
    )
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(4))

    output.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    return output


if __name__ == "__main__":
    print(plot_pk2_sign_map())
