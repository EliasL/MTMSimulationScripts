"""Regenerate the Conti energy disk with its Lagrange grid, without a domain overlay."""

from pathlib import Path

import matplotlib.pyplot as plt

from MTMath.poincareEnergy import generate_energy_grid, plotEnergyField


def generate_figure(resolution=1000, output_stem="Plots/conti_poincare_disk_with_lagrange_grid"):
    """Save PNG and JPEG versions using the existing energy-field renderer."""
    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12}):
        energy = generate_energy_grid(resolution=resolution, energy_lim=[0, 0.37])
        ax = plotEnergyField(
            energy,
            save=False,
            add_title=False,
            remove_max_color=False,
            withYieldSurface=False,
            withGrid=True,
            grid_depth=6,
            withFundamentalDomain=False,
            withSymmetryPoints=True,
            symmetryPoints_kwargs={"depth": 2},
            show_colorbar_cap=False,
            colorbar_label=r"Energy density $\Phi$",
        )
        stem = Path(output_stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "jpg"):
            path = stem.with_suffix(f".{extension}")
            ax.figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
            print(f"Saved {path.resolve()}")
        plt.close(ax.figure)


if __name__ == "__main__":
    generate_figure()
