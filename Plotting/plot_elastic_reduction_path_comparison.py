"""Plot unit-step and batched-integer elastic-reduction paths."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from MTMath.energyFunction import F_from_C
from MTMath.poincareEnergy import C2PoincareDisk, plot_reduction_history
from MTMath.reduction import elastic_reduction_history, in_elastic_domain


EXAMPLE_C = np.array(
    [
        [1543.349514563135, -1724.3300970874102],
        [-1724.3300970874102, 1926.53398058256],
    ],
    dtype=float,
)

NEAR_CENTER_C = np.array(
    [
        [1.0, -1.51],
        [-1.51, 3.0],
    ],
    dtype=float,
)


def multistep_elastic_reduction_history(C, loops=1000):
    """Return the nearest-integer-shear path for one SPD metric."""
    current = np.asarray(C, dtype=float).copy()
    history = [current.copy()]

    for _ in range(loops):
        a = current[0, 0]
        b = current[1, 1]
        c = current[0, 1]
        if in_elastic_domain(a, b, c):
            return np.stack(history)

        denominator = a if a <= b else b
        ratio = -c / denominator
        m = int(np.sign(ratio) * np.floor(abs(ratio) + 0.5))
        if m == 0:
            raise RuntimeError("Multi-step elastic reduction made no progress")

        if a <= b:
            current[0, 1] = c + m * a
            current[1, 0] = current[0, 1]
            current[1, 1] = b + 2.0 * m * c + m * m * a
        else:
            current[0, 1] = c + m * b
            current[1, 0] = current[0, 1]
            current[0, 0] = a + 2.0 * m * c + m * m * b
        history.append(current.copy())

    raise RuntimeError(f"Multi-step elastic reduction did not converge in {loops} steps")


def make_plot(
    C=EXAMPLE_C,
    output_stem=Path("Plots/elastic_reduction_unit_vs_multistep_paths"),
):
    C = np.asarray(C, dtype=float)
    unit_history = elastic_reduction_history(C)
    multi_history = multistep_elastic_reduction_history(C)
    unit_color = "#008E70"
    multi_color = "#8E44AD"

    F = F_from_C(C)
    fig, ax = plot_reduction_history(
        F,
        histories=(
            (unit_history, unit_color, f"Unit steps ({len(unit_history) - 1})"),
            (multi_history, multi_color, f"Batched shears ({len(multi_history) - 1})"),
        ),
        resolution=600,
        grid_depth=6,
        show_grid=True,
        show_colorbar=False,
        show_legend=True,
        show_axes=True,
        lagrange_color="#111111",
        elastic_color="#111111",
        grid_color="#4A4A4A",
        linewidth=2.2,
        white_background=True,
    )

    for history, color, marker, label_offset in (
        (unit_history, unit_color, "o", (-26, 16)),
        (multi_history, multi_color, "D", (16, -24)),
    ):
        end_x, end_y = C2PoincareDisk(history[-1])
        end = np.array([end_x, end_y]) * 300 + 300
        ax.scatter(
            end[0],
            end[1],
            s=75,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=7,
        )
        ax.annotate(
            "unit" if marker == "o" else "batched",
            xy=end,
            xytext=label_offset,
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.8},
            zorder=8,
        )

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"unit steps: {len(unit_history) - 1}")
    print(f"batched steps: {len(multi_history) - 1}")
    print(f"unit endpoint:\n{unit_history[-1]}")
    print(f"batched endpoint:\n{multi_history[-1]}")
    print(png_path)
    print(pdf_path)
    return png_path, pdf_path


def make_near_center_plot():
    return make_plot(
        C=NEAR_CENTER_C,
        output_stem=Path("Plots/elastic_reduction_near_center_counterexample"),
    )


if __name__ == "__main__":
    make_near_center_plot()
