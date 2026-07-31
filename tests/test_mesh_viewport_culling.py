import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import colors, pyplot as plt
from matplotlib.cm import ScalarMappable

from Plotting.pyplotFunctions import (
    _add_additional_elements,
    _plot_mesh_elements,
    calculate_shifts,
)


class MeshViewportCullingTests(unittest.TestCase):
    def _render(self, cull):
        triangle_nodes = np.array(
            [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [10, 10], [11, 10], [10, 11]]
        )
        nodes = np.vstack((triangle_nodes, np.full((2001, 2), 20.0)))
        connectivity = np.array([[0, 1, 2], [3, 4, 5]])
        fig, ax = plt.subplots(figsize=(2, 2), dpi=60)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_axis_off()

        class Data:
            load = 0.0

        mappable = _plot_mesh_elements(
            ax,
            nodes[:, 0],
            nodes[:, 1],
            connectivity,
            np.array([0.25, 0.75]),
            colors.Normalize(0, 1),
            "viridis",
            Data(),
            "energy",
            "none",
            False,
            None,
            False,
            None,
            shifts=[(0, 0)],
            cull_to_view=cull,
        )
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba()).copy()
        paths = len(mappable.get_paths())
        plt.close(fig)
        return image, paths

    def test_culling_omits_offscreen_triangles_without_changing_pixels(self):
        full_image, full_paths = self._render(False)
        culled_image, culled_paths = self._render(True)
        self.assertEqual((full_paths, culled_paths), (2, 1))
        np.testing.assert_array_equal(culled_image, full_image)

    def test_periodic_shifts_cover_viewport_after_large_shear(self):
        class Data:
            BC = "PBC"
            load = 3.0
            size = (1, 1)

        x = np.array([0.0, 1.0, 3.0, 4.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        fig, ax = plt.subplots()
        ax.set(xlim=(0, 1), ylim=(0, 1))
        shifts = calculate_shifts(ax, Data(), x, y)
        plt.close(fig)

        for px in np.linspace(0.05, 0.95, 10):
            for py in np.linspace(0.05, 0.95, 10):
                self.assertTrue(
                    any(
                        0 <= py - dy <= 1
                        and 0 <= px - dx - Data.load * (py - dy) <= 1
                        for dx, dy in shifts
                    )
                )

    def test_single_mesh_colorbar_is_horizontal(self):
        fig, ax = plt.subplots()
        mappable = ScalarMappable(colors.Normalize(0, 1), "viridis")

        class Data:
            BC = "PBC"

        _add_additional_elements(
            ax,
            mappable,
            "energy",
            True,
            None,
            None,
            None,
            False,
            np.empty((0, 2)),
            Data(),
            False,
            colorbar_orientation="horizontal",
        )
        colorbar_ax = fig.axes[-1]
        self.assertGreater(colorbar_ax.get_position().width, colorbar_ax.get_position().height)
        self.assertEqual(colorbar_ax.get_xlabel(), "$E_i$")
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
