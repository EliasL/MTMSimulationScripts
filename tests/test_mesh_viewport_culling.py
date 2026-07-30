import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import colors, pyplot as plt

from Plotting.pyplotFunctions import _plot_mesh_elements


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


if __name__ == "__main__":
    unittest.main()
