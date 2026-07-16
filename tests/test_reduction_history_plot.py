import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from MTMath.poincareEnergy import (
    C2PoincareDisk,
    generate_elastic_quadrant_grid,
    plot_reduction_history,
)
from MTMath.reduction import (
    elastic_reduction,
    elastic_reduction_history,
    lagrange_reduction,
    lagrange_reduction_history,
)


class ReductionHistoryPlotTests(unittest.TestCase):
    def setUp(self):
        self.F = np.array([[-0.43, 1.21], [-1.19, 1.02]])
        self.C = self.F.T @ self.F

    def tearDown(self):
        plt.close("all")

    def test_histories_match_reduction_endpoints_and_reference_path(self):
        lagrange_history = lagrange_reduction_history(self.C)
        elastic_history = elastic_reduction_history(self.C)

        np.testing.assert_allclose(
            lagrange_history[-1], lagrange_reduction(self.C)[0]
        )
        np.testing.assert_allclose(
            elastic_history[-1], elastic_reduction(self.C)[0]
        )

        lagrange_xy = np.column_stack(C2PoincareDisk(lagrange_history))
        elastic_xy = np.column_stack(C2PoincareDisk(elastic_history))
        np.testing.assert_allclose(
            lagrange_xy,
            [
                [-0.14791834, -0.56780341],
                [-0.14791834, 0.56780341],
                [0.22723950, 0.06276970],
                [-0.22723950, 0.06276970],
            ],
            atol=1e-8,
        )
        np.testing.assert_allclose(
            elastic_xy,
            [
                [-0.14791834, -0.56780341],
                [0.22723950, -0.06276970],
            ],
            atol=1e-8,
        )

    def test_quadrant_grid_masks_disk_exterior(self):
        quadrants = generate_elastic_quadrant_grid(resolution=41)
        self.assertEqual(quadrants.shape, (41, 41))
        self.assertTrue(np.isnan(quadrants[0, 0]))
        self.assertEqual(
            set(np.unique(quadrants[np.isfinite(quadrants)])), {0, 1, 2, 3}
        )

    def test_elastic_history_uses_rotation_free_unit_shears(self):
        F = np.array([[1.0, 5.0], [0.0, 1.0]])
        history = elastic_reduction_history(F.T @ F)
        np.testing.assert_allclose(history[:, 0, 1], [5, 4, 3, 2, 1, 0])

    def test_plot_uses_white_background_and_draws_both_histories(self):
        with matplotlib.rc_context({"text.usetex": False}):
            fig, ax = plot_reduction_history(
                self.F,
                resolution=61,
                show_grid=False,
                show_colorbar=False,
            )

        np.testing.assert_allclose(fig.get_facecolor()[:3], (1, 1, 1))
        np.testing.assert_allclose(ax.get_facecolor()[:3], (1, 1, 1))
        self.assertEqual(len(ax.images), 1)
        self.assertEqual(len(ax.texts), 4)
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(ax.get_xlabel(), r"$x_p$")
        self.assertEqual(ax.get_ylabel(), r"$y_p$")


if __name__ == "__main__":
    unittest.main()
