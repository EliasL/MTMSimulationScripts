import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Plotting.pyplotFunctions import (
    base_plot,
    plot_binned_poincare_displacement_field,
)


class PoincareVelocityFieldTests(unittest.TestCase):
    def test_base_plot_preserves_equal_data_scaling(self):
        axis, figure = base_plot(add_title=False)
        try:
            self.assertEqual(axis.get_aspect(), 1.0)
            self.assertEqual(axis.get_adjustable(), "box")
        finally:
            plt.close(figure)

    def test_binned_field_uses_coherence_filtered_mean_displacements(self):
        figure, axis = plt.subplots()
        try:
            quiver, shown_bins, populated_bins = (
                plot_binned_poincare_displacement_field(
                    axis,
                    x=np.array([0.10, 0.12]),
                    y=np.array([0.10, 0.12]),
                    u=np.array([0.02, 0.02]),
                    v=np.array([0.01, 0.01]),
                    grid_size=100,
                    zoom=1,
                    bins=4,
                    min_count=2,
                    min_coherence=0.9,
                    show_colorbar=False,
                )
            )
        finally:
            plt.close(figure)

        self.assertIsNotNone(quiver)
        self.assertEqual(shown_bins, 1)
        self.assertEqual(populated_bins, 1)

    def test_binned_field_can_colour_opaque_wide_arrows_by_supplied_scalar(self):
        figure, axis = plt.subplots()
        try:
            quiver, shown_bins, _ = plot_binned_poincare_displacement_field(
                axis,
                x=np.array([0.10, 0.12, 0.60, 0.62]),
                y=np.array([0.10, 0.12, 0.60, 0.62]),
                u=np.array([0.02, 0.02, 0.03, 0.03]),
                v=np.array([0.01, 0.01, 0.02, 0.02]),
                grid_size=100,
                zoom=1,
                bins=4,
                min_count=2,
                min_coherence=0.9,
                color_values=np.array([2.0, 2.0, 4.0, 4.0]),
                colorbar_label=r"mean $\|\Delta\mathbf{T}\|_F$",
                colorbar_log=True,
                vector_length_from_color=True,
                vector_length_scale=0.7,
            )
            self.assertEqual(axis.figure.axes[-1].get_xlabel(), r"mean $\|\Delta\mathbf{T}\|_F$")
            self.assertEqual(quiver.width, 0.006)
            self.assertTrue(np.all(quiver.get_facecolors()[:, 3] == 1.0))
            self.assertEqual(len(quiver.get_path_effects()), 1)
            self.assertEqual(type(axis.figure.axes[-1]._colorbar.mappable.norm).__name__, "LogNorm")
            calibration = np.median(
                [np.hypot(0.02, 0.01) / 2.0, np.hypot(0.03, 0.02) / 4.0]
            )
            expected_lengths = 50.0 * calibration * np.array([2.0, 4.0]) * 0.7
            self.assertTrue(
                np.allclose(
                    np.sort(np.hypot(quiver.U, quiver.V)),
                    np.sort(expected_lengths),
                )
            )
        finally:
            plt.close(figure)

        self.assertEqual(shown_bins, 2)

    def test_binned_field_otsu_split_preserves_opposite_directions(self):
        figure, axis = plt.subplots()
        try:
            quivers, shown_vectors, populated_bins = (
                plot_binned_poincare_displacement_field(
                    axis,
                    x=np.full(6, 0.10),
                    y=np.full(6, 0.10),
                    u=np.zeros(6),
                    v=np.array([0.2, 0.2, 0.2, -0.2, -0.2, -0.2]),
                    grid_size=100,
                    zoom=1,
                    bins=4,
                    min_count=5,
                    min_coherence=0.9,
                    show_colorbar=False,
                    vector_length_scale=0.7,
                    direction_split_otsu=True,
                )
            )
            self.assertEqual(populated_bins, 1)
            self.assertEqual(shown_vectors, 2)
            self.assertEqual(len(quivers), 2)
            self.assertTrue(all(quiver.pivot == "tail" for quiver in quivers))
            vertical_components = sorted(float(quiver.V[0]) for quiver in quivers)
            self.assertLess(vertical_components[0], 0.0)
            self.assertGreater(vertical_components[1], 0.0)
            self.assertTrue(np.allclose(np.abs(vertical_components), [7.0, 7.0]))
        finally:
            plt.close(figure)

    def test_binned_field_otsu_split_handles_angle_wraparound(self):
        figure, axis = plt.subplots()
        try:
            angles = np.deg2rad([178.0, 179.0, -179.0, -178.0, 1.0, -1.0])
            quivers, shown_vectors, populated_bins = (
                plot_binned_poincare_displacement_field(
                    axis,
                    x=np.full(angles.size, 0.10),
                    y=np.full(angles.size, 0.10),
                    u=0.2 * np.cos(angles),
                    v=0.2 * np.sin(angles),
                    grid_size=100,
                    zoom=1,
                    bins=4,
                    min_count=5,
                    min_coherence=0.9,
                    show_colorbar=False,
                    direction_split_otsu=True,
                )
            )
            self.assertEqual(populated_bins, 1)
            self.assertEqual(shown_vectors, 2)
            self.assertEqual(len(quivers), 2)
            horizontal_components = sorted(float(quiver.U[0]) for quiver in quivers)
            self.assertLess(horizontal_components[0], 0.0)
            self.assertGreater(horizontal_components[1], 0.0)
        finally:
            plt.close(figure)

    def test_binned_field_can_keep_only_bins_with_two_otsu_branches(self):
        figure, axis = plt.subplots()
        try:
            quivers, shown_vectors, populated_bins = (
                plot_binned_poincare_displacement_field(
                    axis,
                    x=np.array([0.10, 0.10, 0.10, 0.10, 0.60, 0.62]),
                    y=np.array([0.10, 0.10, 0.10, 0.10, 0.60, 0.62]),
                    u=np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.2]),
                    v=np.array([0.2, 0.2, -0.2, -0.2, 0.2, 0.2]),
                    grid_size=100,
                    zoom=1,
                    bins=4,
                    min_count=2,
                    min_coherence=0.1,
                    show_colorbar=False,
                    direction_split_otsu=True,
                    require_direction_split=True,
                )
            )
            self.assertEqual(populated_bins, 2)
            self.assertEqual(shown_vectors, 2)
            self.assertEqual(len(quivers), 2)
        finally:
            plt.close(figure)

    def test_binned_field_double_filter_is_applied_after_branch_visibility(self):
        figure, axis = plt.subplots()
        try:
            quiver, shown_vectors, populated_bins = (
                plot_binned_poincare_displacement_field(
                    axis,
                    x=np.full(4, 0.10),
                    y=np.full(4, 0.10),
                    u=np.zeros(4),
                    v=np.array([0.2, 0.2, -0.05, -0.05]),
                    grid_size=100,
                    zoom=1,
                    bins=4,
                    min_count=2,
                    min_coherence=0.1,
                    min_vector_length=0.1,
                    show_colorbar=False,
                    direction_split_otsu=True,
                    require_direction_split=True,
                )
            )
            self.assertIsNone(quiver)
            self.assertEqual(shown_vectors, 0)
            self.assertEqual(populated_bins, 1)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
