import unittest

import matplotlib.pyplot as plt
import numpy as np

from Plotting.pyplotFunctions import (
    draw_periodic_shear_box,
    tile_periodic_mesh,
    wrap_periodic_mesh,
)
from Plotting.meshEventPlotting import MeshState, calculate_periodic_energy_change_field


class PeriodicMeshWrappingTests(unittest.TestCase):
    def setUp(self):
        load = 0.5
        box = np.array([[2.0, load * 2.0], [0.0, 2.0]])
        fractional = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=float,
        )
        self.points = fractional @ box.T
        self.triangles = np.array([[0, 1, 2], [0, 2, 3]])
        self.values = np.array([1.0, 2.0])
        self.reference_indices = np.arange(len(self.points))
        self.load = load

    def test_wrap_maps_sheared_cell_to_unit_square(self):
        polygons, values, wrapped = wrap_periodic_mesh(
            self.points,
            self.triangles,
            self.values,
            self.reference_indices,
            self.load,
            box_size=2.0,
        )

        self.assertEqual(polygons.shape[1:], (3, 2))
        self.assertEqual(values.shape, (len(polygons),))
        self.assertTrue(np.all(wrapped >= 0.0))
        self.assertTrue(np.all(wrapped < 1.0))
        self.assertEqual(set(values), {1.0, 2.0})

    def test_tiling_covers_wide_video_window(self):
        polygons, values, _ = wrap_periodic_mesh(
            self.points,
            self.triangles,
            self.values,
            self.reference_indices,
            self.load,
            box_size=2.0,
        )
        tiled, tiled_values = tile_periodic_mesh(
            polygons, values, xlim=(0.0, 16.0 / 9.0), ylim=(0.0, 1.0)
        )

        self.assertGreater(len(tiled), len(polygons))
        self.assertEqual(tiled_values.shape, (len(tiled),))
        lower = tiled.min(axis=1)
        upper = tiled.max(axis=1)
        self.assertTrue(np.all(upper[:, 0] >= 0.0))
        self.assertTrue(np.all(lower[:, 0] <= 16.0 / 9.0))

    def test_missing_periodic_origin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "refIndex=0"):
            wrap_periodic_mesh(
                self.points,
                self.triangles,
                self.values,
                self.reference_indices + 1,
                self.load,
                box_size=2.0,
            )

    def test_periodic_energy_projection_is_square_and_image_invariant(self):
        triangles = np.array([[0, 1, 2]])
        local_fractional = np.array([[0.0, 0.0], [0.45, 0.0], [0.2, 0.4]])
        box = np.array([[2.0, self.load * 2.0], [0.0, 2.0]])
        points = local_fractional @ box.T

        def state(path, point_coordinates):
            return MeshState(
                path=path,
                points=point_coordinates,
                triangles=triangles,
                reference_indices=np.arange(3),
                point_fields={},
                cell_fields={"energy_field": np.array([3.0])},
            )

        first = state("first.vtu", points)
        second = state("second.vtu", points + box[:, 0])
        change, geometry = calculate_periodic_energy_change_field(
            first,
            second,
            first_load=self.load,
            second_load=self.load,
            box_size=2.0,
            common_grid_resolution=12,
        )

        self.assertEqual(geometry.kind, "grid")
        np.testing.assert_allclose(geometry.x[[0, -1]], [0.0, 1.0])
        np.testing.assert_allclose(geometry.y[[0, -1]], [0.0, 1.0])
        np.testing.assert_allclose(change, 0.0)

    def test_shear_box_outlines_the_deformed_reference_cell(self):
        figure, axis = plt.subplots()
        try:
            corners = draw_periodic_shear_box(
                axis, origin=(3.0, 4.0), load=0.5, box_size=2.0
            )
        finally:
            plt.close(figure)
        np.testing.assert_allclose(
            corners,
            [[3.0, 4.0], [5.0, 4.0], [6.0, 6.0], [4.0, 6.0], [3.0, 4.0]],
        )


if __name__ == "__main__":
    unittest.main()
