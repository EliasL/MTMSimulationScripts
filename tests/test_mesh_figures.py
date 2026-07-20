import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from MTMath.meshUtils import unique_mesh_edges
from Plotting.meshFigures import (
    build_edge_flip_example,
    make_edge_flip_deformation_gradient_figure,
)


def _shared_edge(connectivity: np.ndarray) -> tuple[int, int]:
    edges, counts = np.unique(
        np.sort(
            np.concatenate(
                [
                    connectivity[:, [0, 1]],
                    connectivity[:, [1, 2]],
                    connectivity[:, [2, 0]],
                ]
            ),
            axis=1,
        ),
        axis=0,
        return_counts=True,
    )
    return tuple(edges[counts == 2][0])


class MeshFiguresTests(unittest.TestCase):
    def test_edge_flip_example_changes_diagonal_and_local_F(self):
        example = build_edge_flip_example(shear=0.75)
        initial, loaded, flipped = example.states

        np.testing.assert_allclose(
            initial.deformation_gradients,
            np.broadcast_to(np.eye(2), (2, 2, 2)),
        )
        np.testing.assert_allclose(
            loaded.deformation_gradients,
            np.broadcast_to(example.applied_deformation_gradient, (2, 2, 2)),
        )
        np.testing.assert_allclose(
            flipped.deformation_gradients,
            np.broadcast_to(
                np.array([[1.0, -0.25], [0.0, 1.0]]),
                (2, 2, 2),
            ),
        )
        self.assertEqual(_shared_edge(initial.connectivity), (0, 3))
        self.assertEqual(_shared_edge(flipped.connectivity), (1, 2))
        self.assertEqual(len(unique_mesh_edges(initial.connectivity)), 5)
        self.assertEqual(len(unique_mesh_edges(flipped.connectivity)), 5)

    def test_post_flip_references_remain_at_their_stored_coordinates(self):
        flipped = build_edge_flip_example(shear=0.75).states[-1]
        self.assertAlmostEqual(flipped.reference_elements.min(), 0.0)
        self.assertAlmostEqual(flipped.reference_elements.max(), 1.0)
        self.assertFalse(
            np.allclose(
                flipped.reference_elements.mean(axis=1),
                flipped.current_elements.mean(axis=1),
            )
        )

    def test_edge_flip_figure_has_two_rows_and_two_columns(self):
        figure, axes, example = make_edge_flip_deformation_gradient_figure()
        try:
            self.assertEqual(axes.shape, (2, 2))
            self.assertEqual(len(example.states), 3)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
