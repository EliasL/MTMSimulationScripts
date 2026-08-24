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
            np.array(
                [
                    [[1.0, 1.0], [0.0, 1.0]],
                    [[1.0, 0.0], [1.0, 1.0]],
                ]
            ),
        )
        np.testing.assert_allclose(
            loaded.deformation_gradients,
            np.array(
                [
                    [[1.0, 1.75], [0.0, 1.0]],
                    [[1.75, 0.75], [1.0, 1.0]],
                ]
            ),
        )
        np.testing.assert_allclose(
            flipped.deformation_gradients,
            np.array(
                [
                    [[1.0, 0.75], [0.0, 1.0]],
                    [[0.75, -0.25], [1.0, 1.0]],
                ]
            ),
        )
        np.testing.assert_allclose(
            example.F,
            np.array([[1.0, 1.75], [0.0, 1.0]]),
        )
        np.testing.assert_allclose(loaded.deformation_gradients[0], example.F)
        np.testing.assert_allclose(
            loaded.deformation_gradients[1], example.F @ example.Q
        )
        for state in (loaded, flipped):
            np.testing.assert_allclose(
                state.deformation_gradients[1],
                state.deformation_gradients[0] @ example.Q,
            )
        self.assertAlmostEqual(np.linalg.det(example.Q), 1.0)
        self.assertFalse(
            np.allclose(
                flipped.deformation_gradients[0],
                example.F @ np.linalg.inv(example.Q),
            )
        )
        self.assertFalse(
            np.allclose(
                loaded.deformation_gradients,
                flipped.deformation_gradients,
            )
        )
        self.assertEqual(_shared_edge(initial.connectivity), (0, 3))
        self.assertEqual(_shared_edge(flipped.connectivity), (1, 2))
        self.assertEqual(len(unique_mesh_edges(initial.connectivity)), 5)
        self.assertEqual(len(unique_mesh_edges(flipped.connectivity)), 5)

    def test_post_flip_elements_share_one_canonical_reference(self):
        flipped = build_edge_flip_example(shear=0.75).states[-1]
        np.testing.assert_allclose(
            flipped.canonical_reference_element,
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        )
        np.testing.assert_allclose(
            flipped.reference_elements[0], flipped.reference_elements[1]
        )
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
