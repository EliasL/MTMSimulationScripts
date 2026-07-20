import unittest

import numpy as np

from MTMath.meshUtils import (
    _build_triangular_elements,
    _compute_dN_dX,
    element_deformation_gradients,
    mesh_edge_segments,
    structured_triangle_connectivity,
    structured_triangular_mesh,
    triangle_shape_grads_and_area,
    unique_mesh_edges,
)


class StructuredTriangleMeshTests(unittest.TestCase):
    def test_two_by_two_grid_supports_both_diagonals(self):
        minor = structured_triangle_connectivity((2, 2), diagonal="minor")
        major = structured_triangle_connectivity((2, 2), diagonal="major")

        np.testing.assert_array_equal(minor, [[0, 1, 3], [0, 3, 2]])
        np.testing.assert_array_equal(major, [[0, 1, 2], [1, 3, 2]])

    def test_all_structured_triangles_are_counter_clockwise(self):
        nodes, _ = structured_triangular_mesh((4, 3))
        for diagonal in ("major", "minor"):
            connectivity = structured_triangle_connectivity(
                (4, 3), diagonal=diagonal
            )
            triangles = nodes[connectivity]
            first_edges = triangles[:, 1] - triangles[:, 0]
            second_edges = triangles[:, 2] - triangles[:, 0]
            signed_double_area = (
                first_edges[:, 0] * second_edges[:, 1]
                - first_edges[:, 1] * second_edges[:, 0]
            )
            self.assertTrue(np.all(signed_double_area > 0.0))

    def test_mesh_builder_applies_requested_spacing(self):
        nodes, connectivity = structured_triangular_mesh(
            (3, 2), diagonal="major", dx=2.0, dy=0.5
        )

        np.testing.assert_allclose(
            nodes,
            [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [0.0, 0.5], [2.0, 0.5], [4.0, 0.5]],
        )
        self.assertEqual(connectivity.shape, (4, 3))

    def test_unique_edges_are_not_duplicated_between_elements(self):
        nodes, connectivity = structured_triangular_mesh((2, 2), diagonal="minor")
        edges = unique_mesh_edges(connectivity)
        segments = mesh_edge_segments(nodes, edges)

        self.assertEqual(edges.shape, (5, 2))
        self.assertEqual(segments.shape, (5, 2, 2))
        self.assertEqual(len({tuple(edge) for edge in edges}), 5)

    def test_deformation_gradients_support_load_histories(self):
        reference_nodes, connectivity = structured_triangular_mesh(
            (3, 3), diagonal="major"
        )
        gammas = np.array([-0.5, 0.0, 0.8])
        current_nodes = np.repeat(reference_nodes[None, :, :], len(gammas), axis=0)
        current_nodes[:, :, 0] += gammas[:, None] * reference_nodes[None, :, 1]

        F = element_deformation_gradients(
            reference_nodes, current_nodes, connectivity
        )
        expected = np.repeat(np.eye(2)[None, :, :], len(gammas), axis=0)
        expected[:, 0, 1] = gammas

        self.assertEqual(F.shape, (len(gammas), len(connectivity), 2, 2))
        np.testing.assert_allclose(
            F, np.repeat(expected[:, None, :, :], len(connectivity), axis=1)
        )

    def test_legacy_helpers_delegate_to_public_implementations(self):
        nodes, connectivity = structured_triangular_mesh((3, 3), diagonal="minor")
        expected_gradients, _ = triangle_shape_grads_and_area(nodes[connectivity])

        np.testing.assert_array_equal(
            _build_triangular_elements((3, 3)), connectivity
        )
        np.testing.assert_allclose(
            _compute_dN_dX(nodes, connectivity), expected_gradients
        )

    def test_invalid_diagonal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "major.*minor"):
            structured_triangle_connectivity((2, 2), diagonal="other")


if __name__ == "__main__":
    unittest.main()
