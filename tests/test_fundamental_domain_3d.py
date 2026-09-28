import unittest

import numpy as np

from Plotting.fundamentalDomain3D import fundamental_domain_vertices, square_epn_vertices
from MTMath.reduction import elastic_domain_quadrant


class FundamentalDomain3DTests(unittest.TestCase):
    def test_display_vertices_obey_fundamental_domain_inequalities(self):
        vertices = fundamental_domain_vertices((0.2, 2.0), 3.0)

        self.assertEqual(vertices.shape, (8, 3))
        c11, c22, c12 = vertices.T
        self.assertTrue(np.all(c11 >= 0.0))
        self.assertTrue(np.all(c12 >= 0.0))
        self.assertTrue(np.all(c12 <= c11 / 2.0))
        self.assertTrue(np.all(c11 <= c22))

    def test_invalid_display_window_raises(self):
        with self.assertRaises(ValueError):
            fundamental_domain_vertices((0.2, 3.1), 3.0)

    def test_display_window_can_include_rank_deficient_origin_face(self):
        vertices = fundamental_domain_vertices((0.0, 2.0), 3.0)

        self.assertTrue(np.any(np.all(vertices == 0.0, axis=1)))

    def test_four_epns_cover_the_four_elastic_quadrants(self):
        regions = square_epn_vertices((0.2, 2.0), 3.0)

        self.assertEqual(len(regions), 4)
        for expected_quadrant, (_, vertices) in enumerate(regions):
            # Vertex 6 has non-zero C12 and unequal diagonal entries, so it
            # belongs strictly to one elastic quadrant rather than a shared face.
            c11, c22, c12 = vertices[6]
            metric = np.array([[c11, c12], [c12, c22]])
            self.assertEqual(elastic_domain_quadrant(metric), expected_quadrant)


if __name__ == "__main__":
    unittest.main()
