import unittest

import numpy as np

from MTMath.meshUtils import structured_triangle_connectivity
from MTMath.miniMTM import simpleShearSystem2


class MiniMTMNumericalMeshTests(unittest.TestCase):
    def test_simple_shear_system_uses_shared_major_diagonal_mesh(self):
        shear_values = np.array([0.0, 0.5])
        positions, elements, F, dN_dX = simpleShearSystem2(
            L=3, shearValues=shear_values
        )
        connectivity = np.asarray([element.node_ids for element in elements])
        expected_connectivity = structured_triangle_connectivity(
            (3, 3), diagonal="major"
        )

        np.testing.assert_array_equal(connectivity, expected_connectivity)
        self.assertEqual(positions.shape, (2, 9, 2))
        self.assertEqual(F.shape, (2, 8, 2, 2))
        self.assertEqual(dN_dX.shape, (8, 3, 2))

        expected_F = np.repeat(np.eye(2)[None, :, :], len(shear_values), axis=0)
        expected_F[:, 0, 1] = shear_values
        np.testing.assert_allclose(
            F,
            np.repeat(expected_F[:, None, :, :], len(connectivity), axis=1),
        )


if __name__ == "__main__":
    unittest.main()
