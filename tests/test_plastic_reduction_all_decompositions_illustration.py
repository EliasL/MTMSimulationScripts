import unittest
from collections import Counter

import numpy as np

from Plotting.plasticReductionAllDecompositionsIllustration import (
    M1,
    M2,
    decomposition_table_data,
    generated_symmetry_group,
)


class PlasticReductionAllDecompositionsIllustrationTests(unittest.TestCase):
    def test_m1_m2_generate_eight_unique_decompositions(self):
        group = generated_symmetry_group()
        data = decomposition_table_data()

        self.assertEqual(len(group), 8)
        self.assertEqual(len(data), 8)
        self.assertFalse(np.allclose(M1 @ M2, M2 @ M1))
        self.assertEqual(
            Counter(item["quadrant"] for item in data),
            Counter({0: 2, 1: 2, 2: 2, 3: 2}),
        )

        matrices = {tuple(np.rint(item["M"]).astype(int).ravel()) for item in data}
        self.assertEqual(len(matrices), 8)
        for item in data:
            np.testing.assert_allclose(item["F_e"] @ item["F_p"], data[0]["F_e"] @ data[0]["F_p"])

        r = M1 @ M2
        s = M1
        for column in range(4):
            np.testing.assert_allclose(data[column]["symmetry"], np.linalg.matrix_power(r, column))
            np.testing.assert_allclose(data[column + 4]["symmetry"], np.linalg.matrix_power(r, column) @ s)

        for row_start in (0, 4):
            np.testing.assert_allclose(data[row_start + 2]["F_e"], -data[row_start]["F_e"])
            np.testing.assert_allclose(data[row_start + 2]["F_p"], -data[row_start]["F_p"])
            np.testing.assert_allclose(data[row_start + 3]["F_e"], -data[row_start + 1]["F_e"])
            np.testing.assert_allclose(data[row_start + 3]["F_p"], -data[row_start + 1]["F_p"])


if __name__ == "__main__":
    unittest.main()
