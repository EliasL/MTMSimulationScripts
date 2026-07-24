import unittest

import numpy as np

from MTMath.reduction import lagrange_reduction
from Plotting.lagrangeReductionCorrectnessIllustration import decomposition_data


class LagrangeReductionCorrectnessIllustrationTests(unittest.TestCase):
    def test_lagrange_variants_cover_the_quadrants_but_not_the_known_factors(self):
        data = decomposition_data()
        _, expected_M = lagrange_reduction(data["F"].T @ data["F"])

        np.testing.assert_allclose(data["base_M"], expected_M)
        self.assertEqual(
            [branch["quadrant"] for branch in data["branches"]],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [branch["quadrant"] for branch in data["branches"] if branch["preferred"]],
            [0],
        )
        self.assertFalse(any(branch["matches"] for branch in data["branches"]))

        for branch in data["branches"]:
            np.testing.assert_allclose(branch["F_e"] @ branch["F_p"], data["F"])


if __name__ == "__main__":
    unittest.main()
