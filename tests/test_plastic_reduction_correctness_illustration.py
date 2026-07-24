import unittest

import numpy as np

from Plotting.plasticReductionCorrectnessIllustration import decomposition_data


class PlasticReductionCorrectnessIllustrationTests(unittest.TestCase):
    def test_only_known_quadrant_recovers_the_original_factors(self):
        data = decomposition_data()

        np.testing.assert_allclose(
            data["known_F_e"] @ data["known_F_p"], data["F"]
        )
        np.testing.assert_allclose(
            data["base_M"], np.linalg.inv(data["known_F_p"])
        )
        self.assertEqual(
            [branch["quadrant"] for branch in data["branches"]],
            [3, 2, 1, 0],
        )
        self.assertEqual(
            [branch["quadrant"] for branch in data["branches"] if branch["matches"]],
            [3],
        )
        self.assertEqual(
            [branch["label"] for branch in data["branches"]],
            [
                r"$\mathbf{M}_{3}=\mathbf{M}$",
                r"$\mathbf{M}_{2}=\mathbf{M}\mathbf{m}_1$",
                r"$\mathbf{M}_{1}=\mathbf{M}\mathbf{m}_2$",
                r"$\mathbf{M}_{0}=\mathbf{M}\mathbf{m}_1\mathbf{m}_2$",
            ],
        )

        for branch in data["branches"]:
            np.testing.assert_allclose(branch["F_e"] @ branch["F_p"], data["F"])


if __name__ == "__main__":
    unittest.main()
