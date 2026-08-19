import unittest

import numpy as np

from Plotting.plasticReductionCorrectnessIllustration import (
    M1,
    M2,
    counterclockwise_branch,
    decomposition_data,
)


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
                r"$\mathbf{M}_0^{(c)}=\mathbf{M}\mathbf{m}_1\mathbf{m}_2$",
            ],
        )
        self.assertEqual(
            [
                branch["decomposition_superscript"]
                for branch in data["branches"]
            ],
            ["3", "2", "1", "0,c"],
        )

        for branch in data["branches"]:
            np.testing.assert_allclose(branch["F_e"] @ branch["F_p"], data["F"])

    def test_counterclockwise_partner_uses_the_opposite_rotation(self):
        data = decomposition_data()
        counterclockwise = counterclockwise_branch(data)
        clockwise = data["branches"][-1]

        self.assertEqual(counterclockwise["decomposition_superscript"], "0,cc")

        np.testing.assert_allclose(
            counterclockwise["M"], data["base_M"] @ M2 @ M1
        )
        np.testing.assert_allclose(
            clockwise["M"], data["base_M"] @ M1 @ M2
        )
        np.testing.assert_allclose(
            counterclockwise["M"], -clockwise["M"]
        )
        np.testing.assert_allclose(
            counterclockwise["F_e"] @ counterclockwise["F_p"], data["F"]
        )


if __name__ == "__main__":
    unittest.main()
