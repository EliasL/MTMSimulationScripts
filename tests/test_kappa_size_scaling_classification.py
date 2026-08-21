import unittest

import numpy as np

from Plotting.kappaSizeScalingClassification import _summary_for_size


class KappaSizeScalingClassificationTests(unittest.TestCase):
    def test_fixed_born_threshold_is_size_independent(self):
        data = {
            "kappa": np.array([1.0, 4.0]),
            "recorded_plastic": np.array([False, True]),
        }
        summary = _summary_for_size(
            data,
            50,
            simple_drop_er_det=1.0e-6,
            rho=1.0,
            reference_threshold=3.0,
        )
        self.assertEqual(summary["fixed_threshold"], 3.0)
        self.assertEqual(summary["fixed_mu_over_2"]["tp"], 1)
        self.assertEqual(summary["fixed_mu_over_2"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
