import unittest

import numpy as np

from Plotting.kappaEventClassification import (
    classification_metrics,
    kappa_from_relaxation_energy,
    mu_kappa_threshold,
)


class KappaEventClassificationTests(unittest.TestCase):
    def test_kappa_uses_relaxation_energy_volume_and_strain_increment(self):
        result = kappa_from_relaxation_energy(
            np.array([2.0, 8.0]),
            np.array([0.1, 0.2]),
            10.0,
            rho=1.0,
        )
        np.testing.assert_allclose(result, [20.0, 20.0])

    def test_kappa_rejects_nonpositive_strain_increment(self):
        with self.assertRaisesRegex(ValueError, "Delta gamma"):
            kappa_from_relaxation_energy([1.0], [0.0], 10.0)

    def test_born_bound_includes_density(self):
        np.testing.assert_allclose(
            mu_kappa_threshold(np.array([6.0, 10.0]), rho=2.0),
            [1.5, 2.5],
        )

    def test_classification_metrics(self):
        metrics = classification_metrics(
            np.array([True, True, False, False]),
            np.array([True, False, True, False]),
        )
        self.assertEqual(
            {key: metrics[key] for key in ("tp", "fp", "fn", "tn")},
            {"tp": 1, "fp": 1, "fn": 1, "tn": 1},
        )
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)


if __name__ == "__main__":
    unittest.main()
