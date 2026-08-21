import unittest

import numpy as np

from Plotting.standardPowerlaw import (
    EventDrops,
    kappa_detection_threshold,
    positive_es,
    split_by_kappa,
    split_by_er,
)


class StandardPowerLawProtocolTests(unittest.TestCase):
    def test_event_pair_contract_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            EventDrops(er=[1.0, 2.0], es=[1.0])

    def test_classification_uses_er_before_filtering_es(self):
        drops = EventDrops(
            er=np.array([0.5, 2.0, 3.0, np.nan]),
            es=np.array([-1.0, np.nan, 4.0, 5.0]),
        )

        split = split_by_er(drops, er_det=2.0)

        np.testing.assert_array_equal(
            split.is_rev, [True, False, False, False]
        )
        np.testing.assert_array_equal(
            split.is_irrev, [False, True, True, False]
        )

        np.testing.assert_array_equal(positive_es(drops, split.is_rev), [])
        np.testing.assert_array_equal(positive_es(drops, split.is_irrev), [4.0])

    def test_threshold_is_a_classification_boundary(self):
        drops = EventDrops(er=[1.0, 2.0, 3.0], es=[1.0, 2.0, 3.0])
        split = split_by_er(drops, er_det=2.0)

        self.assertEqual(np.count_nonzero(split.is_rev), 1)
        self.assertEqual(np.count_nonzero(split.is_irrev), 2)

    def test_kappa_detector_classifies_before_filtering_es(self):
        drops = EventDrops(
            er=np.array([1.0, 2.0, 3.0, np.nan]),
            es=np.array([-1.0, np.nan, 4.0, 5.0]),
            kappa=np.array([0.5, 2.0, 3.0, 1.0]),
        )
        split = split_by_kappa(drops, kappa_det=2.0)

        np.testing.assert_array_equal(split.is_rev, [True, False, False, False])
        np.testing.assert_array_equal(split.is_irrev, [False, True, True, False])
        np.testing.assert_array_equal(positive_es(drops, split.is_irrev), [4.0])

    def test_default_kappa_detector_is_mu_over_two(self):
        self.assertAlmostEqual(kappa_detection_threshold(mu=6.0), 3.0)


if __name__ == "__main__":
    unittest.main()
