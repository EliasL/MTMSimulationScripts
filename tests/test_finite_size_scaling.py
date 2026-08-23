import unittest

import numpy as np

from MTMath.finiteSizeScaling import (
    build_log_pdf_curves,
    collapse_variance,
    filter_by_xmin,
    fit_moment_scaling,
    optimize_collapse,
)


class FiniteSizeScalingTests(unittest.TestCase):
    def test_collapse_variance_is_small_for_generating_parameters(self):
        tau = 1.35
        dimension = 0.8
        scaled_x = np.logspace(-3, 0, 80)
        scaled_density = scaled_x ** (-tau) * np.exp(-scaled_x)
        curves = {
            size: (
                scaled_x * size**dimension,
                scaled_density / size ** (dimension * tau),
            )
            for size in (50, 100, 200)
        }
        self.assertLess(collapse_variance(curves, tau, dimension), 1e-20)
        self.assertGreater(collapse_variance(curves, tau + 0.2, dimension + 0.2), 1e-4)

    def test_optimizer_recovers_generating_parameters(self):
        tau = 1.4
        dimension = 0.75
        scaled_x = np.logspace(-3, 0, 60)
        scaled_density = scaled_x ** (-tau) * np.exp(-scaled_x)
        curves = {
            size: (
                scaled_x * size**dimension,
                scaled_density / size ** (dimension * tau),
            )
            for size in (50, 100, 150, 200, 250)
        }
        result = optimize_collapse(
            curves,
            exponent_range=(1.0, 1.8),
            dimension_range=(0.4, 1.1),
            coarse_points=31,
            fine_points=41,
        )
        self.assertAlmostEqual(result["exponent"], tau, delta=0.02)
        self.assertAlmostEqual(result["dimension"], dimension, delta=0.02)
        self.assertFalse(result["boundary"])

    def test_moment_scaling_recovers_dimension_for_scaled_samples(self):
        rng = np.random.default_rng(7)
        base = rng.lognormal(mean=0.0, sigma=0.4, size=200_000)
        dimension = 0.7
        samples = {size: base * size**dimension for size in (50, 100, 150, 200)}
        result = fit_moment_scaling(samples, orders=(1, 2, 3))
        self.assertAlmostEqual(result["dimension"], dimension, delta=0.01)

    def test_log_pdf_builds_one_curve_per_size(self):
        values = {size: np.logspace(-4, -1, 100) * size for size in (50, 100, 200)}
        curves = build_log_pdf_curves(values, bins_per_decade=8)
        self.assertEqual(list(curves), [50.0, 100.0, 200.0])
        self.assertEqual(curves[50.0].count, 100)

    def test_filter_by_xmin_keeps_each_size_tail(self):
        values = {50: np.arange(1.0, 6.0), 100: np.arange(2.0, 12.0)}
        tails = filter_by_xmin(values, {50: 3.0, 100: 7.0})
        np.testing.assert_array_equal(tails[50.0], [3.0, 4.0, 5.0])
        np.testing.assert_array_equal(tails[100.0], [7.0, 8.0, 9.0, 10.0, 11.0])


if __name__ == "__main__":
    unittest.main()
