import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from Plotting.sizeScalingCollapse import (
    _read_mixed_selected,
    collapse_variance,
    exclude_size,
    fit_xmins,
    log_histogram,
    optimize_collapse,
    tail_histogram_curves,
)


class _Fit:
    def __init__(self, xmin):
        self.xmin = xmin


class SizeScalingCollapseTests(unittest.TestCase):
    def test_plateau_accuracy_controls_candidate_spacing(self):
        fake_fit = _Fit(1.0)
        with mock.patch(
            "Plotting.sizeScalingCollapse.make_fit", return_value=fake_fit
        ) as make_fit:
            fit_xmins({50: np.array([1.0, 2.0, 3.0])}, "plateau", 0.1, False, Path("cache"))

        self.assertEqual(
            make_fit.call_args.kwargs["xmin_strategy_kwargs"],
            {"samples_per_decade": 10.0, "tail_decades": 1.0},
        )

    def test_global_min_accuracy_controls_observed_candidate_stride(self):
        fake_fit = _Fit(1.0)
        with mock.patch(
            "Plotting.sizeScalingCollapse.make_fit", return_value=fake_fit
        ) as make_fit:
            fit_xmins(
                {50: np.array([1.0, 2.0, 3.0])},
                "global_min",
                0.1,
                False,
                Path("cache"),
            )

        self.assertEqual(
            make_fit.call_args.kwargs["xmin_strategy_kwargs"],
            {
                "candidate_stride": 10,
                "tail_decades": 1.0,
            },
        )

    def test_mixed_header_reader_uses_each_segments_column_order(self):
        wanted = {
            "load",
            "total_energy",
            "total_energy_change",
            "total_e_change_from_init",
            "avg_sigma12",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.csv"
            path.write_text(
                "load,total_energy,total_energy_change,total_e_change_from_init,avg_sigmaxy\n"
                "0.1,10,1,-2,0.3\n"
                "#HEADER:avg_sigma12,load,total_e_change_from_init,total_energy_change,total_energy\n"
                "0.4,0.2,-3,2,20\n"
            )
            frame = _read_mixed_selected(path, wanted)

        np.testing.assert_allclose(frame["load"], [0.1, 0.2])
        np.testing.assert_allclose(frame["total_energy"], [10.0, 20.0])
        np.testing.assert_allclose(frame["avg_sigma12"], [0.3, 0.4])

    def test_exact_synthetic_collapse_prefers_generating_parameters(self):
        exponent = 1.35
        dimension = 0.8
        scaled_drop = np.logspace(-4, 0, 80)
        scaled_density = scaled_drop**-exponent * np.exp(-scaled_drop)
        curves = {
            size: (
                scaled_drop * size**dimension,
                scaled_density / size ** (dimension * exponent),
            )
            for size in (50, 100, 200)
        }

        true_quality = collapse_variance(curves, exponent, dimension)
        wrong_quality = collapse_variance(curves, exponent + 0.3, dimension + 0.2)

        self.assertLess(true_quality, 1e-25)
        self.assertGreater(wrong_quality, true_quality + 1e-4)

    def test_optimizer_recovers_exact_synthetic_collapse(self):
        exponent = 1.4
        dimension = 0.75
        scaled_drop = np.logspace(-4, 0, 60)
        scaled_density = scaled_drop**-exponent * np.exp(-scaled_drop)
        curves = {
            size: (
                scaled_drop * size**dimension,
                scaled_density / size ** (dimension * exponent),
            )
            for size in (50, 100, 150, 200, 250)
        }
        with tempfile.TemporaryDirectory() as directory:
            result = optimize_collapse(
                curves, Path(directory) / "synthetic.npz", force=True
            )

        self.assertAlmostEqual(float(result["x"]), exponent, delta=0.01)
        self.assertAlmostEqual(float(result["dimension"]), dimension, delta=0.01)
        self.assertFalse(bool(result["boundary"]))

    def test_tail_histogram_is_recomputed_after_xmin_filter(self):
        curves = tail_histogram_curves(
            {50: np.array([1.0, 2.0, 4.0, 8.0])},
            {50: 2.0},
            bins_per_decade=10,
        )
        expected = log_histogram([2.0, 4.0, 8.0], bins_per_decade=10)
        np.testing.assert_allclose(curves[50][0], expected[0])
        np.testing.assert_allclose(curves[50][1], expected[1])

    def test_log_histogram_rejects_too_few_positive_values(self):
        with self.assertRaisesRegex(ValueError, "three positive drops"):
            log_histogram([-1.0, 0.0, 1.0, np.nan])

    def test_exclude_size_keeps_other_collapse_curves(self):
        curves = {size: (np.ones(3), np.ones(3)) for size in (50, 100, 150, 200)}
        self.assertEqual(set(exclude_size(curves, 50)), {100, 150, 200})

    def test_exclude_size_rejects_missing_size(self):
        with self.assertRaisesRegex(ValueError, "missing system size"):
            exclude_size({100: (), 150: (), 200: ()}, 50)


if __name__ == "__main__":
    unittest.main()
