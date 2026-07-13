import tempfile
import unittest
import gzip
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from MTMath.evaluatePowerlawFit import Truncated_Power_Law
from Plotting.energyDropCalculations import calculate_energy_step_data
from Plotting.findXmin import (
    XMIN_STRATEGIES,
    compare_xmin_strategies,
    find_xmin_dks_from_results,
)
from Plotting.plotPowerLaw import get_energy_drops, make_fit


class EnergyDropTests(unittest.TestCase):
    def test_second_order_drop_and_average_scaling(self):
        total_df = pd.DataFrame(
            {
                "load": [0.0, 0.1],
                "total_energy": [10.0, 10.5],
                "avg_sigma12": [2.0, 2.0],
            }
        )
        average_df = total_df.rename(columns={"total_energy": "avg_energy"}).copy()
        average_df["avg_energy"] /= 8.0

        with mock.patch(
            "Plotting.energyDropCalculations._simple_shear_tangent",
            return_value=np.array([4.0]),
        ), mock.patch(
            "Plotting.energyDropCalculations._simple_shear_tangent_gamma0",
            return_value=np.array([4.0]),
        ):
            total, _ = calculate_energy_step_data(
                df=total_df, metadata={"L": 2}, average_energy=False
            )
            average, _ = calculate_energy_step_data(
                df=average_df, metadata={"L": 2}, average_energy=True
            )

        self.assertAlmostEqual(total["E_ip1_pred_second_order"].iloc[0], 10.88)
        self.assertAlmostEqual(total["stress_corrected_drop_second_order"].iloc[0], 0.38)
        self.assertAlmostEqual(
            average["stress_corrected_drop_second_order"].iloc[0], 0.38 / 8.0
        )

    def test_unsupported_energy_function_raises(self):
        df = pd.DataFrame(
            {
                "load": [0.0, 0.1],
                "total_energy": [0.0, 0.1],
                "avg_sigma12": [0.0, 0.1],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "config.conf").write_text(
                "energyFunction = contiTriangular\nbulkModulus = 4\n"
            )
            with self.assertRaisesRegex(ValueError, "contiTriangular"):
                calculate_energy_step_data(
                    directory / "macroData.csv", df=df, metadata={"L": 2}
                )

    def test_plastic_event_alias_and_explicit_positive_sign(self):
        df = pd.DataFrame(
            {
                "load_step": [1, 2, 3],
                "load": [0.0, 0.01, 0.02],
                "total_energy": [0.0, 0.0, 0.0],
                "total_energy_change": [0.0, 1.0, 2.0],
                "avg_sigma12": [0.0, 0.0, 0.0],
                "avg_P12": [0.0, 0.0, 0.0],
                "nr_elements_with_m3_change": [0, 1, 0],
                "LBFGS_Term_reason": [1, 1, 1],
                "CG_Term_reason": [0, 0, 0],
                "FIRE_Term_reason": [0, 0, 0],
            }
        )
        with tempfile.TemporaryDirectory(prefix="simpleShear,s2x2") as tmp:
            csv_path = str(Path(tmp) / "macroData.csv")
            df.to_csv(csv_path, index=False)
            drops, _ = get_energy_drops(
                csv_path,
                df=df,
                strainLim="all",
                onlyStrainedEnergyDrops=True,
            )
            self.assertEqual(len(drops), 1)

            negative, _ = get_energy_drops(
                csv_path,
                df=df,
                strainLim="all",
                stress_corrected=False,
                energy_type="energy_change",
            )
            positive, _ = get_energy_drops(
                csv_path,
                df=df,
                strainLim="all",
                stress_corrected=False,
                energy_type="energy_change",
                drop_sign="positive",
            )
            self.assertEqual(negative.size, 0)
            np.testing.assert_allclose(positive, [1.0, 2.0])


class XminCleanupTests(unittest.TestCase):
    def test_all_comparison_strategies_are_registered(self):
        self.assertTrue(
            {"min_ks", "dip", "max_p", "plateau", "derivative", "dks", "sizer", "sylvain"}
            <= set(XMIN_STRATEGIES)
        )

    def test_dks_reuses_existing_grid(self):
        xmin = find_xmin_dks_from_results(
            [1.0, 10.0, 100.0],
            [0.5, 0.1, 0.09],
        )
        self.assertEqual(xmin, 1.0)

    def test_named_strategy_comparison(self):
        with mock.patch.dict(
            XMIN_STRATEGIES,
            {"low": lambda drops: 2.0, "high": lambda drops: 4.0},
            clear=True,
        ):
            results = compare_xmin_strategies(
                [1.0, 2.0, 3.0, 4.0], strategies=("low", "high")
            )
        self.assertEqual(results["low"].n_tail, 3)
        self.assertEqual(results["high"].n_tail, 1)

    def test_make_fit_uses_accuracy_and_parallel_controls(self):
        class FakeFit:
            def __init__(self, data, **kwargs):
                self.kwargs = kwargs
                self.xmin = 1.0
                self.xmin_fitting_results = {}

        with mock.patch("Plotting.plotPowerLaw.Fit", FakeFit):
            fit = make_fit(
                [1.0, 2.0, 3.0],
                distType=Truncated_Power_Law,
                use_cache=False,
                fast_xmin=True,
                xmin_accuracy=0.1,
                parallel_xmin=True,
            )
        self.assertEqual(fit.kwargs["xmin_samples_per_decade"], 10.0)
        self.assertTrue(fit.kwargs["parallel_xmin"])

    def test_fixed_xmin_cache_does_not_require_search_diagnostics(self):
        class FakeFit:
            def __init__(self, data, xmin=None, **kwargs):
                self.xmin = xmin

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "Plotting.plotPowerLaw.Fit", FakeFit
        ):
            make_fit(
                [1.0, 2.0, 3.0],
                xmin_range=1.0,
                use_cache=True,
                cache_dir=directory,
            )
            cache_path = next(Path(directory).glob("*.json.gz"))
            with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)

        self.assertEqual(payload["xmin"], 1.0)
        self.assertIsNone(payload["xmin_fitting_results"])


if __name__ == "__main__":
    unittest.main()
