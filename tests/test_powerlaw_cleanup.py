import tempfile
import unittest
import gzip
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from MTMath.evaluatePowerlawFit import (
    Fit,
    Truncated_Power_Law,
    evaluate_xmin,
    evaluate_xmin_distances,
)
from Plotting.energyDropCalculations import calculate_energy_step_data
from Plotting import energyDropCalculations
from Plotting.findXmin import (
    DEFAULT_XMIN_COMPARISON_STRATEGIES,
    XMIN_STRATEGIES,
    _log_xmin_candidates,
    annotate_xmin_choices,
    compare_xmin_strategies,
    find_xmin_dks_from_results,
    find_xmin_global_min,
    find_xmin_refined_global_min_from_results,
    find_xmin_simple_drop,
    find_xmin_simple_drop_from_results,
    select_global_min_from_search_details,
)
from Plotting.plotPowerLaw import (
    get_energy_drops,
    make_fit,
    plot_data_and_fit,
    plot_energy_drop_trace,
    plot_ks_distance,
    plot_ks_distance_marker,
)


class EnergyDropTests(unittest.TestCase):
    def test_simple_shear_tangent_uses_spatial_configuration(self):
        material_tensor = np.zeros((1, 2, 2, 2, 2), dtype=float)
        material_tensor[..., 0, 1, 0, 1] = 3.5
        with mock.patch.object(
            energyDropCalculations.ContiEnergy,
            "elasticity_tensor",
            return_value=material_tensor,
        ) as elasticity_tensor:
            tangent = energyDropCalculations._simple_shear_tangent(
                np.array([0.1])
            )

        np.testing.assert_allclose(tangent, [3.5])
        self.assertTrue(elasticity_tensor.call_args.kwargs["eulerian"])

    def test_second_order_drop_and_average_scaling(self):
        total_df = pd.DataFrame(
            {
                "load": [0.0, 0.1],
                "total_energy": [10.0, 10.5],
                "avg_sigma12": [2.0, 2.0],
                "avg_P12": [99.0, 99.0],
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
            total, total_info = calculate_energy_step_data(
                df=total_df, metadata={"L": 2}, average_energy=False
            )
            average, _ = calculate_energy_step_data(
                df=average_df, metadata={"L": 2}, average_energy=True
            )

        self.assertAlmostEqual(total["E_ip1_pred_second_order"].iloc[0], 10.88)
        self.assertEqual(total_info["cauchy_col"], "avg_sigma12")
        self.assertFalse(total_info["used_piola_stress"])
        self.assertAlmostEqual(total["stress_corrected_drop_second_order"].iloc[0], 0.38)
        self.assertAlmostEqual(
            average["stress_corrected_drop_second_order"].iloc[0], 0.38 / 8.0
        )

    def test_energy_prediction_does_not_fall_back_to_average_piola(self):
        df = pd.DataFrame(
            {
                "load": [0.0, 0.1],
                "total_energy": [0.0, 0.1],
                "avg_P12": [1.0, 1.0],
            }
        )

        with self.assertRaisesRegex(KeyError, "Do not substitute 'avg_P12'"):
            calculate_energy_step_data(
                df=df, metadata={"L": 2}, average_energy=False
            )

    def test_unsupported_energy_function_raises(self):
        df = pd.DataFrame(
            {
                "load": [0.0, 0.1],
                "total_energy": [0.0, 0.1],
                "avg_sigma12": [0.0, 0.1],
                "avg_P12": [0.0, 0.1],
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
            {
                "min_ks",
                "global_min",
                "dip",
                "max_p",
                "simpleDrop",
                "derivative",
                "dks",
                "slope",
                "sizer",
                "sylvain",
            }
            <= set(XMIN_STRATEGIES)
        )
        self.assertNotIn("plateau", XMIN_STRATEGIES)

    def test_default_comparison_runs_global_min_last(self):
        self.assertEqual(
            DEFAULT_XMIN_COMPARISON_STRATEGIES,
            ("simpleDrop", "slope", "global_min"),
        )

    def test_simple_drop_starts_from_exactly_100_initial_measurements(self):
        calls = []

        def fake_evaluate_xmin(_drops, candidates, **_kwargs):
            candidates = np.asarray(candidates, dtype=float)
            calls.append(candidates.copy())
            log_candidates = np.log10(candidates)
            distances = (
                0.5
                + 0.02 * (log_candidates - 1.0) ** 2
                - 0.25 * (candidates >= 5.0)
            )
            return (
                distances,
                [[] for _ in candidates],
                np.ones(candidates.size, dtype=bool),
            )

        with mock.patch(
            "Plotting.findXmin.evaluate_xmin_distances",
            side_effect=fake_evaluate_xmin,
        ):
            xmin, details = find_xmin_simple_drop(
                np.geomspace(1.0, 1000.0, 500),
            )

        self.assertEqual(calls[0].size, 100)
        self.assertEqual(details["nr_initial"], 100)
        self.assertEqual(
            len(details["simple_drop_details"]["local_minima"]),
            3,
        )
        self.assertAlmostEqual(xmin, 10.0, delta=0.2)

    def test_simple_drop_does_not_hide_finite_ks_jump_with_noise_flag(self):
        def fake_evaluate_xmin(_drops, candidates, **_kwargs):
            candidates = np.asarray(candidates, dtype=float)
            distances = (np.log10(candidates) - np.log10(5.0)) ** 2
            return (
                distances,
                [[] for _ in candidates],
                np.ones(candidates.size, dtype=bool),
            )

        with mock.patch(
            "Plotting.findXmin.evaluate_xmin_distances",
            side_effect=fake_evaluate_xmin,
        ):
            _, details = find_xmin_simple_drop_from_results(
                np.geomspace(1.0, 100.0, 100),
                [1.0, 2.0, 4.0, 8.0, 16.0],
                [0.5, 0.6, 0.2, 0.21, 0.22],
                [True, False, False, True, True],
            )

        self.assertEqual(details["largest_drop_interval"], (2.0, 4.0))

    def test_refined_global_min_searches_every_rough_local_minimum(self):
        def distance_function(candidates):
            log_candidates = np.log10(np.asarray(candidates, dtype=float))
            return np.minimum(
                0.10 + (log_candidates - np.log10(2.5)) ** 2,
                0.02 + (log_candidates - 1.0) ** 2,
            )

        xmins = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        distances = distance_function(xmins)

        def fake_evaluate_xmin(_drops, candidates, **_kwargs):
            candidates = np.asarray(candidates, dtype=float)
            return (
                distance_function(candidates),
                [[] for _ in candidates],
                np.ones(candidates.size, dtype=bool),
            )

        with mock.patch(
            "Plotting.findXmin.evaluate_xmin_distances",
            side_effect=fake_evaluate_xmin,
        ):
            xmin, details = find_xmin_refined_global_min_from_results(
                np.geomspace(1.0, 100.0, 500),
                xmins,
                distances,
                np.ones(xmins.size, dtype=bool),
            )

        self.assertEqual(
            [item["index"] for item in details["rough_local_minima"]],
            [1, 3],
        )
        self.assertEqual(len(details["local_minima"]), 2)
        self.assertAlmostEqual(xmin, 10.0, delta=0.2)
        self.assertLess(details["selected_distance"], float(np.min(distances)))

    def test_refined_global_min_keeps_finite_noise_flagged_minimum(self):
        xmins = np.asarray([1.0, 2.0, 4.0])
        distances = np.asarray([0.5, 0.1, 0.3])

        def fake_evaluate_xmin(_drops, candidates, **_kwargs):
            candidates = np.asarray(candidates, dtype=float)
            values = 0.1 + (np.log10(candidates) - np.log10(2.0)) ** 2
            return (
                values,
                [[] for _ in candidates],
                np.zeros(candidates.size, dtype=bool),
            )

        with mock.patch(
            "Plotting.findXmin.evaluate_xmin_distances",
            side_effect=fake_evaluate_xmin,
        ):
            _, details = find_xmin_refined_global_min_from_results(
                np.geomspace(1.0, 10.0, 100),
                xmins,
                distances,
                [True, False, True],
            )

        self.assertEqual(
            [item["index"] for item in details["rough_local_minima"]],
            [1],
        )
        self.assertTrue(np.isfinite(details["selected_distance"]))

    def test_global_min_is_selected_after_all_searches_finish(self):
        simple_drop = {
            "evaluated_xmins": [2.0, 2.5],
            "evaluated_distances": [0.20, 0.08],
        }
        refined_global = {
            "evaluated_xmins": [1.0, 4.0, 5.0],
            "evaluated_distances": [0.30, 0.10, 0.05],
        }

        xmin, distance, evaluations = select_global_min_from_search_details(
            simple_drop,
            refined_global,
        )

        self.assertEqual(xmin, 5.0)
        self.assertEqual(distance, 0.05)
        self.assertEqual(len(evaluations), 5)

    def test_log_candidate_ceiling_leaves_one_decade_of_tail(self):
        _, candidates = _log_xmin_candidates(
            [1.0, 2.0, 100.0], samples_per_decade=10
        )
        self.assertAlmostEqual(candidates[-1], 10.0)

    def test_global_min_samples_every_tenth_observed_candidate(self):
        captured = {}

        def fake_evaluate_xmin(_drops, candidates, **_kwargs):
            captured.setdefault("candidates", np.asarray(candidates))
            return -np.asarray(candidates), [[] for _ in candidates], np.ones(
                len(candidates), dtype=bool
            )

        with mock.patch(
            "Plotting.findXmin.evaluate_xmin_distances",
            side_effect=fake_evaluate_xmin,
        ):
            xmin, _ = find_xmin_global_min(np.arange(1.0, 1001.0))

        np.testing.assert_allclose(captured["candidates"], np.arange(1.0, 101.0, 10.0))
        self.assertEqual(xmin, 91.0)

    def test_lightweight_xmin_batches_match_full_fixed_xmin_fits(self):
        drops = np.geomspace(1.0, 100.0, 80)
        candidates = np.array([1.0, 2.0, 5.0])
        full_fits = evaluate_xmin(drops, candidates)
        distances, _, _ = evaluate_xmin_distances(drops, candidates)
        np.testing.assert_allclose(distances, [fit.D for fit in full_fits], atol=1e-12)

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

    def test_make_fit_uses_canonical_analysis_and_parallel_control(self):
        class FakeFit:
            def __init__(self, data, xmin=None, **kwargs):
                self.kwargs = kwargs
                self.xmin = xmin

        analysis = {
            "simple_drop_xmin": 2.0,
            "global_min_xmin": 3.0,
        }
        with mock.patch(
            "Plotting.plotPowerLaw.analyze_xmin",
            return_value=analysis,
        ) as analyze, mock.patch("Plotting.plotPowerLaw.Fit", FakeFit):
            fit = make_fit(
                [1.0, 2.0, 3.0],
                distType=Truncated_Power_Law,
                use_cache=False,
                parallel_xmin=True,
                xmin_search_kwargs={"nr_initial": 100},
            )
        self.assertEqual(fit.xmin, 2.0)
        self.assertIs(fit.xmin_analysis, analysis)
        analyze.assert_called_once_with(
            [1.0, 2.0, 3.0],
            distType=Truncated_Power_Law,
            parallel=True,
            nr_initial=100,
        )

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
        self.assertIsNone(payload["xmin_analysis"])

    def test_global_xmin_annotation_is_conditional(self):
        fig, ax = plt.subplots()
        annotate_xmin_choices(
            ax,
            {
                "simple_drop_xmin": 2.0,
                "global_min_xmin": 4.0,
            },
        )
        self.assertEqual(len(ax.lines), 2)
        self.assertIn("Global min.", ax.lines[1].get_label())
        plt.close(fig)

        fig, ax = plt.subplots()
        annotate_xmin_choices(
            ax,
            {
                "simple_drop_xmin": 2.0,
                "global_min_xmin": 2.0,
            },
        )
        self.assertEqual(len(ax.lines), 1)
        plt.close(fig)


class FlowchartPlotControlTests(unittest.TestCase):
    def test_fit_annotations_title_and_legend_can_be_hidden(self):
        data = np.geomspace(1.0, 100.0, 80)
        fit = Fit(
            data,
            xmin=1.0,
            xmin_distribution=Truncated_Power_Law.name,
            verbose=0,
        )
        fig, ax = plt.subplots()
        plot_data_and_fit(
            fit,
            ax=ax,
            color="C3",
            data_color="C3",
            save=False,
            close=False,
            show_fit_region=False,
            show_cutoff=False,
            show_title=False,
            show_legend=False,
        )
        self.assertEqual(ax.get_title(), "")
        self.assertIsNone(ax.get_legend())
        self.assertEqual(len(ax.patches), 0)
        self.assertEqual(ax.lines[0].get_color(), "C3")
        self.assertEqual(ax.lines[1].get_color(), "C3")
        plt.close(fig)

    def test_ks_panel_title_and_legend_can_be_hidden(self):
        data = np.geomspace(1.0, 100.0, 80)
        fig, ax = plt.subplots()
        plot_ks_distance(
            data,
            xmin=1.0,
            ax=ax,
            save=False,
            close=False,
            set_title=False,
            show_legend=False,
            empirical_color="C3",
            show_inset=True,
        )
        self.assertEqual(ax.get_title(), "")
        self.assertIsNone(ax.get_legend())
        self.assertEqual(ax.lines[0].get_color(), "C3")
        self.assertEqual(len(ax.child_axes), 1)
        self.assertEqual(len(ax.child_axes[0].get_xticks()), 0)
        plt.close(fig)

    def test_ks_marker_checks_both_sides_of_empirical_jumps(self):
        fig, ax = plt.subplots()
        distance = plot_ks_distance_marker(
            ax,
            sorted_data=np.array([0.4, 0.9]),
            ecdf=np.array([0.5, 0.0]),
            model_ccdf=np.array([0.6, 0.1]),
        )
        self.assertAlmostEqual(distance, 0.4)
        plt.close(fig)

    def test_energy_drop_inset_ticks_can_be_suppressed(self):
        strain = np.linspace(0.0, 1.0, 101)
        energy = strain**2
        drop_strain = strain[1:]
        drops = np.full(drop_strain.shape, 1.0e-3)
        drops[50] = 1.0
        fig, ax = plt.subplots()
        drop_ax, inset_ax, inset_drop_ax = plot_energy_drop_trace(
            ax,
            strain,
            energy,
            drop_strain,
            drops,
            min_drop=1.0e-4,
            inset_show_y_ticks=False,
        )
        self.assertEqual(drop_ax.get_yscale(), "linear")
        self.assertEqual(drop_ax.lines[0].get_color(), "C1")
        self.assertEqual(drop_ax.lines[0].get_linestyle(), "-")
        self.assertAlmostEqual(inset_ax.patch.get_alpha(), 0.9)
        self.assertFalse(any(label.get_visible() for label in inset_ax.get_yticklabels()))
        self.assertFalse(
            any(label.get_visible() for label in inset_drop_ax.get_yticklabels())
        )
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
