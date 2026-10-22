import tempfile
import unittest
import json
import gzip
import os
from unittest.mock import patch
from pathlib import Path

import meshio
import numpy as np
import pandas as pd

from Plotting.reconnectingEnergyJumpAndElementDistribution import (
    _edge_flip_element_pairs,
    _completed_simulation_peaks,
    _log_shift_bins,
    _post_yield_energy_stress_frame,
    build_element_lineage_summary,
    read_live_macro_snapshot,
    read_vtu_pair,
    read_vtu_pair_details,
    select_pairs,
)
from Plotting.dataFunctions import get_data_from_name
from Management.reconnectionElementSnapshot import compact_flip_pair, main, read_dump_load


class ReconnectingEnergyJumpTests(unittest.TestCase):
    def test_dump_selection_reads_precise_strain_across_stream_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dump_l0.90.xml.gz"
            with gzip.open(path, "wb") as stream:
                stream.write(b" " * (1024 * 1024 - 3))
                stream.write(b"<load>0.90494</load>")
            self.assertEqual(read_dump_load(path), 0.90494)
            with gzip.open(path, "wb") as stream:
                stream.write(b"<load>nan</load>")
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                read_dump_load(path)

    def test_replay_cli_refuses_to_run_outside_slurm(self):
        args = ["replay", "--manifest", "missing.json", "--executable", "missing",
                "--output-root", "missing", "--task-index", "0", "--nr-threads", "1"]
        with patch.dict(os.environ, {}, clear=True), patch("sys.argv", args):
            with self.assertRaisesRegex(RuntimeError, "must run through Slurm"):
                main()

    def test_replay_cli_requires_one_cpu_and_one_dump(self):
        args = ["replay", "--manifest", "missing.json", "--executable", "missing",
                "--output-root", "missing", "--task-index", "0", "--nr-threads", "1"]
        with patch.dict(os.environ, {"SLURM_JOB_ID": "1", "SLURM_CPUS_PER_TASK": "2"}, clear=True), patch("sys.argv", args):
            with self.assertRaisesRegex(RuntimeError, "one dump with one CPU"):
                main()

    def test_overlapping_checkpoints_do_not_duplicate_a_flip_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "host/samples/run"
            folder.mkdir(parents=True)
            dumps = ["dump_l0.7.xml.gz", "dump_l0.8.xml.gz"]
            metadata = {"status": "complete", "load": 0.81,
                        "before_vtu": "mesh_load=0.81_minStep=1.2_pre.42.vtu",
                        "after_vtu": "mesh_load=0.81_minStep=1.2_post.42.vtu"}
            identity = np.broadcast_to(np.eye(2), (2, 2, 2)).copy()
            data = {"element_indices": np.array([8, 13]),
                    "before_connectivity": np.array([[0, 1, 2], [0, 2, 3]]),
                    "after_connectivity": np.array([[0, 1, 3], [1, 2, 3]]),
                    "before_energy": np.array([0.2, 0.3]),
                    "after_energy": np.array([0.25, 0.28]),
                    "before_stored_sigma12": np.array([0.1, 0.2]),
                    "after_stored_sigma12": np.array([0.11, 0.19]),
                    "before_T": identity, "after_T": identity,
                    "before_F_E": identity, "after_F_E": identity}
            for dump in dumps:
                path = folder / (dump + ".npz")
                np.savez_compressed(path, **data)
                path.with_suffix(".json").write_text(json.dumps(metadata))
            plan = root / "plan.json"
            plan.write_text(json.dumps({"sampling": "test", "runs": [{
                "simulation_name": "run", "host": "host.example", "size": 10,
                "seed": 0, "yield_load": 0.6, "max_load": 1, "dumps": dumps,
            }]}))
            result = build_element_lineage_summary(root, plan, root / "lineage.csv")
            frame = pd.read_csv(result)
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame["before_element_index"].tolist(), [8, 13])
            np.testing.assert_allclose(frame["short_energy_shift"], [0.05, -0.02])
            report = json.loads(result.with_suffix(".sampling.json").read_text())
            self.assertEqual(report["checkpoints"][1]["status"], "duplicate_cycle")

    def test_completed_runs_require_finite_endpoint_at_configured_target(self):
        peaks = pd.DataFrame({
            "simulation_name": ["done", "running", "other_target", "missing"],
            "last_load": [1.0, 0.95, 0.5, np.nan],
            "max_load": [1.0, 1.0, 0.5, 1.0],
        })
        selected = _completed_simulation_peaks(peaks)
        self.assertEqual(selected["simulation_name"].tolist(), ["done", "other_target"])
        with self.assertRaisesRegex(ValueError, "requires columns"):
            _completed_simulation_peaks(peaks.drop(columns="last_load"))

    def test_post_yield_selection_uses_each_peak_and_replaces_stale_labels(self):
        frame = pd.DataFrame({
            "load": [0.4, 0.6, 0.61, 0.7],
            "yield_load": [0.3, 0.6, 0.6, 0.8],
            "regime": ["pre-yield", "post-yield", "pre-yield", "post-yield"],
        })
        selected = _post_yield_energy_stress_frame(frame)
        self.assertEqual(selected.index.tolist(), [0, 2])
        self.assertEqual(selected["regime"].tolist(), ["post-yield", "post-yield"])
        self.assertEqual(frame.loc[0, "regime"], "pre-yield")

    def test_post_yield_selection_requires_a_valid_peak(self):
        with self.assertRaisesRegex(ValueError, "requires columns"):
            _post_yield_energy_stress_frame(pd.DataFrame({"load": [0.7]}))
        with self.assertRaisesRegex(ValueError, "finite"):
            _post_yield_energy_stress_frame(pd.DataFrame({
                "load": [0.7], "yield_load": [np.nan],
            }))

    def test_live_csv_reader_drops_partial_final_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "macroData.csv"
            path.write_bytes(
                b"load,avg_sigma12\n"
                b"0.1,0.3\n"
                b"0.2,0.4"
            )
            original = path.read_bytes()
            frame = read_live_macro_snapshot(path)
            self.assertEqual(len(frame), 1)
            self.assertEqual(float(frame["load"].iloc[0]), 0.1)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(path.with_name("macroData_fixed.csv").exists())

    def test_live_csv_reader_ignores_undeclared_trailing_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "macroData.csv"
            path.write_text(
                "load_step,load,total_energy_change,avg_sigma12\n"
                "7,0.2,-3.0,0.4,unheadered,reversibility,diagnostics\n"
            )
            frame = read_live_macro_snapshot(
                path,
                columns=("load_step", "load", "total_energy_change"),
            )
            self.assertEqual(frame.columns.tolist(), [
                "load_step", "load", "total_energy_change",
            ])
            self.assertEqual(frame.iloc[0].to_dict(), {
                "load_step": "7",
                "load": 0.2,
                "total_energy_change": "-3.0",
            })

    def test_explicit_load_increment_supports_nested_protocol_state_paths(self):
        result = get_data_from_name(
            "/missing/job/data/reversibilityData/irrev_drop_l_1.2/state0_min_gamma.3.vtu",
            load_increment=1e-5,
        )
        self.assertEqual(result["loadIncrement"], 1e-5)

    def test_reconnection_pairs_are_selected_by_numeric_min_step(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for min_step in ("100.200", "20.40"):
                (folder / f"mesh_load=0.2_minStep={min_step}_pre.5.vtu").touch()
                (folder / f"mesh_load=0.2_minStep={min_step}_post.5.vtu").touch()
            selected = select_pairs(folder, 1)
            self.assertEqual(len(selected), 1)
            self.assertIn("minStep=20.40", selected[0][1][0].name)
            self.assertEqual(len(select_pairs(folder, "all")), 2)
            with self.assertRaisesRegex(ValueError, "contains only 2"):
                select_pairs(folder, 3)

    def test_shift_bins_are_logarithmically_spaced_magnitudes(self):
        edges = _log_shift_bins(np.array([-1e-6, 1e-4, -1e-2]))
        self.assertTrue(np.all(edges > 0.0))
        np.testing.assert_allclose(edges[1:] / edges[:-1], edges[1] / edges[0])
        self.assertLessEqual(edges[0], 1e-6)
        self.assertGreaterEqual(edges[-1], 1e-2)

    def test_edge_flip_pairing_ignores_static_elements(self):
        before = np.array([[0, 1, 2], [0, 2, 3], [3, 4, 5]])
        after = np.array([[0, 1, 3], [1, 2, 3], [3, 4, 5]])
        pairs = _edge_flip_element_pairs(
            before, after, Path("before.vtu"), Path("after.vtu")
        )
        np.testing.assert_array_equal(pairs, [[0, 1]])

    def test_vtu_pair_uses_reconnected_elements_and_global_energy_stress(self):
        with tempfile.TemporaryDirectory() as directory:
            before_path = Path(directory) / (
                "simpleShear,s1x1l0.15,1e-05,5PBCt5s0_"
                "load=0.2_minStep=1.2_pre.3.vtu"
            )
            after_path = Path(str(before_path).replace("_pre.", "_post."))
            points = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            )
            before_cells = [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))]
            after_cells = [("triangle", np.array([[0, 1, 3], [1, 2, 3]]))]

            def cell_data(energy, p12):
                return {
                    "energy_field": [np.asarray(energy)],
                    "sigma12": [np.asarray(p12) * 2.0],
                    "C11": [np.array([1.0, 1.1])],
                    "C12": [np.array([0.0, 0.05])],
                    "C22": [np.array([1.0, 1.0])],
                    "G11": [np.array([1.0, 1.1])],
                    "G12": [np.array([0.0, 0.05])],
                    "G22": [np.array([1.0, 1.0])],
                    "T11": [np.array([1.0, 2.0])],
                    "T12": [np.array([1.0, 0.0])],
                    "T21": [np.array([0.0, 1.0])],
                    "T22": [np.array([1.0, 1.0])],
                    "F_E11": [np.ones(2)],
                    "F_E12": [np.zeros(2)],
                    "F_E21": [np.zeros(2)],
                    "F_E22": [np.ones(2)],
                    "nrm3": [np.array([0, 0])],
                }

            meshio.write(
                before_path,
                meshio.Mesh(
                    points,
                    before_cells,
                    point_data={"refIndex": np.arange(4)},
                    cell_data=cell_data([2.5, 3.5], [0.4, 0.6]),
                ),
                binary=False,
            )
            meshio.write(
                after_path,
                meshio.Mesh(
                    points,
                    after_cells,
                    point_data={"refIndex": np.arange(4)},
                    cell_data=cell_data([2.0, 3.0], [0.3, 0.5]),
                ),
                binary=False,
            )
            compact = compact_flip_pair(before_path, after_path)
            np.testing.assert_array_equal(compact["element_indices"], [0, 1])
            np.testing.assert_allclose(compact["before_energy"], [2.5, 3.5])
            np.testing.assert_allclose(compact["after_energy"], [2.0, 3.0])
            np.testing.assert_allclose(compact["before_stored_sigma12"], [0.8, 1.2])
            np.testing.assert_array_equal(compact["before_connectivity"], before_cells[0][1])
            np.testing.assert_array_equal(compact["after_connectivity"], after_cells[0][1])
            np.testing.assert_allclose(compact["before_F_E"] @ compact["before_T"],
                                       [[[1, 1], [0, 1]], [[2, 0], [1, 1]]])
            before, after = read_vtu_pair(
                before_path, after_path, "C", np.linspace(-1.0, 1.0, 11)
            )
            self.assertEqual(before.nr_elements, 2)
            self.assertEqual(before.nr_reconnected_elements, 2)
            self.assertEqual(before.total_energy, 6.0)
            self.assertEqual(after.total_energy, 5.0)
            self.assertEqual(before.average_sigma12, 1.0)
            self.assertEqual(after.average_sigma12, 0.8)
            self.assertEqual(before.flipped_total_energy, 6.0)
            self.assertEqual(after.flipped_total_energy, 5.0)
            self.assertEqual(before.flipped_average_sigma12, 1.0)
            self.assertEqual(after.flipped_average_sigma12, 0.8)
            self.assertEqual(int(before.histogram.sum()), 2)
            self.assertEqual(int(after.histogram.sum()), 2)
            before_T, after_T = read_vtu_pair(
                before_path, after_path, "T_total", np.linspace(-1.0, 1.0, 11)
            )
            self.assertEqual(int(before_T.histogram.sum()), 2)
            self.assertEqual(int(after_T.histogram.sum()), 2)
            _, _, local = read_vtu_pair_details(
                before_path, after_path, "G", np.linspace(-1.0, 1.0, 11)
            )
            np.testing.assert_array_equal(local.element_pairs, [[0, 1]])
            np.testing.assert_allclose(local.before_energy, [[2.5, 3.5]])
            np.testing.assert_allclose(local.short_after_energy, [[2.0, 3.0]])
            np.testing.assert_allclose(local.long_after_energy, [[3.0, 2.0]])
            np.testing.assert_allclose(local.before_sigma12, [[0.8, 1.2]])
            np.testing.assert_allclose(local.short_after_sigma12, [[0.6, 1.0]])
            np.testing.assert_allclose(local.long_after_sigma12, [[1.0, 0.6]])


if __name__ == "__main__":
    unittest.main()
