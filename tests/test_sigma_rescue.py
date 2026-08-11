import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Management.sigmaRescue import (
    DumpRecord,
    SchemaInterval,
    SourceRun,
    SegmentPlan,
    ValidationTolerance,
    build_prefix_plan,
    build_segment_plan,
    inspect_schema_intervals,
    inventory_dumps,
    select_prefix_dump,
    write_prefix_config,
    write_private_config,
    validate_segment,
)
from Plotting.energyDropCalculations import (
    calculate_energy_step_data,
    calculate_stress_step_data,
    infer_energy_prediction_stress_column,
    infer_stress_column,
    validate_sigma12_column,
)


class SigmaRescueTests(unittest.TestCase):
    def test_equal_sigma_and_piola_above_threshold_is_rejected(self):
        frame = pd.DataFrame(
            {
                "load": [0.2, 0.31],
                "avg_sigma12": [1.0, 2.0],
                "avg_P12": [1.0, 2.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "equal to avg_P12"):
            validate_sigma12_column(frame, context="fixture")

        frame["total_energy"] = [10.0, 10.1]
        with self.assertRaisesRegex(ValueError, "equal to avg_P12"):
            calculate_energy_step_data(
                df=frame,
                metadata={"L": 2},
                average_energy=False,
            )

    def test_small_strain_equality_is_not_the_high_strain_guard(self):
        frame = pd.DataFrame(
            {
                "load": [0.2],
                "avg_sigma12": [1.0],
                "avg_P12": [1.0],
            }
        )

        self.assertEqual(validate_sigma12_column(frame), "avg_sigma12")

    def test_piola_is_never_used_when_sigma_is_missing(self):
        frame = pd.DataFrame(
            {
                "load": [0.2, 0.4],
                "avg_P12": [1.0, 2.0],
            }
        )

        with self.assertRaisesRegex(KeyError, "refusing to substitute"):
            infer_stress_column(frame)
        with self.assertRaisesRegex(KeyError, "Do not substitute"):
            infer_energy_prediction_stress_column(frame)

    def test_distinct_sigma_and_piola_are_accepted(self):
        frame = pd.DataFrame(
            {
                "load": [0.31, 0.4],
                "avg_sigma12": [0.2, 0.3],
                "avg_P12": [0.25, 0.35],
            }
        )

        self.assertEqual(validate_sigma12_column(frame), "avg_sigma12")
        self.assertEqual(infer_stress_column(frame), "avg_sigma12")

    def test_energy_mismatch_marks_sigma_sentinel_and_continues(self):
        header = (
            "load_step,load,total_energy,avg_energy,avg_sigma12,"
            "avg_init_sigma12,avg_sigma12_change_from_init\n"
        )
        source_rows = "1,0.15,10,10,0.1,0.1,0\n2,0.16,9,9,0.2,0.2,0\n3,0.17,8,8,0.3,0.3,0\n"
        replay_rows = "1,0.15,10,10,0.11,0.11,0\n2,0.16,9.1,9.1,0.22,0.22,0\n3,0.17,8,8,0.33,0.33,0\n"
        source = SourceRun(
            name="run",
            size=50,
            seed=0,
            server="test",
            folder="/source/run",
            config_path="/source/run/config.conf",
            macro_path="/source/run/macroData.csv",
            config_sha256="config",
            macro_sha256="macro",
        )
        segment = SegmentPlan(
            segment_id="segment",
            run_name="run",
            size=50,
            seed=0,
            server="test",
            start_dump=DumpRecord("/source/run/dumps/dump.xml.gz", 0.14, 1, "dump", "test"),
            stop_load=0.17,
            expected_first_step=1,
            expected_last_step=3,
            expected_first_load=0.15,
            expected_last_load=0.17,
            output_directory="/rescue/segment",
        )

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path = directory / "source.csv"
            replay_path = directory / "replay.csv"
            source_path.write_text(header + source_rows)
            replay_path.write_text(header + replay_rows)
            validated = validate_segment(
                segment,
                replay_path,
                source_path,
                ValidationTolerance(),
            )
            result = json.loads((validated.parent / "result.json").read_text())
            frame = pd.read_csv(validated)

        self.assertEqual(result["status"], "validated_with_sentinels")
        self.assertEqual(result["invalid_rows"], 1)
        self.assertEqual(frame.loc[1, "avg_sigma12"], -1)
        self.assertEqual(frame.loc[1, "avg_init_sigma12"], -1)
        self.assertEqual(frame.loc[1, "avg_sigma12_change_from_init"], -1)

    def test_energy_and_stress_calculations_ignore_sigma_sentinel_steps(self):
        frame = pd.DataFrame(
            {
                "load": [0.2, 0.21, 0.22],
                "total_energy": [10.0, 9.9, 9.8],
                "avg_sigma12": [0.1, -1.0, 0.2],
                "avg_sigma12_change_from_init": [0.0, -1.0, 0.1],
            }
        )

        energy_steps, _ = calculate_energy_step_data(
            df=frame, metadata={"L": 2}, average_energy=False
        )
        stress_steps, _ = calculate_stress_step_data(df=frame)

        self.assertTrue(energy_steps["stress_corrected_drop_second_order"].isna().all())
        self.assertTrue(stress_steps["inter_strain_drop"].isna().all())

    def test_schema_intervals_preserve_old_and_new_eras(self):
        old_header = (
            "load_step,load,total_energy,total_energy_change,"
            "total_init_energy,total_e_change_from_init,avg_sigmaxy,avg_Pxy\n"
        )
        new_header = (
            "#HEADER:load_step,load,total_energy,total_energy_change,"
            "total_init_energy,total_e_change_from_init,avg_sigma12,avg_P12\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "macroData.csv"
            path.write_text(
                old_header
                + "1,0.15,10,0,10,0,0.1,0.2\n"
                + "2,0.31,11,1,10,1,0.11,0.21\n"
                + new_header
                + "3,0.32,12,1,10,2,0.22,0.23\n"
            )
            intervals = inspect_schema_intervals(path)

        self.assertEqual([interval.sigma_status for interval in intervals], ["bad-old", "correct-new"])
        self.assertEqual((intervals[0].first_step, intervals[0].last_step), (1, 2))
        self.assertEqual((intervals[1].first_step, intervals[1].last_step), (3, 3))
        self.assertEqual(intervals[0].header[-2:], ("avg_sigmaxy", "avg_Pxy"))
        self.assertEqual(intervals[1].header[-2:], ("avg_sigma12", "avg_P12"))

    def test_schema_parser_rejects_unknown_stress_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "macroData.csv"
            path.write_text("load_step,load,total_energy\n1,0.15,10\n")

            with self.assertRaisesRegex(ValueError, "exactly one sigma schema"):
                inspect_schema_intervals(path)

    def test_dump_inventory_is_sorted_and_fingerprinted(self):
        with tempfile.TemporaryDirectory() as directory:
            dump_dir = Path(directory) / "dumps"
            dump_dir.mkdir()
            (dump_dir / "dump_l0.20.xml.gz").write_bytes(
                gzip.compress(b"<load>0.20</load><loadSteps>2</loadSteps>")
            )
            (dump_dir / "dump_l0.15.xml.gz").write_bytes(
                gzip.compress(b"<load>0.15</load><loadSteps>1</loadSteps>")
            )

            records = inventory_dumps(dump_dir, format_era="test")

        self.assertEqual([record.load for record in records], [0.15, 0.2])
        self.assertTrue(all(len(record.sha256) == 64 for record in records))
        self.assertTrue(np.all(np.isfinite([record.load for record in records])))

    def test_segment_plan_covers_bad_rows_between_dumps(self):
        source = SourceRun(
            name="simpleShear,s50x50l0.15,1e-05,1.0PBCt2LBFGSEpsx1e-06s0",
            size=50,
            seed=0,
            server="test",
            folder="/source/run",
            config_path="/source/run/config.conf",
            macro_path="/source/run/macroData.csv",
            config_sha256="config",
            macro_sha256="macro",
        )
        interval = SchemaInterval(
            first_step=2,
            last_step=6,
            first_load=0.16,
            last_load=0.2,
            header=("load_step", "load", "avg_sigmaxy", "avg_Pxy"),
            sigma_status="bad-old",
            row_keys=((2, 0.16), (3, 0.17), (4, 0.18), (5, 0.19), (6, 0.2)),
        )
        dumps = [
            DumpRecord(f"/source/run/dumps/dump_l{load}.xml.gz", load, 1, str(load), "test")
            for load in (0.15, 0.17, 0.19)
        ]

        segments = build_segment_plan(source, [interval], dumps, Path("/rescue"))

        self.assertEqual(len(segments), 3)
        self.assertEqual(
            [(segment.expected_first_step, segment.expected_last_step) for segment in segments],
            [(2, 3), (4, 5), (6, 6)],
        )
        self.assertEqual([segment.start_dump.load for segment in segments], [0.15, 0.17, 0.19])

    def test_segment_plan_rejects_bad_rows_before_first_dump(self):
        source = SourceRun(
            name="run",
            size=50,
            seed=0,
            server="test",
            folder="/source/run",
            config_path="/source/run/config.conf",
            macro_path="/source/run/macroData.csv",
            config_sha256="config",
            macro_sha256="macro",
        )
        interval = SchemaInterval(
            first_step=1,
            last_step=1,
            first_load=0.15,
            last_load=0.15,
            header=("load_step", "load", "avg_sigmaxy", "avg_Pxy"),
            sigma_status="bad-old",
            row_keys=((1, 0.15),),
        )
        dump = DumpRecord("/source/run/dumps/dump_l0.2.xml.gz", 0.2, 1, "dump", "test")

        with self.assertRaisesRegex(ValueError, "before the first usable dump"):
            build_segment_plan(source, [interval], [dump], Path("/rescue"))

    def test_prefix_plan_covers_rows_before_first_dump(self):
        source = SourceRun(
            name="run",
            size=50,
            seed=0,
            server="test",
            folder="/source/run",
            config_path="/source/run/config.conf",
            macro_path="/source/run/macroData.csv",
            config_sha256="config",
            macro_sha256="macro",
        )
        interval = SchemaInterval(
            first_step=1,
            last_step=6,
            first_load=0.15,
            last_load=0.2,
            header=("load_step", "load", "avg_sigmaxy", "avg_Pxy"),
            sigma_status="bad-old",
            row_keys=((1, 0.15), (2, 0.16), (3, 0.17), (4, 0.18), (5, 0.19), (6, 0.2)),
        )
        dumps = [
            DumpRecord(
                "/source/run/dumps/dump_l0.17.xml.gz",
                0.17,
                1,
                "dump",
                "test",
                state_load=0.17,
                state_step=3,
            ),
            DumpRecord(
                "/source/run/dumps/dump_l0.19.xml.gz",
                0.19,
                1,
                "dump",
                "test",
                state_load=0.19,
                state_step=5,
            ),
        ]

        prefix = build_prefix_plan(source, [interval], dumps, Path("/rescue"))
        self.assertIsNotNone(prefix)
        assert prefix is not None
        self.assertEqual((prefix.expected_first_step, prefix.expected_last_step), (1, 3))
        self.assertEqual(prefix.stop_load, 0.17)

        segments = build_segment_plan(
            source, [interval], dumps, Path("/rescue"), prefix=prefix
        )
        self.assertEqual(
            [(segment.expected_first_step, segment.expected_last_step) for segment in segments],
            [(4, 5), (6, 6)],
        )
        self.assertEqual([segment.start_dump.state_step for segment in segments], [3, 5])

    def test_private_config_changes_only_output_controls_and_stop_load(self):
        source = SourceRun(
            name="run",
            size=50,
            seed=0,
            server="test",
            folder="/source/run",
            config_path="/source/run/config.conf",
            macro_path="/source/run/macroData.csv",
            config_sha256="config",
            macro_sha256="macro",
        )
        dump = DumpRecord("/source/run/dumps/dump_l0.2.xml.gz", 0.2, 1, "dump", "test")
        segment = build_segment_plan(
            source,
            [
                SchemaInterval(
                    first_step=2,
                    last_step=2,
                    first_load=0.2,
                    last_load=0.2,
                    header=("load_step", "load", "avg_sigmaxy", "avg_Pxy"),
                    sigma_status="bad-old",
                    row_keys=((2, 0.2),),
                )
            ],
            [dump],
            Path("/rescue"),
        )[0]

        with tempfile.TemporaryDirectory() as directory:
            source_config = Path(directory) / "config.conf"
            destination = Path(directory) / "segment" / "config.conf"
            source_config.write_text(
                "rows = 50\nminimizer = LBFGS\nLBFGSEpsx = 1e-6\n"
                "maxLoad = 1.0\nwriteDumps = 1\n"
            )
            write_private_config(source_config, destination, segment)
            private = destination.read_text()

        self.assertIn("rows = 50", private)
        self.assertIn("minimizer = LBFGS", private)
        self.assertIn("maxLoad = 0.2", private)
        self.assertIn("writeDumps = 0", private)
        self.assertIn("writeDebugVTUs = 0", private)
        self.assertIn("writeMeshVTUs = 0", private)

    def test_prefix_config_keeps_dump_and_stops_at_exact_load(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "config.conf"
            destination = directory / "prefix" / "config.conf"
            source.write_text("startLoad = 0.15\nmaxLoad = 1.0\nwriteDumps = 0\n")
            write_prefix_config(
                source,
                destination,
                run_name="prefix-test",
                stop_load=0.17215000000002215,
            )
            private = destination.read_text()

        self.assertIn("maxLoad = 0.17215000000002215", private)
        self.assertIn("writeDumps = 1", private)
        self.assertIn("name = prefix-test", private)
        self.assertIn("showProgress = -1", private)

    def test_prefix_dump_selection_allows_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            dump_dir = Path(directory)
            dumps = [
                dump_dir / "dump_l0.1523.mtsb",
                dump_dir / "dump_l0.1547.mtsb",
            ]
            for path in dumps:
                path.write_bytes(b"dump")

            selected, inventory = select_prefix_dump(
                dumps,
                stop_load=0.15466,
                load_increment=0.00003,
            )

        self.assertEqual(selected.name, "dump_l0.1547.mtsb")
        self.assertEqual([load for _, load in inventory], [0.1523, 0.1547])


if __name__ == "__main__":
    unittest.main()
