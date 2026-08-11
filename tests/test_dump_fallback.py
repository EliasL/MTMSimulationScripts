from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Management.configGenerator import SimulationConfig
from Management.queueLocalJobs import get_batch_script
from Management.simulationManager import SimulationManager


def make_manager(tmp_path):
    manager = object.__new__(SimulationManager)
    manager.outputPath = str(tmp_path)
    manager.subfolderName = "simulation"
    manager.program_path = "MTS2D"
    manager.configObj = SimpleNamespace(makeDumpAt=-1)
    manager.conf_file = "config.conf"
    manager.taskName = None
    dump_dir = tmp_path / manager.subfolderName / "dumps"
    dump_dir.mkdir(parents=True)
    return manager, dump_dir


class DumpFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manager, self.dump_dir = make_manager(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_latest_dump_is_deleted_and_previous_dump_is_used(self):
        previous = self.dump_dir / "dump_l0.1.xml"
        previous.write_text("valid")
        latest = self.dump_dir / "dump_l0.2.xml"
        latest.touch()

        with patch("Management.simulationManager.run_command", return_value=0) as run:
            self.manager.resumeSimulation(build=False)

        self.assertFalse(latest.exists())
        self.assertIn(str(previous), run.call_args.args[0])

    def test_unreadable_latest_is_quarantined_and_previous_is_used(self):
        previous = self.dump_dir / "dump_l0.1.xml"
        previous.write_text("valid")
        latest = self.dump_dir / "dump_l0.2.xml"
        latest.write_text("invalid")

        with patch("Management.simulationManager.run_command", side_effect=[2, 0]):
            self.manager.resumeSimulation(build=False)

        self.assertFalse(latest.exists())
        self.assertEqual(
            (self.dump_dir / f"broken_{latest.name}").read_text(), "invalid"
        )

    def test_second_unreadable_dump_crashes_after_quarantining_both(self):
        dumps = [
            self.dump_dir / "dump_l0.1.xml",
            self.dump_dir / "dump_l0.2.xml",
        ]
        for dump in dumps:
            dump.write_text("invalid")

        with patch("Management.simulationManager.run_command", side_effect=[2, 2]):
            with self.assertRaisesRegex(RuntimeError, "Could not load dump"):
                self.manager.resumeSimulation(build=False)

        self.assertTrue(
            all((self.dump_dir / f"broken_{dump.name}").exists() for dump in dumps)
        )

    def test_unrelated_failure_does_not_quarantine_dump(self):
        dump = self.dump_dir / "dump_l0.2.xml"
        dump.write_text("valid")

        with patch("Management.simulationManager.run_command", return_value=7):
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                self.manager.resumeSimulation(build=False)

        self.assertTrue(dump.exists())
        self.assertFalse((self.dump_dir / f"broken_{dump.name}").exists())

    def test_slurm_time_limit_is_one_day_and_five_minutes(self):
        script = get_batch_script("true", "test", 1, "/tmp")
        self.assertIn("#SBATCH --time=1-00:05:00", script)

    def test_slurm_nice_is_only_added_when_requested(self):
        normal = get_batch_script("true", "test", 1, "/tmp")
        low_priority = get_batch_script("true", "test", 1, "/tmp", nice=10000)
        self.assertNotIn("#SBATCH --nice=", normal)
        self.assertIn("#SBATCH --nice=10000", low_priority)

    def test_edge_locking_is_disabled_by_default(self):
        self.assertEqual(SimulationConfig().reconnectEdgeLocking, 0)


if __name__ == "__main__":
    unittest.main()
