import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from Management.vtuBeforeReconnectionExtraction import (
    extract_simulation,
    find_completed_event_step,
    reconnection_pairs,
    write_extraction_config,
)


BASE_CONFIG = """\
# Preserve this comment and unknown future options.
name = simpleShear,s10x10l0.15,0.01,1.0PBCedgeFlipt1s3
rows = 10
cols = 10
scenario = simpleShear
logDuringMinimization = 0
writeDumps = 1
plasticityEventThreshold = 0.05
futureOption = keep-me
"""


class VtuBeforeReconnectionExtractionTests(unittest.TestCase):
    def test_config_patch_preserves_unknown_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.conf"
            destination = directory / "destination.conf"
            source.write_text(BASE_CONFIG)

            name = write_extraction_config(source, destination)
            result = destination.read_text()

            self.assertEqual(
                name,
                "simpleShear,s10x10l0.15,0.01,1.0PBCedgeFlipt1"
                "logDuringMinimization1s3",
            )
            self.assertIn("futureOption = keep-me", result)
            self.assertIn("maxLoad = 1e100", result)
            self.assertIn("logDuringMinimization = 1", result)
            self.assertIn("fullMinimizationLogging = 0", result)
            self.assertIn("writeDumps = 0", result)
            self.assertIn("plasticityEventThreshold = 0", result)

    def test_only_a_completed_step_with_a_pair_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            periodic = root / "step10"
            event = root / "step11"
            active = root / "step12"
            periodic.mkdir()
            event.mkdir()
            active.mkdir()
            (event / "mesh_pre.11.vtu").write_text("before")
            (event / "mesh_post.11.vtu").write_text("after")

            self.assertEqual(find_completed_event_step(root), event)

    def test_integration_stops_only_its_child_and_copies_the_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            simulation = root / "simulation"
            dumps = simulation / "dumps"
            dumps.mkdir(parents=True)
            (simulation / "config.conf").write_text(BASE_CONFIG)
            (dumps / "dump_l0.2.xml.gz").write_text("fake dump")

            fake_mts2d = root / "fake_mts2d.py"
            fake_mts2d.write_text(
                "#!/usr/bin/env python3\n"
                + textwrap.dedent(
                    """
                    import re
                    import sys
                    import time
                    from pathlib import Path

                    args = sys.argv[1:]
                    config = Path(args[args.index("-c") + 1])
                    output = Path(args[args.index("-o") + 1])
                    name = re.search(
                        r"^\\s*name\\s*=\\s*(.+?)\\s*$",
                        config.read_text(),
                        re.MULTILINE,
                    ).group(1)
                    root = output / name / "data/minimizationData"
                    (root / "step20").mkdir(parents=True)
                    event = root / "step21"
                    event.mkdir()
                    (event / "mesh_pre.21.vtu").write_text("before")
                    (event / "mesh_post.21.vtu").write_text("after")
                    (event / "collection.pvd").write_text("complete")
                    time.sleep(30)
                    """
                )
            )
            fake_mts2d.chmod(0o755)

            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            try:
                results = extract_simulation(
                    simulation, fake_mts2d, poll_interval=0.01, timeout=5
                )
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate()
                unrelated.wait()

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].name, "dump_l0.2.xml.gz_step21")
            self.assertEqual(len(reconnection_pairs(results[0])), 1)
            self.assertEqual((results[0] / "mesh_pre.21.vtu").read_text(), "before")
            self.assertFalse((simulation / "beforeReconnectionVtuData/.work").exists())


if __name__ == "__main__":
    unittest.main()
