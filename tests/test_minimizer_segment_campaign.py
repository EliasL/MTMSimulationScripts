import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from ClusterJobs.minimizer_segment_campaign import (
    FIRE_EPS,
    MEMORY,
    source_boundaries,
    segment_config,
)
from ClusterJobs.run_minimizer_segment import last_macro_load, truncate_to_restart
from Management.queueLocalJobs import get_batch_script


class MinimizerSegmentCampaignTests(unittest.TestCase):
    def test_batch_script_requests_campaign_memory(self):
        script = get_batch_script("true", "test", 2, "/tmp", memory=MEMORY)
        self.assertIn("#SBATCH --mem=4G", script)

    def test_source_boundaries_use_dump_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(10, 51):
                nominal = index / 10
                with gzip.open(root / f"dump_l{nominal:.1f}.xml.gz", "wt") as stream:
                    stream.write(f"<load>{nominal + 0.0123:.17g}</load>\n")
            boundaries = source_boundaries(root)
        self.assertEqual(len(boundaries), 41)
        self.assertAlmostEqual(boundaries[0]["actual_load"], 1.0123)
        self.assertAlmostEqual(boundaries[-1]["actual_load"], 5.0123)

    def test_minimizers_receive_matching_epsx(self):
        self.assertEqual(MEMORY, "4G")
        lbfgs = segment_config(1.0, 1.1, "LBFGS", "lbfgs")
        cg = segment_config(1.0, 1.1, "CG", "cg")
        fire = segment_config(1.0, 1.1, "FIRE", "fire")
        self.assertEqual(lbfgs.LBFGSEpsx, 1e-6)
        self.assertEqual(cg.CGEpsx, 1e-6)
        self.assertEqual(fire.FIREEpsx, 0.0)
        self.assertEqual(fire.eps, FIRE_EPS)
        self.assertTrue(all(c.nrThreads == 2 for c in (lbfgs, cg, fire)))
        self.assertTrue(all(c.writeMeshVTUs == 1 for c in (lbfgs, cg, fire)))
        self.assertTrue(all(c.writeDumps == 1 for c in (lbfgs, cg, fire)))

    def test_resume_truncates_only_rows_and_vtus_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            macro = output / "macroData.csv"
            with macro.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["load_step", "load"])
                writer.writeheader()
                for step in (10, 11, 12):
                    writer.writerow({"load_step": step, "load": step / 10})
            data = output / "data"
            data.mkdir()
            for step in (10, 11, 12):
                (data / f"frame.{step}.vtu").touch()
            (output / "collection.pvd").write_text(
                '<DataSet timestep="10"/>\n'
                '<DataSet timestep="11"/>\n'
                '<DataSet timestep="12"/>\n'
            )

            truncate_to_restart(output, 11)

            self.assertEqual(last_macro_load(macro), 1.1)
            self.assertTrue((data / "frame.10.vtu").exists())
            self.assertTrue((data / "frame.11.vtu").exists())
            self.assertFalse((data / "frame.12.vtu").exists())
            self.assertNotIn('timestep="12"', (output / "collection.pvd").read_text())


if __name__ == "__main__":
    unittest.main()
