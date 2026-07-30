import tempfile
import unittest
from pathlib import Path

from Plotting.makeAnimations import prepare_animation_inputs, simulation_uses_reconnection


class AnimationInputTests(unittest.TestCase):
    def _simulation_folder(self, root):
        (root / "macroData.csv").touch()
        (root / "collection.pvd").write_text(
            '<VTKFile><Collection><DataSet file="sample.0.vtu"/></Collection></VTKFile>'
        )

    def test_empty_macro_argument_still_uses_standard_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._simulation_folder(root)
            _, macro, _, _ = prepare_animation_inputs(root, macroData="")
            self.assertEqual(Path(macro), (root / "macroData.csv").resolve())

    def test_metadata_free_rendering_requires_explicit_opt_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._simulation_folder(root)
            _, macro, _, _ = prepare_animation_inputs(root, useMetadata=False)
            self.assertIsNone(macro)

    def test_reconnection_mode_is_read_from_simulation_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.conf").write_text("reconnectionMethod = edgeFlip\n")
            self.assertTrue(simulation_uses_reconnection(root))
            (root / "config.conf").write_text("reconnectionMethod = none\n")
            self.assertFalse(simulation_uses_reconnection(root))


if __name__ == "__main__":
    unittest.main()
