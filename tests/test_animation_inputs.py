import tempfile
import unittest
from pathlib import Path

from Plotting.makeAnimations import (
    get_vtu_strains,
    prepare_animation_inputs,
    select_vtu_files,
    simulation_uses_reconnection,
)


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

    def test_default_frame_selection_targets_uniform_strain_intervals(self):
        files = [
            f"sample_load={load}_.{index}.vtu"
            for index, load in enumerate((0.0, 0.1, 0.4, 1.0))
        ]
        selected = select_vtu_files(files, 3)
        self.assertEqual(get_vtu_strains(selected), [0.0, 0.4, 1.0])

    def test_index_sampling_remains_available_as_an_opt_out(self):
        files = [f"sample_{index}.vtu" for index in range(6)]
        self.assertEqual(
            select_vtu_files(files, 3, constant_strain_rate=False),
            [files[0], files[3], files[5]],
        )

    def test_strain_rate_sampling_preserves_a_reversal(self):
        strains = (0.0, 0.5, 1.0, 0.5, 0.0)
        files = [
            f"sample_load={strain}_.{index}.vtu"
            for index, strain in enumerate(strains)
        ]
        selected = select_vtu_files(files, 5)
        self.assertEqual(get_vtu_strains(selected), list(strains))


if __name__ == "__main__":
    unittest.main()
