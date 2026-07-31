import unittest
from unittest.mock import patch

import numpy as np

from Plotting.dataFunctions import VTUData
from Plotting.pyplotFunctions import (
    _make_integer_shear_bins,
    canonical_plastic_shear_counts,
    get_plastic_shear_counts,
    process_frame,
)


def horizontal(n):
    return np.array([[1.0, n], [0.0, 1.0]])


def vertical(n):
    return np.array([[1.0, 0.0], [n, 1.0]])


class PlasticShearCountTests(unittest.TestCase):
    def test_signed_horizontal_and_vertical_counts(self):
        matrices = np.stack([horizontal(3), horizontal(-2), vertical(4), vertical(-1)])
        h, v = canonical_plastic_shear_counts(matrices)
        np.testing.assert_array_equal(h, [3, -2, 0, 0])
        np.testing.assert_array_equal(v, [0, 0, 4, -1])

    def test_opposite_shears_cancel(self):
        h, v = canonical_plastic_shear_counts(horizontal(3) @ horizontal(-2))
        self.assertEqual(int(h), 1)
        self.assertEqual(int(v), 0)

    def test_mixed_decomposition_counts_each_direction(self):
        h, v = canonical_plastic_shear_counts(horizontal(2) @ vertical(-3))
        self.assertEqual(int(h), 2)
        self.assertEqual(int(v), -3)

    def test_frame_retry_returns_the_successful_path(self):
        calls = []

        def flaky_frame(path):
            calls.append(path)
            if len(calls) == 1:
                raise SyntaxError("transient PNG read failure")
            return path

        result = process_frame({"frameFunction": flaky_frame, "path": "frame.png"})
        self.assertEqual(result, "frame.png")
        self.assertEqual(len(calls), 2)

    def test_reconnection_mode_selects_T_or_F_P(self):
        fields = {
            "T11": np.array([1.0]),
            "T12": np.array([3.0]),
            "T21": np.array([0.0]),
            "T22": np.array([1.0]),
            "F_P11": np.array([1.0]),
            "F_P12": np.array([2.0]),
            "F_P21": np.array([0.0]),
            "F_P22": np.array([1.0]),
        }

        class FakeVTU:
            def get_cell_data(self, name):
                return fields[name]

        with patch("Plotting.pyplotFunctions.VTUData", return_value=FakeVTU()):
            h_recon, _, _, source_recon = get_plastic_shear_counts(
                "frame.vtu", reconnecting=True
            )
            h_fixed, _, _, source_fixed = get_plastic_shear_counts(
                "frame.vtu", reconnecting=False
            )

        self.assertEqual(int(h_recon[0]), 3)
        self.assertEqual(source_recon, "T")
        self.assertEqual(int(h_fixed[0]), 2)
        self.assertEqual(source_fixed, "F_P")

    def test_integer_shear_bins_are_symmetric_and_grouped(self):
        _, _, labels = _make_integer_shear_bins(50)
        self.assertEqual(
            labels,
            [
                "-41+",
                "-40..-21",
                "-20..-11",
                "-10..-5",
                "-4..-2",
                "-1",
                "0",
                "1",
                "2..4",
                "5..10",
                "11..20",
                "21..40",
                "41+",
            ],
        )

    def test_total_branch_uses_exported_values_when_available(self):
        data = VTUData.__new__(VTUData)
        components = {
            (1, 1): np.array([2.0]),
            (1, 2): np.array([3.0]),
            (2, 1): np.array([4.0]),
            (2, 2): np.array([5.0]),
        }
        with patch.object(data, "get_matrix_components", return_value=components):
            np.testing.assert_allclose(
                data.get_T_total(), [[[2.0, 3.0], [4.0, 5.0]]]
            )


if __name__ == "__main__":
    unittest.main()
