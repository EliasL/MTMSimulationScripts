import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from Plotting.plot_plastic_quadrant_decomposition import (
    decomposition_data,
    match2_decomposition_data,
    same_length_decomposition_data,
)


class PlasticQuadrantDecompositionPathTests(unittest.TestCase):
    @staticmethod
    def _terminal_metric(representative):
        F_e = representative["F_e"]
        return F_e.T @ F_e

    def assert_shortest_distinct_lift(self, short, alternative, paths):
        target_C = self._terminal_metric(short)
        self.assertEqual(
            len(short["path"]),
            min(path["depth"] for path in paths),
        )
        np.testing.assert_allclose(
            self._terminal_metric(alternative),
            target_C,
            atol=1e-12,
        )
        np.testing.assert_array_equal(alternative["M"], -short["M"])

        distinct_lift_depths = [
            path["depth"]
            for path in paths
            if np.allclose(path["C"], target_C, atol=1e-12, rtol=1e-12)
            and not np.array_equal(path["M"], short["M"])
        ]
        self.assertTrue(distinct_lift_depths)
        self.assertEqual(len(alternative["path"]), min(distinct_lift_depths))

    def test_original_uses_pr_then_shortest_rotated_lift(self):
        total_F, representatives, paths = decomposition_data()
        short, long = representatives

        self.assertEqual(short["path"], ("U-",))
        self.assertEqual(
            long["path"],
            ("U+", "L-", "U+", "U+", "L-"),
        )
        self.assert_shortest_distinct_lift(short, long, paths)

        intermediate_F, second_representatives, second_paths = (
            same_length_decomposition_data(total_F, long)
        )
        self.assertEqual(intermediate_F.shape, (2, 2))
        pr, other = second_representatives
        self.assertEqual(pr["path"], ("U+", "L+"))
        self.assertEqual(
            other["path"],
            ("U+", "U+", "U+", "L-", "U+", "U+"),
        )
        self.assert_shortest_distinct_lift(pr, other, second_paths)

        same_lift_depth_four = [
            path
            for path in second_paths
            if path["path"] == ("L+", "U-", "L-", "U+")
        ]
        self.assertEqual(len(same_lift_depth_four), 1)
        np.testing.assert_array_equal(same_lift_depth_four[0]["M"], pr["M"])

    def test_match2_uses_shortest_quarter_and_half_turn_lifts(self):
        _total_F, representatives, paths = match2_decomposition_data()
        short, match_90, match_180 = representatives

        self.assertEqual(short["path"], ("U-",))
        self.assertEqual(match_90["path"], ("L-", "U+"))
        self.assertEqual(
            match_180["path"],
            ("U+", "L-", "U+", "U+", "L-"),
        )
        quarter_turn = np.array([[0, 1], [-1, 0]])
        np.testing.assert_array_equal(match_90["M"], short["M"] @ quarter_turn)
        self.assert_shortest_distinct_lift(short, match_180, paths)

    def test_match2_can_use_the_alternate_initial_state(self):
        total_F, representatives, _ = decomposition_data()
        _short, long = representatives
        alternate_F, _, _ = same_length_decomposition_data(total_F, long)

        _alternate_F, representatives, _ = match2_decomposition_data(alternate_F)
        short, match_90, match_180 = representatives

        self.assertEqual(short["path"], ("U+", "L+"))
        self.assertEqual(match_90["path"], ("U+", "U+", "L-"))
        self.assertEqual(
            match_180["path"],
            ("U+", "U+", "U+", "L-", "U+", "U+"),
        )


if __name__ == "__main__":
    unittest.main()
