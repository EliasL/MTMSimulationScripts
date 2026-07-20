import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from Plotting.meshLockingFigure import classify_protocol_nodes


class MeshLockingFigureTests(unittest.TestCase):
    def test_protocol_node_roles_cover_the_constrained_l_boundary(self):
        x, y = np.meshgrid(np.arange(4, dtype=float), np.arange(4, dtype=float))
        nodes = np.column_stack([x.ravel(), y.ravel()])
        fixed_status = (nodes[:, 0] == 0) | (nodes[:, 1] == 0)

        groups = classify_protocol_nodes(nodes, fixed_status)

        self.assertEqual(set(groups["fixed"]), {0, 1, 4})
        self.assertEqual(set(groups["x_loaded"]), {8, 12})
        self.assertEqual(set(groups["y_loaded"]), {2, 3})
        constrained = np.concatenate(
            [groups["fixed"], groups["x_loaded"], groups["y_loaded"]]
        )
        np.testing.assert_array_equal(
            np.sort(constrained), np.flatnonzero(fixed_status)
        )


if __name__ == "__main__":
    unittest.main()
