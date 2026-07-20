import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from MTMath.poincareEnergy import drawC, generate_poincare_disk


class DrawCTests(unittest.TestCase):
    def setUp(self):
        self.C0 = np.eye(2)
        self.C1 = np.array([[1.25, 0.5], [0.5, 1.0]])

    def tearDown(self):
        plt.close("all")

    def test_single_tensor_accepts_scalar_and_sequence_labels(self):
        fig, axes = plt.subplots(1, 2)

        drawC(axes[0], self.C0, scatter=True, label="scalar", fontsize=10)
        drawC(axes[1], self.C0, scatter=True, label=["sequence"])

        self.assertEqual([text.get_text() for text in axes[0].texts], ["scalar"])
        self.assertEqual(axes[0].texts[0].get_fontsize(), 10)
        self.assertEqual([text.get_text() for text in axes[1].texts], ["sequence"])

    def test_batched_scatter_accepts_numpy_labels(self):
        _, ax = plt.subplots()
        labels = np.array(["first", "second"])

        drawC(ax, np.stack([self.C0, self.C1]), scatter=True, label=labels)

        self.assertEqual([text.get_text() for text in ax.texts], labels.tolist())
        self.assertEqual(ax.collections[0].get_offsets().shape, (2, 2))

    def test_arrow_accepts_list_input_and_labels_midpoint(self):
        _, ax = plt.subplots()

        drawC(
            ax,
            [self.C0, self.C1],
            arrow=True,
            label=["path"],
            linestyle="--",
            alpha=0.6,
        )

        labels = [text.get_text() for text in ax.texts if text.get_text()]
        self.assertEqual(labels, ["path"])

    def test_line_and_both_shading_modes_still_work(self):
        fig, axes = plt.subplots(1, 3)
        C_grid = generate_poincare_disk(resolution=20, zoom=2)

        drawC(axes[0], np.stack([self.C0, self.C1]))
        drawC(axes[1], C_grid, grid_size=20, zoom=2, shade=True)
        drawC(
            axes[2],
            C_grid,
            grid_size=20,
            zoom=2,
            shade=True,
            shade_values=1.0,
        )

        self.assertEqual(len(axes[0].lines), 1)
        self.assertEqual(len(axes[1].images), 1)
        self.assertEqual(len(axes[2].images), 1)
        self.assertEqual(tuple(axes[1].images[0].get_extent()), (0, 20, 0, 20))

    def test_label_count_must_match_point_count(self):
        _, ax = plt.subplots()

        with self.assertRaisesRegex(ValueError, "1 label.*2 plotted point"):
            drawC(
                ax,
                np.stack([self.C0, self.C1]),
                scatter=True,
                label="only one",
            )


if __name__ == "__main__":
    unittest.main()
