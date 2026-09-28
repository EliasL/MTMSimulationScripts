import tempfile
import unittest
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from Plotting.makePlots import getPrettyLabel
from Plotting.minimizationEnergyPlot import plot_minimization_energy


class MinimizationEnergyPlotTests(unittest.TestCase):
    def test_pretty_function_evaluations_label(self):
        self.assertEqual(getPrettyLabel("nr_func_evals"), "Nr. func. evaluations")

    def test_energy_is_normalized_and_zoom_geometry_is_added(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "macroData.csv"
            pd.DataFrame(
                {
                    "nr_func_evals": [0, 1, 2, 3],
                    "total_energy": [40.0, 32.0, 28.0, 24.0],
                    "nr_reconnect_cycles": [0, 0, 1, 1],
                }
            ).to_csv(csv_path, index=False)

            fig, ax, inset = plot_minimization_energy(
                csv_path,
                volume=4,
                zoom_fraction=(0.1, 0.1),
                zoom_xlim=(None, 3),
                zoom_ylim=(6.1, 6.2),
            )
            self.assertEqual(ax.get_xlabel(), "Nr. func. evaluations")
            self.assertEqual(ax.lines[0].get_label(), r"$\langle E\rangle/V_0$")
            self.assertEqual(list(ax.lines[0].get_ydata()), [10.0, 8.0, 7.0])
            self.assertEqual(ax.lines[1].get_label(), "Final energy")
            self.assertEqual(ax.lines[1].get_markerfacecolor(), "none")
            self.assertEqual(len(ax.patches), 1)
            self.assertEqual(len(fig.artists), 2)
            self.assertEqual(len(inset.lines), 2)
            self.assertEqual(ax.get_yscale(), "linear")
            self.assertEqual(ax.get_legend()._loc, 2)
            self.assertEqual(len(ax.collections), 1)
            self.assertEqual(len(inset.collections), 1)
            zoom_box = ax.patches[0]
            self.assertAlmostEqual(zoom_box.get_height(), 0.1)
            self.assertAlmostEqual(zoom_box.get_x() + zoom_box.get_width(), 3)
            self.assertAlmostEqual(
                zoom_box.get_y() + 0.5 * zoom_box.get_height(),
                6.15,
            )
            matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    unittest.main()
