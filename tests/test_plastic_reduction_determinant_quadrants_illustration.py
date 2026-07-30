import unittest
from collections import Counter

import numpy as np

from Plotting.plasticReductionAllDecompositionsIllustration import (
    decomposition_table_data,
)
from Plotting.plasticReductionDeterminantQuadrantsIllustration import (
    PANEL_COLOR_STRENGTH,
    _elastic_transform_label,
    grouped_disk_points,
    panel_background_color,
    quadrant_palette,
    reduction_history_data,
)


class PlasticReductionDeterminantQuadrantsIllustrationTests(unittest.TestCase):
    def test_determinant_sectors_use_distinct_four_color_palettes(self):
        positive = quadrant_palette(1)
        negative = quadrant_palette(-1)

        self.assertEqual(positive.shape, (4, 4))
        self.assertEqual(negative.shape, (4, 4))
        self.assertFalse(np.allclose(positive, negative))

    def test_each_determinant_disk_has_two_unique_metric_points(self):
        data = decomposition_table_data()

        self.assertEqual(
            Counter(item["determinant"] for item in data),
            Counter({1: 4, -1: 4}),
        )
        for determinant in (1, -1):
            groups = grouped_disk_points(data, determinant)
            self.assertEqual(len(groups), 2)
            self.assertEqual(sorted(map(len, groups.values())), [2, 2])

    def test_panel_backgrounds_follow_determinant_and_quadrant(self):
        data = decomposition_table_data()
        for item in data:
            palette_color = quadrant_palette(item["determinant"])[
                item["quadrant"], :3
            ]
            expected = 1.0 - PANEL_COLOR_STRENGTH * (1.0 - palette_color)
            background = np.asarray(panel_background_color(item))
            np.testing.assert_allclose(background, expected)

    def test_reduction_history_switches_sheets_on_reflections(self):
        histories = reduction_history_data()

        self.assertEqual(
            histories["lagrange"]["determinants"],
            (1, -1, -1, 1),
        )
        self.assertEqual(
            histories["elastic"]["determinants"],
            (1, 1),
        )
        self.assertEqual(len(histories["lagrange"]["transforms"]), 3)
        self.assertEqual(
            _elastic_transform_label(histories["elastic"]["history"], 0),
            r"$\mathbf{E}_{12}(1)$",
        )


if __name__ == "__main__":
    unittest.main()
