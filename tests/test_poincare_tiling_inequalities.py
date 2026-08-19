import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from MTMath.poincareEnergy import (
    _fundamental_domain_inequality_mask,
    _generate_shear_transformation_path_counts,
    generate_poincare_disk,
    generatePoincareCTilingRegions,
    generatePoincareCTilingMultiplicity,
    generateShearTransformations,
    getCFundamental,
    plotPoincareCTilingInequalities,
    transformed_fundamental_domain_inequalities,
)


class PoincareTilingInequalityTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_raw_shear_labels_are_consistent_for_both_application_orders(self):
        for left_applied in (False, True):
            _, labels = generateShearTransformations(2, leftApplied=left_applied)
            self.assertEqual(labels[0], "")
            self.assertTrue(all("$" not in label for label in labels))

    def test_transformed_inequalities_match_legacy_mask_away_from_boundaries(self):
        transformations, _ = generateShearTransformations(2, leftApplied=False)
        transform = transformations[4]
        C, _ = generate_poincare_disk(resolution=96, returnMask=True)
        coefficients = transformed_fundamental_domain_inequalities(transform)
        fast_mask = _fundamental_domain_inequality_mask(C, coefficients)
        _, legacy_mask = getCFundamental(
            96,
            transformation=transform,
            returnMask=True,
        )

        values = np.stack(
            [
                coefficients[0, 0] * C[..., 0, 0]
                + coefficients[0, 1] * C[..., 0, 1]
                + coefficients[0, 2] * C[..., 1, 1],
                coefficients[1, 0] * C[..., 0, 0]
                + coefficients[1, 1] * C[..., 0, 1]
                + coefficients[1, 2] * C[..., 1, 1],
                coefficients[2, 0] * C[..., 0, 0]
                + coefficients[2, 1] * C[..., 0, 1]
                + coefficients[2, 2] * C[..., 1, 1],
            ]
        )
        away_from_boundary = np.all(np.abs(values) > 1e-8, axis=0)
        np.testing.assert_array_equal(
            fast_mask[away_from_boundary], legacy_mask[away_from_boundary]
        )

    def test_regions_dedupe_signed_congruence_transforms(self):
        transforms, _ = generateShearTransformations(4, leftApplied=False)
        tile_ids, specifications = generatePoincareCTilingRegions(
            depth=4,
            quadrants="a",
            leftApplied=False,
            grid_size=96,
        )
        self.assertLess(len(specifications), len(transforms))
        self.assertTrue(np.any(tile_ids >= 0))
        self.assertTrue(np.all(tile_ids < len(specifications)))

    def test_path_generation_excludes_immediate_inverse_moves(self):
        entries = _generate_shear_transformation_path_counts(
            depth=3,
            leftApplied=False,
            collect_labels=True,
        )
        labels = [path for entry in entries for path in entry["path_labels"]]
        inverse = {"r": "l", "l": "r", "u": "d", "d": "u"}
        self.assertEqual(len(labels), 53)
        self.assertTrue(
            all(
                all(inverse.get(first) != second for first, second in zip(path, path[1:]))
                for path in labels
            )
        )

    def test_multiplicity_counts_non_backtracking_paths_on_each_domain(self):
        counts, _, specifications, centroids = generatePoincareCTilingMultiplicity(
            depth=3,
            quadrants="a",
            leftApplied=False,
            grid_size=96,
            collect_labels=True,
        )
        self.assertEqual(sum(spec["path_count"] for spec in specifications), 53)
        self.assertEqual(
            sum(len(spec["path_labels"]) for spec in specifications), 53
        )
        self.assertLess(len(specifications), 53)
        self.assertGreater(int(counts.max()), 1)
        self.assertTrue(any(centroid is not None for centroid in centroids))

    def test_fast_plot_uses_one_raster_layer_for_one_quadrant(self):
        _, ax = plt.subplots()
        plotPoincareCTilingInequalities(
            ax=ax,
            save=False,
            grid_size=64,
            depth=1,
            quadrants="a",
            use_labels=True,
            leftApplied=False,
        )
        self.assertEqual(len(ax.images), 1)
        self.assertTrue(all("$$" not in text.get_text() for text in ax.texts))


if __name__ == "__main__":
    unittest.main()
