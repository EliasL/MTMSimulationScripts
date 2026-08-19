import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from MTMath.poincareEnergy import C2PoincareDisk
from Plotting.fedBoundarySimpleShearIllustration import (
    BOUNDARIES,
    boundary_metrics,
    make_figure,
    reduction_example,
    simple_shear_metrics,
    unit_shear_matrix,
)


class FEDBoundarySimpleShearIllustrationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_boundary_faces_are_determinant_one_and_satisfy_their_equations(self):
        for spec in BOUNDARIES:
            with self.subTest(boundary=spec.key):
                C = boundary_metrics(spec, resolution=31)
                determinant = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] ** 2
                np.testing.assert_allclose(determinant, 1.0, atol=1e-12)
                np.testing.assert_allclose(
                    2.0 * np.abs(C[:, 0, 1]),
                    C[:, 0, 0] if spec.diagonal == "11" else C[:, 1, 1],
                    atol=1e-12,
                )
                smaller = C[:, 0, 0] if spec.diagonal == "11" else C[:, 1, 1]
                larger = C[:, 1, 1] if spec.diagonal == "11" else C[:, 0, 0]
                self.assertTrue(np.all(smaller <= larger + 1e-12))

    def test_simple_shear_reaches_matching_boundary_at_half_unit(self):
        for spec in BOUNDARIES:
            with self.subTest(boundary=spec.key):
                _, C = simple_shear_metrics(spec, 0.5)
                np.testing.assert_allclose(
                    2.0 * abs(C[0, 1]),
                    C[0, 0] if spec.diagonal == "11" else C[1, 1],
                )
                smaller = C[0, 0] if spec.diagonal == "11" else C[1, 1]
                larger = C[1, 1] if spec.diagonal == "11" else C[0, 0]
                self.assertLessEqual(smaller, larger + 1e-12)
                x, y = C2PoincareDisk(C)
                self.assertTrue(np.isfinite(x))
                self.assertTrue(np.isfinite(y))

    def test_reduction_uses_the_opposite_unit_shear_and_returns_inside(self):
        for spec in BOUNDARIES:
            with self.subTest(boundary=spec.key):
                C_out, C_reduced, M = reduction_example(spec)
                expected_argument = -spec.sign
                np.testing.assert_allclose(M, unit_shear_matrix(spec, expected_argument))

                outside_limit = min(C_out[0, 0], C_out[1, 1])
                reduced_limit = min(C_reduced[0, 0], C_reduced[1, 1])
                self.assertGreater(2.0 * abs(C_out[0, 1]), outside_limit)
                self.assertLessEqual(2.0 * abs(C_reduced[0, 1]), reduced_limit + 1e-12)

    def test_figure_writes_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            png_path, pdf_path = make_figure(
                output_stem=Path(tmp) / "fed_boundary_simple_shear",
                boundary_resolution=80,
                dpi=100,
            )
            self.assertTrue(png_path.is_file())
            self.assertTrue(pdf_path.is_file())


if __name__ == "__main__":
    unittest.main()
