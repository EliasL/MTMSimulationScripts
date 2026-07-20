import unittest
from unittest import mock

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import MTMath.poincareTiling as poincare_tiling
from MTMath.energyFunction import F_from_C, SShear


class TryAllRotationsTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plastic_reduction_bfs_returns_two_distinct_center_candidates(self):
        F = SShear(1.6, s_conponent=(0, 1))

        candidates = poincare_tiling.plasticReductionBFS(
            F.T @ F,
            max_depth=5,
            plot=False,
        )

        np.testing.assert_allclose(
            candidates,
            [
                [[1.0, -0.4], [-0.4, 1.16]],
                [[1.16, 0.4], [0.4, 1.0]],
            ],
            atol=1e-12,
        )

    def test_rotation_search_uses_bfs_lifts_without_lagrange_reduction(self):
        def return_existing_axis(ax=None, **_kwargs):
            return ax

        bfs_implementation = poincare_tiling.plasticReductionBFS
        with mock.patch.object(
            poincare_tiling,
            "lagrange_reduction",
            side_effect=AssertionError("Lagrange reduction must not be used"),
        ), mock.patch.object(
            poincare_tiling,
            "plasticReductionBFS",
            wraps=bfs_implementation,
        ) as bfs_mock, mock.patch.object(
            poincare_tiling,
            "drawPoincareGrid",
            side_effect=return_existing_axis,
        ), mock.patch.object(
            poincare_tiling.plt,
            "show",
        ):
            _, axes, matches, candidate_paths = poincare_tiling.tryAllRotations(
                n_theta=50_000,
                save=False,
                show=False,
            )

        bfs_mock.assert_called_once()
        self.assertEqual([match[3] for match in matches], [0, 1])
        np.testing.assert_allclose(
            [match[1] * 180 / np.pi for match in matches],
            [0.0, 0.0],
            atol=0.01,
        )
        np.testing.assert_allclose(
            matches[0][0],
            [[1.0, -0.4], [0.0, 1.0]],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            matches[1][0],
            [[0.4, 1.0], [-1.0, 0.0]],
            atol=1e-12,
        )
        self.assertEqual(len(candidate_paths), 7)
        np.testing.assert_allclose(
            [
                path["matches"][0][0] * 180 / np.pi
                for path in candidate_paths
            ],
            np.zeros(7),
            atol=1e-12,
        )
        plot_labels = [text.get_text() for text in axes[0].texts]
        self.assertIn(r"$\tilde{\mathbf{F}}_0$" + "\n" + r"$0.00^\circ$", plot_labels)
        self.assertIn(r"$\tilde{\mathbf{F}}_1$" + "\n" + r"$0.00^\circ$", plot_labels)
        self.assertFalse(any("m_1" in label or "m_2" in label for label in plot_labels))
        self.assertEqual(
            axes[1].get_xlabel(), r"Cauchy shear stress $\sigma_{12}$"
        )
        self.assertEqual(
            axes[1].get_ylabel(),
            r"Cauchy $N_1=(\sigma_{11} - \sigma_{22})/2$",
        )

    def test_plastic_reduction_bfs_uses_cauchy_signature_by_default(self):
        class CauchyOnlyEnergy:
            calls = 0

            @classmethod
            def cauchy_from_C(cls, C):
                cls.calls += 1
                return np.asarray(C, dtype=float)

            @classmethod
            def S_from_C(cls, _C):
                raise AssertionError("PK2 stress should not be used")

        F = SShear(1.6, s_conponent=(0, 1))
        candidates = poincare_tiling.plasticReductionBFS(
            F.T @ F,
            max_depth=5,
            eFunc=CauchyOnlyEnergy,
            plot=False,
        )

        self.assertEqual(candidates.shape, (2, 2, 2))
        self.assertGreater(CauchyOnlyEnergy.calls, 0)

    def test_every_bfs_path_lift_preserves_the_unrotated_stress_id(self):
        F = SShear(1.6, s_conponent=(0, 1))
        candidates, paths = poincare_tiling.plasticReductionBFS(
            F.T @ F,
            max_depth=5,
            plot=False,
            return_paths=True,
        )

        reference_id = poincare_tiling.getIdOfF(F)
        for path in paths:
            candidate_number = path["candidate_index"]
            path_F = F @ path["M"]
            np.testing.assert_allclose(
                path_F.T @ path_F,
                candidates[candidate_number],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                poincare_tiling.getIdOfF(path_F),
                reference_id,
                atol=1e-12,
            )

        # The canonical square root has the same C but not the same spatial
        # orientation.  Using it here reproduces the bug fixed in
        # tryAllRotations.
        for candidate in candidates:
            canonical_F = F_from_C(candidate)
            self.assertFalse(
                np.allclose(poincare_tiling.getIdOfF(canonical_F), reference_id)
            )


if __name__ == "__main__":
    unittest.main()
