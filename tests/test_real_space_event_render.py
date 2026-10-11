import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Plotting import meshEventPlotting as mesh_plot
from Plotting.real_space_events import render
from Plotting.real_space_events.models import EventSheetLayout, RenderOptions


class EventRenderTests(unittest.TestCase):
    def test_geometry_recovers_energy_and_work_conjugate_stress(self):
        from MTMath.energyFunction import ContiEnergy
        from Plotting.real_space_events.legacy_fields import recover_legacy_fields, current_area_cauchy_mean
        shear = np.array([[1., .137], [0., 1.]])
        points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]) @ shear.T
        triangles = np.array([[0, 1, 2], [1, 2, 3]])  # opposite orientations
        edges = (points[triangles][:, 1:] - points[triangles][:, :1]).swapaxes(-1, -2)
        positive = edges.copy()
        positive[np.linalg.det(positive) < 0, :, 1] *= -1
        energy = .5*ContiEnergy.energy_from_F(positive)
        rounded = lambda a: np.array([float(f"{x:.5g}") for x in a])
        fields = {f"F_Fix{i+1}{j+1}": rounded(edges[:, i, j])
                  for i in range(2) for j in range(2)}
        fields.update(energy_field=rounded(energy), sigma12=np.full(2, 100.))
        state = mesh_plot.MeshState(path=Path("legacy"), points=points,
                                    triangles=triangles, reference_indices=np.arange(4),
                                    point_fields={}, cell_fields=fields)
        recovered = recover_legacy_fields(state, 1.)
        np.testing.assert_allclose(recovered.cell_fields["energy_field"], energy, atol=1e-14)
        h = 1e-6
        plus, minus = np.array([[1., h], [0., 1.]]), np.array([[1., -h], [0., 1.]])
        derivative = (.5*ContiEnergy.energy_from_F(plus @ positive).sum()
                      -.5*ContiEnergy.energy_from_F(minus @ positive).sum())/(2*h)
        self.assertAlmostEqual(current_area_cauchy_mean(recovered), derivative, places=7)
        np.testing.assert_array_equal(state.cell_fields["sigma12"], [100., 100.])

    def test_color_scale_ignores_largest_thirty_percent_by_magnitude(self):
        values = np.array([-1000., -3., -2., -1., 0., 1., 2., 3., 500., 900.])
        self.assertAlmostEqual(render._energy_color_limit(values, 0.7), 152.1)
        self.assertEqual(render._energy_color_limit([0., 0.], 0.7), np.finfo(float).eps)
        self.assertAlmostEqual(render._energy_color_limit(values, 0.9), 910.)
        self.assertEqual(RenderOptions(output_root=Path(".")).energy_color_quantile, 0.97)
        self.assertAlmostEqual(render._energy_color_limit(values, 0.95), 955.)
        self.assertAlmostEqual(render._energy_color_limit(values, 0.97), 973.)

    def test_legacy_stress_uses_force_reference_and_current_area(self):
        from Plotting.real_space_events.rebuild_downloaded_atlas import legacy_starting_cauchy_mean
        from types import SimpleNamespace
        state = SimpleNamespace(
            points=np.array([[0., 0.], [1., 0.], [0., .5], [1., 1.5]]),
            triangles=np.array([[0, 1, 2], [1, 3, 2]]),
            cell_fields={"P11": np.array([2., 4.]), "P12": np.array([1., 2.]),
                         "F_Fix21": np.array([.5, 1.]), "F_Fix22": np.ones(2),
                         "sigma12": np.array([100., 100.])})
        self.assertAlmostEqual(legacy_starting_cauchy_mean(state, 1.), 4.)

    def test_pre_yield_rows_do_not_enter_scatter_or_legend(self):
        event = pd.Series(dict(delta_rev_u=0.01, delta_E_S_over_V0=0.1,
                               rev_u_cut=0.02, yield_regime="post"))
        pre_event = event.copy()
        pre_event.update(pd.Series(dict(delta_rev_u=100., delta_E_S_over_V0=100.,
                                        yield_regime="pre")))
        catalog = pd.DataFrame([event, pre_event])
        fig, ax = plt.subplots()
        try:
            render._plot_reversibility_scatter(ax, catalog, event, EventSheetLayout())
            self.assertLess(ax.get_xlim()[1], 1.)
            self.assertFalse(any("pre-yield" in text.get_text()
                                 for text in ax.get_legend().get_texts()))
            np.testing.assert_allclose(ax.collections[-1].get_offsets(), [[0.01, 0.1]])
            np.testing.assert_allclose(ax.lines[1].get_xdata(), [0.01, 0.01])
            np.testing.assert_allclose(ax.lines[2].get_ydata(), [0.1, 0.1])
        finally:
            plt.close(fig)

    def test_renderer_compares_equal_load_pairs_and_adds_four_connectors(self):
        names = ("state0_min_gamma", "state1_affine_gamma_plus",
                 "state2_relaxed_gamma_plus", "state3_affine_gamma_minus",
                 "state4_relaxed_gamma")
        points = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.]])
        triangles = np.array([[0, 1, 2], [1, 3, 2]])
        states = {}
        for i, name in enumerate(names):
            deformed = points.copy()
            deformed[3, 0] += i * 0.001
            states[name] = mesh_plot.MeshState(
                path=Path(name), points=deformed, triangles=triangles,
                reference_indices=np.arange(4), point_fields={},
                cell_fields={"energy_field": np.array([10., 20.]) - [0, 1, 2, 3, 7][i],
                             "nrm3": np.zeros(2), "deltaNrm3": np.zeros(2)})
        event = pd.Series(dict(delta_rev_u=0.01, delta_E_S_over_V0=0.1,
                               rev_u_cut=0.02, yield_regime="post"))
        captured = []
        original_save = plt.Figure.savefig

        def save(fig, *args, **kwargs):
            captured.append(fig)
            return original_save(fig, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(render, "_load_states", return_value=states), \
             patch.object(mesh_plot, "calculate_energy_change_field",
                          wraps=mesh_plot.calculate_energy_change_field) as changes, \
             patch.object(plt.Figure, "savefig", save):
            render.render_event_pdf(event, None, Path(tmp)/"event.png",
                                    RenderOptions(output_root=Path(tmp), rasterized_dpi=40))
        self.assertIs(changes.call_args_list[0].args[0], states[names[1]])
        self.assertIs(changes.call_args_list[0].args[1], states[names[2]])
        self.assertIs(changes.call_args_list[1].args[0], states[names[3]])
        self.assertIs(changes.call_args_list[1].args[1], states[names[4]])
        fig = captured[0]
        self.assertIsNone(fig._suptitle)
        norms = [(c.norm.vmin, c.norm.vmax) for ax in fig.axes[:6]
                 for c in ax.collections if c.get_array() is not None]
        self.assertEqual(norms, [(-4., 4.)]*4)
        self.assertEqual(len(fig.artists), 4)
        for connector in fig.artists:
            self.assertEqual(connector.xy2[1], 1)
            self.assertEqual(connector.get_edgecolor()[:3], (0., 0., 0.))
        titles = [axis.get_title() for axis in fig.axes]
        self.assertIn(r"(b) $E_1^{(e)}-E_2^{(e)}$", titles)
        self.assertIn(r"(d) $E_3^{(e)}-E_4^{(e)}$", titles)
        from matplotlib.quiver import Quiver
        quivers = [collection for axis in fig.axes for collection in axis.collections
                   if isinstance(collection, Quiver)]
        self.assertEqual(len(quivers), 3)
        for quiver in quivers:
            self.assertIsInstance(quiver, mesh_plot.LengthScaledQuiver)
            x, y = quiver._h_arrows(np.array([0., 10., 20.]))
            np.testing.assert_allclose(x[0], 0.)
            np.testing.assert_allclose(y[0], 0.)
            np.testing.assert_allclose(x[2, 3]-x[2, 2], 2*(x[1, 3]-x[1, 2]))
            self.assertAlmostEqual(y[2, 2], 2*y[1, 2])


if __name__ == "__main__":
    unittest.main()
