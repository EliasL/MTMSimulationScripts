import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from MTMath.meshUtils import structured_triangular_mesh
from Plotting.mesh_plotting import (
    MeshFigure,
    MeshStyle,
    draw_mesh_edges,
    draw_node_labels,
    draw_triangle_mesh,
)


class IllustrativeMeshPlottingTests(unittest.TestCase):
    def setUp(self):
        self.nodes, self.connectivity = structured_triangular_mesh(
            (2, 2), diagonal="minor"
        )

    def tearDown(self):
        plt.close("all")

    def test_triangle_mesh_draws_unique_edges_and_nodes(self):
        _, ax = plt.subplots()
        artists = draw_triangle_mesh(ax, self.nodes, self.connectivity)

        self.assertEqual(len(artists.faces.get_paths()), 2)
        self.assertEqual(len(artists.edges.get_segments()), 5)
        self.assertEqual(artists.nodes.get_offsets().shape, (4, 2))

    def test_style_can_draw_an_outline_only_mesh(self):
        _, ax = plt.subplots()
        artists = draw_triangle_mesh(
            ax,
            self.nodes,
            self.connectivity,
            style=MeshStyle(
                color="tab:orange",
                linestyle="--",
                draw_faces=False,
                draw_nodes=False,
            ),
        )

        self.assertIsNone(artists.faces)
        self.assertIsNone(artists.nodes)
        self.assertEqual(len(artists.edges.get_segments()), 5)

    def test_explicit_edge_overlay_can_highlight_a_diagonal(self):
        _, ax = plt.subplots()
        edge = np.array([[0, 3]])
        artist = draw_mesh_edges(
            ax, self.nodes, edge, color="black", linewidth=3.0, zorder=5
        )

        self.assertEqual(len(artist.get_segments()), 1)
        np.testing.assert_allclose(artist.get_segments()[0], self.nodes[edge[0]])

    def test_labels_support_individual_offsets(self):
        _, ax = plt.subplots()
        labels = draw_node_labels(
            ax,
            self.nodes,
            ["a", "b"],
            node_ids=[0, 3],
            offsets=[[0.1, 0.0], [-0.1, 0.2]],
        )

        self.assertEqual([label.get_text() for label in labels], ["a", "b"])
        np.testing.assert_allclose(labels[0].get_position(), self.nodes[0] + [0.1, 0.0])
        np.testing.assert_allclose(labels[1].get_position(), self.nodes[3] + [-0.1, 0.2])

    def test_axis_bound_wrapper_tracks_drawn_mesh_bounds(self):
        _, ax = plt.subplots()
        mesh = MeshFigure(ax)
        mesh.draw_mesh(self.nodes, self.connectivity)
        mesh.configure_axis(padding_fraction=0.1)

        self.assertEqual(ax.get_aspect(), 1.0)
        np.testing.assert_allclose(ax.get_xlim(), (-0.1, 1.1))
        np.testing.assert_allclose(ax.get_ylim(), (-0.1, 1.1))
        self.assertFalse(any(spine.get_visible() for spine in ax.spines.values()))


if __name__ == "__main__":
    unittest.main()
