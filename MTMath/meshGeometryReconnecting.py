import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.colors import ListedColormap
from .poincareEnergy import plotPoincareDisk, C2Plane, generate_poincare_disk
from matplotlib.colors import BoundaryNorm


class DraggableTriangulation:
    def triangle_basis_indices(self, tri, diag_pair):
        """Given a triangle (i,j,k) and the diagonal pair (d0,d1),
        return (o, u, v) where o is the vertex NOT on the diagonal, and
        vectors are taken as a = p[u]-p[o], b = p[v]-p[o]."""
        i, j, k = tri
        d0, d1 = diag_pair
        if i != d0 and i != d1:
            return i, j, k
        if j != d0 and j != d1:
            return j, i, k
        return k, i, j

    def reference_triangle_basis_indices(self, tri):
        """Return a fixed basis chosen from the reference geometry."""
        i, j, k = tri
        ri = self.reference_points[i]
        rj = self.reference_points[j]
        rk = self.reference_points[k]
        edges = [
            (float(np.sum((ri - rj) * (ri - rj))), (i, j)),
            (float(np.sum((rj - rk) * (rj - rk))), (j, k)),
            (float(np.sum((rk - ri) * (rk - ri))), (k, i)),
        ]
        edges.sort(key=lambda item: item[0])
        (_, (u1, v1)), (_, (u2, v2)) = edges[0], edges[1]
        if u1 == u2 or u1 == v2:
            other_b = v2 if u1 == u2 else u2
            return u1, v1, other_b
        if v1 == u2 or v1 == v2:
            other_b = v2 if v1 == u2 else u2
            return v1, u1, other_b
        raise RuntimeError(
            f"Expected the two shortest reference edges in triangle {tri} to share one vertex."
        )

    def basis_indices(self, tri, diag_pair):
        """Return the currently active basis for a triangle."""
        if self.use_reference_basis:
            return self.reference_triangle_basis_indices(tri)
        return self.triangle_basis_indices(tri, diag_pair)

    def __init__(self, ax, points, ax_g=None, poincare_transformation="triangular"):
        assert points.shape == (4, 2), "Provide exactly four 2D points"
        self.ax = ax
        self.initial_points = np.copy(points)
        self.points = points
        self.reference_points = np.copy(points)
        self.ax_g = ax_g  # optional axes for (G11,G22) scatter
        self.poincare_transformation = poincare_transformation
        self.g_scatter = None
        self.g_colors = ["tab:green", "tab:orange"]
        self.alt_region_lines = []
        self.use_reference_basis = False

        # Initial triangulation uses diagonal (0, 2): triangles (0,1,2) and (0,2,3)
        self.diagonal_02 = True
        self.edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # outer quad edges
            (0, 2),  # initial diagonal
        ]

        # Heatmap (prediction matrix) showing how many elements would violate if the selected point moved
        self.heatmap_im = None
        self.heatmap_visible = True
        self.grid_size = 500  # pixels per axis
        self.shear_step = 0.05  # shear increment per arrow key press

        # Debug toggles/handles
        self.debug = False
        self.debug_text = None
        self._last_cur = None
        self._last_rec = None
        self._last_idx = None
        self._grid_xs = None
        self._grid_ys = None

        # Heatmap showing count of violated elements across current (2) and after reconnect (2)
        # Index mapping (by count):
        #   0 -> white  (0/4 violated)
        #   1 -> green  (1/4 violated)
        #   2 -> yellow (2/4 violated)
        #   3 -> orange (3/4 violated)
        #   4 -> red    (4/4 violated)
        alpha = 0.2
        self.heatmap_cmap = ListedColormap(
            [
                (1.0, 1.0, 1.0, alpha),  # 0: white
                (0.0, 1.0, 0.0, alpha),  # 1: green
                (1.0, 1.0, 0.0, alpha),  # 2: yellow
                (1.0, 0.65, 0.0, alpha),  # 3: orange
                (1.0, 0.0, 0.0, alpha),  # 4: red
            ]
        )
        # NaN handling for imshow: fully transparent so they don't look like valid white bins
        self.heatmap_cmap.set_bad((1.0, 1.0, 1.0, 0.0))

        # Discrete norm for 5 bins (0..4)
        self.heatmap_norm = BoundaryNorm(
            boundaries=np.arange(-0.5, 5.5, 1),
            ncolors=self.heatmap_cmap.N,
            clip=True,
        )
        # Draw points and edges
        self.point_artist = ax.plot(
            points[:, 0], points[:, 1], "o", color="red", picker=5
        )[0]
        self.line_artists = []
        for i, j in self.edges:
            (ln,) = ax.plot(
                [self.points[i, 0], self.points[j, 0]],
                [self.points[i, 1], self.points[j, 1]],
                "-",
                color="blue",
            )
            self.line_artists.append(ln)

        self.dragging_index = None

        # Keep track of the last selected point (persists after mouse release)
        self.selected_index = points.shape[0] - 2  # default to second last point

        # Selected point backdrop (slightly larger dot behind selected point)
        self.selected_back_artist = self.ax.plot(
            [self.points[self.selected_index, 0]],
            [self.points[self.selected_index, 1]],
            "o",
            color="black",
            alpha=0.3,
            markersize=12,  # slightly larger than the main marker
            zorder=self.point_artist.get_zorder() - 1,  # draw behind main dots
        )[0]

        # Triangle face patches (for coloring by Gram-matrix criterion)
        self.face_patches = []
        self.update_element_color()

        # Storage for element vectors (arrows) and Gram matrix labels
        self.vector_artists = []  # list[FancyArrowPatch]
        self.G_texts = []  # list[Text]

        # Initial draw for element vectors and Gram matrices
        self.update_element_arrows()
        self.update_gram_labels()
        self.update_g_scatter()

        # Optional scatter plot: one dot per element in Poincaré disk
        if self.ax_g is not None:
            self.ax_g.set_title("Elements in Poincaré disk")
            self.ax_g.set_xlabel("x")
            self.ax_g.set_ylabel("y")
            self.ax_g.set_aspect("equal", adjustable="box")
            # Initialize with current values
            g_points = self.compute_poincare_points()
            self.g_scatter = self.ax_g.scatter(
                g_points[:, 0], g_points[:, 1], c=self.g_colors, zorder=3
            )

        # Connect events
        self.cid_press = self.point_artist.figure.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cid_release = self.point_artist.figure.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cid_motion = self.point_artist.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )
        self.cid_key = self.point_artist.figure.canvas.mpl_connect(
            "key_press_event", self.on_key
        )

        self.update_selected_marker()
        self.update_heatmap()

    def nearest_point_index(self, x, y) -> int:
        """Return index of closest vertex to (x,y)."""
        diffs = self.points - np.array([x, y])
        return int(np.argmin(np.sum(diffs * diffs, axis=1)))

    def update_selected_marker(self):
        """Position/visibility of the backdrop circle for the selected point."""
        if getattr(self, "selected_back_artist", None) is None:
            return

        # If we don't show the heatmap, we don't show the selected node
        self.selected_back_artist.set_visible(self.heatmap_visible)

        p = self.points[self.selected_index]
        self.selected_back_artist.set_data([p[0]], [p[1]])
        # Keep it slightly larger than the main points
        try:
            base_ms = self.point_artist.get_markersize()
            self.selected_back_artist.set_markersize(base_ms * 1.6)
        except Exception:
            pass

    # ---------- Interaction handlers ----------
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        contains, _ = self.point_artist.contains(event)
        if not contains:
            return
        # find nearest point
        self.dragging_index = self.nearest_point_index(event.xdata, event.ydata)
        self.selected_index = self.dragging_index
        self.update_selected_marker()
        self.update_heatmap()

    def on_release(self, event):
        self.dragging_index = None
        # self.update_gram_labels()

    def on_motion(self, event):
        if self.dragging_index is None:
            return
        if event.inaxes != self.ax:
            return
        # update point position
        self.points[self.dragging_index] = [event.xdata, event.ydata]
        self.update()
        # Update debug readout under cursor
        if (
            self.debug
            and self._last_cur is not None
            and event.xdata is not None
            and event.ydata is not None
            and self._grid_xs is not None
            and self._grid_ys is not None
        ):
            # Map mouse position to nearest grid cell
            ix = int(
                np.clip(
                    np.searchsorted(self._grid_xs, event.xdata),
                    0,
                    len(self._grid_xs) - 1,
                )
            )
            iy = int(
                np.clip(
                    np.searchsorted(self._grid_ys, event.ydata),
                    0,
                    len(self._grid_ys) - 1,
                )
            )
            cur_v = int(self._last_cur[iy, ix])
            rec_v = int(self._last_rec[iy, ix])
            idx_v = int(self._last_idx[iy, ix])
            if self.debug_text is not None:
                self.debug_text.set_text(f"cur={cur_v} rec={rec_v} idx={idx_v}")

    def on_key(self, event):
        if event.key == "d":
            self.debug = not self.debug
            if self.debug and self.debug_text is None:
                self.debug_text = self.ax.text(
                    0.02, 0.90, "", transform=self.ax.transAxes, va="top"
                )
        elif event.key == "r":
            self.reconnect()
            return
        elif event.key == "t":
            self.reset()
            return
        elif event.key == "down":
            # Negative shear in x with respect to y
            self.apply_shear(kx=-self.shear_step)
            return
        elif event.key == "up":
            # Positive shear in x with respect to y
            self.apply_shear(kx=+self.shear_step)
            return
        elif event.key == "left":
            # Negative shear in y with respect to x
            self.apply_shear(ky=-self.shear_step)
            return
        elif event.key == "right":
            # Positive shear in y with respect to x
            self.apply_shear(ky=+self.shear_step)
            return
        elif event.key == "b":
            # Toggle heatmap visibility
            self.heatmap_visible = not self.heatmap_visible
            if self.heatmap_im is not None:
                self.update_heatmap()
                self.heatmap_im.set_visible(self.heatmap_visible)
            # No geometry change, just redraw
            self.ax.figure.canvas.draw_idle()
            return
        elif event.key == "f":
            self.use_reference_basis = not self.use_reference_basis
        self.update()

    def compute_diagonal_indices(self):
        """Return the tuple of point indices forming the current diagonal."""
        return (0, 2) if self.diagonal_02 else (1, 3)

    def compute_diagonal_length(self):
        """Compute the Euclidean length of the current diagonal."""
        i, j = self.compute_diagonal_indices()
        dx = self.points[i, 0] - self.points[j, 0]
        dy = self.points[i, 1] - self.points[j, 1]
        return float(np.hypot(dx, dy))

    def triangles(self, flipped: bool = False):
        """Return the two triangles as tuples of vertex indices.
        If flipped=True, returns the configuration after flipping the diagonal."""
        d02 = self.diagonal_02 ^ bool(flipped)
        return [(0, 1, 2), (0, 2, 3)] if d02 else [(1, 2, 3), (1, 3, 0)]

    def element_vectors(self):
        """For each triangle, return (i, j, k, origin_point, a, b, centroid)."""
        tris = self.triangles(False)
        diag_pair = self.compute_diagonal_indices()
        out = []
        for tri in tris:
            i, j, k = tri
            pi, pj, pk = self.points[i], self.points[j], self.points[k]
            o, u, v = self.basis_indices(tri, diag_pair)
            po = self.points[o]
            a = self.points[u] - po
            b = self.points[v] - po

            centroid = (pi + pj + pk) / 3.0
            out.append((i, j, k, po.copy(), a, b, centroid))
        return out

    def gram_matrix(self, a, b):
        """Return 2x2 Gram matrix of vectors a and b."""
        aa = float(np.dot(a, a))
        ab = float(np.dot(a, b))
        bb = float(np.dot(b, b))
        return np.array([[aa, ab], [ab, bb]], dtype=float)

    def apply_shear(self, kx=0.0, ky=0.0):
        """Apply a simple shear transform to all points.
        kx: shear factor for x with respect to y (x' = x + kx*y)
        ky: shear factor for y with respect to x (y' = y + ky*x)
        """
        M = np.array([[1.0, kx], [ky, 1.0]], dtype=float)
        self.points = self.points @ M
        self.update()

    def compute_poincare_points(self):
        """Return an (2,2) array with rows [x, y] for the two current triangles,
        where (x, y) are the Poincaré disk coordinates mapped from the element
        Gram matrix G computed from the two shortest edges (see element_vectors).
        """
        pts = []
        for i, j, k, origin, a, b, centroid in self.element_vectors():
            G = self.gram_matrix(a, b)
            x, y = C2Plane(G, transformation=self.poincare_transformation)
            zoom = 1
            x = x * zoom * self.grid_size / 2 + self.grid_size / 2
            y = y * zoom * self.grid_size / 2 + self.grid_size / 2
            pts.append([float(x), float(y)])
        return np.asarray(pts, dtype=float)

    def draw_alternative_region(self, ax=None, color="orange", linewidth=1.0, n=200):
        if ax is None:
            ax = self.ax_g
        if ax is None:
            return
        for line in self.alt_region_lines:
            try:
                line.remove()
            except Exception:
                pass
        self.alt_region_lines = []
        grid_size = self.grid_size
        zoom = 1
        G, r_mask = generate_poincare_disk(
            grid_size, zoom, returnMask=True, transformation=self.poincare_transformation
        )

        a = G[..., 0, 0]
        b = 0.5 * (G[..., 0, 1] + G[..., 1, 0])
        c = G[..., 1, 1]

        case_ac = a <= c
        b_lo = np.where(case_ac, -0.5 * a, -0.5 * c)
        b_hi = np.where(
            case_ac,
            np.minimum(1.5 * a, (3.0 * a + c) / 4.0),
            np.minimum(1.5 * c, (a + 3.0 * c) / 4.0),
        )
        region = (b >= b_lo) & (b <= b_hi)
        region = np.where(r_mask, False, region)

        extent = [
            (grid_size / 2) * (1 - 1 / zoom),
            (grid_size / 2) * (1 + 1 / zoom),
            (grid_size / 2) * (1 - 1 / zoom),
            (grid_size / 2) * (1 + 1 / zoom),
        ]
        cs = ax.contour(
            region.astype(float),
            levels=[0.5],
            colors=color,
            linewidths=linewidth,
            origin="lower",
            extent=extent,
            zorder=2,
        )
        if hasattr(cs, "collections"):
            self.alt_region_lines.extend(cs.collections)
        elif hasattr(cs, "artists"):
            self.alt_region_lines.extend(cs.artists)
        else:
            self.alt_region_lines.append(cs)

    def update_g_scatter(self):
        """Update the scatter plot of Poincaré disk coordinates for current triangles."""
        if self.ax_g is None or self.g_scatter is None:
            return
        g_points = self.compute_poincare_points()
        self.g_scatter.set_offsets(g_points)
        self.g_scatter.set_color(self.g_colors)

    def violates(self, G, eps=1e-18):
        """Return True if G[0,1] < 0 or G[0,1] > min(G[0,0], G[1,1])."""
        ab = float(G[0, 1])
        aa = float(G[0, 0])
        bb = float(G[1, 1])
        return ab < 0 - eps or ab - eps > min(aa, bb)

    def update_element_color(self):
        """Recreate triangle face patches and color them based on the Gram-matrix criterion."""
        # Remove old patches
        for p in getattr(self, "face_patches", []):
            p.remove()
        self.face_patches = []

        # Build new patches for current triangulation
        for i, j, k, origin, a, b, centroid in self.element_vectors():
            G = self.gram_matrix(a, b)
            bad = self.violates(G, eps=1e-10)
            facecolor = (
                (1.0, 0.0, 0.0, 0.3) if bad else (0.0, 0.0, 0.0, 0.0)
            )  # red with alpha if bad, transparent otherwise
            poly = Polygon(
                self.points[[i, j, k]],
                closed=True,
                facecolor=facecolor,
                edgecolor="none",
            )
            self.ax.add_patch(poly)
            self.face_patches.append(poly)

    def update_heatmap(self):
        """Compute and display a 5-level heatmap by counting violations across four elements (two now + two after reconnect)."""
        if not self.heatmap_visible:
            # No need to update when not visible
            return

        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin

        H = W = int(self.grid_size)
        xs = np.linspace(xmin, xmax, W)
        ys = np.linspace(ymin, ymax, H)
        X, Y = np.meshgrid(xs, ys)
        self._grid_xs = xs
        self._grid_ys = ys

        def tri_violation_map(tri, diag_pair):
            o, u, v = self.basis_indices(tri, diag_pair)

            def comp(idx):
                if idx == self.selected_index:
                    return X, Y
                else:
                    return self.points[idx, 0], self.points[idx, 1]

            pox, poy = comp(o)
            pux, puy = comp(u)
            pvx, pvy = comp(v)
            axx = pux - pox
            axy = puy - poy
            bxx = pvx - pox
            bxy = pvy - poy
            aa = axx * axx + axy * axy
            bb = bxx * bxx + bxy * bxy
            ab = axx * bxx + axy * bxy
            m = np.logical_or(ab < 0, ab > np.minimum(aa, bb))
            if np.isscalar(m):
                m = np.full((H, W), bool(m))
            return m

        # Current triangulation
        tris_cur = self.triangles(False)
        dpair_cur = self.compute_diagonal_indices()
        m0c = tri_violation_map(tris_cur[0], dpair_cur)
        m1c = tri_violation_map(tris_cur[1], dpair_cur)
        cur = m0c.astype(np.uint8) + m1c.astype(np.uint8)

        # After reconnect (flip diagonal)
        tris_rec = self.triangles(True)
        # diagonal pair after flip is the opposite
        dpair_rec = (1, 3) if dpair_cur == (0, 2) else (0, 2)
        m0r = tri_violation_map(tris_rec[0], dpair_rec)
        m1r = tri_violation_map(tris_rec[1], dpair_rec)
        rec = m0r.astype(np.uint8) + m1r.astype(np.uint8)

        # Count-of-violations index: total violated elements across both states (0..4)
        idx = np.clip(cur + rec, 0, 4).astype(np.uint8)  # 0..4
        self._last_cur = cur
        self._last_rec = rec
        self._last_idx = idx

        if self.heatmap_im is None:
            self.heatmap_im = self.ax.imshow(
                idx,
                extent=(xmin, xmax, ymin, ymax),
                origin="lower",
                cmap=self.heatmap_cmap,
                norm=self.heatmap_norm,  # <— key
                interpolation="nearest",
                zorder=0.05,
            )
        else:
            self.heatmap_im.set_data(idx)
            self.heatmap_im.set_extent((xmin, xmax, ymin, ymax))
        self.heatmap_im.set_visible(self.heatmap_visible)
        self.ax.figure.canvas.draw_idle()

    # ---------- Geometry ops ----------
    def reconnect(self):
        """Flip the internal diagonal: (0,2) <-> (1,3)."""
        if self.diagonal_02:
            # replace (0,2) with (1,3)
            self._replace_edge((0, 2), (1, 3))
        else:
            # replace (1,3) with (0,2)
            self._replace_edge((1, 3), (0, 2))
        self.diagonal_02 = not self.diagonal_02

        # Update only what is not covered by `update()`
        self.update_edges_and_nodes()

        # Do the rest (faces, vectors/grams, labels, scatter) once via the centralized updater
        self.update()

    def _replace_edge(self, old_edge, new_edge):
        # normalize order (i<j) for robust equality
        old_edge = tuple(sorted(old_edge))
        new_edge = tuple(sorted(new_edge))
        self.edges = [tuple(sorted(e)) for e in self.edges]
        # swap in list
        self.edges = [e for e in self.edges if e != old_edge]
        self.edges.append(new_edge)

    def update_element_arrows(self):
        """Update (not recreate) the element vectors (arrows) only."""
        elems = self.element_vectors()

        # Ensure we have exactly two arrow artists per element
        needed = 2 * len(elems)
        while len(self.vector_artists) < needed:
            arr = FancyArrowPatch(
                posA=(0.0, 0.0),
                posB=(0.0, 0.0),
                arrowstyle="->",
                mutation_scale=12,
                color="green",
                lw=2,
                zorder=5,
            )
            self.ax.add_patch(arr)
            self.vector_artists.append(arr)
        while len(self.vector_artists) > needed:
            art = self.vector_artists.pop()
            art.remove()

        # Update arrows (two per triangle: a then b)
        for idx, (i, j, k, origin, a, b, centroid) in enumerate(elems):
            arr1 = self.vector_artists[2 * idx]
            arr2 = self.vector_artists[2 * idx + 1]
            arr1.set_zorder(5)
            arr2.set_zorder(5)
            arr1.set_positions(
                (origin[0], origin[1]), (origin[0] + a[0], origin[1] + a[1])
            )
            arr2.set_positions(
                (origin[0], origin[1]), (origin[0] + b[0], origin[1] + b[1])
            )
            # Color the vectors to match the Gram-matrix/legend color for this element
            if hasattr(self, "g_colors") and len(self.g_colors) > 0:
                vec_color = self.g_colors[idx % len(self.g_colors)]
                arr1.set_color(vec_color)
                arr2.set_color(vec_color)

    def update_gram_labels(self):
        """Update Gram-matrix text labels and the optional (G11,G22) scatter."""
        elems = self.element_vectors()

        # Ensure exactly one text label per element
        while len(self.G_texts) < len(elems):
            self.G_texts.append(
                self.ax.text(
                    0,
                    0,
                    "",
                    va="top",
                    ha="left",
                    transform=self.ax.transAxes,
                    fontfamily="monospace",  # fast native font
                    usetex=False,  # make sure not to trigger LaTeX
                )
            )
        while len(self.G_texts) > len(elems):
            txt = self.G_texts.pop()
            txt.remove()

        # Update labels
        for idx, (i, j, k, origin, a, b, centroid) in enumerate(elems):
            G = self.gram_matrix(a, b)
            label = f"{G[0, 0]:4.1f}, {G[0, 1]:4.1f}\n{G[1, 0]:4.1f}, {G[1, 1]:4.1f}"
            score = min(G[0, 1], min(G[0, 0], G[1, 1]) - G[0, 1])

            # label += "\n" + str(score)
            self.G_texts[idx].set_transform(self.ax.transAxes)
            self.G_texts[idx].set_ha("left")
            self.G_texts[idx].set_va("top")

            base_y = 0.98
            spacing = 0.10
            pos = (0.02, base_y - idx * spacing)
            self.G_texts[idx].set_position(pos)
            self.G_texts[idx].set_text(label)
            self.G_texts[idx].set_bbox(
                dict(
                    boxstyle="round,pad=0.2",
                    fc=self.g_colors[idx % len(self.g_colors)]
                    if hasattr(self, "g_colors")
                    else "w",
                    ec="0.5",
                    alpha=0.3,
                )
            )

    # ---------- Rendering updates ----------
    def update_edges_and_nodes(self):
        # Ensure number of artists matches number of edges
        if len(self.line_artists) != len(self.edges):
            # Recreate artists from scratch
            for ln in self.line_artists:
                ln.remove()
            self.line_artists = []
            for i, j in self.edges:
                (ln,) = self.ax.plot(
                    [self.points[i, 0], self.points[j, 0]],
                    [self.points[i, 1], self.points[j, 1]],
                    "-",
                    color="blue",
                )
                self.line_artists.append(ln)

    def update(self):
        # update point scatter
        self.point_artist.set_data(self.points[:, 0], self.points[:, 1])
        # update each edge
        for ln, (i, j) in zip(self.line_artists, self.edges):
            ln.set_data(
                [self.points[i, 0], self.points[j, 0]],
                [self.points[i, 1], self.points[j, 1]],
            )
        self.update_selected_marker()
        self.update_element_color()
        self.update_element_arrows()
        self.update_gram_labels()
        self.update_g_scatter()
        self.update_heatmap()
        self.ax.figure.canvas.draw_idle()

    def reset(self):
        self.points = np.copy(self.initial_points)
        self.update()


def run_reconnection_demo():
    points = np.array(
        [
            [2, 1.3],  # A0
            [1, 1],  # C0
            [1.0, 0.00],  # B0
            [2, 0.5],  # D0
        ]
    )

    points = np.array(
        [
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ]
    )

    fig, (ax, ax_g) = plt.subplots(1, 2, figsize=(10, 5))
    padding = 4.25
    ax.set_xlim(points[:, 0].min() - padding, points[:, 0].max() + padding)
    ax.set_ylim(points[:, 1].min() - padding, points[:, 1].max() + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Triangulation")

    dt = DraggableTriangulation(ax, points, ax_g=ax_g, poincare_transformation="triangular")
    # Draw disk first so it's firmly in the background
    plotPoincareDisk(
        ax=ax_g,
        grid_size=dt.grid_size,
        depth=4,
        transformation=dt.poincare_transformation,
    )
    dt.draw_alternative_region()

    plt.show()


if __name__ == "__main__":
    run_reconnection_demo()
