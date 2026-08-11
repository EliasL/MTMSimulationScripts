from typing import Literal, Tuple
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


Vec2 = Tuple[float, float] | np.ndarray
Vec2_ = Tuple[float, float] | np.ndarray | None


class LatticeFigure:
    """Minimal helper for making 2D lattice vector figures on a single Axes.

    The figure/subplot layout is created outside this class; you pass an `ax`
    to the constructor and all drawing methods act on that internal axis.
    """

    def __init__(
        self,
        ax,
        margin: float = 0.5,
        point_fmt: str = "ko",
        grid_linestyle: str = "--",
        grid_color: str = "gray",
        vector_color: str = "black",
        limits: Tuple[float, float, float, float] | None = None,
        font_size: int = 25,
        label_spacing: float = 0.1,
        origin: Vec2 = (0.0, 0.0),
        basis: Tuple[Vec2, Vec2] | None = None,
    ) -> None:
        self.ax = ax
        self.margin = margin
        self.point_fmt = point_fmt
        self.grid_linestyle = grid_linestyle
        self.grid_color = grid_color
        self.vector_color = vector_color
        self._limits = limits
        self._vector_endpoints = np.empty((0, 2), dtype=float)
        self.font_size = font_size
        self.origin = (float(origin[0]), float(origin[1]))
        self._basis = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        if basis is not None:
            self.set_basis(*basis)

    def set_basis(self, e1: Vec2, e2: Vec2) -> None:
        self._basis = (
            np.array([float(e1[0]), float(e1[1])]),
            np.array([float(e2[0]), float(e2[1])]),
        )

    def _auto_limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if self._vector_endpoints.size == 0:
            max_abs = 0.0
        else:
            max_abs = float(np.max(np.abs(self._vector_endpoints)))

        half = max_abs + float(self.margin)
        xlim = (-half, half)
        ylim = (-half, half)
        return xlim, ylim

    def _set_lims(self, xlim: Vec2_, ylim: Vec2_) -> Tuple[Vec2, Vec2]:
        if xlim is None and self._limits is not None:
            xmin, xmax, _, _ = self._limits
            xlim = (xmin, xmax)
        if ylim is None and self._limits is not None:
            _, _, ymin, ymax = self._limits
            ylim = (ymin, ymax)

        if xlim is None or ylim is None:
            auto_xlim, auto_ylim = self._auto_limits()
            xlim = auto_xlim if xlim is None else xlim
            ylim = auto_ylim if ylim is None else ylim
        return xlim, ylim

    def limits(self, xlim: Vec2_ = None, ylim: Vec2_ = None, margin=None):
        e1, e2 = self._basis

        (xmin, xmax), (ymin, ymax) = self._set_lims(xlim, ylim)
        if margin is None:
            margin = self.margin
        xmin -= margin  # * max(abs(e1[0]), abs(e2[0]))
        xmax += margin  # * max(abs(e1[0]), abs(e2[0]))
        ymin -= margin  # * max(abs(e1[1]), abs(e2[1]))
        ymax += margin  # * max(abs(e1[1]), abs(e2[1]))
        return xmin, xmax, ymin, ymax

    def drawGrid(self, linestyle: str = "--", color: str = "gray", **kwargs) -> None:
        if self._limits is None:
            raise ValueError("drawGrid requires self.limits to be set.")

        e1, e2 = self._basis
        if e1.shape != (2,) or e2.shape != (2,):
            raise ValueError("drawGrid requires a 2D basis to be set.")

        ox, oy = self.origin
        origin = np.array([ox, oy], dtype=float)

        xmin, xmax, ymin, ymax = self.limits()
        limits = (float(xmin), float(xmax), float(ymin), float(ymax))

        def line_segment_in_box(p0: np.ndarray, v: np.ndarray):
            t_min = -np.inf
            t_max = np.inf
            for p_i, v_i, min_i, max_i in (
                (p0[0], v[0], limits[0], limits[1]),
                (p0[1], v[1], limits[2], limits[3]),
            ):
                if abs(v_i) < 1e-12:
                    if p_i < min_i or p_i > max_i:
                        return None
                    continue
                t1 = (min_i - p_i) / v_i
                t2 = (max_i - p_i) / v_i
                t_low = t1 if t1 < t2 else t2
                t_high = t2 if t1 < t2 else t1
                if t_low > t_min:
                    t_min = t_low
                if t_high < t_max:
                    t_max = t_high
                if t_min > t_max:
                    return None
            p_start = p0 + t_min * v
            p_end = p0 + t_max * v
            return p_start, p_end

        def draw_family(direction: np.ndarray, offset: np.ndarray) -> None:
            offset_norm = float(np.linalg.norm(offset))
            if offset_norm < 1e-12:
                return
            plot_kwargs = dict(kwargs)
            line_alpha = plot_kwargs.pop("alpha", 0.7)
            line_width = plot_kwargs.pop("linewidth", 1)
            max_dim = max(
                abs(limits[0]), abs(limits[1]), abs(limits[2]), abs(limits[3])
            )
            max_steps = int(np.ceil((2.0 * max_dim + 2.0) / offset_norm)) + 1
            for k in range(-max_steps, max_steps + 1):
                p0 = origin + offset * float(k)
                segment = line_segment_in_box(p0, direction)
                if segment is None:
                    continue
                p_start, p_end = segment
                self.ax.plot(
                    [p_start[0], p_end[0]],
                    [p_start[1], p_end[1]],
                    linewidth=line_width,
                    alpha=line_alpha,
                    linestyle=linestyle,
                    color=color,
                    zorder=0,
                    **plot_kwargs,
                )

        # First family: lines parallel to e1, offset by e2.
        draw_family(e1, e2)
        # Second family: lines parallel to e2, offset by e1.
        draw_family(e2, e1)

    def draw_lattice_points(
        self,
        origin: Vec2_ = None,
        xlim_: Vec2_ = None,
        ylim_: Vec2_ = None,
        maxDepth: int = 50,
        basis: Tuple[Vec2, Vec2] | None = None,
    ) -> None:
        xmin, xmax, ymin, ymax = self.limits(xlim=xlim_, ylim=ylim_, margin=0.1)

        e1, e2 = (
            self._basis
            if basis is None
            else (
                np.array([float(basis[0][0]), float(basis[0][1])]),
                np.array([float(basis[1][0]), float(basis[1][1])]),
            )
        )
        if origin is None:
            ox, oy = self.origin
        else:
            ox, oy = float(origin[0]), float(origin[1])

        for i in range(-maxDepth, maxDepth + 1):
            for j in range(-maxDepth, maxDepth + 1):
                px, py = (i * e1) + (j * e2)
                px += ox
                py += oy
                if xmin <= px <= xmax and ymin <= py <= ymax:
                    self.ax.plot(px, py, self.point_fmt, zorder=1)

    def draw_vector(
        self,
        v: Vec2,
        label: str | None = None,
        label_pos: Vec2 | None = None,
        color: str | None = None,
        origin: Vec2_ = None,
        ha: Literal["left", "center", "right"] | None = "left",
        va: Literal["top", "center", "bottom", "baseline", "center_baseline"]
        | None = "center",
        label_spacing=0.1,
        **quiver_kwargs,
    ) -> None:
        c = self.vector_color if color is None else color

        # Prepare quiver kwargs with sane defaults while allowing caller overrides.
        qkw = dict(quiver_kwargs)
        qkw.setdefault("headwidth", 5)
        qkw.setdefault("headlength", 7)
        qkw.setdefault("headaxislength", 6)

        if origin is None:
            ox, oy = self.origin
        else:
            ox, oy = float(origin[0]), float(origin[1])
        vx, vy = float(v[0]), float(v[1])

        end = (ox + vx, oy + vy)

        # Matplotlib's Quiver does not reliably support dashed shafts. For arrowless
        # segments (e.g. parallelogram closure edges), fall back to a plain Line2D
        # while keeping the public API consistent.
        arrowless = (
            float(qkw.get("headwidth", 0)) == 0
            and float(qkw.get("headlength", 0)) == 0
            and float(qkw.get("headaxislength", 0)) == 0
        )

        linestyle = qkw.get("linestyle", "-")
        if arrowless and linestyle not in ("-", "solid", None):
            lw = float(qkw.get("linewidth", 2.0))
            self.ax.plot(
                [ox, end[0]],
                [oy, end[1]],
                linestyle=linestyle,
                linewidth=lw,
                color=c,
                zorder=3,
            )
        else:
            self.ax.quiver(
                ox,
                oy,
                vx,
                vy,
                angles="xy",
                scale_units="xy",
                scale=1,
                color=c,
                zorder=3,
                **qkw,
            )

        # Track endpoints for automatic axis limits.
        if (ox, oy) == (0.0, 0.0):
            self._vector_endpoints = np.vstack(
                [self._vector_endpoints, np.array([[end[0], end[1]]], dtype=float)]
            )
        else:
            self._vector_endpoints = np.vstack(
                [
                    self._vector_endpoints,
                    np.array([[ox, oy], [end[0], end[1]]], dtype=float),
                ]
            )

        if label is not None:
            # Label position is specified relative to the vector midpoint
            mx = ox + 0.5 * vx
            my = oy + 0.5 * vy

            if label_pos is None:
                lx, ly = mx, my
            else:
                lx = mx + float(label_pos[0])
                ly = my + float(label_pos[1])

            ha = "left" if ha is None else ha
            va = "center" if va is None else va

            # Push label away from anchor according to alignment
            dx = 0.0
            dy = 0.0
            if ha == "left":
                dx = label_spacing
            elif ha == "right":
                dx = -label_spacing

            if va == "bottom":
                dy = label_spacing
            elif va == "top":
                dy = -label_spacing

            self.ax.text(
                lx + dx,
                ly + dy,
                label,
                fontsize=self.font_size,
                ha=ha,
                va=va,
            )

    def draw_parallelogram(
        self,
        a: Vec2,
        b: Vec2,
        origin: Vec2_ = None,
        color: str | None = None,
        extra_linestyle: str = "--",
        labels: Tuple[str | None, str | None] | str = (None, None),
        has=(None, "right"),  # horizontal label anchors
        vas=("top", None),  # vertical label anchors
        spacing=(0.1, 0.1),
        **quiver_kwargs,
    ) -> None:
        """Draw the fundamental parallelogram spanned by `a` and `b`.

        The two "extra" edges (those not starting at `origin`) are drawn without
        arrowheads and are dashed by default.

        All segments are drawn via `draw_vector`.
        """

        c = self.vector_color if color is None else color

        if origin is None:
            ox, oy = self.origin
        else:
            ox, oy = float(origin[0]), float(origin[1])
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])

        if isinstance(labels, str):
            base = labels.strip()
            # Accept either a bare LaTeX fragment like r"\\mathbf{e}" or a math-wrapped
            # string like r"$\\mathbf{e}$". Internally we always add exactly one pair of $.
            if base.startswith("$") and base.endswith("$") and len(base) >= 2:
                base = base[1:-1]
            labels = (rf"${base}_1$", rf"${base}_2$")

        # Base edges (with arrowheads)
        self.draw_vector(
            (ax, ay),
            origin=(ox, oy),
            color=c,
            label=labels[0],
            ha=has[0],
            va=vas[0],
            label_spacing=spacing[0],
            **quiver_kwargs,
        )
        self.draw_vector(
            (bx, by),
            origin=(ox, oy),
            color=c,
            label=labels[1],
            ha=has[1],
            va=vas[1],
            label_spacing=spacing[1],
            **quiver_kwargs,
        )

        # Extra edges: dashed and without arrowheads.
        # Remove arrow-related kwargs to avoid duplicate keyword errors.
        extra_kwargs = dict(quiver_kwargs)
        extra_kwargs.update(
            {
                "headwidth": 0,
                "headlength": 0,
                "headaxislength": 0,
                "linestyle": extra_linestyle,
            }
        )

        self.draw_vector(
            (bx, by),
            origin=(ox + ax, oy + ay),
            color=c,
            **extra_kwargs,
        )
        self.draw_vector(
            (ax, ay),
            origin=(ox + bx, oy + by),
            color=c,
            **extra_kwargs,
        )

    def style_axis(
        self,
        xlim_: Vec2_ = None,
        ylim_: Vec2_ = None,
        set_ax_lims=True,
        draw_grid=True,
        hide_ticklabels: bool = True,
        equal_aspect: bool = True,
        maxDepth: int = 50,
        draw_points: bool = True,
    ) -> None:
        # Ensure grid/points stay behind vectors.
        self.ax.set_axisbelow(True)
        if draw_points:
            self.draw_lattice_points(
                xlim_=xlim_,
                ylim_=ylim_,
                maxDepth=maxDepth,
            )
        if draw_grid:
            self.drawGrid()

        xmin, xmax, ymin, ymax = self.limits(xlim_, ylim_)
        self.ax.set_xticks(range(int(xmin), int(xmax) + 1))
        self.ax.set_yticks(range(int(ymin), int(ymax) + 1))

        if set_ax_lims:
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)

        if hide_ticklabels:
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])

        if equal_aspect:
            self.ax.set_aspect("equal")

        # Hide tick marks without disabling the axis (so grid stays visible).
        self.ax.tick_params(axis="both", which="both", length=0)
        self.ax.set_frame_on(False)

    def add_corner_label(
        self,
        text: str,
        color: str = "black",
        dx: float = 0.1,
        dy: float = 0.1,
        fontsize: int | None = None,
    ) -> None:
        xmin, xmax, ymin, ymax = self.limits()
        fs = self.font_size if fontsize is None else fontsize
        self.ax.text(
            xmin + dx,
            ymax - dy,
            text,
            fontsize=fs,
            ha="left",
            va="top",
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
        )


def integer_shear_examples():
    # Define vectors e1 and three versions of e2
    e1: Vec2 = (1, 0)
    e2_list = [(-1, 1), (0, 1), (1, 1)]  # Different cases for e2

    # Create the figure (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, e2 in enumerate(e2_list):
        lf = LatticeFigure(axes[i])

        # Vectors
        lf.draw_vector(e1, label=r"$\mathbf{e_1}$", label_pos=(0.5, 0.1), color="black")

        # Match prior e2 label placement logic
        e2_label_x = e2[0] / 2 + 0.1 + (0.1 if e2[0] == 1 else 0)
        e2_label_y = e2[1] / 2
        lf.draw_vector(
            e2,
            label=r"$\mathbf{e_2}$",
            label_pos=(e2_label_x, e2_label_y),
            color="black",
        )

        # Axis styling
        lf.style_axis()

    plt.tight_layout()
    plt.show()


def three_bases_same_lattice():
    """Show two different lattice bases that generate the same Z^2 lattice."""

    # Base lattice vectors
    e1: Vec2 = (1, 0)
    e2: Vec2 = (0, 1)
    o1 = (-2, 0)

    # New basis (same lattice): e1bar = e1 - e2, e2bar = e2
    e1bar: Vec2 = (e1[0] - e2[0], e1[1] - e2[1])
    e2bar: Vec2 = e2
    o2 = (0, 0)

    # New basis (same lattice): e1bar = e1 - e2, e2bar = e2
    e1hat: Vec2 = (-e1bar[0], e1bar[1])
    e2hat: Vec2 = e2
    o3 = (3, 0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    lf = LatticeFigure(ax, limits=(-2, 3, -1, 1))

    # Fundamental parallelograms
    lf.draw_parallelogram(e1, e2, labels=r"\mathbf{a}", origin=o1, color="black")
    lf.draw_parallelogram(
        e1bar,
        e2bar,
        labels=r"\bar{\mathbf{a}}",
        has=("right", "right"),
        origin=o2,
        color="tab:blue",
        spacing=(0, 0.1),
    )
    lf.draw_parallelogram(
        e1hat,
        e2hat,
        labels=r"\hat{\mathbf{a}}",
        has=("left", None),
        origin=o3,
        color="tab:red",
        spacing=(0, 0.1),
    )
    lf.style_axis()
    path = Path("Plots/LatticeBases.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"Fig saved to : {path}")
    plt.show()


def two_lattices_side_by_side():
    """Plot two lattices with different bases side by side on the same axes."""

    fig, ax = plt.subplots(figsize=(6, 5))

    # Bases for the two lattices.
    e1_left: Vec2 = (0.5, 1.5)
    e2_left: Vec2 = (0.0, 2.0)
    e1_right: Vec2 = (-0.5, 1.5)
    e2_right: Vec2 = (0.0, 2.0)

    # Place them side by side by shifting their origins.
    o_left: Vec2 = (-1.5, 0.0)
    o_right: Vec2 = (1.5, 0.0)

    # Visible regions for each lattice.
    limits1 = (-3.0, -0.6, -0.5, 3.0)
    limits2 = (0.6, 3.0, -0.5, 3.0)

    # Draw basis vectors for each lattice.
    lf1 = LatticeFigure(ax, limits=limits1, basis=(e1_left, e2_left), origin=o_left)
    lf2 = LatticeFigure(ax, limits=limits2, basis=(e1_right, e2_right), origin=o_right)
    lf1.draw_parallelogram(e1_left, e2_left, labels=r"\mathbf{e}", color="black")
    lf2.draw_parallelogram(
        e1_right,
        e2_right,
        labels=r"\bar\mathbf{e}",
        color="tab:blue",
        spacing=(0, 0.1),
        has=("right", "left"),
    )

    lf1.style_axis(set_ax_lims=False)
    lf2.style_axis(set_ax_lims=False)

    plt.tight_layout()
    path = Path("Plots/TwoLattices.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"Fig saved to : {path}")
    plt.show()


def four_lattices_translation_rotation(
    translation: float | Vec2 = 0.2,
    rotation_deg: float = 10.0,
):
    """Show translation and rotation freedoms using two lattice pairs."""

    fig, ax = plt.subplots(figsize=(9, 4))

    # Base lattice vectors.
    e1: Vec2 = (1.0, 0.0)
    e2: Vec2 = (0.0, 1.0)

    if isinstance(translation, (int, float)):
        t = np.array([float(translation), float(translation)])
    else:
        t = np.array([float(translation[0]), float(translation[1])])

    theta = np.deg2rad(rotation_deg)
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    e1_rot = rot @ np.array(e1, dtype=float)
    e2_rot = rot @ np.array(e2, dtype=float)

    # Left/right bounding boxes so the two pairs stay separate.
    half_span_x = 1.1
    half_span_y = 1.5
    x_center_left = -1.7
    x_center_right = 1.7
    y_center = 0.5
    left_limits = (
        x_center_left - half_span_x,
        x_center_left + half_span_x,
        y_center - half_span_y,
        y_center + half_span_y,
    )
    right_limits = (
        x_center_right - half_span_x,
        x_center_right + half_span_x,
        y_center - half_span_y,
        y_center + half_span_y,
    )

    # Translation pair (left).
    o_left: Vec2 = (x_center_left, 0.0)
    o_left_shift: Vec2 = (o_left[0] + t[0], o_left[1] + t[1])

    lf_left = LatticeFigure(
        ax,
        limits=left_limits,
        basis=(e1, e2),
        origin=o_left,
        point_fmt="ko",
        grid_color="lightgray",
        vector_color="black",
    )
    lf_left_shift = LatticeFigure(
        ax,
        limits=left_limits,
        basis=(e1, e2),
        origin=o_left_shift,
        point_fmt="bo",
        grid_color="lightgray",
        vector_color="tab:blue",
    )

    lf_left.draw_parallelogram(e1, e2, color="black")
    lf_left_shift.draw_parallelogram(e1, e2, color="tab:blue")

    lf_left.style_axis(set_ax_lims=False, draw_grid=True, draw_points=True)
    lf_left_shift.style_axis(set_ax_lims=False, draw_grid=True, draw_points=True)

    lf_left.add_corner_label("A")
    # Rotation pair (right).
    o_right: Vec2 = (x_center_right, 0.0)

    lf_right = LatticeFigure(
        ax,
        limits=right_limits,
        basis=(e1, e2),
        origin=o_right,
        point_fmt="ko",
        grid_color="lightgray",
        vector_color="black",
    )
    lf_right_rot = LatticeFigure(
        ax,
        limits=right_limits,
        basis=(e1_rot, e2_rot),
        origin=o_right,
        point_fmt="ro",
        grid_color="tab:red",
        vector_color="tab:red",
    )

    lf_right.draw_parallelogram(e1, e2, color="black")
    lf_right_rot.draw_parallelogram(e1_rot, e2_rot, color="tab:red")

    lf_right.style_axis(set_ax_lims=False, draw_grid=True, draw_points=True)
    lf_right_rot.style_axis(set_ax_lims=False, draw_grid=True, draw_points=True)

    lf_right.add_corner_label("B")

    plt.tight_layout()
    path = Path("Plots/TranslationAndRotation.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"Fig saved to : {path}")
    plt.show()


def lagrange_reduction_bases_row(
    save_path: str | Path = "Plots/LagrangeReductionBasesRow.pdf",
    show: bool = True,
):
    """Plot e^0 -> e^1 -> e^2 -> e~ for the Lagrange reduction example."""

    # Basis matrices (columns are basis vectors).
    e0 = np.array([[1.0, 0.0], [-1.0, 1.0]])
    m1 = np.array([[1.0, 0.0], [0.0, -1.0]])
    m2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    m3 = np.array([[1.0, -1.0], [0.0, 1.0]])

    e1 = e0 @ m1
    e2 = e1 @ m2
    e_tildebar = e2 @ m3

    # Keep these strict so accidental convention changes fail loudly.
    if not np.array_equal(e1, np.array([[1.0, 0.0], [-1.0, -1.0]])):
        raise ValueError("Unexpected e^1 basis from e^0 m1.")
    if not np.array_equal(e2, np.array([[0.0, 1.0], [-1.0, -1.0]])):
        raise ValueError("Unexpected e^2 basis from e^1 m2.")
    if not np.array_equal(e_tildebar, np.array([[0.0, 1.0], [-1.0, 0.0]])):
        raise ValueError("Unexpected reduced basis from e^2 m3.")

    bases = [e0, e1, e2, e_tildebar]
    titles = [
        r"$\bar{\mathbf{e}}^0$",
        r"$\bar{\mathbf{e}}^1$",
        r"$\bar{\mathbf{e}}^2$",
        r"$\tilde{\bar{\mathbf{e}}}$",
    ]

    max_abs = float(np.max(np.abs(np.hstack(bases))))
    lim = max_abs + 0.3
    common_limits = (-lim, lim, -lim, lim)

    fig, axes = plt.subplots(1, 4, figsize=(8, 2), sharex=True, sharey=True)
    title_y = 1.02
    for i, (ax, basis, title) in enumerate(zip(axes, bases, titles)):
        e1_vec = tuple(basis[:, 0])
        e2_vec = tuple(basis[:, 1])
        x=-0.1 
        e1_label_pos = [(x,0),(x,0),(x,0),(0.0, -0.2)][i]
        e2_label_pos = [(x,0),(x,0),(x,0),(-0.12, 0.0)][i]
        label_base = (
            r"\bar{\mathbf{e}}"
            if i < 3
            else r"\tilde{\bar{\mathbf{e}}}"
        )
        lf = LatticeFigure(
            ax,
            limits=common_limits,
            basis=(e1_vec, e2_vec),
            font_size=16,
            margin=0.2,
        )
        # Closure edges only (dashed) so e1/e2 arrows can be color-coded clearly.
        lf.draw_vector(
            e1_vec,
            origin=e2_vec,
            color="0.55",
            headwidth=0,
            headlength=0,
            headaxislength=0,
            linestyle="--",
            linewidth=1.6,
        )
        lf.draw_vector(
            e2_vec,
            origin=e1_vec,
            color="0.55",
            headwidth=0,
            headlength=0,
            headaxislength=0,
            linestyle="--",
            linewidth=1.6,
        )
        lf.draw_vector(
            e1_vec,
            label=rf"${label_base}_1^{i}$",
            label_pos=e1_label_pos,
            color="tab:red",
            width=.017,
            ha="left" if i < 2 else "right",
            va="bottom",
        )
        lf.draw_vector(
            e2_vec,
            label=rf"${label_base}_2^{i}$",
            label_pos=e2_label_pos,
            color="tab:blue",
            width=.017,
            ha="right" if i < 2 else "left",
            va="bottom",
        )
        lf.style_axis(
            set_ax_lims=True,
            draw_grid=True,
            draw_points=True,
            maxDepth=6,
            hide_ticklabels=True,
        )
        ax.set_title(title, fontsize=17, y=title_y, pad=0)

    # Place transform labels between neighboring panels, aligned with title level.
    transition_labels = [
        r"$\times \mathbf{m}_1 \rightarrow$",
        r"$\times \mathbf{m}_2 \rightarrow$",
        r"$\times \mathbf{m}_3 \rightarrow$",
    ]
    for i, text in enumerate(transition_labels):
        axes[i].text(
            1.03,
            title_y+0.05,
            text,
            transform=axes[i].transAxes,
            ha="center",
            va="center",
            fontsize=16,
        )

    fig.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Fig saved to : {out}")
    if show:
        plt.show()
    return fig, axes


def coordinate_index_conventions(
    save_path: str | Path = "Plots/CoordinateIndexConventions.pdf",
    show: bool = False,
):
    """Illustrate reference/current Cartesian and lattice-vector conventions.

    The reference lattice is generated from unit vectors
    ``a_1^0=(1, 0)`` and ``a_2^0=(1/2, sqrt(3)/2)``.  The current lattice is
    obtained from the homogeneous deformation

    ``F = [[1, 0], [1/4, 1]]``.

    The lattice origin is placed at ``(1, 1)`` in each panel so that the
    figure emphasizes the transformation of lattice vectors rather than the
    arbitrary translation used to draw them.
    """

    e1: Vec2 = (1.0, 0.0)
    e2: Vec2 = (0.0, 1.0)
    a1: Vec2 = (1.0, 0.0)
    a2: Vec2 = (0.5, float(np.sqrt(3.0) / 2.0))
    deformation = np.array([[1.0, 0.0], [0.25, 1.0]])
    a1_current = deformation @ np.asarray(a1, dtype=float)
    a2_current = deformation @ np.asarray(a2, dtype=float)

    lattice_origin: Vec2 = (1.0, 1.0)
    common_limits = (-0.20, 3.20, -0.20, 3.20)
    arrow_kwargs = dict(
        width=0.006,
        headwidth=3.0,
        headlength=4.0,
        headaxislength=3.5,
    )

    def draw_configuration(
        ax,
        lattice_basis,
        *,
        cartesian_labels,
        lattice_label_base,
        title,
    ):
        lattice_basis_1, lattice_basis_2 = lattice_basis
        third_family_basis = (
            lattice_basis_2[0] - lattice_basis_1[0],
            lattice_basis_2[1] - lattice_basis_1[1],
        )

        # The coordinate grid remains orthonormal in both frames.  Only its
        # labels change from the reference basis E_I to the current basis e_i.
        cartesian = LatticeFigure(
            ax,
            limits=common_limits,
            basis=(e1, e2),
            origin=(0.0, 0.0),
            margin=0.0,
            font_size=10,
        )
        cartesian.drawGrid(linestyle=":", color="0.55", alpha=0.30)

        # The two calls provide the three line families of the triangular
        # lattice: horizontal, +60 degrees, and -60 degrees.  After F is
        # applied, these become the corresponding sheared line families.
        lattice = LatticeFigure(
            ax,
            limits=common_limits,
            basis=(lattice_basis_1, lattice_basis_2),
            origin=lattice_origin,
            margin=0.0,
            point_fmt="C0.",
            vector_color="tab:blue",
            font_size=10,
        )
        lattice.drawGrid(linestyle="-", color="tab:blue", alpha=0.20)
        third_family = LatticeFigure(
            ax,
            limits=common_limits,
            basis=(lattice_basis_1, third_family_basis),
            origin=lattice_origin,
            margin=0.0,
        )
        third_family.drawGrid(linestyle="-", color="tab:blue", alpha=0.20)
        lattice.draw_lattice_points(
            xlim_=common_limits[:2],
            ylim_=common_limits[2:],
            maxDepth=8,
        )

        cartesian.draw_vector(
            e1,
            label=rf"$\mathbf{{{cartesian_labels[0]}}}_1$",
            label_pos=(0.38, -0.10),
            color="0.15",
            ha="center",
            va="top",
            **arrow_kwargs,
        )
        cartesian.draw_vector(
            e2,
            label=rf"$\mathbf{{{cartesian_labels[1]}}}_2$",
            label_pos=(-0.10, 0.38),
            color="0.15",
            ha="right",
            va="center",
            **arrow_kwargs,
        )

        lattice.draw_vector(
            lattice_basis_1,
            origin=lattice_origin,
            label=rf"${lattice_label_base}_1$",
            label_pos=(0.0, -0.10),
            color="tab:blue",
            ha="center",
            va="top",
            label_spacing=0.08,
            **arrow_kwargs,
        )
        lattice.draw_vector(
            lattice_basis_2,
            origin=lattice_origin,
            label=rf"${lattice_label_base}_2$",
            label_pos=(-0.08, 0.0),
            color="tab:blue",
            ha="right",
            va="center",
            label_spacing=0.08,
            **arrow_kwargs,
        )

        ax.set_xlim(common_limits[:2])
        ax.set_ylim(common_limits[2:])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(0.0, 4.0, 1.0))
        ax.set_yticks(np.arange(0.0, 4.0, 1.0))
        ax.tick_params(
            axis="both",
            which="both",
            length=2.5,
            width=0.6,
            colors="0.35",
            labelsize=8,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=10.5, pad=3)

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.1), sharex=True, sharey=True)
    draw_configuration(
        axes[0],
        (a1, a2),
        cartesian_labels=("E", "E"),
        lattice_label_base=r"\mathbf{a}^{0}",
        title="Reference Configuration",
    )
    draw_configuration(
        axes[1],
        (a1_current, a2_current),
        cartesian_labels=("e", "e"),
        lattice_label_base=r"\mathbf{a}",
        title="Current Configuration",
    )
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    fig.subplots_adjust(wspace=0.04, left=0.06, right=0.98, bottom=0.08, top=0.86)

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Fig saved to : {output}")
    if show:
        plt.show()
    return fig, axes


if __name__ == "__main__":
    # integer_shear_examples()
    # three_bases_same_lattice()
    # two_lattices_side_by_side()
    #four_lattices_translation_rotation()
    lagrange_reduction_bases_row()

    # f = np.array(((1, 0), (0.25, 1)))
    # v = np.array((8, 0))
    # v1 = np.array((5, 0.25))
    # v2 = np.array((3, 0.25))
    # print(f @ v1)
    # print(f @ v2)
    pass 
