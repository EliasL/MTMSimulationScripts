from typing import Literal, Tuple
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


Vec2 = Tuple[float, float]


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
    ) -> None:
        self.ax = ax
        self.margin = margin
        self.point_fmt = point_fmt
        self.grid_linestyle = grid_linestyle
        self.grid_color = grid_color
        self.vector_color = vector_color
        self.limits = limits
        self._vector_endpoints = np.empty((0, 2), dtype=float)
        self.font_size = font_size

    def draw_vector(
        self,
        v: Vec2,
        label: str | None = None,
        label_pos: Vec2 | None = None,
        color: str | None = None,
        origin: Vec2 = (0.0, 0.0),
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
        origin: Vec2 = (0.0, 0.0),
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

    def _auto_limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if self._vector_endpoints.size == 0:
            max_abs = 0.0
        else:
            max_abs = float(np.max(np.abs(self._vector_endpoints)))

        half = max_abs + float(self.margin)
        xlim = (-half, half)
        ylim = (-half, half)
        return xlim, ylim

    def style_axis(
        self,
        xlim: Tuple[float, float] | None = None,
        ylim: Tuple[float, float] | None = None,
        hide_ticklabels: bool = True,
        equal_aspect: bool = True,
    ) -> None:
        if self.limits is not None:
            xmin, xmax, ymin, ymax = self.limits
            xlim = (xmin - self.margin, xmax + self.margin)
            ylim = (ymin - self.margin, ymax + self.margin)
        else:
            if xlim is None or ylim is None:
                auto_xlim, auto_ylim = self._auto_limits()
                xlim = auto_xlim if xlim is None else xlim
                ylim = auto_ylim if ylim is None else ylim

        # Use floor/ceil to cover the visible region without adding extra buffer.
        ix_min = int(np.floor(xlim[0]))
        ix_max = int(np.ceil(xlim[1]))
        iy_min = int(np.floor(ylim[0]))
        iy_max = int(np.ceil(ylim[1]))

        # Ensure grid/points stay behind vectors.
        self.ax.set_axisbelow(True)
        # Draw points
        for x in range(ix_min, ix_max + 1):
            for y in range(iy_min, iy_max + 1):
                self.ax.plot(x, y, self.point_fmt, zorder=1)

        self.ax.set_xticks(range(ix_min, ix_max + 1))
        self.ax.set_yticks(range(iy_min, iy_max + 1))
        self.ax.grid(
            True,
            which="both",
            linestyle=self.grid_linestyle,
            color=self.grid_color,
            zorder=0,
        )

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

        if hide_ticklabels:
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])

        if equal_aspect:
            self.ax.set_aspect("equal")

        # Hide tick marks without disabling the axis (so grid stays visible).
        self.ax.tick_params(axis="both", which="both", length=0)
        self.ax.set_frame_on(False)


def simple_integer_shear_transformation():
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
    lf.draw_parallelogram(e1, e2, labels=r"\mathbf{e}", origin=o1, color="black")
    lf.draw_parallelogram(
        e1bar,
        e2bar,
        labels=r"\bar{\mathbf{e}}",
        has=("right", "right"),
        origin=o2,
        color="tab:blue",
        spacing=(0, 0.1),
    )
    lf.draw_parallelogram(
        e1hat,
        e2hat,
        labels=r"\hat{\mathbf{e}}",
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


if __name__ == "__main__":
    # simple_integer_shear_transformation()
    three_bases_same_lattice()
