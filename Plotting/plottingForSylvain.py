"""Plot MTS2D VTU files colored by one scalar field.

Usage:
    python plottingForSylvain.py mesh.vtu
    python plottingForSylvain.py mesh.vtu sigma12 --output sigma12.png
    python plottingForSylvain.py before.vtu after.vtu sigma12 --output diff.png

With only a VTU file, the script lists the available fields.
This script assumes vtuDataForSylvain.py is in the same folder.
"""

import argparse
from pathlib import Path
import re
import sys

import numpy as np

from vtuDataForSylvain import VTUData


def plot_vtu(vtu_file, field_name, output=None, show=True):
    data = VTUData(vtu_file)
    values, location, resolved_name = data.field(field_name)
    return plot_values(
        data,
        values,
        location,
        title=resolved_name,
        colorbar_label=resolved_name,
        output=output,
        show=show,
    )


def plot_vtu_difference(
    first_vtu_file,
    second_vtu_file,
    field_name,
    output=None,
    show=True,
):
    first_vtu_file, second_vtu_file, difference_description = order_difference_files(
        first_vtu_file,
        second_vtu_file,
    )
    first = VTUData(first_vtu_file)
    second = VTUData(second_vtu_file)
    assert_same_mesh(first, second)

    first_values, first_location, first_name = first.field(field_name)
    second_values, second_location, second_name = second.field(field_name)
    if first_location != second_location:
        raise ValueError(
            f"Field {field_name!r} is {first_location} data in the first VTU, "
            f"but {second_location} data in the second VTU."
        )
    if first_values.shape != second_values.shape:
        raise ValueError(
            f"Field {field_name!r} has shape {first_values.shape} in the first VTU, "
            f"but {second_values.shape} in the second VTU."
        )

    difference = second_values - first_values
    return plot_values(
        second,
        difference,
        first_location,
        title=f"{first_name}: {difference_description}",
        colorbar_label=f"{first_name} change",
        output=output,
        show=show,
        center_zero=True,
    )


def order_difference_files(first_vtu_file, second_vtu_file):
    first_role = pre_post_role(first_vtu_file)
    second_role = pre_post_role(second_vtu_file)
    if {first_role, second_role} == {"pre", "post"}:
        warn_if_names_do_not_match(first_vtu_file, second_vtu_file)
        if first_role == "pre":
            return first_vtu_file, second_vtu_file, "post - pre"
        return second_vtu_file, first_vtu_file, "post - pre"

    warn(
        "The two VTU files are not a pre/post pair. Expected names ending in "
        "'_pre.<step>.vtu' and '_post.<step>.vtu'. Plotting second - first."
    )
    return first_vtu_file, second_vtu_file, "second - first"


def pre_post_role(vtu_file):
    match = re.search(r"_(pre|post)(?=(?:\.\d+)?\.vtu$)", Path(vtu_file).name)
    return match.group(1) if match else None


def warn_if_names_do_not_match(first_vtu_file, second_vtu_file):
    first_path = Path(first_vtu_file)
    second_path = Path(second_vtu_file)
    if first_path.parent != second_path.parent:
        warn("The two VTU files are not in the same folder.")
    if normalized_pre_post_name(first_path.name) != normalized_pre_post_name(
        second_path.name
    ):
        warn("The two VTU files do not look like matching pre/post names.")


def normalized_pre_post_name(name):
    name = re.sub(r"_(pre|post)(?=(?:\.\d+)?\.vtu$)", "_STATE", name)
    return re.sub(r"nrM=\d+", "nrM=*", name)


def warn(message):
    print(f"Warning: {message}", file=sys.stderr)


def assert_same_mesh(first, second):
    if first.points.shape != second.points.shape:
        raise ValueError(
            f"The two VTU files have different point array shapes: "
            f"{first.points.shape} and {second.points.shape}."
        )
    if first.triangles.shape != second.triangles.shape:
        raise ValueError(
            f"The two VTU files have different triangle array shapes: "
            f"{first.triangles.shape} and {second.triangles.shape}."
        )
    if not np.allclose(first.points, second.points):
        warn("The two VTU files have different point coordinates. Plotting on the second mesh.")
    if not np.array_equal(first.triangles, second.triangles):
        warn(
            "The two VTU files have different triangle connectivity. "
            "Subtracting fields by cell order and plotting on the second mesh."
        )


def plot_values(
    data,
    values,
    location,
    title,
    colorbar_label,
    output=None,
    show=True,
    center_zero=False,
):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    points = data.points
    triangles = data.triangles
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    norm = None
    if center_zero:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmin < 0 < vmax:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    if location == "cell":
        mesh = ax.tripcolor(
            triangulation,
            facecolors=values,
            edgecolors="none",
            cmap="coolwarm" if center_zero else "viridis",
            norm=norm,
        )
    elif location == "point":
        mesh = ax.tripcolor(
            triangulation,
            values,
            shading="gouraud",
            edgecolors="none",
            cmap="coolwarm" if center_zero else "viridis",
            norm=norm,
        )
    else:
        raise ValueError(f"Unhandled field location {location!r}")

    ax.triplot(triangulation, color="black", linewidth=0.05, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label=colorbar_label)

    if output is not None:
        fig.savefig(output, dpi=300)
    if show:
        plt.show()
    return fig, ax


def print_fields(vtu_file):
    data = VTUData(vtu_file)
    print("Cell fields:")
    for name in data.cell_field_names:
        print(f"  {name}")
    print("Point fields:")
    for name in data.point_field_names:
        print(f"  {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot an MTS2D VTU mesh colored by a scalar field."
    )
    parser.add_argument(
        "positional",
        nargs="+",
        help=(
            "Use: mesh.vtu | mesh.vtu field | first.vtu second.vtu field. "
            "The three-argument form plots second - first."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional image path. If omitted, only an interactive window is shown.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save without opening an interactive window.",
    )
    args = parser.parse_args()
    if len(args.positional) == 1:
        print_fields(args.positional[0])
    elif len(args.positional) == 2:
        plot_vtu(
            args.positional[0],
            args.positional[1],
            args.output,
            show=not args.no_show,
        )
    elif len(args.positional) == 3:
        plot_vtu_difference(
            args.positional[0],
            args.positional[1],
            args.positional[2],
            args.output,
            show=not args.no_show,
        )
    else:
        parser.error("Expected 1, 2, or 3 positional arguments.")


if __name__ == "__main__":
    main()
