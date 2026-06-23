from pathlib import Path

import meshio
import numpy as np


class VTUData:
    """Small reader for MTS2D VTU files.

    This file is intentionally standalone. Keep it in the same folder as
    plottingForSylvain.py and run that script directly.
    """

    def __init__(self, vtu_file):
        self.path = Path(vtu_file)
        if not self.path.is_file():
            raise FileNotFoundError(f"No VTU file found at {self.path}")
        self.mesh = meshio.read(self.path)

    @property
    def points(self):
        return np.asarray(self.mesh.points)

    @property
    def triangles(self):
        cells = getattr(self.mesh, "cells_dict", {})
        if "triangle" in cells:
            return np.asarray(cells["triangle"], dtype=int)
        if len(self.mesh.cells) == 1 and self.mesh.cells[0].type == "triangle":
            return np.asarray(self.mesh.cells[0].data, dtype=int)
        available = [cell.type for cell in self.mesh.cells]
        raise ValueError(f"Expected triangle cells, found {available}")

    @property
    def point_field_names(self):
        return sorted(self.mesh.point_data)

    @property
    def cell_field_names(self):
        return sorted(self.mesh.cell_data)

    def field(self, name):
        """Return (values, location, resolved_name) for a scalar field."""
        if name in self.mesh.cell_data:
            values = self._single_cell_block(name)
            return self._scalar(values, name), "cell", name

        if name in self.mesh.point_data:
            values = self.mesh.point_data[name]
            return self._scalar(values, name), "point", name

        raise KeyError(
            f"Field {name!r} was not found. Available cell fields: "
            f"{self.cell_field_names}. Available point fields: {self.point_field_names}."
        )

    def _single_cell_block(self, name):
        blocks = self.mesh.cell_data[name]
        if len(blocks) != 1:
            raise ValueError(
                f"Cell field {name!r} has {len(blocks)} blocks; expected one."
            )
        return blocks[0]

    def _scalar(self, values, name):
        values = np.asarray(values)
        if values.ndim == 2 and values.shape[1] == 1:
            return values[:, 0]
        if values.ndim == 1:
            return values
        raise ValueError(
            f"Field {name!r} has shape {values.shape}; expected a scalar field."
        )
