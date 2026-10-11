"""Recover legacy atlas fields from full-precision geometry, validating exports."""

from dataclasses import replace

import numpy as np

from MTMath.energyFunction import ContiEnergy


def recover_legacy_fields(state, volume):
    """Recompute Conti-square energy and Cauchy stress for unit lattice triangles.

    Legacy F_Fix is the matrix of the two current triangle edges, with a
    reference area of 1/2. Its exported components and scalar fields have only
    five significant digits; the point coordinates retain full precision.
    Clockwise local bases are reflected to a positive material orientation,
    a symmetry of the square-lattice potential, before stress evaluation.
    """
    fields = state.cell_fields
    if len(state.triangles) != round(2*volume) or "T11" in fields:
        raise ValueError("Legacy recovery requires non-reconnecting unit lattice triangles.")
    if not all(f"F_Fix{i}{j}" in fields for i in (1, 2) for j in (1, 2)):
        raise ValueError("Missing legacy fixed-reference deformation fields.")
    points = state.points[state.triangles]
    deformation = (points[:, 1:] - points[:, :1]).swapaxes(-1, -2).copy()
    exported = np.stack([fields[f"F_Fix{i}{j}"] for i in (1, 2) for j in (1, 2)], axis=-1).reshape(-1, 2, 2)
    def export_tolerance(values):
        magnitude = np.maximum(np.abs(values), 1e-300)
        return .5*10.**(np.floor(np.log10(magnitude))-4) + 1e-10
    if np.any(np.abs(deformation-exported) > export_tolerance(exported)):
        raise ValueError("Triangle geometry does not match the legacy force-reference matrix.")
    determinants = np.linalg.det(deformation)
    areas = .5*np.abs(determinants)
    if np.any(areas <= 0) or not np.isclose(areas.sum(), volume, rtol=1e-9):
        raise ValueError("Legacy mesh has invalid current areas or total volume.")
    deformation[determinants < 0, :, 1] *= -1
    energy = .5*ContiEnergy.energy_from_F(deformation)
    sigma = ContiEnergy.cauchy_from_F(deformation)
    if np.any(np.abs(energy-fields["energy_field"]) > export_tolerance(fields["energy_field"])):
        raise ValueError("Recomputed potential does not match the exported energy within its rounding precision.")
    recovered = dict(fields)
    recovered["legacy_exported_energy_field"] = fields["energy_field"]
    recovered["legacy_exported_sigma12"] = fields["sigma12"]
    recovered["energy_field"] = energy
    recovered["sigma12"] = sigma[:, 0, 1]
    recovered["current_area"] = areas
    return replace(state, cell_fields=recovered)


def current_area_cauchy_mean(state):
    return float(np.average(state.cell_fields["sigma12"],
                            weights=state.cell_fields["current_area"]))
