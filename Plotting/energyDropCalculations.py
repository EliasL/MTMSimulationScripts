from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from Management.updateCSV import read_macrodata_csv
from MTMath.energyFunction import ContiEnergy
from .dataFunctions import get_metadata


PLASTIC_EVENT_COLUMNS = (
    "nr_elements_with_m3_fix_change",
    "nr_elements_with_m3_change",
)


def infer_plastic_event_column(df, *, required=True):
    """Return the available CSV column containing per-step plastic events."""
    column = next((name for name in PLASTIC_EVENT_COLUMNS if name in df), None)
    if column is None and required:
        raise KeyError(f"Missing plastic-event column; tried {PLASTIC_EVENT_COLUMNS}.")
    return column


def infer_energy_column(df, average_energy=False):
    if average_energy is True:
        candidates = ["avg_energy"]
    elif average_energy is False:
        candidates = ["total_energy", "energy"]
    elif average_energy is None:
        candidates = ["total_energy", "energy", "avg_energy"]
    else:
        raise ValueError("average_energy must be True, False, or None.")

    for col in candidates:
        if col in df:
            return col
    raise KeyError(f"No energy column found. Tried {candidates}.")


def infer_stress_column(df):
    """Return the preferred shear-stress column for general legacy analyses."""
    sigma_col = "avg_sigma12"
    piola_col = "avg_P12"
    if sigma_col in df:
        sigma = np.asarray(df[sigma_col], dtype=float)
        if not np.all(sigma == 0):
            return sigma_col
    if piola_col in df:
        print(
            "Warning: avg_sigma12 is unavailable or identically zero; using "
            "avg_P12 for a legacy stress analysis."
        )
        return piola_col
    raise KeyError("No stress column found")


def infer_energy_prediction_stress_column(df):
    """Return Cauchy shear stress for MTS2D's spatial affine shear increment.

    MTS2D applies each load step by left multiplication, so an element follows
    ``dF/dgamma = K F``. The exact generalized shear stress is therefore the
    reference-area average of ``(P F.T)[0, 1] = J * sigma[0, 1]``. Existing CSV
    files do not store that average, and ``avg_sigma12`` is the intended simple
    approximation for the nearly isochoric simulations. Never silently replace
    it with ``avg_P12``: PK1's second index belongs to each element's reference
    map, so differently oriented element references can cancel in the raw
    component average. It is also not conjugate to this loading path once
    element deformation gradients become heterogeneous.
    """
    stress_col = "avg_sigma12"
    if stress_col in df:
        if "avg_P12" in df:
            warnings.warn(
                "Using avg_sigma12 for the affine-step energy prediction; "
                "avg_P12 is intentionally ignored.",
                RuntimeWarning,
                stacklevel=2,
            )
        return stress_col
    raise KeyError(
        "Missing Cauchy shear-stress column 'avg_sigma12'. Do not substitute "
        "'avg_P12' in the affine-step energy prediction."
    )


def stress_corrected_drop_column(correction_order=2, tangent="current"):
    if correction_order == 1:
        return "stress_corrected_drop_first_order"
    if correction_order == 2:
        if tangent in {"current", "gamma", "gamma_i"}:
            return "stress_corrected_drop_second_order"
        if tangent in {"gamma0", "zero", "gamma_zero"}:
            return "stress_corrected_drop_second_order_gamma0"
    raise ValueError(
        "Expected correction_order=1 or correction_order=2 with "
        "tangent in {'current', 'gamma0'}."
    )


def volume_from_metadata(meta):
    if "L" in meta and meta["L"] is not None:
        L = float(meta["L"])
        return float(L * L)
    dims = meta.get("dims") or meta.get("N")
    if dims:
        n1, n2 = dims
        return float(n1 * n2)
    return None


def _simple_shear_tangent(load_i, *, bulk_modulus=4.0):
    """Return the spatial tangent a_1212 along upper simple shear."""
    tangent = np.full_like(load_i, np.nan, dtype=float)
    finite = np.isfinite(load_i)
    if not np.any(finite):
        return tangent

    F_i = np.zeros((int(np.sum(finite)), 2, 2), dtype=float)
    F_i[..., 0, 0] = 1.0
    F_i[..., 0, 1] = load_i[finite]
    F_i[..., 1, 1] = 1.0
    tangent[finite] = ContiEnergy.elasticity_tensor(
        F_i, K=bulk_modulus, eulerian=True
    )[
        ..., 0, 1, 0, 1
    ]
    return tangent


def _simple_shear_tangent_gamma0(shape, *, bulk_modulus=4.0):
    """Return the spatial tangent a_1212 at gamma=0."""
    F0 = np.zeros((1, 2, 2), dtype=float)
    F0[..., 0, 0] = 1.0
    F0[..., 1, 1] = 1.0
    tangent0 = ContiEnergy.elasticity_tensor(
        F0, K=bulk_modulus, eulerian=True
    )[..., 0, 1, 0, 1]
    return np.full(shape, float(tangent0[0]), dtype=float)


def _read_simulation_config(csv_path):
    if csv_path is None:
        return {}
    config_path = csv_path.parent / "config.conf"
    if not config_path.exists():
        return {}

    values = {}
    for raw_line in config_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        values[key] = value
    return values


def calculate_energy_step_data(
    csv_file_path=None,
    *,
    df=None,
    metadata=None,
    average_energy=None,
):
    """
    Compute per-load-step internal-energy drops and Taylor predictions using
    the averaged Cauchy shear stress and spatial tangent a_1212:

        E_hat_{n+1} = E_n + V_0 <sigma_12>_n delta_gamma_n
            + 0.5 V_0 a_1212,n delta_gamma_n^2.

    ``<sigma_12>`` approximates the exact generalized stress
    ``<(P F.T)_12>_A0 = <J sigma_12>_A0`` for MTS2D's left-multiplicative
    affine shear step. The spatial second-order tangent is evaluated along the
    homogeneous simple-shear path and retained as an approximation; it is not a
    measured macroscopic tangent.

    The stress-corrected drop is stored as predicted E_hat_{n+1} minus the
    measured E_{n+1}. Positive values are therefore energy drops relative to
    the elastic prediction.
    """
    csv_path = Path(csv_file_path) if csv_file_path is not None else None
    if df is None:
        if csv_path is None:
            raise ValueError("Provide either csv_file_path or df.")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = read_macrodata_csv(csv_path)
    if metadata is None:
        metadata = get_metadata(str(csv_path)) if csv_path is not None else {}

    if "load" not in df.columns:
        raise KeyError("Missing 'load' column.")

    load = np.asarray(df["load"], dtype=float)
    if load.ndim != 1 or load.size < 2:
        raise ValueError("'load' must be 1D with at least 2 points.")

    reference_volume = volume_from_metadata(metadata)
    if reference_volume is None:
        source = str(csv_path) if csv_path is not None else "provided dataframe"
        raise ValueError(f"Could not infer system volume from metadata for {source}")
    nr_elements = 2 * reference_volume

    energy_col = infer_energy_column(df, average_energy=average_energy)
    stress_col = infer_energy_prediction_stress_column(df)
    use_average_energy = (
        energy_col.startswith("avg_") if average_energy is None else bool(average_energy)
    )

    energy = np.asarray(df[energy_col], dtype=float)
    cauchy_stress = np.asarray(df[stress_col], dtype=float)
    if energy.shape != load.shape:
        raise ValueError(f"Energy shape mismatch: {energy.shape} vs load {load.shape}")
    if cauchy_stress.shape != load.shape:
        raise ValueError(
            f"Cauchy stress shape mismatch: {cauchy_stress.shape} vs load {load.shape}"
        )

    energy_total = energy * nr_elements if use_average_energy else energy
    load_i = load[:-1]
    load_ip1 = load[1:]
    delta_gamma = np.diff(load)
    sigma12_i = cauchy_stress[:-1]
    e_i = energy_total[:-1]
    e_real_next = energy_total[1:]

    config = _read_simulation_config(csv_path)
    energy_function = config.get("energyFunction", "contiSquare")
    if energy_function != "contiSquare":
        raise ValueError(
            "Second-order energy correction only supports energyFunction="
            f"'contiSquare', got {energy_function!r}."
        )
    bulk_modulus = float(config.get("bulkModulus", 4.0))

    a1212_i = _simple_shear_tangent(load_i, bulk_modulus=bulk_modulus)
    a1212_gamma0_i = _simple_shear_tangent_gamma0(
        load_i.shape, bulk_modulus=bulk_modulus
    )

    e_pred_next = e_i + reference_volume * sigma12_i * delta_gamma
    e_pred_next_second_order = (
        e_pred_next + 0.5 * reference_volume * a1212_i * delta_gamma**2
    )
    e_pred_next_second_order_gamma0 = (
        e_pred_next + 0.5 * reference_volume * a1212_gamma0_i * delta_gamma**2
    )

    prediction_error = e_real_next - e_pred_next
    second_order_prediction_error = e_real_next - e_pred_next_second_order
    second_order_gamma0_prediction_error = (
        e_real_next - e_pred_next_second_order_gamma0
    )

    abs_prediction_error = np.abs(prediction_error)
    abs_second_order_prediction_error = np.abs(second_order_prediction_error)
    abs_second_order_gamma0_prediction_error = np.abs(
        second_order_gamma0_prediction_error
    )

    relative_prediction_error = np.full_like(abs_prediction_error, np.nan)
    relative_second_order_prediction_error = np.full_like(
        abs_second_order_prediction_error, np.nan
    )
    relative_second_order_gamma0_prediction_error = np.full_like(
        abs_second_order_gamma0_prediction_error, np.nan
    )
    denom = np.abs(e_real_next)
    nonzero = denom > 0
    relative_prediction_error[nonzero] = abs_prediction_error[nonzero] / denom[nonzero]
    relative_second_order_prediction_error[nonzero] = (
        abs_second_order_prediction_error[nonzero] / denom[nonzero]
    )
    relative_second_order_gamma0_prediction_error[nonzero] = (
        abs_second_order_gamma0_prediction_error[nonzero] / denom[nonzero]
    )

    drop_scale = nr_elements if use_average_energy else 1.0
    first_order_drop = -prediction_error / drop_scale
    second_order_drop = -second_order_prediction_error / drop_scale
    second_order_gamma0_drop = -second_order_gamma0_prediction_error / drop_scale

    step_df = pd.DataFrame(
        {
            "load_i": load_i,
            "load_ip1": load_ip1,
            "delta_gamma": delta_gamma,
            "sigma12_i": sigma12_i,
            "stress_i": sigma12_i,
            "a1212_i": a1212_i,
            "a1212_gamma0_i": a1212_gamma0_i,
            # Backward-compatible aliases for downstream analysis code.
            "simple_shear_tangent_i": a1212_i,
            "simple_shear_tangent_gamma0_i": a1212_gamma0_i,
            "E_i": e_i,
            "E_ip1_pred": e_pred_next,
            "E_ip1_pred_second_order": e_pred_next_second_order,
            "E_ip1_pred_second_order_gamma0": e_pred_next_second_order_gamma0,
            "E_ip1_real": e_real_next,
            "prediction_error": prediction_error,
            "abs_prediction_error": abs_prediction_error,
            "relative_prediction_error": relative_prediction_error,
            "second_order_prediction_error": second_order_prediction_error,
            "abs_second_order_prediction_error": abs_second_order_prediction_error,
            "relative_second_order_prediction_error": relative_second_order_prediction_error,
            "second_order_gamma0_prediction_error": second_order_gamma0_prediction_error,
            "abs_second_order_gamma0_prediction_error": abs_second_order_gamma0_prediction_error,
            "relative_second_order_gamma0_prediction_error": relative_second_order_gamma0_prediction_error,
            "stress_corrected_drop_first_order": first_order_drop,
            "stress_corrected_drop_second_order": second_order_drop,
            "stress_corrected_drop_second_order_gamma0": second_order_gamma0_drop,
        }
    )
    info = {
        "csv_path": str(csv_path) if csv_path is not None else None,
        "reference_volume": reference_volume,
        "volume": reference_volume,
        "nr_elements": nr_elements,
        "stress_col": stress_col,
        "cauchy_col": stress_col,
        "energy_col": energy_col,
        "converted_avg_energy_to_total": use_average_energy,
        "used_piola_stress": False,
        "energy_function": energy_function,
        "bulk_modulus": bulk_modulus,
    }
    return step_df, info


def extract_energy_drops_from_dataframe(
    df,
    *,
    csv_file_path=None,
    metadata=None,
    strain_lim=(-np.inf, np.inf),
    energy_key="total_e_change_from_init",
    average_energy=False,
    stress_corrected=True,
    correction_order=2,
    tangent="current",
    drop_sign="negative",
    min_drop=0.0,
    plastic_only=False,
):
    """Extract one positive drop array and its row mask from one dataframe."""
    if drop_sign not in {"negative", "positive"}:
        raise ValueError("drop_sign must be 'negative' or 'positive'.")
    if not np.isfinite(min_drop) or min_drop < 0:
        raise ValueError("min_drop must be finite and nonnegative.")

    info = {}
    if stress_corrected:
        steps, info = calculate_energy_step_data(
            csv_file_path,
            df=df,
            metadata=metadata,
            average_energy=average_energy,
        )
        drop_column = stress_corrected_drop_column(correction_order, tangent)
        step_drops = np.asarray(steps[drop_column], dtype=float)
        signed_change = np.zeros(len(df), dtype=float)
        signed_change[1:] = -step_drops
        info["drop_column"] = drop_column
    else:
        if energy_key not in df:
            raise KeyError(f"Missing energy-change column {energy_key!r}.")
        signed_change = np.asarray(df[energy_key], dtype=float).copy()
        if drop_sign == "positive":
            signed_change *= -1.0
        info["energy_col"] = energy_key
        info["drop_column"] = energy_key

    if "load_step" in df and df["load_step"].iloc[0] == 1:
        signed_change[0] = 0.0
    strain = np.asarray(df["load"], dtype=float)
    mask = (
        (-signed_change > min_drop)
        & (strain > strain_lim[0])
        & (strain < strain_lim[1])
    )

    if plastic_only:
        plastic_col = infer_plastic_event_column(df)
        mask &= np.asarray(df[plastic_col] >= 1, dtype=bool)
        info["plastic_event_col"] = plastic_col

    mask = np.asarray(mask, dtype=bool)
    return -signed_change[mask], mask, signed_change, info
