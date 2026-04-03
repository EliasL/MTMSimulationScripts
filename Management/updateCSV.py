import csv
import re
from pathlib import Path

import pandas as pd
import numpy as np

def update_df_header(
    df: pd.DataFrame,
    add_total_columns: bool = True,
    add_extrap_energy:bool =True,
    L: int | None = None,
    nr_elements: int | None = None,
):
    # Mapping of old column names to new column names
    rename_map = {
        "Load": "load",
        "Avg energy": "avg_energy",
        "Max energy": "max_energy",
        "Avg RSS": "avg_RSS",
        "Nr plastic deformations": "nr_plastic_deformations",
        "Nr FIRE iterations": "nr_iterations",
        "Nr LBFGS iterations": "nr_iterations",
        "Nr CG iterations": "nr_iterations",
        "Nr FIRE func evals": "nr_func_evals",
        "Nr LBFGS func evals": "nr_func_evals",
        "Nr CG iterations.1": "nr_func_evals",
        "FIRE Term reason": "FIRE_Term_reason",
        "LBFGS Term reason": "LBFGS_Term_reason",
        "CG Term reason": "CG_Term_reason",
        "Run time": "run_time",
        "Est time remaining": "est_time_remaining",
        "maxX": "maxX",
        "minX": "minX",
        "maxY": "maxY",
        "minY": "minY",
        #
        "avg_init_energy_change": "avg_e_change_from_init",
        "avg_RSS": "avg_P12",
        "max_plastic_deformation": "max_m3_nr",
        # Umut headers (Note energy is NOT averaged)
        "Alpha": "load",
        "PreEnergy": "init_energy",
        "PostEnergy": "energy",
        "PreStress": "avg_init_sigma12",
        "PostStress": "avg_sigma12",
        "EnergyChange": "total_e_change_from_init",
        "StressChange": "avg_sigma_change_from_init",
        # Change from xy to 12
        "avg_sigmaxy":"avg_sigma12",
        "avg_Pxy":"avg_P12",
        "avg_init_sigmaxy":"avg_init_sigma12",
    }

    # Rename columns if they exist in the DataFrame
    df = df.rename(columns=rename_map)

    if add_total_columns:
        if nr_elements is None and L is not None:
            nr_elements = int(L) * int(L) * 2
        if nr_elements is not None:
            for col in df.columns:
                if col.startswith("avg_"):
                    total_col = "total_" + col[4:]
                    if total_col not in df.columns:
                        df[total_col] = df[col] * nr_elements
    
    if add_extrap_energy:
        # This energy uses the two previous local minima energies to estimate
        # the energy increase over the strain step. 
        # See equation 5 in Avalanches in the Athermal Quasistatic Limit of Sheared Amorphous Solids: An Atomistic Perspective
        # ΔE = En − E_n+1 + V σ_n δγ
        del_gamma = np.diff(df["load"])
        sigma = df["avg_sigma12"]

    return df


OLD_MACRODATA_HEADER = [
    "load_step",
    "load",
    "avg_energy",
    "avg_energy_change",
    "avg_init_energy",
    "avg_init_energy_change",
    "max_energy",
    "max_force",
    "avg_RSS",
    "nr_plastic_deformations",
    "max_plastic_deformation",
    "max_positive_plastic_jump",
    "max_negative_plastic_jump",
    "nr_iterations",
    "nr_func_evals",
    "LBFGS_Term_reason",
    "CG_Term_reason",
    "FIRE_Term_reason",
    "run_time",
    "minimization_time",
    "write_time",
    "est_time_remaining",
    "cmX",
    "cmY",
    "maxX",
    "minX",
    "maxY",
    "minY",
]

NEW_MACRODATA_HEADER = [
    "load_step",
    "load",
    "total_energy",
    "total_energy_change",
    "total_init_energy",
    "total_e_change_from_init",
    "avg_energy",
    "avg_energy_change",
    "avg_init_energy",
    "avg_e_change_from_init",
    "min_iter_total_energy_change",
    "min_iter_avg_energy_change",
    "max_energy",
    "max_force",
    "avg_sigma12",
    "avg_init_sigma12",
    "avg_sigmaxy_change_from_init",
    "avg_P12",
    "nr_plastic_deformations",
    "nr_red_q1",
    "nr_red_q2",
    "nr_red_q3",
    "nr_red_q4",
    "nr_red_q1_fixed",
    "nr_red_q2_fixed",
    "nr_red_q3_fixed",
    "nr_red_q4_fixed",
    "max_m3_nr",
    "max_positive_plastic_jump",
    "max_negative_plastic_jump",
    "nr_iterations",
    "nr_func_evals",
    "LBFGS_Term_reason",
    "CG_Term_reason",
    "FIRE_Term_reason",
    "run_time",
    "minimization_time",
    "write_time",
    "est_time_remaining",
    "cmX",
    "cmY",
    "maxX",
    "minX",
    "maxY",
    "minY",
]

MID_MACRODATA_HEADER = [
    "load_step",
    "load",
    "avg_energy",
    "avg_energy_change",
    "avg_init_energy",
    "avg_e_change_from_init",
    "max_energy",
    "max_force",
    "avg_sigma12",
    "avg_init_sigma12",
    "avg_sigmaxy_change_from_init",
    "avg_P12",
    "nr_plastic_deformations",
    "max_m3_nr",
    "max_positive_plastic_jump",
    "max_negative_plastic_jump",
    "nr_iterations",
    "nr_func_evals",
    "LBFGS_Term_reason",
    "CG_Term_reason",
    "FIRE_Term_reason",
    "run_time",
    "minimization_time",
    "write_time",
    "est_time_remaining",
    "cmX",
    "cmY",
    "maxX",
    "minX",
    "maxY",
    "minY",
]

DEFAULT_OLD_TO_NEW_RENAME = {
    "avg_init_energy_change": "avg_e_change_from_init",
    "avg_RSS": "avg_P12",
    "max_plastic_deformation": "max_m3_nr",
    "avg_sigmaxy": "avg_sigma12",
    "avg_init_sigmaxy": "avg_init_sigma12",
    "avg_Pxy": "avg_P12",
}

SIGMAXY_MACRODATA_HEADER = [
    "load_step",
    "load",
    "total_energy",
    "total_energy_change",
    "total_init_energy",
    "total_e_change_from_init",
    "avg_energy",
    "avg_energy_change",
    "avg_init_energy",
    "avg_e_change_from_init",
    "min_iter_total_energy_change",
    "min_iter_avg_energy_change",
    "max_energy",
    "max_force",
    "avg_sigmaxy",
    "avg_init_sigmaxy",
    "avg_sigmaxy_change_from_init",
    "avg_Pxy",
    "nr_plastic_deformations",
    "nr_red_q1",
    "nr_red_q2",
    "nr_red_q3",
    "nr_red_q4",
    "nr_red_q1_fixed",
    "nr_red_q2_fixed",
    "nr_red_q3_fixed",
    "nr_red_q4_fixed",
    "max_m3_nr",
    "max_positive_plastic_jump",
    "max_negative_plastic_jump",
    "nr_iterations",
    "nr_func_evals",
    "LBFGS_Term_reason",
    "CG_Term_reason",
    "FIRE_Term_reason",
    "run_time",
    "minimization_time",
    "write_time",
    "est_time_remaining",
    "cmX",
    "cmY",
    "maxX",
    "minX",
    "maxY",
    "minY",
]


def fix_mixed_macrodata_csv(
    csv_path: str | Path,
    out_path: str | Path | None = None,
    *,
    inplace: bool = True,
    old_header: list[str] | None = None,
    new_header: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
    L: int | None = None,
    nr_elements: int | None = None,
    infer_elements_from_path: bool = True,
    fill_value: str = "0",
) -> Path:
    """
    Fix a macroData.csv where the header changes mid-file.

    The output uses the new header throughout. Rows written with the old header
    are mapped into the new columns, and any missing new columns are filled with
    `fill_value`.
    """
    csv_path = Path(csv_path)
    if out_path is None:
        if inplace:
            out_path = csv_path.with_suffix(".tmp.csv")
        else:
            out_path = csv_path.with_name(f"{csv_path.stem}_fixed{csv_path.suffix}")
    out_path = Path(out_path)

    old_header = list(old_header or OLD_MACRODATA_HEADER)
    new_header = list(new_header or NEW_MACRODATA_HEADER)
    if rename_map is None:
        rename_map = dict(DEFAULT_OLD_TO_NEW_RENAME)
    else:
        merged = dict(DEFAULT_OLD_TO_NEW_RENAME)
        merged.update(rename_map)
        rename_map = merged

    if nr_elements is None:
        if L is None and infer_elements_from_path:
            match = re.search(r"s(\d+)x(\d+)", str(csv_path))
            if match:
                lx = int(match.group(1))
                ly = int(match.group(2))
                nr_elements = 2 * lx * ly
        elif L is not None:
            nr_elements = int(L) * int(L) * 2

    new_index = {name: i for i, name in enumerate(new_header)}

    def _header_key(row: list[str]) -> list[str]:
        return [c.strip().lower().replace(" ", "_") for c in row]

    def _build_mapping(header: list[str]) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for i, col in enumerate(header):
            new_col = rename_map.get(col, col)
            idx = new_index.get(new_col)
            if idx is not None:
                mapping[i] = idx
        return mapping

    default_old_mapping = _build_mapping(old_header)
    default_mid_mapping = _build_mapping(MID_MACRODATA_HEADER)
    default_new_mapping = _build_mapping(new_header)
    default_sigmaxy_mapping = _build_mapping(SIGMAXY_MACRODATA_HEADER)
    header_len_map = {
        len(old_header): default_old_mapping,
        len(MID_MACRODATA_HEADER): default_mid_mapping,
        len(new_header): default_new_mapping,
    }

    known_headers: dict[tuple[str, ...], dict[int, int]] = {}
    for header, mapping in (
        (old_header, default_old_mapping),
        (MID_MACRODATA_HEADER, default_mid_mapping),
        (SIGMAXY_MACRODATA_HEADER, default_sigmaxy_mapping),
        (new_header, default_new_mapping),
    ):
        key = tuple(_header_key(header))
        known_headers[key] = mapping

    def _try_float(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _fill_totals_from_avgs(row_out: list[str]) -> list[str]:
        if nr_elements is None:
            return row_out
        energy_avg_cols = [
            "avg_energy",
            "avg_energy_change",
            "avg_init_energy",
            "avg_e_change_from_init",
        ]
        energy_total_cols = [
            "total_energy",
            "total_energy_change",
            "total_init_energy",
            "total_e_change_from_init",
        ]

        def _maybe_rescale_energy_avgs() -> None:
            if nr_elements is None:
                return 
            idx_avg = new_index.get("avg_energy")
            idx_max = new_index.get("max_energy")
            if idx_avg is None or idx_max is None:
                return
            avg_val = _try_float(row_out[idx_avg])
            max_val = _try_float(row_out[idx_max])
            if avg_val is None or max_val is None or max_val <= 0:
                return
            # If "avg_energy" is orders of magnitude larger than max element energy,
            # it is likely a total value written under the old header.
            if avg_val <= max_val * 10:
                return
            for col in energy_avg_cols:
                idx = new_index.get(col)
                if idx is None or idx >= len(row_out):
                    continue
                v = _try_float(row_out[idx])
                if v is None:
                    continue
                row_out[idx] = f"{v / nr_elements:.15g}"
            # Clear totals so they get recomputed from corrected averages.
            for col in energy_total_cols:
                idx = new_index.get(col)
                if idx is None or idx >= len(row_out):
                    continue
                row_out[idx] = fill_value

        _maybe_rescale_energy_avgs()
        total_from_avg = {
            "avg_energy": "total_energy",
            "avg_energy_change": "total_energy_change",
            "avg_init_energy": "total_init_energy",
            "avg_e_change_from_init": "total_e_change_from_init",
        }
        for avg_col, total_col in total_from_avg.items():
            idx_avg = new_index.get(avg_col)
            idx_total = new_index.get(total_col)
            if idx_avg is None or idx_total is None:
                continue
            if idx_avg >= len(row_out) or idx_total >= len(row_out):
                continue
            avg_val = _try_float(row_out[idx_avg])
            if avg_val is None:
                continue
            total_raw = row_out[idx_total]
            total_str = "" if total_raw is None else str(total_raw).strip()
            if total_str not in ("", str(fill_value)):
                # Already has a total value.
                continue
            if total_str == str(fill_value) and avg_val == 0.0:
                # Preserve explicit zeros when avg is zero.
                continue
            row_out[idx_total] = f"{avg_val * nr_elements:.15g}"
        return row_out

    def _map_row(row: list[str], mapping: dict[int, int]) -> list[str]:
        out = [fill_value] * len(new_header)
        for i, value in enumerate(row):
            idx = mapping.get(i)
            if idx is None:
                continue
            out[idx] = value
        return _fill_totals_from_avgs(out)

    def _coerce_length(row: list[str], length: int) -> list[str]:
        if len(row) < length:
            return row + [fill_value] * (length - len(row))
        if len(row) > length:
            return row[:length]
        return row

    current_mapping: dict[int, int] | None = None
    current_expected_len: int | None = None
    with (
        open(csv_path, "r", newline="") as f_in,
        open(out_path, "w", newline="") as f_out,
    ):
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(new_header)

        for row in reader:
            if not row:
                continue

            row_key = _header_key(row)
            mapping = known_headers.get(tuple(row_key))
            if mapping is not None:
                current_mapping = mapping
                current_expected_len = len(row)
                continue
            if row_key and row_key[0] == "load_step":
                # Skip any stray header-like line.
                continue

            mapping = current_mapping
            row_len = len(row)
            if mapping is None or (
                current_expected_len is not None and row_len != current_expected_len
            ):
                mapping = header_len_map.get(row_len, default_new_mapping)
                current_mapping = mapping
                current_expected_len = row_len

            writer.writerow(_map_row(row, mapping))

    if inplace:
        out_path.replace(csv_path)
        return csv_path
    return out_path


def fix_csv_files(paths):
    def _fix_entry(entry):
        if isinstance(entry, list):
            return [_fix_entry(item) for item in entry]
        if isinstance(entry, tuple):
            return tuple(_fix_entry(item) for item in entry)
        return fix_mixed_macrodata_csv(entry, inplace=False)

    return _fix_entry(paths)
