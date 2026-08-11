import csv
import re
from pathlib import Path

import pandas as pd
import numpy as np

HEADER_RENAME_MAP = {
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
    "avg_sigmaxy_change_from_init":"avg_sigma12_change_from_init",
    # Reversibility column rename
    "rev_d": "rev_u_diff",
    # More changes
    "nr_plastic_deformations":"nr_elements_with_m3_fix_change",
}

def update_df_header(
    df: pd.DataFrame,
    add_total_columns: bool = True,
    L: int | None = None,
    nr_elements: int | None = None,
):
    # Mapping of old column names to new column names
    rename_map = dict(HEADER_RENAME_MAP)

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

    return df


def get_fixed_csv_path(path: str | Path) -> Path:
    path = Path(path)
    if path.stem.endswith("_fixed"):
        return path
    return path.with_name(f"{path.stem}_fixed{path.suffix}")




def read_macrodata_csv(
    csv_path,
    *,
    fix_mixed=True,
    update_header=True,
    warn_on_dtype=True,
    **update_kwargs,
):
    import warnings
    from pandas.errors import DtypeWarning

    csv_path = Path(csv_path)
    if fix_mixed and not csv_path.stem.endswith("_fixed"):
        fixed_path = get_fixed_csv_path(csv_path)
        try:
            if fixed_path.exists():
                if not csv_path.exists():
                    csv_path = fixed_path
                else:
                    fixed_mtime = fixed_path.stat().st_mtime
                    src_mtime = csv_path.stat().st_mtime
                    if fixed_mtime >= src_mtime:
                        csv_path = fixed_path
        except OSError:
            pass

    def _read():
        with warnings.catch_warnings(record=True) as warn_list:
            if warn_on_dtype:
                warnings.simplefilter("always", DtypeWarning)
            df_local = pd.read_csv(csv_path)
        if not isinstance(df_local.index, pd.RangeIndex):
            raise pd.errors.ParserError(
                f"Unexpected {type(df_local.index).__name__} while reading {csv_path}. "
                "The CSV contains rows wider than its header."
            )
        has_dtype_warning = any(
            issubclass(w.category, DtypeWarning) for w in warn_list
        )
        return df_local, has_dtype_warning

    effective_path = csv_path
    try:
        df, dtype_warn = _read()
    except Exception:
        if not fix_mixed:
            raise
        effective_path = fix_mixed_macrodata_csv(csv_path, inplace=False)
        csv_path = effective_path
        df, dtype_warn = _read()

    if dtype_warn and warn_on_dtype:
        print(f"Mixed dtypes detected in {csv_path}")
        if fix_mixed:
            effective_path = fix_mixed_macrodata_csv(csv_path, inplace=False)
            csv_path = effective_path
            df, dtype_warn = _read()
            if dtype_warn:
                print(f"Mixed dtypes persist after fix in {csv_path}")
    if update_header:
        df = update_df_header(df, **update_kwargs)
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
    "avg_sigma11",
    "avg_sigma12",
    "avg_sigma22",
    "avg_init_sigma11",
    "avg_init_sigma12",
    "avg_init_sigma22",
    "avg_sigma12_change_from_init",
    "avg_P11",
    "avg_P12",
    "avg_P21",
    "avg_P22",
    "avg_init_P11",
    "avg_init_P12",
    "avg_init_P21",
    "avg_init_P22",
    "participationFraction",
    "m3_participationFraction",
    "nr_elements_with_m3_fix_change",
    "nr_red_q1",
    "nr_red_q2",
    "nr_red_q3",
    "nr_red_q4",
    "nr_red_q1_fixed",
    "nr_red_q2_fixed",
    "nr_red_q3_fixed",
    "nr_red_q4_fixed",
    "max_m3_nr",
    "sum_m3",
    "max_positive_plastic_jump",
    "max_negative_plastic_jump",
    "nr_iterations",
    "nr_func_evals",
    "nr_edge_flips",
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
    "is_reversible",
    "rev_u_diff",
    "rev_energy_diff",
    "rev_sigma_12_diff",
    "rev_sigma_trace_diff",
    "rev_sigma11_diff",
    "rev_sigma22_diff",
    "rev_p11_diff",
    "rev_p12_diff",
    "rev_p21_diff",
    "rev_p22_diff",
]

RECONNECT_REVERSIBILITY_MACRODATA_HEADER = (
    NEW_MACRODATA_HEADER[:31]
    + ["nr_elements_with_m3_change"]
    + NEW_MACRODATA_HEADER[32:36]
    + NEW_MACRODATA_HEADER[40:47]
    + [
        "nr_total_edge_flips",
        "nr_reconnect_cycles",
        "reconnect_stop_reason",
        "rejected_reconnect_energy_delta",
        "edge_flip_chosen_minus_other_energy",
        "edge_flip_always_chose_lower_energy",
    ]
    + NEW_MACRODATA_HEADER[47:]
)

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

DEFAULT_OLD_TO_NEW_RENAME = dict(HEADER_RENAME_MAP)

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
    "avg_sigma11",
    "avg_sigmaxy",
    "avg_sigma22",
    "avg_init_sigma11",
    "avg_init_sigmaxy",
    "avg_init_sigma22",
    "avg_sigmaxy_change_from_init",
    "avg_P11",
    "avg_Pxy",
    "avg_P21",
    "avg_P22",
    "avg_init_P11",
    "avg_init_P12",
    "avg_init_P21",
    "avg_init_P22",
    "participationFraction",
    "m3_participationFraction",
    "nr_elements_with_m3_fix_change",
    "nr_red_q1",
    "nr_red_q2",
    "nr_red_q3",
    "nr_red_q4",
    "nr_red_q1_fixed",
    "nr_red_q2_fixed",
    "nr_red_q3_fixed",
    "nr_red_q4_fixed",
    "max_m3_nr",
    "sum_m3",
    "max_positive_plastic_jump",
    "max_negative_plastic_jump",
    "nr_iterations",
    "nr_func_evals",
    "nr_edge_flips",
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
    "is_reversible",
    "rev_u_diff",
    "rev_energy_diff",
    "rev_sigma_12_diff",
    "rev_sigma_trace_diff",
    "rev_sigma11_diff",
    "rev_sigma22_diff",
    "rev_p11_diff",
    "rev_p12_diff",
    "rev_p21_diff",
    "rev_p22_diff",
]


def fix_mixed_macrodata_csv(
    csv_path: str | Path,
    out_path: str | Path | None = None,
    *,
    inplace: bool = False,
    old_header: list[str] | None = None,
    new_header: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
    L: int | None = None,
    nr_elements: int | None = None,
    infer_elements_from_path: bool = True,
    fill_value: str = "0",
    warn_on_drop: bool = True,
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
            out_path = get_fixed_csv_path(csv_path)
    out_path = Path(out_path)

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

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        df = None

    if df is not None and isinstance(df.index, pd.RangeIndex):
        df = update_df_header(
            df,
            add_total_columns=False,
            
            nr_elements=nr_elements,
        )
        df.to_csv(out_path, index=False)
        print(f"Fixed CSV written to {out_path}")
        if inplace:
            out_path.replace(csv_path)
            return csv_path
        return out_path

    header_line_idx = None
    header_line = None
    first_header = None
    rows_before: list[list[str]] = []
    rows_after: list[list[str]] = []
    pending_bad_row: tuple[int, int, int, list[str]] | None = None
    schema_was_inferred = False

    def _normalize_header(header_row: list[str]) -> list[str]:
        return [h.strip() for h in header_row if h.strip()]

    def _parse_header_token(token_row: list[str]) -> list[str]:
        header_first = token_row[0].split(":", 1)[1]
        return _normalize_header([header_first] + token_row[1:])

    def _append_row(rows: list[list[str]], row: list[str], expected_len: int, line_no: int) -> None:
        if len(row) != expected_len:
            raise ValueError(
                f"Row length mismatch in {csv_path} at line {line_no}: "
                f"expected {expected_len}, got {len(row)}."
            )
        rows.append(row)

    def _infer_header(row_length: int) -> list[str] | None:
        candidates = [
            header
            for header in (
                new_header,
                RECONNECT_REVERSIBILITY_MACRODATA_HEADER,
            )
            if header is not None and len(header) == row_length
        ]
        if len(candidates) != 1:
            return None
        return list(candidates[0])

    def _validate_header_transition(source_header, target_header):
        source = [rename_map.get(name, name) for name in source_header]
        target = [rename_map.get(name, name) for name in target_header]
        if len(target) != len(set(target)):
            raise ValueError(f"Duplicate columns in inferred schema for {csv_path}.")
        if source != target[: len(source)]:
            raise ValueError(
                f"Inferred schema for {csv_path} does not preserve the original "
                "column order. Refusing to relabel the data."
            )

    def _validate_repaired_data(df):
        if df.empty:
            return

        duration_columns = {
            "run_time",
            "minimization_time",
            "write_time",
            "est_time_remaining",
        }
        text_columns = {
            column
            for column in df.columns
            if column.endswith("_Term_reason") or column == "reconnect_stop_reason"
        }
        numeric_columns = set(df.columns) - duration_columns - text_columns
        numeric_values = {}
        for column in numeric_columns:
            values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            if np.any(~np.isfinite(values)):
                raise ValueError(
                    f"Non-numeric or non-finite values found in numeric column "
                    f"{column!r} while repairing {csv_path}."
                )
            numeric_values[column] = values

        load_step = numeric_values["load_step"]
        if not np.allclose(load_step, np.round(load_step)):
            raise ValueError(f"Non-integer load_step values found while repairing {csv_path}.")
        if len(load_step) > 1 and np.any(np.diff(load_step) <= 0):
            raise ValueError(f"load_step is not strictly increasing in {csv_path}.")

        for column in (column for column in df.columns if column.startswith("nr_")):
            values = numeric_values[column]
            if not np.allclose(values, np.round(values)) or np.any(values < 0):
                raise ValueError(
                    f"Invalid count values found in column {column!r} while "
                    f"repairing {csv_path}."
                )

        for column in ("is_reversible",):
            if column in numeric_values and not np.all(np.isin(numeric_values[column], [0, 1])):
                raise ValueError(
                    f"Expected only 0/1 values in {column!r} while repairing {csv_path}."
                )

        for column in ("participationFraction", "m3_participationFraction"):
            if column in numeric_values:
                values = numeric_values[column]
                if np.any((values < 0) | (values > 1)):
                    raise ValueError(
                        f"Values outside [0, 1] found in {column!r} while repairing {csv_path}."
                    )

    with open(csv_path, "r", newline="") as f_in:
        reader = csv.reader(f_in)
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue
            token = row[0].strip()
            if token.lower().startswith("#header:"):
                header_line_idx = line_no
                header_line = _parse_header_token(row)
                continue
            if first_header is None:
                first_header = _normalize_header(row)
                continue
            if header_line is None:
                if pending_bad_row is not None:
                    prev_line, expected_len, got_len, prev_row = pending_bad_row
                    inferred_header = _infer_header(got_len)
                    if inferred_header is not None and len(row) == got_len:
                        header_line = inferred_header
                        schema_was_inferred = True
                        rows_after.extend((prev_row, row))
                        pending_bad_row = None
                        print(
                            f"Detected unmarked header change in {csv_path} at line "
                            f"{prev_line}: using the {got_len}-column schema."
                        )
                        continue
                    raise ValueError(
                        f"Row length mismatch in {csv_path} at line {prev_line}: "
                        f"expected {expected_len}, got {got_len}. "
                        f"Encountered additional data at line {line_no} without a #HEADER line. "
                        "Only a corrupted final row is allowed in this mode."
                    )
                if len(row) != len(first_header):
                    pending_bad_row = (line_no, len(first_header), len(row), row)
                    continue
                rows_before.append(row)
            else:
                _append_row(rows_after, row, len(header_line), line_no)

    if header_line is not None and pending_bad_row is not None:
        bad_line, expected_len, got_len, _ = pending_bad_row
        raise ValueError(
            f"Row length mismatch in {csv_path} at line {bad_line}: "
            f"expected {expected_len}, got {got_len}. "
            "A #HEADER line was found later, so this mismatch is unexpected."
        )

    if header_line is None and pending_bad_row is not None:
        _, _, got_len, bad_row = pending_bad_row
        inferred_header = _infer_header(got_len)
        if inferred_header is not None:
            header_line = inferred_header
            schema_was_inferred = True
            rows_after.append(bad_row)
            pending_bad_row = None

    if header_line is None:
        if pending_bad_row is None:
            raise ValueError(
                f"Parser error while reading {csv_path}, but no #HEADER line found. "
                "No single corrupted final row was detected."
            )
        bad_line, expected_len, got_len, _ = pending_bad_row
        if first_header is None:
            raise ValueError(
                f"Parser error while reading {csv_path}, but no header line found."
            )
        print(
            f"Warning: dropping corrupted final row in {csv_path} "
            f"(line {bad_line}, expected {expected_len}, got {got_len})."
        )
        df = pd.DataFrame(rows_before, columns=first_header)
        df = update_df_header(
            df,
            add_total_columns=False,
            
            nr_elements=nr_elements,
        )
        df.to_csv(out_path, index=False)
        print(f"Fixed CSV written to {out_path}")
        if inplace:
            out_path.replace(csv_path)
            return csv_path
        return out_path

    new_header = new_header or header_line
    source_header = old_header or first_header or []
    if schema_was_inferred or rows_before:
        _validate_header_transition(source_header, new_header)
    old_header = new_header if not rows_before else source_header

    def _rename_headers(headers: list[str]) -> list[str]:
        return [rename_map.get(col, col) for col in headers]

    old_header_renamed = _rename_headers(old_header)
    new_header_renamed = _rename_headers(new_header)
    missing = [col for col in old_header_renamed if col not in new_header_renamed]
    if missing:
        missing_str = ", ".join(missing)
        print(f"Error: missing mapped columns from new header in {csv_path}: {missing_str}")
        raise ValueError(
            f"Unable to map old header into new header for {csv_path}. "
            f"Missing columns: {missing_str}"
        )

    df_first = pd.DataFrame(rows_before, columns=old_header)
    df_second = pd.DataFrame(rows_after, columns=new_header)

    df_first = update_df_header(
        df_first,
        add_total_columns=False,
        
        nr_elements=nr_elements,
    )
    df_second = update_df_header(
        df_second,
        add_total_columns=False,
        
        nr_elements=nr_elements,
    )

    for col in df_second.columns:
        if col not in df_first.columns:
            df_first[col] = fill_value

    extra_cols = [col for col in df_first.columns if col not in df_second.columns]
    if extra_cols:
        extra_str = ", ".join(extra_cols)
        print(
            f"Error: unmatched columns from early header in {csv_path}: {extra_str}"
        )
        raise ValueError(
            f"Unable to merge headers for {csv_path}. "
            f"Unmatched columns: {extra_str}"
        )

    df_first = df_first[df_second.columns]
    df_out = pd.concat([df_first, df_second], ignore_index=True)
    if not df_out.empty:
        def _is_numeric(val) -> bool:
            if val is None:
                return True
            s = str(val).strip()
            if s == "" or s == fill_value:
                return True
            try:
                float(s)
                return True
            except ValueError:
                return False

        last_row = df_out.iloc[-1]
        if "load_step" in df_out and not _is_numeric(last_row["load_step"]):
            print(f"Warning: dropping corrupted final row in {csv_path}")
            df_out = df_out.iloc[:-1]
    _validate_repaired_data(df_out)
    df_out.to_csv(out_path, index=False)
    print(f"Fixed CSV written to {out_path}")

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
        path = Path(entry)
        if path.stem.endswith("_fixed"):
            return path
        fixed_path = get_fixed_csv_path(path)
        try:
            if fixed_path.exists():
                if not path.exists():
                    return fixed_path
                if fixed_path.stat().st_mtime >= path.stat().st_mtime:
                    return fixed_path
        except OSError:
            pass
        return fix_mixed_macrodata_csv(path, inplace=False)

    return _fix_entry(paths)
