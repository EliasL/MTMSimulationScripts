import pandas as pd


def update_df_header(
    df: pd.DataFrame,
    add_total_columns: bool = True,
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
        "avg_RSS": "avg_Pxy",
        # Umut headers (Note energy is NOT averaged)
        "Alpha": "load",
        "PreEnergy": "init_energy",
        "PostEnergy": "energy",
        "PreStress": "avg_init_sigmaxy",
        "PostStress": "avg_sigmaxy",
        "EnergyChange": "e_change_from_init",
        "StressChange": "avg_sigma_change_from_init",
    }

    # Rename columns if they exist in the DataFrame
    df = df.rename(columns=rename_map)

    if add_total_columns:
        if nr_elements is None and L is not None:
            nr_elements = int(L) * int(L) * 2
        if nr_elements is not None:
            for col in df.columns:
                if col.startswith("avg_"):
                    total_col = col[4:]
                    if total_col not in df.columns:
                        df[total_col] = df[col] * nr_elements

    return df
