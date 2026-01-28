import pandas as pd


def update_df_header(df: pd.DataFrame):
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
        "PreEnergy": "avg_init_energy",
        "PostEnergy": "avg_energy",
        "PreStress": "avg_init_sigmaxy",
        "PostStress": "avg_sigmaxy",
        "EnergyChange": "avg_e_change_from_init",
        "StressChange": "avg_sigma_change_from_init",
    }

    # Rename columns if they exist in the DataFrame
    df = df.rename(columns=rename_map)

    return df
