import os
import numpy as np
import pandas as pd

from methods.load.results import load_results
from methods.config import (
    NFE, SEED, ISLANDS, OUTPUT_DIR, FIG_DIR, PROCESSED_DATA_DIR,
    reservoir_options, policy_type_options,
    reservoir_min_release, reservoir_max_release, reservoir_capacity
)
from methods.load.observations import get_observational_training_data
from methods.reservoir.model import Reservoir

# === Ensure output directory exists
os.makedirs(FIG_DIR, exist_ok=True)

# === Config
RESERVOIR_NAMES = reservoir_options
POLICY_TYPES = policy_type_options
SELECTION_LABELS = [
    "Best Release NSE", "Best Storage NSE", "Best Average NSE", "Best All Average NSE"
]
obj_labels = {
    "obj1": "Release NSE",
    "obj2": "Release q20 Abs % Bias",
    "obj3": "Release q80 Abs % Bias",
    "obj4": "Storage NSE",
}
obj_cols = list(obj_labels.values())

# === Load all best solutions
best_policy_params = {}
for reservoir_name in RESERVOIR_NAMES:
    best_policy_params[reservoir_name] = {}
    for policy_type in POLICY_TYPES:
        fname = f"{OUTPUT_DIR}/MMBorg_{ISLANDS}M_{policy_type}_{reservoir_name}_nfe{NFE}_seed{SEED}.csv"
        obj_df, var_df = load_results(fname, obj_labels=obj_labels)
        obj_df = obj_df[obj_df["Storage NSE"] <= 10]
        var_df = var_df.loc[obj_df.index]

        if len(obj_df) == 0:
            print(f"[!] Skipped: No valid results for {reservoir_name} | {policy_type}")
            continue

        best_policy_params[reservoir_name][policy_type] = {
            "Best Release NSE": {
                "idx": obj_df["Release NSE"].idxmax(),
                "vars": var_df.loc[obj_df["Release NSE"].idxmax()].values
            },
            "Best Storage NSE": {
                "idx": obj_df["Storage NSE"].idxmax(),
                "vars": var_df.loc[obj_df["Storage NSE"].idxmax()].values
            },
            "Best Average NSE": {
                "idx": obj_df[["Release NSE", "Storage NSE"]].mean(axis=1).idxmax(),
                "vars": var_df.loc[obj_df[["Release NSE", "Storage NSE"]].mean(axis=1).idxmax()].values
            },
            "Best All Average NSE": {
                "idx": obj_df[obj_cols].mean(axis=1).idxmax(),
                "vars": var_df.loc[obj_df[obj_cols].mean(axis=1).idxmax()].values
            }
        }

# === Simulate and Save Plots
for reservoir_name in RESERVOIR_NAMES:
    inflow_obs, release_obs, storage_obs = get_observational_training_data(
        reservoir_name=reservoir_name,
        data_dir=PROCESSED_DATA_DIR,
        as_numpy=False,
        scaled_inflows=True
    )
    datetime_index = inflow_obs.index
    inflow_array = inflow_obs.values.flatten()
    release_obs_array = release_obs.values.flatten()
    storage_obs_array = storage_obs.values.flatten()

    for policy_type in POLICY_TYPES:
        for selection_label in SELECTION_LABELS:
            try:
                params = best_policy_params[reservoir_name][policy_type][selection_label]["vars"]
                reservoir = Reservoir(
                    inflow=inflow_array,
                    dates=datetime_index,
                    capacity=reservoir_capacity[reservoir_name],
                    policy_type=policy_type,
                    policy_params=params,
                    release_min=reservoir_min_release[reservoir_name],
                    release_max=reservoir_max_release[reservoir_name],
                    name=reservoir_name
                )
                reservoir.run()

                # === Save figure
                fname = f"{FIG_DIR}/dynamics_{reservoir_name}_{policy_type}_{selection_label.replace(' ', '_')}_nfe{NFE}_seed{SEED}.png"
                reservoir.plot(
                    save=True,
                    fname=fname,
                    storage_obs=storage_obs_array,
                    release_obs=release_obs_array,
                    release_log_scale=False
                )
                print(f"[✓] Saved: {fname}")

            except Exception as e:
                print(f"[!] Failed: {reservoir_name} | {policy_type} | {selection_label} — {e}")