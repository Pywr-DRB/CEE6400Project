import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # ← Missing in your original code!

from methods.load.results import load_results
from methods.config import (
    NFE, SEED, ISLANDS, OUTPUT_DIR, FIG_DIR, PROCESSED_DATA_DIR,
    reservoir_options, policy_type_options,
    reservoir_min_release, reservoir_max_release, reservoir_capacity
)
from methods.load.observations import get_observational_training_data
from methods.reservoir.model import Reservoir
from methods.config import inflow_bounds_by_reservoir

# === Ensure output directory exists
os.makedirs(FIG_DIR, exist_ok=True)

# NEW: where to save simulated time series
TS_DIR = os.path.join(OUTPUT_DIR, "sim_timeseries")
os.makedirs(TS_DIR, exist_ok=True)

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
NUM_OBJECTIVES = len(obj_cols)
results_dict = {}

def read_borg_ref_matrix(path):
    """
    Read whitespace-delimited borg.ref into a 2D float array.
    Assumes no header, decisions first, then the 4 objectives at the end.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing borg.ref at: {path}")
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rows.append([float(x) for x in s.split()])
    if not rows:
        raise ValueError(f"No parseable rows in {path}")
    return np.array(rows, dtype=float)

def build_obj_var_from_ref_PARSE_ONLY(reservoir_name, policy_type):
    """
    Parse `moeaframework/outputs/Policy_{policy_type}/runtime/{reservoir_name}/borg.ref`
    into var_df (decisions) and obj_df (objectives), assuming the last 4 columns are:
      [Release NSE, Release q20 Abs % Bias, Release q80 Abs % Bias, Storage NSE]
    """
    ref_path = os.path.join(
        "moeaframework", "outputs", f"Policy_{policy_type}", "runtime",
        reservoir_name, "borg.ref"
    )
    mat = read_borg_ref_matrix(ref_path)

    if mat.shape[1] <= NUM_OBJECTIVES:
        raise ValueError(
            f"borg.ref seems to have ≤{NUM_OBJECTIVES} columns; need decisions + {NUM_OBJECTIVES} objectives."
        )

    dec_len = mat.shape[1] - NUM_OBJECTIVES
    dec = mat[:, :dec_len]
    objs = mat[:, dec_len:]

    var_df = pd.DataFrame(dec, columns=[f"x{i+1}" for i in range(dec.shape[1])])
    obj_df = pd.DataFrame(objs, columns=obj_cols)

    # If your borg.ref stores “minimize” versions (e.g., -NSE, +bias), but your CSV used “maximize”/“higher is better”,
    # flip here. Since you said “already where they need to be”, we leave as-is.
    # Example if needed:
    # obj_df["Release NSE"] *= -1
    # obj_df["Storage NSE"] *= -1
    # obj_df["Release q20 Abs % Bias"] *= -1
    # obj_df["Release q80 Abs % Bias"] *= -1

    # Clean up any inf/nan weirdness
    obj_df = obj_df.replace([np.inf, -np.inf], np.nan).dropna()
    var_df = var_df.loc[obj_df.index]

    return obj_df, var_df

# === Load all best solutions and run simulations
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

    results_dict[reservoir_name] = {}

    for policy_type in POLICY_TYPES:
        obj_df, var_df = build_obj_var_from_ref_PARSE_ONLY(reservoir_name, policy_type)
        obj_df = obj_df[obj_df["Storage NSE"] <= 10]
        var_df = var_df.loc[obj_df.index]

        if len(obj_df) == 0:
            print(f"[!] Skipped: No valid results for {reservoir_name} | {policy_type}")
            continue

        best_params = {
            "Best Release NSE": obj_df["Release NSE"].idxmax(),
            "Best Storage NSE": obj_df["Storage NSE"].idxmax(),
            "Best Average NSE": obj_df[["Release NSE", "Storage NSE"]].mean(axis=1).idxmax(),
            "Best All Average NSE": obj_df[obj_cols].mean(axis=1).idxmax()
        }

        results_dict[reservoir_name][policy_type] = {}

        for label, idx in best_params.items():
            try:
                params = var_df.loc[idx].values
                reservoir = Reservoir(
                    inflow=inflow_array,
                    dates=datetime_index,
                    capacity=reservoir_capacity[reservoir_name],
                    policy_type=policy_type,
                    policy_params=params,
                    release_min=reservoir_min_release[reservoir_name],
                    release_max=reservoir_max_release[reservoir_name],
                    name=reservoir_name,
                    inflow_min=inflow_bounds_by_reservoir[reservoir_name]["I_min"],
                    inflow_max=inflow_bounds_by_reservoir[reservoir_name]["I_max"]
                )
                reservoir.run()

                # Optional: also grab storage if you want to keep it
                release_sim = np.array(reservoir.release_array).flatten()
                storage_sim = np.array(getattr(reservoir, "storage_array", np.full_like(release_sim, np.nan))).flatten()

                # Save a clean CSV of the time series
                safe_label = (
                    label.replace(" ", "_")
                        .replace("/", "-")
                        .replace(":", "-")
                )
                ts_path = os.path.join(
                    TS_DIR,
                    f"ts_{reservoir_name}_{policy_type}_{safe_label}.csv"
                )

                ts_df = pd.DataFrame({
                    "date": pd.to_datetime(datetime_index),         # ISO-8601 when saved
                    "release_obs_mgd": release_obs_array,
                    "release_sim_mgd": release_sim,
                    # Uncomment if you want these as well:
                    # "storage_obs_mg": storage_obs_array,
                    # "storage_sim_mg": storage_sim,
                })
                # Keep only finite rows (avoids NaNs breaking downstream tools)
                mask = np.isfinite(ts_df["release_obs_mgd"]) | np.isfinite(ts_df["release_sim_mgd"])
                ts_df.loc[mask].to_csv(ts_path, index=False)
                print(f"[✓] Saved releases: {ts_path}")

                results_dict[reservoir_name][policy_type][label] = {
                    "dates": datetime_index,
                    "release_obs": release_obs_array,
                    "release_sim": np.array(reservoir.release_array).flatten()
                }
            except Exception as e:
                print(f"[!] Failed: {reservoir_name} | {policy_type} | {label} — {e}")


# === Plotting Function (run after simulation)
def plot_all_reservoirs_3panel(results_dict):
    for reservoir_name, policy_dict in results_dict.items():
        for policy_type, selections in policy_dict.items():
            for selection_label, data in selections.items():
                dates = data["dates"]
                release_obs = data["release_obs"]
                release_sim = data["release_sim"]

                fig, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

                # Panel 1: Time series
                axs[0].plot(dates, release_obs, label="Observed", color="black", linewidth=2)
                axs[0].plot(dates, release_sim, label="Simulated", color="blue", alpha=0.7)
                axs[0].set_ylabel("Release [mgd]")
                axs[0].set_title("Time Series")
                axs[0].legend()

                # Panel 2: Scatter plot
                axs[1].scatter(release_obs, release_sim, alpha=0.6, color="purple")
                axs[1].plot([min(release_obs), max(release_obs)],
                            [min(release_obs), max(release_obs)],
                            color="gray", linestyle="--")
                axs[1].set_xlabel("Observed Release")
                axs[1].set_ylabel("Simulated Release")
                axs[1].set_title("Observed vs Simulated")

                # Panel 3: Flow duration curve
                sorted_obs = np.sort(release_obs)[::-1]
                sorted_sim = np.sort(release_sim)[::-1]
                exceedance = np.linspace(0, 100, len(sorted_obs))

                axs[2].plot(exceedance, sorted_obs, label="Observed", color="black", linewidth=2)
                axs[2].plot(exceedance, sorted_sim, label="Simulated", color="blue", alpha=0.7)
                axs[2].set_xlabel("Exceedance Probability [%]")
                axs[2].set_ylabel("Release [mgd]")
                axs[2].set_title("Flow Duration Curve")
                axs[2].legend()
                axs[2].grid(True, which="both", ls="--", alpha=0.5)

                fig.suptitle(f"{reservoir_name} — {policy_type} — {selection_label}", fontsize=14)
                fig_path = f"{FIG_DIR}/panelbest3_{reservoir_name}_{policy_type}_{selection_label.replace(' ', '_')}.png"
                plt.savefig(fig_path, dpi=300)
                plt.close()
                print(f"[✓] Saved: {fig_path}")


# === Call the plotting function
plot_all_reservoirs_3panel(results_dict)
