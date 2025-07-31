import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods.reservoir.model import Reservoir
from methods.load.observations import get_observational_training_data
from methods.metrics.objectives import ObjectiveCalculator
from methods.metrics.objectives import RELEASE_METRICS, STORAGE_METRICS

from methods.utils import get_overlapping_datetime_indices


from methods.config import reservoir_min_release, reservoir_max_release, reservoir_capacity
from methods.config import SEED, METRICS, EPSILONS, NFE
from methods.config import policy_n_params, policy_param_bounds
from methods.config import SEED, cfs_to_mgd

from methods.sampling import generate_policy_param_samples

from methods.config import DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR
import os

import sys

# # Create a logs directory if it doesn't exist
# LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
# os.makedirs(LOG_DIR, exist_ok=True)

# # Redirect all stdout to a log file
# log_file_path = os.path.join(LOG_DIR, "reservoir_simulation_log.txt")
# sys.stdout = open(log_file_path, "w")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

from methods.config import (
    reservoir_min_release, 
    reservoir_max_release, 
    reservoir_capacity, 
    policy_n_params, 
    policy_param_bounds,
    PROCESSED_DATA_DIR
)
# np.random.seed(SEED)

# Policies to test
test_policies = [
    # "RBF",
    #"PiecewiseLinear",
    "STARFIT"
]

# Reservoirs to test
test_reservoirs = [
     "fewalter",
    'beltzvilleCombined',
    "prompton"
]


# # Random draw of parameters from LHS
# n_samples =  1
# test_params = {
#     p : generate_policy_param_samples(p, n_samples, 
#                                       sample_type='latin')[0] for p in test_policies
# }
# Sets of test parameters
test_params = np.array([
    15.08, 5.0, 20.0,         # NORhi mu, min, max
    0.0, -15.0,               # NORhi alpha, beta
    9.0, 1.6, 14.2,           # NORlo mu, min, max
    -1.0, -30.0,              # NORlo alpha, beta
    0.2118, -0.0357, 0.1302, -0.0248,  # Harmonics
    -0.123, 0.183, 0.732     # Adjustments
])

baseline_starfit_params = {
    "fewalter": np.array([
        15.08, 5.0, 20.0,
        0.0, -15.0,
        9.0, 1.6, 14.2,
        -1.0, -30.0,
        0.2118, -0.0357, 0.1302, -0.0248,
        -0.123, 0.183, 0.732
    ]),
    "beltzvilleCombined": np.array([
        100.0, -99999.0, np.inf,
        0.0, 0.0,
        90.0, 85.0, 91.0,
        -0.56, 2.28,
        0.2824, -0.0585, 0.1903, 0.005,
        -0.338, 17.016, 0.788
    ]),
    "prompton": np.array([
        15.08, -99999.0, np.inf,
        -0.23, -2.01,
        12.58, 12.31, 15.52,
        0.2, -4.08,
        0.2118, -0.0357, 0.1302, -0.0248,
        -0.123, 0.183, 0.732
    ])
}

line_break = "##############################################"


if __name__ == "__main__":

    # setup objective function
    METRICS = RELEASE_METRICS + STORAGE_METRICS

    # Setup objective function
    release_obj_func = ObjectiveCalculator(metrics=RELEASE_METRICS)
    storage_obj_func = ObjectiveCalculator(metrics=STORAGE_METRICS)

    for RESERVOIR_NAME in test_reservoirs:

        # Load observed data
        inflow_obs, release_obs, storage_obs = get_observational_training_data(
        reservoir_name=RESERVOIR_NAME,
        data_dir=PROCESSED_DATA_DIR,
        as_numpy=False,
        scaled_inflows=True
        )
        datetime_index = inflow_obs.index
        inflow_obs = inflow_obs.values.flatten()
        release_obs = release_obs.values.flatten()
        storage_obs = storage_obs.values.flatten()
        
        for POLICY_TYPE in test_policies:

            # === Build reservoir ===
            reservoir = Reservoir(
                inflow=inflow_obs,
                dates=datetime_index,
                capacity=reservoir_capacity[RESERVOIR_NAME],
                policy_type=POLICY_TYPE,
                policy_params=baseline_starfit_params[RESERVOIR_NAME],
                release_min=reservoir_min_release[RESERVOIR_NAME],
                release_max=reservoir_max_release[RESERVOIR_NAME],
                name=RESERVOIR_NAME
            )
            # === Constraint Check ===
            if not reservoir.policy.test_nor_constraint():
                print("[!] STARFIT NOR constraint violated.")
            else:
                print("[✓] STARFIT NOR constraint passed.")
            # Run 
            reservoir.run()

    

            # Retrieve simulated release data
            sim_release = reservoir.release_array.astype(np.float64)
            sim_release += reservoir.spill_array.astype(np.float64) # add spill to release
            sim_storage = reservoir.storage_array.astype(np.float64)
    
            # Check that simulation results are real numbered
            if np.isnan(sim_release).any() or np.isnan(sim_storage).any():
                print(f"Simulation generated NaN for {RESERVOIR_NAME}, {POLICY_TYPE} with parameters {vars}.")
                # return [9999.99] * NOBJS, [1.0]

            # Calculate the objectives
            release_objs = release_obj_func.calculate(obs=release_obs, sim=sim_release)
            storage_objs = storage_obj_func.calculate(obs=storage_obs, sim=sim_storage)

            # === Reporting ===

            print(f"Reservoir: {RESERVOIR_NAME} | Policy: {POLICY_TYPE}")

            print(f"\n### Release Objectives: ###")
            for metric_name, objective_value in zip(METRICS, release_objs):
                print(f"{metric_name}: {objective_value:.4f}")

            print(f"\n### Storage Objectives: ###")
            for metric_name, objective_value in zip(METRICS, storage_objs):
                print(f"{metric_name}: {objective_value:.4f}")
            print(line_break)


            # Plot policy function
            #reservoir.policy.plot()

            # Plot STARFIT 3D policy surface
            if POLICY_TYPE == "STARFIT":
                reservoir.policy.plot_policy_surface(save=True, fname=f"{RESERVOIR_NAME}_policy_surface.png")
                reservoir.policy.plot_nor(fname=f"{RESERVOIR_NAME}_nor.png")
            
            # Plot sim and obs dynamics (storage and release)
            reservoir.plot(
                save=True,
                fname=f"{RESERVOIR_NAME}_dynamics.png",
                storage_obs=storage_obs,
                release_obs=release_obs,
                release_log_scale=False
            )

