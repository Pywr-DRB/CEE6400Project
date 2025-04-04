import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods.reservoir.model import Reservoir
from methods.load.observations import load_observations
from methods.metrics.objectives import ObjectiveCalculator

from methods.utils import get_overlapping_datetime_indices


from methods.config import reservoir_min_release, reservoir_max_release, reservoir_capacity
from methods.config import policy_n_params, policy_param_bounds
from methods.config import SEED, cfs_to_mgd, METRICS, OUTPUT_DIR, FIG_DIR


# Policies to test
test_policies = [
    # "RBF",
    "PiecewiseLinear"
]

# Reservoirs to test
test_reservoirs = [
    "fewalter",
    # 'beltzvilleCombined',
]

policy_type = 'PiecewiseLinear'
reservoir_name = 'fewalter'
NFE = 30000


# load parameters from Borg output
fname = f"{OUTPUT_DIR}/MMBorg_3M_{policy_type}_{reservoir_name}_nfe{NFE}_seed71.csv"
solutions = pd.read_csv(fname)

var_cols = [f'var{i+1}' for i in range(10)]
obj_cols = [f'obj{i+1}' for i in range(4)]
params = solutions.loc[:, var_cols].values
objs = solutions.loc[:, obj_cols].values


line_break = "##############################################"


if __name__ == "__main__":
    metrics = METRICS
    obj_func = ObjectiveCalculator(metrics=METRICS)


    for RESERVOIR_NAME in test_reservoirs:

        # Load observed data
        inflow_obs = load_observations(datatype='inflow', 
                                   reservoir_name=RESERVOIR_NAME, 
                                   data_dir="./data/", as_numpy=False)

        release_obs = load_observations(datatype='release', 
                                        reservoir_name=RESERVOIR_NAME, 
                                        data_dir="./data/", as_numpy=False)

        storage_obs = load_observations(datatype='storage',
                                        reservoir_name=RESERVOIR_NAME, 
                                        data_dir="./data/", as_numpy=False)
        
        # get only overlapping data range
        dt = get_overlapping_datetime_indices(inflow_obs, release_obs, storage_obs)
        
        # subset data
        inflow_obs = inflow_obs.loc[dt,:].values
        release_obs = release_obs.loc[dt,:].values
        storage_obs = storage_obs.loc[dt,:].values
        
    
        # scale inflow to have mass balance with release
        scale_factor = np.sum(release_obs) / np.sum(inflow_obs)
        inflow_scaled = inflow_obs * scale_factor
        print(f'Reservoir: {RESERVOIR_NAME} | inflow scaled by {scale_factor}')

        
        for POLICY_TYPE in test_policies:
            
            # Define the reservoir model
            reservoir = Reservoir(
                inflow = inflow_scaled,
                capacity = reservoir_capacity[RESERVOIR_NAME],
                policy_type = POLICY_TYPE,
                policy_params = params[0],
                release_min = reservoir_min_release[RESERVOIR_NAME],
                release_max =  reservoir_max_release[RESERVOIR_NAME],
                initial_storage = storage_obs[0],
                name = RESERVOIR_NAME,
            )
            
            # Run 
            reservoir.run()

            # Get simulated results
            release_sim = reservoir.release_array
            storage_sim = reservoir.storage_array
            
            # Calculate the objectives
            release_objs = obj_func.calculate(obs=release_obs, sim=release_sim)
            storage_objs = obj_func.calculate(obs=storage_obs, sim=storage_sim)
            
            ### Print summary
            print(line_break)
            print(f"Reservoir: {RESERVOIR_NAME} | Policy: {POLICY_TYPE}")
            
            print(f"\n### Release Objectives: ###")
            for metric_name, objective_value in zip(metrics, release_objs):
                print(f"{metric_name}: {objective_value}")
            
            print(f"\n### Storage Objectives: ###")
            for metric_name, objective_value in zip(metrics, storage_objs):
                print(f"{metric_name}: {objective_value}")
            print(line_break)



            # Plot policy function
            reservoir.policy.plot()
            
            # Plot sim and obs dynamics (storage and release)
            reservoir.plot(
                save=False,
                storage_obs=storage_obs,
                release_obs=release_obs,
                release_log_scale=True
            )

