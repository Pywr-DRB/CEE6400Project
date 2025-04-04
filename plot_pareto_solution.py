import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods.reservoir.model import Reservoir
from methods.load.observations import load_observations, get_observational_training_data
from methods.metrics.objectives import ObjectiveCalculator

from methods.utils import get_overlapping_datetime_indices


from methods.config import reservoir_min_release, reservoir_max_release, reservoir_capacity
from methods.config import policy_n_params, policy_param_bounds
from methods.config import SEED, cfs_to_mgd, METRICS, OUTPUT_DIR, FIG_DIR, DATA_DIR


# Policies to test
POLICY_TYPE = 'PiecewiseLinear'
RESERVOIR_NAME = 'fewalter'
NFE = 30000


# load parameters from Borg output
fname = f"{OUTPUT_DIR}/MMBorg_3M_{POLICY_TYPE}_{RESERVOIR_NAME}_nfe{NFE}_seed71.csv"
solutions = pd.read_csv(fname)

var_cols = [c for c in solutions.columns if 'var' in c]
obj_cols = [c for c in solutions.columns if 'obj' in c]

all_params = solutions.loc[:, var_cols].values
all_objs = solutions.loc[:, obj_cols].values

# get the solutions best for each objective
best_solutions = {}
for i, obj_col in enumerate(obj_cols):
    # get the index of the best solution for this objective
    best_index = np.argmin(all_objs[:, i])
    best_solutions[obj_col] = all_params[best_index, :]

best_avg_index = np.argmin(np.mean(all_objs, axis=0))
best_solutions['avg_all'] = all_params[best_avg_index, :]

best_avg_index = np.argmin(np.mean(all_objs[:, [0, 2]], axis=1))
best_solutions['avg_nse'] = all_params[best_avg_index, :]

line_break = "##############################################"


if __name__ == "__main__":
    metrics = METRICS
    obj_func = ObjectiveCalculator(metrics=METRICS)


    inflow_obs, release_obs, storage_obs = get_observational_training_data(
        reservoir_name=RESERVOIR_NAME,
        data_dir = DATA_DIR,
        as_numpy=False
    )

    datetime = inflow_obs.index
    inflow_obs = inflow_obs.values
    release_obs = release_obs.values
    storage_obs = storage_obs.values
    
    
    # scale inflow to have mass balance with release
    scale_factor = np.sum(release_obs) / np.sum(inflow_obs)
    inflow_scaled = inflow_obs * scale_factor
    print(f'Reservoir: {RESERVOIR_NAME} | inflow scaled by {scale_factor}')


    for solution_type in best_solutions.keys():

        params = best_solutions[solution_type]

        # Define the reservoir model
        reservoir = Reservoir(
            inflow = inflow_scaled,
            dates= datetime,
            capacity = reservoir_capacity[RESERVOIR_NAME],
            policy_type = POLICY_TYPE,
            policy_params = params,
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
        print(f"Reservoir: {RESERVOIR_NAME} | Policy: {POLICY_TYPE} | Solution: {solution_type}")
        
        print(f"\n### Release Objectives: ###")
        for metric_name, objective_value in zip(metrics, release_objs):
            print(f"{metric_name}: {objective_value}")
        
        print(f"\n### Storage Objectives: ###")
        for metric_name, objective_value in zip(metrics, storage_objs):
            print(f"{metric_name}: {objective_value}")
        print(line_break)



        # # Plot policy function
        fig_fname = f"{FIG_DIR}/policy_functions/"
        fig_fname += f"{POLICY_TYPE}_{RESERVOIR_NAME}_{solution_type}.png"
        reservoir.policy.plot(fname=fig_fname, save=True)
        
        # Plot sim and obs dynamics (storage and release)
        
        fig_fname = f"{FIG_DIR}/solution_dynamics/"
        fig_fname += f"{POLICY_TYPE}_{RESERVOIR_NAME}_{solution_type}.png"
        
        reservoir.plot(
            save=True,
            storage_obs=storage_obs,
            release_obs=release_obs,
            release_log_scale=False,
            fname=fig_fname,
            title=f"{POLICY_TYPE} {RESERVOIR_NAME} {solution_type}",
        )

