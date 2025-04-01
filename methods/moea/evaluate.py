import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods.reservoir.model import Reservoir
from methods.load.observations import load_observations
from methods.metrics.objectives import ObjectiveCalculator

from methods.utils import get_overlapping_datetime_indices


from methods.config import reservoir_min_release, reservoir_max_release, reservoir_capacity
from methods.config import policy_n_params, policy_param_bounds
from methods.config import SEED, cfs_to_mgd

from methods.sampling import generate_policy_param_samples


def get_evaluation_function(reservoir_name, 
                            policy_type, 
                            metrics):
    
    ### Load observed data
    inflow_obs = load_observations(datatype='inflow', 
                                reservoir_name=reservoir_name, 
                                data_dir="./data/", as_numpy=False)

    release_obs = load_observations(datatype='release', 
                                    reservoir_name=reservoir_name, 
                                    data_dir="./data/", as_numpy=False)

    storage_obs = load_observations(datatype='storage',
                                    reservoir_name=reservoir_name, 
                                    data_dir="./data/", as_numpy=False)
    
    # get overlapping datetime indices, 
    # when all data is available for this reservoir
    dt = get_overlapping_datetime_indices(inflow_obs, release_obs, storage_obs)
    
    # subset data
    inflow_obs = inflow_obs.loc[dt,:].values
    release_obs = release_obs.loc[dt,:].values
    storage_obs = storage_obs.loc[dt,:].values
    

    # Setup objective function
    obj_func = ObjectiveCalculator(metrics=metrics)
    
    
    ### Evaluation function
    def evaluate(*vars):
        """
        Runs one reservoir operation function evaluation.
        
        Args:
            *vars: The reservoir policy parameters to evaluate.
            
        Returns:
            tuple: The objective values
        """
        ### Setup reservoir
        reservoir = Reservoir(
            inflow = inflow_obs,
            capacity = reservoir_capacity[reservoir_name],
            policy_type = policy_type,
            policy_params = list(vars),
            release_min = reservoir_min_release[reservoir_name],
            release_max = reservoir_max_release[reservoir_name],
            initial_storage = storage_obs[0],
            name = reservoir_name,
        )
        
        # Re-assign the reservoir policy params
        reservoir.policy.policy_params = list(vars)
        reservoir.policy.parse_policy_params()
        
        # Reset the reservoir simulation
        reservoir.reset()
        
        # Run the simulation
        reservoir.run()
        
        # Retrieve simulated release data
        sim_release = reservoir.release_array
        sim_storage = reservoir.storage_array 
        
        # Calculate the objectives
        objectives = obj_func.calculate(obs=release_obs, 
                                        sim=sim_release)
        
        return objectives,
    
    
    # Return the evaluation function
    return evaluate
