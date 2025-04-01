"""
Runs MMBorgMOEA optimization.

Credit to Chung-Yi Lin & Sai Veena Sunkara for the original code,
which has been modified for use with the reservoir problem.

For more info, see:
https://github.com/philip928lin/BorgTraining/tree/main (Private)

"""


import os
import sys
import numpy as np
from pathnavigator import PathNavigator

from methods.reservoir.model import Reservoir
from methods.load.observations import load_observations
from methods.metrics.objectives import ObjectiveCalculator
from methods.utils import get_overlapping_datetime_indices
from methods.config import reservoir_min_release, reservoir_max_release, reservoir_capacity
from methods.config import policy_n_params, policy_param_bounds
from methods.config import SEED, METRICS, EPSILONS


root_dir = os.path.expanduser("./")
pn = PathNavigator(root_dir)
pn.chdir()


### Get POLICY_TYPE and RESERVOIR_NAME from sys.argv
assert len(sys.argv) > 2, "POLICY_TYPE and RESERVOIR_NAME must be provided by command line."

POLICY_TYPE = sys.argv[1]  
RESERVOIR_NAME = sys.argv[2]



##### Settings ####################################################################

### Policy settings
NVARS = policy_n_params[POLICY_TYPE]
BOUNDS = policy_param_bounds[POLICY_TYPE]

### Objectives
METRICS = METRICS
NOBJS = len(METRICS) * 2   # x2 since storage and release objectives
EPSILONS = EPSILONS + EPSILONS

### Borg Settings
NCONSTRS = 0
NFE = 10000            # Number of function evaluation 
runtime_freq = 250      # output frequency
islands = 3             # 1 = MW, >1 = MM  # Note the total NFE is islands * nfe
borg_seed = SEED


### Other
SCALE_INFLOW = True   # if True, scale inflow based on observed release volume


### Load observed data #######################################

inflow_obs = load_observations(datatype='inflow', 
                            reservoir_name=RESERVOIR_NAME, 
                            data_dir="./data/", as_numpy=False)

release_obs = load_observations(datatype='release', 
                                reservoir_name=RESERVOIR_NAME, 
                                data_dir="./data/", as_numpy=False)

storage_obs = load_observations(datatype='storage',
                                reservoir_name=RESERVOIR_NAME, 
                                data_dir="./data/", as_numpy=False)

# get overlapping datetime indices, 
# when all data is available for this reservoir
dt = get_overlapping_datetime_indices(inflow_obs, release_obs, storage_obs)

# subset data
inflow_obs = inflow_obs.loc[dt,:].values
release_obs = release_obs.loc[dt,:].values
storage_obs = storage_obs.loc[dt,:].values


# scale inflow, so that the total inflow volume is equal to the total release volume
if SCALE_INFLOW:
    scale_factor = np.sum(release_obs) / np.sum(inflow_obs)
    inflow_obs = inflow_obs * scale_factor

# Setup objective function
obj_func = ObjectiveCalculator(metrics=METRICS)


### Evaluation function: 
# function(*vars) -> (objs, constrs)
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
        capacity = reservoir_capacity[RESERVOIR_NAME],
        policy_type = POLICY_TYPE,
        policy_params = list(vars),
        release_min = reservoir_min_release[RESERVOIR_NAME],
        release_max = reservoir_max_release[RESERVOIR_NAME],
        initial_storage = storage_obs[0],
        name = RESERVOIR_NAME,
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
    release_objs = obj_func.calculate(obs=release_obs,
                                        sim=sim_release)
    storage_objs = obj_func.calculate(obs=storage_obs,
                                        sim=sim_storage)
    
    objectives = []
    for obj in release_objs:
        objectives.append(obj)
    for obj in storage_objs:
        objectives.append(obj)
    
    return objectives,


borg_settings = {
    "numberOfVariables": NVARS,
    "numberOfObjectives": NOBJS,
    "numberOfConstraints": NCONSTRS,
    "function": evaluate,
    "epsilons": EPSILONS,
    "bounds": BOUNDS,
    "directions": None,  # default is to minimize all objectives. keep this unchanged.
    "seed": SEED
}


if __name__ == "__main__":

    from borg import *
    Configuration.startMPI()

    
    ##### Borg Setup ####################################################################

    borg = Borg(**borg_settings)

    ##### Parallel borg - solvempi #########################################################
    # Make output and checkpoints directories
    pn.mkdir("outputs") 
    pn.outputs.mkdir("checkpoints")

    if islands == 1:
        fname_base = pn.outputs.get() / f"MWBorg_{POLICY_TYPE}_{RESERVOIR_NAME}_nfe{NFE}_seed{borg_seed}"
    else:
        fname_base = pn.outputs.get() / f"MMBorg_{islands}M_{POLICY_TYPE}_{RESERVOIR_NAME}_nfe{NFE}_seed{borg_seed}"
    
    # Runtime
    if islands == 1: # Master slave version
        runtime_filename = f"{fname_base}.runtime"
    else:
        # For MMBorg, the filename should include one %d which gets replaced by the island index
        runtime_filename = f"{fname_base}_%d.runtime"


    solvempi_settings = {
        "islands": islands,
        "maxTime": None,
        "maxEvaluations": NFE,  # Total NFE is islands * maxEvaluations if island > 1
        "initialization": None,
        "runtime": runtime_filename,
        "allEvaluations": None,
        "frequency": runtime_freq,
    }

    result = borg.solveMPI(**solvempi_settings)

    ##### Save results #####################################################################
    if result is not None:
        # The result will only be returned from one node
        with open(f"{fname_base}.csv", "w") as file:
            # You may add header here
            file.write(",".join(
                [f"var{i+1}" for i in range(NVARS)]
                + [f"obj{i+1}" for i in range(NOBJS)]
                + [f"constr{i+1}" for i in range(NCONSTRS)]
                ) + "\n")
            result.display(out=file, separator=",")

        # for MOEAFramework-5.0
        with open(f"{fname_base}.set", "w") as file:
            # You may add header here
            file.write("# Version=5\n")
            file.write(f"# NumberOfVariables={NVARS}\n")
            file.write(f"# NumberOfObjectives={NOBJS}\n")
            file.write(f"# NumberOfConstraints={NCONSTRS}\n")
            for i, bound in enumerate(borg_settings["bounds"]):
                file.write(f"# Variable.{i+1}.Definition=RealVariable({bound[0]},{bound[1]})\n")
            if borg_settings.get("directions") is None:
                for i in range(NOBJS):
                    file.write(f"# Objective.{i+1}.Definition=Minimize\n")
            else:
                for i, direction in enumerate(borg_settings["directions"]):
                    if direction == "min":
                        file.write(f"# Objective.{i+1}.Definition=Minimize\n")
                    elif direction == "max":
                        file.write(f"# Objective.{i+1}.Definition=Maximize\n")
            file.write(f"//NFE={NFE}\n") # if using check point or multi island, the NFE may not be correct.
            result.display(out=file, separator=" ")
            file.write("#\n")
        
        # Write the dictionary to a file in a readable format
        with open(f"{fname_base}.info", 'w') as file:
            file.write("\nBorg settings\n")
            file.write("=================\n")
            for key, value in borg_settings.items():
                file.write(f"{key}: {value}\n")
            file.write("\nBorg solveMPI settings\n")
            file.write("=================\n")
            for key, value in solvempi_settings.items():
                file.write(f"{key}: {value}\n")

        if islands == 1:
            print(f"Master: Completed {fname_base}")
        elif islands > 1:
            print(f"Multi-master controller: Completed {fname_base}")

    ##### End MPI #########################################################################
    Configuration.stopMPI()