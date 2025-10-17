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
import pandas as pd

from pathnavigator import PathNavigator

# Optimization level setting imported from this repo
from methods.reservoir.model import Reservoir
from methods.load.observations import get_observational_training_data
from methods.metrics.objectives import ObjectiveCalculator
from methods.config import SEED, RELEASE_METRICS, STORAGE_METRICS, EPSILONS, NFE, ISLANDS
from methods.config import DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR

# Import settings from pywrdrb
from methods.config import policy_n_params, policy_param_bounds
from methods.config import reservoir_capacity, INERTIA_BY_RESERVOIR, release_max_by_reservoir


root_dir = os.path.expanduser("./")
pn = PathNavigator(root_dir, max_depth=2)
pn.chdir()

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

### Get POLICY_TYPE and RESERVOIR_NAME from sys.argv
assert len(sys.argv) > 2, "POLICY_TYPE and RESERVOIR_NAME must be provided by command line."

POLICY_TYPE = sys.argv[1]  
RESERVOIR_NAME = sys.argv[2]

RESERVOIR_NAME = str(RESERVOIR_NAME)
POLICY_TYPE = str(POLICY_TYPE)

##### Settings ####################################################################

### Policy settings
NVARS = policy_n_params[POLICY_TYPE]
BOUNDS = policy_param_bounds[POLICY_TYPE]

### Objectives
METRICS = RELEASE_METRICS + STORAGE_METRICS
NOBJS = len(METRICS)   # x2 since storage and release objectives
EPSILONS = EPSILONS 

### Borg Settings
NCONSTRS = 1 if POLICY_TYPE == 'STARFIT' else 0

runtime_freq = 250      # output frequency
islands = ISLANDS             # 1 = MW, >1 = MM  # Note the total NFE is islands * nfe

borg_seed = int(sys.argv[3]) if len(sys.argv) > 3 else SEED

### Load observed data #######################################

inflow_obs, release_obs, storage_obs = get_observational_training_data(
    reservoir_name=RESERVOIR_NAME,
    data_dir = PROCESSED_DATA_DIR,
    as_numpy=False,
    inflow_type='inflow_pub' #specify type of inflow data here options: 'inflow', 'inflow_scaled', 'inflow_pub'
)

# Keep datetime
datetime = inflow_obs.index

if len(datetime) < 365:
    print(f"Warning: Only {len(datetime)} days of data available for reservoir '{RESERVOIR_NAME}'.")

# keep arrays
inflow_obs = inflow_obs.values.flatten().astype(np.float64)
release_obs = release_obs.values.flatten().astype(np.float64)
storage_obs = storage_obs.values.flatten().astype(np.float64)

initial_storage_obs = storage_obs[0] 
cap = reservoir_capacity[RESERVOIR_NAME]
R_MAX = release_max_by_reservoir[RESERVOIR_NAME]
iset = INERTIA_BY_RESERVOIR[RESERVOIR_NAME]

# Setup objective function
# Release objectives (uses release inertia settings)
release_obj_func = ObjectiveCalculator(
    metrics=RELEASE_METRICS,
    inertia_tau=iset["release"]["tau"],
    inertia_scale_release=iset["release"]["scale"],
    inertia_release_scale_value=(R_MAX if iset["release"]["scale"] == "value" else None),
)

# Storage objectives (uses storage inertia settings)
storage_obj_func = ObjectiveCalculator(
    metrics=STORAGE_METRICS,
    inertia_tau=iset["storage"]["tau"],
    inertia_scale_storage=iset["storage"]["scale"],
    inertia_storage_scale_value=iset["storage"]["scale_value"],
)

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
        inflow = inflow_obs, # or inflow_scaled or inflow_obs
        dates= datetime,
        capacity = reservoir_capacity[RESERVOIR_NAME],
        policy_type = POLICY_TYPE,
        policy_params = list(vars),
        initial_storage = initial_storage_obs,
        name = RESERVOIR_NAME,
    )
    reservoir.policy.debug = False

    if POLICY_TYPE == 'STARFIT':
        valid = reservoir.policy.test_nor_constraint()
        if not valid:
            with open("violated_params_borg.log", "a") as f:
                f.write(f"[FAIL] {RESERVOIR_NAME} @ {str(pd.Timestamp.now())}:\n")
                f.write(f"{list(vars)}\n\n")
            return [9999.99] * NOBJS, [1.0]

    # Reset the reservoir simulation
    reservoir.reset()
    
    # Run the simulation
    reservoir.run()

     # quick guards
    if (reservoir.storage_array < -1e-9).any():
        print("[WARN] Negative storage encountered.")
    avail = np.r_[initial_storage_obs, reservoir.storage_array[:-1]] + inflow_obs
    if (reservoir.release_array - avail > 1e-6).any():
        print("[WARN] Release > available water at some steps.")

    # summary = reservoir.policy.get_violation_summary()
    # if any(summary.values()):           # only print when something actually happened
    #     print("[violations]", summary)
    
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

    
    objectives = []
    for obj in release_objs:
        objectives.append(obj)
    for obj in storage_objs:
        objectives.append(obj)
        
    if NCONSTRS > 0:    
        return objectives, [0.0]
    else:
        return objectives,

borg_settings = {
    "numberOfVariables": NVARS,
    "numberOfObjectives": NOBJS,
    "numberOfConstraints": NCONSTRS,
    "function": evaluate,
    "epsilons": EPSILONS,
    "bounds": BOUNDS,
    "directions": None,  # default is to minimize all objectives. keep this unchanged.
    "seed": borg_seed
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