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

from methods.moea.evaluate import get_evaluation_function


##### Set for parallel borg
from borg import *
Configuration.startMPI()

root_dir = os.path.expanduser("./")
pn = PathNavigator(root_dir)
pn.chdir()


##### Settings ####################################################################

### Policy settings
RESERVOIR_NAME = "fewalter"
POLICY_TYPE = "RBF"
NVARS = policy_n_params[POLICY_TYPE]
BOUNDS = policy_param_bounds[POLICY_TYPE]


### Objectives
METRICS = METRICS
NOBJS = len(METRICS)


### Borg Settings
NCONSTRS = 0
NFE = 5000            # Number of function evaluation 
runtime_freq = 250      # output frequency
islands = 1             # 1 = MW, >1 = MM  # Note the total NFE is islands * nfe


### Evaluation function: function(*vars) -> (objs, constrs)
evaluate_func = get_evaluation_function(
    RESERVOIR_NAME, 
    POLICY_TYPE, 
    METRICS
)


borg_settings = {
    "numberOfVariables": NVARS,
    "numberOfObjectives": NOBJS,
    "numberOfConstraints": NCONSTRS,
    "function": evaluate_func,
    "epsilons": EPSILONS,
    "bounds": BOUNDS,
    "directions": None,  # default is to minimize all objectives. keep this unchanged.
    "seed": SEED
}


### job ID
job_id = "00000" # Default value
if len(sys.argv) > 1:
    job_id = sys.argv[1]  # Capture the job id from the command line


### Random seed for Borg
borg_seed = SEED
if len(sys.argv) > 2 and sys.argv[2] != "None":
    borg_seed = int(sys.argv[2])  # Capture the seed from the command line



if __name__ == "__main__":
    
    ##### Borg Setup ####################################################################

    borg = Borg(**borg_settings)

    ##### Parallel borg - solvempi #########################################################
    # Make output and checkpoints directories
    pn.mkdir("outputs") 
    pn.outputs.mkdir("checkpoints")

    # Runtime
    if islands == 1: # Master slave version
        runtime_filename = pn.outputs.get() / f"parallel_{job_id}_nfe{NFE}_seed{borg_seed}.runtime"
    else:
        # For MMBorg, the filename should include one %d which gets replaced by the island index
        runtime_filename = pn.outputs.get() / f"parallel_{job_id}_nfe{NFE}_seed{borg_seed}_%d.runtime"
        
    # Checkpoint
    newCheckpointFileBase_filename = pn.outputs.checkpoints.get() / f"parallel_{job_id}_nfe{nfe}_seed{borg_seed}"

    # Load previous checkpoint (the file must already exist)
    oldCheckpointFile_filename = pn.outputs.checkpoints.get() / "parallel_108649_nfe300_seed1_nfe100.checkpoint"

    # Evaluation
    evaluationFile_filename = pn.outputs.get() / f"{job_id}_nfe{NFE}_seed{borg_seed}.eval"

    solvempi_settings = {
        "islands": islands,
        "maxTime": None,
        "maxEvaluations": NFE,  # Total NFE is islands * maxEvaluations if island > 1
        "initialization": None,
        "runtime": runtime_filename,
        "allEvaluations": None,
        "frequency": runtime_freq,
        # "newCheckpointFileBase": newCheckpointFileBase_filename, # Output checkpoint
        #"oldCheckpointFile": oldCheckpointFile_filename, # Load checkpoint if uncommented
        #"evaluationFile": evaluationFile_filename
    }

    result = borg.solveMPI(**solvempi_settings)

    ##### Save results #####################################################################
    if result is not None:
        # The result will only be returned from one node
        with open(pn.outputs.get() / f"{job_id}_nfe{NFE}_seed{borg_seed}.csv", "w") as file:
            # You may add header here
            file.write(",".join(
                [f"var{i+1}" for i in range(NVARS)]
                + [f"obj{i+1}" for i in range(NOBJS)]
                + [f"constr{i+1}" for i in range(NCONSTRS)]
                ) + "\n")
            result.display(out=file, separator=",")

        # for MOEAFramework-5.0
        with open(pn.outputs.get() / f"{job_id}_nfe{NFE}_seed{borg_seed}.set", "w") as file:
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
        with open(pn.outputs.get() / f"{job_id}_nfe{NFE}_seed{borg_seed}.info", 'w') as file:
            file.write("\nBorg settings\n")
            file.write("=================\n")
            for key, value in borg_settings.items():
                file.write(f"{key}: {value}\n")
            file.write("\nBorg solveMPI settings\n")
            file.write("=================\n")
            for key, value in solvempi_settings.items():
                file.write(f"{key}: {value}\n")

        if islands == 1:
            print(f"Master: Completed dps_borg_{job_id}_nfe{NFE}_seed{borg_seed}.csv")
        elif islands > 1:
            print(f"Multi-master controller: Completed dps_borg_{job_id}_nfe{NFE}_seed{borg_seed}.csv")

    ##### End MPI #########################################################################
    Configuration.stopMPI()