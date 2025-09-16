#!/bin/bash
#SBATCH --job-name=ResBorg
#SBATCH --output=./logs/ResBorg.out
#SBATCH --error=./logs/ResBorg.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive

# Load Python module
module load python/3.11.5

# Activate Python virtual environment #change based on your directory
source /home/fs02/pmr82_0001/ms3654/envs/borg-env/bin/activate

# Define function to submit a single job iteration
submit_job() {
    local POLICY_TYPE=$1
    local RESERVOIR_NAME=$2

    datetime=$(date '+%Y-%m-%d %H:%M:%S')
    n_processors=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

    echo "Running: POLICY_TYPE=$POLICY_TYPE, RESERVOIR_NAME=$RESERVOIR_NAME"
    echo "Datetime: $datetime"
    echo "Total processors: $n_processors"

    # Run with MPI
    time mpirun --oversubscribe -np $n_processors python 03_parallel_borg_run.py "$POLICY_TYPE" "$RESERVOIR_NAME"
    echo "Finished: POLICY_TYPE=$POLICY_TYPE, RESERVOIR_NAME=$RESERVOIR_NAME"
    echo "#############################################"
    
    wait
}


# Arrays of policy types and reservoir names

# "RBF" "PWL" "STARFIT"
POLICY_TYPES=("RBF" "PWL" "STARFIT")
# "fewalter" "prompton" "beltzvilleCombined"
RESERVOIR_NAMES=("fewalter" "prompton" "beltzvilleCombined")

# Loop through all combinations of reservoir names and policy types
for POLICY_TYPE in "${POLICY_TYPES[@]}"; do
    for RESERVOIR_NAME in "${RESERVOIR_NAMES[@]}"; do
        echo "Submitting job for $POLICY_TYPE - $RESERVOIR_NAME"
        submit_job "$POLICY_TYPE" "$RESERVOIR_NAME"
    done
done

echo "All jobs completed."