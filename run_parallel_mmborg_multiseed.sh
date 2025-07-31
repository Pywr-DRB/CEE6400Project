#!/bin/bash
#SBATCH --job-name=CustomMultiseed        # Job name
#SBATCH --output=./logs/%j.out  # Standard output log file with job ID
#SBATCH --error=./logs/%j.err   # Standard error log file with job ID
#SBATCH --nodes=3                         # Number of nodes to use
#SBATCH --ntasks-per-node=40               # Number of tasks (processes) per node
#SBATCH --exclusive                        # Use the node exclusively for this job
#SBATCH --mail-type=END                    # Send email at job end
#SBATCH --mail-user=ms3654@cornell.edu     # Email for notifications

# Load Python module
module load python/3.11.5

# Activate Python virtual environment #change based on your directory
source /home/fs02/pmr82_0001/ms3654/envs/borg-env/bin/activate

# Define function to submit a single job iteration
submit_job() {
    local seed=$1
    local POLICY_TYPE=$2
    local RESERVOIR_NAME=$3

    datetime=$(date '+%Y-%m-%d %H:%M:%S')
    n_processors=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

    echo "[JobID $SLURM_JOB_ID] Running dps_borg simulation with seed $seed ..."
    echo "Number of nodes: $SLURM_NNODES"
    echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
    echo "Running: POLICY_TYPE=$POLICY_TYPE, RESERVOIR_NAME=$RESERVOIR_NAME"
    echo "Datetime: $datetime"
    echo "Total processors: $n_processors"

    # Run with MPI
    time mpirun --oversubscribe -np $n_processors python 03_parallel_borg_run.py "$POLICY_TYPE" "$RESERVOIR_NAME" "$seed"
    echo "Finished: POLICY_TYPE=$POLICY_TYPE, RESERVOIR_NAME=$RESERVOIR_NAME"
    echo "#############################################"
    
    wait
}


# === Define parameters === #
POLICY_TYPES=("RBF" "PiecewiseLinear" "STARFIT")
RESERVOIR_NAMES=("fewalter" "prompton" "beltzvilleCombined")

# === Loop through all combinations === #
for seed in {1..10}; do 
    for POLICY_TYPE in "${POLICY_TYPES[@]}"; do
        for RESERVOIR_NAME in "${RESERVOIR_NAMES[@]}"; do
            echo "Submitting job for $POLICY_TYPE - $RESERVOIR_NAME - SEED $seed"
            submit_job "$seed" "$POLICY_TYPE" "$RESERVOIR_NAME" 
        done
    done
done

echo "All multi-seed jobs submitted."