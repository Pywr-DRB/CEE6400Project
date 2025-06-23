#!/bin/bash
#SBATCH --job-name=PolInf        # Job name
#SBATCH --output=./logs/%j.out  # Standard output log file with job ID
#SBATCH --error=./logs/%j.err   # Standard error log file with job ID
#SBATCH --nodes=4                          # Number of nodes to use
#SBATCH --ntasks-per-node=40               # Number of tasks (processes) per node
#SBATCH --exclusive                        # Use the node exclusively for this job
#SBATCH --mail-type=END                    # Send email at job end
#SBATCH --mail-user=tja73@cornell.edu     # Email for notifications

# Remember to create ./logs/ first!

module load python/3.11.5
source venv/bin/activate

# Function to submit the job
submit_job() {
    local seed=$1
    # Print start message and the number of nodes and tasks per node
    datetime=$(date '+%Y-%m-%d %H:%M:%S')
    n_processors=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

    echo "[JobID $SLURM_JOB_ID] Running dps_borg simulation with seed $seed ..."
    echo "Number of nodes: $SLURM_NNODES"
    echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
    echo "Total number of processors: $n_processors"
    echo "Datetime: $datetime"

    # Run the script with MPI and time the execution
    time mpirun --oversubscribe -np $n_processors python parallel_DTLZ2_example.py $SLURM_JOB_ID $seed

    # Ensure the job finishes before proceeding to the next
    wait
}


# Define function to submit a single job iteration
submit_job() {
    local POLICY_TYPE=$1
    local RESERVOIR_NAME=$2
    local seed=$3

    datetime=$(date '+%Y-%m-%d %H:%M:%S')
    n_processors=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

    echo "Running: POLICY_TYPE=$POLICY_TYPE, RESERVOIR_NAME=$RESERVOIR_NAME"
    echo "Datetime: $datetime"
    echo "Total processors: $n_processors"

    # Run with MPI
    time mpirun --oversubscribe -np $n_processors python 03_parallel_borg_run.py "$POLICY_TYPE" "$RESERVOIR_NAME" $seed
    echo "Finished seed $seed - POLICY_TYPE=$POLICY_TYPE - RESERVOIR_NAME=$RESERVOIR_NAME"
    echo "#############################################"
    
    wait
}

# Loop to submit jobs with different seeds
POLICY_TYPE="PiecewiseLinear"
RESERVOIR_NAME="fewalter"

for seed in {1..10}; do 
    submit_job "$POLICY_TYPE" "$RESERVOIR_NAME" $seed
done



# Copy runtime files to separate folders
echo "Copying runtime and refset files to respective folders..."
for seed in {1..10}; do
   cp outputs/MMBorg_4M_PiecewiseLinear_fewalter_nfe30000_seed${seed}_*.runtime ./runtime/ 2>/dev/null
   cp outputs/MMBorg_4M_PiecewiseLinear_fewalter_nfe30000_seed${seed}.set refsets/ 2>/dev/null
done