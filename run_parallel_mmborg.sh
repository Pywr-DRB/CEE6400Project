#!/bin/bash
#SBATCH --job-name=ResBorg        # Job name
#SBATCH --output=./logs/ResBorg.out  
#SBATCH --error=./logs/ResBorg.err   
#SBATCH --nodes=2                          # Number of nodes to use
#SBATCH --ntasks-per-node=40               # Number of tasks (processes) per node
#SBATCH --exclusive                        # Use the node exclusively for this job

# Remember to create ./logs/ first!

# Load Python module
module load python/3.11.5

# Activate Python virtual environment
source venv/bin/activate

# Function to submit the job
submit_job() {
    # Print start message and the number of nodes and tasks per node
    datetime=$(date '+%Y-%m-%d %H:%M:%S')
    n_processors=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

    echo "Number of nodes: $SLURM_NNODES"
    echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
    echo "Total number of processors: $n_processors"
    echo "Datetime: $datetime"

    # Run the script with MPI and time the execution
    time mpirun --oversubscribe -np $n_processors python parallel_borg_run.py $SLURM_JOB_ID

    # Ensure the job finishes before proceeding to the next
    wait
}


submit_job