#!/bin/bash
#SBATCH --job-name=SFT
#SBATCH --output=./logs/starfit_test.out
#SBATCH --error=./logs/starfit_test.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Load Python module
module load python/3.11.5

# Activate Python virtual environment
source venv/bin/activate

python test_reservoir_policies.py