#!/bin/bash
#SBATCH --job-name=PreprocessObs
#SBATCH --output=logs/preprocess.out
#SBATCH --error=logs/preprocess.err
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G

module load python/3.11.5
source venv/bin/activate

echo "Starting observational data retrieval..."
python 01_retrieve_data.py

echo "Processing raw data..."
python 02_process_data.py

echo "Done."
