#!/bin/bash
#SBATCH --job-name=Postproc
#SBATCH --output=logs/postprocess.out
#SBATCH --error=logs/postprocess.err
#SBATCH --ntasks=1
#SBTACH --nodes=1

module load python/3.11.5
source venv/bin/activate

### Run pareto solutions
echo "Running pareto solution policy simulations..."
python plot_pareto_solution.py

### Make figures
echo "Starting figure generation..."
python 0x_make_figures.py

