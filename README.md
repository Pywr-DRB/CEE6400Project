# CEE6400Project
Marilyn &amp; Trevor's course project for CEE6400 Spring 2025

Parametric reservoir policies (STARFIT, RBF, PWL) are optimized with MMBorgMOEA and validated in both a standalone reservoir simulator and Pywr-DRB. This repo provides a reproducible workflow from data prep → optimization → post-processing/figures → validation.

# Repo Structure (key files)

CEE6400Project/
├── 01_retrieve_data.py
├── 02_process_data.py
├── 03_parallel_borg_run.py
├── 04_make_figures.py
├── 05_validate.py
├── run_preprocessing.sh
├── run_parallel_mmborg.sh
├── run_postprocessing_and_figures.sh
├── methods/
│ ├── config.py # single source of truth for config/context
│ ├── reservoir/model.py # standalone Reservoir model
│ ├── load/ # loaders for results & observations
│ └── plotting/ # figure builders (Pareto, axes, dynamics, 9-panel, errors)
├── obs_data/{raw,processed,pub_reconstruction}
├── outputs/ # BORG CSVs
├── figures/ # figures (fig1..fig5 and subfolders)
├── logs/ # SLURM logs
├── borg.py, libborg*.so, MOEAFramework-5.0/, moeaframework/
└── requirements.txt


# Resources:
- [BorgTraining (GitHub)](https://github.com/philip928lin/BorgTraining)
- [Everything You Need to Run Borg MOEA and Python Wrapper – Part 2 (WaterProgramming)](https://waterprogramming.wordpress.com/2025/02/19/everything-you-need-to-run-borg-moea-and-python-wrapper-part-2/)


****

# Workflow

Start by cloning the following repositories:
```
git clone https://github.com/Pywr-DRB/CEE6400Project.git
git clone https://github.com/philip928lin/BorgTraining.git
```

These instructions assume you have the following folder structure. You may have a different folder structure, in which you will need to modify the paths when copying files or navigating folders. 

```
/
./CEE6400Project/
./BorgTraining/
```

We will work in the CEE6400Project/ directory, and will move other code to this folder as needed.

```
cd ./CEE6400Project/
```

## Setup virtual environment

The `CEE6400Project/requirements.txt` contains all requirements, so we will use that when creating a virtual environment.

On Hopper:
```
module load python/3.11.5
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Setup MMBorgMOEA

Rather than compiling MMBorgMOEA from scratch, these instructions simply copy the already compiled `libborg*.sc` files from the `BorgTraining` (private) repo. If you encounter errors, you may need to clone the `MMBorgMOEA` repo and recompile, following the instructions in the WaterProgramming guide. 


We need to move several files from the `BorgTraining` folder to the `CEE6400Project` folder, to be accessed during the optimization:
```
cp ./BorgTraining/borg.py ./CEE6400Project/
cp -r ./BorgTraining/MOEAFramework-5.0/ ./CEE6400Project/
cp -r ./BorgTraining/moeaframework/ ./CEE6400Project/
cp ./BorgTraining/libborg.so ./CEE6400Project/
cp ./BorgTraining/libborgms.so ./CEE6400Project/
cp ./BorgTraining/libborgmm.so ./CEE6400Project/
```

## Running MMBorgMOEA

The following are designed to run on Hopper.  If using a different machine, errors may arise. 

Navigate back to the `CEE6400Project` folder:
```
cd ./CEE6400Project
```

The `03_parallel_borg_run.py` file is used to execute the MMBorgMOEA optimization for a specific `POLICY_TYPE` and `RESERVOIR_NAME`.

The `POLICY_TYPE` and `RESERVOIR_NAME` must be provided as command line arguments when running the script.   

Submit the multi-reservoir × multi-policy sweep on Hopper:

The `run_parallel_mmborg.sh` script loops through different `POLICY_TYPE` and `RESERVOIR_NAME` options.  

```
sbatch run_parallel_mmborg.sh
```

The following are dfined in the `methods/config.py`, and can be changed by modifying that figure:
- Metrics
- Epsilon values
- Parameter bounds
- Seed number

## Post-processing and Figures 

After optimization completes, generate Pareto comparisons, parallel axes, dynamics, and validation figures:

```
sbatch run_postprocessing_and_figures.sh
```

That script runs:

`04_make_figures.py` → Fig 1–4

- Fig 1: Pareto Front Comparison (per reservoir, across policies)

- Fig 2: Parallel Axes (all solutions; and highlighted “best” picks)

- Fig 3: System Dynamics diagnostics (quantiles, spaghetti + FDC, weekly scatter)

- Fig 4: 9-panel validation plots over a historical window

`05_validate.py` → Fig 5 (validation overlays & error diagnostics)

- 2×1 overlay: Independent vs Pywr-DRB (Parametric) vs Pywr-DRB (Default)

- Error time series & error-vs-percentile (with seasonal/decadal panels)

- Optional CSV series saving and numerical comparisons

## Reproducibility knobs

Key parameters and settings are centralized in `methods/config.py`:

- `SEED` (default: 71)  
- `NFE` (default: 30000)  
- `ISLANDS` (default: 4)  
- `EPSILONS`, `METRICS`, `OBJ_FILTER_BOUNDS`  
- All policy parameter bounds (STARFIT/RBF/PWL)  
- Per-reservoir capacities, inflow bounds, release min/max  

Additional reproducibility controls:

- **SLURM job files:**  
  - `run_parallel_mmborg.sh` (nodes, tasks per node, module loads, environment path)  
  - `run_postprocessing_and_figures.sh` (module, environment, figure scripts)  

**Note:** There are currently two `config.py` files — one in this repo and one in the Pywr-DRB repo. They are kept the same for now, but the **main configuration** lives in the Pywr-DRB branch.  

## Policy sources

The parametric policy classes (**STARFIT, RBF, PWL**) used here originate from the  
[Pywr-DRB repository](https://github.com/Pywr-DRB/Pywr-DRB) (feature branch for parametric releases).  
This ensures that the policies optimized in this repo are consistent with those implemented in Pywr-DRB  
for validation and comparison.  
