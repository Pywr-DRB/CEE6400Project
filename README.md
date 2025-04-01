# CEE6400Project
Marilyn &amp; Trevor's course project for CEE6400 Spring 2025


# File Descriptions
- `writeup.md` contains a draft of the project report.
- `literature.md` contains a list of relevant literature to be used to guide methods and be cited in the report.
- `config.py` contains specifications to be used to control project methods.


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

Navigate back to the CEE6400Project/ folder.
```
cd ../CEE6400Project/
```

We need to move several files to the CEE6400Project folder, to be accessed during the optimization:
```
cp ../BorgTraining/borg.py ./
cp -r ../BorgTraining/MOEAFramework-5.0/ ./
cp -r ../BorgTraining/moeaframework/ ./
cp ../BorgTraining/libborg.so ./
cp ../BorgTraining/libborgms.so ./
cp ../BorgTraining/libborgmm.so ./
```

## Running MMBorgMOEA

The following are designed to run on Hopper.  If using a different machine, errors may arise. 

The `parallel_borg_run.py` file is used to execute the MMBorgMOEA optimization for a specific `POLICY_TYPE` and `RESERVOIR_NAME`.

The `POLICY_TYPE` and `RESERVOIR_NAME` must be provided as command line arguments when running the script.   

The `run_parallel_mmborg.sh` script loops through different `POLICY_TYPE` and `RESERVOIR_NAME` options.  

```
sbatch run_parallel_mmborg.sh
```

The following are dfined in the `methods/config.py`, and can be changed by modifying that figure:
- Metrics
- Epsilon values
- Parameter bounds
- Seed number

