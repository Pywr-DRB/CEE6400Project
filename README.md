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


Start by cloning the repositories:
```
git clone https://github.com/Pywr-DRB/CEE6400Project.git
git clone https://github.com/philip928lin/BorgTraining.git
git clone 
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

The `CEE6400Project/requirements.txt` contains all requirements (also for BorgMOEA), so we will use that when creating a virtual environment.

On Hopper:
```
module load python/3.11.5
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Setup MMBorgMOEA

Assuming that you have the `MMBorgMOEA` repo already available in your working directory.

```
cd ./MMBorgMOEA
mpicc -shared -fPIC -O3 -o libborgmm.so borgmm.c mt19937ar.c -lm
```

Navigate back to the CEE6400Project/ folder.
```
cd ../CEE6400Project/
```

We need to move several files to the CEE6400Project folder, to be accessed during the optimization:
```
cp ../BorgTraining/borg.py ./
cp -r ../BorgTraining/MOEAFramework-5.0/ ./
cp -r ../BorgTraining/moeaframework/ ./
cp ../MMBorgMOEA/libborg.so ./
cp ../MMBorgMOEA/libborgms.so ./
cp ../MMBorgMOEA/libborgmm.so ./
```

## Running MMBorgMOEA
