# Instructions
These instructions supplement but shouldn't replace the MMBorg Bootcamp powerpoint in `Box/Reed.GroupShare/Training/MMBorg Bootcamp/` and the instructions in the `BorgTraining-Python` repo.

Clone the `passNFE_ALH_PyCheckpoint` branch of the `MMBorgMOEA` repository. 

I already had the master branch of `MMBorgMOEA` resposity cloned, so I switched to the specific branch using:

```
git checkout -b passNFE_ALH_PyCheckpoint
git fetch
git pull origin passNFE_ALH_PyCheckpoint
git reset --hard origin/passNFE_ALH_PyCheckpoint
```

Comment out lines 2842 and 2843 of `borg.c`:
```
// BORG_Check_scan1(fscanf(file, "Next Guassian: %lg\n", &nextNextGaussian));
// BORG_Check_scan1(fscanf(file, "Have Next Guassian: %d\n", &haveNextNextGaussian));
```


Compile:
```
mpicc -shared -fPIC -03 -o libborgmm.so borgmm.c mt19937ar.c -lm
```


Copy files to `CEE6400Project/`:
```
cp borg.py libborgmm.so borgmm.c mt19937ar.c <PATH-TO>/CEE6400Project/
```

Replace the `borg.py` file with the version from the Box training materials. 


Run multi-seed optimization:
```
sbatch S1-run_parallel_multiseed.sh
```


## MOEAFramework 5.0 processing

Run:
```
sbatch S0-MOEAFramework5-install.sh

cd MOEAFramework-5.0/

./cli BuildProblem --problemName PolInf --language python --numberOfVariables 15 --numberOfObjectives 4 --lowerBound -10.0 --upperBound 10.0

cd native/PolInf/

make

cp PolInf.jar ../../lib/

cd ../../../

sbatch S2-MOEAFramework5-gen-approx-refsets.sh

sbatch S3-MOEAFramework3-runtime-metrics.sh

sbatch S4-extract-dvs.sh refsets/PiecewiseLinear_fewalter_refset.ref refsets/refset_dvs.csv
sbatch S4-extract-objs.sh refsets/PiecewiseLinear_fewalter_refset.ref refsets/refset_objs.csv

python S5-plot-metrics.py
```

