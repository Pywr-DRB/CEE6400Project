# MOEA Framework 5.0 — Custom Problem Workflow (6 Objectives)

General background is in:
- https://waterprogramming.wpcomstaging.com/2025/08/14/mm-borg-moea-python-wrapper-checkpointing-runtime-and-operator-dynamics-using-moea-framework-5-0/
- https://waterprogramming.wpcomstaging.com/2025/03/18/introducing-moeaframework-v5-0/

This is a **copy-pasteable** tutorial that follows the blog “runtime dynamics” flow but uses **our repo layout** and **scripts**.  
It installs MOEA v5, organizes results by **Policy → Reservoir**, builds our **custom problems** (PWL/STARFIT/RBF; 6 objectives), and runs the **4-step MOEA workflow** (runtime→set, header, refset, metrics).

We assume file names like:
```
MMBorg_4M_<POLICY>_<RESERVOIR>_nfe..._seed..._<island>.runtime
```
and a **6-objective** analysis.

---

## 0) Install MOEA Framework 5.0 (CLI)

```bash
# install_moea5.sh
set -euo pipefail

MOEAFrameworkURL="https://github.com/MOEAFramework/MOEAFramework/releases/download/v5.0/MOEAFramework-5.0.tar.gz"
MOEAFrameworkTar="MOEAFramework-5.0.tar.gz"
MOEAFramework5Path="MOEAFramework-5.0"
cliPath="$MOEAFramework5Path/cli"

if [ ! -d "$MOEAFramework5Path" ]; then
  echo "[INFO] Downloading MOEAFramework-5.0..."
  curl -L -o "$MOEAFrameworkTar" "$MOEAFrameworkURL"
  tar -xzf "$MOEAFrameworkTar"
  rm "$MOEAFrameworkTar"
fi

chmod 775 "$cliPath" || true
echo "[OK] MOEAFramework CLI ready at $cliPath"
```

Run:
```bash
bash install_moea5.sh
cd MOEAFramework-5.0
```

## Step 0.1 — Post-process & organize results by Policy / Reservoir

> Move/sync `./outputs/` from the main repo into `./moeaframework/outputs/Policy_<POLICY>/runtime/<RESERVOIR>/`.

Run:
```bash
bash postprocess_to_moea.sh
```
---

## 1) Build Custom External Problems (6 objectives)

> If you intentionally run a 4-objective experiment, change `--numberOfObjectives` and the epsilon vectors later. Otherwise keep all at **6**.

```bash
# From MOEAFramework-5.0/

# PWL (15 vars, 6 objs, 0 constraints)
./cli BuildProblem   --problemName PWL_Custom   --language python   --numberOfVariables 15   --numberOfObjectives 4   --numberOfConstraints 0   --lowerBound -1e6   --upperBound  1e6

# STARFIT (17 vars, 6 objs, 1 constraint)
./cli BuildProblem   --problemName STARFIT_Custom   --language python   --numberOfVariables 17   --numberOfObjectives 4   --numberOfConstraints 0   --lowerBound -1e6   --upperBound  1e6

# RBF (14 vars, 6 objs, 0 constraints)
./cli BuildProblem   --problemName RBF_Custom   --language python   --numberOfVariables 14   --numberOfObjectives 4   --numberOfConstraints 0   --lowerBound -1e6   --upperBound  1e6
```

Build and expose the jar(s):
```bash
cd native
# e.g., cd python/PWL_Custom  (repeat for STARFIT_Custom, RBF_Custom)
make
make run
cp **/*.jar ../../lib/
cd ..
```

---

## 2) Create v5 Header (required for tools)

Create `1-header-file.txt` in `MOEAFramework-5.0/`:
```text
# Version=5
# Notes=Custom 6-objective reservoir policy problems (PWL/STARFIT/RBF). Keep epsilon vectors length=6.
```
## 3) — Convert, header, merge, evaluate (our 4-step MOEA workflow)

> Wraps the tutorial’s “generate .set”, “patch header”, “merge .ref”, and “metrics” steps.  
> Uses the **organized** layout (`Policy_<POLICY>/runtime/<RESERVOIR>`).

```bash
# run_moea_workflow.sh
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- \"$(dirname \"${BASH_SOURCE[0]}\")\" &>/dev/null && pwd)"

export EPSILON=\"0.01,0.01,0.02,0.01,0.01,0.01\"  # 6 objectives
POLICIES=(\"STARFIT\" \"RBF\" \"PWL\")
RESERVOIRS=(\"beltzvilleCombined\" \"fewalter\" \"prompton\" \"blueMarsh\")
declare -A NUM_DVS=( [\"STARFIT\"]=17 [\"RBF\"]=14 [\"PWL\"]=15 )
export NUM_OBJS=6
export OUT_ROOT=\"outputs\"

export SEED_FROM=\"${SEED_FROM:-1}\"
export SEED_TO=\"${SEED_TO:-10}\"

CLI_ARG=\"${1:-}\"
if [[ \"$CLI_ARG\" == \"--cli\" ]]; then
  CLI=\"${2:?usage: $0 [--cli /path/to/cli]}\"; shift 2
else
  if   [[ -x \"$SCRIPT_DIR/cli\" ]]; then CLI=\"$SCRIPT_DIR/cli\"
  elif [[ -x \"$SCRIPT_DIR/MOEAFramework-5.0/cli\" ]]; then CLI=\"$SCRIPT_DIR/MOEAFramework-5.0/cli\"
  elif [[ -x \"$SCRIPT_DIR/../MOEAFramework-5.0/cli\" ]]; then CLI=\"$SCRIPT_DIR/../MOEAFramework-5.0/cli\"
  else
    echo \"ERROR: MOEAFramework cli not found. Pass --cli /path/to/cli\" >&2; exit 1
  fi
fi
[[ -x \"$CLI\" ]] || { echo \"ERROR: CLI not executable: $CLI (chmod +x)\"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo \"ERROR: python3 not found in PATH\"; exit 1; }

echo \"== MOEA Workflow (policy/reservoir layout) ==\"
echo \"CLI       : $CLI\"
echo \"OUT_ROOT  : $OUT_ROOT\"
echo \"EPSILON   : $EPSILON\"
echo \"NUM_OBJS  : $NUM_OBJS\"
echo \"SEEDS     : ${SEED_FROM}-${SEED_TO}\"
echo \"Policies  : ${POLICIES[*]}\"
echo \"Reservoirs: ${RESERVOIRS[*]}\"
echo

echo \">> [1/4] runtime -> set\"
bash \"$SCRIPT_DIR/1-moeaframework_merge_files.sh\" --cli \"$CLI\"
echo

echo \">> [2/4] append header -> *_header.set\"
python3 \"$SCRIPT_DIR/append_header.py\" --outputs-root \"$OUT_ROOT\" --seed-from \"$SEED_FROM\" --seed-to \"$SEED_TO\"
echo

echo \">> [3/4] merge *_header.set -> <reservoir>.ref\"
EPSILON=\"$EPSILON\" DIMENSION=\"$NUM_OBJS\" SEED_FROM=\"$SEED_FROM\" SEED_TO=\"$SEED_TO\" \
  bash \"$SCRIPT_DIR/2-moeaframework_gen_refset.sh\" --cli \"$CLI\"
echo

echo \">> [4/4] metrics vs <reservoir>.ref\"
EPSILON=\"$EPSILON\" SEED_FROM=\"$SEED_FROM\" SEED_TO=\"$SEED_TO\" \
  bash \"$SCRIPT_DIR/3-moeaframework_gen_runtime.sh\" --cli \"$CLI\"
echo
echo \"== Done ==\"
```

Run:
```bash
# from moeaframework/
bash run_moea_workflow.sh --cli \"$(pwd)/MOEAFramework-5.0/cli\"
```

## Policy sanity table

| Policy   | Vars | Objs | Constr | Notes                                 |
|----------|-----:|-----:|-------:|---------------------------------------|
| PWL      |   15 |    6 |      0 | Piecewise Linear control rule         |
| STARFIT  |   17 |    6 |      1 | Seasonal harmonic + NOR constraint    |
| RBF      |   14 |    6 |      0 | Radial Basis Function policy          |

---

## Directory layout (after organization + workflow)

```
moeaframework/
  MOEAFramework-5.0/
  sync_and_organize_moea.sh
  run_moea_workflow.sh
  1-moeaframework_merge_files.sh
  2-moeaframework_gen_refset.sh
  3-moeaframework_gen_runtime.sh
  append_header.py
  outputs/
    Policy_STARFIT/
      runtime/<reservoir>/*.runtime
      refsets/<reservoir>/*_header.set, seed*.ref, <reservoir>.ref
      metrics/<reservoir>/*.metric
    Policy_RBF/...
    Policy_PWL/...
```

---

## Notes / gotchas

- Keep **problem names** consistent (`PWL_Custom`, `STARFIT_Custom`, `RBF_Custom`) between **BuildProblem**, merge, and metrics.
- **EPSILON length must equal #objectives (6)**.
- Header files: you already maintain **per-policy** headers; the header appender is **safe** (adds only if missing).
- Don’t mix **4-objective** and **6-objective** runs in one workflow.