#!/usr/bin/env bash
set -e

# Get the absolute path to the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EPSILON="0.01,0.01,0.01,0.01"
POLICIES=("STARFIT" "RBF" "PiecewiseLinear")
RESERVOIRS=("beltzvilleCombined" "fewalter" "prompton")

# Define the number of DVs per policy (used by step 3)
declare -A NUM_DVS
NUM_DVS["STARFIT"]=17
NUM_DVS["RBF"]=14
NUM_DVS["PiecewiseLinear"]=15

# Fixed number of objectives
NUM_OBJS=4
export NUM_DVS NUM_OBJS  # step 3 reads these

for POLICY in "${POLICIES[@]}"; do
  echo "Running diagnostics for policy: $POLICY"
  RUNTIME_DIR="outputs/Policy_${POLICY}/runtime"

  for RES in "${RESERVOIRS[@]}"; do
    DIR="${RUNTIME_DIR}/${RES}"
    if [ ! -d "$DIR" ]; then
      echo "Skipping missing directory: $DIR"
      continue
    fi

    echo "Processing: $POLICY – $RES"
    cd "$DIR"

    # Step 1: runtime -> refsets/*.set (with header)
    bash "$SCRIPT_DIR/1-moeaframework_merge_files.sh" "$EPSILON"

    # Step 2: merge per-seed -> seed*.ref; then union -> borg.ref
    bash "$SCRIPT_DIR/2-moeaframework_gen_refset.sh" "$EPSILON"

    # Step 3: evaluate metrics vs borg.ref
    bash "$SCRIPT_DIR/3-moeaframework_gen_runtime.sh" "$EPSILON"

    cd - > /dev/null
  done
done

echo "MOEA diagnostics complete for all policies and reservoirs."
