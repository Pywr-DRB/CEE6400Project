#!/bin/bash
set -e

# Get the absolute path to the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EPSILON="0.01,0.01,0.01,0.01"
POLICIES=("STARFIT" "RBF" "PiecewiseLinear")
RESERVOIRS=("beltzvilleCombined" "fewalter" "prompton")

# Define the number of DVs per policy
declare -A NUM_DVS
NUM_DVS["STARFIT"]=17
NUM_DVS["RBF"]=14
NUM_DVS["PiecewiseLinear"]=15

# Fixed number of objectives
NUM_OBJS=4

for POLICY in "${POLICIES[@]}"; do
    echo "🔁 Running diagnostics for policy: $POLICY"
    RUNTIME_DIR="outputs/Policy_${POLICY}/runtime"

    for RES in "${RESERVOIRS[@]}"; do
        DIR="${RUNTIME_DIR}/${RES}"
        if [ ! -d "$DIR" ]; then
            echo "⚠️  Skipping missing directory: $DIR"
            continue
        fi

        echo "📂 Processing: $POLICY – $RES"
        cd "$DIR"

        # Step 1: Merge and generate raw .ref file
        bash "$SCRIPT_DIR/1-moeaframework_merge_files.sh" $EPSILON
        bash "$SCRIPT_DIR/2-moeaframework_gen_refset.sh" $EPSILON

        # Step 4: Generate runtime metrics
        bash "$SCRIPT_DIR/3-moeaframework_gen_runtime.sh" $EPSILON

        cd - > /dev/null
    done
done

echo "✅ MOEA diagnostics complete for all policies and reservoirs."
