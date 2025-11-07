#!/bin/bash
#SBATCH --job-name=MOEAMulti
#SBATCH --output=./logs/MOEAMulti.out
#SBATCH --error=./logs/MOEAMulti.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH --mail-type=END
#SBATCH --mail-user=ms3654@cornell.edu

set -euo pipefail

# Always operate from the directory where you ran `sbatch`.
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

SCRIPT_DIR="$SLURM_SUBMIT_DIR"   # project root

# ===== Config =====
export EPSILON="0.01,0.01,0.01,0.01"
POLICIES=("STARFIT" "RBF" "PWL")
RESERVOIRS=("beltzvilleCombined" "fewalter" "prompton" "blueMarsh")
declare -A NUM_DVS=( ["STARFIT"]=17 ["RBF"]=14 ["PWL"]=15 )
export NUM_OBJS=6
export OUT_ROOT="outputs"

export SEED_FROM="${SEED_FROM:-1}"
export SEED_TO="${SEED_TO:-10}"

# Optional filters (uncomment to limit scope)
# export POLICIES_FILTER="Policy_*"                            # Glob for policy dirs under OUT_ROOT
# export RESERVOIRS_FILTER="*"                                 # Glob for reservoirs under runtime/refsets

# ===== CLI autodetect / override =====
CLI_ARG="${1:-}"
if [[ "$CLI_ARG" == "--cli" ]]; then
  CLI="${2:?usage: $0 [--cli /path/to/cli]}"; shift 2
else
  if   [[ -x "$SCRIPT_DIR/cli" ]]; then CLI="$SCRIPT_DIR/cli"
  elif [[ -x "$SCRIPT_DIR/MOEAFramework-5.0/cli" ]]; then CLI="$SCRIPT_DIR/MOEAFramework-5.0/cli"
  elif [[ -x "$SCRIPT_DIR/../MOEAFramework-5.0/cli" ]]; then CLI="$SCRIPT_DIR/../MOEAFramework-5.0/cli"
  else
    echo "ERROR: MOEAFramework cli not found. Pass --cli /path/to/cli" >&2
    exit 1
  fi
fi
[[ -x "$CLI" ]] || { echo "ERROR: CLI not executable: $CLI (chmod +x)"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found in PATH"; exit 1; }

echo "== MOEA Workflow (flat layout) =="
echo "PWD       : $PWD"
echo "CLI       : $CLI"
echo "OUT_ROOT  : $OUT_ROOT"
echo "EPSILON   : $EPSILON"
echo "NUM_OBJS  : $NUM_OBJS"
echo "SEEDS     : ${SEED_FROM}-${SEED_TO}"
echo "Policies  : ${POLICIES[*]}"
echo "Reservoirs: ${RESERVOIRS[*]}"
echo

# ===== Steps =====
echo ">> [1/4] runtime -> set"
bash "$SCRIPT_DIR/1-moeaframework_merge_files.sh" --cli "$CLI"
echo

echo ">> [2/4] append header -> *_header.set"
python3 "$SCRIPT_DIR/append_header.py" --outputs-root "$OUT_ROOT" --seed-from "$SEED_FROM" --seed-to "$SEED_TO"
echo

echo ">> [3/4] merge *_header.set -> <reservoir>.ref"
EPSILON="$EPSILON" DIMENSION="$NUM_OBJS" SEED_FROM="$SEED_FROM" SEED_TO="$SEED_TO" \
  bash "$SCRIPT_DIR/2-moeaframework_gen_refset.sh" --cli "$CLI"
echo

echo ">> [4/4] metrics vs <reservoir>.ref"
EPSILON="$EPSILON" SEED_FROM="$SEED_FROM" SEED_TO="$SEED_TO" \
  bash "$SCRIPT_DIR/3-moeaframework_gen_runtime.sh" --cli "$CLI"
echo

echo "== Done =="
