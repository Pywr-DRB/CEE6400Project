#!/usr/bin/env bash
# 2-moeaframework_gen_refset.sh
# Step 10 + 11:
#  - *_header.set (islands) -> seed<SEED>.ref
#  - seed*.ref (all seeds)  -> all.ref per reservoir
#  - optional policy_all.ref across reservoirs
set -euo pipefail
shopt -s nullglob

########## CLI lookup ##########

CLI_ARG="${1:-}"
if [[ "$CLI_ARG" == "--cli" ]]; then
  CLI="${2:?usage: $0 [--cli /path/to/cli] [--no-policy-merge]}"
  shift 2
else
  SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
  if   [[ -x "$SCRIPT_DIR/cli" ]]; then CLI="$SCRIPT_DIR/cli"
  elif [[ -x "$SCRIPT_DIR/MOEAFramework-5.0/cli" ]]; then CLI="$SCRIPT_DIR/MOEAFramework-5.0/cli"
  elif [[ -x "$SCRIPT_DIR/../MOEAFramework-5.0/cli" ]]; then CLI="$SCRIPT_DIR/../MOEAFramework-5.0/cli"
  else
    echo "ERROR: Could not find MOEAFramework cli. Pass --cli /path/to/cli" >&2
    exit 1
  fi
fi

# Verify CLI is executable
if [[ ! -x "$CLI" ]]; then
  echo "Error: CLI at $CLI is not executable. Run:"
  echo "  chmod +x \"$CLI\""
  exit 1
fi

########## Config / continuity vars ##########
OUT_ROOT="outputs"

# Epsilon & objectives: 4 objectives → 4 eps
EPSILON="${EPSILON:-0.01,0.01,0.01,0.01}"
DIMENSION="${DIMENSION:-4}"

# Seed window 
NUM_SEEDS="${NUM_SEEDS:-10}"
SEED_FROM="${SEED_FROM:-1}"
SEED_TO="${SEED_TO:-$NUM_SEEDS}"
all_seeds="$(seq "$SEED_FROM" "$SEED_TO")"

# Not strictly used by the merger logic,
# but defined here to match previous scripts/vars)
setDir="./refsets"
runtimeDir="./runtime"
metricDir="./metrics"
refFile_name="${refFile_name:-custom_refset}"

# Optional flag to skip policy-wide merge
DO_POLICY_MERGE="${DO_POLICY_MERGE:-true}"
if [[ "${1:-}" == "--no-policy-merge" ]]; then
  DO_POLICY_MERGE=false
  shift
fi

echo ">> Using CLI: $CLI"
echo ">> EPSILON=$EPSILON  SEEDS=$SEED_FROM..$SEED_TO  POLICY_MERGE=$DO_POLICY_MERGE"

# Helper: Must get Problem from header or override; no fallback.
get_problem_for_policy () {
  local policy_dir="$1"
  if [[ -n "${PROBLEM_OVERRIDE:-}" ]]; then
    echo "$PROBLEM_OVERRIDE"; return
  fi
  local header_file="$policy_dir/1-header-file.txt"
  if [[ ! -f "$header_file" ]]; then
    echo "ERROR: Missing $header_file; set PROBLEM_OVERRIDE or add '# Problem=...' to header." >&2
    exit 1
  fi
  local prob
  prob="$(awk -F= '/^# *Problem=/{print $2; exit}' "$header_file" | tr -d '[:space:]')"
  if [[ -z "$prob" ]]; then
    echo "ERROR: Could not find '# Problem=' in $header_file; set PROBLEM_OVERRIDE." >&2
    exit 1
  fi
  echo "$prob"
}

# Helper: ensure a .set has a Version=5 header; if missing, prepend policy header
ensure_version5_header () {
  local set_file="$1"
  local header_file="$2"
  if ! grep -q '^# *Version=5' "$set_file"; then
    echo "     - Adding header to $(basename "$set_file")"
    local tmp
    tmp="$(mktemp)"
    {
      cat "$header_file"
      echo
      cat "$set_file"
    } > "$tmp"
    mv "$tmp" "$set_file"
  fi
}

############################################
# Step 10: Merge islands per seed -> seedX.ref
############################################
echo ">> Step 10: Merging islands per seed ..."
for policy_dir in "${OUT_ROOT}"/Policy_*; do
  refsets_root="${policy_dir}/refsets"
  [[ -d "$refsets_root" ]] || continue
  POLICY_NAME="$(basename "$policy_dir")"
  PROBLEM="$(get_problem_for_policy "$policy_dir")"
  HEADER_FILE="$policy_dir/1-header-file.txt"
  echo ">> Policy: $POLICY_NAME | Problem: $PROBLEM"

  for rdir in "${refsets_root}"/*/; do
    reservoir="$(basename "$rdir")"
    dest_dir="${policy_dir}/refsets_merged/${reservoir}"
    mkdir -p "$dest_dir"
    echo "   - Reservoir: ${reservoir}"

    for seed in $all_seeds; do
      # collect islands for this seed
      inputs=( "$rdir"/*_seed${seed}_*_header.set )
      if [[ ! -e "${inputs[0]:-}" ]]; then
        continue
      fi

      # Continuity: ensure each input has Version=5 header (safety net)
      for f in "${inputs[@]}"; do
        ensure_version5_header "$f" "$HEADER_FILE"
      done

      out_file="${dest_dir}/seed${seed}.ref"
      echo "     * Seed ${seed}: ${#inputs[@]} island(s) -> $(basename "$out_file")"

      if [[ -f "$out_file" ]]; then
        "$CLI" ResultFileMerger \
          --problem "$PROBLEM" \
          --epsilon "$EPSILON" \
          --output  "$out_file" \
          "${inputs[@]}" \
          --overwrite
      else
        "$CLI" ResultFileMerger \
          --problem "$PROBLEM" \
          --epsilon "$EPSILON" \
          --output  "$out_file" \
          "${inputs[@]}"
      fi
    done
  done
done

####################################################
# Step 11: Merge seeds per reservoir -> all.ref
#          (and optional policy_all.ref per policy)
####################################################
echo ">> Step 11: Merging all seeds per reservoir -> all.ref ..."
for policy_dir in "${OUT_ROOT}"/Policy_*; do
  merged_root="${policy_dir}/refsets_merged"
  [[ -d "$merged_root" ]] || continue
  POLICY_NAME="$(basename "$policy_dir")"
  PROBLEM="$(get_problem_for_policy "$policy_dir")"
  echo ">> Policy: $POLICY_NAME | Problem: $PROBLEM"

  # Per-reservoir merge to all.ref
  for rdir in "${merged_root}"/*/; do
    reservoir="$(basename "$rdir")"
    seed_refs=( "$rdir"/seed*.ref )
    if [[ ! -e "${seed_refs[0]:-}" ]]; then
      echo "   - ${reservoir}: (no seed*.ref)"
      continue
    fi

    out_dir="${policy_dir}/refsets_master/${reservoir}"
    mkdir -p "$out_dir"
    out_file="${out_dir}/all.ref"
    echo "   - Reservoir: ${reservoir} (seeds: ${#seed_refs[@]}) -> $(basename "$out_file")"

    if [[ -f "$out_file" ]]; then
      "$CLI" ResultFileMerger \
        --problem "$PROBLEM" \
        --epsilon "$EPSILON" \
        --output  "$out_file" \
        "${seed_refs[@]}" \
        --overwrite
    else
      "$CLI" ResultFileMerger \
        --problem "$PROBLEM" \
        --epsilon "$EPSILON" \
        --output  "$out_file" \
        "${seed_refs[@]}"
    fi
  done

  # Optional policy-wide merge across reservoirs
  if [[ "$DO_POLICY_MERGE" == "true" ]]; then
    all_refs=( "$policy_dir"/refsets_master/*/all.ref )
    if [[ -e "${all_refs[0]:-}" ]]; then
      out_file="${policy_dir}/refsets_master/policy_all.ref"
      echo "   * Policy-wide merge across ${#all_refs[@]} reservoirs -> $(basename "$out_file")"
      if [[ -f "$out_file" ]]; then
        "$CLI" ResultFileMerger \
          --problem "$PROBLEM" \
          --epsilon "$EPSILON" \
          --output  "$out_file" \
          "${all_refs[@]}" \
          --overwrite
      else
        "$CLI" ResultFileMerger \
          --problem "$PROBLEM" \
          --epsilon "$EPSILON" \
          --output  "$out_file" \
          "${all_refs[@]}"
      fi
    else
      echo "   * No per-reservoir all.ref found; skipping policy-wide merge."
    fi
  fi
done

echo ">> Done."
echo "   Per-seed refs:       outputs/Policy_*/refsets_merged/<reservoir>/seed<SEED>.ref"
echo "   Per-reservoir all:   outputs/Policy_*/refsets_master/<reservoir>/all.ref"
echo "   Policy-wide master:  outputs/Policy_*/refsets_master/policy_all.ref (if enabled)"
