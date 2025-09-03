#!/usr/bin/env bash
# 3-moeaframework_gen_metrics.sh
# Generate runtime metrics for each policy / reservoir / runtime (*.runtime).
# Uses the reservoir-level merged reference set:
#   outputs/Policy_*/refsets_master/<reservoir>/all.ref
# Falls back to policy_all.ref if reservoir ref is missing.

set -euo pipefail
shopt -s nullglob

########## CLI lookup ##########
CLI_ARG="${1:-}"
if [[ "$CLI_ARG" == "--cli" ]]; then
  CLI="${2:?usage: $0 [--cli /path/to/cli]}"
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

# Executable check 
if [[ ! -x "$CLI" ]]; then
  echo "Error: CLI at $CLI is not executable. Run:"
  echo "  chmod 775 \"$CLI\""
  exit 1
fi

########## Config ##########
OUT_ROOT="outputs"

# 4 objectives => 4 epsilons (override by exporting EPSILON="..."):
EPSILON="${EPSILON:-0.01,0.01,0.01,0.01}"

# Optional seed/island windows (not strictly needed since we glob)
NUM_SEEDS="${NUM_SEEDS:-10}"
NUM_MASTERS="${NUM_MASTERS:-4}"   # not used; we glob *.runtime

# Chooses which reference to use:
#   reservoir -> use refsets_master/<reservoir>/all.ref (default)
#   policy    -> use refsets_master/policy_all.ref
REF_SOURCE="${REF_SOURCE:-reservoir}"

echo ">> Using CLI: $CLI"
echo ">> EPSILON=$EPSILON  REF_SOURCE=$REF_SOURCE"

# Helper: problem per policy from the header (must exist; no fallback)
get_problem_for_policy () {
  local policy_dir="$1"
  if [[ -n "${PROBLEM_OVERRIDE:-}" ]]; then
    echo "$PROBLEM_OVERRIDE"; return
  fi
  local header="$policy_dir/1-header-file.txt"
  if [[ ! -f "$header" ]]; then
    echo "ERROR: Missing $header" >&2; exit 1
  fi
  local prob
  prob="$(awk -F= '/^# *Problem=/{print $2; exit}' "$header" | tr -d '[:space:]')"
  if [[ -z "$prob" ]]; then
    echo "ERROR: Could not find '# Problem=' in $header" >&2; exit 1
  fi
  echo "$prob"
}

# Header safety: prepend policy header if runtime is missing '# Version=5'
ensure_version5_header () {
  local runtime_file="$1"
  local header_file="$2"
  if ! grep -q '^# *Version=5' "$runtime_file"; then
    echo "     - Adding header to $(basename "$runtime_file")"
    local tmp; tmp="$(mktemp)"
    {
      cat "$header_file"
      echo
      cat "$runtime_file"
    } > "$tmp"
    mv "$tmp" "$runtime_file"
  fi
}

# Pick reference file for a reservoir (or policy)
pick_reference () {
  local policy_dir="$1"
  local reservoir="$2"
  local ref_file=""
  case "$REF_SOURCE" in
    reservoir)
      ref_file="$policy_dir/refsets_master/$reservoir/all.ref"
      if [[ ! -f "$ref_file" ]]; then
        # fallback:
        ref_file="$policy_dir/refsets_master/policy_all.ref"
      fi
      ;;
    policy)
      ref_file="$policy_dir/refsets_master/policy_all.ref"
      ;;
    *)
      echo "ERROR: Unknown REF_SOURCE=$REF_SOURCE (use reservoir|policy)" >&2; exit 1;;
  esac
  if [[ ! -f "$ref_file" ]]; then
    echo "ERROR: Reference file not found for $policy_dir ($reservoir). Looked for:" >&2
    echo "       - $policy_dir/refsets_master/$reservoir/all.ref" >&2
    echo "       - $policy_dir/refsets_master/policy_all.ref" >&2
    exit 1
  fi
  echo "$ref_file"
}

############################################
# Walk policies / reservoirs / runtimes
############################################
for policy_dir in "${OUT_ROOT}"/Policy_*; do
  [[ -d "$policy_dir/runtime" ]] || continue
  POLICY_NAME="$(basename "$policy_dir")"
  PROBLEM="$(get_problem_for_policy "$policy_dir")"
  HEADER_FILE="$policy_dir/1-header-file.txt"

  echo ">> Policy: $POLICY_NAME | Problem: $PROBLEM"

  for rdir in "$policy_dir/runtime"/*/; do
    reservoir="$(basename "$rdir")"
    ref_file="$(pick_reference "$policy_dir" "$reservoir")"

    metrics_dir="$policy_dir/metrics/$reservoir"
    mkdir -p "$metrics_dir"

    echo "   - Reservoir: $reservoir"
    echo "     Reference: $(realpath "$ref_file" 2>/dev/null || echo "$ref_file")"

    for runtime_file in "$rdir"/*.runtime; do
      [[ -f "$runtime_file" ]] || continue
      base="$(basename "$runtime_file" .runtime)"
      metric_file="$metrics_dir/${base}.metric"

      # Ensure header on input runtime
      ensure_version5_header "$runtime_file" "$HEADER_FILE"

      # Run evaluator (conditional overwrite)
      if [[ -f "$metric_file" ]]; then
        echo "     * $(basename "$runtime_file") -> $(basename "$metric_file") (overwrite)"
        "$CLI" MetricsEvaluator \
          --problem "$PROBLEM" \
          --epsilon "$EPSILON" \
          --input   "$runtime_file" \
          --output  "$metric_file" \
          --reference "$ref_file" \
          --overwrite
      else
        echo "     * $(basename "$runtime_file") -> $(basename "$metric_file")"
        "$CLI" MetricsEvaluator \
          --problem "$PROBLEM" \
          --epsilon "$EPSILON" \
          --input   "$runtime_file" \
          --output  "$metric_file" \
          --reference "$ref_file"
      fi

      # Strip leading '#' in the first line (nice-to-have)
      if [[ -f "$metric_file" ]]; then
        sed -i '1s/^#\s*//' "$metric_file"
      fi
    done
  done
done

echo ">> Done."
echo "Metrics written under: outputs/Policy_*/metrics/<reservoir>/*.metric"
