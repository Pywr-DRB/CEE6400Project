#!/usr/bin/env bash
# Step 4: Compute metrics (*.metric) using <reservoir>.ref (seed-filtered).
set -euo pipefail
shopt -s nullglob

CLI_ARG="${1:-}"
if [[ "$CLI_ARG" == "--cli" ]]; then
  CLI="${2:?usage: $0 [--cli /path/to/cli]}"; shift 2
else
  SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
  if   [[ -x "$SCRIPT_DIR/cli" ]]; then CLI="$SCRIPT_DIR/cli"
  elif [[ -x "$SCRIPT_DIR/MOEAFramework-5.0/cli" ]]; then CLI="$SCRIPT_DIR/MOEAFramework-5.0/cli"
  elif [[ -x "$SCRIPT_DIR/../MOEAFramework-5.0/cli" ]]; then CLI="$SCRIPT_DIR/../MOEAFramework-5.0/cli"
  else echo "ERROR: Could not find MOEAFramework cli. Pass --cli /path/to/cli" >&2; exit 1; fi
fi
[[ -x "$CLI" ]] || { echo "ERROR: $CLI not executable (chmod +x)"; exit 1; }

OUT_ROOT="outputs"
EPSILON="${EPSILON:-0.01,0.01,0.01,0.01,0.01,0.01}"
SEED_FROM="${SEED_FROM:-1}"
SEED_TO="${SEED_TO:-10}"

get_problem_for_policy () {
  local policy_dir="$1"
  if [[ -n "${PROBLEM_OVERRIDE:-}" ]]; then echo "$PROBLEM_OVERRIDE"; return; fi
  local header="$policy_dir/1-header-file.txt"
  [[ -f "$header" ]] || { echo "ERROR: Missing $header" >&2; exit 1; }
  local prob; prob="$(awk -F= '/^# *Problem=/{print $2; exit}' "$header" | tr -d '[:space:]')"
  [[ -n "$prob" ]] || { echo "ERROR: add '# Problem=...' to $header or set PROBLEM_OVERRIDE" >&2; exit 1; }
  echo "$prob"
}

ensure_version5_header () {
  local runtime_file="$1"; local header_file="$2"
  if ! grep -q '^# *Version=5' "$runtime_file"; then
    local tmp; tmp="$(mktemp)"; { cat "$header_file"; echo; cat "$runtime_file"; } > "$tmp"; mv "$tmp" "$runtime_file"
  fi
}

pick_reference_flat () {
  local policy_dir="$1"; local reservoir="$2"
  local ref_file="$policy_dir/refsets/$reservoir/$reservoir.ref"
  [[ -f "$ref_file" ]] || { echo "ERROR: Missing reservoir ref: $ref_file" >&2; exit 1; }
  echo "$ref_file"
}

echo ">> Using CLI: $CLI"
echo ">> EPSILON=$EPSILON  SEEDS=${SEED_FROM}-${SEED_TO}"

for policy_dir in "${OUT_ROOT}"/Policy_*; do
  [[ -d "$policy_dir/runtime" ]] || continue
  POLICY_NAME="$(basename "$policy_dir")"
  PROBLEM="$(get_problem_for_policy "$policy_dir")"
  HEADER_FILE="$policy_dir/1-header-file.txt"
  echo ">> Policy: $POLICY_NAME | Problem: $PROBLEM"

  for rdir in "$policy_dir/runtime"/*/; do
    reservoir="$(basename "$rdir")"
    ref_file="$(pick_reference_flat "$policy_dir" "$reservoir")"

    metrics_dir="$policy_dir/metrics/$reservoir"
    mkdir -p "$metrics_dir"

    echo "   - Reservoir: $reservoir"
    echo "     Reference: $(realpath "$ref_file" 2>/dev/null || echo "$ref_file")"

    for runtime_file in "$rdir"/*.runtime; do
      [[ -f "$runtime_file" ]] || continue
      base="$(basename "$runtime_file" .runtime)"
      # Seed filter
      if [[ "$base" =~ _seed([0-9]+)([_.]|$) ]]; then
        seed="${BASH_REMATCH[1]}"
        if (( seed < SEED_FROM || seed > SEED_TO )); then
          continue
        fi
      else
        continue
      fi

      metric_file="$metrics_dir/${base}.metric"
      ensure_version5_header "$runtime_file" "$HEADER_FILE"

      if [[ -f "$metric_file" ]]; then
        echo "     * $(basename "$runtime_file") -> $(basename "$metric_file") (overwrite)"
        "$CLI" MetricsEvaluator --problem "$PROBLEM" --epsilon "$EPSILON" \
          --input "$runtime_file" --output "$metric_file" --reference "$ref_file" --overwrite
      else
        echo "     * $(basename "$runtime_file") -> $(basename "$metric_file")"
        "$CLI" MetricsEvaluator --problem "$PROBLEM" --epsilon "$EPSILON" \
          --input "$runtime_file" --output "$metric_file" --reference "$ref_file"
      fi
      # Optional: tidy leading '#'
      [[ -f "$metric_file" ]] && sed -i '1s/^#\s*//' "$metric_file" || true
    done
  done
done

echo ">> Done. Metrics in: outputs/Policy_*/metrics/<reservoir>/*.metric"
