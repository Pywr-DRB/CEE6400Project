#!/usr/bin/env bash
# Step 1: Convert *.runtime -> *.set per policy/reservoir (seed-filtered).
set -euo pipefail

CLI_ARG="${1:-}"
if [[ "$CLI_ARG" == "--cli" ]]; then
  CLI="${2:?usage: $0 [--cli /path/to/cli]}"; shift 2
else
  SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
  if   [[ -x "$SCRIPT_DIR/cli" ]]; then CLI="$SCRIPT_DIR/cli"
  elif [[ -x "$SCRIPT_DIR/MOEAFramework-5.0/cli" ]]; then CLI="$SCRIPT_DIR/MOEAFramework-5.0/cli"
  elif [[ -x "$SCRIPT_DIR/../MOEAFramework-5.0/cli" ]]; then CLI="$SCRIPT_DIR/../MOEAFramework-5.0/cli"
  else echo "ERROR: MOEAFramework cli not found. Pass --cli /path/to/cli" >&2; exit 1; fi
fi
[[ -x "$CLI" ]] || { echo "ERROR: $CLI not executable (chmod +x)"; exit 1; }

OUT_ROOT="outputs"
SEED_FROM="${SEED_FROM:-1}"
SEED_TO="${SEED_TO:-10}"

echo ">> Using CLI: $CLI"
echo ">> Converting runtime -> set (seeds ${SEED_FROM}-${SEED_TO})..."

shopt -s nullglob
for policy_dir in "${OUT_ROOT}"/Policy_*; do
  [[ -d "$policy_dir/runtime" ]] || continue
  echo ">> Policy: $(basename "$policy_dir")"
  for reservoir_dir in "${policy_dir}/runtime"/*/; do
    reservoir="$(basename "$reservoir_dir")"
    out_dir="${policy_dir}/refsets/${reservoir}"
    mkdir -p "$out_dir"
    echo "   - Reservoir: ${reservoir}"
    for runtime_file in "${reservoir_dir}"/*.runtime; do
      base="$(basename "${runtime_file}" .runtime)"
      # Parse seed number from filename (expects ..._seed<NUM>_...)
      if [[ "$base" =~ _seed([0-9]+)([_.]|$) ]]; then
        seed="${BASH_REMATCH[1]}"
        if (( seed < SEED_FROM || seed > SEED_TO )); then
          continue
        fi
      else
        # If no seed tag, skip (or remove this 'continue' to include)
        continue
      fi
      out_file="${out_dir}/${base}.set"
      echo "     * ${base}.runtime -> ${base}.set"
      "$CLI" ResultFileConverter --input "${runtime_file}" --output "${out_file}"
    done
  done
done
echo ">> Done."
