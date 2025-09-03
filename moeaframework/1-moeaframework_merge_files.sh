#!/usr/bin/env bash
set -euo pipefail

# Optional override: allow `./script.sh --cli /path/to/cli`
CLI_ARG="${1:-}"
if [[ "$CLI_ARG" == "--cli" ]]; then
  CLI="${2:?usage: $0 [--cli /path/to/cli]}"
  shift 2
else
  # Auto-detect cli relative to this script
  SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
  if   [[ -x "$SCRIPT_DIR/cli" ]]; then CLI="$SCRIPT_DIR/cli"
  elif [[ -x "$SCRIPT_DIR/MOEAFramework-5.0/cli" ]]; then CLI="$SCRIPT_DIR/MOEAFramework-5.0/cli"
  elif [[ -x "$SCRIPT_DIR/../MOEAFramework-5.0/cli" ]]; then CLI="$SCRIPT_DIR/../MOEAFramework-5.0/cli"
  else
    echo "ERROR: Could not find MOEAFramework cli. Pass --cli /path/to/cli" >&2
    exit 1
  fi
fi

OUT_ROOT="outputs"

echo ">> Using CLI: $CLI"
echo ">> Converting runtime -> set for all policies/reservoirs..."

shopt -s nullglob
for policy_dir in "${OUT_ROOT}"/Policy_*; do
  [[ -d "$policy_dir/runtime" ]] || continue

  echo ">> Policy: $(basename "$policy_dir")"

  # Create a sibling refsets folder mirroring runtime/<reservoir> subfolders
  for reservoir_dir in "${policy_dir}/runtime"/*/; do
    reservoir="$(basename "$reservoir_dir")"
    refset_dir="${policy_dir}/refsets/${reservoir}"
    mkdir -p "$refset_dir"

    echo "   - Reservoir: ${reservoir}"

    for runtime_file in "${reservoir_dir}"/*.runtime; do
      base="$(basename "${runtime_file}" .runtime)"
      out_file="${refset_dir}/${base}.set"

      echo "     * ${base}.runtime -> ${base}.set"
      "${CLI}" ResultFileConverter \
        --input  "${runtime_file}" \
        --output "${out_file}"
    done
  done
done

echo ">> Done."
