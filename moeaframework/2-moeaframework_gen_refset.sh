#!/usr/bin/env bash
# Step 3: Merge *_header.set -> <reservoir>.ref (seed-filtered).
set -euo pipefail
shopt -s nullglob

# CLI lookup
CLI_ARG="${1:-}"
if [[ "$CLI_ARG" == "--cli" ]]; then
  CLI="${2:?usage: $0 [--cli /path/to/cli]}"; shift 2
else
  SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
  if   [[ -x "$SCRIPT_DIR/cli" ]] ; then CLI="$SCRIPT_DIR/cli"
  elif [[ -x "$SCRIPT_DIR/MOEAFramework-5.0/cli" ]]; then CLI="$SCRIPT_DIR/MOEAFramework-5.0/cli"
  elif [[ -x "$SCRIPT_DIR/../MOEAFramework-5.0/cli" ]]; then CLI="$SCRIPT_DIR/../MOEAFramework-5.0/cli"
  else echo "ERROR: MOEAFramework cli not found. Pass --cli /path/to/cli" >&2; exit 1; fi
fi
[[ -x "$CLI" ]] || { echo "ERROR: $CLI not executable (chmod +x)"; exit 1; }

OUT_ROOT="outputs"
EPSILON="${EPSILON:-0.01,0.01,0.01,0.01,0.01,0.01}"
DIMENSION="${DIMENSION:-6}"
SEED_FROM="${SEED_FROM:-1}"
SEED_TO="${SEED_TO:-10}"

get_problem_for_policy () {
  local policy_dir="$1"
  if [[ -n "${PROBLEM_OVERRIDE:-}" ]]; then echo "$PROBLEM_OVERRIDE"; return; fi
  local header_file="$policy_dir/1-header-file.txt"
  [[ -f "$header_file" ]] || { echo "ERROR: $header_file missing" >&2; exit 1; }
  local prob
  prob="$(awk -F= '/^# *Problem=/{print $2; exit}' "$header_file" | tr -d '[:space:]')"
  [[ -n "$prob" ]] || { echo "ERROR: add '# Problem=...' to $header_file or set PROBLEM_OVERRIDE" >&2; exit 1; }
  echo "$prob"
}

ensure_version5_header () {
  local set_file="$1"; local header_file="$2"
  if ! grep -q '^# *Version=5' "$set_file"; then
    local tmp; tmp="$(mktemp)"; { cat "$header_file"; echo; cat "$set_file"; } > "$tmp"; mv "$tmp" "$set_file"
  fi
}

count_commas() { awk -F, '{print NF}' <<<"$1"; }
eps_count="$(count_commas "$EPSILON")"
if [[ "$eps_count" -ne "$DIMENSION" ]]; then
  echo "WARNING: EPSILON has $eps_count entries; DIMENSION=$DIMENSION" >&2
fi

echo ">> Using CLI: $CLI"
echo ">> EPSILON=$EPSILON  DIMENSION=$DIMENSION  SEEDS=${SEED_FROM}-${SEED_TO}"

for policy_dir in "${OUT_ROOT}"/Policy_*; do
  refsets_root="${policy_dir}/refsets"
  [[ -d "$refsets_root" ]] || continue
  POLICY="$(basename "$policy_dir")"
  PROBLEM="$(get_problem_for_policy "$policy_dir")"
  HEADER_FILE="$policy_dir/1-header-file.txt"
  echo ">> Policy: $POLICY | Problem: $PROBLEM"

  for rdir in "${refsets_root}"/*/; do
    reservoir="$(basename "$rdir")"
    echo "   - Reservoir: ${reservoir}"

    # Prefer *_header.set; else *.set
    mapfile -t candidates < <(ls "$rdir"/*_header.set 2>/dev/null || true)
    if [[ "${#candidates[@]}" -eq 0 ]]; then
      mapfile -t candidates < <(ls "$rdir"/*.set 2>/dev/null || true)
    fi
    if [[ "${#candidates[@]}" -eq 0 ]]; then
      echo "     (no .set files)"; continue
    fi

    # Filter by seed window
    inputs=()
    for f in "${candidates[@]}"; do
      fname="$(basename "$f")"
      if [[ "$fname" =~ _seed([0-9]+)([_.]|$) ]]; then
        seed="${BASH_REMATCH[1]}"
        if (( seed < SEED_FROM || seed > SEED_TO )); then
          continue
        fi
      else
        # No seed tag? skip to be strict.
        continue
      fi
      inputs+=("$f")
    done

    if [[ "${#inputs[@]}" -eq 0 ]]; then
      echo "     (no files in seed window ${SEED_FROM}-${SEED_TO})"; continue
    fi

    for f in "${inputs[@]}"; do ensure_version5_header "$f" "$HEADER_FILE"; done

    out_file="${rdir}/${reservoir}.ref"
    echo "     * Merge seeds ${SEED_FROM}-${SEED_TO} -> $(basename "$out_file")  (n=${#inputs[@]})"
    if [[ -f "$out_file" ]]; then
      "$CLI" ResultFileMerger --problem "$PROBLEM" --epsilon "$EPSILON" --output "$out_file" "${inputs[@]}" --overwrite
    else
      "$CLI" ResultFileMerger --problem "$PROBLEM" --epsilon "$EPSILON" --output "$out_file" "${inputs[@]}"
    fi
  done
done

echo ">> Done. <reservoir>.ref written per reservoir."
