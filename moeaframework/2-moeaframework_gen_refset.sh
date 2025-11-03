#!/usr/bin/env bash
# Builds references in two stages:
#   (A) Per seed: merge *_header.set across islands -> seed<seed>.ref
#   (B) Global: merge seed*.ref -> <reservoir>.ref
set -euo pipefail
shopt -s nullglob

# ===== CLI autodetect / override =====
CLI_ARG="${1:-}"
if [[ "$CLI_ARG" == "--cli" ]]; then
  CLI="${2:?usage: $0 [--cli /path/to/cli]}"; shift 2
else
  if   [[ -x "./cli" ]]; then CLI="./cli"
  elif [[ -x "./MOEAFramework-5.0/cli" ]]; then CLI="./MOEAFramework-5.0/cli"
  elif [[ -x "../MOEAFramework-5.0/cli" ]]; then CLI="../MOEAFramework-5.0/cli"
  else echo "ERROR: MOEAFramework cli not found. Pass --cli /path/to/cli" >&2; exit 1; fi
fi
[[ -x "$CLI" ]] || { echo "ERROR: $CLI not executable (chmod +x)"; exit 1; }

# ===== Config =====
OUT_ROOT="${OUT_ROOT:-outputs}"
EPSILON="${EPSILON:-0.01,0.01,0.01,0.01}"  # must match NO
SEED_FROM="${SEED_FROM:-1}"
SEED_TO="${SEED_TO:-10}"

eps_count() { awk -F, '{print NF}' <<<"$EPSILON"; }

read_meta() {
  # $1 = file
  # prints: Problem NO NV NC Version
  awk -F= '
    BEGIN{P=NO=NV=NC=V=""}
    /^# *Version=/             {gsub(/^[# ]*Version=/,""); gsub(/ /,""); V=$0}
    /^# *Problem=/             {gsub(/^[# ]*Problem=/,""); gsub(/ /,""); P=$0}
    /^# *NumberOfObjectives=/  {gsub(/^[# ]*NumberOfObjectives=/,""); gsub(/ /,""); NO=$0}
    /^# *NumberOfVariables=/   {gsub(/^[# ]*NumberOfVariables=/,""); gsub(/ /,""); NV=$0}
    /^# *NumberOfConstraints=/ {gsub(/^[# ]*NumberOfConstraints=/,""); gsub(/ /,""); NC=$0}
    END{print P,NO,NV,NC,V}
  ' "$1"
}

echo ">> [3/4] Build references (reading metadata from *_header.set)"
for policy_dir in "${OUT_ROOT}"/Policy_*; do
  ref_root="$policy_dir/refsets"
  [[ -d "$ref_root" ]] || continue
  echo ">> Policy: $(basename "$policy_dir")"

  for rdir in "$ref_root"/*/; do
    [[ -d "$rdir" ]] || continue
    reservoir="$(basename "$rdir")"
    echo "   - Reservoir: $reservoir"

    # Quick inventory of inputs
    hdr_sets=( "$rdir"/*_header.set )
    if [[ ${#hdr_sets[@]} -eq 0 || ! -e "${hdr_sets[0]}" ]]; then
      echo "     (no *_header.set files; did you run append_header.py?)"
      continue
    fi

    # Read canonical metadata from the first header.set
    read P0 NO0 NV0 NC0 VER0 < <(read_meta "${hdr_sets[0]}")
    if [[ "$VER0" != "5" ]]; then
      echo "     ERROR: Expected # Version=5 in ${hdr_sets[0]} (got $VER0)"; continue
    fi
    # Validate all files match this metadata
    mixed=0
    for f in "${hdr_sets[@]}"; do
      read P NO NV NC VER < <(read_meta "$f")
      if [[ "$VER" != "5" || "$P" != "$P0" || "$NO" != "$NO0" || "$NV" != "$NV0" || "$NC" != "$NC0" ]]; then
        echo "     ! META MISMATCH in $(basename "$f")"
        echo "       got: Problem=$P NO=$NO NV=$NV NC=$NC Ver=$VER"
        echo "       exp: Problem=$P0 NO=$NO0 NV=$NV0 NC=$NC0 Ver=5"
        mixed=1
      fi
    done
    if (( mixed )); then
      echo "     !! aborting merge due to mixed metadata"
      continue
    fi

    # Check epsilon length vs NO
    if [[ "$(eps_count)" -ne "$NO0" ]]; then
      echo "     ERROR: EPSILON has $(eps_count) entries but NO=$NO0; fix EPSILON and retry."
      continue
    fi

    # (A) per-seed merges
    for ((seed=SEED_FROM; seed<=SEED_TO; seed++)); do
      mapfile -t per_island < <(ls "$rdir"/*_seed${seed}_*_header.set 2>/dev/null || true)
      [[ "${#per_island[@]}" -gt 0 ]] || continue
      out_seed_ref="$rdir/seed${seed}.ref"
      echo "     * seed ${seed}: merge ${#per_island[@]} islands -> $(basename "$out_seed_ref")"
      if [[ -f "$out_seed_ref" ]]; then
        # "$CLI" ResultFileMerger --problem "$P0" --epsilon "$EPSILON" --output "$out_seed_ref" "${per_island[@]}" --overwrite
        "$CLI" ResultFileMerger --problem "$P0" --epsilon "$EPSILON" \
          --overwrite --output "$out_seed_ref" "${per_island[@]}"
      else
        "$CLI" ResultFileMerger --problem "$P0" --epsilon "$EPSILON" --output "$out_seed_ref" "${per_island[@]}"
      fi
    done

    # (B) global merge across seeds
    mapfile -t seed_refs < <(ls "$rdir"/seed*.ref 2>/dev/null || true)
    if [[ "${#seed_refs[@]}" -eq 0 ]]; then
      echo "     (no seed refs found; skipping global merge)"
      continue
    fi
    out_global="$rdir/${reservoir}.ref"
    echo "     * global: merge ${#seed_refs[@]} seed refs -> $(basename "$out_global")"
    if [[ -f "$out_global" ]]; then
      # "$CLI" ResultFileMerger --problem "$P0" --epsilon "$EPSILON" --output "$out_global" "${seed_refs[@]}" --overwrite
      "$CLI" ResultFileMerger --problem "$P0" --epsilon "$EPSILON" \
        --overwrite --output "$out_global" "${seed_refs[@]}"
    else
      "$CLI" ResultFileMerger --problem "$P0" --epsilon "$EPSILON" --output "$out_global" "${seed_refs[@]}"
    fi
  done
done
echo ">> Done."