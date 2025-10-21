#!/usr/bin/env bash
# sync_and_organize_moea.sh
# 1) copy/move ./outputs → ./moeaframework/outputs
# 2) organize into Policy_<POLICY>/runtime/<RESERVOIR>/
set -euo pipefail

# -------- Args (with defaults) --------
SRC="${1:-$(pwd)/outputs}"                # where MMBorg_*/*MWBorg_* were written
MOEA_DIR="${2:-$(pwd)/moeaframework}"     # destination folder containing 'outputs/'
MODE="${3:-copy}"                         # copy|move

# Optional overrides via env:
#   POLICIES="STARFIT RBF PWL" RESERVOIRS="beltzvilleCombined fewalter prompton"
POLICIES_STR="${POLICIES:-STARFIT RBF PWL}"
RESERVOIRS_STR="${RESERVOIRS:-beltzvilleCombined fewalter prompton blueMarsh}"

# -------- Guards --------
if [[ ! -d "$SRC" ]]; then echo "SRC not found: $SRC"; exit 1; fi
mkdir -p "$MOEA_DIR/outputs"

# -------- 1) Sync/Move --------
echo "[1/2] Syncing $SRC → $MOEA_DIR/outputs (mode=$MODE)"
if [[ "$MODE" == "move" ]]; then
  rsync -av --progress --remove-source-files "$SRC"/ "$MOEA_DIR/outputs"/
  # clean up emptied dirs
  find "$SRC" -type d -empty -delete || true
else
  rsync -av --progress "$SRC"/ "$MOEA_DIR/outputs"/
fi

# -------- 2) Organize --------
echo "[2/2] Organizing into Policy_<POLICY>/runtime/<RESERVOIR>/"
read -r -a POLICIES_ARR <<< "$POLICIES_STR"
read -r -a RESERVOIRS_ARR <<< "$RESERVOIRS_STR"
OUT="$MOEA_DIR/outputs"

# make folders
for p in "${POLICIES_ARR[@]}"; do
  for r in "${RESERVOIRS_ARR[@]}"; do
    mkdir -p "$OUT/Policy_${p}/runtime/${r}"
  done
done

# move all result artifacts into their buckets
shopt -s nullglob
for f in "$OUT"/MMBorg_* "$OUT"/MWBorg_*; do
  [[ -e "$f" ]] || continue
  base="$(basename "$f")"
  moved=0
  for p in "${POLICIES_ARR[@]}"; do
    for r in "${RESERVOIRS_ARR[@]}"; do
      if [[ "$base" == *"_${p}_${r}"* ]]; then
        tgt="$OUT/Policy_${p}/runtime/${r}"
        echo "  → $base → $tgt/"
        mv "$f" "$tgt"/
        moved=1; break 2
      fi
    done
  done
  if [[ $moved -eq 0 ]]; then
    echo "  ! unmatched: $base (left in $OUT)"
  fi
done

echo "Done. Files organized under: $OUT/Policy_<POLICY>/runtime/<RESERVOIR>/"
