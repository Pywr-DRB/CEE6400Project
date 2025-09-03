#!/usr/bin/env bash
set -euo pipefail

# --- args ---
SRC="${1:-$(pwd)/outputs}"                # where MMBorg_* were written
MOEA_DIR="${2:-$(pwd)/moeaframework}"     # your moeaframework folder
MODE="${3:-copy}"                         # copy|move

if [[ ! -d "$SRC" ]]; then echo "SRC not found: $SRC"; exit 1; fi
mkdir -p "$MOEA_DIR/outputs" "$MOEA_DIR/figures"

echo "[1/4] Syncing outputs → $MOEA_DIR/outputs (mode=$MODE)"
if [[ "$MODE" == "move" ]]; then
  rsync -av --progress --remove-source-files "$SRC"/ "$MOEA_DIR/outputs"/
  # cleanup any empty dirs after move
  find "$SRC" -type d -empty -delete || true
else
  rsync -av --progress "$SRC"/ "$MOEA_DIR/outputs"/
fi

echo "[2/4] Organizing into Policy_<POLICY>/runtime/<RESERVOIR>/"
POLICIES=(STARFIT RBF PiecewiseLinear)
RESERVOIRS=(beltzvilleCombined fewalter prompton)
OUT="$MOEA_DIR/outputs"

# make folders
for p in "${POLICIES[@]}"; do
  for r in "${RESERVOIRS[@]}"; do
    mkdir -p "$OUT/Policy_${p}/runtime/${r}"
  done
done

shopt -s nullglob
# move any islands/seed artifacts (.csv .set .info .runtime etc.)
for f in "$OUT"/MMBorg_*; do
  base="$(basename "$f")"
  moved=0
  for p in "${POLICIES[@]}"; do
    for r in "${RESERVOIRS[@]}"; do
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

echo "[3/4] (Optional) Generate pooled borg.ref with your MOEA scripts if needed"
pushd "$MOEA_DIR" >/dev/null
if [[ -x ./1-moeaframework_merge_files.sh && -x ./2-moeaframework_gen_refset.sh && -x ./3-moeaframework_gen_runtime.sh ]]; then
  # only run if borg.ref appears missing anywhere
  need_run=0
  while IFS= read -r -d '' rt; do
    if [[ ! -f "$rt/borg.ref" ]]; then need_run=1; break; fi
  done < <(find "$MOEA_DIR/outputs" -type d -path "*/runtime/*" -print0)

  if [[ $need_run -eq 1 ]]; then
    echo "  Running MOEA steps 1→3 to produce borg.ref files…"
    bash ./1-moeaframework_merge_files.sh
    bash ./2-moeaframework_gen_refset.sh
    bash ./3-moeaframework_gen_runtime.sh
  else
    echo "  borg.ref files already present—skipping MOEA steps."
  fi
else
  echo "  MOEA scripts not executable or missing—skipping."
fi
popd >/dev/null

echo "[4/4] Plotting from pooled borg.ref and saving time-series CSVs"
# write a tiny marker env so the plotter knows where to work
export MOEA_ROOT="$MOEA_DIR"
python "$MOEA_DIR/plot_best_from_borgref.py"

echo "Done. Figures and CSVs in: $MOEA_DIR/figures/"
