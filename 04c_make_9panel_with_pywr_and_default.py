#!/usr/bin/env python3
# 04c_make_9panel_with_pywr_and_default.py
"""
Selections → Independent sim + Pywr-DRB Parametric overlays + Pywr-DRB Default overlays → 9-panel

What you get on each 9-panel (per pick):
  • Sim:     Independent Reservoir run (primary series)
  • Pywr (Param): Pywr-DRB with inline Parametric policy (overlay)
  • Pywr (Default): your cached default Pywr-DRB time series from CSV (overlay)
  • Obs:     Observed release/storage if available

Assumptions:
  - Your exported default CSVs live in DRB_DEFAULT_SERIES (env) or ./pywr_data/_default_series/
  - Filenames: <reservoir>_default_{release,storage}.csv with columns: date,value
  - Your repo layout has top-level dirs “Policy_Optimization” and “Pywr-DRB”

Example:
  python 04c_make_9panel_with_pywr_and_default.py \
      --reservoirs fewalter beltzvilleCombined \
      --policies STARFIT RBF \
      --val-start 2019-01-01 --val-end 2024-12-31
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Path bootstrap (no reliance on methods.config) --------------------
def _find_up(name: str, start: Path | None = None) -> Path:
    p = Path.cwd() if start is None else Path(start).resolve()
    for q in [p, *p.parents]:
        cand = q / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find '{name}' above {p}")

PO_REPO       = _find_up("CEE6400Project")
PYWR_DRB_REPO = _find_up("Release_Policy_DRB")
PYWR_SRC      = PYWR_DRB_REPO / "src"

# ensure import paths (front of sys.path, no dups)
for p in [str(PYWR_SRC), str(PO_REPO)]:
    if p in sys.path:
        sys.path.remove(p)
for p in [str(PYWR_SRC), str(PO_REPO)]:
    sys.path.insert(0, p)

# -------------------- Project imports --------------------
import pywrdrb  # editable install in PYWR_SRC
from methods.config import (
    NFE, SEED, ISLANDS,
    OBJ_LABELS, OBJ_FILTER_BOUNDS,
    reservoir_options as RESERVOIR_NAMES,
    policy_type_options as POLICY_TYPES,
    OUTPUT_DIR as CFG_OUTPUT_DIR,
    FIG_DIR as CFG_FIG_DIR,
    PROCESSED_DATA_DIR,
    reservoir_capacity,
)
from methods.load.results import load_results
from methods.load.observations import get_observational_training_data
from methods.reservoir.model import Reservoir
from methods.plotting.selection_unified import build_unified_picks, DESIRED_PICKS_ORDER
from methods.plotting.plot_release_storage_9panel import plot_release_storage_9panel

# -------------------- Small metrics (local) --------------------
def _align(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    if a is None or b is None:
        return (pd.Series(dtype=float), pd.Series(dtype=float))
    z = pd.concat([a.astype(float), b.astype(float)], axis=1).dropna()
    if z.empty: return (pd.Series(dtype=float), pd.Series(dtype=float))
    return z.iloc[:, 0], z.iloc[:, 1]

def nse(sim: pd.Series, obs: pd.Series) -> float:
    s, o = _align(sim, obs)
    if len(s) < 2: return np.nan
    denom = ((o - o.mean())**2).sum()
    if denom == 0: return np.nan
    return 1.0 - ((s - o)**2).sum() / denom

def rmse(sim: pd.Series, obs: pd.Series) -> float:
    s, o = _align(sim, obs)
    if len(s) == 0: return np.nan
    return float(np.sqrt(((s - o)**2).mean()))

def bias(sim: pd.Series, obs: pd.Series) -> float:
    s, o = _align(sim, obs)
    if len(s) == 0: return np.nan
    mo = float(o.mean())
    if mo == 0: return np.nan
    return float((s.mean() - mo) / mo)

# -------------------- Helpers --------------------
def _obj_cols() -> List[str]:
    return list(OBJ_LABELS.values())

def _params_for_row(var_df: pd.DataFrame, idx_row) -> np.ndarray:
    try:
        return var_df.loc[int(idx_row)].to_numpy(dtype=float)
    except Exception:
        return var_df.iloc[int(idx_row)].to_numpy(dtype=float)

def _ensure_dir(p: Path | str) -> Path:
    p = p if isinstance(p, Path) else Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

# -------------------- Independent sim --------------------
def run_independent_once(
    RES: str, POL: str, theta: np.ndarray,
    start: str, end: str, inflow_type: str = "inflow_pub"
) -> Tuple[pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    inflow_df, release_df, storage_df = get_observational_training_data(
        reservoir_name=RES, data_dir=PROCESSED_DATA_DIR, as_numpy=False, inflow_type=inflow_type
    )
    slicer = slice(start, end)
    dates = inflow_df.loc[slicer].index
    inflow = inflow_df.loc[slicer].iloc[:, 0].values

    obs_R = (release_df.loc[slicer][RES] if (release_df is not None and RES in release_df.columns) else None)
    obs_S = (storage_df.loc[slicer][RES] if RES in storage_df.columns else storage_df.loc[slicer].iloc[:, 0])

    model = Reservoir(
        inflow=inflow, dates=dates,
        capacity=reservoir_capacity[RES],
        policy_type=POL, policy_params=np.asarray(theta, float),
        initial_storage=float(obs_S.iloc[0]) if obs_S is not None and len(obs_S) else None,
        name=RES
    )
    model.run()
    sim_S = pd.Series(model.storage_array.flatten(), index=dates, name=RES)
    sim_R = pd.Series((model.release_array + model.spill_array).flatten(), index=dates, name=RES)
    return sim_S, sim_R, obs_S, obs_R

# -------------------- Pywr-DRB Parametric --------------------
def run_pywr_parametric_once(
    RES: str, POL: str, theta: np.ndarray,
    pywr_start: str, pywr_end: str,
    inflow_type: str = "pub_nhmv10_BC_withObsScaled",
    work_dir: Path | str = (Path(CFG_FIG_DIR) / "_pywr_parametric")
) -> tuple[pd.Series, pd.Series]:
    work_dir   = _ensure_dir(work_dir)
    tmp_models = _ensure_dir(PYWR_DRB_REPO / "_tmp_models")
    params_str = ",".join(str(float(x)) for x in np.asarray(theta, float).ravel().tolist())
    tag        = f"Parametric_{POL}_{RES}"

    options = {
        "release_policy_dict": {
            RES: {
                "class_type": "ParametricReservoirRelease",
                "policy_type": POL,
                "policy_id":   "inline",
                "params":      params_str,
            }
        }
    }

    model_json = tmp_models / f"model_{tag}.json"
    h5         = Path(work_dir) / f"output_{tag}.hdf5"

    mb = pywrdrb.ModelBuilder(start_date=pywr_start, end_date=pywr_end,
                              inflow_type=inflow_type, options=options)
    mb.make_model()
    mb.write_model(str(model_json))

    model = pywrdrb.Model.load(str(model_json))
    rec   = pywrdrb.OutputRecorder(model, str(h5))
    _     = model.run()

    dataP = pywrdrb.Data(print_status=False,
                         results_sets=["res_storage", "reservoir_downstream_gage"],
                         output_filenames=[str(h5)])
    dataP.load_output()
    key = h5.stem

    R_full = dataP.reservoir_downstream_gage[key][0][RES].astype(float).rename("pywr_parametric_release")
    S_full = dataP.res_storage[key][0][RES].astype(float).rename("pywr_parametric_storage")
    return R_full, S_full

def run_pywr_parametric_for_candidates(
    RES: str, POL: str, thetas: dict[str, np.ndarray],
    val_start: str, val_end: str,
    pywr_start: str = "1983-10-01", pywr_end: str = "2023-12-31",
    inflow_type: str = "pub_nhmv10_BC_withObsScaled",
    save_series: bool = False,
    outdir: Path | str = (Path(CFG_FIG_DIR) / "_pywr_parametric")
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.Series]], Tuple[pd.Series, pd.Series]]:
    outdir = _ensure_dir(outdir)
    slicer = slice(val_start, val_end)
    runs, metrics = {}, []

    inflow_df, release_df, storage_df = get_observational_training_data(
        reservoir_name=RES, data_dir=PROCESSED_DATA_DIR, as_numpy=False, inflow_type="inflow_pub"
    )
    obs_R = release_df.loc[slicer][RES] if (release_df is not None and RES in release_df.columns) else None
    obs_S = storage_df.loc[slicer][RES] if RES in storage_df.columns else storage_df.loc[slicer].iloc[:, 0]

    for label, theta in thetas.items():
        R_full, S_full = run_pywr_parametric_once(
            RES, POL, np.asarray(theta, float),
            pywr_start=pywr_start, pywr_end=pywr_end,
            inflow_type=inflow_type, work_dir=outdir
        )
        R, S = R_full.loc[slicer], S_full.loc[slicer]
        runs[label] = {"R": R, "S": S}

        metrics.append({
            "Candidate": label,
            "Release NSE": nse(R, obs_R),
            "Storage NSE": nse(S, obs_S),
            "Release RMSE": rmse(R, obs_R),
            "Storage RMSE": rmse(S, obs_S),
            "Release Bias (rel.)": bias(R, obs_R),
            "Storage Bias (rel.)": bias(S, obs_S),
        })

        if save_series:
            tag = f"{RES}_{POL}_{label.replace(' ', '_').replace('/', '-')}_{val_start}_to_{val_end}"
            S.to_csv(Path(outdir) / f"{tag}_pywr_storage.csv")
            R.to_csv(Path(outdir) / f"{tag}_pywr_release.csv")

    metrics_df = pd.DataFrame(metrics).sort_values(
        ["Release NSE", "Storage NSE"], ascending=[False, False]
    ).reset_index(drop=True)

    if save_series:
        (Path(outdir) / f"{RES}_{POL}_pywr_parametric_metrics_{val_start}_to_{val_end}.csv"
         ).write_text(metrics_df.to_csv(index=False))

    return metrics_df, runs, (obs_S, obs_R)

# -------------------- Default series (CSV) loader --------------------
def _default_dir(explicit: Optional[str]) -> Path:
    if explicit: return Path(explicit).resolve()
    env = os.environ.get("DRB_DEFAULT_SERIES", "").strip()
    if env: return Path(env).resolve()
    # fallback to project ./pywr_data/_default_series
    candidate = _find_up("pywr_data", start=PO_REPO) / "_default_series"
    return candidate.resolve()

def load_default_series_csv(default_dir: Path, reservoir: str) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Returns (default_release_MGD, default_storage_MG) as Series or (None, None) if missing.
    CSV schema: columns 'date','value'
    """
    def _read_one(p: Path) -> Optional[pd.Series]:
        if not p.exists(): return None
        df = pd.read_csv(p)
        if "date" not in df.columns or "value" not in df.columns or df.empty:
            return None
        s = pd.Series(df["value"].astype(float).values,
                      index=pd.to_datetime(df["date"].astype(str)), name=p.stem)
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        return s.sort_index()

    r_csv = default_dir / f"{reservoir}_default_release.csv"
    s_csv = default_dir / f"{reservoir}_default_storage.csv"
    return _read_one(r_csv), _read_one(s_csv)

# -------------------- Main driver --------------------
def main():
    ap = argparse.ArgumentParser(description="Selections → Indie sim + Pywr Param + Pywr Default → 9-panel")
    ap.add_argument("--reservoirs", nargs="*", default=RESERVOIR_NAMES, help="Subset of reservoirs")
    ap.add_argument("--policies",   nargs="*", default=POLICY_TYPES,   help="Subset of policies")
    ap.add_argument("--val-start",  default="2019-01-01", help="Validation start (YYYY-MM-DD)")
    ap.add_argument("--val-end",    default="2024-12-31", help="Validation end (YYYY-MM-DD)")
    ap.add_argument("--max-picks",  type=int, default=6,  help="Max picks per (reservoir, policy)")
    ap.add_argument("--default-series-dir", default="",   help="Path to _default_series (optional)")
    ap.add_argument("--save-series", action="store_true", help="Save Pywr Param per-candidate series")
    args = ap.parse_args()

    outdir = _ensure_dir(Path(CFG_FIG_DIR) / "fig4_validation_9panel")
    default_dir = _default_dir(args.default_series_dir)
    print(f"[info] Using default series dir: {default_dir}")

    obj_cols = _obj_cols()
    slicer = slice(args.val_start, args.val_end)

    for RES in args.reservoirs:
        print(f"\n=== Reservoir: {RES} ===")

        # Observations (for the plotting window)
        inflow_df, release_df, storage_df = get_observational_training_data(
            reservoir_name=RES, data_dir=PROCESSED_DATA_DIR, as_numpy=False, inflow_type="inflow_pub"
        )
        obs_R = (release_df.loc[slicer][RES] if (release_df is not None and RES in release_df.columns) else None)
        obs_S = (storage_df.loc[slicer][RES] if RES in storage_df.columns else storage_df.loc[slicer].iloc[:, 0])

        # Default series (overlay)
        default_R_full, default_S_full = load_default_series_csv(default_dir, RES)
        default_R = default_R_full.loc[slicer] if default_R_full is not None else None
        default_S = default_S_full.loc[slicer] if default_S_full is not None else None

        for POL in args.policies:
            # Load optimization outputs
            csv = Path(CFG_OUTPUT_DIR) / f"MMBorg_{ISLANDS}M_{POL}_{RES}_nfe{NFE}_seed{SEED}.csv"
            try:
                obj_df, var_df = load_results(
                    str(csv),
                    obj_labels=OBJ_LABELS,
                    filter=True,
                    obj_bounds=OBJ_FILTER_BOUNDS
                )
            except Exception as e:
                print(f"[WARN] load failed {RES}/{POL}: {e}")
                continue
            if obj_df is None or obj_df.empty or var_df is None or var_df.empty:
                print(f"[INFO] No solutions for {RES}/{POL}.")
                continue

            # Unified selections
            picks_map = build_unified_picks(obj_df, obj_cols)
            want_labels = list(DESIRED_PICKS_ORDER) + [f"Best {c}" for c in obj_cols]
            chosen = [(lab, idx) for lab, idx in picks_map.items() if lab in want_labels]
            if args.max_picks and len(chosen) > args.max_picks:
                chosen = chosen[:args.max_picks]
            if not chosen:
                print(f"[INFO] No picks found for {RES}/{POL}.")
                continue

            # thetas for selected candidates
            thetas: Dict[str, np.ndarray] = {lab: _params_for_row(var_df, idx) for lab, idx in chosen}

            # (a) independent runs (primary “Sim”)
            indie_runs = {}
            for lab, th in thetas.items():
                sim_S, sim_R, _, _ = run_independent_once(RES, POL, th, args.val_start, args.val_end)
                indie_runs[lab] = {"S": sim_S, "R": sim_R}

            # (b) Pywr-DRB Parametric runs (overlay)
            _, pywr_runs, _ = run_pywr_parametric_for_candidates(
                RES, POL, thetas,
                val_start=args.val_start, val_end=args.val_end,
                save_series=args.save_series
            )

            # 9-panels
            for k, (lab, _) in enumerate(chosen, start=1):
                sim_S = indie_runs[lab]["S"]
                sim_R = indie_runs[lab]["R"]
                pywr_S = pywr_runs.get(lab, {}).get("S")
                pywr_R = pywr_runs.get(lab, {}).get("R")

                save_path = outdir / f"{RES}_{POL}_{lab.replace(' ', '_')}_{k}_9panel.png"
                try:
                    plot_release_storage_9panel(
                        reservoir=RES,
                        sim_release=sim_R,
                        sim_storage_MG=sim_S,
                        obs_release=obs_R,                 # optional
                        obs_storage_MG=obs_S,              # optional
                        pywr_param_release=pywr_R,         # overlay: Parametric
                        pywr_param_storage_MG=pywr_S,
                        pywr_default_release=default_R,    # overlay: Default (CSV)
                        pywr_default_storage_MG=default_S,
                        nor_lo_pct=None, nor_hi_pct=None,  # add if you have %capacity bands
                        start=args.val_start, end=args.val_end,
                        policy_label=POL, pick_label=lab,
                        save_path=str(save_path),
                    )
                    print(f"[✓] 9-panel saved: {save_path.name}")
                except Exception as e:
                    print(f"[9-panel FAIL] {RES}/{POL}/{lab}: {e}")

    print("\nDONE.\n")

if __name__ == "__main__":
    main()
