#!/usr/bin/env python3
# 05_validate.py
"""
05_validate.py — Validate & compare parametric reservoir policies in Pywr-DRB

Overview
--------
This script automates a full validation workflow that:
  1) Loads multi-objective optimization (BORG) results, selects "focal" solutions
     per reservoir/policy (Best Release NSE, Best Storage NSE, Best Average NSE,
     Best Average All) using OBJ_LABELS and OBJ_FILTER_BOUNDS.
  2) Runs an independent reservoir simulation (methods.reservoir.model.Reservoir)
     with the chosen policy parameters over a validation window using observed
     inflows (MGD) and compares simulated release = (release + spill) and storage.
  3) Builds and runs a Pywr-DRB model with an inline ParametricReservoirRelease
     node that uses the same policy parameters (“parametric run”).
  4) Builds and runs a default Pywr-DRB model with no overrides (“default run”).
  5) Generates comparison figures and (optionally) CSV time series for analysis.

Generated Figures
-----------------
Saved under:  FIG_DIR/fig5_validation (override with --outdir)

- nine_panel/RES_POLICY_PICK_9panel.png
    9-panel diagnostic for Pywr-DRB (parametric) vs observations over the
    validation window; storage plotted in MG and percent-of-capacity panels.

- dynamics_2x1/RES_POLICY_PICK_2x1_dynamics.png
    2×1 overlay: (top) storage, (bottom) release/flow for:
    Independent run, Pywr-DRB (Parametric), and Pywr-DRB (Default).

- error_ts_release/*.png
    Time-series of release residuals (Sim − Obs) with optional rolling mean and
    LOWESS trend, plus decadal skill badges (NSE, KGE) when data permit.

- error_vs_pct_release/*.png
    Scatter of release residual vs observed flow percentile (0–100) with
    optional LOWESS trend and decade/season color coding.

- error_vs_pct_seasons_release/*.png
    2×2 seasonal panels (Winter/Spring/Summer/Fall) showing residual vs flow
    percentile, colored by decade.

Saved Series (optional)
-----------------------
When --save-series is set, CSVs for independent, parametric, and default runs
are written for both storage and release (sliced to the validation window).

Comparisons (optional)
----------------------
With --compare-series the script compares independent vs Pywr-DRB (parametric)
series using numpy.allclose with user-specified tolerances:
  • rtol: relative tolerance
  • atol: absolute tolerance
See NumPy documentation for details; (rtol=0, atol=0) enforces exact equality.

Inputs & Dependencies
---------------------
- Reads BORG output CSVs from OUTPUT_DIR using methods.load.results.load_results.
- Observations via methods.load.observations.get_observational_training_data.
- Reservoir capacity and policy metadata from methods.config.
- Pywr-DRB model construction/run via pywrdrb.ModelBuilder/Model/Data.
- If Prompton downstream gage observations are missing, the script can attempt
  an NWIS daily-values fetch (cfs→MGD) to fill releases for plotting.

Key Assumptions
---------------
- Units: inflow/release/spill in MGD; storage in MG.
- Validation window applies to figures and CSV outputs; Pywr-DRB runs can cover
  a larger window (controlled by --pywr-start/--pywr-end) and are sliced later.
- Policy parameter vectors are taken from the selected BORG row; bounds are
  checked against policy_param_bounds (out-of-bounds values are only warned).

CLI
---
Examples:

# All default reservoirs/policies, all focal picks, full figure set:
python 05_validate.py --save-series --compare-series --rtol 1e-8 --atol 1e-4

# Subset: STARFIT for FE Walter & Prompton, two focal picks:
python 05_validate.py \
  --reservoirs fewalter prompton \
  --policies STARFIT \
  --picks "Best Release NSE" "Best Average All" \
  --save-series --compare-series

# Alternate Pywr-DRB period & inflow dataset:
python 05_validate.py \
  --pywr-start 1983-10-01 --pywr-end 2023-12-31 \
  --pywr-inflow-type pub_nhmv10_BC_withObsScaled

Arguments
---------
--reservoirs            List of reservoir names (default: methods.config.reservoir_options)
--policies              List of policy types (STARFIT|RBF|PWL; default: methods.config.policy_type_options)
--picks                 Focal picks to plot (default: 4 picks listed above)
--val-start/--val-end   Validation window for figures/CSVs (default: 1980-01-01 to 2018-12-31)
--pywr-start/--pywr-end Pywr-DRB model run window (can exceed validation window)
--pywr-inflow-type      Pywr-DRB inflow dataset key (default provided)
--outdir                Output directory root (default: FIG_DIR/fig5_validation)
--save-series           Write CSVs for all plotted series
--compare-series        Compare independent vs parametric series (rtol/atol)
--rtol/--atol           Tolerances for numpy.allclose in comparisons
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==== project imports ====
from methods.config import (
    NFE, SEED, ISLANDS,
    OBJ_LABELS, OBJ_FILTER_BOUNDS,
    reservoir_options, policy_type_options,
    OUTPUT_DIR, FIG_DIR, PROCESSED_DATA_DIR,
    reservoir_capacity, policy_n_params, policy_param_bounds,
)
from methods.load.results import load_results
from methods.load.observations import get_observational_training_data
from methods.reservoir.model import Reservoir
from methods.plotting.plot_release_storage_9panel import plot_release_storage_9panel

from methods.plotting.plot_dynamics_2x1 import plot_2x1_dynamics
from methods.plotting.plot_error_diagnostics import (
    plot_error_time_series_enhanced,
    plot_error_vs_flow_percentile_enhanced,
    plot_seasonal_decadal_panels,
)

import pywrdrb

# ---------- small helpers ----------
def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def parse_params_inline(s: str, expected_n: int) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(vals) != expected_n:
        raise ValueError(f"Expected {expected_n} params, got {len(vals)}.")
    return np.array(vals, dtype=float)

def compare_series(a: pd.Series, b: pd.Series, name: str, rtol=0.0, atol=0.0):
    a = a.astype(float); b = b.astype(float)
    idx = a.index.intersection(b.index)
    a = a.loc[idx]; b = b.loc[idx]
    if len(a) == 0:
        print(f"[WARN] {name}: no overlapping index to compare.")
        return False
    if rtol == 0.0 and atol == 0.0:
        eq = (a.values == b.values).all()
        print(f"[COMPARE:{name}] exact equality: {eq} over {len(a)} steps."); return eq
    ok = np.allclose(a.values, b.values, rtol=rtol, atol=atol, equal_nan=True)
    print(f"[COMPARE:{name}] allclose rtol={rtol} atol={atol}: {ok} over {len(a)} steps."); return ok

def _robust_limits(series_list, lo=1.0, hi=99.0, pad=0.06):
    vals = pd.concat([pd.Series(s, dtype=float) for s in series_list if s is not None], axis=0)
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None
    ql, qh = np.nanpercentile(vals.values, [lo, hi])
    span = qh - ql
    return (ql - pad*span, qh + pad*span)

def _compute_common_axes(indie_R, pywr_Rv, def_Rv, obs_release,
                         indie_S, pywr_Sv, def_Sv, obs_storage):
    # Release y-lims (MGD)
    r_ylim = _robust_limits([indie_R, pywr_Rv, def_Rv, obs_release])
    # Storage y-lims (MG) — keep MG here; 9-panel has its own %cap axis.
    s_ylim = _robust_limits([indie_S, pywr_Sv, def_Sv, obs_storage])
    return s_ylim, r_ylim

# ---- optionally fill Prompton obs gage via NWIS if missing ----
def _maybe_fetch_prompton_obs_if_missing(data_obj, base_key, reservoir_name):
    if reservoir_name.lower() != "prompton":
        return
    try:
        df_sim_gage = data_obj.reservoir_downstream_gage[base_key][0]
        df_obs_gage = data_obj.reservoir_downstream_gage["obs"][0]
        if "prompton" in df_obs_gage.columns and not df_obs_gage["prompton"].isna().all():
            return
        from dataretrieval import nwis
        def _tz_naive(idx):
            idx = pd.to_datetime(idx)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            return idx
        idx_sim = _tz_naive(df_sim_gage.index)
        s_date, e_date = idx_sim.min().strftime("%Y-%m-%d"), idx_sim.max().strftime("%Y-%m-%d")
        site, param = "01429000", "00060"
        print(f"Retrieving Prompton DV {param} {s_date}..{e_date} from NWIS...")
        df_raw = nwis.get_record(sites=site, start=s_date, end=e_date, parameterCd=param, service="dv")
        if df_raw is None or len(df_raw) == 0:
            print("NWIS returned no DV; skipping."); return
        df_raw.index = _tz_naive(pd.to_datetime(df_raw.index)); df_raw.sort_index(inplace=True)
        col = next((c for c in df_raw.columns if "00060" in c and ("Mean" in c or c.endswith("_Mean"))), None)
        if col is None:
            non_qual = [c for c in df_raw.columns if "qual" not in c.lower()]
            col = non_qual[0] if non_qual else df_raw.columns[0]
        CFS_TO_MGD = 0.646317
        df_promp = df_raw[[col]].rename(columns={col:"prompton"}).astype(float)
        df_promp["prompton"] *= CFS_TO_MGD
        target_index = df_obs_gage.index if len(df_obs_gage.index) else idx_sim
        df_promp = df_promp.reindex(target_index)
        df_obs_gage = df_obs_gage.copy()
        if len(df_obs_gage.index) == 0:
            df_obs_gage = pd.DataFrame(index=target_index)
        df_obs_gage.loc[:, "prompton"] = df_promp["prompton"].values
        data_obj.reservoir_downstream_gage["obs"][0] = df_obs_gage
        print("Prompton appended to observations (MGD).")
    except Exception as e:
        print(f"Prompton NWIS append skipped: {e}")

# ---------- core: pick best solutions from BORG CSVs ----------
def load_focal_picks():
    policy_types = policy_type_options
    reservoirs   = reservoir_options
    obj_labels   = OBJ_LABELS
    res2pol2_obj = {}
    res2pol2_var = {}

    for res in reservoirs:
        res2pol2_obj[res] = {}
        res2pol2_var[res] = {}
        for pol in policy_types:
            fname = f"{OUTPUT_DIR}/MMBorg_{ISLANDS}M_{pol}_{res}_nfe{NFE}_seed{SEED}.csv"
            # load with filter=True (your chosen bounds)
            obj_df, var_df = load_results(fname, obj_labels=obj_labels, filter=True, obj_bounds=OBJ_FILTER_BOUNDS)
            if obj_df is None or len(obj_df)==0:
                print(f"[SEL] Skip {res}/{pol}: no solutions.")
                continue

            # focal indices
            idx_best_release = obj_df["Release NSE"].idxmax()
            idx_best_storage = obj_df["Storage NSE"].idxmax()
            idx_best_average = obj_df[["Release NSE","Storage NSE"]].mean(axis=1).idxmax()

            min_obj = obj_df.copy()
            min_obj["Release NSE"] = -min_obj["Release NSE"]
            min_obj["Storage NSE"] = -min_obj["Storage NSE"]
            scaled = (min_obj - min_obj.min()) / (min_obj.max() - min_obj.min())
            idx_best_all_avg = scaled.mean(axis=1).idxmin()

            highlight = {
                idx_best_release: "Best Release NSE",
                idx_best_storage: "Best Storage NSE",
                idx_best_average: "Best Average NSE",
                idx_best_all_avg: "Best Average All",
            }
            obj_df = obj_df.copy()
            obj_df["highlight"] = ["Other"]*len(obj_df)
            for i, lab in highlight.items():
                if i in obj_df.index:
                    obj_df.loc[i, "highlight"] = lab

            res2pol2_obj[res][pol] = obj_df
            res2pol2_var[res][pol] = var_df

    return res2pol2_obj, res2pol2_var

# ---------- pywr run (parametric + default) ----------
def run_pywr_pair(res_name, policy, params_vec, start, end, inflow_type, outdir):
    # parametric
    options = {
        "release_policy_dict": {
            res_name: {
                "class_type": "ParametricReservoirRelease",
                "policy_type": policy,
                "policy_id":   "inline",
                "params":      ",".join(str(x) for x in np.asarray(params_vec, float).tolist()),
            }
        }
    }
    mb = pywrdrb.ModelBuilder(start_date=start, end_date=end, inflow_type=inflow_type, options=options)
    mb.make_model()
    model_json = os.path.join(outdir, f"model_Parametric_{policy}_{res_name}.json")
    mb.write_model(model_json)
    model = pywrdrb.Model.load(model_json)
    h5 = os.path.join(outdir, f"output_Parametric_{policy}_{res_name}.hdf5")
    rec = pywrdrb.OutputRecorder(model, h5); _ = model.run()
    assert os.path.exists(h5), "Parametric run: missing HDF5 output."

    # default
    mb_def = pywrdrb.ModelBuilder(start_date=start, end_date=end, inflow_type=inflow_type)
    mb_def.make_model()
    model_def_json = os.path.join(outdir, "model_default.json")
    mb_def.write_model(model_def_json)
    model_def = pywrdrb.Model.load(model_def_json)
    h5_def = os.path.join(outdir, "output_default.hdf5")
    rec_def = pywrdrb.OutputRecorder(model_def, h5_def); _ = model_def.run()
    assert os.path.exists(h5_def), "Default run: missing HDF5 output."

    # load outputs
    results_sets = ['major_flow','res_storage','reservoir_downstream_gage','lower_basin_mrf_contributions']
    dataP = pywrdrb.Data(print_status=True, results_sets=results_sets, output_filenames=[h5])
    dataP.load_output(); dataP.load_observations()
    kP = os.path.splitext(os.path.basename(h5))[0]
    _maybe_fetch_prompton_obs_if_missing(dataP, kP, res_name)
    dfP_R = dataP.reservoir_downstream_gage[kP][0][res_name].astype(float).rename("pywr_release")
    dfP_S = dataP.res_storage[kP][0][res_name].astype(float).rename("pywr_storage")

    dataD = pywrdrb.Data(print_status=True, results_sets=results_sets, output_filenames=[h5_def])
    dataD.load_output(); dataD.load_observations()
    kD = os.path.splitext(os.path.basename(h5_def))[0]
    dfD_R = dataD.reservoir_downstream_gage[kD][0][res_name].astype(float).rename("default_release")
    dfD_S = dataD.res_storage[kD][0][res_name].astype(float).rename("default_storage")

    return (dfP_R, dfP_S), (dfD_R, dfD_S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reservoirs", nargs="*", default=reservoir_options,
                    help="subset like: fewalter prompton blueMarsh beltzvilleCombined")
    ap.add_argument("--policies", nargs="*", default=policy_type_options, choices=["STARFIT","RBF","PWL"])
    ap.add_argument("--picks", nargs="*", default=["Best Release NSE","Best Storage NSE","Best Average NSE","Best Average All"])
    ap.add_argument("--val-start", default="1980-01-01")
    ap.add_argument("--val-end",   default="2018-12-31")
    ap.add_argument("--pywr-start", default="1983-10-01")
    ap.add_argument("--pywr-end",   default="2023-12-31")
    ap.add_argument("--pywr-inflow-type", default="pub_nhmv10_BC_withObsScaled")
    ap.add_argument("--use-opt-window", action="store_true",
                    help="Run validation strictly on the optimization (inflow_pub) window.")
    ap.add_argument("--align-inflow", action="store_true",
                    help="Use the same inflow family as optimization (inflow_pub) for Pywr-DRB.")
    ap.add_argument("--outdir", default=os.path.join(FIG_DIR, "fig5_validation"))
    ap.add_argument("--save-series", action="store_true")
    ap.add_argument("--compare-series", action="store_true")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    outdir_9 = ensure_dir(os.path.join(outdir, "nine_panel"))
    outdir_dyn = ensure_dir(os.path.join(outdir, "dynamics_2x1"))
    print(f"[info] writing to {outdir}")

    # helper: for a given reservoir, compute the optimization (public) window
    def _opt_window_for_res(res):
        inflow_pub_df, release_pub_df, storage_pub_df = get_observational_training_data(
            reservoir_name=res, data_dir=PROCESSED_DATA_DIR, as_numpy=False, inflow_type="inflow_pub"
        )
        # Use the strict common index across what was optimized against
        idx = inflow_pub_df.index
        return str(idx.min().date()), str(idx.max().date())
    
    # 1) load all results; mark focal picks
    sol_objs, sol_vars = load_focal_picks()

    # 2) loop reservoirs/policies/picks → simulate + figures
    for res in args.reservoirs:
            # Override windows/inflow to match optimization, if requested
        val_start, val_end = args.val_start, args.val_end
        pywr_start, pywr_end = args.pywr_start, args.pywr_end
        pywr_inflow_type = args.pywr_inflow_type

        if args.use_opt_window:
            val_start, val_end = _opt_window_for_res(res)   # same window as optimization
            pywr_start, pywr_end = val_start, val_end       # force Pywr to same span
            print(f"[opt-window] {res}: {val_start} → {val_end}")

        if args.align_inflow:
            pywr_inflow_type = "pub_nhmv10_BC_withObsScaled"                 # same inflow family as optimization
            print(f"[align-inflow] {res}: Pywr inflow_type set to 'pub_nhmv10_BC_withObsScaled'")

        if res not in sol_objs:
            print(f"[Skip] {res}: no results dict.")
            continue

        # observed for validation window (for 9-panel)
        inflow_df, release_df, storage_df = get_observational_training_data(
            reservoir_name=res, data_dir=PROCESSED_DATA_DIR, as_numpy=False, inflow_type="inflow_pub"
        )
        slicer = slice(val_start, val_end)
        obs_release = (release_df.loc[slicer][res] if res in release_df.columns else None)
        obs_storage = (storage_df.loc[slicer][res] if res in storage_df.columns else storage_df.loc[slicer].iloc[:,0])

        for pol in args.policies:
            if pol not in sol_objs.get(res, {}):
                print(f"[Skip] {res}/{pol}: no solutions.")
                continue

            obj_df = sol_objs[res][pol]
            var_df = sol_vars[res][pol]

            for pick in args.picks:
                mask = (obj_df["highlight"] == pick)
                if mask.sum()==0:
                    print(f"[Skip] {res}/{pol}/{pick}: not found.")
                    continue
                params = var_df.loc[mask].iloc[0].values  # single vector

                # 2a) independent reservoir run over the validation window (using obs inflow)
                inflow_df2, release_df2, storage_df2 = get_observational_training_data(
                    reservoir_name=res, data_dir=PROCESSED_DATA_DIR, as_numpy=False, inflow_type="inflow_pub"
                )
                dt_idx = inflow_df2.index
                inflow = inflow_df2[res].to_numpy().astype(float).ravel()
                storage0 = float(storage_df2[res].iloc[0])

                indie = Reservoir(
                    inflow=inflow, dates=dt_idx, capacity=float(reservoir_capacity[res]),
                    policy_type=pol, policy_params=params, initial_storage=storage0, name=res
                )
                indie.reset(); indie.run()
                indie_S_full = pd.Series(indie.storage_array.flatten(), index=dt_idx, name=res)
                indie_R_full = pd.Series((indie.release_array + indie.spill_array).flatten(), index=dt_idx, name=res)
                indie_S = indie_S_full.loc[slicer]; indie_R = indie_R_full.loc[slicer]

                # 2b) pywr parametric & default (full period requested for pywr), then slice to validation for figs
                (pywr_R, pywr_S), (def_R, def_S) = run_pywr_pair(
                    res, pol, params, pywr_start, pywr_end, pywr_inflow_type, outdir
                )

                pywr_Rv = pywr_R.loc[slicer]; pywr_Sv = pywr_S.loc[slicer]
                def_Rv  = def_R.loc[slicer];  def_Sv  = def_S.loc[slicer]
                indie_R = indie_R_full.loc[slicer]; indie_S = indie_S_full.loc[slicer]

                # compute per-reservoir common axes
                s_ylim, r_ylim = _compute_common_axes(
                    indie_R, pywr_Rv, def_Rv, obs_release,
                    indie_S, pywr_Sv, def_Sv, obs_storage
                )
                # 3) nine-panel validation figure (pywr parametric vs obs; storage in % cap)
                nine_png = os.path.join(outdir_9, f"{res}_{pol}_{pick.replace(' ','_')}_9panel.png")
                plot_release_storage_9panel(
                    reservoir=res,
                    sim_release=pywr_Rv,
                    obs_release=obs_release,
                    sim_storage_MG=pywr_Sv,
                    obs_storage_MG=obs_storage,
                    start=val_start, end=val_end,
                    save_path=nine_png
                )
                print(f"[✓] saved 9-panel: {nine_png}")

                # 4) 2×1 storage/release dynamics overlay (independent vs pywr vs default)
                tag = f"{res}_{pol}_{pick.replace(' ','_')}"
                dyn_png = os.path.join(outdir_dyn, f"{res}_{pol}_{pick.replace(' ','_')}_2x1_dynamics.png")
                plot_2x1_dynamics(
                    reservoir=res, policy=pol,
                    indie_R=indie_R, indie_S=indie_S,
                    pywr_R=pywr_Rv, pywr_S=pywr_Sv,
                    def_R=def_Rv,  def_S=def_Sv,
                    obs_R=obs_release, obs_S=obs_storage,
                    date_label=f"{val_start} to {val_end}",
                    ylims_storage=s_ylim, ylims_release=r_ylim,
                    save_path=dyn_png,
                )

                period_lbl = f"{res} {pol} {pick} ({val_start}–{val_end})"
                # releases
                plot_error_time_series_enhanced(
                    df_obs=obs_release.to_frame(res),   # DataFrame with column named res
                    df_sim=pywr_Rv.to_frame(res),
                    reservoirs=[res],
                    start=args.val_start, end=args.val_end,
                    period_label=period_lbl,
                    save_folder=os.path.join(outdir, "error_ts_release"),
                    acceptable_band=None,  # e.g., 50 for ±50 MGD band
                )
                plot_error_vs_flow_percentile_enhanced(
                    df_obs=obs_release.to_frame(res),
                    df_sim=pywr_Rv.to_frame(res),
                    reservoirs=[res],
                    period_label=period_lbl,
                    save_folder=os.path.join(outdir, "error_vs_pct_release"),
                )
                plot_seasonal_decadal_panels(
                    df_obs=obs_release.to_frame(res),
                    df_sim=pywr_Rv.to_frame(res),
                    reservoirs=[res],
                    period_label=period_lbl,
                    save_folder=os.path.join(outdir, "error_vs_pct_seasons_release"),
                )

                # 5) optional saves & comparisons (indie vs pywr parametric)
                if args.save_series:
                    indie_R.to_csv(os.path.join(outdir, f"{tag}_indie_release.csv"), header=True)
                    indie_S.to_csv(os.path.join(outdir, f"{tag}_indie_storage.csv"), header=True)
                    pywr_Rv.to_csv(os.path.join(outdir, f"{tag}_pywr_release.csv"), header=True)
                    pywr_Sv.to_csv(os.path.join(outdir, f"{tag}_pywr_storage.csv"), header=True)
                    def_Rv.to_csv(os.path.join(outdir, f"{tag}_default_release.csv"), header=True)
                    def_Sv.to_csv(os.path.join(outdir, f"{tag}_default_storage.csv"), header=True)

                if args.compare_series:
                    compare_series(indie_R, pywr_Rv, f"{res}:{pol}:{pick} release (indie vs pywr)", rtol=args.rtol, atol=args.atol)
                    compare_series(indie_S, pywr_Sv, f"{res}:{pol}:{pick} storage (indie vs pywr)", rtol=args.rtol, atol=args.atol)

    print("DONE.")

if __name__ == "__main__":
    main()
