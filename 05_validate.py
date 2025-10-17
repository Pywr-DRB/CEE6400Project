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
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== project imports ====
from methods.config import (
    NFE, SEED, ISLANDS,
    OBJ_LABELS, OBJ_FILTER_BOUNDS,
    reservoir_options, policy_type_options,
    OUTPUT_DIR, FIG_DIR, PROCESSED_DATA_DIR,
    reservoir_capacity,
)
from methods.load.results import load_results
from methods.load.observations import get_observational_training_data
from methods.reservoir.model import Reservoir

from methods.plotting.plot_release_storage_9panel import plot_release_storage_9panel
from methods.plotting.plot_dynamics_2x1 import plot_2x1_dynamics
from methods.plotting.plot_error_diagnostics import (
    plot_error_time_series_enhanced_multi,
    plot_error_vs_flow_percentile_enhanced_multi,
    plot_seasonal_decadal_panels,
)
from methods.plotting.plot_sixpanel_timeseries_exceedance_fdc import (
    plot_sixpanel_timeseries_exceedance_fdc
)

# advanced selection utilities (same as 04)
from methods.plotting.selection_utils import (
    compute_and_apply_advanced_highlights,  # adds 'highlight_adv' + returns cand_map
    ADVANCED_COLORS,
)

import pywrdrb


# ---------- helpers ----------
def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def safe_name(s) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s)).strip('_') or "pick"

def compare_series(a: pd.Series, b: pd.Series, name: str, rtol=0.0, atol=0.0):
    a = a.astype(float); b = b.astype(float)
    idx = a.index.intersection(b.index)
    a = a.loc[idx]; b = b.loc[idx]
    if len(a) == 0:
        print(f"[WARN] {name}: no overlapping index to compare.")
        return False
    if rtol == 0.0 and atol == 0.0:
        ok = (a.values == b.values).all()
        print(f"[COMPARE:{name}] exact equality: {ok} over {len(a)} steps.")
        return ok
    ok = np.allclose(a.values, b.values, rtol=rtol, atol=atol, equal_nan=True)
    print(f"[COMPARE:{name}] allclose rtol={rtol} atol={atol}: {ok} over {len(a)} steps.")
    return ok

# objective senses (same as 04)
SENSES_ALL = {
    "Release NSE": "max",
    "Q20 Log Release NSE": "max",
    "Q80 Release Abs % Bias": "min",
    "Release Inertia": "max",
    "Storage KGE": "max",
    "Storage Inertia": "max",
}

# unified default picks (legacy + advanced)
DEFAULT_PICKS = [
    # legacy
    "Best Release NSE", "Best Storage KGE", "Best Average NSE", "Best Average All",
    # advanced from selection_utils
    "Compromise L2 (Euclidean)", "Tchebycheff L∞", "Manhattan L1",
    "ε-constraint Release NSE ≥ Q50", "Diverse #1 (FPS)", "Diverse #2 (FPS)",
]


# ---------- optionally fill Prompton obs gage via NWIS if missing ----------
def _maybe_fetch_prompton_obs_if_missing(data_obj, base_key, reservoir_name):
    if str(reservoir_name).lower() != "prompton":
        return
    try:
        df_sim_gage = data_obj.reservoir_downstream_gage[base_key][0]
        df_obs_gage = data_obj.reservoir_downstream_gage["obs"][0]
        if ("prompton" in df_obs_gage.columns) and (not df_obs_gage["prompton"].isna().all()):
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
        # pick a mean column
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


# ---------- load BORG and build legacy+advanced picks ----------
def load_picks_with_advanced():
    res2pol2_obj = {}
    res2pol2_var = {}
    res2pol2_candmap = {}

    for res in reservoir_options:
        res2pol2_obj[res] = {}
        res2pol2_var[res] = {}
        res2pol2_candmap[res] = {}
        for pol in policy_type_options:
            fname = f"{OUTPUT_DIR}/MMBorg_{ISLANDS}M_{pol}_{res}_nfe{NFE}_seed{SEED}.csv"
            try:
                obj_df, var_df = load_results(
                    fname, obj_labels=OBJ_LABELS, filter=True, obj_bounds=OBJ_FILTER_BOUNDS
                )
            except Exception as e:
                print(f"[SEL] Skip {res}/{pol}: failed to load ({e}).")
                continue
            if obj_df is None or len(obj_df) == 0:
                print(f"[SEL] Skip {res}/{pol}: no solutions.")
                continue

            # ---- legacy highlights (Storage KGE!) ----
            idx_best_release = obj_df["Release NSE"].idxmax()
            idx_best_storage = obj_df["Storage KGE"].idxmax()
            idx_best_average = obj_df[["Release NSE", "Storage KGE"]].mean(axis=1).idxmax()

            min_obj = obj_df.copy()
            min_obj["Release NSE"] = -min_obj["Release NSE"]
            min_obj["Storage KGE"] = -min_obj["Storage KGE"]
            scaled = (min_obj - min_obj.min()) / (min_obj.max() - min_obj.min())
            idx_best_all_avg = scaled.mean(axis=1).idxmin()

            highlight = {
                idx_best_release: "Best Release NSE",
                idx_best_storage: "Best Storage KGE",
                idx_best_average: "Best Average NSE",
                idx_best_all_avg: "Best Average All",
            }
            obj_df = obj_df.copy()
            obj_df["highlight"] = ["Other"] * len(obj_df)
            for i, lab in highlight.items():
                if i in obj_df.index:
                    obj_df.loc[i, "highlight"] = lab

            # ---- advanced highlights (same as 04) ----
            advanced_objectives = [
                "Release NSE",
                "Storage KGE",
                "Q20 Log Release NSE",
                "Q80 Release Abs % Bias",
                "Release Inertia",
                "Storage Inertia",
            ]
            obj_df_aug, cand_df, cand_map = compute_and_apply_advanced_highlights(
                obj_df,
                objectives=advanced_objectives,
                senses=SENSES_ALL,
                bounds=OBJ_FILTER_BOUNDS,
                eps_qs=(0.5, 0.8),
                add_k_diverse=2,
                include_hv=False,
                out_label_col="highlight_adv",
            )

            res2pol2_obj[res][pol] = obj_df_aug
            res2pol2_var[res][pol] = var_df
            res2pol2_candmap[res][pol] = cand_map

    return res2pol2_obj, res2pol2_var, res2pol2_candmap


def get_pick_indices(solution_objs, cand_maps, reservoir_name: str, policy_type: str, label: str):
    """
    Return list of row indices for requested pick, searching:
      1) legacy 'highlight' column
      2) advanced cand_map entries
      3) 'highlight_adv' column
    """
    out = []

    df = solution_objs.get(reservoir_name, {}).get(policy_type)
    if df is not None and "highlight" in df.columns:
        out += df.index[df["highlight"] == label].tolist()

    cand_map = (cand_maps.get(reservoir_name, {}) or {}).get(policy_type, {}) or {}
    if label in cand_map and cand_map[label] is not None:
        val = cand_map[label]
        if isinstance(val, (list, tuple, np.ndarray, pd.Index, pd.Series)):
            out += list(pd.Index(val))
        else:
            out.append(val)

    if df is not None and "highlight_adv" in df.columns:
        out += df.index[df["highlight_adv"] == label].tolist()

    # dedupe preserving order
    seen, deduped = set(), []
    for idx in out:
        key = int(idx) if isinstance(idx, (int, np.integer)) or str(idx).isdigit() else idx
        if key not in seen:
            seen.add(key)
            deduped.append(idx)
    return deduped


# ---------- pywr runs ----------
def run_pywr_parametric(res_name, policy, params_vec, start, end, inflow_type, outdir):
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
    _ = pywrdrb.OutputRecorder(model, h5); _ = model.run()

    results_sets = ['major_flow','res_storage','reservoir_downstream_gage','lower_basin_mrf_contributions']
    dataP = pywrdrb.Data(print_status=False, results_sets=results_sets, output_filenames=[h5])
    dataP.load_output(); dataP.load_observations()
    kP = os.path.splitext(os.path.basename(h5))[0]
    _maybe_fetch_prompton_obs_if_missing(dataP, kP, res_name)
    dfP_R = dataP.reservoir_downstream_gage[kP][0][res_name].astype(float).rename("pywr_release")
    dfP_S = dataP.res_storage[kP][0][res_name].astype(float).rename("pywr_storage")
    return dfP_R, dfP_S


def run_pywr_default(start, end, inflow_type, outdir):
    """Build & run default model. Returns (dfD_R_dict, dfD_S_dict) for all reservoirs or (None, None) on failure."""
    try:
        mb_def = pywrdrb.ModelBuilder(start_date=start, end_date=end, inflow_type=inflow_type)
        mb_def.make_model()
        model_def_json = os.path.join(outdir, "model_default.json")
        mb_def.write_model(model_def_json)
        model_def = pywrdrb.Model.load(model_def_json)
        h5_def = os.path.join(outdir, "output_default.hdf5")
        _ = pywrdrb.OutputRecorder(model_def, h5_def); _ = model_def.run()

        results_sets = ['major_flow','res_storage','reservoir_downstream_gage','lower_basin_mrf_contributions']
        dataD = pywrdrb.Data(print_status=False, results_sets=results_sets, output_filenames=[h5_def])
        dataD.load_output(); dataD.load_observations()
        kD = os.path.splitext(os.path.basename(h5_def))[0]

        # Build dicts for convenient per-res extraction later
        dfD_R = {res: dataD.reservoir_downstream_gage[kD][0][res].astype(float).rename("default_release")
                 for res in reservoir_options if res in dataD.reservoir_downstream_gage[kD][0].columns}
        dfD_S = {res: dataD.res_storage[kD][0][res].astype(float).rename("default_storage")
                 for res in reservoir_options if res in dataD.res_storage[kD][0].columns}
        return dfD_R, dfD_S
    except Exception as e:
        # This is where the malformed STARFIT CSV crash was happening
        print(f"[WARN] Default Pywr run failed; continuing without default series. Reason: {e}")
        return None, None


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reservoirs", nargs="*", default=reservoir_options,
                    help="subset like: fewalter prompton blueMarsh beltzvilleCombined")
    ap.add_argument("--policies", nargs="*", default=policy_type_options, choices=["STARFIT","RBF","PWL"])
    ap.add_argument("--picks", nargs="*", default=DEFAULT_PICKS,
                    help="Will match both legacy 'highlight' and advanced 'highlight_adv' / cand_map.")
    ap.add_argument("--val-start", default="1980-01-01")
    ap.add_argument("--val-end",   default="2018-12-31")
    ap.add_argument("--pywr-start", default="1983-10-01")
    ap.add_argument("--pywr-end",   default="2023-12-31")
    ap.add_argument("--pywr-inflow-type", default="pub_nhmv10_BC_withObsScaled")
    ap.add_argument("--use-opt-window", action="store_true",
                    help="Plot strictly on the optimization (inflow_pub) window.")
    ap.add_argument("--align-inflow", action="store_true",
                    help="Force 'pub_nhmv10_BC_withObsScaled' for the Pywr-DRB runs.")
    ap.add_argument("--skip-default-run", action="store_true",
                    help="Do not build/run Pywr default model (avoids STARFIT CSV dependency).")
    ap.add_argument("--outdir", default=os.path.join(FIG_DIR, "fig5_validation"))
    ap.add_argument("--save-series", action="store_true")
    ap.add_argument("--compare-series", action="store_true")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    outdir_9  = ensure_dir(os.path.join(outdir, "nine_panel"))
    outdir_dyn = ensure_dir(os.path.join(outdir, "dynamics_2x1"))
    outdir_six = ensure_dir(os.path.join(outdir, "six_panel"))
    print(f"[info] writing to {outdir}")

    # compute the optimization (public) window per-reservoir
    def _opt_window_for_res(res):
        inflow_pub_df, release_pub_df, storage_pub_df = get_observational_training_data(
            reservoir_name=res, data_dir=PROCESSED_DATA_DIR, as_numpy=False, inflow_type="inflow_pub"
        )
        idx = inflow_pub_df.index
        return str(idx.min().date()), str(idx.max().date())

    # 1) load results + advanced picks
    sol_objs, sol_vars, cand_maps = load_picks_with_advanced()

    # 2) (optional) run Pywr default ONCE and cache results
    def_R_all, def_S_all = (None, None)
    if not args.skip_default_run:
        def_R_all, def_S_all = run_pywr_default(args.pywr_start, args.pywr_end,
                                                "pub_nhmv10_BC_withObsScaled" if args.align_inflow else args.pywr_inflow_type,
                                                outdir)

    # 3) loop reservoirs/policies/picks
    for res in args.reservoirs:
        pywr_start, pywr_end = args.pywr_start, args.pywr_end
        pywr_inflow_type = "pub_nhmv10_BC_withObsScaled" if args.align_inflow else args.pywr_inflow_type

        if res not in sol_objs:
            print(f"[Skip] {res}: no results dict."); continue

        # choose the PLOT window
        if args.use_opt_window:
            PLOT_START, PLOT_END = _opt_window_for_res(res)
            print(f"[plot-window] {res}: {PLOT_START} → {PLOT_END} (optimization window)")
        else:
            PLOT_START, PLOT_END = args.val_start, args.val_end
            print(f"[plot-window] {res}: {PLOT_START} → {PLOT_END} (user window)")
        plot_slicer = slice(PLOT_START, PLOT_END)

        # observed series for the plot window
        inflow_df, release_df, storage_df = get_observational_training_data(
            reservoir_name=res, data_dir=PROCESSED_DATA_DIR, as_numpy=False, inflow_type="inflow_pub"
        )
        obs_release = (release_df.loc[plot_slicer][res] if res in release_df.columns else None)
        obs_storage = (storage_df.loc[plot_slicer][res] if res in storage_df.columns else storage_df.loc[plot_slicer].iloc[:,0])

        for pol in args.policies:
            if pol not in sol_objs.get(res, {}):
                print(f"[Skip] {res}/{pol}: no solutions."); continue

            obj_df = sol_objs[res][pol]
            var_df = sol_vars[res][pol]

            for pick in args.picks:
                idxs = get_pick_indices(sol_objs, cand_maps, res, pol, pick)
                if not idxs:
                    print(f"[Skip] {res}/{pol}/{pick}: not found."); continue
                # take first match
                params = var_df.loc[idxs[0]].values

                # Independent run on full inflow_pub index; slice to plot window
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
                indie_S = pd.Series(indie.storage_array.flatten(), index=dt_idx, name=res).loc[plot_slicer]
                indie_R = pd.Series((indie.release_array + indie.spill_array).flatten(), index=dt_idx, name=res).loc[plot_slicer]

                # Pywr parametric for (res, pol)
                pywr_R, pywr_S = run_pywr_parametric(res, pol, params, pywr_start, pywr_end, pywr_inflow_type, outdir)
                pywr_Rv = pywr_R.loc[plot_slicer]
                pywr_Sv = pywr_S.loc[plot_slicer]

                # default series (if available)
                def_Rv = def_Sv = None
                if def_R_all is not None and res in def_R_all:
                    def_Rv = def_R_all[res].loc[plot_slicer]
                if def_S_all is not None and res in def_S_all:
                    def_Sv = def_S_all[res].loc[plot_slicer]

                # 9-panel (parametric vs obs)
                nine_png = os.path.join(outdir_9, f"{res}_{pol}_{safe_name(pick)}_9panel.png")
                plot_release_storage_9panel(
                    reservoir=res,
                    sim_release=pywr_Rv,
                    obs_release=obs_release,
                    sim_storage_MG=pywr_Sv,
                    obs_storage_MG=obs_storage,
                    start=PLOT_START, end=PLOT_END,
                    save_path=nine_png
                )
                print(f"[✓] saved 9-panel: {nine_png}")

                # 2×1 dynamics overlay (indie, parametric, default if present)
                dyn_png = os.path.join(outdir_dyn, f"{res}_{pol}_{safe_name(pick)}_2x1_dynamics.png")
                plot_2x1_dynamics(
                    reservoir=res, policy=pol,
                    indie_R=indie_R, indie_S=indie_S,
                    pywr_R=pywr_Rv, pywr_S=pywr_Sv,
                    def_R=def_Rv,  def_S=def_Sv,
                    obs_R=obs_release, obs_S=obs_storage,
                    date_label=f"{PLOT_START} to {PLOT_END}",
                    ylims_storage=None, ylims_release=None,
                    save_path=dyn_png,
                    yscale_storage=None, yscale_release=None,
                    max_date_ticks=10,
                )

                # 6-panel comparisons
                six_png = os.path.join(outdir_six, f"{res}_{pol}_{safe_name(pick)}_6panel.png")
                plot_sixpanel_timeseries_exceedance_fdc(
                    reservoir=res,
                    capacity_MG=float(reservoir_capacity[res]),
                    indie_R=indie_R, indie_S=indie_S,
                    pywr_R=pywr_Rv,  pywr_S=pywr_Sv,
                    def_R=def_Rv,    def_S=def_Sv,
                    obs_R=obs_release, obs_S=obs_storage,
                    date_label=f"{PLOT_START} to {PLOT_END}",
                    save_path=six_png,
                    downsample_step=2,
                    max_date_ticks=10,
                )
                print(f"[✓] saved 6-panel: {six_png}")

                # residual diagnostics (release)
                sim_release_dict = {"Independent": indie_R, "Pywr Parametric": pywr_Rv}
                if def_Rv is not None:
                    sim_release_dict["Pywr Default"] = def_Rv

                err_ts_folder = ensure_dir(os.path.join(outdir, "error_ts_release_multi"))
                plot_error_time_series_enhanced_multi(
                    obs=obs_release,
                    sim_dict=sim_release_dict,
                    title_prefix=f"{res} {pol} {pick} ({PLOT_START}–{PLOT_END}) — Release",
                    start=PLOT_START, end=PLOT_END,
                    save_path=os.path.join(err_ts_folder, f"error_timeseries_release_{res}_{pol}_{safe_name(pick)}.png"),
                    window_days=60, lowess_frac=0.10,
                    acceptable_band=None,
                    max_date_ticks=10,
                    put_skill_outside=True,
                )

                err_pct_folder = ensure_dir(os.path.join(outdir, "error_vs_pct_release_multi"))
                plot_error_vs_flow_percentile_enhanced_multi(
                    obs=obs_release,
                    sim_dict=sim_release_dict,
                    title_prefix=f"{res} {pol} {pick} ({PLOT_START}–{PLOT_END}) — Release",
                    save_path=os.path.join(err_pct_folder, f"error_vs_flow_percentile_release_{res}_{pol}_{safe_name(pick)}.png"),
                    acceptable_band=None,
                    lowess_frac=0.10,
                    max_x_ticks=6,
                )

                if obs_release is not None:
                    plot_seasonal_decadal_panels(
                        df_obs=obs_release.to_frame(res),
                        df_sim=pywr_Rv.to_frame(res),
                        reservoirs=[res],
                        period_label=f"{res} {pol} {pick} ({PLOT_START}–{PLOT_END})",
                        save_folder=os.path.join(outdir, "error_vs_pct_seasons_release"),
                    )

                # optional saves & comparisons
                tag = f"{res}_{pol}_{safe_name(pick)}"
                if args.save_series:
                    indie_R.to_csv(os.path.join(outdir, f"{tag}_indie_release.csv"), header=True)
                    indie_S.to_csv(os.path.join(outdir, f"{tag}_indie_storage.csv"), header=True)
                    pywr_Rv.to_csv(os.path.join(outdir, f"{tag}_pywr_release.csv"), header=True)
                    pywr_Sv.to_csv(os.path.join(outdir, f"{tag}_pywr_storage.csv"), header=True)
                    if def_Rv is not None:
                        def_Rv.to_csv(os.path.join(outdir, f"{tag}_default_release.csv"), header=True)
                    if def_Sv is not None:
                        def_Sv.to_csv(os.path.join(outdir, f"{tag}_default_storage.csv"), header=True)

                if args.compare_series:
                    compare_series(indie_R, pywr_Rv, f"{res}:{pol}:{pick} release (indie vs pywr)",
                                   rtol=args.rtol, atol=args.atol)
                    compare_series(indie_S, pywr_Sv, f"{res}:{pol}:{pick} storage (indie vs pywr)",
                                   rtol=args.rtol, atol=args.atol)

    print("DONE.")


if __name__ == "__main__":
    main()