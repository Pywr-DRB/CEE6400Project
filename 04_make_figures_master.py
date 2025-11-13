#!/usr/bin/env python3
# Research/CEE6400Project/04_make_figures_master.py
"""
Unified figure generator (minimal wiring only).

Figures:
- Fig1: Pareto front comparison
- Fig2: Parallel axes (with Baseline row)
- Fig3: System dynamics for selected picks (independent sim)
- Fig4: Validation 9-panel (independent vs obs; optional Pywr overlays if provided)
- Fig5: Objective & decision ranges by policy (per-reservoir visual)

All figure settings, colors, objective labels/bounds, and senses come
from methods.config and methods.plotting.* modules.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- project config & utilities (source of truth) ----------------
from methods.config import (
    # Paths & run setup
    OUTPUT_DIR, FIG_DIR, PROCESSED_DATA_DIR, NFE, SEED, ISLANDS,
    # Entities
    reservoir_options as RESERVOIR_NAMES,
    policy_type_options as POLICY_TYPES,
    reservoir_capacity,
    # Objectives & plotting senses
    OBJ_LABELS, OBJ_FILTER_BOUNDS, SENSES_ALL,
    # Baseline discovery (these should live in config)
    BASELINE_DIR_NAME, BASELINE_INFLOW_TAG, VAL_START, VAL_END,
)

# Styles & colors (source of truth)
from methods.plotting.styles import policy_type_colors as POLICY_COLORS
from methods.plotting.styles import ADVANCED_COLORS  # for unified picks legend

# Selection helpers (source of truth)
from methods.plotting.selection_unified import (
    build_unified_picks,
    filter_better_than_baseline,
    append_baseline_row,                
    DESIRED_PICKS_ORDER,
    baseline_series_from_df,           
)

# Loaders / sims / plotters
from methods.load.results import load_results
from methods.load.observations import get_observational_training_data
from methods.reservoir.model import Reservoir

from methods.plotting.plot_pareto_front_comparison import plot_pareto_front_comparison
from methods.plotting.plot_parallel_axis_baseline import (
    custom_parallel_coordinates as parallel_with_baseline,
)
from methods.plotting.plot_parallel_axis import (
    custom_parallel_coordinates as parallel_no_baseline,
)
from methods.plotting.plot_reservoir_storage_release_distributions import plot_storage_release_distributions
from methods.plotting.plot_release_storage_9panel import plot_release_storage_9panel

# Optional: per-reservoir “policy range” figure generator
from methods.plotting.plot_bounds_tables import make_reservoir_visual

DISABLE_BASELINE_FILTER = True  # set False to re-enable
# ------------------------------- helpers -------------------------------------

def _obj_cols() -> List[str]:
    """Axis labels (ordered) from config."""
    return list(OBJ_LABELS.values())

def _sense_is_max(col: str) -> bool:
    return str(SENSES_ALL.get(col, "max")).lower().startswith("max")

def _minmaxs_all() -> List[str]:
    return ['max' if _sense_is_max(c) else 'min' for c in _obj_cols()]

def _ensure_fig_subdirs():
    (Path(FIG_DIR) / "fig1_pareto_front_comparison").mkdir(parents=True, exist_ok=True)
    (Path(FIG_DIR) / "fig2_parallel_axes").mkdir(parents=True, exist_ok=True)
    (Path(FIG_DIR) / "fig3_dynamics").mkdir(parents=True, exist_ok=True)
    (Path(FIG_DIR) / "fig4_validation_9panel").mkdir(parents=True, exist_ok=True)
    (Path(FIG_DIR) / "fig5_policy_range_viz").mkdir(parents=True, exist_ok=True)

def _baseline_dir() -> Path:
    return Path(FIG_DIR) / f"{BASELINE_DIR_NAME}_{BASELINE_INFLOW_TAG}"

def load_baseline_metrics_df(reservoir: str) -> Optional[pd.DataFrame]:
    """
    Expect CSV created by your baseline run:
      baseline_objectives_{reservoir}_{VAL_START}_to_{VAL_END}.csv
    with columns: ['metric', 'pywr_baseline'] (wide also ok; we only pass through).
    """
    p = _baseline_dir() / f"baseline_objectives_{reservoir}_{VAL_START}_to_{VAL_END}.csv"
    if not p.exists():
        print(f"[BASELINE] Missing for {reservoir}: {p.name}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[BASELINE] Could not read {p.name}: {e}")
        return None

def _params_for_row(var_df: pd.DataFrame, idx_row) -> np.ndarray:
    """Robust row extraction by label or integer position."""
    try:
        return var_df.loc[int(idx_row)].to_numpy(dtype=float)
    except Exception:
        return var_df.iloc[int(idx_row)].to_numpy(dtype=float)

# -------------------------------- main flow -----------------------------------

def main():
    _ensure_fig_subdirs()

    obj_cols = _obj_cols()
    minmaxs = _minmaxs_all()

    # Robust color dicts (include Baseline & Other)
    policy_colors_with_baseline = {**POLICY_COLORS, "Baseline": "black"}
    adv_colors_with_baseline = {**ADVANCED_COLORS}
    adv_colors_with_baseline.setdefault("Baseline", "black")
    adv_colors_with_baseline.setdefault("Other", "lightgray")

    # ====== Load all results & build unified selections ======
    solution_objs: Dict[str, Dict[str, pd.DataFrame]] = {}
    solution_vars: Dict[str, Dict[str, pd.DataFrame]] = {}
    pick_maps:     Dict[str, Dict[str, Dict[str, int]]] = {}
    baselines:     Dict[str, Optional[pd.DataFrame]] = {}
    ideal_point = [1.0, 1.0]

    for res in RESERVOIR_NAMES:
        print(f"\n=== Reservoir: {res} ===")
        baselines[res] = load_baseline_metrics_df(res)
        solution_objs[res], solution_vars[res], pick_maps[res] = {}, {}, {}

        for pol in POLICY_TYPES:
            csv = Path(OUTPUT_DIR) / f"MMBorg_{ISLANDS}M_{pol}_{res}_nfe{NFE}_seed{SEED}.csv"
            try:
                obj_df, var_df = load_results(
                    str(csv),
                    obj_labels=OBJ_LABELS,
                    filter=True,
                    obj_bounds=OBJ_FILTER_BOUNDS
                )
            except Exception as e:
                print(f"[WARN] load failed {res}/{pol}: {e}")
                obj_df, var_df = pd.DataFrame(), pd.DataFrame()

            if obj_df.empty:
                print(f"[INFO] No solutions for {res}/{pol}.")
                continue

            # If a baseline metrics table exists, optionally filter vs baseline; otherwise, keep as-is.
            # if baselines[res] is not None:
            #     try:
            #         obj_df = filter_better_than_baseline(obj_df, baselines[res], margin=0.0)
            #     except Exception as e:
            #         print(f"[INFO] Baseline filter skipped for {res}/{pol}: {e}")
            # If a baseline exists AND we're not disabling, filter; else keep everything.
            if (not DISABLE_BASELINE_FILTER) and (baselines[res] is not None):
                try:
                    obj_df = filter_better_than_baseline(obj_df, baselines[res], margin=0.0)
                except Exception as e:
                    print(f"[INFO] Baseline filter skipped for {res}/{pol}: {e}")
            if obj_df.empty:
                print(f"[INFO] After filter, none left for {res}/{pol}.")
                continue

            solution_objs[res][pol] = obj_df
            solution_vars[res][pol] = var_df
            pick_maps[res][pol]     = build_unified_picks(obj_df, obj_cols)

    # ---------- diagnostics ----------
    def summarize_ranges(solution_objs, cols):
        for res, pols in solution_objs.items():
            print(f"\n[RANGES] {res}")
            for pol, df in pols.items():
                print(f"  {pol}:")
                for c in cols:
                    if c in df.columns:
                        v = pd.to_numeric(df[c], errors="coerce")
                        print(
                            f"    {c:24s} "
                            f"min={v.min():.4g} p25={v.quantile(0.25):.4g} "
                            f"med={v.median():.4g} p75={v.quantile(0.75):.4g} "
                            f"max={v.max():.4g} NaN={v.isna().sum()}"
                        )

    def summarize_param_ranges(solution_vars: dict):
        for res, pols in solution_vars.items():
            print(f"\n[PARAM RANGES] {res}")
            for pol, df in pols.items():
                if df is None or df.empty:
                    continue
                print(f"  {pol}:")
                for c in df.columns:
                    if not c.lower().startswith("obj"):
                        v = pd.to_numeric(df[c], errors="coerce")
                        if v.notna().any():
                            print(
                                f"    {c:24s} "
                                f"min={v.min():.4g} p25={v.quantile(0.25):.4g} "
                                f"med={v.median():.4g} p75={v.quantile(0.75):.4g} "
                                f"max={v.max():.4g} NaN={v.isna().sum()}"
                            )

    summarize_ranges(solution_objs, obj_cols)
    summarize_param_ranges(solution_vars)

    # ====== FIG 1: Pareto front comparison (object-based) ======
    print("\n#### Figure 1 - Pareto Front Comparison #####")
    for reservoir in RESERVOIR_NAMES:
        obj_dfs, labels = [], []
        for policy in POLICY_TYPES:
            df = solution_objs.get(reservoir, {}).get(policy)
            if df is not None and not df.empty:
                obj_dfs.append(df)
                labels.append(policy)
        if not obj_dfs:
            print(f"[Fig1] Skip {reservoir}: no solutions.")
            continue

        fname = Path(FIG_DIR, "fig1_pareto_front_comparison", f"{reservoir}.png")
        try:
            plot_pareto_front_comparison(
                obj_dfs, labels,
                obj_cols=["Release NSE", "Storage NSE"],
                ideal=[1.0, 1.0],
                title=f"Pareto Front Comparison – {reservoir}",
                fname=str(fname)
            )
        except Exception as e:
            print(f"[Fig1] {reservoir} failed: {e}")

    # ====== FIG 2: Parallel axes (per policy; with Baseline row) ======
    print("\n#### Figure 2 - Parallel Axes ####")
    for res in RESERVOIR_NAMES:
        base_df = baselines.get(res)

        for pol in POLICY_TYPES:
            obj_df = solution_objs.get(res, {}).get(pol)
            if obj_df is None or obj_df.empty:
                continue

            # A) All solutions per reservoir/policy (+ Baseline as a pseudo-row)
            dfA = obj_df.copy()
            dfA["policy"] = pol
            dfA = append_baseline_row(dfA, base_df, label_col="policy", label_value="Baseline")
            fA = Path(FIG_DIR, "fig2_parallel_axes", f"all_sols_{res}_{pol}.png")

            try:
                parallel_with_baseline(
                    objs=dfA,
                    columns_axes=obj_cols,
                    axis_labels=obj_cols,
                    ideal_direction="top",
                    minmaxs=minmaxs,
                    color_by_continuous=None,
                    color_by_categorical="policy",
                    color_dict_categorical=policy_colors_with_baseline,
                    zorder_by=None, zorder_num_classes=None, zorder_direction="ascending",
                    alpha_base=0.35, brushing_dict=None, alpha_brush=0.08,
                    lw_base=1.2, fontsize=9, figsize=(11, 6),
                    bottom_pad=0.22, legend_pad=0.08, fname=str(fA)
                )
            except Exception as e:
                print(f"[Fig2A] {res}/{pol} failed: {e}")

            # B) Unified highlighted selections (+ Baseline)
            picks = pick_maps.get(res, {}).get(pol, {})
            dfH = obj_df.copy()
            dfH["highlight"] = "Other"
            for lab, idx in picks.items():
                if idx in dfH.index:
                    dfH.loc[idx, "highlight"] = lab
            dfH = append_baseline_row(dfH, base_df, label_col="highlight", label_value="Baseline")

            local_colors = {**adv_colors_with_baseline}
            for oc in obj_cols:
                local_colors.setdefault(f"Best {oc}", "crimson" if _sense_is_max(oc) else "teal")

            fB = Path(FIG_DIR, "fig2_parallel_axes", f"unified_picks_{res}_{pol}.png")
            try:
                parallel_with_baseline(
                    objs=dfH,
                    columns_axes=obj_cols,
                    axis_labels=obj_cols,
                    ideal_direction="top",
                    minmaxs=minmaxs,
                    color_by_continuous=None,
                    color_by_categorical="highlight",
                    color_dict_categorical=local_colors,
                    zorder_by=0, zorder_num_classes=10, zorder_direction="ascending",
                    alpha_base=0.9, brushing_dict=None, alpha_brush=0.1,
                    lw_base=1.5, fontsize=9, figsize=(12, 6),
                    bottom_pad=0.28, legend_pad=0.18, fname=str(fB)
                )
            except Exception as e:
                print(f"[Fig2B] {res}/{pol} failed: {e}")

        # C) All policies combined (+ Baseline)
        frames = []
        for pol in POLICY_TYPES:
            df = solution_objs.get(res, {}).get(pol)
            if df is None or df.empty:
                continue
            x = df.copy()
            x["policy"] = pol
            frames.append(x)
        if frames:
            combined = pd.concat(frames, axis=0).sample(frac=1).reset_index(drop=True)
            combined = append_baseline_row(combined, base_df, label_col="policy", label_value="Baseline")
            fC = Path(FIG_DIR, "fig2_parallel_axes", f"all_policies_{res}.png")
            try:
                parallel_with_baseline(
                    objs=combined,
                    columns_axes=obj_cols,
                    axis_labels=obj_cols,
                    ideal_direction="top",
                    minmaxs=minmaxs,
                    color_by_continuous=None,
                    color_by_categorical="policy",
                    color_dict_categorical=policy_colors_with_baseline,
                    zorder_by=None, zorder_num_classes=None, zorder_direction="ascending",
                    alpha_base=0.3, brushing_dict=None, alpha_brush=0.1,
                    lw_base=1.0, fontsize=9, figsize=(11, 6),
                    bottom_pad=0.22, legend_pad=0.08, fname=str(fC)
                )
            except Exception as e:
                print(f"[Fig2C] {res} failed: {e}")

    # ====== FIG 3: Dynamics (independent sim for DESIRED picks) ======
    print("\n##### Figure 3 - System dynamics (independent) #####")
    for res in RESERVOIR_NAMES:
        inflow_df, release_df, storage_df = get_observational_training_data(
            reservoir_name=res,
            data_dir=PROCESSED_DATA_DIR,
            as_numpy=False,
            inflow_type="inflow_pub"
        )
        if inflow_df.empty or storage_df.empty:
            print(f"[Fig3] Skip {res}: missing obs.")
            continue

        dates = inflow_df.index
        inflow_obs  = inflow_df.values
        release_obs = (release_df.values if release_df is not None else None)
        storage_obs = storage_df.values

        for pol in POLICY_TYPES:
            obj_df = solution_objs.get(res, {}).get(pol)
            var_df = solution_vars.get(res, {}).get(pol)
            if obj_df is None or var_df is None or obj_df.empty or var_df.empty:
                continue

            picks = pick_maps.get(res, {}).get(pol, {})
            want_labels = list(DESIRED_PICKS_ORDER) + [f"Best {c}" for c in obj_cols]
            want = [(lab, idx) for lab, idx in picks.items() if lab in want_labels]

            for k, (label, idx_row) in enumerate(want, start=1):
                theta = _params_for_row(var_df, idx_row)

                sim = Reservoir(
                    inflow=inflow_obs, dates=dates,
                    capacity=reservoir_capacity[res],
                    policy_type=pol, policy_params=theta,
                    initial_storage=storage_obs[0], name=res
                )
                sim.run()
                sim_S = sim.storage_array.flatten()
                sim_R = (sim.release_array + sim.spill_array).flatten()

                out = Path(FIG_DIR, "fig3_dynamics", f"{res}_{pol}_{label.replace(' ', '_')}_{k}")
                try:
                    fig, _ = plot_storage_release_distributions(
                        obs_storage=storage_obs.flatten(),
                        obs_release=(release_obs.flatten() if release_obs is not None else None),
                        sim_storage=sim_S, sim_release=sim_R,
                        obs_inflow=inflow_obs.flatten(), datetime=dates,
                        storage_distribution=True, inflow_scatter=False, inflow_vs_release=True,
                        fname=str(out) + "__quantIR.png"
                    )
                    plt.close(fig)
                except Exception as e:
                    print(f"[Fig3] {res}/{pol}/{label} failed: {e}")

    # ====== FIG 4: Validation 9-panel (independent vs obs; overlays optional) ======
    print("\n##### Figure 4 - Validation 9-panel #####")
    VSTART, VEND, VINFLOW = VAL_START, VAL_END, "inflow_pub"  # use config dates
    for res in RESERVOIR_NAMES:
        inflow_df, release_df, storage_df = get_observational_training_data(
            reservoir_name=res, data_dir=PROCESSED_DATA_DIR, as_numpy=False, inflow_type=VINFLOW
        )
        if inflow_df.empty:
            print(f"[Fig4] Skip {res}: no inflow data.")
            continue

        slicer = slice(VSTART, VEND)
        inflow_win  = inflow_df.loc[slicer][res] if res in inflow_df.columns else inflow_df.loc[slicer].iloc[:, 0]
        release_win = (release_df.loc[slicer][res] if (release_df is not None and res in release_df.columns) else None)
        storage_win = storage_df.loc[slicer][res] if res in storage_df.columns else storage_df.loc[slicer].iloc[:, 0]
        if len(inflow_win) == 0 or len(storage_win) == 0:
            print(f"[Fig4] Skip {res}: empty validation window.")
            continue

        dt_index = inflow_win.index

        for pol in POLICY_TYPES:
            obj_df = solution_objs.get(res, {}).get(pol)
            var_df = solution_vars.get(res, {}).get(pol)
            if obj_df is None or var_df is None or obj_df.empty or var_df.empty:
                continue

            picks = pick_maps.get(res, {}).get(pol, {})
            want_labels = list(DESIRED_PICKS_ORDER) + [f"Best {c}" for c in obj_cols]
            want = [(lab, idx) for lab, idx in picks.items() if lab in want_labels]

            for k, (label, idx_row) in enumerate(want, start=1):
                theta = _params_for_row(var_df, idx_row)

                model = Reservoir(
                    inflow=inflow_win.values, dates=dt_index,
                    capacity=reservoir_capacity[res],
                    policy_type=pol, policy_params=theta,
                    initial_storage=float(storage_win.iloc[0]), name=res
                )
                model.run()
                sim_S = pd.Series(model.storage_array.flatten(), index=dt_index, name=res)
                sim_R = pd.Series((model.release_array + model.spill_array).flatten(), index=dt_index, name=res)

                out = Path(FIG_DIR, "fig4_validation_9panel",
                           f"{res}_{pol}_{label.replace(' ', '_')}_{k}_9panel.png")

                try:
                    plot_release_storage_9panel(
                        reservoir=res,
                        sim_release=sim_R, sim_storage_MG=sim_S,
                        obs_release=(release_win if release_win is not None else None),
                        obs_storage_MG=storage_win,
                        # Pywr overlays (set to real series if you have them):
                        pywr_param_release=None,
                        pywr_param_storage_MG=None,
                        pywr_default_release=None,
                        pywr_default_storage_MG=None,
                        # Optional NOR %capacity bands (set if available):
                        nor_lo_pct=None,
                        nor_hi_pct=None,
                        # Labels
                        start=str(VSTART), end=str(VEND),
                        policy_label=pol, pick_label=label,
                        save_path=str(out),
                    )
                    print(f"[Fig4] Saved {out}")
                except Exception as e:
                    print(f"[Fig4] {res}/{pol}/{label} failed: {e}")

    # ====== FIG 5: Policy objective & decision ranges ======
    print("\n##### Figure 5 - Policy objective/decision ranges #####")
    outdir5 = Path(FIG_DIR) / "fig5_policy_range_viz"
    outdir5.mkdir(parents=True, exist_ok=True)
    for res in RESERVOIR_NAMES:
        try:
            make_reservoir_visual(res, outdir5)
        except Exception as e:
            print(f"[Fig5] Skip {res}: {e}")

    print("\nDONE.\n")


if __name__ == "__main__":
    main()
