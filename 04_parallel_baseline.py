import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import re
import os
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from methods.config import (
    NFE, SEED, ISLANDS,
    OBJ_LABELS, OBJ_FILTER_BOUNDS,
    reservoir_options, policy_type_options,
    OUTPUT_DIR, FIG_DIR, PROCESSED_DATA_DIR,
    reservoir_capacity, n_rbfs, n_rbf_inputs, n_segments, n_pwl_inputs
)
from methods.reservoir.model import Reservoir
from methods.load.results import load_results
from methods.load.observations import get_observational_training_data

from methods.plotting.plot_parallel_axis_baseline import custom_parallel_coordinates
from methods.plotting.selection_utils import (
    compute_and_apply_advanced_highlights,
    ADVANCED_COLORS
)

BASELINE_OPT_BY_RES = {
    "blueMarsh": {
        "neg_nse": -0.185603,
        "Q20_abs_pbias_release": 83.524277,
        "neg_kge": 1.302547,
        "Q20_abs_pbias_storage": 36.457770,
    },
    "beltzvilleCombined": {
        "neg_nse": 0.630813,
        "Q20_abs_pbias_release": 207.037890,
        "neg_kge": 4.173522,
        "Q20_abs_pbias_storage": 76.270459,
    },
    "fewalter": {
        "neg_nse": -0.432571,
        "Q20_abs_pbias_release": 22.021222,
        "neg_kge": -0.499028,
        "Q20_abs_pbias_storage": 259.943947,
    },
    "prompton": {
        "neg_nse": -0.550374,
        "Q20_abs_pbias_release": 11.119965,
        "neg_kge": 2.596013,
        "Q20_abs_pbias_storage": 309.722063,
    },
}


# def _flip_if_neg(key: str, val: float) -> float:
#     k = str(key).lower()
#     return (-float(val)) if (k.startswith("neg_") or k.endswith("_neg") or "_neg_" in k) else float(val)

# def baseline_series_from_opt_keys(reservoir: str) -> pd.Series | None:
#     opt_vals = BASELINE_OPT_BY_RES.get(reservoir)
#     if not opt_vals:
#         print(f"[BASELINE] No in-memory baseline for reservoir='{reservoir}'.")
#         return None
#     out = {}
#     for opt_key, pretty_label in OBJ_LABELS.items():
#         out[pretty_label] = _flip_if_neg(opt_key, opt_vals.get(opt_key, np.nan))
#     obj_cols = list(OBJ_LABELS.values())
#     return pd.Series(out).reindex(obj_cols)

import re

# explicit “what pretty label comes from which baseline key”
# _BASELINE_ALIASES = {
#     "Release NSE":      "neg_nse",
#     "Q20 Abs % Bias":   "Q20_abs_pbias",
#     "Storage KGE":      "neg_kge",
# }
_BASELINE_ALIASES = {
    "Release NSE":                   "neg_nse",
    "Q20 Abs % Bias (Release)":      "Q20_abs_pbias_release",
    "Storage KGE":                   "neg_kge",
    "Q80 Abs % Bias (Storage)":      "Q80_abs_pbias_storage",
}

def _flip_if_neg(key: str, val: float) -> float:
    k = str(key).lower()
    return (-float(val)) if (k.startswith("neg_") or k.endswith("_neg") or "_neg_" in k) else float(val)

def _canon(s: str) -> str:
    """lower + strip non-alnum for fuzzy fallback"""
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def baseline_series_from_opt_keys(reservoir: str) -> pd.Series | None:
    """
    Build a Series of baseline metrics in *plot label* space (obj_cols),
    pulling values from BASELINE_OPT_BY_RES[reservoir] and flipping signs
    where needed.
    """
    opt_vals = BASELINE_OPT_BY_RES.get(reservoir)
    if not opt_vals:
        print(f"[BASELINE] No in-memory baseline for reservoir='{reservoir}'.")
        return None

    # We’ll produce values in the order of your plotted columns
    obj_cols = list(OBJ_LABELS.values())

    # 1) start from explicit aliases
    pretty_to_key = dict(_BASELINE_ALIASES)

    # 2) if OBJ_LABELS can directly invert and matches your baseline keys, allow that too
    #    (this covers cases where your OBJ_LABELS keys already equal the baseline keys)
    for opt_key, pretty in OBJ_LABELS.items():
        if opt_key in opt_vals:
            pretty_to_key.setdefault(pretty, opt_key)

    # 3) fuzzy fallback: if still unmapped, try to match by canonicalized tokens
    canon_baseline_keys = { _canon(k): k for k in opt_vals.keys() }
    canon_opt_keys      = { _canon(k): k for k in OBJ_LABELS.keys() }
    for pretty in obj_cols:
        if pretty in pretty_to_key:
            continue
        # try: find OBJ_LABELS key for this pretty, then map to baseline by canon
        # (useful if the OBJ_LABELS key is close to the baseline key but not identical)
        # e.g., "Q20_log_neg_nse" vs "q20lognegnse"
        # a) get the opt-key (from OBJ_LABELS) if any
        candidates = [k for k, p in OBJ_LABELS.items() if p == pretty]
        if candidates:
            ck = _canon(candidates[0])
            if ck in canon_baseline_keys:
                pretty_to_key[pretty] = canon_baseline_keys[ck]
                continue
        # b) last-ditch: try to canon the pretty label itself and look in baseline keys
        cp = _canon(pretty)
        if cp in canon_baseline_keys:
            pretty_to_key[pretty] = canon_baseline_keys[cp]

    # 4) build the output series in obj_cols order
    out_vals = {}
    missing = []
    for pretty in obj_cols:
        k = pretty_to_key.get(pretty, None)
        if k is None or (k not in opt_vals):
            out_vals[pretty] = np.nan
            missing.append(pretty)
        else:
            out_vals[pretty] = _flip_if_neg(k, opt_vals[k])

    if missing:
        print(f"[BASELINE MAP] Missing for {reservoir}: {', '.join(missing)}")

    return pd.Series(out_vals).reindex(obj_cols)

def append_baseline_row(obj_df: pd.DataFrame, baseline_metrics: pd.Series | None,
                        label_col: str, label_value: str = "Baseline",
                        obj_cols: list[str] | None = None) -> pd.DataFrame:
    if baseline_metrics is None or obj_df is None or obj_df.empty:
        return obj_df
    out = obj_df.copy()
    if obj_cols is None:
        obj_cols = [c for c in out.columns if c in baseline_metrics.index]
    row = {}
    for c in obj_cols:
        row[c] = float(baseline_metrics.get(c, np.nan))
    for c in out.columns:
        if c in row: continue
        row[c] = out.iloc[0][c] if len(out) else np.nan
    row[label_col] = label_value
    out = pd.concat([out, pd.DataFrame([row], columns=out.columns)], ignore_index=True)
    for c in obj_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# Labels/colors
POLICY_TYPES = policy_type_options
RESERVOIR_NAMES = reservoir_options

policy_colors = {
    "STARFIT": "blue",
    "RBF": "orange",
    "PWL": "green",
    "Baseline": "black",
}
highlight_colors = {
    "Best Release NSE": "red",
    "Best Storage KGE": "green",
    "Best Average NSE": "purple",
    "Best Average All": "blue",
    "Baseline": "black",
    "Other": "lightgray",
}


senses_all = {
    "Release NSE": "max",
    "Q20 Abs % Bias (Release)": "min",
    "Storage KGE": "max",
    "Q80 Abs % Bias (Storage)": "min",
}


REMAKE_PARALLEL_PLOTS = True

if __name__ == "__main__":
    inflow_obs, release_obs, storage_obs = get_observational_training_data(
        reservoir_name="prompton", data_dir=PROCESSED_DATA_DIR,
        as_numpy=False, inflow_type="inflow_pub"
    )
    print(f"Inflows shape: {inflow_obs.shape}")

    Path(FIG_DIR, "fig_parallel_axes").mkdir(parents=True, exist_ok=True)

    solution_objs, solution_vars = {}, {}
    solution_adv_maps, solution_adv_cands = {}, {}

    obj_labels = OBJ_LABELS
    obj_cols = list(obj_labels.values())
    minmaxs_all = ["max" if senses_all[c] == "max" else "min" for c in obj_cols]

    # --- load per reservoir/policy ---
    for reservoir_name in RESERVOIR_NAMES:
        solution_objs[reservoir_name] = {}
        solution_vars[reservoir_name] = {}

        for policy_type in POLICY_TYPES:
            fname = f"{OUTPUT_DIR}/MMBorg_{ISLANDS}M_{policy_type}_{reservoir_name}_nfe{NFE}_seed{SEED}.csv"
            try:
                obj_df, var_df = load_results(
                    fname, obj_labels=obj_labels, filter=True, obj_bounds=OBJ_FILTER_BOUNDS
                )
                print(f"[FLT] {reservoir_name}/{policy_type}: {len(obj_df)} rows")
            except Exception as e:
                print(f"[WARN] load failed {reservoir_name}/{policy_type}: {e}")
                obj_df, var_df = pd.DataFrame(), pd.DataFrame()

            if len(obj_df) == 0:
                print(f"Warning: No solutions for {policy_type}/{reservoir_name}.")
                continue

            # focal picks
            idx_best_release = obj_df["Release NSE"].idxmax()
            idx_best_storage = obj_df["Storage KGE"].idxmax()
            idx_best_average = obj_df[["Release NSE", "Storage KGE"]].mean(axis=1).idxmax()

            min_obj_df = obj_df.copy()
            min_obj_df["Release NSE"] = -min_obj_df["Release NSE"]
            min_obj_df["Storage KGE"] = -min_obj_df["Storage KGE"]
            scaled_min_obj_df = (min_obj_df - min_obj_df.min()) / (min_obj_df.max() - min_obj_df.min())
            idx_best_all_avg = scaled_min_obj_df.mean(axis=1).idxmin()

            # legacy highlight labels
            highlight_label_dict = {
                idx_best_release: "Best Release NSE",
                idx_best_storage: "Best Storage KGE",
                idx_best_average: "Best Average NSE",
                idx_best_all_avg: "Best Average All",
            }
            obj_df["highlight"] = [highlight_label_dict.get(idx, "Other") for idx in obj_df.index]

            # advanced highlight labels
            
            adv_objectives = obj_cols
            obj_df_aug, cand_df, cand_map = compute_and_apply_advanced_highlights(
                obj_df, objectives=adv_objectives, senses=senses_all, bounds=OBJ_FILTER_BOUNDS,
                eps_qs=(0.5, 0.8), add_k_diverse=2, include_hv=False, out_label_col="highlight_adv"
            )

            solution_objs[reservoir_name][policy_type] = obj_df_aug
            solution_vars[reservoir_name][policy_type] = var_df
            solution_adv_maps.setdefault(reservoir_name, {})[policy_type] = cand_map
            solution_adv_cands.setdefault(reservoir_name, {})[policy_type] = cand_df

    def has_solutions(reservoir_name, policy_type):
        return (reservoir_name in solution_objs and
                policy_type in solution_objs[reservoir_name] and
                solution_objs[reservoir_name][policy_type] is not None and
                len(solution_objs[reservoir_name][policy_type]) > 0)

    def reservoir_has_any(reservoir_name):
        d = solution_objs.get(reservoir_name, {})
        return any((df is not None) and (len(df) > 0) for df in d.values())

    # --------- plotting ---------
    print("#### Figure 2 - Parallel Axis Plot #####")
    if REMAKE_PARALLEL_PLOTS:

        # (A) all solutions per reservoir & policy (color by policy; Baseline added to 'policy')
        for reservoir_name in RESERVOIR_NAMES:
            for policy_type in POLICY_TYPES:
                if not has_solutions(reservoir_name, policy_type):
                    continue
                obj_df = solution_objs[reservoir_name][policy_type].copy()
                #print the head of the dataframe
                print(obj_df.head())
                obj_df["policy"] = policy_type
                baseline_series = baseline_series_from_opt_keys(reservoir_name)
                obj_df = append_baseline_row(obj_df, baseline_series, label_col="policy",
                                             label_value="Baseline", obj_cols=obj_cols)
                print(obj_df.head())
                fname = f"{FIG_DIR}/fig_parallel_axes/all_sols_{reservoir_name}_{policy_type}.png"
                custom_parallel_coordinates(
                    objs=obj_df, columns_axes=obj_cols, axis_labels=obj_cols,
                    ideal_direction="top", minmaxs=minmaxs_all,
                    color_by_continuous=None, color_by_categorical="policy",
                    color_dict_categorical=policy_colors,
                    zorder_by=0, zorder_num_classes=10, zorder_direction="ascending",
                    alpha_base=0.7, brushing_dict=None, alpha_brush=0.05,
                    lw_base=1.5, fontsize=9, figsize=(11, 6),
                    bottom_pad=0.22, legend_pad=0.08, fname=fname
                )

        # (B) best solutions (color by legacy highlight; Baseline added to 'highlight')
        for reservoir_name in RESERVOIR_NAMES:
            for policy_type in POLICY_TYPES:
                if not has_solutions(reservoir_name, policy_type):
                    continue
                obj_df = solution_objs[reservoir_name][policy_type].copy()
                print(obj_df.head())
                if "highlight" not in obj_df.columns:
                    obj_df["highlight"] = "Other"
                baseline_series = baseline_series_from_opt_keys(reservoir_name)
                obj_df = append_baseline_row(obj_df, baseline_series, label_col="highlight",
                                             label_value="Baseline", obj_cols=obj_cols)
                print(obj_df.head())
                fname = f"{FIG_DIR}/fig_parallel_axes/compare_best_sols_{reservoir_name}_{policy_type}.png"
                custom_parallel_coordinates(
                    objs=obj_df, columns_axes=obj_cols, axis_labels=obj_cols,
                    ideal_direction="top", minmaxs=minmaxs_all,
                    color_by_continuous=None, color_by_categorical="highlight",
                    color_dict_categorical=highlight_colors,
                    zorder_by=0, zorder_num_classes=10, zorder_direction="ascending",
                    alpha_base=0.9, brushing_dict=None, alpha_brush=0.1,
                    lw_base=1.5, fontsize=9, figsize=(11, 6),
                    bottom_pad=0.22, legend_pad=0.08, fname=fname
                )

        # (C) advanced selections (color by highlight_adv; Baseline added to 'highlight_adv')
        for reservoir_name in RESERVOIR_NAMES:
            for policy_type in POLICY_TYPES:
                if not has_solutions(reservoir_name, policy_type):
                    continue
                obj_df_adv = solution_objs[reservoir_name][policy_type].copy()
                print(obj_df_adv.head())
                baseline_series = baseline_series_from_opt_keys(reservoir_name)
                obj_df_adv = append_baseline_row(
                    obj_df_adv, baseline_series, label_col="highlight_adv",
                    label_value="Baseline", obj_cols=obj_cols
                )
                print(obj_df_adv.head())
                fname = f"{FIG_DIR}/fig_parallel_axes/advanced_picks_{reservoir_name}_{policy_type}.png"
                custom_parallel_coordinates(
                    objs=obj_df_adv, columns_axes=obj_cols, axis_labels=obj_cols,
                    ideal_direction="top", minmaxs=minmaxs_all,
                    color_by_continuous=None, color_by_categorical="highlight_adv",
                    color_dict_categorical=ADVANCED_COLORS | {"Baseline": "black"},
                    zorder_by=0, zorder_num_classes=10, zorder_direction="ascending",
                    alpha_base=0.9, brushing_dict=None, alpha_brush=0.1,
                    lw_base=1.5, fontsize=9, figsize=(11, 6),
                    bottom_pad=0.3, legend_pad=0.2, fname=fname
                )

        # (D) all policies combined per reservoir (color by policy; Baseline added to 'policy')
        for reservoir_name in RESERVOIR_NAMES:
            if not reservoir_has_any(reservoir_name):
                continue
            obj_list = []
            for policy_type in POLICY_TYPES:
                if has_solutions(reservoir_name, policy_type):
                    df = solution_objs[reservoir_name][policy_type].copy()
                    df["policy"] = policy_type
                    obj_list.append(df)
            if not obj_list:
                continue
            combined_df = pd.concat(obj_list, axis=0)
            print(combined_df.head())
            combined_df["policy"] = combined_df["policy"].astype(str)
            baseline_series = baseline_series_from_opt_keys(reservoir_name)
            combined_df = append_baseline_row(combined_df, baseline_series, label_col="policy",
                                              label_value="Baseline", obj_cols=obj_cols)
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)
            print(combined_df.head())

            fname = f"{FIG_DIR}/fig_parallel_axes/all_policies_{reservoir_name}.png"
            custom_parallel_coordinates(
                objs=combined_df, columns_axes=obj_cols, axis_labels=obj_cols,
                ideal_direction="top", minmaxs=minmaxs_all,
                color_by_continuous=None, color_by_categorical="policy",
                color_dict_categorical=policy_colors,
                zorder_by=None, zorder_num_classes=None, zorder_direction="ascending",
                alpha_base=0.3, brushing_dict=None, alpha_brush=0.1,
                lw_base=1.0, fontsize=9, figsize=(11, 6),
                bottom_pad=0.22, legend_pad=0.08, fname=fname
            )

    print("DONE!")
