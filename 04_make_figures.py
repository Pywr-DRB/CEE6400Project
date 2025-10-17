import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import re
import os

warnings.filterwarnings("ignore")

from methods.config import NFE, SEED, ISLANDS
from methods.config import OBJ_LABELS, OBJ_FILTER_BOUNDS
from methods.config import reservoir_options, policy_type_options
from methods.config import OUTPUT_DIR, FIG_DIR, PROCESSED_DATA_DIR
from methods.config import reservoir_capacity

from methods.reservoir.model import Reservoir
from methods.load.results import load_results
from methods.load.observations import get_observational_training_data

from methods.plotting.plot_pareto_front_comparison import plot_pareto_front_comparison
from methods.plotting.plot_parallel_axis import custom_parallel_coordinates
from methods.plotting.plot_reservoir_storage_release_distributions import plot_storage_release_distributions
from methods.plotting.plot_release_storage_9panel import plot_release_storage_9panel
from methods.plotting.selection_utils import (
    compute_and_apply_advanced_highlights,  # one-call helper
    ADVANCED_COLORS
)

# ======= PARAM NAME MAPS (match run_simple_model.py) =======
from methods.config import n_rbfs, n_rbf_inputs, n_segments, n_pwl_inputs


def safe_name(label: str) -> str:
    """Turn any label into a filesystem-friendly token."""
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', str(label))
    s = re.sub(r'_+', '_', s).strip('_')
    return s if s else "pick"


def get_param_names_for_policy(policy: str):
    policy = str(policy).upper()
    if policy == "STARFIT":
        return [
            "NORhi_mu", "NORhi_min", "NORhi_max", "NORhi_alpha", "NORhi_beta",
            "NORlo_mu", "NORlo_min", "NORlo_max", "NORlo_alpha", "NORlo_beta",
            "Release_alpha1", "Release_alpha2", "Release_beta1", "Release_beta2",
            "Release_c", "Release_p1", "Release_p2",
        ]
    if policy == "RBF":
        labels = ["storage", "inflow", "doy"][:n_rbf_inputs]
        names = []
        for i in range(1, n_rbfs + 1):
            names.append(f"w{i}")
        for i in range(1, n_rbfs + 1):
            for v in labels:
                names.append(f"c{i}_{v}")
        for i in range(1, n_rbfs + 1):
            for v in labels:
                names.append(f"r{i}_{v}")
        return names
    if policy == "PWL":
        names = []
        block_labels = ["storage", "inflow", "day"][:n_pwl_inputs]
        for lab in block_labels:
            for k in range(1, n_segments):
                names.append(f"{lab}_x{k}")
            for k in range(1, n_segments + 1):
                names.append(f"{lab}_theta{k}")
        return names
    raise ValueError(f"Unknown policy '{policy}'")


def print_params_flat(policy_type: str, params_1d):
    """Flat index → name → value (works for all policies)."""
    names = get_param_names_for_policy(policy_type)
    assert len(params_1d) == len(names), f"Length mismatch: got {len(params_1d)} values, expected {len(names)}"
    print(f"\n--- Parameters ({policy_type}) ---")
    for i, (n, v) in enumerate(zip(names, params_1d)):
        print(f"[{i:02d}] {n:16s} = {float(v): .6f}")


def print_params_pretty(policy_type: str, params_1d):
    """
    Nicely grouped printers for RBF and PWL; STARFIT uses flat by design.
    """
    policy = policy_type.upper()
    names = get_param_names_for_policy(policy)
    assert len(params_1d) == len(names), f"Length mismatch: got {len(params_1d)} values, expected {len(names)}"

    if policy == "STARFIT":
        print_params_flat(policy, params_1d)
        return

    if policy == "RBF":
        print(f"\n--- Parameters (RBF) n_rbfs={n_rbfs}, n_inputs={n_rbf_inputs} ---")
        idx = 0
        print("Weights:")
        for i in range(1, n_rbfs + 1):
            print(f"  w{i} = {float(params_1d[idx]): .6f}")
            idx += 1
        print("Centers c[i, var]:")
        for i in range(1, n_rbfs + 1):
            row = []
            for var in ["storage", "inflow", "doy"][:n_rbf_inputs]:
                row.append(float(params_1d[idx])); idx += 1
            print(f"  c{i} = {row}")
        print("Scales r[i, var]:")
        for i in range(1, n_rbfs + 1):
            row = []
            for var in ["storage", "inflow", "doy"][:n_rbf_inputs]:
                row.append(float(params_1d[idx])); idx += 1
            print(f"  r{i} = {row}")
        return

    if policy == "PWL":
        print(f"\n--- Parameters (PWL) n_segments={n_segments}, n_inputs={n_pwl_inputs} ---")
        per_block = 2 * n_segments - 1
        blocks = ["storage", "inflow", "day"][:n_pwl_inputs]
        for b, lab in enumerate(blocks):
            block = params_1d[b*per_block:(b+1)*per_block]
            xs     = block[:n_segments-1]
            thetas = block[n_segments-1:]
            print(f"{lab.capitalize()} block:")
            for i, x in enumerate(xs, start=1):
                print(f"  x{i}     = {float(x): .6f}")
            for i, th in enumerate(thetas, start=1):
                print(f"  theta{i} = {float(th): .6f}")
        return

    print_params_flat(policy, params_1d)


POLICY_TYPES = policy_type_options
print(f"Policy types: {POLICY_TYPES}")
RESERVOIR_NAMES = reservoir_options
print(f"Reservoirs: {RESERVOIR_NAMES}")

REMAKE_PARALLEL_PLOTS = True
REMAKE_DYNAMICS_PLOTS = True

reservoir_labels = {
    'beltzvilleCombined': 'Beltzville',
    'fewalter': 'FE Walter',
    'prompton': 'Prompton',
    'blueMarsh': 'Blue Marsh',
}

policy_labels = {
    'STARFIT': 'STARFIT',
    'RBF': 'RBF',
    'PWL': 'PWL',
}

policy_colors = {
    'STARFIT': 'blue',
    'RBF': 'orange',
    'PWL': 'green',
}

# Sense map for all objectives (used for axis direction and advanced picks)
senses_all = {
    "Release NSE": "max",
    "Q20 Log Release NSE": "max",
    "Q80 Release Abs % Bias": "min",
    "Release Inertia": "max",
    "Storage KGE": "max",
    "Storage Inertia": "max",
}

# unified list of picks you want to run
DESIRED_PICKS = [
    "Best Release NSE", "Best Storage KGE", "Best Average NSE", "Best Average All",
    "Compromise L2 (Euclidean)", "Tchebycheff L∞", "Manhattan L1",
    "ε-constraint Release NSE ≥ Q50", "Diverse #1 (FPS)", "Diverse #2 (FPS)",
]


# ---------- NEW: helpers to safely query solutions ----------
def has_solutions(reservoir_name: str, policy_type: str) -> bool:
    """True iff filtered DF exists and is non-empty for (reservoir, policy)."""
    return (
        reservoir_name in solution_objs and
        policy_type in solution_objs[reservoir_name] and
        solution_objs[reservoir_name][policy_type] is not None and
        len(solution_objs[reservoir_name][policy_type]) > 0
    )


def reservoir_has_any(reservoir_name: str) -> bool:
    """True iff reservoir has at least one policy with solutions."""
    d = solution_objs.get(reservoir_name, {})
    return any((df is not None) and (len(df) > 0) for df in d.values())


def get_obj_df(reservoir_name: str, policy_type: str):
    return solution_objs.get(reservoir_name, {}).get(policy_type)


def _params_for_row(var_df: pd.DataFrame, row_idx):
    """Row-safe accessor: prefer .loc by index label, fall back to .iloc if needed."""
    try:
        return var_df.loc[row_idx].values
    except Exception:
        return var_df.iloc[int(row_idx)].values


if __name__ == "__main__":

    # Load one reservoir's obs to print basic info (kept from original)
    inflow_obs, release_obs, storage_obs = get_observational_training_data(
        reservoir_name='prompton',
        data_dir=PROCESSED_DATA_DIR,
        as_numpy=False,
        inflow_type='inflow_pub'  # 'inflow', 'inflow_scaled', 'inflow_pub'
    )
    print(f"Inflows shape: {inflow_obs.shape}")
    print(f"Datetime: {inflow_obs.index}")
    print(f"Min storage: {storage_obs.min()}")
    print(f"Max storage: {storage_obs.max()}")
    print(f"Min release: {release_obs.min()}")
    print(f"Max release: {release_obs.max()}")

    # --- ensure figure subfolders exist ---
    Path(FIG_DIR, "fig1_pareto_front_comparison").mkdir(parents=True, exist_ok=True)
    Path(FIG_DIR, "fig2_parallel_axes").mkdir(parents=True, exist_ok=True)
    Path(FIG_DIR, "fig3_dynamics").mkdir(parents=True, exist_ok=True)
    Path(FIG_DIR, "fig4_validation_9panel").mkdir(parents=True, exist_ok=True)

    ####################################################
    ### Load & process data ############################
    ####################################################
    solution_objs = {}       # dict[reservoir][policy] -> obj_df
    solution_vars = {}       # dict[reservoir][policy] -> var_df
    solution_adv_maps = {}   # dict[reservoir][policy] -> cand_map
    solution_adv_cands = {}  # dict[reservoir][policy] -> cand_df

    obj_labels = OBJ_LABELS
    obj_cols = list(obj_labels.values())

    # Direction per axis for plotting
    minmaxs_all = ['max' if senses_all[c] == 'max' else 'min' for c in obj_cols]

    # For each reservoir/policy: load raw + filtered; store only if non-empty
    for reservoir_name in RESERVOIR_NAMES:
        solution_objs[reservoir_name] = {}
        solution_vars[reservoir_name] = {}

        for policy_type in POLICY_TYPES:
            fname = f"{OUTPUT_DIR}/MMBorg_{ISLANDS}M_{policy_type}_{reservoir_name}_nfe{NFE}_seed{SEED}.csv"

            # Raw (only for logging)
            try:
                obj_df_raw, var_df_raw = load_results(
                    fname, obj_labels=obj_labels, filter=False, obj_bounds=None
                )
                print(f"[RAW] {reservoir_name}/{policy_type}: {len(obj_df_raw)} rows before filter")
            except Exception as e:
                print(f"[WARN] Could not load RAW for {reservoir_name}/{policy_type}: {e}")
                obj_df_raw, var_df_raw = pd.DataFrame(), pd.DataFrame()

            # Filtered
            try:
                obj_df, var_df = load_results(
                    fname, obj_labels=obj_labels, filter=True, obj_bounds=OBJ_FILTER_BOUNDS
                )
                print(f"[FLT] {reservoir_name}/{policy_type}: {len(obj_df)} rows after filter")
            except Exception as e:
                print(f"[WARN] Could not load FILTERED for {reservoir_name}/{policy_type}: {e}")
                obj_df, var_df = pd.DataFrame(), pd.DataFrame()

            if len(obj_df) == 0:
                print(f"Warning: No solutions found for {policy_type} with {reservoir_name}. Skipping.")
                continue

            solution_objs[reservoir_name][policy_type] = obj_df
            solution_vars[reservoir_name][policy_type] = var_df

            # Focal solutions
            idx_best_release = obj_df["Release NSE"].idxmax()
            idx_best_storage = obj_df["Storage KGE"].idxmax()
            idx_best_average = obj_df[["Release NSE", "Storage KGE"]].mean(axis=1).idxmax()

            print(f"Stats for {policy_type} {reservoir_name}:")
            print(f"Best Release NSE: {idx_best_release} = {obj_df['Release NSE'][idx_best_release]}")
            print(f"Best Storage KGE: {idx_best_storage} = {obj_df['Storage KGE'][idx_best_storage]}")
            print(f"Best Average NSE: {idx_best_average} = {obj_df[['Release NSE','Storage KGE']].mean(axis=1)[idx_best_average]}")

            # best average all (normalize to minimization)
            min_obj_df = obj_df.copy()
            min_obj_df["Release NSE"] = -min_obj_df["Release NSE"]
            min_obj_df["Storage KGE"] = -min_obj_df["Storage KGE"]
            scaled_min_obj_df = (min_obj_df - min_obj_df.min()) / (min_obj_df.max() - min_obj_df.min())
            idx_best_all_avg = scaled_min_obj_df.mean(axis=1).idxmin()

            # print parameter sets (best picks)
            try:
                params_best_all     = var_df.loc[idx_best_all_avg].values
                params_best_release = var_df.loc[idx_best_release].values
                params_best_storage = var_df.loc[idx_best_storage].values
                params_best_average = var_df.loc[idx_best_average].values

                print_params_flat(policy_type, params_best_all)
                print_params_pretty(policy_type, params_best_all)

                print("\n[Params] Best Release NSE:")
                print_params_flat(policy_type, params_best_release)
                print("\n[Params] Best Storage KGE:")
                print_params_flat(policy_type, params_best_storage)
                print("\n[Params] Best Average NSE:")
                print_params_flat(policy_type, params_best_average)
            except Exception as e:
                print(f"[WARN] Could not print named parameters for {reservoir_name}/{policy_type}: {e}")

            # Legacy highlight labels
            highlight_label_dict = {
                idx_best_release: "Best Release NSE",
                idx_best_storage: "Best Storage KGE",
                idx_best_average: "Best Average NSE",
                idx_best_all_avg: "Best Average All",
            }
            obj_df["highlight"] = [
                highlight_label_dict.get(idx, "Other") for idx in obj_df.index
            ]

            # Advanced selections (all objectives)
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
                senses=senses_all,
                bounds=OBJ_FILTER_BOUNDS,
                eps_qs=(0.5, 0.8),
                add_k_diverse=2,
                include_hv=False,
                out_label_col="highlight_adv",
            )

            solution_objs[reservoir_name][policy_type] = obj_df_aug
            solution_vars[reservoir_name][policy_type] = var_df
            solution_adv_maps.setdefault(reservoir_name, {})[policy_type] = cand_map
            solution_adv_cands.setdefault(reservoir_name, {})[policy_type] = cand_df

    # safer debug
    for res in RESERVOIR_NAMES:
        print(f"solution_objs[{res}] policies: {list(solution_objs.get(res, {}).keys())}")

    def get_pick_indices(reservoir_name: str, policy_type: str, label: str):
        """
        Return a list of row indices (dataframe index labels) for the requested label.
        Works with legacy 'highlight' and advanced picks from cand_map or 'highlight_adv'.
        """
        out = []

        # 1) legacy 'highlight'
        df = solution_objs.get(reservoir_name, {}).get(policy_type)
        if df is not None and "highlight" in df.columns:
            out += df.index[df["highlight"] == label].tolist()

        # 2) advanced cand_map
        cand_map = solution_adv_maps.get(reservoir_name, {}).get(policy_type, {}) or {}
        if label in cand_map and cand_map[label] is not None:
            val = cand_map[label]
            if isinstance(val, (list, tuple, np.ndarray, pd.Index, pd.Series)):
                out += list(pd.Index(val))
            else:
                out.append(val)

        # 3) advanced column fallback
        if df is not None and "highlight_adv" in df.columns:
            out += df.index[df["highlight_adv"] == label].tolist()

        # dedupe, preserve order
        seen, deduped = set(), []
        for idx in out:
            try:
                key = int(idx)
            except Exception:
                key = idx
            if key not in seen:
                seen.add(key)
                deduped.append(idx)
        return deduped

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

    summarize_ranges(solution_objs, obj_cols)

    #################################################
    m = "#### Figure 1 - Pareto Front Comparison #####"
    #################################################
    print(m)

    plot_obj_cols = ["Release NSE", "Storage KGE"]
    ideal_point = [1.0, 1.0]

    for reservoir in RESERVOIR_NAMES:
        if not reservoir_has_any(reservoir):
            print(f"[Fig1] Skip {reservoir}: no solutions for any policy.")
            continue

        obj_dfs, labels = [], []
        for policy in POLICY_TYPES:
            if not has_solutions(reservoir, policy):
                print(f"[Fig1] Skip {reservoir}/{policy}: no solutions.")
                continue
            obj_dfs.append(solution_objs[reservoir][policy])
            labels.append(policy_labels[policy])

        if not obj_dfs:
            continue

        fname = f"{FIG_DIR}/fig1_pareto_front_comparison/{reservoir}.png"
        plot_pareto_front_comparison(
            obj_dfs,
            labels,
            obj_cols=plot_obj_cols,
            ideal=ideal_point,
            title=f"Pareto Front Comparison - {reservoir_labels.get(reservoir, reservoir)}",
            fname=fname
        )

    ####################################################
    print("#### Figure 2 - Parallel Axis Plot #####")
    ####################################################

    if REMAKE_PARALLEL_PLOTS:
        # (A) All solutions per reservoir & policy
        print("Plotting all solutions for each reservoir & policy...")
        for reservoir_name in RESERVOIR_NAMES:
            for policy_type in POLICY_TYPES:
                if not has_solutions(reservoir_name, policy_type):
                    print(f"[Fig2-all] Skip {reservoir_name}/{policy_type}: no solutions.")
                    continue
                obj_df = solution_objs[reservoir_name][policy_type].copy()
                fname1 = f"{FIG_DIR}/fig2_parallel_axes/all_sols_{reservoir_name}_{policy_type}.png"
                custom_parallel_coordinates(
                    objs=obj_df,
                    columns_axes=obj_cols,
                    axis_labels=obj_cols,
                    ideal_direction='top',
                    minmaxs=minmaxs_all,
                    color_by_continuous=0,
                    color_palette_continuous=None,
                    color_by_categorical=None,
                    color_palette_categorical=None,
                    colorbar_ticks_continuous=None,
                    color_dict_categorical=None,
                    zorder_by=0,
                    zorder_num_classes=10,
                    zorder_direction='ascending',
                    alpha_base=0.7,
                    brushing_dict=None,
                    alpha_brush=0.05,
                    lw_base=1.5,
                    fontsize=9,
                    figsize=(11, 6),
                    bottom_pad=0.22,
                    legend_pad=0.08,
                    fname=fname1
                )

        # (B) Highlight best solutions
        print("Plotting best solutions for each reservoir & policy...")
        highlight_colors = {
            "Best Release NSE": "red",
            "Best Storage KGE": "green",
            "Best Average NSE": "purple",
            "Best Average All": "blue",
            "Other": "lightgray"
        }
        for reservoir_name in RESERVOIR_NAMES:
            for policy_type in POLICY_TYPES:
                if not has_solutions(reservoir_name, policy_type):
                    print(f"[Fig2-best] Skip {reservoir_name}/{policy_type}: no solutions.")
                    continue
                obj_df = solution_objs[reservoir_name][policy_type].copy()
                fname1 = f"{FIG_DIR}/fig2_parallel_axes/compare_best_sols_{reservoir_name}_{policy_type}.png"
                custom_parallel_coordinates(
                    objs=obj_df,
                    columns_axes=obj_cols,
                    axis_labels=obj_cols,
                    ideal_direction='top',
                    minmaxs=minmaxs_all,
                    color_by_continuous=None,
                    color_palette_continuous=None,
                    color_by_categorical='highlight',
                    color_palette_categorical=None,
                    colorbar_ticks_continuous=None,
                    color_dict_categorical=highlight_colors,
                    zorder_by=0,
                    zorder_num_classes=10,
                    zorder_direction='ascending',
                    alpha_base=0.9,
                    brushing_dict=None,
                    alpha_brush=0.1,
                    lw_base=1.5,
                    fontsize=9,
                    figsize=(11, 6),
                    bottom_pad=0.22,
                    legend_pad=0.08,
                    fname=fname1
                )

        # (C) Advanced selections highlighted
        print("Plotting advanced selections for each reservoir & policy...")
        for reservoir_name in RESERVOIR_NAMES:
            for policy_type in POLICY_TYPES:
                if not has_solutions(reservoir_name, policy_type):
                    print(f"[Fig2-adv] Skip {reservoir_name}/{policy_type}: no solutions.")
                    continue
                obj_df_adv = solution_objs[reservoir_name][policy_type].copy()
                fname_adv = f"{FIG_DIR}/fig2_parallel_axes/advanced_picks_{reservoir_name}_{policy_type}.png"
                custom_parallel_coordinates(
                    objs=obj_df_adv,
                    columns_axes=obj_cols,
                    axis_labels=obj_cols,
                    ideal_direction='top',
                    minmaxs=minmaxs_all,
                    color_by_continuous=None,
                    color_palette_continuous=None,
                    color_by_categorical='highlight_adv',
                    color_palette_categorical=None,
                    colorbar_ticks_continuous=None,
                    color_dict_categorical=ADVANCED_COLORS,
                    zorder_by=0,
                    zorder_num_classes=10,
                    zorder_direction='ascending',
                    alpha_base=0.9,
                    brushing_dict=None,
                    alpha_brush=0.1,
                    lw_base=1.5,
                    fontsize=9,
                    figsize=(11, 6),
                    bottom_pad=0.3,
                    legend_pad=0.2,
                    fname=fname_adv
                )

        # (D) All policies combined per reservoir
        print("Plotting all solutions, all policies for each reservoir...")
        for reservoir_name in RESERVOIR_NAMES:
            if not reservoir_has_any(reservoir_name):
                print(f"[Fig2-allpol] Skip {reservoir_name}: no solutions.")
                continue

            obj_list = []
            for policy_type in POLICY_TYPES:
                if has_solutions(reservoir_name, policy_type):
                    df = solution_objs[reservoir_name][policy_type].copy()
                    df['policy'] = policy_type
                    obj_list.append(df)

            if not obj_list:
                continue

            combined_df = pd.concat(obj_list, axis=0)
            if combined_df.empty:
                print(f"[Fig2-allpol] Skip {reservoir_name}: combined empty.")
                continue

            combined_df = combined_df.sample(frac=1).reset_index(drop=True)
            fname1 = f"{FIG_DIR}/fig2_parallel_axes/all_policies_{reservoir_name}.png"
            custom_parallel_coordinates(
                objs=combined_df,
                columns_axes=obj_cols,
                axis_labels=obj_cols,
                ideal_direction='top',
                minmaxs=minmaxs_all,
                color_by_continuous=None,
                color_palette_continuous=None,
                color_by_categorical='policy',
                color_palette_categorical=None,
                colorbar_ticks_continuous=None,
                color_dict_categorical=policy_colors,
                zorder_by=None,
                zorder_num_classes=None,
                zorder_direction='ascending',
                alpha_base=0.3,
                brushing_dict=None,
                alpha_brush=0.1,
                lw_base=1.0,
                fontsize=9,
                figsize=(11, 6),
                bottom_pad=0.22,
                legend_pad=0.08,
                fname=fname1
            )

    ####################################################
    print("##### Figure 3 - system dynamics ######")
    ####################################################

    if REMAKE_DYNAMICS_PLOTS:

        for reservoir_name in RESERVOIR_NAMES:

            # Load observed arrays for sim
            inflow_df, release_df, storage_df = get_observational_training_data(
                reservoir_name=reservoir_name,
                data_dir=PROCESSED_DATA_DIR,
                as_numpy=False,
                inflow_type='inflow_pub'
            )
            if inflow_df.empty or storage_df.empty:
                print(f"[Fig3] Skip {reservoir_name}: missing obs.")
                continue

            datetime = inflow_df.index
            inflow_obs = inflow_df.values
            release_obs = release_df.values
            storage_obs = storage_df.values

            for policy_type in POLICY_TYPES:
                if not has_solutions(reservoir_name, policy_type):
                    print(f"[Fig3] Skip {reservoir_name}/{policy_type}: no solutions.")
                    continue

                var_df = solution_vars.get(reservoir_name, {}).get(policy_type)
                if var_df is None or var_df.empty:
                    print(f"[Fig3] Skip {reservoir_name}/{policy_type}: no variable DF.")
                    continue

                for pick_label in DESIRED_PICKS:
                    idxs = get_pick_indices(reservoir_name, policy_type, pick_label)
                    if not idxs:
                        print(f"[Fig3] {reservoir_name}/{policy_type}: no '{pick_label}' pick; skip.")
                        continue

                    for k, idx_row in enumerate(idxs, start=1):
                        params = _params_for_row(var_df, idx_row)

                        reservoir = Reservoir(
                            inflow=inflow_obs, dates=datetime,
                            capacity=reservoir_capacity[reservoir_name],
                            policy_type=policy_type, policy_params=params,
                            initial_storage=storage_obs[0], name=reservoir_name
                        )

                        # optional policy surface
                        if hasattr(reservoir.policy, "plot_surfaces_for_different_weeks"):
                            surf_dir = Path(FIG_DIR, "figX_policy_surfaces"); surf_dir.mkdir(parents=True, exist_ok=True)
                            reservoir.policy.plot_surfaces_for_different_weeks(
                                fname=str(surf_dir / f"{safe_name(reservoir_name)}_{policy_type}_{safe_name(pick_label)}_{k}_surface.png"),
                                save=True, grid=40, n_weeks=5
                            )
                        elif hasattr(reservoir.policy, "plot_policy_surface"):
                            reservoir.policy.plot_policy_surface(
                                save=True,
                                fname=f"{safe_name(reservoir_name)}_{policy_type}_{safe_name(pick_label)}_{k}_policy_surface.png"
                            )
                        else:
                            reservoir.policy.plot(N=41)

                        reservoir.run()
                        sim_storage = reservoir.storage_array
                        sim_release = reservoir.release_array + reservoir.spill_array

                        base = f"{FIG_DIR}/fig3_dynamics/{safe_name(reservoir_name)}_{policy_type}_{safe_name(pick_label)}_{k}"

                        fig1, _ = plot_storage_release_distributions(
                            obs_storage=storage_obs.flatten(), obs_release=release_obs.flatten(),
                            sim_storage=sim_storage.flatten(), sim_release=sim_release.flatten(),
                            obs_inflow=inflow_obs.flatten(), datetime=datetime,
                            storage_distribution=True, inflow_scatter=False, inflow_vs_release=True,
                            fname=f"{base}__quantIR.png"
                        ); plt.close(fig1)

    ####################################################
    ### Figure 4 - simulation dynamics (validation) ####
    ####################################################

    if REMAKE_DYNAMICS_PLOTS:
        print("##### Figure 4 - Validation 9-panel plots #####")

        VAL_START = "1980-01-01"
        VAL_END   = "2018-12-31"
        VAL_INFLOW_TYPE = "inflow_pub"

        for reservoir_name in RESERVOIR_NAMES:

            inflow_df, release_df, storage_df = get_observational_training_data(
                reservoir_name=reservoir_name,
                data_dir=PROCESSED_DATA_DIR,
                as_numpy=False,
                inflow_type=VAL_INFLOW_TYPE
            )
            if inflow_df.empty:
                print(f"[Fig4] Skip {reservoir_name}: no inflow data.")
                continue

            slicer = slice(VAL_START, VAL_END)
            inflow_win   = inflow_df.loc[slicer][reservoir_name] if reservoir_name in inflow_df.columns else inflow_df.loc[slicer].iloc[:, 0]
            release_win  = release_df.loc[slicer][reservoir_name] if reservoir_name in release_df.columns else None
            storage_win  = storage_df.loc[slicer][reservoir_name] if reservoir_name in storage_df.columns else storage_df.loc[slicer].iloc[:, 0]

            if len(inflow_win) == 0 or len(storage_win) == 0:
                print(f"[Fig4] Skip {reservoir_name}: empty validation window.")
                continue

            dt_index = inflow_win.index
            for policy_type in POLICY_TYPES:
                if not has_solutions(reservoir_name, policy_type):
                    print(f"[Fig4] Skip {reservoir_name}/{policy_type}: no solutions.")
                    continue

                obj_df = solution_objs[reservoir_name][policy_type]
                var_df = solution_vars[reservoir_name][policy_type]
                if obj_df is None or obj_df.empty or var_df is None or var_df.empty:
                    print(f"[Fig4] Skip {reservoir_name}/{policy_type}: empty obj/var DF.")
                    continue

                for pick_label in DESIRED_PICKS:
                    idxs = get_pick_indices(reservoir_name, policy_type, pick_label)
                    if not idxs:
                        print(f"[Fig4] {reservoir_name}/{policy_type}: no '{pick_label}' pick; skip.")
                        continue

                    for k, idx_row in enumerate(idxs, start=1):
                        params_1d = _params_for_row(var_df, idx_row)

                        model = Reservoir(
                            inflow=inflow_win.values, dates=dt_index,
                            capacity=reservoir_capacity[reservoir_name],
                            policy_type=policy_type, policy_params=params_1d,
                            initial_storage=float(storage_win.iloc[0]), name=reservoir_name
                        )
                        model.run()

                        sim_storage = pd.Series(model.storage_array.flatten(), index=dt_index, name=reservoir_name)
                        sim_release = pd.Series((model.release_array + model.spill_array).flatten(), index=dt_index, name=reservoir_name)

                        out = f"{FIG_DIR}/fig4_validation_9panel/{safe_name(reservoir_name)}_{policy_type}_{safe_name(pick_label)}_{k}_9panel.png"
                        plot_release_storage_9panel(
                            reservoir=reservoir_name,
                            sim_release=sim_release, obs_release=release_win if release_win is not None else None,
                            sim_storage_MG=sim_storage, obs_storage_MG=storage_win,
                            start=VAL_START, end=VAL_END, save_path=out
                        )
                        print(f"[Fig4] Saved 9-panel: {out}")

    print("DONE!")
