import pandas as pd
import numpy as np
import warnings
from pathlib import Path
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

# ======= PARAM NAME MAPS (match run_simple_model.py) =======
from methods.config import n_rbfs, n_rbf_inputs, n_segments, n_pwl_inputs

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
        # weights
        print("Weights:")
        for i in range(1, n_rbfs + 1):
            print(f"  w{i} = {float(params_1d[idx]): .6f}")
            idx += 1
        # centers
        print("Centers c[i, var]:")
        for i in range(1, n_rbfs + 1):
            row = []
            for var in ["storage", "inflow", "doy"][:n_rbf_inputs]:
                row.append(float(params_1d[idx])); idx += 1
            print(f"  c{i} = {row}")
        # scales
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

    # fallback
    print_params_flat(policy, params_1d)

def make_var_df_with_names(policy_type: str, var_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of var_df with friendly column names (in correct order).
    Assumes var_df columns are var1..varN (or equivalent positional order).
    """
    names = get_param_names_for_policy(policy_type)
    assert len(var_df.columns) >= len(names), "var_df has fewer columns than expected parameters."
    df = var_df.copy().iloc[:, :len(names)]
    df.columns = names
    return df


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


if __name__ == "__main__":
    
    ## Load data
    inflow_obs, release_obs, storage_obs = get_observational_training_data(
                reservoir_name='prompton',
                data_dir = PROCESSED_DATA_DIR,
                as_numpy=False,
                scaled_inflows=True
            )
    print(f"Inflows shape: {inflow_obs.shape}")
    print(f"Datetime: {inflow_obs.index}")
    print(f"Min storage: {storage_obs.min()}")
    print(f"Max storage: {storage_obs.max()}")
    print(f"Min release: {release_obs.min()}")
    print(f"Max release: {release_obs.max()}")

    # --- make sure figure subfolders exist ---
    Path(FIG_DIR, "fig1_pareto_front_comparison").mkdir(parents=True, exist_ok=True)
    Path(FIG_DIR, "fig2_parallel_axes").mkdir(parents=True, exist_ok=True)
    Path(FIG_DIR, "fig3_dynamics").mkdir(parents=True, exist_ok=True)

    ####################################################
    ### Load & process data ############################
    ####################################################
    
    # nested dict to hold solutions
    # dict[reservoir_name][policy_type] = df
    solution_objs = {}
    solution_vars = {}
        
    obj_labels = OBJ_LABELS
    obj_cols = list(obj_labels.values())
    
    # For each reservoir
    for reservoir_name in RESERVOIR_NAMES:
        solution_objs[reservoir_name] = {}
        solution_vars[reservoir_name] = {}
        
        
        for policy_type in POLICY_TYPES:

            ### Load output data & filter
            # Borg output fname
            fname = f"{OUTPUT_DIR}/MMBorg_{ISLANDS}M_{policy_type}_{reservoir_name}_nfe{NFE}_seed{SEED}.csv"
            
            obj_df, var_df = load_results(fname, obj_labels=obj_labels,
                                          filter=True, obj_bounds=OBJ_FILTER_BOUNDS)
            

            if len(obj_df) == 0:
                print(f"Warning: No solutions found for {policy_type} with {reservoir_name}.")
                continue

            # Print the number of solutions
            msg = f"#### Reservoir: {reservoir_name} Policy: {policy_type} ####\n"
            msg += f"Number of solutions after filter: {len(obj_df)}\n"
            print(msg)
            
            
            ### Find focal solutions
            # Different metrics for 'best' solutions
            idx_best_release = obj_df["Release NSE"].idxmax()
            idx_best_storage = obj_df["Storage NSE"].idxmax()
            idx_best_average = obj_df[["Release NSE", "Storage NSE"]].mean(axis=1).idxmax()

            print(f"Stats for {policy_type} {reservoir_name}:")
            print(f"Best Release NSE: {idx_best_release} with value {obj_df['Release NSE'][idx_best_release]}")
            print(f"Best Storage NSE: {idx_best_storage} with value {obj_df['Storage NSE'][idx_best_storage]}")
            print(f"Best Average NSE: {idx_best_average} with value {obj_df[['Release NSE', 'Storage NSE']].mean(axis=1)[idx_best_average]}")


            # for best average all, we need to convert NSE to negative again 
            # then scale all objs to have equal weight
            # then find idxmin() of the mean scale score
            min_obj_df = obj_df.copy()
            min_obj_df["Release NSE"] = -min_obj_df["Release NSE"]
            min_obj_df["Storage NSE"] = -min_obj_df["Storage NSE"]
            scaled_min_obj_df = (min_obj_df - min_obj_df.min()) / (min_obj_df.max() - min_obj_df.min())
            idx_best_all_avg = scaled_min_obj_df.mean(axis=1).idxmin()
            
            # ================== INSERT THIS BLOCK ==================
            # Print parameters for each focal pick (uses the helpers you added)
            try:
                # Best Average All (main ask)
                params_best_all = solution_vars[reservoir_name][policy_type].iloc[idx_best_all_avg].values
                print_params_flat(policy_type, params_best_all)
                print_params_pretty(policy_type, params_best_all)

                # (Optional) also print others:
                params_best_release = solution_vars[reservoir_name][policy_type].iloc[idx_best_release].values
                params_best_storage = solution_vars[reservoir_name][policy_type].iloc[idx_best_storage].values
                params_best_average = solution_vars[reservoir_name][policy_type].iloc[idx_best_average].values

                print("\n[Params] Best Release NSE:")
                print_params_flat(policy_type, params_best_release)

                print("\n[Params] Best Storage NSE:")
                print_params_flat(policy_type, params_best_storage)

                print("\n[Params] Best Average NSE:")
                print_params_flat(policy_type, params_best_average)

            except Exception as e:
                print(f"[WARN] Could not print named parameters for {reservoir_name}/{policy_type}: {e}")
            # =======================================================
            highlight_label_dict = {
                idx_best_release: "Best Release NSE",
                idx_best_storage: "Best Storage NSE",
                idx_best_average: "Best Average NSE",
                idx_best_all_avg: "Best Average All",
            }
            
            highlight_labels = []
            for idx in obj_df.index:
                if idx in list(highlight_label_dict.keys()):
                    highlight_labels.append(highlight_label_dict[idx])            
                else:
                    highlight_labels.append("Other")

            # Add highlight labels to the dataframe
            obj_df["highlight"] = highlight_labels


            ### Save solution data for later plots
            solution_objs[reservoir_name][policy_type] = obj_df
            solution_vars[reservoir_name][policy_type] = var_df
            
    for policy_type in solution_objs['fewalter'].keys():
        print(f'solution_objs has policy type: {policy_type}')
    for policy_type in solution_objs['beltzvilleCombined'].keys():
        print(f'solution_objs has policy type: {policy_type}')
    for policy_type in solution_objs['prompton'].keys():
        print(f'solution_objs has policy type: {policy_type}')
    #################################################
    m="#### Figure 1 - Pareto Front Comparison #####"
    #################################################
    print(m)
    
    plot_obj_cols = ["Release NSE", "Storage NSE"]
    ideal_point = [1.0, 1.0]
    
    for reservoir in RESERVOIR_NAMES:
        obj_dfs = []
        labels = []
        for policy in POLICY_TYPES:
            obj_dfs.append(solution_objs[reservoir][policy])
            labels.append(policy_labels[policy])
        
        
        fname = f"{FIG_DIR}/fig1_pareto_front_comparison/"
        fname += f"{reservoir}.png"
        
        # Plot the pareto front comparison
        plot_pareto_front_comparison(
            obj_dfs,
            labels,
            obj_cols=plot_obj_cols,
            ideal = ideal_point,
            title=f"Pareto Front Comparison - {reservoir_labels[reservoir]}",
            fname=fname
        )

    ####################################################
    print("#### Figure 2 - Parallel Axis Plot #####")
    ####################################################
    
    if REMAKE_PARALLEL_PLOTS:
        ### First plot: All solutions for each reservoir & policy
        print("Plotting all solutions for each reservoir & policy...")
        
        for reservoir_name in RESERVOIR_NAMES:
            for policy_type in POLICY_TYPES:
                
                obj_df = solution_objs[reservoir_name][policy_type].copy()
                
                # figure name
                fname1 = f"{FIG_DIR}/fig2_parallel_axes/"
                fname1 += f"all_sols_{reservoir_name}_{policy_type}.png"
                
                custom_parallel_coordinates(
                    objs=obj_df,
                    columns_axes=obj_cols,
                    axis_labels=obj_cols,
                    ideal_direction='top',
                    minmaxs=['max','min','min','max'],
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
                    fname=fname1
                    )
        
        
        ### Second plot: Highlight best solutions
        print("Plotting best solutions for each reservoir & policy...")
        
        color_by_categorical = 'highlight'
        highlight_colors = {
            "Best Release NSE": "red",
            "Best Storage NSE": "green",
            "Best Average NSE": "purple",
            "Best Average All": "blue",
            "Other": "lightgray"
        }
        
        # Loop through each reservoir and policy
        for reservoir_name in RESERVOIR_NAMES:
            for policy_type in POLICY_TYPES:
        
                # get the dataframe
                obj_df = solution_objs[reservoir_name][policy_type].copy()
        
                # figure name
                fname1 = f"{FIG_DIR}/fig2_parallel_axes/"
                fname1 += f"compare_best_sols_{reservoir_name}_{policy_type}.png"
                custom_parallel_coordinates(
                    objs=obj_df,
                    columns_axes=obj_cols,
                    axis_labels=obj_cols,
                    ideal_direction='top',
                    minmaxs=['max','min','min','max'],
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
                    fname=fname1
                    )
                
        ### Third: All solutions for each reservoir, all policies
        print("Plotting all solutions, all policies for each reservoir...")
        
        color_by_categorical = 'policy'
        color_dict_categorical = policy_colors
        
        for reservoir_name in RESERVOIR_NAMES:
            # Now, per reservoir, concatenate across policies
            obj_list = []
            policy_list = []

            for policy_type in POLICY_TYPES:
                obj_df = solution_objs[reservoir_name][policy_type].copy()
                obj_df['policy'] = policy_type  # add policy column
                obj_list.append(obj_df)

            # Concatenate all policies into one dataframe
            combined_df = pd.concat(obj_list, axis=0)
            
            # Shuffle the dataframe to mix policies
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)
            
            # Now plot
            fname1 = f"{FIG_DIR}/fig2_parallel_axes/"
            fname1 += f"all_policies_{reservoir_name}.png"
            custom_parallel_coordinates(
                objs=combined_df,
                columns_axes=obj_cols,
                axis_labels=obj_cols,
                ideal_direction='top',
                minmaxs=['max', 'min', 'min', 'max'],
                color_by_continuous=None,
                color_palette_continuous=None,
                color_by_categorical=color_by_categorical,
                color_palette_categorical=None,
                colorbar_ticks_continuous=None,
                color_dict_categorical=color_dict_categorical,
                zorder_by=None,
                zorder_num_classes=None,
                zorder_direction='ascending',
                alpha_base=0.3,
                brushing_dict=None,
                alpha_brush=0.1,
                lw_base=1.0,
                fontsize=9,
                figsize=(11, 6),
                fname=fname1
            )
            
    ####################################################
    print("##### Figure 3 - system dynamics ######")
    ####################################################
    
    if REMAKE_DYNAMICS_PLOTS:
        
        for reservoir_name in RESERVOIR_NAMES:
            
            ## Load data
            inflow_obs, release_obs, storage_obs = get_observational_training_data(
                reservoir_name=reservoir_name,
                data_dir = PROCESSED_DATA_DIR,
                as_numpy=False,
                scaled_inflows=True
            )
            
            # Keep just arrays
            datetime = inflow_obs.index
            inflow_obs = inflow_obs.values
            release_obs = release_obs.values
            storage_obs = storage_obs.values
            
            for policy_type in POLICY_TYPES:
                
                # Loop through 'best' solutions
                solution_types = [
                    "Best Release NSE",
                    "Best Storage NSE",
                    "Best Average NSE",
                    "Best Average All"
                ]
                
                for solution_type in solution_types:
                    
                    params = solution_vars[reservoir_name][policy_type].loc[
                        solution_objs[reservoir_name][policy_type]['highlight'] == solution_type
                    ].values.flatten()
                    
                    # Unable to find best storage NSE for RBF... skip.
                    if len(params) == 0:
                        print(f"Warning: No parameters found for {solution_type} for {reservoir_name} with {policy_type}.")
                        continue
                    
                    print_params_pretty(policy_type, params)

                    # Define the reservoir model
                    reservoir = Reservoir(
                        inflow = inflow_obs,
                        dates= datetime,
                        capacity = reservoir_capacity[reservoir_name],
                        policy_type = policy_type,
                        policy_params = params,
                        initial_storage = storage_obs[0],
                        name = reservoir_name,
                    )
                    
                    # Run 
                    reservoir.run()
                    
                    # output data
                    sim_storage = reservoir.storage_array
                    sim_release = reservoir.release_array
                    sim_release = sim_release + reservoir.spill_array
            
                    ## Plot dynamics
                    fig_fname = f"{FIG_DIR}/fig3_dynamics/"
                    fig_fname += f"{reservoir_name}_{policy_type}_{solution_type}.png"
            
            
                    plot_storage_release_distributions(
                        obs_storage=storage_obs.flatten(),
                        obs_release=release_obs.flatten(),
                        sim_storage=sim_storage.flatten(),
                        sim_release=sim_release.flatten(),
                        obs_inflow=inflow_obs.flatten(),
                        datetime=datetime,
                        storage_distribution=True,
                        inflow_scatter=False,
                        inflow_vs_release=True,
                        fname=fig_fname
                    )
                
    ####################################################
    ### Figure 4 - simulation dynamics     #############
    ### during historic validation period  #############
    ####################################################
    
    #TODO
    
    
    print("DONE!")