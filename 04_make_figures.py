import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from methods.config import NFE, SEED, ISLANDS
from methods.config import OBJ_LABELS, OBJ_FILTER_BOUNDS
from methods.config import reservoir_options, policy_type_options
from methods.config import OUTPUT_DIR, FIG_DIR, PROCESSED_DATA_DIR
from methods.config import reservoir_min_release, reservoir_max_release, reservoir_capacity

from methods.reservoir.model import Reservoir
from methods.load.results import load_results
from methods.load.observations import get_observational_training_data

from methods.plotting.plot_pareto_front_comparison import plot_pareto_front_comparison
from methods.plotting.plot_parallel_axis import custom_parallel_coordinates
from methods.plotting.plot_reservoir_storage_release_distributions import plot_storage_release_distributions

POLICY_TYPES = policy_type_options
RESERVOIR_NAMES = reservoir_options

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
    'PiecewiseLinear': 'Piecewise Linear',
}

policy_colors = {
        'STARFIT': 'blue',
        'RBF': 'orange',
        'PiecewiseLinear': 'green',
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
                    
                    # Define the reservoir model
                    reservoir = Reservoir(
                        inflow = inflow_obs,
                        dates= datetime,
                        capacity = reservoir_capacity[reservoir_name],
                        policy_type = policy_type,
                        policy_params = params,
                        release_min = reservoir_min_release[reservoir_name],
                        release_max =  reservoir_max_release[reservoir_name],
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