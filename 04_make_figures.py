import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from methods.config import NFE, SEED, METRICS, ISLANDS
from methods.config import OBJ_LABELS, OBJ_FILTER_BOUNDS
from methods.config import reservoir_options, policy_type_options
from methods.config import OUTPUT_DIR, FIG_DIR
from methods.load.results import load_results, filter_solutions

from methods.plotting.plot_pareto_front_comparison import plot_pareto_front_comparison
from methods.plotting.plot_parallel_axis import custom_parallel_coordinates

POLICY_TYPES = policy_type_options
RESERVOIR_NAMES = reservoir_options


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
            
    ####################################################
    ###  ######################################
    ####################################################


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
    ### Figure 3 - Parallel axis plot      #############
    ### for specific policies & reservoirs #############
    ####################################################
    
    #TODO
    
    ####################################################
    ### Figure 4 - simulation dynamics     #############
    ### during historic validation period  #############
    ####################################################
    
    #TODO
    
    
    print("DONE!")