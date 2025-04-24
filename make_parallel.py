import pandas as pd
import os

from methods.metrics.objectives import ObjectiveCalculator
from methods.config import NFE, SEED, METRICS, ISLANDS
from methods.config import reservoir_options, policy_type_options
from methods.config import OUTPUT_DIR, FIG_DIR
from methods.plotting.plot_parallel_axis import custom_parallel_coordinates
from methods.load.results import load_results

# make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

if __name__ == "__main__":
    ####################################################
    ### Load data ######################################
    ####################################################
    POLICY_TYPES = policy_type_options
    RESERVOIR_NAMES = reservoir_options

    color_by_categorical = 'policy'
    color_dict_categorical = {
        'STARFIT': 'blue',
        'RBF': 'orange',
        'PiecewiseLinear': 'green',
    }

    obj_labels = {
        "obj1": "Release NSE",
        "obj2": "Release q20 Abs % Bias",
        "obj3": "Release q80 Abs % Bias",
        "obj4": "Storage NSE",
    }
    obj_cols = list(obj_labels.values())

    solution_objs = {}
    solution_vars = {}

    for reservoir_name in RESERVOIR_NAMES:
        solution_objs[reservoir_name] = {}
        solution_vars[reservoir_name] = {}

        for policy_type in POLICY_TYPES:
            fname = f"{OUTPUT_DIR}/MMBorg_{ISLANDS}M_{policy_type}_{reservoir_name}_nfe{NFE}_seed{SEED}.csv"
            obj_df, var_df = load_results(fname, obj_labels=obj_labels)
            
            # Filter bad solutions
            obj_df = obj_df[obj_df["Storage NSE"] <= 10]
            var_df = var_df.loc[obj_df.index]

            solution_objs[reservoir_name][policy_type] = obj_df
            solution_vars[reservoir_name][policy_type] = var_df




    # Labels for the policies
    labels = POLICY_TYPES

    
    for reservoir_name in RESERVOIR_NAMES:
    
        # Make a list of dataframes for each reservoir
        combined_df = []
    
        for policy_type in POLICY_TYPES:
            # Append the dataframe to the list
            obj_df = solution_objs[reservoir_name][policy_type].copy()
            obj_df["policy"] = policy_type  # add for combined plot
        # brushing_dict = {
            #     obj_cols.index("Release NSE"): (10, '<'),
            #     obj_cols.index("Storage NSE"): (10, '<'),
            # }
        # figure name
            fname1 = f"{FIG_DIR}/parallel_comparison_{reservoir_name}_{policy_type}_{NFE}nfe.png"
            custom_parallel_coordinates(
                objs=obj_df,
                columns_axes=obj_cols,
                axis_labels=obj_cols,
                ideal_direction='bottom',
                minmaxs=['min','min','min','min'],
                color_by_continuous=0,
                color_palette_continuous=None,
                color_by_categorical=None,
                color_palette_categorical=None,
                colorbar_ticks_continuous=None,
                color_dict_categorical=None,
                zorder_by=0,
                zorder_num_classes=10,
                zorder_direction='ascending',
                alpha_base=0.8,
                brushing_dict=None,
                alpha_brush=0.05,
                lw_base=1.5,
                fontsize=9,
                figsize=(11, 6),
                fname=fname1
                )
            print(f"Figure saved to {fname1}")
            combined_df.append(obj_df)
        
        # === Combined PAP for all policies
        combined_df = pd.concat(combined_df, ignore_index=True)

        fname2 = f"{FIG_DIR}/parallel_comparison_combined_{reservoir_name}_{NFE}nfe.png"
        custom_parallel_coordinates(
            objs=combined_df,
            columns_axes=obj_cols,
            axis_labels=obj_cols,
            ideal_direction='bottom',
            minmaxs=['min','min','min','min'],
            color_by_continuous=None,
            color_palette_continuous=None,
            color_by_categorical='policy',
            color_palette_categorical=None,
            color_dict_categorical=color_dict_categorical,
            colorbar_ticks_continuous=None,
            zorder_by=None,
            zorder_num_classes=10,
            zorder_direction='ascending',
            alpha_base=0.8,
            brushing_dict=None,
            alpha_brush=0.05,
            lw_base=1.5,
            fontsize=9,
            figsize=(11, 6),
            fname=fname2
            )
        print(f"Combined policy figure saved to {fname2}")
    
print("All figures saved successfully.")

