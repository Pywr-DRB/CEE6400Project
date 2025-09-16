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
        'PWL': 'green',
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

    # Load all results first
    for reservoir_name in RESERVOIR_NAMES:
        solution_objs[reservoir_name] = {}

        for policy_type in POLICY_TYPES:
            fname = f"{OUTPUT_DIR}/MMBorg_{ISLANDS}M_{policy_type}_{reservoir_name}_nfe{NFE}_seed{SEED}.csv"
            obj_df, var_df = load_results(fname, obj_labels=obj_labels)

            # Filter bad solutions
            obj_df = obj_df[obj_df["Storage NSE"] <= 10]
            var_df = var_df.loc[obj_df.index]

            solution_objs[reservoir_name][policy_type] = obj_df

    # Now, per reservoir, concatenate across policies
    for reservoir_name in RESERVOIR_NAMES:
        obj_list = []
        policy_list = []

        for policy_type in POLICY_TYPES:
            obj_df = solution_objs[reservoir_name][policy_type].copy()
            obj_df['policy'] = policy_type  # add policy column
            obj_list.append(obj_df)
        
        # Concatenate all policies into one dataframe
        combined_df = pd.concat(obj_list, axis=0)

        # Now plot
        fname1 = f"{FIG_DIR}/parallel_comparison_{reservoir_name}_all_policies_{NFE}nfe.png"
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
            alpha_base=0.1,
            brushing_dict=None,
            alpha_brush=0.05,
            lw_base=1.5,
            fontsize=9,
            figsize=(11, 6),
            fname=fname1
        )
        print(f"Figure saved to {fname1}")
