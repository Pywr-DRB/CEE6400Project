import pandas as pd

from methods.config import NFE, SEED, METRICS, ISLANDS
from methods.config import reservoir_options, policy_type_options
from methods.config import OUTPUT_DIR, FIG_DIR
from methods.plotting.plot_pareto_front_comparison import plot_pareto_front_comparison
from methods.load.results import load_results


reservoir_labels = {
    'beltzvilleCombined': 'Beltzville',
    'fewalter': 'FE Walter',
    'prompton': 'Prompton',
}



if __name__ == "__main__":
    
    ####################################################
    ### Load data ######################################
    ####################################################
    POLICY_TYPES = policy_type_options
    RESERVOIR_NAMES = reservoir_options
    
    # nested dict to hold solutions
    # dict[reservoir_name][policy_type] = df
    solution_objs = {}
    solution_vars = {}
        
    obj_labels = {
        "obj1": "Release NSE",
        "obj2": "Release q20 Abs % Bias",
        "obj3": "Release q80 Abs % Bias",
        "obj4": "Storage NSE",
    }
    obj_cols = list(obj_labels.values())
    
    # For each reservoir
    for reservoir_name in RESERVOIR_NAMES:
        solution_objs[reservoir_name] = {}
        solution_vars[reservoir_name] = {}
        
        
        for policy_type in POLICY_TYPES:
            
            # Borg output fname
            fname = f"{OUTPUT_DIR}/MMBorg_{ISLANDS}M_{policy_type}_{reservoir_name}_nfe{NFE}_seed{SEED}.csv"
            
            obj_df, var_df = load_results(fname, obj_labels=obj_labels)
            solution_objs[reservoir_name][policy_type] = obj_df
            solution_vars[reservoir_name][policy_type] = var_df
            
                    
    ####################################################
    ### Figure 1 - Pareto Front Comparison #############
    ####################################################
    
        
    ####################################################
    ### Figure 2 - Parallel axis plot      #############
    ### for specific policies & reservoirs #############
    ####################################################
    
    #TODO
    
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