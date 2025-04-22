import pandas as pd

from methods.config import NFE, SEED
from methods.config import reservoir_options, policy_type_options
from methods.config import OUTPUT_DIR, FIG_DIR
from methods.plotting.plot_pareto_front_comparison import plot_pareto_front_comparison


if __name__ == "__main__":
    
    
    POLICY_TYPES = policy_type_options
    RESERVOIR_NAMES = reservoir_options
    
    
    obj_labels = {
        "obj1": "Release -NSE",
        "obj2": "Release q20 Abs % Bias",
        "obj3": "Release q80 Abs % Bias",
        "obj4": "Storage -NSE",
    }
    
    # For each reservoir
    for reservoir_name in RESERVOIR_NAMES:

        # Make a list of dataframes containing pareto front data
        # for different policy types
        obj_dfs = []
        for policy_type in POLICY_TYPES:
            
            # Borg output fname
            fname = f"{OUTPUT_DIR}/MMBorg_3M_{policy_type}_{reservoir_name}_nfe{NFE}_seed{SEED}.csv"
            
            # Load dataframes
            df = pd.read_csv(fname)

            # Keep only the objectives
            obj_cols = [col for col in df.columns if col.startswith("obj")]
            print(obj_cols)
            df = df.loc[:, obj_cols]

            # Rename the columns
            df.rename(columns=obj_labels, inplace=True)
            obj_cols = list(df.columns)

            
            obj_dfs.append(df.loc[:, obj_cols])
                
        # Labels for the policies
        labels = POLICY_TYPES
        print(obj_cols)
        plot_obj_cols = [obj_cols[0], obj_cols[3]]
        
        
        # Plot comparison
        plot_pareto_front_comparison(obj_dfs = obj_dfs, 
                                    labels = labels,
                                    obj_cols = plot_obj_cols,
                                    fname=f"{FIG_DIR}/pareto_front_comparison_{reservoir_name}_{NFE}nfe.png")