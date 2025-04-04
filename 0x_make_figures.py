import pandas as pd

from methods.config import OUTPUT_DIR, FIG_DIR
from methods.plotting.plot_pareto_front_comparison import plot_pareto_front_comparison


if __name__ == "__main__":
    
    
    POLICY_TYPES = ["RBF", "PiecewiseLinear"]
    reservoir_name = "fewalter"
    NFE = 30000
    
    obj_labels = {
        "obj1": "Release -NSE",
        "obj2": "Release Abs % Bias",
        "obj3": "Storage -NSE",
        "obj4": "Storage Abs % Bias",
    }
    
    # Load dataframes of results
    obj_dfs = []
    for policy_type in POLICY_TYPES:
        
        # Borg output fname
        fname = f"{OUTPUT_DIR}/MMBorg_3M_{policy_type}_{reservoir_name}_nfe{NFE}_seed71.csv"
        
        # Load dataframes
        df = pd.read_csv(fname)

        # Keep only the objectives
        obj_cols = [col for col in df.columns if col.startswith("obj")]
        df = df.loc[:, obj_cols]

        # Rename the columns
        df.rename(columns=obj_labels, inplace=True)
        obj_cols = list(obj_labels.values())

        
        obj_dfs.append(df.loc[:, obj_cols])
            
    # Labels for the policies
    labels = POLICY_TYPES
    plot_obj_cols = [obj_cols[0], obj_cols[2]]
    
    
    # Plot
    plot_pareto_front_comparison(obj_dfs = obj_dfs, 
                                  labels = labels,
                                  obj_cols = plot_obj_cols,
                                  fname=f"{FIG_DIR}/pareto_front_comparison_{reservoir_name}_{NFE}nfe.png")