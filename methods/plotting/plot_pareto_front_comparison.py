import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods.config import OUTPUT_DIR, FIG_DIR


def plot_pareto_front_comparison(obj_dfs, 
                                 labels,
                                 obj_cols,
                                 fname=None):
    """
    """
    
    assert(len(obj_dfs) == len(labels)), "Number of dataframes must match number of labels"
    for i, df in enumerate(obj_dfs):
        for col in obj_cols:
            assert(col in df.columns), f"Column {col} not found in {i}th dataframe"
    
    # Create a 2 or 3D scatter of pareto fronts
    use_x_upper = 0.0
    use_y_upper = 0.0
    if len(obj_cols) == 2:
        fig, ax = plt.subplots()
        for i, df in enumerate(obj_dfs):
            ax.scatter(df[obj_cols[0]], df[obj_cols[1]], label=labels[i], alpha=0.3)
            ax.set_xlabel(obj_cols[0])
            ax.set_ylabel(obj_cols[1])
        
            # get 90% confidence interval
            x_upper = np.percentile(df[obj_cols[0]], 90)
            y_upper = np.percentile(df[obj_cols[1]], 90)
            if x_upper > use_x_upper:
                use_x_upper = x_upper
            if y_upper > use_y_upper:
                use_y_upper = y_upper
        
        ax.set_xlim(-1.0, use_x_upper)
        ax.set_ylim(-1.0, use_y_upper)
        ax.legend()
        ax.set_title("Pareto Front Comparison")
        ax.grid()
    
    elif len(obj_cols) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, df in enumerate(obj_dfs):
            ax.scatter(df[obj_cols[0]], df[obj_cols[1]], df[obj_cols[2]], label=labels[i], alpha=0.5)
            ax.set_xlabel(obj_cols[0])
            ax.set_ylabel(obj_cols[1])
            ax.set_zlabel(obj_cols[2])
            ax.legend()
            ax.set_title("Pareto Front Comparison")
            ax.grid()
    

    if fname is not None:
        plt.savefig(fname)
    
    plt.show()
    return
