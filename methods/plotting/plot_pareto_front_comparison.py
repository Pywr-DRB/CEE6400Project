# methods/plotting/plot_pareto_front_comparison.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_pareto_front_comparison(obj_dfs, 
                                 labels,
                                 obj_cols,
                                 title="Pareto Front Comparison",
                                 x_lims=None,
                                 y_lims=None,
                                 ideal=None,
                                 fname=None):
    """
    """
    
    assert(len(obj_dfs) == len(labels)), "Number of dataframes must match number of labels"
    for i, df in enumerate(obj_dfs):
        for col in obj_cols:
            assert(col in df.columns), f"Column {col} not found in {i}th dataframe"
    
    
    # Create a 2D scatter of pareto fronts
    fig, ax = plt.subplots()
    for i, df in enumerate(obj_dfs):
        ax.scatter(df[obj_cols[0]], df[obj_cols[1]], label=labels[i], alpha=0.3, zorder=5)
        ax.set_xlabel(obj_cols[0], fontsize=12)
        ax.set_ylabel(obj_cols[1], fontsize=12)
    
    if ideal is not None:
        ax.scatter(ideal[0], ideal[1], color='gold', 
                   label='Ideal', marker='*', s=500, zorder=5)
    
    if x_lims is not None:
        ax.set_xlim(x_lims[0], x_lims[1])
    if y_lims is not None:
        ax.set_ylim(y_lims[0], y_lims[1])       
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=10, ncol=4)
    ax.set_title(title)
    ax.grid(zorder=0)
    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, dpi=200)
    
    plt.show()
    return