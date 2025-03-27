import pandas as pd
import numpy as np

def get_overlapping_datetime_indices(*dfs):
    """
    Get the indices of the overlapping datetime indices of the input dataframes.
    
    Args:
        *dfs: DataFrames with datetime indices.
    
    Returns:
        pd.DatetimeIndex: The overlapping datetime indices.
    """
    indices = dfs[0].dropna().index
    for df in dfs[1:]:
        indices = indices.intersection(df.dropna().index)

    indices = indices.dropna()
    
    # keep only continuous non-missing values
    keep_indices = []
    for i in range(len(indices) - 1):
        if indices[i] + pd.Timedelta('1D') == indices[i+1]:
            keep_indices.append(indices[i])
    keep_indices.append(indices[-1])
    indices = pd.DatetimeIndex(keep_indices)
    return indices