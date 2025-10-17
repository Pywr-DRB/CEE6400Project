import pandas as pd
from pywrdrb.release_policies.config import OBJ_FILTER_BOUNDS



def filter_solutions(df: pd.DataFrame,
                     obj_bounds = OBJ_FILTER_BOUNDS) -> pd.DataFrame:
    
    """
    Filter solutions based on objective bounds.
    Args:
        df (pd.DataFrame): DataFrame containing the solutions.
        obj_bounds (dict): Dictionary with objective bounds.
            Keys are objective names and values are tuples of (min, max).
    Returns:
        pd.DataFrame: Filtered DataFrame.    
    """
    # check that keys of obj_bounds are in the DataFrame
    for obj in obj_bounds.keys():
        if obj not in df.columns:
            raise ValueError(f"Objective {obj} in obj_bounds not found in DataFrame columns.")


    # Filter the DataFrame based on the objective bounds
    for obj, bounds in obj_bounds.items():
        min_bound, max_bound = bounds
        df = df[(df[obj] >= min_bound) & (df[obj] <= max_bound)]
    
    # reset index
    df.reset_index(drop=True, inplace=True)
    
    return df



def load_results(file_path: str,
                 obj_labels=None,
                 filter=False,
                 obj_bounds=OBJ_FILTER_BOUNDS) -> pd.DataFrame:
    """
    Load results from a CSV file and return a DataFrame.

    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the results.
    """
    
    # Load the results from the CSV file
    results = pd.read_csv(file_path)
    
    # Relabel objectives if obj_labels are provided
    obj_cols = [col for col in results.columns if col.startswith("obj")]
    var_cols = [col for col in results.columns if col.startswith("var")]
    
    if obj_labels is not None:
        for col in obj_cols:
            # Rename the column using the provided labels
            new_col = obj_labels.get(col, col)
            results.rename(columns={col: new_col}, inplace=True)

        # Change obj_cols to use the new labels
        obj_cols = [obj_labels.get(col, col) for col in obj_cols]
        

    # Modify the sign of the objectives,
    # only for specific objs (eg. NSE)
    for col in obj_cols:
        # NSE 
        if ('nse' in col.lower()):
            results[col] = -results[col]
        elif ('kge' in col.lower()):
            results[col] = -results[col]
        elif ('inertia' in col.lower()):
            results[col] = -results[col]


    # Filter the results based on the objective bounds
    # if filter and obj_bounds is not None:
    #     if 'prompton' not in file_path:    
    #         results = filter_solutions(results, obj_bounds)
    #     else:
    #         # prompton has f'd up Storage NSE
    #         obj_bounds['Storage NSE'] = (-10, 1.0)
    #         results = filter_solutions(results, obj_bounds)

    if filter and obj_bounds is not None:
        before = len(results)
        results = filter_solutions(results, obj_bounds=obj_bounds)
        after = len(results)
        print(f"[FILTER] {file_path}: {before} → {after} rows")

    # separate objectives and variables
    results_obj = results.loc[:, obj_cols]
    results_var = results.loc[:, var_cols]
    
    return results_obj, results_var