import pandas as pd

def load_results(file_path: str,
                 obj_labels=None) -> pd.DataFrame:
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


    # separate objectives and variables
    results_obj = results.loc[:, obj_cols]
    results_var = results.loc[:, var_cols]

    return results_obj, results_var