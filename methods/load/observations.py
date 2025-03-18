import pandas as pd

# Define the inflow and demand timeseries
def load_observations(datatype,
                      reservoir_name=None,
                      data_dir = "../../data/",
                      as_numpy=False):
    """
    Loads observational data (inflow, storage or release).
    
    Args:
        datatype (str): The type of data to load. Must be 'inflow', 'storage' or 'release'.
        reservoir_name (str): Name of the reservoir to load data for. If None, all data is returned.
        as_numpy (bool): If True, return data as a numpy array. If False, return as a pandas DataFrame.


    Returns:
        np.array: Inflow timeseries for the given reservoir.
    """
    if datatype not in ["inflow", "storage", "release"]:
        raise ValueError(f"Invalid datatype '{datatype}'. Must be 'inflow', 'storage' or 'release'.")
    
    filepath = f"{data_dir}{datatype}.csv"

    df = pd.read_csv(filepath)
    
    if (reservoir_name is not None) and (reservoir_name not in df.columns):
        raise ValueError(f"Reservoir '{reservoir_name}' not found in {filepath}. Check CSV headers.")
    
    if reservoir_name is None:
        return df
    elif as_numpy:
        return df[reservoir_name].values
    else:
        return df[[reservoir_name]]
