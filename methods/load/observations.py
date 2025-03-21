import pandas as pd

def load_observations(datatype,
                      reservoir_name=None,
                      data_dir = "./data/",
                      as_numpy=True):
    """
    Loads observational data (inflow, storage or release).
    
    Args:
        datatype (str): The type of data to load. Must be 'inflow', 'storage' or 'release'.
        reservoir_name (str): Name of the reservoir to load data for. If None, all data is returned.
    
    Returns:
        np.array: Inflow timeseries for the given reservoir as a numpy array.
    """
    if datatype not in ["inflow", "storage", "release"]:
        raise ValueError(f"Invalid datatype '{datatype}'. Must be 'inflow', 'storage' or 'release'.")
    
    filepath = f"{data_dir}{datatype}.csv"

    df = pd.read_csv(filepath, index_col = 0, parse_dates=True)
    df.index = pd.to_datetime(df.index.date)
    
    
    # set 0.0 to NaN
    df = df.replace(0.0, pd.NA)
    
    if (reservoir_name is not None) and (reservoir_name not in df.columns):
        raise ValueError(f"Reservoir '{reservoir_name}' not found in {filepath}. Check CSV headers.")
    
    if reservoir_name is None:
        if not as_numpy:
            return df
        else:
            return df.to_numpy()  # Return the whole DataFrame as a numpy array
    else:
        if not as_numpy:
            return df[[reservoir_name]]
        else:        
            return df[reservoir_name].values  # Return the specified reservoir's data as a numpy array
