import os
import pandas as pd

from methods.utils import get_overlapping_datetime_indices

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



def get_observational_training_data(reservoir_name, 
                           data_dir,
                           as_numpy=True):
    """
    Loads training data (inflow, release and storage)
    for a given reservoir, for maximum overlapping timeperiod.
    
    Args:
        reservoir_name (str): Name of the reservoir to load data for.
        data_dir (str): Directory where the data files are located.
        as_numpy (bool): If True, returns numpy arrays. If False, returns pandas DataFrames.
    
    Returns:
        tuple: A tuple containing inflow, release and storage data as numpy arrays or DataFrames.
    """

    assert os.path.exists(data_dir), \
        f"Data directory '{data_dir}' does not exist. Please check the data_dir path."
    
    
    inflow_obs = load_observations(datatype='inflow', 
                                reservoir_name=reservoir_name, 
                                data_dir=data_dir, as_numpy=False)

    release_obs = load_observations(datatype='release', 
                                    reservoir_name=reservoir_name, 
                                    data_dir=data_dir, as_numpy=False)

    storage_obs = load_observations(datatype='storage',
                                    reservoir_name=reservoir_name, 
                                    data_dir=data_dir, as_numpy=False)
    
    # get overlapping datetime indices, 
    # when all data is available for this reservoir
    dt = get_overlapping_datetime_indices(inflow_obs, release_obs, storage_obs)

    assert len(dt) > 0, \
        f"No overlapping datetime indices found for reservoir '{reservoir_name}'. "

    # subset data
    inflow_obs = inflow_obs.loc[dt,:]
    release_obs = release_obs.loc[dt,:]
    storage_obs = storage_obs.loc[dt,:]
    
    if as_numpy:
        # Return just arrays
        return inflow_obs.values, release_obs.values, storage_obs.values
    else:
        # Return the DataFrames
        return inflow_obs, release_obs, storage_obs
