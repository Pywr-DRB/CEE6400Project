import os
import pandas as pd
import numpy as np

from methods.utils import get_overlapping_datetime_indices

def load_observations(datatype,
                      reservoir_name=None,
                      data_dir = "../obs_data/processed",
                      as_numpy=True):
    """
    Loads observational data (inflow, storage or release).
    
    Args:
        datatype (str): The type of data to load. Must be 'inflow', 'storage' or 'release'.
        reservoir_name (str): Name of the reservoir to load data for. If None, all data is returned.
    
    Returns:
        np.array: Inflow timeseries for the given reservoir as a numpy array.
    """
    if datatype not in ["inflow", "inflow_scaled", "storage", "release"]:
        raise ValueError(f"Invalid datatype '{datatype}'. Must be 'inflow', 'storage' or 'release'.")
    
    filepath = f"{data_dir}/{datatype}.csv"

    df = pd.read_csv(filepath, index_col = 0, parse_dates=True)
    df.index = pd.to_datetime(df.index.date)
    
    
    # set 0.0 to NaN
    df = df.replace(0.0, pd.NA)
    
    if (reservoir_name is not None) and (reservoir_name not in df.columns.to_list()):
        print(f"Warning: '{reservoir_name}' not found in {filepath}. Columns: {df.columns.to_list()}")
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
                           as_numpy=True,
                           scaled_inflows=True):
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
    
    if scaled_inflows:
        # Load scaled inflow observations
        inflow_obs = load_observations(datatype='inflow_scaled', 
                                       reservoir_name=reservoir_name, 
                                       data_dir=data_dir, as_numpy=False)
    else:
        # Load raw inflow observations
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


def scale_inflow_observations(inflow_obs, release_obs):
    """
    Scales inflow observations based on release observations,
    assuming that total inflow volme is equal to total release volume.
    
    Scaling is applied on a monthly basis.
    
    Args:
        inflow_obs (pd.DataFrame): Inflow observations.
        release_obs (pd.DataFrame): Release observations.
    
    Returns:
        pd.DataFrame: Scaled inflow observations.
    """
    assert isinstance(inflow_obs, pd.DataFrame), \
        "inflow_obs must be a pandas DataFrame."
    assert isinstance(release_obs, pd.DataFrame), \
        "release_obs must be a pandas DataFrame."
    assert inflow_obs.index.equals(release_obs.index), \
        "inflow_obs and release_obs must have the same index."
    
    # get monthly inflow and release volumes
    inflow_monthly = inflow_obs.resample('MS').sum()
    release_monthly = release_obs.resample('MS').sum()
    
    # get monthly scaling factor
    scale_factor = release_monthly / inflow_monthly
    
    # make sure scaling is >= 1.0
    scale_factor = scale_factor.clip(lower=1.0)
    
    # apply scaling factor to inflow observations
    inflow_scaled = inflow_obs.copy()
    
    # Apply for each month, year in the series
    for year in inflow_scaled.index.year.unique():
        for month in inflow_scaled.index.month.unique():
            
            # skip if no data for this month
            datetime = pd.to_datetime(f"{year}-{month}-01")
            if datetime not in inflow_scaled.index:
                continue
            
            # Get the scaling factor for this month
            scale = scale_factor.loc[(scale_factor.index.year == year) &
                                     (scale_factor.index.month == month)].values[0]
            
            # Apply the scaling factor to the inflow observations
            inflow_scaled.loc[(inflow_scaled.index.year == year) & 
                             (inflow_scaled.index.month == month), :] *= scale
    
    return inflow_scaled