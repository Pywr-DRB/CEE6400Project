import pandas as pd

from methods.load.observations import load_observations, get_overlapping_datetime_indices
from methods.load.observations import scale_inflow_observations

from methods.config import DATA_DIR, reservoir_options

if __name__ == "__main__":
    
    ##############################################################
    ### Load data ################################################
    ##############################################################
    # raw observations for all reservoirs
    inflow_obs = load_observations(datatype='inflow',
                                   reservoir_name = None,
                                   data_dir = DATA_DIR,
                                   as_numpy=False)
    
    release_obs = load_observations(datatype='release',
                                     reservoir_name = None,
                                     data_dir = DATA_DIR,
                                     as_numpy=False)
    
    storage_obs = load_observations(datatype='storage',
                                        reservoir_name = None,
                                        data_dir = DATA_DIR,
                                        as_numpy=False)
    
    ##############################################################
    ### Make scaled inflow dataset, with NA default values #######
    ##############################################################
    inflow_obs_scaled = inflow_obs.copy()
    inflow_obs_scaled.loc[:,:] = float('nan')

    # apply scaling 1 at a time
    for reservoir in reservoir_options:
        
        # Get overlapping datetime indices, for inflows and releases
        dt = get_overlapping_datetime_indices(inflow_obs.loc[:, [reservoir]], 
                                            release_obs.loc[:, [reservoir]])
        
        # Scale inflow observations, to match the release volume
        res_inflow_scaled = scale_inflow_observations(inflow_obs.loc[dt, [reservoir]], 
                                                      release_obs.loc[dt, [reservoir]])
        
        # Store scaled inflow observations
        inflow_obs_scaled[reservoir] = res_inflow_scaled[reservoir]
        
    # Save scaled inflow observations
    inflow_obs_scaled.to_csv(DATA_DIR + '/inflow_scaled.csv')