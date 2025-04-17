from methods.load import load_observations

def get_release_minmax_release_dict():
    
    # load observed data
    release_obs = load_observations(datatype='release',
                                    data_dir="./obs_data/processed", as_numpy=False)
    
    # 'min' release is calculated using 0.01 quantile of the observed release
    # 'max' release is calculated using 0.99 quantile of the observed release
    
    release_min_dict = {}
    release_max_dict = {}
    
    for reservoir_name in release_obs.columns:
        release_min_dict[reservoir_name] = release_obs[reservoir_name].quantile(0.005)
        release_max_dict[reservoir_name] = release_obs[reservoir_name].quantile(0.995)
        
    return release_min_dict, release_max_dict