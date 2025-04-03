"""
Contains configuration specifications for the project.
"""
import numpy as np
from methods.utils.release_constraints import get_release_minmax_release_dict


### Random ##################
SEED = 71
DEBUG = True

### Constants ###############
cfs_to_mgd = 0.645932368556


### Reservoirs ###############
reservoir_options = [
    'blueMarsh',
    'beltzvilleCombined',
    'fewalter',
    'prompton',
]


### Polcy Settings ###############

policy_type_options = [
    "STARFIT",
    "RBF",
    "PiecewiseLinear",
]


## RBF
n_rbfs = 2              # Number of radial basis functions (RBFs) used in the policy
n_rbf_inputs = 2
n_rbf_params = n_rbfs * (2 * n_rbf_inputs + 1)
rbf_param_bounds = [[0.0, 1.0]] * n_rbf_params


## STARFIT
n_starfit_params = 17         # Number of parameters in STARFIT policy
# param order = [ NORhi_mu, NORhi_min, NORhi_max, NORhi_alpha, NORhi_beta,
#                  NORlo_mu, NORlo_min, NORlo_max, NORlo_alpha, NORlo_beta,
#                  Release_alpha1, Release_alpha2, Release_beta1, Release_beta2,
#                  Release_c, Release_p1, Release_p2]

starfit_param_bounds = [
    [14.0244, 16.1356],   # NORhi_mu
    [0.0, 2.0],           # NORhi_min
    [93.0, 100.0],        # NORhi_max
    [-0.2461, -0.2139],   # NORhi_alpha
    [-2.1507, -1.8693],   # NORhi_beta
    [11.6994, 13.4606],   # NORlo_mu
    [11.4483, 13.1717],   # NORlo_min
    [14.4336, 16.6064],   # NORlo_max
    [0.1860, 0.2140],     # NORlo_alpha
    [-4.3656, -3.7944],   # NORlo_beta
    [-4.08, 240.0],       # Release_alpha1
    [-0.5901, 84.7844],   # Release_alpha2
    [-1.2104, 83.9024],   # Release_beta1
    [-52.3545, 0.4454],   # Release_beta2
    [-1.4, 63.516],       # Release_c
    [0.0, 17.02],         # Release_p1
    [0.0, 0.957]          # Release_p2
]



## Piecewise Linear
use_inflow_predictor = True
n_segments = 3         # linear segments 

# num params =  (2 * n_segments + 1) * n_predictors
if use_inflow_predictor:
    n_piecewise_linear_params = (2 * n_segments - 1) * 2   
else:
    n_piecewise_linear_params = 2 * n_segments - 1 

# param order = [ segment_breakpoints, slopes] 
piecewise_linear_param_bounds = [
    [[i/(n_segments-1), (i+1)/(n_segments-1)] for i in range(n_segments-1)] +   # # Segment breakpoints (x_i) in [0.0, 1.0]
    [[0.0, np.pi/2]] * n_segments  # Segment slopes (θ_i) in [0.0, π/3]
][0]

# double the parameter bounds, if inflow predictor is used
if use_inflow_predictor:
    piecewise_linear_param_bounds = piecewise_linear_param_bounds + piecewise_linear_param_bounds
    

## Dictionaries of configurations
policy_n_params = {
    "STARFIT": n_starfit_params,
    "RBF": n_rbf_params,                  
    "PiecewiseLinear": n_piecewise_linear_params,  
}

policy_param_bounds = {
    "STARFIT": starfit_param_bounds,
    "RBF": rbf_param_bounds,
    "PiecewiseLinear": piecewise_linear_param_bounds,
}


#### RESERVOIR CONSTRAINTS ##############

# Reservoir capacities (in Million Gallons - MG)
reservoir_capacity = {
    "prompton": 16800,              # 16,800 MG
    "beltzvilleCombined": 22300,     # 22,300 MG (approximate to spillway crest)
    "fewalter": 36000,               # 36,000 MG
    "blueMarsh": 16300              # 16,300 MG
}


# Conservation releases at lower reservoirs
# Specified in the DRBC Water Code Table 4
drbc_conservation_releases = {
    "blueMarsh": 50 * cfs_to_mgd,
    "beltzvilleCombined": 35 * cfs_to_mgd,
    "fewalter": 50 * cfs_to_mgd,
}

# Observed min/max releases
obs_release_min, obs_release_max = get_release_minmax_release_dict()


# Assign min/max releases for each reservoir
reservoir_min_release = {}
reservoir_max_release = {}
for r in reservoir_options:
    reservoir_max_release[r] = obs_release_max[r]
    
    # Set min (conservation) releases from WaterCode
    if r in drbc_conservation_releases:
        reservoir_min_release[r] = drbc_conservation_releases[r]
    else:
        reservoir_min_release[r] = obs_release_min[r]
    



