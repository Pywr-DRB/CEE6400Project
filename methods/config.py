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


### MOEA Objectives ##########

# Metrics used for Borg
METRICS = [
    'neg_nse',           # Negative Nash Sutcliffe Efficiency
    'abs_pbias',         # Absolute Percent Bias
]


metric_epsilons = {
    'neg_nse': 0.01,
    'neg_kge': 0.01,
    'abs_pbias': 0.1,
    'rmse': 0.1,
}

for m in METRICS:
    for q in ["Q20", "Q80"]:
        metric_epsilons[f"{q}_{m}"] = metric_epsilons[m]
    metric_epsilons[f"log_{m}"] = metric_epsilons[m]

EPSILONS = [metric_epsilons[m] for m in METRICS]


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
starfit_param_bounds = [
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
    





# # Conservation releases if applied (in MGD)
# # Minimum releases are set according to typical drought operation targets.
# reservoir_min_release = {
#     "prompton": 3.9,               # 6 cfs = 3.9 MGD (drought release)
#     "beltzvilleCombined": 9.7,      # 15 cfs = 9.7 MGD (drought release)
#     "fewalter": 27.8,               # 43 cfs = 27.8 MGD (drought release)
#     "blueMarsh": 13.6               # 21 cfs = 13.6 MGD (drought release)
# }

# # Maximum releases are estimated from typical operating scenarios
# # These are not absolute maximums but common release levels during normal conditions.
# reservoir_max_release = {
#     "prompton": 3.9,                # Same as min release (minimal pass-through flow)
#     "beltzvilleCombined": 22.6,      # 35 cfs = 22.6 MGD (normal release)
#     "fewalter": 32.0,               # 50 cfs = 32 MGD (normal release)
#     "blueMarsh": 32.0               # 50 cfs = 32 MGD (normal release, including supply withdrawal)
# }

