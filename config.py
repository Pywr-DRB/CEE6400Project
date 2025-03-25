"""
Contains configuration specifications for the project.
"""

starfit_policy_options = {
    #add specific options here
}

rbf_policy_options = {
    #add specific options here
}

piecewise_linear_policy_options = {
    #add specific options here
}

#### RESERVOIR CONSTRAINTS ##############
# Conservation releases if applied (in MGD)
# Minimum releases are set according to typical drought operation targets.
reservoir_min_release = {
    "prompton": 3.9,               # 6 cfs = 3.9 MGD (drought release)
    "beltzvilleCombined": 9.7,      # 15 cfs = 9.7 MGD (drought release)
    "fewalter": 27.8,               # 43 cfs = 27.8 MGD (drought release)
    "blueMarsh": 13.6               # 21 cfs = 13.6 MGD (drought release)
}

# Maximum releases are estimated from typical operating scenarios
# These are not absolute maximums but common release levels during normal conditions.
reservoir_max_release = {
    "prompton": 3.9,                # Same as min release (minimal pass-through flow)
    "beltzvilleCombined": 22.6,      # 35 cfs = 22.6 MGD (normal release)
    "fewalter": 32.0,               # 50 cfs = 32 MGD (normal release)
    "blueMarsh": 32.0               # 50 cfs = 32 MGD (normal release, including supply withdrawal)
}

# Reservoir capacities (in Million Gallons - MG)
# Based on total storage capacity to full pool (spillway crest elevation).
reservoir_capacity = {
    "prompton": 16800,              # 16,800 MG
    "beltzvilleCombined": 22300,     # 22,300 MG (approximate to spillway crest)
    "fewalter": 36000,               # 36,000 MG
    "blueMarsh": 16300              # 16,300 MG
}