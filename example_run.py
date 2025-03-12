import numpy as np
from methods.reservoir.model import Reservoir

# Define the inflow and demand timeseries
inflow = np.array([1, 2, 3, 4, 5])

# Sets of test parameters
test_params = {
    "PiecewiseLinear": [3, 0.3, 0.6, 0.78, 0.2, 1.0],
    "RBF": [],
    "STARFIT": []
}

test_policy = "PiecewiseLinear"

# Define the reservoir
reservoir = Reservoir(
    inflow = inflow,
    capacity = 10,
    policy_type = test_policy,
    policy_params = test_params[test_policy],
    release_min = 0,
    release_max = 7.5,
    initial_storage = None
)


# Run the reservoir simulation
reservoir.run()

reservoir.policy.plot()