import numpy as np
from methods.reservoir.model import Reservoir

# Define the inflow and demand timeseries
inflow = np.array([1, 2, 3, 4, 5])

# Define the reservoir
reservoir = Reservoir(
    inflow = inflow,
    capacity = 10,
    policy_type = "PiecewiseLinear",
    policy_options = None,
    policy_params = [3, 0.3, 0.6, 0.78, 0.2, 0.78],
    release_min = 0,
    release_max = 7.5,
    initial_storage = None
)


# Run the reservoir simulation
reservoir.run()

reservoir.policy.plot()