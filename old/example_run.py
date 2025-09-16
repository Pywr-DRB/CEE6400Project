import numpy as np
from methods.reservoir.model import Reservoir
import pandas as pd

from methods.load import load_observations

from methods.metrics.objectives import ObjectiveCalculator
from methods.config import reservoir_min_release, reservoir_max_release, reservoir_capacity

# Set reservoir name
RESERVOIR_NAME = 'fewalter'
POLICY_TYPE = "STARFIT"

# Load inflow & STARFIT data
inflow = load_observations(datatype='inflow', reservoir_name=RESERVOIR_NAME, data_dir="./data/", as_numpy=True)

# Load actual observed release data
observed = load_observations(datatype='release', reservoir_name=RESERVOIR_NAME, data_dir="./data/", as_numpy=True)

# Sets of test parameters
test_params = {
    "PiecewiseLinear": [3, 0.3, 0.6, 0.78, 0.2, 1.0],
    "RBF": [],
    "STARFIT": np.array([
        15.08, 5.0, 20.0,         # NORhi mu, min, max
        0.0, -15.0,               # NORhi alpha, beta
        9.0, 1.6, 14.2,           # NORlo mu, min, max
        -1.0, -30.0,              # NORlo alpha, beta
        0.2118, -0.0357, 0.1302, -0.0248,  # Release harmonic
        -0.123, 0.183, 0.732     # Release adjustment
    ])
}

#datetime
datetime_index = inflow_obs.loc[dt,:].index

# Define the reservoir
reservoir = Reservoir(
    inflow = inflow,
    dates= datetime
    capacity = reservoir_capacity[RESERVOIR_NAME],
    policy_type = POLICY_TYPE,
    policy_params = test_params[POLICY_TYPE],
    release_min = reservoir_min_release[RESERVOIR_NAME],
    release_max =  reservoir_max_release[RESERVOIR_NAME],
    initial_storage = None,
    name = RESERVOIR_NAME,
)


# Run the reservoir simulation
reservoir.run()

reservoir.policy.plot()

# Get simulated release data
simulated = reservoir.release_array  


# Initialize ObjectiveCalculator with all metrics you want to test
metrics_to_test = ['nse', 'rmse', 'kge']
ObjFunc = ObjectiveCalculator(metrics=metrics_to_test)

# Calculate the objectives
objectives = ObjFunc.calculate(obs=observed, sim=simulated)

# Print the results for each metric
print("Objective Calculation Results:")
for metric_name, objective_value in zip(metrics_to_test, objectives):
    print(f"{metric_name}: {objective_value}")