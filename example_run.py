import numpy as np
from methods.reservoir.model import Reservoir
import pandas as pd

from methods.load import load_observations

from methods.metrics.objectives import ObjectiveCalculator
from config import reservoir_min_release, reservoir_max_release, reservoir_capacity

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
    "STARFIT": np.array([  # Example STARFIT optimization parameters
        0.5, 0.3, 0.7, 0.2, 0.4,  # NORhi parameters
        0.3, 0.1, 0.5, 0.1, 0.2,  # NORlo parameters
        0.4, 0.3, 0.6, 0.5,  # Seasonal release params
        1.0, 0.7, 0.8  # Release coefficients
    ])}



# Define the reservoir
reservoir = Reservoir(
    inflow = inflow,
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
