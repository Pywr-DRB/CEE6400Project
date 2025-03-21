import numpy as np
from methods.reservoir.model import Reservoir
import pandas as pd

from methods.load import load_observations

from methods.metrics.objectives import ObjectiveCalculator

# Set reservoir name
reservoir_name = "blueMarsh"

# Load inflow & STARFIT data
inflow = load_observations(datatype='inflow', reservoir_name=reservoir_name, data_dir="./data/")


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


test_policy = "STARFIT"


# Define the reservoir
reservoir = Reservoir(
    inflow = inflow,
    capacity = 10,
    policy_type = test_policy,
    policy_params = test_params[test_policy],
    release_min = 0,
    release_max = 8,
    initial_storage = None,
    name = reservoir_name
)


# Run the reservoir simulation
reservoir.run()

reservoir.policy.plot()

# Mock observed and simulated data
observed = np.array([1, 2, 3, 4, 5])
simulated = np.array([1.1, 2.1, 2.9, 3.8, 4.9])

# Initialize ObjectiveCalculator with all metrics you want to test
metrics_to_test = ['nse', 'rmse', 'kge']
ObjFunc = ObjectiveCalculator(metrics=metrics_to_test)

# Calculate the objectives
objectives = ObjFunc.calculate(obs=observed, sim=simulated)

# Print the results for each metric
print("Objective Calculation Results:")
for metric_name, objective_value in zip(metrics_to_test, objectives):
    print(f"{metric_name}: {objective_value}")