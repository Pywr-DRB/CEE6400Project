import numpy as np
from methods.reservoir.model import Reservoir
import pandas as pd

# Define the inflow and demand timeseries

def load_inflow(reservoir_name, filepath="data/inflow.csv"):
    """
    Loads inflow data for a specific reservoir.
    
    Args:
        reservoir_name (str): Name of the reservoir.
        filepath (str): Path to the inflow CSV file.

    Returns:
        np.array: Inflow timeseries for the given reservoir.
    """
    df = pd.read_csv(filepath)
    if reservoir_name not in df.columns:
        raise ValueError(f"Reservoir '{reservoir_name}' not found in {filepath}. Check CSV headers.")
    return df[reservoir_name].values

starfit_df = pd.read_csv("data/drb_model_istarf_conus.csv")

# Set reservoir name
reservoir_name = "blueMarsh"

# Load inflow & STARFIT data
inflow = load_inflow(reservoir_name)

# Sets of test parameters
test_params = {
    "PiecewiseLinear": [3, 0.3, 0.6, 0.78, 0.2, 1.0],
    "RBF": [],
    "STARFIT":{"starfit_df": starfit_df,  "reservoir_name": reservoir_name}
}

test_policy = "STARFIT"


# Define the reservoir
reservoir = Reservoir(
    inflow = inflow,
    capacity = 10,
    policy_type = test_policy,
    policy_params = test_params[test_policy],
    release_min = 0,
    release_max = 8,
    initial_storage = None
)


# Run the reservoir simulation
reservoir.run()

reservoir.policy.plot()