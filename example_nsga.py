import numpy as np
import pandas as pd
from platypus import Problem, Real, NSGAII
from methods.reservoir.model import Reservoir
from methods.load import load_observations
from methods.metrics.objectives import ObjectiveCalculator


# Set reservoir name
reservoir_name = "blueMarsh"

# Load inflow & STARFIT data
inflow = load_observations(datatype='inflow', reservoir_name=reservoir_name, data_dir="./data/")

# Load actual observed release data
observed = load_observations(datatype='release', reservoir_name=reservoir_name, data_dir="./data/")

# Initialize ObjectiveCalculator with all metrics you want to test
metrics_to_test = ['nse', 'rmse', 'kge']
ObjFunc = ObjectiveCalculator(metrics=metrics_to_test)


# Define the Evaluation Function for NSGA-II
def evaluate(vars):
    # Define the reservoir with the NSGA-II generated parameters
    reservoir = Reservoir(
        inflow=inflow,
        capacity=10,
        policy_type="STARFIT",
        policy_params=vars,
        release_min=0,
        release_max=8,
        initial_storage=None,
        name=reservoir_name
    )

    # Run the simulation
    reservoir.run()
    
    # Retrieve simulated release data
    simulated = reservoir.release_array

    # Calculate the objectives
    objectives = ObjFunc.calculate(obs=observed, sim=simulated)
    
    # Return objective values as a tuple (for NSGA-II)
    return tuple(objectives)


# Initialize the NSGA-II Problem
num_variables = 17  # Number of parameters to optimize
problem = Problem(num_variables, len(metrics_to_test))

# Define the parameter range for optimization (Adjust these as needed)
for i in range(num_variables):
    problem.types[i] = Real(0.0, 1.0)  # Adjust the range as needed

# Attach the evaluation function
problem.function = evaluate

# Initialize the NSGA-II algorithm
algorithm = NSGAII(problem)

# Run the optimization
algorithm.run(100) 


# Extract the results
results = np.array([solution.objectives for solution in algorithm.result])

# Print the best solutions
print("Optimization Results:")
for metric_name, best_value in zip(metrics_to_test, results[0]):
    print(f"{metric_name}: {best_value}")


# Plot the Pareto Front (2D Plot Example)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(results[:, 0], results[:, 1], c=results[:, 2], cmap='viridis', s=20)
plt.colorbar(label='KGE')
plt.title("Pareto Front: NSE vs RMSE")
plt.xlabel("NSE")
plt.ylabel("RMSE")
plt.show()
