from methods.policies.abstract_policy import AbstractPolicy
import numpy as np
import matplotlib.pyplot as plt

class PiecewiseLinear(AbstractPolicy):
    """
    A piecewise linear policy class.

    This policy defines a reservoir release function as a series of linear 
    segments. Each segment is defined by:
    - x_i: the lower bound of the segment (storage threshold)
    - θ_i: the angle of the segment relative to the x-axis

    The first segment always starts at x_0 = 0. The number of segments is 
    determined by the first value in `policy_params`. The remaining values 
    define x_i and θ_i for the segments.

    Attributes:
        Reservoir (Reservoir): The reservoir instance associated with the policy.
        n_segments (int): The number of linear segments in the policy.
        segment_x_bounds (list): Storage thresholds defining segment boundaries.
        segment_theta_vals (list): Angles defining segment slopes.
        slopes (list): Computed slopes for each segment.
        intercepts (list): Computed intercepts for each segment.
    """

    def __init__(self, Reservoir, policy_params):
        """
        Initializes the PiecewiseLinear policy.

        Args:
            Reservoir (Reservoir): The reservoir instance.
            policy_params (list): A list of policy parameters where:
                - policy_params[0] defines the number of segments (M).
                - The next M-1 values define x_i (excluding x_0 = 0).
                - The last M values define θ_i.
        """
        self.Reservoir = Reservoir
        self.n_segments = int(policy_params[0])  # First param is the number of segments
        self.policy_params = policy_params
        self.parse_policy_params()

    def validate_policy_params(self):
        """
        Validates the policy parameters.
        """
        expected_n_params = 2 * self.n_segments - 1
        actual_n_params = len(self.policy_params) - 1  # Excluding num_segments

        if expected_n_params != actual_n_params:
            raise ValueError(f"Incorrect number of parameters. Expected {expected_n_params}, got {actual_n_params}.")

    def parse_policy_params(self):
        """Parses policy parameters into segment boundaries and slopes."""

        self.validate_policy_params()

        # Storage thresholds (segment boundaries)
        self.segment_x_bounds = [0.0] + self.policy_params[1:self.n_segments]

        # Scale segment boundaries by reservoir capacity
        self.segment_x_bounds = [x * self.Reservoir.capacity for x in self.segment_x_bounds]

        # Append reservoir capacity as the last segment boundary
        self.segment_x_bounds.append(self.Reservoir.capacity)

        ## Convert to mx+b form for each segment
        # Angles (converted to slopes)
        self.segment_theta_vals = self.policy_params[self.n_segments:]
        self.segment_slopes = [np.tan(theta) for theta in self.segment_theta_vals]

        # Compute intercepts for each segment
        self.segment_intercepts = [0.0]  # First segment starts at (0,0)
        for i in range(1, self.n_segments):
            prev_x = self.segment_x_bounds[i - 1]
            prev_b = self.segment_intercepts[i - 1]
            x_delta = self.segment_x_bounds[i] - prev_x
            b = prev_b + self.segment_slopes[i - 1] * (x_delta)
            self.segment_intercepts.append(b)

    def evaluate(self, X):
        """
        Evaluate the policy function at a given storage level.

        Args:
            X (float): The current storage level.

        Returns:
            float: The computed release.
        """
        segment_idx = None
        for i in range(self.n_segments):
            if X >= self.segment_x_bounds[i] and X < self.segment_x_bounds[i + 1]:
                segment_idx = i
                break
            elif X >= self.Reservoir.capacity:
                segment_idx = self.n_segments - 1
                break
        if segment_idx is None:
            msg = f"No segment index found for storage {X}."
            msg += f"Storage bounds: {self.segment_x_bounds}"
            raise ValueError(msg)

        # calculate release
        x_delta = X - self.segment_x_bounds[segment_idx]
        release = self.segment_slopes[segment_idx] * x_delta + self.segment_intercepts[segment_idx]

        release = self.enforce_constraints(release)
        
        return release 


    def get_release(self, timestep):
        """
        Computes the reservoir release for a given timestep based on the 
        current storage level.

        Args:
            timestep (int): The current time step index.

        Returns:
            float: The computed release.
        """
        I_t = self.Reservoir.inflow_array[timestep]

        if timestep == 0:
            S_t = self.Reservoir.initial_storage
        else:
            S_t = self.Reservoir.storage_array[timestep - 1]

        release = self.evaluate(X = S_t)
        release = self.enforce_constraints(release)
        release = min(release, S_t + I_t)
        return release

    def plot(self, fname="PiecewiseLinearPolicy.png"):
        """
        Plots the piecewise linear policy function.
        """
        fig, ax = plt.subplots()
        xs = np.linspace(0.0, self.Reservoir.capacity, 100)
        ys = [self.evaluate(x) for x in xs]

        ax.plot(xs, ys)
        ax.set_xlabel("Storage")
        ax.set_ylabel("Release")
        ax.set_title("Piecewise Linear Policy")
        plt.savefig(f"./figures/{fname}")
        plt.show()
