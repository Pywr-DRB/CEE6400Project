from methods.policies.abstract_policy import AbstractPolicy
import numpy as np
import matplotlib.pyplot as plt

from methods.config import policy_n_params, policy_param_bounds
from methods.config import n_segments, use_inflow_predictor

from methods.config import SEED, DEBUG


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

    def __init__(self, 
                 Reservoir, 
                 policy_params,
                 use_inflow_predictor=use_inflow_predictor,
                 debug=DEBUG):
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
        self.use_inflow_predictor = use_inflow_predictor
        self.debug = debug
        
        # num of linear segments
        self.n_segments = n_segments
        self.param_bounds = policy_param_bounds["PiecewiseLinear"]
        self.n_params = policy_n_params["PiecewiseLinear"]
          
        self.policy_params = policy_params
        self.parse_policy_params()

    def validate_policy_params(self):
        """
        Validates the policy parameters.
        """
        # Check if the number of parameters is correct
        assert len(self.policy_params) == self.n_params, \
            f"PiecewiseLinear policy expected {self.n_params} parameters, got {len(self.policy_params)}."
        
        # check parameter bounds
        for i, p in enumerate(self.policy_params):
            bounds = self.param_bounds[i]
            assert (p >= bounds[0]) and (p <= bounds[1]), \
                f"Parameter with index {i} is out of bounds {bounds}. Value: {p}."
        
    def parse_policy_params(self):
        """Parses policy parameters into segment boundaries and slopes."""
        self.validate_policy_params()

        def parse_segment_params(segment_params, M = self.n_segments):
            """
            Decomposes the policy parameters into segment boundaries, slopes, and intercepts.
            
            Args:
                segment_params (list): A slice of the policy parameters, must be of length 2M-1 for M segments.
                
            Returns:
                tuple: A tuple containing:
                    - x_bounds (list): Segment boundaries, length n_segments-1.
                    - slopes (list): Slopes of the segments, length n_segments.
                    - intercepts (list): Intercepts of the segments, length n_segments.
            """
            
            x_bounds = [0.0] + list(segment_params[:M - 1]) + [1.0]
            theta_vals = segment_params[M - 1:]
            slopes = [np.tan(theta) for theta in theta_vals]

            intercepts = [0.0]
            for i in range(1, M):
                dx = x_bounds[i] - x_bounds[i - 1]
                b = intercepts[i - 1] + slopes[i - 1] * dx
                intercepts.append(b)
            return x_bounds, slopes, intercepts

        # if using inflow predictor, split the policy params into two halves
        # one for storage and one for inflow functions
        if self.use_inflow_predictor:
            midpoint = len(self.policy_params) // 2
            s_params = self.policy_params[:midpoint]
            i_params = self.policy_params[midpoint:]

            (self.storage_bounds,
            self.storage_slopes,
            self.storage_intercepts) = parse_segment_params(s_params)

            (self.inflow_bounds,
            self.inflow_slopes,
            self.inflow_intercepts) = parse_segment_params(i_params)
        else:
            (self.storage_bounds,
            self.storage_slopes,
            self.storage_intercepts) = parse_segment_params(self.policy_params)



    def evaluate(self, S, I=None):
        
        """
        Evaluate the piecewise linear policy function.
        
        Args:
            S (float): Prior storage level.
            I (float, optional): Current inflow. Defaults to None.
            
        Returns:
            float: The computed release.
        """
        
        def segment_eval(x, bounds, slopes, intercepts):
            """
            Resolves the piecewise linear function for a given x.
            
            Uses function:
            f(x) = m_i * (x - x_i) + b_i
            
            where:
                - m_i is the slope of the segment
                - x_i is the lower bound of the segment
                - b_i is the intercept of the segment
            
            
            Args:
                x (float): The input value.
                bounds (list): The segment boundaries.
                slopes (list): The slopes of the segments.
                intercepts (list): The intercepts of the segments.
            
            Returns:
                float: The evaluated value at x.
            """
            for i in range(self.n_segments):
                if bounds[i] <= x < bounds[i + 1]:
                    dx = x - bounds[i]
                    return slopes[i] * dx + intercepts[i]
            if x >= bounds[-1]:
                dx = x - bounds[-2]
                return slopes[-1] * dx + intercepts[-1]
            raise ValueError(f"Value {x} outside bounds: {bounds}")

        z = segment_eval(S, self.storage_bounds, self.storage_slopes, self.storage_intercepts)
        z = max(0.0, min(1.0, z)) 
        
        if self.use_inflow_predictor:
            assert I is not None, "Inflow input required but not provided."
            zI = segment_eval(I, self.inflow_bounds, self.inflow_slopes, self.inflow_intercepts)
            zI = max(0.0, min(1.0, zI))
            
            # divide by 2 to account for the two predictors
            z = (z + zI)/2.0
            
        # Impose bound limits
        z = max(0.0, min(1.0, z)) 
        
        return z



    def get_release(self, timestep):
        """
        Computes the reservoir release for a given timestep based on the 
        current storage level.

        Args:
            timestep (int): The current time step index.

        Returns:
            float: The computed release.
        """
        
        # Get state variables
        I_t = self.Reservoir.inflow_array[timestep]
        S_t = self.Reservoir.initial_storage if timestep == 0 else self.Reservoir.storage_array[timestep - 1]

        # scale S and I to [0, 1]
        S_norm = S_t / self.Reservoir.capacity
        I_norm = I_t / self.Reservoir.inflow_max

        if self.use_inflow_predictor:
            z = self.evaluate(S_norm, I_norm)
        else:
            z = self.evaluate(S_norm) 
        
        # scale back to [0, release_max]
        release = z * self.Reservoir.release_max
        
        # Enforce constraints (defined in AbstractPolicy)
        release = self.enforce_constraints(release)
        release = min(release, S_t + I_t)
        return release
    
    
    
    def plot_3d_policy(self,
                fname="PiecewiseLinearPolicy3D.png",
                save=False):
        """
        Plots the piecewise linear policy function in 3D,
        when use_inflow_predictor is True.
    
        """
        xS_max = self.Reservoir.capacity
        xI_max = self.Reservoir.inflow_max
        
        xS = np.linspace(0.0, 1.0, 100)
        xI = np.linspace(0.0, 1.0, 100)
        
        X, Y = np.meshgrid(xS, xI)
        Z = np.zeros_like(X)
        for i in range(len(xS)):
            for j in range(len(xI)):
                Z[i, j] = self.evaluate(X[i, j], Y[i, j])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel('Storage')
        ax.set_ylabel('Inflow')
        ax.set_zlabel('Release')
        ax.set_title('Piecewise Linear Policy 3D Plot')
        if save:
            plt.savefig(f"./figures/{fname}")
        plt.show()
    

    def plot_storage_policy(self, 
             fname="PiecewiseLinearPolicy.png",
             save=False):
        """
        Plots the piecewise linear policy function.
        """
        
        fig, ax = plt.subplots()
        
        xs = np.linspace(0.0, self.Reservoir.capacity, 100)

        ys = [self.evaluate(x) for x in xs]

        ax.plot(xs, ys)
        
        # Add segment vertical lines
        for x in self.segment_x_bounds:
            ax.axvline(x=x, color='k', 
                       linestyle='--', alpha=0.5,
                       label='Segment Boundaries' if x == 0 else None)
        
        # Add max release line
        ax.axhline(y=self.Reservoir.release_max, 
                   color='r', linestyle='--', alpha=0.5, 
                   label='Max Release')
        
        ax.set_xlabel("Storage")
        ax.set_ylabel("Release")
        ax.set_xlim(0, self.Reservoir.capacity)
        ax.set_title("Piecewise Linear Policy")
        ax.legend()
        if save:
            plt.savefig(f"./figures/{fname}")
        plt.show()


    def plot(self, 
             fname=None,
             save=False):
        """
        Plot the piecewise linear policy function.

        Args:
            fname (str): Filename for saving the plot.
            save (bool): Whether to save the plot as a file.
        """
        
        if self.use_inflow_predictor:
            self.plot_3d_policy(fname=fname, save=save)
        else:
            self.plot_storage_policy(fname=fname, save=save)
        
        