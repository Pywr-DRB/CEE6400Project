# import numpy as np
# import matplotlib.pyplot as plt

# from methods.policies.abstract_policy import AbstractPolicy
# from methods.config import policy_n_params, policy_param_bounds
# from methods.config import n_segments, n_piecewise_linear_inputs, drbc_conservation_releases


# class PiecewiseLinear(AbstractPolicy):
#     """
#     A piecewise linear policy class.

#     This policy defines a reservoir release function as a series of linear 
#     segments. Each segment is defined by:
#     - x_i: the lower bound of the segment (storage threshold)
#     - θ_i: the angle of the segment relative to the x-axis

#     The first segment always starts at x_0 = 0. The number of segments is 
#     determined by the first value in `policy_params`. The remaining values 
#     define x_i and θ_i for the segments.

#     Attributes:
#         Reservoir (Reservoir): The reservoir instance associated with the policy.
#         n_segments (int): The number of linear segments in the policy.
#         segment_x_bounds (list): Storage thresholds defining segment boundaries.
#         segment_theta_vals (list): Angles defining segment slopes.
#         slopes (list): Computed slopes for each segment.
#         intercepts (list): Computed intercepts for each segment.
#     """

#     def __init__(self, 
#                  Reservoir, 
#                  policy_params):
#         """
#         Initializes the PiecewiseLinear policy.

#         Args:
#             Reservoir (Reservoir): The reservoir instance.
#             policy_params (list): A list of policy parameters where:
#                 - policy_params[0] defines the number of segments (M).
#                 - The next M-1 values define x_i (excluding x_0 = 0).
#                 - The last M values define θ_i.
#         """
        
#         # Reservoir sim data
#         self.Reservoir = Reservoir
#         self.dates = Reservoir.dates
        
#         # Policy parameters
#         self.n_segments = n_segments
#         self.n_inputs = n_piecewise_linear_inputs
#         self.param_bounds = policy_param_bounds["PiecewiseLinear"]
#         self.n_params = policy_n_params["PiecewiseLinear"]
        
#         # X (input) max and min values
#         # used to normalize the input data
#         # X = [storage, inflow, day_of_year]
#         self.x_min = np.array([0.0, 
#                                self.Reservoir.inflow_min,
#                                1.0])
        
#         self.x_max = np.array([self.Reservoir.capacity, 
#                                self.Reservoir.inflow_max,
#                                366.0])
        
#         self.policy_params = policy_params
#         self.parse_policy_params()

#     def validate_policy_params(self):
#         """
#         Validates the policy parameters.
#         """
#         # Check if the number of parameters is correct
#         assert len(self.policy_params) == self.n_params, \
#             f"PiecewiseLinear policy expected {self.n_params} parameters, got {len(self.policy_params)}."
        
#         # check parameter bounds
#         for i, p in enumerate(self.policy_params):
#             bounds = self.param_bounds[i]
#             assert (p >= bounds[0]) and (p <= bounds[1]), \
#                 f"Parameter with index {i} is out of bounds {bounds}. Value: {p}."
            
#         return
        
#     def parse_policy_params(self):
#         """
#         Parses policy parameters into segment boundaries and slopes.
#         """

#         # Validate the policy parameters
#         self.validate_policy_params()

#         def parse_segment_params(segment_params, M = self.n_segments):
#             """
#             Decomposes the policy parameters into segment boundaries, slopes, and intercepts.
            
#             Args:
#                 segment_params (list): A slice of the policy parameters, must be of length 2M-1 for M segments.
                
#             Returns:
#                 tuple: A tuple containing:
#                     - x_bounds (list): Segment boundaries, length n_segments-1.
#                     - slopes (list): Slopes of the segments, length n_segments.
#                     - intercepts (list): Intercepts of the segments, length n_segments.
#             """
            
#             x_bounds = [0.0] + list(segment_params[:M - 1]) + [1.0]
#             theta_vals = segment_params[M - 1:]
#             slopes = [np.tan(theta) for theta in theta_vals]

#             intercepts = [0.0]
#             for i in range(1, M):
#                 dx = x_bounds[i] - x_bounds[i - 1]
#                 b = intercepts[i - 1] + slopes[i - 1] * dx
#                 intercepts.append(b)
#             return x_bounds, slopes, intercepts


#         ### Params contains [storage_params, inflow_params, day_of_year_params]
#         # split params in thirds

#         n_param_subset = len(self.policy_params) // 3
        
#         s_params = self.policy_params[:n_param_subset]
#         i_params = self.policy_params[n_param_subset:(2 * n_param_subset)]
#         d_params = self.policy_params[(2 * n_param_subset):]

#         # Calculate and store the segment boundaries, slopes, and intercepts
#         # for storage, inflow, and day of year functions
#         (self.storage_bounds,
#         self.storage_slopes,
#         self.storage_intercepts) = parse_segment_params(s_params)

#         (self.inflow_bounds,
#         self.inflow_slopes,
#         self.inflow_intercepts) = parse_segment_params(i_params)

#         (self.day_bounds,
#         self.day_slopes,
#         self.day_intercepts) = parse_segment_params(d_params)


#     def evaluate(self, X):
#         """
#         Evaluate the piecewise linear policy function.
        
#         Args:
#             X (list): A list of input values, including normalized:
#                 - Storage (S)
#                 - Inflow (I)
#                 - Week of year (W)
        
#         Returns:
#             float: The computed release.
#         """
#         # Separate inputs [storage, inflow, day_of_year]
#         S, I, D = X
        
#         assert I is not None, "Inflow input required but not provided."
#         assert S is not None, "Storage input required but not provided."
#         assert D is not None, "Day of year input required but not provided."

        
#         def segment_eval(x, bounds, slopes, intercepts):
#             """
#             Resolves the piecewise linear function for a given x.
            
#             Uses function:
#             f(x) = m_i * (x - x_i) + b_i
            
#             where:
#                 - m_i is the slope of the segment
#                 - x_i is the lower bound of the segment
#                 - b_i is the intercept of the segment
            
            
#             Args:
#                 x (float): The input value.
#                 bounds (list): The segment boundaries.
#                 slopes (list): The slopes of the segments.
#                 intercepts (list): The intercepts of the segments.
            
#             Returns:
#                 float: The evaluated value at x.
#             """
#             for i in range(self.n_segments):
#                 if bounds[i] <= x < bounds[i + 1]:
#                     dx = x - bounds[i]
#                     return slopes[i] * dx + intercepts[i]
#             if x >= bounds[-1]:
#                 dx = x - bounds[-2]
#                 return slopes[-1] * dx + intercepts[-1]
#             raise ValueError(f"Value {x} outside bounds: {bounds}")

#         zS = segment_eval(S, self.storage_bounds, self.storage_slopes, self.storage_intercepts)
#         zS = max(0.0, min(1.0, zS)) 
        
#         zI = segment_eval(I, self.inflow_bounds, self.inflow_slopes, self.inflow_intercepts)
#         zI = max(0.0, min(1.0, zI))

#         zD = segment_eval(D, self.day_bounds, self.day_slopes, self.day_intercepts)
#         zD = max(0.0, min(1.0, zD))

#         # Compute the final release value
#         z = (zS + zI + zD) / 3.0

#         # Impose bound limits
#         z = max(0.0, min(1.0, z)) 
#         return z



#     def get_release(self, timestep):
#         """
#         Computes the reservoir release for a given timestep based on the 
#         current storage level.

#         Args:
#             timestep (int): The current time step index.

#         Returns:
#             float: The computed release.
#         """
        
#         # Get state variables
#         I_t = self.Reservoir.inflow_array[timestep]
#         S_t = self.Reservoir.initial_storage if timestep == 0 else self.Reservoir.storage_array[timestep - 1]
#         day_of_year = self.dates[timestep].timetuple().tm_yday

#         # make sure I_t and S_t are float
#         I_t = float(I_t)
#         S_t = float(S_t)
#         day_of_year = float(day_of_year)

#         # inputs  = [storage, inflow, day_of_year]
#         X = np.array([S_t, I_t, day_of_year])

#         # Normalize X
#         X_norm = np.zeros(self.n_inputs)
#         for i in range(self.n_inputs):
#             X_norm[i] = (X[i] - self.x_min[i]) / (self.x_max[i] - self.x_min[i])        
#             X_norm[i] = max(0.0, min(1.0, X_norm[i])) # enforce bounds [0, 1]
        
#         # Compute release
#         release  = self.evaluate(X_norm) * self.Reservoir.release_max

#         # Enforce constraints (defined in AbstractPolicy)
#         release = self.enforce_constraints(release)
#         release = min(release, S_t + I_t)

#         reservoir_name = self.Reservoir.name
#         if reservoir_name in drbc_conservation_releases:
#             R_min = drbc_conservation_releases[reservoir_name]
#             release = max(release, R_min)
        
#         return release
    
    
    
#     def plot_surfaces_for_different_weeks(self, fname=None, save=False):
#         """
#         Creates a 3D plot with inflow (X), storage (Y), release (Z),
#         and multiple surfaces for different weeks of the year.

#         Args:
#             fname (str): Filename to save the plot.
#             save (bool): Flag to save the plot.
#         """
#         inflow = np.linspace(0.0, 1.0, 30)
#         storage = np.linspace(0.0, 1.0, 30)
#         weeks = np.linspace(0.0, 1.0, 5)  # Select 5 representative weeks for clarity

#         I, S = np.meshgrid(inflow, storage)

#         fig = plt.figure(figsize=(12, 9))
#         ax = fig.add_subplot(111, projection='3d')

#         cmap = plt.cm.viridis

#         for idx, week in enumerate(weeks):
#             Z = np.zeros(I.shape)
#             for i in range(I.shape[0]):
#                 for j in range(I.shape[1]):
#                     Z[i, j] = self.evaluate([S[i, j], I[i, j], week])

#             color = cmap(idx / len(weeks))
#             ax.plot_surface(I, S, Z, color=color, alpha=0.6, label=f'Week {week:.2f}')

#         ax.set_xlabel('Inflow')
#         ax.set_ylabel('Storage')
#         ax.set_zlabel('Release')
#         ax.set_title('3D Policy Output for Different Weeks')

#         # Create a custom legend
#         custom_lines = [plt.Line2D([0], [0], linestyle="none", marker='s', markersize=10,
#                                 markerfacecolor=cmap(i / len(weeks)), alpha=0.6)
#                         for i in range(len(weeks))]
#         ax.legend(custom_lines, [f'Week {w:.2f}' for w in weeks], loc='upper left')

#         if save:
#             assert fname is not None, "Filename must be provided to save the plot."
#             plt.savefig(fname)
#         plt.show()
    

#     def plot_storage_policy(self, 
#              fname="PiecewiseLinearPolicy.png",
#              save=False):
#         """
#         Plots the piecewise linear policy function.
#         """
        
#         fig, ax = plt.subplots()
        
#         xs = np.linspace(0.0, self.Reservoir.capacity, 100)

#         ys = [self.evaluate(x) for x in xs]

#         ax.plot(xs, ys)
        
#         # Add segment vertical lines
#         for x in self.segment_x_bounds:
#             ax.axvline(x=x, color='k', 
#                        linestyle='--', alpha=0.5,
#                        label='Segment Boundaries' if x == 0 else None)
        
#         # Add max release line
#         ax.axhline(y=self.Reservoir.release_max, 
#                    color='r', linestyle='--', alpha=0.5, 
#                    label='Max Release')
        
#         ax.set_xlabel("Storage")
#         ax.set_ylabel("Release")
#         ax.set_xlim(0, self.Reservoir.capacity)
#         ax.set_title("Piecewise Linear Policy")
#         ax.legend()
#         if save:
#             plt.savefig(f"./figures/{fname}")
#         plt.show()


#     def plot(self, 
#              fname=None,
#              save=False):
#         """
#         Plot the piecewise linear policy function.

#         Args:
#             fname (str): Filename for saving the plot.
#             save (bool): Whether to save the plot as a file.
#         """
#         self.plot_surfaces_for_different_weeks(fname=fname, save=save)
#         # self.plot_storage_policy(fname=fname, save=save)
        
        