# from abc import ABC, abstractmethod

# class AbstractPolicy(ABC):
#     """
#     Abstract policy class to enforce shared implementation 
#     across different parameterized policies.
#     """
#     @abstractmethod
#     def __init__(self, 
#                  Reservoir, 
#                  policy_params):
#         """
#         Initialize policy class.
        
#         Parameters
#         ----------
#         Reservoir : Reservoir
#             Reservoir object associated with the policy.
#         policy_params : dict
#             Dictionary of policy parameters.
#         """
#         pass

    
#     @abstractmethod
#     def validate_policy_params(self):
#         """
#         Validate policy parameters, ensuring:
#         - All required parameters are present
#         - Parameters are of the correct type
#         - Parameters are within valid ranges
#         """
#         pass
    
#     @abstractmethod
#     def parse_policy_params(self):
#         """
#         Parse policy parameters which will be provided 
#         as an array of values. 

#         Assign these values to the corresponding
#         attribute variables.
#         """
#         pass


#     def enforce_constraints(self, release):
#         """
#         Enforce constraints on the release.

#         Args:
#             release (float): The computed release.

#         Returns:
#             float: The release after enforcing max/min constraints.
#         """
#         return max(self.Reservoir.release_min, min(self.Reservoir.release_max, release))


#     @abstractmethod
#     def get_release(self, timestep):
#         """
#         Get the release for the current timestep,
#         based on state information from the Reservoir object.
        
#         Uses the evaluate method to compute the release,
#         then enforces constraints on the release.
#         """
#         pass
    
#     @abstractmethod
#     def plot(self):
#         """
#         Plot the policy function f(state) -> release.
#         """
#         pass