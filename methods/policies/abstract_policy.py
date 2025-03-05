from abc import ABC, abstractmethod

from methods.reservoir.model import Reservoir


class AbstractPolicy(ABC):
    """
    Abstract policy class to enforce shared implementation 
    across different parameterized policies.
    """
    @abstractmethod
    def __init__(self, policy_params):
        """
        Initialize policy class.
        
        Parameters
        ----------
        policy_params : dict
            Dictionary of policy parameters.
        """
        pass

    
    @abstractmethod
    def validate_policy_params(self):
        """
        Validate policy parameters, ensuring:
        - All required parameters are present
        - Parameters are of the correct type
        - Parameters are within valid ranges
        """
        pass
    
    @abstractmethod
    def parse_policy_params(self):
        """
        Parse policy parameters which will be provided 
        as an array of values. 

        Assign these values to the corresponding
        attribute variables.
        """
        pass
    
    @abstractmethod
    def get_release(self, Reservoir, timestep):
        """
        Get the release for the current timestep,
        based on state information from the Reservoir object.
        """
        pass