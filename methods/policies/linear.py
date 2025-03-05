from methods.policies.abstract_policy import AbstractPolicy
import numpy as np

class PiecewiseLinear(AbstractPolicy):
    def __init__(self, 
                 Reservoir, 
                 policy_params):
        
        """
        A piecewise linear policy class.
        
        The policy consists of M linear segments, 
        each defined by:
        - theta : the angle of the line above the x-axis
        - the x-value lower bound of the linear region
        
        Since the x-value lower bound of the first linear region is x=0, 
        that leaves 5 parameters to be defined. 
        """
        
        self.Reservoir = Reservoir
        self.policy_params = policy_params
        
        pass
    
    def parse_policy_params(self):
        self.x_0 = 0.0
        self.x_1 = self.policy_params[0]
        self.x_2 = self.policy_params[1]
        self.theta_1 = self.policy_params[0]
        self.theta_2 = self.policy_params[1]
        self.theta_3 = self.policy_params[2]
        
        # calculate the slope and intercept from theta
        self.m_1 = np.tan(self.theta_1)
        self.m_2 = np.tan(self.theta_2)
        self.m_3 = np.tan(self.theta_3)
        self.b_1 = 0
        self.b_2 = self.m_1 * self.x_1
        self.b_3 = self.b_2 + self.m_2 * (self.x_2 - self.x_1)
        
        # Get max release for each segment
        self.max_release_1 = self.m_1 * self.x_1
        self.max_release_2 = self.max_release_1 + self.m_2 * (self.x_2 - self.x_1)
        self.max_release_3 = self.max_release_2 + self.m_3 * (self.x_3 - self.x_2)
        return
    
    
    
    def validate_policy_params(self):
        pass
    

    
    def get_release(self, timestep):
        
        I_t = self.Reservoir.inflow_array[timestep]
        S_t = self.Reservoir.storage_array[timestep]
    
        # match case based on current regime
        if S_t < self.x_1:
            release = self.m_1 * S_t
        elif self.x_2 <= S_t < self.x_2:
            release = self.max_release_1 + (self.m_2 * S_t + self.b_2)
        
        pass
    