import numpy as np
from methods.policies import PiecewiseLinear, RBF, STARFIT


class Reservoir():
    def __init__(self, 
                 inflow, 
                 capacity,
                 policy_type,
                 policy_params,
                 release_min = None,
                 release_max = None,
                 initial_storage = None):
        """
        Reservoir model class, used to simulate the operation of a reservoir.
        
        Simulation data is stored as class attributes, to be accessed
        by the policy to inform release decisions, and to be retrieved
        after simulation for objective calculation.
        
        Parameters
        ----------
        inflow : np.array
            Array of inflow values for each timestep.
        capacity : float
            Maximum storage capacity of the reservoir.
        policy_type : str
            Type of policy to use for operation. Options are: "PiecewiseLinear", "RBF", "STARFIT"
        policy_params : np.array
            Array of policy parameters to be optimized.
        release_min : float, optional
            Minimum release constraint. Default is None.
        release_max : float, optional
            Maximum release constraint. Default is None.
        initial_storage : float, optional
            Initial storage level of the reservoir. Default is 0.8 * capacity.
            
        Methods
        -------
        run()
            Run the reservoir simulation for the given inflow and demand timeseries.
        
        get_release(t)
            Get the release for the given timestep, from the policy class.
            
        Returns
        -------
        None
        """
        
        # Input timeseries
        self.inflow_array = inflow
        self.T = len(inflow)
        
        # Reservoir characteristics
        self.capacity = capacity
        self.release_max = release_max
        self.release_min = release_min
        self.initial_storage = 0.8 * capacity if initial_storage is None else initial_storage

        # reset simulation variables
        self.reset()
        
        # initialize policy class
        self.policy = None
        self.initalize_policy(policy_type, 
                               policy_params)
    
    def reset(self):
        """
        Reset the variable arrays prior to simulation.
        """
        self.storage_array = np.zeros(self.T)
        self.release_array = np.zeros(self.T)
        self.spill_array = np.zeros(self.T)
        self.storage_array[0] = self.initial_storage
        return
    
    def initalize_policy(self, 
                          policy_type,
                          policy_params):
        """
        Initialize the policy class based on the policy type.
        
        Each policy class will have a different set of parameters,
        which will be parsed inside the unique classes. 
        
        Parameters
        ----------
        policy_type : str
            Type of policy to use for operation. Options are: "PiecewiseLinear", "RBF", "STARFIT"
        policy_options : dict
            Dictionary of optional kwargs to pass to the policy class.
        policy_params : np.array
            Array of policy parameters to be optimized.        
        """    
    
        if policy_type == 'PiecewiseLinear':
            self.policy = PiecewiseLinear(Reservoir=self, 
                                          policy_params=policy_params)
            
        elif policy_type == 'RBF':
            self.policy = RBF(Reservoir=self,
                              policy_params = policy_params)
            
        elif policy_type == 'STARFIT':
            self.policy = STARFIT(Reservoir=self,
                                  policy_params=policy_params)
        
        else:
            raise ValueError(f"Invalid policy type: {policy_type}")
        
        self.policy.validate_policy_params()
        return 
    
    
    def get_release(self, timestep):
        """
        Use the policy class to get the release for the given timestep,
        the policy class can access reservoir attributes to make decisions.
        """
        return self.policy.get_release(timestep=timestep)
    
    
    
    def run(self):
        """
        Run the reservoir simulation.
        """
        
        # reset simulation variables
        self.reset()
        
        # make sure policy is initialized
        if self.policy is None:
            raise ValueError("Policy not initialized.")
        
        # run simulation
        for t in range(1, self.T):
            # get target release from policy
            release_target = self.get_release(timestep=t)
            
            # enforce constraints on release
            release_allowable = min(release_target, self.storage_array[t-1] + self.inflow_array[t])
            self.release_array[t] = release_allowable

            # update
            self.storage_array[t] = self.storage_array[t-1] + self.inflow_array[t] - self.release_array[t]

            # Check for spill
            if self.storage_array[t] > self.capacity:
                self.spill_array[t] = self.storage_array[t] - self.capacity
                self.storage_array[t] = self.capacity
            else:
                self.spill_array[t] = 0.0
            
        return
