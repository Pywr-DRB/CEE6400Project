from methods.policies.abstract_policy import AbstractPolicy

class STARFIT(AbstractPolicy):
    def __init__(self, 
                 Reservoir, 
                 policy_params):
        pass
    
    def validate_policy_params(self):
        pass
    
    def get_release(self, timestep):
        # access inflow, demand, storage, and capacity from Reservoir object
        I_t = self.Reservoir.inflow_array[timestep]
        S_t = self.Reservoir.storage_array[timestep]

        pass