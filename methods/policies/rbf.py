from methods.policies.abstract_policy import AbstractPolicy

class RBF(AbstractPolicy):
    def __init__(self,
                 Reservoir,
                 policy_params):
        
        pass
    
    def validate_policy_params(self):
        pass
    
    def parse_policy_params(self):
        pass

    def evaluate(self, X):
        """
        Evaluate the policy function.

        Args:
            X (np.array): Policy input data.

        Returns:
            float: The computed release.
        """
        pass
    
    def get_release(self, timestep):
        
        # Get X data
        X = np.array([])
        
        # Compute release
        release  = self.evaluate(X)
        
        # Enforce constraints (defined in AbstractPolicy)
        release = self.enforce_constraints(release)
        
        return release

    def plot(self):
        pass