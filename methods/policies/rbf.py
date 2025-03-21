from methods.policies.abstract_policy import AbstractPolicy

class RBF(AbstractPolicy):
    def __init__(self,
                 Reservoir,
                 policy_params):
        
        self.Reservoir = Reservoir
        
        self.nRBFs = policy_params // 3
        self.policy_params = policy_params
        

    def validate_policy_params(self):
        assert len(self.policy_params) == 3 * self.nRBFs, "Incorrect number of policy parameters."
        return
    
    def parse_policy_params(self):
 
        self.c = self.policy_params[0::3]
        self.r = self.policy_params[1::3]
        w = self.policy_params[2::3]
        
        # Normalize the weights
        w_norm = []
        if np.sum(w) != 0:
            for w_i in w:
                w_norm.append(w_i / np.sum(w))
        else:
            w_norm = (1/nRBFs)*np.ones(len(w))
        
        self.w = w_norm
        
        return 

    def evaluate(self, X):
        """
        Evaluate the policy function.

        Args:
            X (np.array): Policy input data.

        Returns:
            float: The computed release.
        """
        
        z = 0.0
        
        for i in range(nRBFs):
            # Avoid division by zero
            if r[i] != 0:
                z = z + self.w[i] * np.exp(-((x - self.c[i])/self.r[i])**2)
            else:
                z = z + self.w[i] * np.exp(-((x - self.c[i])/(10**-6))**2)
                
        # Impose bound limits
        z = max(0, min(1, z)) 
                       
        return z * self.Reservoir.max_release
    
    def get_release(self, timestep):
        
        # Get X data
        X = np.array([])
        
        # Normalize X
        X_norm = (X - X.min()) / (X.max() - X.min())
        
        # Compute release
        release  = self.evaluate(X_norm)
        
        # Enforce constraints (defined in AbstractPolicy)
        release = self.enforce_constraints(release)
        
        return release

    def plot(self):
        pass