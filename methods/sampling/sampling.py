import numpy as np

from SALib.sample import latin, saltelli

from methods.config import policy_n_params, policy_param_bounds
from methods.config import policy_type_options


def generate_policy_param_samples(policy_type, 
                                  N, 
                                  sample_type='latin'):
    """
    Generates Latin Hypercube Samples (LHS) for policy parameters.
    
    Scales the samples to the defined bounds for the specified policy type.
    
    Args:
        policy_type (str): The type of policy for which to generate parameters.
        N (int): The number of samples to generate.
        sample_type (str): The type of sampling method ('latin' or 'saltelli').
    
    Returns:
        np.ndarray: An array of shape (N, n_params) containing the LHS samples.
    """

    assert policy_type in policy_type_options, \
        f"Invalid policy type: {policy_type}. Options: {policy_type_options}."
    
    assert N > 0, "Number of samples (N) must be a positive integer."
    
    assert sample_type in ['latin', 'saltelli'], \
        "Invalid sample type. Options: ['latin', 'saltelli']."
    
    
    # Define problem dictionary
    problem = {
        'num_vars': policy_n_params[policy_type],
        'names': [f'x{i}' for i in range(policy_n_params[policy_type])],
        'bounds': policy_param_bounds[policy_type],
    }

    # Generate samples
    if sample_type == 'latin':
        samples = latin.sample(problem, N=N)
        
    elif sample_type == 'saltelli':
        # N should be a multiple of 2
        N = N + 1 if (N % 2) else N
              
        samples = saltelli.sample(problem, 
                                  N=N, 
                                  calc_second_order=False)
    else:
        raise ValueError("Invalid sample type. Options: ['latin', 'saltelli'].")
        
    return samples