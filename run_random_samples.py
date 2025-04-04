import sys
import numpy as np
import matplotlib.pyplot as plt

from methods.sampling import generate_policy_param_samples
from methods.config import FIG_DIR


policy_type = "RBF"   # "RBF", "PiecewiseLinear"

# The parallel_borg_run.py script is designed to be run from the command line.
# It expects two arguments (sys.argv) to be the policy_type and reservoir_name.
# However, we can accomodate this by modifying sys.argv directly.
sys.argv = ["run_random_samples.py"] # filename
sys.argv.append(policy_type) # policy_type
sys.argv.append("fewalter")  # reservoir_name


from parallel_borg_run import evaluate, NOBJS, NCONSTRS


if __name__ == "__main__":

    samples = generate_policy_param_samples(policy_type=policy_type, 
                                            N = 100, 
                                            sample_type='latin')
    

    objectives = np.zeros((len(samples), NOBJS))
    
    for i, sample in enumerate(samples): 
        # Evaluate the sample
        if NCONSTRS > 0:            
            obj, const = evaluate(*sample)
        else:
            obj, = evaluate(*sample)
            
        # Store the objectives
        objectives[i, :] = obj
        
    # Print objective range
    print("Objective 1 range: ", objectives[:, 0].min(), objectives[:, 0].max())
    print("Objective 2 range: ", objectives[:, 1].min(), objectives[:, 1].max())
    
    # Make a scatter plot
    plt.scatter(objectives[:, 0], objectives[:, 1])
    plt.xlabel("Release -NSE")
    plt.ylabel("Release pBias")
    plt.title(f"{policy_type} Random Parameter Samples")
    plt.savefig(f"{FIG_DIR}/random_samples/{policy_type}_random_samples.png")
    plt.show()