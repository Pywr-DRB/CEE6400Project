import numpy as np
import matplotlib.pyplot as plt

from parallel_borg_run import evaluate

from methods.sampling import generate_policy_param_samples

if __name__ == "__main__":
    samples = generate_policy_param_samples("RBF", 100, sample_type='latin')
    objectives = np.zeros((len(samples), 2))

    for i, sample in enumerate(samples):
        
        # Evaluate the sample
        obj, = evaluate(*sample)
        
        # Store the objectives
        objectives[i, :] = obj
        
    # Print objective range
    print("Objective 1 range: ", objectives[:, 0].min(), objectives[:, 0].max())
    print("Objective 2 range: ", objectives[:, 1].min(), objectives[:, 1].max())
    
    # Make a scatter plot
    plt.scatter(objectives[:, 0], objectives[:, 1])
    plt.xlabel("-NSE")
    plt.ylabel("pBias")
    plt.title("Random Samples")
    plt.savefig("random_samples.png")
    plt.show()
