import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
 
sns.set_style('white')
 
# define constants 
NFE = 30000
freq = 250
num_output = int(NFE/freq)
num_masters = 4
num_procs = 160
num_seeds = 10

algorithm = 'MM Borg'
problem = 'PolInf'
folder_name = 'metrics/'
metric_name = 'Hypervolume'  # MOEAFramework is weird where the columns start with a '#'

# plot the hypervolume over time
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.set_xlim([freq, NFE+freq])
ax.set_ylim([0, 1])
ax.set_ylabel('Hypervolume')

for s in range(num_seeds):
    hvol_matrix = np.zeros((num_masters, num_output), dtype=float)
    seed = s + 1
    for m in range(num_masters):
        # read the CSV file
        
        filename = f'{folder_name}/MMBorg_4M_PiecewiseLinear_fewalter_nfe30000_seed{seed}_{m}.metric'
        
        # check if the file exists, if not skip it
        if not os.path.exists(filename):
            print(f"File {filename} does not exist")
            continue
        else:
            runtime_bs = pd.read_csv(filename, delimiter=' ', header=0, index_col=False)
            print(runtime_bs.shape)
            hvol_matrix[m, :] = runtime_bs[metric_name].values
    
    mean_hvol = np.mean(hvol_matrix, axis=0)
    upper_hvol = np.max(hvol_matrix, axis=0)
    lower_hvol = np.min(hvol_matrix, axis=0)
    # plot the range of hypervolume values
    ax.fill_between(np.arange(freq, NFE+freq, freq), lower_hvol, upper_hvol, alpha=0.5)
    # plot the median hypervolume value
    ax.plot(np.arange(freq, NFE+freq, freq), mean_hvol, linewidth=2.0, label=f'Seed {seed}')

    ax.set_xlabel('NFE')

plt.legend(loc='lower right', fontsize=12)
        
plt.savefig(f'HVol_{problem}.png')