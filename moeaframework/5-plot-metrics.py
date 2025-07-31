import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIGURATION ===
sns.set_style('white')
metrics = [
    'MaximumParetoFrontError',
    '#Hypervolume',
    'GenerationalDistance',
    'InvertedGenerationalDistance',
    'Spacing',
    'EpsilonIndicator'
]
policies = ['STARFIT', 'RBF', 'PiecewiseLinear']
reservoirs = ['beltzvilleCombined', 'fewalter', 'prompton']
nfe = 30000
freq = 250
num_output = int(nfe / freq)
num_masters = 4
seeds = range(1, 11)

colors_fill = ['lightcoral', 'lightseagreen', 'lightsteelblue', 'lightpink', 'lightgreen'] * 2
colors_line = ['crimson', 'teal', 'royalblue', 'deeppink', 'forestgreen'] * 2

# === MAIN LOOP ===
for policy in policies:
    for metric_name in metrics:
        for reservoir in reservoirs:
            folder = f'outputs/Policy_{policy}/runtime/{reservoir}'
            if not os.path.isdir(folder):
                print(f"Skipping missing folder: {folder}")
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlim([freq, nfe + freq])
            ax.set_ylim([0, 1])  # Adjust per metric if needed
            ax.set_ylabel(metric_name)
            ax.set_xlabel('Function Evaluations')

            x_vals = np.arange(freq, nfe + freq, freq)

            for s_idx, seed in enumerate(seeds):
                metric_matrix = np.zeros((num_masters, num_output), dtype=float)

                for m in range(num_masters):
                    filename = f'{folder}/MMBorg_4M_{policy}_{reservoir}_nfe{nfe}_seed{seed}_{m}.metrics'
                    if not os.path.exists(filename):
                        print(f"Missing: {filename}")
                        continue

                    df = pd.read_csv(filename, sep=r'\s+', header=0)
                    df.columns = df.columns.str.strip()  # Clean column names

                    if metric_name not in df.columns:
                        print(f"{metric_name} not found in {filename}")
                        continue

                    metric_matrix[m, :] = df[metric_name].values

                if np.all(metric_matrix == 0):
                    continue

                mean_vals = np.mean(metric_matrix, axis=0)
                upper_vals = np.max(metric_matrix, axis=0)
                lower_vals = np.min(metric_matrix, axis=0)

                ax.fill_between(x_vals, lower_vals, upper_vals, alpha=0.4, color=colors_fill[s_idx])
                ax.plot(x_vals, mean_vals, label=f'Seed {seed}', linewidth=2, color=colors_line[s_idx])

            plt.legend(loc='lower right', fontsize=10, ncol=2)
            plt.title(f"{metric_name} over NFE\n{policy} – {reservoir}")
            plt.tight_layout()

            fname = f"{metric_name}_{policy}_{reservoir}.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"Plot saved as {fname}")

print("All plots generated successfully.")
