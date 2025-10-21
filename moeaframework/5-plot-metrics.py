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
policies = ['STARFIT', 'RBF', 'PWL']
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


#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
metrics = [
    'MaximumParetoFrontError',
    'Hypervolume',                 # alias below also accepts '#Hypervolume'
    'GenerationalDistance',
    'InvertedGenerationalDistance',
    'Spacing',
    'EpsilonIndicator'
]
policies   = ['STARFIT', 'RBF', 'PWL']
reservoirs = ['beltzvilleCombined', 'fewalter', 'prompton']
nfe  = 30000
freq = 250
seeds = range(1, 11)              # 1..10

# Derived
num_output = int(nfe // freq)
x_vals = np.arange(freq, nfe + freq, freq)

# Regex to extract seed and master from filenames if present
# e.g., "..._seed7_2.metric" -> seed=7, master=2
RX = re.compile(r'_seed(\d+)_([0-9]+)\.metric$')

def read_metric_file(path, metric_name):
    """Read a metric file and return a numpy array (length num_output) for metric_name."""
    # whitespace-delimited, first line has already had leading '#' stripped by the workflow
    df = pd.read_csv(path, sep=r'\s+', engine='python')
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # Accept both 'Hypervolume' and '#Hypervolume'
    if metric_name == 'Hypervolume' and '#Hypervolume' in cols and 'Hypervolume' not in cols:
        use_col = '#Hypervolume'
    else:
        use_col = metric_name

    if use_col not in df.columns:
        raise KeyError(f"Column '{metric_name}' not found in {os.path.basename(path)} (available: {df.columns.tolist()})")

    # If file is longer/shorter than expected, clamp or pad to num_output
    values = df[use_col].to_numpy(dtype=float)
    if len(values) >= num_output:
        return values[:num_output]
    else:
        out = np.full((num_output,), np.nan, dtype=float)
        out[:len(values)] = values
        return out

for policy in policies:
    for metric_name in metrics:
        for reservoir in reservoirs:
            metrics_dir = f'outputs/Policy_{policy}/metrics/{reservoir}'
            if not os.path.isdir(metrics_dir):
                print(f"[skip] Missing folder: {metrics_dir}")
                continue

            # Gather all files; we’ll group by seed and master index (if in filename)
            files = glob.glob(os.path.join(metrics_dir, '*.metric'))
            if not files:
                print(f"[skip] No metric files in {metrics_dir}")
                continue

            # Organize: seed -> list of masters’ arrays
            seed_to_series = {s: [] for s in seeds}

            # Try to use filename pattern; if seed not in name, we’ll ignore it
            for path in files:
                m = RX.search(os.path.basename(path))
                if not m:
                    # If your filenames don’t encode seed/master, you can change this:
                    # fallback: try to find seed=X in name:
                    m2 = re.search(r'seed(\d+)', os.path.basename(path))
                    if not m2:
                        # Unknown seed; skip
                        continue
                    seed = int(m2.group(1))
                    master_idx = None
                else:
                    seed = int(m.group(1))
                    master_idx = int(m.group(2))

                if seed not in seed_to_series:
                    continue  # outside configured range

                try:
                    arr = read_metric_file(path, metric_name)
                except Exception as e:
                    print(f"[warn] {e}")
                    continue

                seed_to_series[seed].append(arr)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlim([freq, nfe + freq])
            # Light auto y-scaling; special-case hypervolume to [0,1]
            if metric_name.lower().endswith('hypervolume'):
                ax.set_ylim([0, 1])

            ax.set_ylabel(metric_name)
            ax.set_xlabel('Function Evaluations')

            any_plotted = False
            for seed, series_list in seed_to_series.items():
                if not series_list:
                    continue
                mat = np.vstack(series_list)  # shape: (#masters, num_output)
                # Drop columns that are all NaN (in case of padding)
                valid = ~np.all(np.isnan(mat), axis=0)
                if not valid.any():
                    continue
                mean_vals  = np.nanmean(mat[:, valid], axis=0)
                lower_vals = np.nanmin(mat[:, valid], axis=0)
                upper_vals = np.nanmax(mat[:, valid], axis=0)

                ax.fill_between(x_vals[valid], lower_vals, upper_vals, alpha=0.25)
                ax.plot(x_vals[valid], mean_vals, linewidth=2, label=f'Seed {seed}')
                any_plotted = True

            if not any_plotted:
                plt.close(fig)
                print(f"[skip] Nothing to plot for {policy} / {reservoir} / {metric_name}")
                continue

            ax.legend(loc='lower right', fontsize=10, ncol=2)
            ax.set_title(f"{metric_name} over NFE\n{policy} – {reservoir}")
            fig.tight_layout()

            out_name = f"{metric_name}_{policy}_{reservoir}.png".replace('#','')  # sanitize '#'
            fig.savefig(out_name, dpi=300)
            plt.close(fig)
            print(f"[ok] Saved {out_name}")

print("All plots generated.")
