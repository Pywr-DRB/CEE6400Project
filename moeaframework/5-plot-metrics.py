#!/usr/bin/env python3
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================== CONFIG ====================
POLICIES   = ['STARFIT', 'RBF', 'PWL']
RESERVOIRS = ['beltzvilleCombined', 'fewalter', 'prompton', 'blueMarsh']
METRICS    = [
    'MaximumParetoFrontError',
    'Hypervolume',                 # accepts both "Hypervolume" and "#Hypervolume"
    'GenerationalDistance',
    'InvertedGenerationalDistance',
    'Spacing',
    'EpsilonIndicator'
]

NFE        = 30000
FREQ       = 250
SEEDS      = list(range(1, 11))      # 1..10
MASTERS    = [0, 1, 2, 3]            # limit to these “islands/masters”

OUT_ROOT   = 'outputs'
FIG_DIR    = 'figures_moea_metrics'   # single folder at repo root for ALL figs
SAVE_NESTED = False                   # True -> group into subfolders by policy/reservoir

# Appearance
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
ALPHA_FILL = 0.28

# ==================== DERIVED ====================
NUM_OUT = NFE // FREQ
X = np.arange(FREQ, NFE + FREQ, FREQ)   # [250, 500, ..., NFE]
assert len(X) == NUM_OUT

# Expected filename pattern (your current layout)
# MMBorg_4M_<POLICY>_<RESERVOIR>_nfe<NFE>_seed<SEED>_<MASTER>.metric
FNAME_FMT = "MMBorg_4M_{policy}_{reservoir}_nfe{nfe}_seed{seed}_{master}.metric"

def metrics_dir(policy: str, reservoir: str) -> str:
    return os.path.join(OUT_ROOT, f"Policy_{policy}", "metrics", reservoir)

def find_files_for_seed(policy: str, reservoir: str, seed: int) -> list[str]:
    """Return existing metric file paths for (policy, reservoir, seed) limited to MASTERS."""
    mdir = metrics_dir(policy, reservoir)
    if not os.path.isdir(mdir):
        return []
    paths = []
    for m in MASTERS:
        fname = FNAME_FMT.format(policy=policy, reservoir=reservoir, nfe=NFE, seed=seed, master=m)
        p = os.path.join(mdir, fname)
        if os.path.exists(p):
            paths.append(p)
    return paths

def read_metric_series(path: str, metric: str) -> np.ndarray:
    """Read metric column, accept Hypervolume or #Hypervolume; pad/truncate to NUM_OUT."""
    df = pd.read_csv(path, sep=r'\s+', engine='python')
    df.columns = [c.strip() for c in df.columns]

    use_col = metric
    if metric == 'Hypervolume' and 'Hypervolume' not in df.columns and '#Hypervolume' in df.columns:
        use_col = '#Hypervolume'

    if use_col not in df.columns:
        # Quietly skip unusable file for this metric
        raise KeyError(f"{os.path.basename(path)} lacks '{metric}' (has {df.columns.tolist()})")

    vals = df[use_col].to_numpy(dtype=float)
    if len(vals) >= NUM_OUT:
        return vals[:NUM_OUT]
    out = np.full(NUM_OUT, np.nan, dtype=float)
    out[:len(vals)] = vals
    return out

def y_limits_for(metric: str, previews: list[np.ndarray]):
    """Optional metric-specific y-limits; return None to auto-scale."""
    name = metric.lower()
    if 'hypervolume' in name:
        return (0.0, 1.0)
    if 'generationaldistance' in name or 'invertedgenerationaldistance' in name or 'spacing' in name:
        data = np.hstack([a[np.isfinite(a)] for a in previews if a is not None]) if previews else np.array([])
        if data.size:
            lo = max(0.0, np.nanpercentile(data, 1))
            hi = np.nanpercentile(data, 99) * 1.05
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return (lo, hi)
    return None

def out_path_for(metric: str, policy: str, reservoir: str) -> str:
    safe_metric = metric.replace('#', '')  # sanitize
    if SAVE_NESTED:
        nested = os.path.join(FIG_DIR, f"Policy_{policy}", reservoir)
        os.makedirs(nested, exist_ok=True)
        return os.path.join(nested, f"{safe_metric}_{policy}_{reservoir}.png")
    else:
        os.makedirs(FIG_DIR, exist_ok=True)
        return os.path.join(FIG_DIR, f"{safe_metric}_{policy}_{reservoir}.png")

def main():
    print(f"[info] scanning outputs in: {OUT_ROOT}")
    os.makedirs(FIG_DIR, exist_ok=True)

    for policy in POLICIES:
        for reservoir in RESERVOIRS:
            mdir = metrics_dir(policy, reservoir)
            if not os.path.isdir(mdir):
                print(f"[skip] missing folder: {mdir}")
                continue

            # ---------- Preview pass for y-limits ----------
            previews = {metric: [] for metric in METRICS}
            for seed in SEEDS:
                files = find_files_for_seed(policy, reservoir, seed)
                if not files:
                    continue
                for metric in METRICS:
                    series = []
                    for f in files:
                        try:
                            series.append(read_metric_series(f, metric))
                        except Exception:
                            pass
                    if series:
                        with np.errstate(all='ignore'):
                            previews[metric].append(np.nanmean(np.vstack(series), axis=0))

            # ---------- Plot pass ----------
            for metric in METRICS:
                # If literally nothing available for this metric at this (policy,reservoir), skip
                if not previews[metric]:
                    print(f"[skip] nothing to plot for {policy} / {reservoir} / {metric}")
                    continue

                fig, ax = plt.subplots()

                # Optional y-limit tuning by metric
                ylims = y_limits_for(metric, previews[metric])
                if ylims:
                    ax.set_ylim(*ylims)

                any_plotted = False
                for seed in SEEDS:
                    files = find_files_for_seed(policy, reservoir, seed)
                    if not files:
                        continue
                    series = []
                    for f in files:
                        try:
                            series.append(read_metric_series(f, metric))
                        except Exception:
                            continue
                    if not series:
                        continue

                    mat = np.vstack(series)  # (#masters, NUM_OUT)
                    valid = ~np.all(np.isnan(mat), axis=0)
                    if not valid.any():
                        continue

                    mean_vals  = np.nanmean(mat[:, valid], axis=0)
                    lower_vals = np.nanmin(mat[:, valid], axis=0)
                    upper_vals = np.nanmax(mat[:, valid], axis=0)

                    ax.fill_between(X[valid], lower_vals, upper_vals, alpha=ALPHA_FILL)
                    ax.plot(X[valid], mean_vals, linewidth=2, label=f"Seed {seed}")
                    any_plotted = True

                if not any_plotted:
                    plt.close(fig)
                    print(f"[skip] nothing to plot after scan for {policy} / {reservoir} / {metric}")
                    continue

                ax.set_xlim([FREQ, NFE + FREQ])
                ax.set_xlabel('Function Evaluations')
                ax.set_ylabel(metric)
                ax.set_title(f"{metric} over NFE\n{policy} – {reservoir}")
                ax.legend(loc='lower right', fontsize=9, ncol=2)
                fig.tight_layout()

                out_png = out_path_for(metric, policy, reservoir)
                fig.savefig(out_png, dpi=300)
                plt.close(fig)
                print(f"[ok] {out_png}")

    print("[done] all plots generated.")

if __name__ == "__main__":
    main()
