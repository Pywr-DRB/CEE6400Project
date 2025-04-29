import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.linspace(0, 1, len(sorted_data))
    return sorted_data, cdf


def plot_annual_storage_distribution(df, label, 
                                     color='black', ax=None, 
                                     quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    """
    Plot shaded annual quantiles of storage by day-of-year.
    
    Parameters:
        df (DataFrame): Must contain columns 'datetime', 'year', 'doy', and a storage column
        label (str): Legend label
        color (str): Base color for shading
        ax (matplotlib.axes.Axes): Axes to plot on
        quantiles (list): List of quantiles to compute
    """
    if ax is None:
        ax = plt.gca()
    
    # Infer which storage column is present
    storage_col = [col for col in df.columns if 'storage' in col.lower()][0]
    
    # Group by DOY, compute quantiles
    grouped = df.groupby('doy')[storage_col].quantile(quantiles).unstack()
    
    # Plot quantile fills
    for q_lo, q_hi in zip(quantiles[:2], quantiles[-1:-3:-1]):
        ax.fill_between(grouped.index,
                        grouped[q_lo],
                        grouped[q_hi],
                        color=color,
                        alpha=q_lo,
                        label = label + f" {int(q_lo*100)}% - {int(q_hi*100)}%",
                        edgecolor='none')
    
    # Plot median
    ax.plot(grouped.index, grouped[0.5], color=color, lw=1.5, label=label + " Median")
    
    return ax


def plot_annual_storage_timeseries(df, label=None, color='black', ax=None):
    if ax is None:
        ax = plt.gca()
        
    storage_col = [col for col in df.columns if 'storage' in col.lower()][0]
    
    for y in sorted(df['year'].unique()):
        subset = df[df['year'] == y]
        ax.plot(subset['doy'], subset[storage_col], 
                color=color, alpha=0.3, 
                label=label if y == df['year'].min() else "")
    
    return ax

# def plot_annual_inflow_release_distribution(df, 
#                                             label=None, color='black',
#                                             quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], 
#                                             ax=None):
#     """
#     Plot inflow vs release curve for a single dataset.
    
#     Parameters:
#         inflow (array-like): Inflow values
#         release (array-like): Corresponding release values
#         label (str): Legend label
#         color (str): Line color
#         ax (matplotlib.axes.Axes): Axis to plot on
#     """
#     if ax is None:
#         ax = plt.gca()
    
    
#     # Sort df based on obs_inflow
#     sorted_idx = np.argsort(df['obs_inflow'])
#     df = df.iloc[sorted_idx]
    
#     # Infer which storage column is present
#     inflow_col = 'obs_inflow'
#     release_col = [col for col in df.columns if 'release' in col.lower()][0]
    
#     # Group by DOY, compute quantiles
#     grouped = df.groupby('doy')[release_col].quantile(quantiles).unstack()
    
    
#     # Plot quantile fills
#     for q_lo, q_hi in zip(quantiles[:-1], quantiles[-1::-1]):
        
#     # Plot median
#     release_median = ...
#     ax.plot(...)
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     return ax

def plot_annual_inflow_release_distribution(df, 
                                            label=None, 
                                            color='black',
                                            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], 
                                            ax=None):
    """
    Plot quantile bands of inflow vs release curves, aggregated across years.
    
    Parameters:
        df (DataFrame): Must contain 'obs_inflow', 'release', 'year'
        label (str): Legend label
        color (str): Base color for quantile fills and median line
        quantiles (list): Quantile levels to compute
        ax (matplotlib.axes.Axes): Matplotlib axis object
    """
    if ax is None:
        ax = plt.gca()

    # Infer release column
    inflow_col = 'obs_inflow'
    release_col = [col for col in df.columns if 'release' in col.lower()][0]

    # Collect inflow-release curves for each year
    inflow_vals = []
    release_vals = []
    for year, group in df.groupby('year'):
        inflow = np.array(group[inflow_col])
        release = np.array(group[release_col])
        sorted_idx = np.argsort(inflow)
        inflow_sorted = inflow[sorted_idx]
        release_sorted = release[sorted_idx]
        inflow_vals.append(inflow_sorted)
        release_vals.append(release_sorted)

    # Stack all years into a 2D array (interpolation ensures aligned inflows)
    inflow_common = np.linspace(np.percentile(np.concatenate(inflow_vals), 1),
                                np.percentile(np.concatenate(inflow_vals), 99),
                                200)

    release_interp = np.array([
        np.interp(inflow_common, inflow_vals[i], release_vals[i])
        for i in range(len(inflow_vals))
    ])

    # Compute quantiles at each inflow level
    release_q = np.quantile(release_interp, q=quantiles, axis=0)

    # Plot quantile bands
    for q_lo, q_hi in zip(quantiles[:2], quantiles[-1:-3:-1]):
        lo = release_q[quantiles.index(q_lo)]
        hi = release_q[quantiles.index(q_hi)]
        ax.fill_between(inflow_common, lo, hi, 
                        color=color, 
                        alpha=q_lo, 
                        edgecolor='none',
                        label=f"{label} {int(q_lo*100)}–{int(q_hi*100)}%")

    # Plot median
    ax.plot(inflow_common, release_q[quantiles.index(0.5)], 
            color=color, lw=1.5, label=f"{label} Median")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Observed Inflow")
    ax.set_ylabel("Release")
    ax.set_title("Annual Inflow–Release Distribution")
    ax.legend()

    return ax



def plot_fdc(data, 
             label=None, 
             color='black', 
             ax=None, 
             logscale=True):
    """
    Plot the empirical CDF of the given data.
    
    Parameters:
        data (array-like): Data to compute the CDF for
        label (str): Label for the plot
        color (str): Color of the plot line
        ax (matplotlib.axes.Axes): Axes to plot on
    """
    sorted_data, cdf = compute_cdf(data)
    if ax is None:
        ax = plt.gca()
    ax.plot(cdf*100, sorted_data, label=label, color=color)
    
    if logscale:
        ax.set_yscale('log')
    return ax

def plot_storage_release_distributions(obs_storage, obs_release, 
                                       sim_storage, sim_release,
                                       obs_inflow,
                                       datetime,
                                       storage_distribution=True,
                                       inflow_vs_release=True,
                                       fname=None):

    """
    Plot observed vs simulated reservoir storage and release dynamics.
    
    Parameters:
        obs_storage (array-like): Observed storage volume
        obs_release (array-like): Observed release volume
        sim_storage (array-like): Simulated storage volume
        sim_release (array-like): Simulated release volume
        datetime (array-like): Corresponding datetime values (must be same length as others)
    """
    # Convert datetime to pandas Series
    dt = pd.to_datetime(datetime)
    df = pd.DataFrame({
        'datetime': dt,
        'year': dt.year,
        'doy': dt.dayofyear,
        'obs_storage': obs_storage,
        'sim_storage': sim_storage,
        'obs_release': obs_release,
        'sim_release': sim_release,
        'obs_inflow': obs_inflow
    })

    # Create figure with 2 vertical subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, 
                                   figsize=(5, 8), 
                                   sharex=False, constrained_layout=True)

    # Top Plot: Storage dynamics by year
    
    if storage_distribution:
        ax1 = plot_annual_storage_distribution(df[['datetime', 'year', 'doy', 'obs_storage']].copy(),
                                               label='Observed', color='black', ax=ax1)
        ax1 = plot_annual_storage_distribution(df[['datetime', 'year', 'doy', 'sim_storage']].copy(),
                                               label='Simulated', color='orange', ax=ax1)
    else:
        ax1 = plot_annual_storage_timeseries(df[['datetime', 'year', 'doy', 'obs_storage']].copy(),
                                             label='Observed', color='black', ax=ax1)
        ax1 = plot_annual_storage_timeseries(df[['datetime', 'year', 'doy', 'sim_storage']].copy(),
                                             label='Simulated', color='orange', ax=ax1)

    
    ax1.set_ylabel("Storage (MG)")
    ax1.set_xlabel("Day of Year")
    # ax1.legend()

    # Bottom Plot: CDF of releases
    if inflow_vs_release:
        ax2 = plot_annual_inflow_release_distribution(df[['datetime', 'year', 'doy', 'obs_inflow', 'obs_release']].copy(),
                                                 label='Observed', color='black', ax=ax2)
        ax2 = plot_annual_inflow_release_distribution(df[['datetime', 'year', 'doy', 'obs_inflow', 'sim_release']].copy(),
                                                    label='Simulated', color='orange', ax=ax2)
        
        ax2.set_ylabel("Release (MGD)")
        ax2.set_xlabel("Inflow (MGD)")
        
    else:
        ax2 = plot_fdc(obs_release, label='Observed', 
                    color='black', ax=ax2)
        ax2 = plot_fdc(sim_release, label='Simulated', 
                    color='orange', ax=ax2)
        
        ax2.set_ylabel("Release (MGD)")
        ax2.set_xlabel("Non-exceedance Probability (%)")
        
    # Get legend handles and labels
    handles, labels = ax2.get_legend_handles_labels()
    # Set legend below the plot
    ax2.legend(handles, labels,
               ncol=2, 
               loc='upper center', 
               bbox_to_anchor=(0.5, -0.2), 
               borderaxespad=0., 
               fontsize='small')
    
    if fname is not None:
        plt.savefig(fname, dpi=250)
        
    return fig, (ax1, ax2)
