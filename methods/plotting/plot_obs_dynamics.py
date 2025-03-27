import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def plot_obs_reservoir_dynamics(I, S, R, reservoir_name, 
                                start_date = None,
                                end_date = None,
                                timescale = 'daily',
                                log=True,
                                fname=None,
                                title=None):
    """
    Plot a 3x1 grid of inflow, storage, and release for a given reservoir.
    
    Args:
        I (pd.DataFrame): Inflow data.
        S (pd.DataFrame): Storage data.
        R (pd.DataFrame): Release data.
        reservoir_name (str): Name of the reservoir to plot.
        start_date (str): Start date for the plot. Format: 'YYYY-MM-DD'.
        end_date (str): End date for the plot. Format: 'YYYY-MM-DD'.
        timescale (str): Timescale for the plot. Options: 'daily', 'weekly', 'monthly'.
        log (bool): Whether to use a log scale for the y-axis.
        fname (str): Filename to save the plot. If None, the plot will not be saved.
        title (str): Title of the plot.
    
    Returns:
        fig (matplotlib.figure.Figure): Figure object.
        axs (list): List of axes objects.
    """    
    
    # subset data
    I = I[[reservoir_name]]
    S = S[[reservoir_name]]
    R = R[[reservoir_name]]
    
    if start_date is not None:
        I = I.loc[start_date:]
        S = S.loc[start_date:]
        R = R.loc[start_date:]
    if end_date is not None:
        I = I.loc[:end_date]
        S = S.loc[:end_date]
        R = R.loc[:end_date]
        
    if timescale == 'monthly':
        I = I.resample('M').sum()
        S = S.resample('M').mean()
        R = R.resample('M').sum()
    elif timescale == 'daily':
        pass
    elif timescale == 'weekly':
        I = I.resample('W').sum()
        S = S.resample('W').mean()
        R = R.resample('W').sum()
    else:
        raise ValueError('timescale must be "daily", "weekly", or "monthly"')
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 10),
                            sharex=True)
    
    axs[0].plot(I.index, I[reservoir_name], label='Inflow')
    axs[0].set_ylabel('Inflow (MGD)')
    
    axs[1].plot(S.index, S[reservoir_name], label='Storage')
    axs[1].set_ylabel('Storage (MGD)')
    
    axs[2].plot(R.index, R[reservoir_name], label='Release')
    axs[2].set_ylabel('Release (MGD)')
    axs[2].set_xlabel('Date')
    
    if log:
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        axs[2].set_yscale('log')
    if title is not None:
        plt.suptitle(title)
        
    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()
    return fig, axs