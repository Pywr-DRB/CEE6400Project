import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_obs_reservoir_dynamics(I, S, R, reservoir_name, 
                                start_date=None,
                                end_date=None,
                                timescale='daily',
                                log=True,
                                title=None,
                                save=True, 
                                save_dir="figures"):
    
    # Ensure output folder exists if saving
    if save:
        os.makedirs(save_dir, exist_ok=True)

    I = I[[reservoir_name]] if reservoir_name in I.columns else pd.DataFrame(index=S.index)
    S = S[[reservoir_name]]
    R = R[[reservoir_name]] if reservoir_name in R.columns else pd.DataFrame(index=S.index)

    # Subset by date
    for df in [I, S, R]:
        if start_date:
            df.drop(df[df.index < start_date].index, inplace=True)
        if end_date:
            df.drop(df[df.index > end_date].index, inplace=True)

    # Resample
    if timescale == 'monthly':
        I = I.resample('ME').sum()
        S = S.resample('ME').mean()
        R = R.resample('ME').sum()
    elif timescale == 'weekly':
        I = I.resample('WE').sum()
        S = S.resample('WE').mean()
        R = R.resample('WE').sum()

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Inflow (hardcoded exception for Blue Marsh)
    if reservoir_name.lower() == "bluemarsh":
        axs[0].text(0.5, 0.5, "No inflow data for Blue Marsh", 
                    ha='center', va='center', transform=axs[0].transAxes)
        axs[0].set_ylim(0, 1)
    else:
        axs[0].plot(I.index, I[reservoir_name], label='Inflow')
    axs[0].set_ylabel('Inflow (MGD)')

    # Storage
    axs[1].plot(S.index, S[reservoir_name], label='Storage')
    axs[1].set_ylabel('Storage (MGD)')

    # Release
    axs[2].plot(R.index, R[reservoir_name], label='Release')
    axs[2].set_ylabel('Release (MGD)')
    axs[2].set_xlabel('Date')

    # Log scale (skip inflow panel for Blue Marsh)
    if log:
        if reservoir_name.lower() != "bluemarsh":
            axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        axs[2].set_yscale('log')

    plt.suptitle(title or f"{reservoir_name} Reservoir Observed Data")
    plt.tight_layout()

    if save:
        sdate = start_date or I.index.min().strftime("%Y-%m-%d")
        edate = end_date or I.index.max().strftime("%Y-%m-%d")
        safe_name = reservoir_name.replace(" ", "_").lower()
        filename = f"{safe_name}_{sdate}_to_{edate}_{timescale}.png"
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path, dpi=300)
        print(f"Saved plot to: {full_path}")
        
    plt.show()