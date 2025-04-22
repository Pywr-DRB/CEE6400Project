import pandas as pd
import numpy as np
import os
from datetime import datetime
from dataretrieval import nwis
import matplotlib.pyplot as plt

ACRE_FEET_TO_MG = 0.325851  # Acre-feet to million gallons


class ObservedDataRetriever:
    def __init__(
        self,
        start_date="1945-01-01",
        end_date=None,
        out_dir="processed_data",
        default_stat_code="00003",  # mean
    ):
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        self.out_dir = out_dir
        self.default_stat_code = default_stat_code
        os.makedirs(self.out_dir, exist_ok=True)

    def get(self, gauges, param_cd="00060", stat_cd=None, label_map=None):
        stat_cd = stat_cd or self.default_stat_code
        all_dfs = []

        for g in gauges:
            try:
                data = nwis.get_dv(
                    sites=g,
                    parameterCd=param_cd,
                    statCd=stat_cd,
                    start=self.start_date,
                    end=self.end_date,
                )[0]
                data.reset_index(inplace=True)
                data["datetime"] = pd.to_datetime(data["datetime"])
                mean_col = f"{param_cd}_Mean"

                if mean_col not in data.columns:
                    raise ValueError(f"Expected column '{mean_col}' not found for site {g}")

                col_name = label_map.get(g, g) if label_map else g
                data.set_index("datetime", inplace=True)
                renamed = data[[mean_col]].rename(columns={mean_col: col_name})
                all_dfs.append(renamed)

            except Exception as e:
                print(f"Failed to retrieve {g}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        df_combined = pd.concat(all_dfs, axis=1)
        print(f"Retrieved data for: {df_combined.columns.tolist()}")
        return df_combined

    def convert_elevation_to_storage(self, elevation_df, storage_curve_dict):
        storage_dfs = []

        for col in elevation_df.columns:
            try:
                curve_file = storage_curve_dict.get(col)
                if not curve_file or not os.path.exists(curve_file):
                    raise FileNotFoundError(f"Missing storage curve: {curve_file}")

                curve = pd.read_csv(curve_file).set_index("Elevation (ft)")
                storage = elevation_df[col].apply(
                    lambda elev: np.interp(elev, curve.index, curve["Acre-Ft"]) * ACRE_FEET_TO_MG
                    if pd.notnull(elev) and isinstance(elev, (int, float, np.floating)) else np.nan
                )
                storage.name = col
                storage_dfs.append(storage)
            except Exception as e:
                print(f"Failed to convert {col}: {e}")
                storage_dfs.append(pd.Series(name=col))

        return pd.concat(storage_dfs, axis=1)
    
    def postprocess_and_save(self, df, reservoir_to_gauges, outfile_path):
        result_df = pd.DataFrame(index=df.index)

        for res, gauges in reservoir_to_gauges.items():
            valid_gauges = [g for g in gauges if g in df.columns]
            if valid_gauges:
                result_df[res] = df[valid_gauges].sum(axis=1)
            else:
                result_df[res] = np.nan
                print(f"No valid gauges found for {res}")

        result_df.index.name = "datetime"
        
        ### Set zeros to NaN
        result_df = result_df.where(result_df > 0, other=float('nan'))
        
        ### Apply forward fill to fill nans of less than 3 days
        result_df = result_df.ffill(limit=3)        
        
        result_df.to_csv(outfile_path)
        print(f"Saved aggregated data to: {outfile_path}")

    def save_to_csv(self, df, name):
        filename = os.path.join(self.out_dir, f"{name}.csv")
        df.index = pd.to_datetime(df.index).date  # strip time and tz
        df.index.name = "datetime"
        df.to_csv(filename)
        print(f"Saved: {filename}")

    def find_missing_dates(self, df):
        full_range = pd.date_range(self.start_date, self.end_date, freq="D")
        missing_dates = full_range.difference(df.index)
        return missing_dates

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
