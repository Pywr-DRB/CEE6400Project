import os
import numpy as np
import pandas as pd
from datetime import datetime
from dataretrieval import nwis

from methods.config import ACRE_FEET_TO_MG, DATA_DIR, PROCESSED_DATA_DIR


class ObservedDataRetriever:
    def __init__(
        self,
        start_date="1945-01-01",
        end_date=None,
        out_dir=PROCESSED_DATA_DIR,
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


