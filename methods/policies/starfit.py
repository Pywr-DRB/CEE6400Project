from methods.policies.abstract_policy import AbstractPolicy
import numpy as np
import pandas as pd
from math import sin, cos, pi

class STARFIT(AbstractPolicy):
    """
    STARFIT policy class for reservoir operation.

    This policy infers release rules using STARFIT-derived operating parameters.
    """

    def __init__(self, Reservoir, policy_params):
        """
        Initializes the STARFIT policy.

        Args:
            Reservoir (Reservoir): The reservoir instance.
            policy_params (dict): Dictionary containing STARFIT parameters.
        """
        self.Reservoir = Reservoir
        self.policy_params = policy_params
        self.parse_policy_params()

    def validate_policy_params(self):
        """
        Validates the STARFIT policy parameters.
        Ensures all required parameters exist in starfit_df.
        """
        if "starfit_df" not in self.policy_params:
            raise ValueError("STARFIT policy requires 'starfit_df' in policy_params.")

        df = self.policy_params["starfit_df"]
        required_columns = [
            "reservoir", "GRanD_CAP_MG", "GRanD_MEANFLOW_MGD",
            "Release_max", "Release_min", "NORhi_mu", "NORhi_min", "NORhi_max",
            "NORhi_alpha", "NORhi_beta", "NORlo_mu", "NORlo_min", "NORlo_max",
            "NORlo_alpha", "NORlo_beta", "Release_alpha1", "Release_alpha2",
            "Release_beta1", "Release_beta2", "Release_c", "Release_p1", "Release_p2"
        ]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required STARFIT column: {col}")
    

    def parse_policy_params(self):
        """
        Parses STARFIT parameters and assigns them for calculations.
        """
        self.validate_policy_params()
        df = self.policy_params["starfit_df"]
        reservoir_name = self.policy_params["reservoir_name"]

        # Use modified STARFIT parameters for DRBC reservoirs
        modified_starfit_reservoir_list = ["blueMarsh", "fewalter", "beltzvilleCombined"]
    
        if reservoir_name in modified_starfit_reservoir_list:
            self.starfit_name = "modified_" + reservoir_name
        else:
            self.starfit_name = reservoir_name

        if self.starfit_name not in df["reservoir"].values:
            raise ValueError(f"STARFIT parameters not found for '{self.starfit_name}'. Check spelling in CSV.")


        # Debugging: Print the first few rows of STARFIT DataFrame
        print("\nSTARFIT DataFrame Preview:")
        print(df.head())

        # Ensure the reservoir exists in the dataset
        if reservoir_name not in df["reservoir"].values:
            raise ValueError(f"STARFIT parameters not found for reservoir '{reservoir_name}'. Check spelling!")

        res_data = df[df["reservoir"] == reservoir_name].iloc[0]

        # Debug: Print loaded parameters
        print(f"\nLoading STARFIT parameters for reservoir: {reservoir_name}")
        print(res_data)

        # Assign storage & inflow parameters

        # Use adjusted storage values if available
        if not pd.isna(res_data["Adjusted_CAP_MG"]) and not pd.isna(res_data["Adjusted_MEANFLOW_MGD"]):
            self.S_cap = res_data["Adjusted_CAP_MG"]
            self.I_bar = res_data["Adjusted_MEANFLOW_MGD"]
        else:
            self.S_cap = res_data["GRanD_CAP_MG"]
            self.I_bar = res_data["GRanD_MEANFLOW_MGD"]
        self.R_max = (res_data["Release_max"] + 1) * self.I_bar
        self.R_min = (res_data["Release_min"] + 1) * self.I_bar

        # Print Loaded STARFIT Parameters for Debugging
        print(f"\nLoaded STARFIT Params for {self.starfit_name}:")
        print(f"  S_cap = {self.S_cap}, I_bar = {self.I_bar}")
        print(f"  R_max = {self.R_max}, R_min = {self.R_min}")

        # Normal Operating Range (NOR) Parameters
        self.NORhi_mu, self.NORhi_min, self.NORhi_max = res_data["NORhi_mu"], res_data["NORhi_min"], res_data["NORhi_max"]
        self.NORhi_alpha, self.NORhi_beta = res_data["NORhi_alpha"], res_data["NORhi_beta"]
        self.NORlo_mu, self.NORlo_min, self.NORlo_max = res_data["NORlo_mu"], res_data["NORlo_min"], res_data["NORlo_max"]
        self.NORlo_alpha, self.NORlo_beta = res_data["NORlo_alpha"], res_data["NORlo_beta"]

        # Seasonal release parameters
        self.Release_alpha1, self.Release_alpha2 = res_data["Release_alpha1"], res_data["Release_alpha2"]
        self.Release_beta1, self.Release_beta2 = res_data["Release_beta1"], res_data["Release_beta2"]
        self.Release_c, self.Release_p1, self.Release_p2 = res_data["Release_c"], res_data["Release_p1"], res_data["Release_p2"]

        # Starting month for harmonic calculations
        self.start_month = self.policy_params.get("start_month", "Oct")

    def sinNpi(self, day, N):
        """
        Computes the sine component for seasonal harmonic adjustments.
        """
        offset = 39 if self.start_month == "Jan" else 0
        return sin(N * pi * (day + offset) / 52)

    def cosNpi(self, day, N):
        """
        Computes the cosine component for seasonal harmonic adjustments.
        """
        offset = 39 if self.start_month == "Jan" else 0
        return cos(N * pi * (day + offset) / 52)

    def release_harmonic(self, time):
        """
        Computes the harmonic seasonal adjustment for releases.
        """
        return (
            self.Release_alpha1 * self.sinNpi(time, 2)
            + self.Release_alpha2 * self.sinNpi(time, 4)
            + self.Release_beta1 * self.cosNpi(time, 2)
            + self.Release_beta2 * self.cosNpi(time, 4)
        )

    def calc_NOR_hi(self, time):
        """
        Computes the upper Normal Operating Range (NOR) bound.
        """
        NOR_hi = (
            self.NORhi_mu
            + self.NORhi_alpha * self.sinNpi(time, 2)
            + self.NORhi_beta * self.cosNpi(time, 2)
        )
        return np.clip(NOR_hi, self.NORhi_min, self.NORhi_max) / 100

    def calc_NOR_lo(self, time):
        """
        Computes the lower Normal Operating Range (NOR) bound.
        """
        NOR_lo = (
            self.NORlo_mu
            + self.NORlo_alpha * self.sinNpi(time, 2)
            + self.NORlo_beta * self.cosNpi(time, 2)
        )
        return np.clip(NOR_lo, self.NORlo_min, self.NORlo_max) / 100

    def standardize_inflow(self, I_t):
        """
        Standardizes inflow based on the annual mean inflow.
        """
        return (I_t - self.I_bar) / self.I_bar

    def percent_storage(self, S_t):
        """
        Converts storage to a fraction of total capacity.
        """
        return S_t / self.S_cap

    def get_release(self, timestep):
        """
        Computes the reservoir release for a given timestep based on STARFIT rules.
        """
        I_t = self.Reservoir.inflow_array[timestep]
        S_t = self.Reservoir.storage_array[timestep - 1] if timestep > 0 else self.Reservoir.initial_storage

        S_hat = self.percent_storage(S_t)
        target_R = min(self.R_max, max(self.R_min, S_hat * self.S_cap))

        return self.enforce_constraints(target_R)

    def plot(self, fname="STARFIT_Release_TimeSeries.png"):
        """
        Plots the STARFIT release time series over simulation time.

        Args:
            fname (str): Filename to save the plot. Default is "STARFIT_Release_TimeSeries.png".
        """
        import matplotlib.pyplot as plt
        import os
        
        # Ensure we have valid data
        if len(self.Reservoir.release_array) == 0:
            raise ValueError("Release data is empty. Ensure the simulation has been run.")

        # Time series (assuming daily)
        timesteps = np.arange(len(self.Reservoir.release_array))

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
    
        # Plot STARFIT simulated releases
        ax.plot(timesteps, self.Reservoir.release_array, label="STARFIT Release", color="r")

        # Labels & legend
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Release (MGD)")
        ax.set_title(f"STARFIT Release - {self.policy_params['reservoir_name']}")
        ax.legend()
        ax.grid()

        # Save the figure
        save_path = os.path.join("figures", fname)
        os.makedirs("figures", exist_ok=True)  # Ensure directory exists
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")

        plt.show()