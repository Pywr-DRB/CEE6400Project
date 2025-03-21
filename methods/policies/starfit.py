from methods.policies.abstract_policy import AbstractPolicy
import numpy as np
import pandas as pd
from math import sin, cos, pi

class STARFIT(AbstractPolicy):
    """
    STARFIT policy class for reservoir operation.

    This policy infers release rules using STARFIT-derived operating parameters.
    STARFIT is a seasonal time series model that uses harmonic adjustments to
    determine reservoir releases.

    Attributes:
        Reservoir (Reservoir): The reservoir instance associated with the policy.
        policy_params (dict): Dictionary containing STARFIT parameters.
        S_cap (float): Total capacity of the reservoir.
        I_bar (float): Mean annual inflow to the reservoir.
        R_max (float): Maximum release from the reservoir.
        R_min (float): Minimum release from the reservoir.
        NORhi_mu (float): Mean upper Normal Operating Range (NOR) bound.
        NORhi_min (float): Minimum upper NOR bound.
        NORhi_max (float): Maximum upper NOR bound.
        NORhi_alpha (float): Alpha parameter for upper NOR bound.
        NORhi_beta (float): Beta parameter for upper NOR bound.
        NORlo_mu (float): Mean lower NOR bound.
        NORlo_min (float): Minimum lower NOR bound.
        NORlo_max (float): Maximum lower NOR bound.
        NORlo_alpha (float): Alpha parameter for lower NOR bound.
        NORlo_beta (float): Beta parameter for lower NOR bound.
        Release_alpha1 (float): Alpha1 parameter for seasonal release adjustment.
        Release_alpha2 (float): Alpha2 parameter for seasonal release adjustment.
        Release_beta1 (float): Beta1 parameter for seasonal release adjustment.
        Release_beta2 (float): Beta2 parameter for seasonal release adjustment.
        Release_c (float): C parameter for seasonal release adjustment.
        Release_p1 (float): P1 parameter for seasonal release adjustment.
        Release_p2 (float): P2 parameter for seasonal release adjustment.
        start_month (str): Starting month for harmonic calculations.
        starfit_name (str): Name of the STARFIT reservoir.


    """

    def __init__(self, Reservoir, policy_params):
        """
        Initialize STARFIT policy with reservoir reference and parameters array.

        Args:
            Reservoir (Reservoir): Reservoir instance associated with the policy.
            policy_params (dict): Dictionary containing STARFIT parameters.
            starfit_df (pd.DataFrame): DataFrame containing STARFIT parameters.
            reservoir_name (str): Name of the reservoir associated with STARFIT parameters.
        """
        self.Reservoir = Reservoir
        self.reservoir_name = self.Reservoir.name if hasattr(self.Reservoir, "name") else None
        if self.reservoir_name is None:
            raise ValueError("Reservoir object must have a 'name' attribute.")
        self.starfit_df = pd.read_csv("data/drb_model_istarf_conus.csv")
        self.load_starfit_params()
        self.policy_params = policy_params
        self.parse_policy_params(policy_params)
    
    def load_starfit_params(self):
        """
        Load known parameters (S_cap, I_bar, R_max, R_min) from the STARFIT CSV.
        """
        if self.reservoir_name not in self.starfit_df["reservoir"].values:
            raise ValueError(f"STARFIT parameters not found for '{self.reservoir_name}'. Check spelling!")

        res_data = self.starfit_df[self.starfit_df["reservoir"] == self.reservoir_name].iloc[0]

        # Assign storage & inflow parameters
        if not pd.isna(res_data["Adjusted_CAP_MG"]) and not pd.isna(res_data["Adjusted_MEANFLOW_MGD"]):
            self.S_cap = res_data["Adjusted_CAP_MG"]
            self.I_bar = res_data["Adjusted_MEANFLOW_MGD"]
        else:
            self.S_cap = res_data["GRanD_CAP_MG"]
            self.I_bar = res_data["GRanD_MEANFLOW_MGD"]

        # Calculate R_min and R_max
        self.R_max = (res_data["Release_max"] + 1) * self.I_bar
        self.R_min = (res_data["Release_min"] + 1) * self.I_bar

    
    def parse_policy_params(self, policy_params):
        """
        Parses STARFIT parameters from an array.
        """
        if len(policy_params) != 17:
            raise ValueError(f"Expected 17 parameters, but received {len(policy_params)}.")

        # Unpack the array into the expected order
        (
            self.NORhi_mu, self.NORhi_min, self.NORhi_max, 
            self.NORhi_alpha, self.NORhi_beta,
            self.NORlo_mu, self.NORlo_min, self.NORlo_max,
            self.NORlo_alpha, self.NORlo_beta,
            self.Release_alpha1, self.Release_alpha2,
            self.Release_beta1, self.Release_beta2,
            self.Release_c, self.Release_p1, self.Release_p2
        ) = policy_params
    

    def validate_policy_params(self):
        """
        Ensures all required parameters are assigned correctly.
        """
        required_params = [
            "NORhi_mu", "NORhi_min", "NORhi_max", "NORhi_alpha", "NORhi_beta",
            "NORlo_mu", "NORlo_min", "NORlo_max", "NORlo_alpha", "NORlo_beta",
            "Release_alpha1", "Release_alpha2", "Release_beta1", "Release_beta2",
            "Release_c", "Release_p1", "Release_p2", 
            "S_cap", "I_bar", "R_min", "R_max"
        ]
        
        for param in required_params:
            if not hasattr(self, param):
                raise ValueError(f"Missing required parameter '{param}' in STARFIT class.")
    

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
        pass