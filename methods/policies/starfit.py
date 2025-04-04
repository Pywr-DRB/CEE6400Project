from methods.policies.abstract_policy import AbstractPolicy
import numpy as np
import pandas as pd
from math import sin, cos, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from methods.config import policy_n_params, policy_param_bounds


class STARFIT(AbstractPolicy):
    """
    STARFIT policy class for reservoir operation.

    This policy infers release rules using STARFIT-derived operating parameters.
    STARFIT is a seasonal time series model that uses harmonic adjustments to
    determine reservoir releases.
    """

    def __init__(self, Reservoir, policy_params):
        self.Reservoir = Reservoir
        self.reservoir_name = self.Reservoir.name if hasattr(self.Reservoir, "name") else None
        if self.reservoir_name is None:
            raise ValueError("Reservoir object must have a 'name' attribute.")

        self.starfit_df = pd.read_csv("data/drb_model_istarf_conus.csv")
        self.load_starfit_params()

        self.n_params = policy_n_params["STARFIT"]
        self.param_bounds = policy_param_bounds["STARFIT"]
        self.policy_params = policy_params
        self.parse_policy_params(policy_params)

        self.dates = None  # To be assigned externally for date-aware seasonal logic

    def load_starfit_params(self):
        if self.reservoir_name not in self.starfit_df["reservoir"].values:
            raise ValueError(f"STARFIT parameters not found for '{self.reservoir_name}'. Check spelling!")

        res_data = self.starfit_df[self.starfit_df["reservoir"] == self.reservoir_name].iloc[0]

        if not pd.isna(res_data["Adjusted_CAP_MG"]) and not pd.isna(res_data["Adjusted_MEANFLOW_MGD"]):
            self.S_cap = res_data["Adjusted_CAP_MG"]
            self.I_bar = res_data["Adjusted_MEANFLOW_MGD"]
        else:
            self.S_cap = res_data["GRanD_CAP_MG"]
            self.I_bar = res_data["GRanD_MEANFLOW_MGD"]

        self.R_max = (res_data["Release_max"] + 1) * self.I_bar
        self.R_min = (res_data["Release_min"] + 1) * self.I_bar

    def parse_policy_params(self, policy_params):
        if len(policy_params) != self.n_params:
            raise ValueError(f"Expected 17 parameters, but received {len(policy_params)}.")

        (
            self.NORhi_mu, self.NORhi_min, self.NORhi_max, 
            self.NORhi_alpha, self.NORhi_beta,
            self.NORlo_mu, self.NORlo_min, self.NORlo_max,
            self.NORlo_alpha, self.NORlo_beta,
            self.Release_alpha1, self.Release_alpha2,
            self.Release_beta1, self.Release_beta2,
            self.Release_c, self.Release_p1, self.Release_p2
        ) = policy_params
        
        self.validate_policy_params()
        
        # Setup the self.weekly_NORhi/lo_array        
        self.calculate_weekly_NOR()

    def validate_policy_params(self):
        assert len(self.policy_params) == self.n_params
        for i, p in enumerate(self.policy_params):
            bounds = self.param_bounds[i]
            assert bounds[0] <= p <= bounds[1], f"Parameter {i} out of bounds: {p} not in {bounds}"

    def sinNpi(self, t, N):
        return sin(N * pi * (t + 39) / 52)

    def cosNpi(self, t, N):
        return cos(N * pi * (t + 39) / 52)

    def get_week_index(self, timestep):
        if self.dates is not None:
            return self.dates[timestep].isocalendar().week % 52
        else:
            return timestep % 52

    def release_harmonic(self, time):
        return (
            self.Release_alpha1 * self.sinNpi(time, 2)
            + self.Release_alpha2 * self.sinNpi(time, 4)
            + self.Release_beta1 * self.cosNpi(time, 2)
            + self.Release_beta2 * self.cosNpi(time, 4)
        )

    def calc_NOR_hi(self, time):
        NOR_hi = (
            self.NORhi_mu
            + self.NORhi_alpha * self.sinNpi(time, 2)
            + self.NORhi_beta * self.cosNpi(time, 2)
        )
        return np.clip(NOR_hi, self.NORhi_min, self.NORhi_max) / 100

    def calc_NOR_lo(self, time):
        NOR_lo = (
            self.NORlo_mu
            + self.NORlo_alpha * self.sinNpi(time, 2)
            + self.NORlo_beta * self.cosNpi(time, 2)
        )
        return np.clip(NOR_lo, self.NORlo_min, self.NORlo_max) / 100

    def calculate_weekly_NOR(self):
        #TODO: Double check this
        
        # loop through timestep 
        weekly_NORhi = []
        weekly_NORlo = []
        for t in range(52):
            NOR_hi = self.calc_NOR_hi(t)
            NOR_lo = self.calc_NOR_lo(t)
            weekly_NORhi.append(NOR_hi)
            weekly_NORlo.append(NOR_lo)
        
        # make sure NOR_hi is always greater than NOR_lo
        weekly_NORhi = np.array(weekly_NORhi)
        weekly_NORlo = np.array(weekly_NORlo)
        self.weekly_NORhi_array = weekly_NORhi
        self.weekly_NORlo_array = weekly_NORlo

    
    def test_nor_constraint(self):
        #TODO: Double check this
        
        self.calculate_weekly_NOR()
        if np.any(self.weekly_NORhi_array < self.weekly_NORlo_array):
            return False
        else:
            return True

    def percent_storage(self, S_t):
        return S_t / self.S_cap

    def get_release(self, timestep):
        I_t = self.Reservoir.inflow_array[timestep]
        S_t = self.Reservoir.storage_array[timestep - 1] if timestep > 0 else self.Reservoir.initial_storage

        S_hat = self.percent_storage(S_t)
        week = self.get_week_index(timestep)
        harmonic = self.release_harmonic(week)

        #TODO: Double check release formula
        target_R = (S_hat + harmonic) * self.S_cap + self.Release_c
        target_R = np.clip(target_R, self.R_min, self.R_max)

        #TODO: Need to enforce NORhi/lo bounds
        # nor_lo = self.weekly_NORhi_array[week]
        # nor_hi = self.weekly_NORlo_array[week]
        
        return self.enforce_constraints(target_R)

    def plot(self, fname="STARFIT_Storage_vs_Release.png"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.Reservoir.storage_array, self.Reservoir.release_array, s=10, alpha=0.7, label="Simulated")
        ax.set_title(f"STARFIT Policy Curve - {self.reservoir_name}")
        ax.set_xlabel("Storage (MG)")
        ax.set_ylabel("Release (MGD)")
        ax.grid(True)
        ax.legend()
        plt.savefig(f"./figures/{fname}", dpi=300)
        plt.show()

    def plot_policy_surface(self, save=True, fname="STARFIT_PolicySurface3D.png"):
        perc_storage = np.linspace(0, 1, 100)
        weeks = np.linspace(0, 52, 100)

        X, Y = np.meshgrid(perc_storage, weeks)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                S_hat = X[i, j]
                t = Y[i, j]
                harmonic = (
                    self.Release_alpha1 * np.sin(2 * np.pi * t / 52)
                    + self.Release_alpha2 * np.sin(4 * np.pi * t / 52)
                    + self.Release_beta1 * np.cos(2 * np.pi * t / 52)
                    + self.Release_beta2 * np.cos(4 * np.pi * t / 52)
                )
                target_R = (S_hat + harmonic) * self.S_cap + self.Release_c
                Z[i, j] = np.clip(target_R, self.R_min, self.R_max)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='k', linewidth=0.2)
        ax.set_xlabel("Percent Storage")
        ax.set_ylabel("Week of Year")
        ax.set_zlabel("Release (MGD)")
        ax.set_title(f"STARFIT Policy Surface - {self.reservoir_name}")

        if save:
            plt.savefig(f"./figures/{fname}", dpi=300)
        plt.show()