import numpy as np
import pandas as pd
from math import sin, cos, pi
import matplotlib.pyplot as plt
import os

from methods.policies.abstract_policy import AbstractPolicy
from methods.config import policy_n_params, policy_param_bounds
from methods.config import DATA_DIR, FIG_DIR


class STARFIT(AbstractPolicy):
    """
    STARFIT policy class for reservoir operation.

    STARFIT is a seasonal time series model that uses harmonic adjustments to
    determine reservoir releases using a normal operating range (NOR) framework.
    """

    def __init__(self, Reservoir, policy_params):
        self.Reservoir = Reservoir
        self.reservoir_name = getattr(self.Reservoir, "name", None)
        if self.reservoir_name is None:
            raise ValueError("Reservoir object must have a 'name' attribute.")

        self.dates = self.Reservoir.dates

        self.starfit_df = pd.read_csv(f"{DATA_DIR}/drb_model_istarf_conus.csv")
        self.load_starfit_params()

        self.n_params = policy_n_params["STARFIT"]
        self.param_bounds = policy_param_bounds["STARFIT"]
        self.policy_params = policy_params
        self.parse_policy_params()

        self.WATER_YEAR_OFFSET = 0  

        self.S_cap = Reservoir.capacity
        self.R_min = self.Reservoir.release_min
        self.R_max = self.Reservoir.release_max
    

        self.log_path = f"STARFIT_release_log_{self.reservoir_name}.txt"
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

        print(f"[DEBUG] R_max: {self.R_max}, I_bar: {self.I_bar}, R_min: {self.R_min}")
        print(f"[DEBUG] Reservoir.capacity: {self.Reservoir.capacity:.2f} | S_cap (from CSV): {self.S_cap:.2f}")

    def load_starfit_params(self):
        if self.reservoir_name not in self.starfit_df["reservoir"].values:
            raise ValueError(f"STARFIT parameters not found for '{self.reservoir_name}'.")

        res_data = self.starfit_df[self.starfit_df["reservoir"] == self.reservoir_name].iloc[0]
        self.S_cap = res_data["Adjusted_CAP_MG"] if not pd.isna(res_data["Adjusted_CAP_MG"]) else res_data["GRanD_CAP_MG"]
        self.I_bar = res_data["Adjusted_MEANFLOW_MGD"] if not pd.isna(res_data["Adjusted_MEANFLOW_MGD"]) else res_data["GRanD_MEANFLOW_MGD"]
        self.R_max = (res_data["Release_max"] + 1) * self.I_bar
        self.R_min = (res_data["Release_min"] + 1) * self.I_bar

    def parse_policy_params(self):
        self.validate_policy_params()
        (
            self.NORhi_mu, self.NORhi_min, self.NORhi_max, 
            self.NORhi_alpha, self.NORhi_beta,
            self.NORlo_mu, self.NORlo_min, self.NORlo_max,
            self.NORlo_alpha, self.NORlo_beta,
            self.Release_alpha1, self.Release_alpha2,
            self.Release_beta1, self.Release_beta2,
            self.Release_c, self.Release_p1, self.Release_p2
        ) = self.policy_params

    def validate_policy_params(self):
        assert len(self.policy_params) == self.n_params
        for i, p in enumerate(self.policy_params):
            bounds = self.param_bounds[i]
            assert bounds[0] <= p <= bounds[1], f"Parameter {i} out of bounds: {p} not in {bounds}"
    
    def test_nor_constraint(self):
        self.calculate_weekly_NOR()
        if np.any(self.weekly_NORhi_array < self.weekly_NORlo_array):
            # Optional: Write violating parameters to a file
            with open("violated_params.log", "a") as f:
                f.write(f"\nViolation for {self.reservoir_name} at {pd.Timestamp.now()}:\n")
                f.write(f"{self.policy_params}\n")
                f.write("--------\n")
            return False
        else:
            return True
    
    def evaluate(self, X):
        S_t, I_t, W_t = X
        S_hat = S_t / self.S_cap
        I_hat = (I_t - self.I_bar) / self.I_bar

        def get_c(W_t):
            doy = int((W_t * 365 + self.WATER_YEAR_OFFSET) % 365)
            return pi * doy / 365

        def get_NOR_hi(c):
            return np.clip(self.NORhi_mu + self.NORhi_alpha * sin(2 * c) + self.NORhi_beta * cos(2 * c),
                       self.NORhi_min, self.NORhi_max) / 100

        def get_NOR_lo(c):
            return np.clip(self.NORlo_mu + self.NORlo_alpha * sin(2 * c) + self.NORlo_beta * cos(2 * c),
                       self.NORlo_min, self.NORlo_max) / 100

        def get_harmonic(c):
            return (self.Release_alpha1 * sin(2 * c) +
                self.Release_alpha2 * sin(4 * c) +
                self.Release_beta1 * cos(2 * c) +
                self.Release_beta2 * cos(4 * c))

        def get_epsilon(S_hat, I_hat, NOR_hi, NOR_lo):
            A_t = (S_hat - NOR_lo) / (NOR_hi + 1e-6)
            return self.Release_c + self.Release_p1 * A_t + self.Release_p2 * I_hat

        def decision_logic(S_hat, I_t, harmonic, epsilon, NOR_hi, NOR_lo):
            if NOR_lo <= S_hat <= NOR_hi:
                return min(self.I_bar * (harmonic + epsilon) + self.I_bar, self.R_max)
            elif S_hat > NOR_hi:
                return min((self.S_cap * (S_hat - NOR_hi) + I_t * 7) / 7, self.R_max)
            else:
                return self.R_min

        c = get_c(W_t)
        NOR_hi = get_NOR_hi(c)
        NOR_lo = get_NOR_lo(c)
        harmonic = get_harmonic(c)
        epsilon = get_epsilon(S_hat, I_hat, NOR_hi, NOR_lo)
        release = decision_logic(S_hat, I_t, harmonic, epsilon, NOR_hi, NOR_lo)

        return release

    def get_release(self, timestep):
        # Get state variables
        I_t = self.Reservoir.inflow_array[timestep]
        S_t = self.Reservoir.initial_storage if timestep == 0 else self.Reservoir.storage_array[timestep - 1]
        week_of_year = self.dates[timestep].timetuple().tm_yday / 365.0

        # make sure I_t and S_t are float
        I_t = float(I_t)
        S_t = float(S_t)
        week_of_year = float(week_of_year)

        release = self.evaluate([S_t, I_t, week_of_year])
        release = self.enforce_constraints(release)
        release = max(min(release, I_t + S_t), I_t + S_t - self.Reservoir.capacity)
        release = max(0, release)
        
        return release
        

    def plot(self, fname="STARFIT_Storage_vs_Release.png", save=True):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.Reservoir.storage_array, self.Reservoir.release_array, s=10, alpha=0.7, label="Simulated")
        ax.set_title(f"STARFIT Policy Curve - {self.reservoir_name}")
        ax.set_xlabel("Storage (MG)")
        ax.set_ylabel("Release (MGD)")
        ax.grid(True)
        ax.legend()

        if save:
            print(f"Saving policy plot to {fname}")
            plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)

        plt.show()

    def calculate_weekly_NOR(self):
        weekly_NORhi = []
        weekly_NORlo = []

        dummy_dates = pd.date_range("2020-10-01", periods=52, freq='W')
        for dt in dummy_dates:
            doy = (dt.timetuple().tm_yday) % 365
            c = pi * doy / 365

            NOR_hi = self.NORhi_mu + self.NORhi_alpha * sin(c * 2) + self.NORhi_beta * cos(c * 2)
            NOR_lo = self.NORlo_mu + self.NORlo_alpha * sin(c * 2) + self.NORlo_beta * cos(c * 2)

            weekly_NORhi.append(np.clip(NOR_hi, self.NORhi_min, self.NORhi_max) / 100)
            weekly_NORlo.append(np.clip(NOR_lo, self.NORlo_min, self.NORlo_max) / 100)

        self.weekly_NORhi_array = np.array(weekly_NORhi)
        self.weekly_NORlo_array = np.array(weekly_NORlo)


    def plot_nor(self, 
                 percent_storage=True,
                 fname=None):
        """
        Plot of NORhi and NORlo w/r/t week of year.
        
        Args:
            percent_storage : 
            fname (str)
        """
        self.calculate_weekly_NOR()
        nor_hi = self.weekly_NORhi_array
        nor_lo = self.weekly_NORlo_array
        xs = np.arange(1, 53)
        
        fig, ax = plt.subplots()        
        ax.fill_between(xs, nor_lo, nor_hi)
        if fname is not None:
            plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
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
            plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
        plt.show()