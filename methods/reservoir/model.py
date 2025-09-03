import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods.policies import PiecewiseLinear, RBF, STARFIT
from methods.utils.conversions import cfs_to_mgd


class Reservoir():
    def __init__(self, 
                 inflow, 
                 capacity,
                 policy_type,
                 policy_params,
                 dates = None,
                 release_min = None,
                 release_max = None,
                 inflow_max = None,
                 inflow_min = None,
                 initial_storage = None,
                 start_date = None,
                 name = None
                 ):
        """
        Reservoir model class, used to simulate the operation of a reservoir.
        
        Simulation data is stored as class attributes, to be accessed
        by the policy to inform release decisions, and to be retrieved
        after simulation for objective calculation.
        
        Parameters
        ----------
        inflow : np.array
            Array of inflow values for each timestep.
        capacity : float
            Maximum storage capacity of the reservoir.
        policy_type : str
            Type of policy to use for operation. Options are: "PiecewiseLinear", "RBF", "STARFIT"
        policy_params : np.array
            Array of policy parameters to be optimized.
        release_min : float, optional
            Minimum release constraint. Default is None.
        release_max : float, optional
            Maximum release constraint. Default is None.
        inflow_max : float, optional
            Maximum inflow used for scaling inflow inputs. If None, use max of observations.
        inflow_min : float, optional
            Minimum inflow used for scaling inflow inputs. If None, use 0.0.
        initial_storage : float, optional
            Initial storage level of the reservoir. Default is 0.8 * capacity.
        dates : pd.DatetimeIndex, optional
            Dates for the simulation timeseries. Default is None. 
        
        Methods
        -------
        run()
            Run the reservoir simulation for the given inflow and demand timeseries.
        
        get_release(t)
            Get the release for the given timestep, from the policy class.
            
        Returns
        -------
        None
        """
        try:
            inflow = np.array(inflow).flatten()
        except Exception as e:
            raise ValueError(f"Inflow must be a 1D array. Got {inflow.shape}") from e
        
        # Input timeseries
        self.inflow_array = inflow
        self.T = len(inflow)
        self.start_date = pd.to_datetime(start_date) if start_date is not None else None
        self.doy = pd.Series(pd.date_range(start=self.start_date, periods=self.T)).dt.dayofyear.values if self.start_date is not None else None
        
        # Dates
        assert dates is not None, "Dates must be provided to the Reservoir during initalization."
        self.dates = pd.to_datetime(dates)
        
        # Reservoir characteristics
        self.capacity = capacity
        self.initial_storage = 0.8 * capacity if initial_storage is None else initial_storage
        
        # Conservation releases
        conservation_releases = {
            "blueMarsh": cfs_to_mgd(50),
            "beltzvilleCombined": cfs_to_mgd(35),
            "nockamixon": cfs_to_mgd(11),
            "fewalter": cfs_to_mgd(50),
        }

        self.name = name 
        self.release_max = release_max
        self.release_min = release_min if release_min else conservation_releases.get(self.name, 0) 

        # Per-reservoir inflow bounds
        if inflow_min is not None and inflow_max is not None:
            self.inflow_min = float(inflow_min)
            self.inflow_max = float(inflow_max)
        else:
            self.inflow_max = np.max(inflow) if inflow_max is None else inflow_max
            self.inflow_min = 0.0 if inflow_min is None else inflow_min
  
        # reset simulation variables
        self.reset()
        
        # initialize policy class
        self.policy = None
        self.initalize_policy(policy_type, 
                               policy_params)
    
    def reset(self):
        """
        Reset the variable arrays prior to simulation.
        """
        self.storage_array = np.zeros(self.T)
        self.release_array = np.zeros(self.T)
        self.spill_array = np.zeros(self.T)
        return
    
    def initalize_policy(self, 
                          policy_type,
                          policy_params):
        """
        Initialize the policy class based on the policy type.
        
        Each policy class will have a different set of parameters,
        which will be parsed inside the unique classes. 
        
        Parameters
        ----------
        policy_type : str
            Type of policy to use for operation. Options are: "PiecewiseLinear", "RBF", "STARFIT"
        policy_options : dict
            Dictionary of optional kwargs to pass to the policy class.
        policy_params : np.array
            Array of policy parameters to be optimized.        
        """    
    
        if policy_type == 'PiecewiseLinear':
            self.policy = PiecewiseLinear(Reservoir=self, 
                                          policy_params=policy_params)
            
        elif policy_type == 'RBF':
            self.policy = RBF(Reservoir=self,
                              policy_params = policy_params)
            
        elif policy_type == 'STARFIT':
            self.policy = STARFIT(Reservoir=self,
                                  policy_params=policy_params)
        
        else:
            raise ValueError(f"Invalid policy type: {policy_type}")
        
        self.policy.validate_policy_params()
        return 
    
    
    def get_release(self, timestep):
        """
        Use the policy class to get the release for the given timestep,
        the policy class can access reservoir attributes to make decisions.
        """
        return self.policy.get_release(timestep=timestep)
    
    
    
    def run(self):
        """
        Run the reservoir simulation.
        """
        
        # reset simulation variables
        self.reset()
        
        # make sure policy is initialized
        if self.policy is None:
            raise ValueError("Policy not initialized.")
        
        # run simulation
        for t in range(0, self.T):
            # get target release from policy
            release_target = self.get_release(timestep=t)
            
            # enforce constraints on release
            release_allowable = min(release_target, self.storage_array[t-1] + self.inflow_array[t])
            self.release_array[t] = release_allowable

            # update
            if t == 0:
                self.storage_array[t] = self.initial_storage + self.inflow_array[t] - self.release_array[t]
            else:
                self.storage_array[t] = self.storage_array[t-1] + self.inflow_array[t] - self.release_array[t]

            # Check for spill
            if self.storage_array[t] > self.capacity:
                self.spill_array[t] = self.storage_array[t] - self.capacity
                self.storage_array[t] = self.capacity
            else:
                self.spill_array[t] = 0.0
        
        return

    def get_results(self):
        # Make a pd.DataFrame of the results
        results = {
            "inflow": self.inflow_array,
            "release": self.release_array,
            "storage": self.storage_array,
            "spill": self.spill_array
        }
        
        results = pd.DataFrame(results)
        
        return results

    def plot(self,
             fname=None,
             save=False,
             storage_obs=None,
             release_obs=None,
             release_log_scale=True,
             release_smooth_window=None,
             title=None):
        """
        Plot storage and release data.
        
        Args:
            storage_obs (pd.Series): Observed storage data for the simulation period.
            release_obs (pd.Series): Observed release data for the simulation period.
            save (bool): Whether to save the plot as a file.
            fname (str): Filename for saving the plot.
        
        """
        
        storage_sim = self.storage_array
        release_sim = self.release_array
        
        
        ### plot simulated and obs storage and release
        fig, ax = plt.subplots(2, 1, figsize=(10, 8),
                                sharex=True)
        
        ax[0].plot(storage_obs, label='Observed', color='blue', alpha=0.5)
        ax[0].plot(storage_sim, label='Simulated', color='orange', alpha=0.5)
        ax[0].set_ylabel('Storage (MGD)')
        ax[0].set_xlim(0, self.T)
        ax[0].set_title(f"Reservoir Dynamics – {self.name.upper()}")

        
        # smooth releases
        if release_smooth_window is not None:
            w = release_smooth_window
            release_obs = pd.Series(release_obs.flatten()).rolling(window=w).mean().values
            release_sim = pd.Series(release_sim.flatten()).rolling(window=w).mean().values
        
        ax[1].plot(release_obs, label='Observed', color='blue', alpha=0.5)
        ax[1].plot(release_sim, label='Simulated', color='orange', alpha=0.5)
        if release_log_scale:
            ax[1].set_yscale('log')

        ax[1].set_ylabel('Release (MGD)')
        
        ax[1].set_xlabel('Time')
        ax[0].legend()
        
        if title is not None:
            fig.suptitle(title)
        
        if save:
            if fname is None:
                raise ValueError("Filename must be provided when save=True.")
            plt.savefig(fname, dpi=300)
        
        plt.tight_layout()
        plt.show()