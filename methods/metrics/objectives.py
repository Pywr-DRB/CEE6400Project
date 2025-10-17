"""
Used to calculate the objectives after Reservoir.run()
"""
import hydroeval as he
import numpy as np

class ObjectiveCalculator():
    """
    Used to calculate objectives used for policy parameter optimization.
    
    Each metric is transformed such that minimization is best;
    for example 'neg_nse' is the negative of the NSE, 
    and -1.0 would be a perfect match.
    
    Example usage:
    ObjFunc = ObjectiveCalculator(metrics=['nse', 'rmse', 'kge'])
    obs = [1, 2, 3, 4, 5]
    sim = [1.1, 2.2, 3.3, 4.4, 5.5]
    objs = ObjFunc.calculate(obs, sim)
    
    
    Uses the hydroeval package for some of the calculations:
    https://thibhlln.github.io/hydroeval/index.html
    
    Hallouin, T. (2025). hydroeval: an evaluator for streamflow time series 
        in Python. Zenodo. https://doi.org/10.5281/zenodo.2591217
    """
    def __init__(self,
                 metrics,
                 capacity_mg: float | None = None,
                 inertia_tau: float = 0.02,
                 inertia_scale_release: str = "range",
                 inertia_release_scale_value: float | None = None,
                 inertia_scale_storage: str = "value",
                 inertia_storage_scale_value: float | None = None):

        base_valid = [
            'neg_nse', 'rmse', 'neg_kge', 'abs_pbias', 'nrmse',
            'neg_inertia_release', 'neg_inertia_storage'
        ]

        # Build valid list with transforms, BUT skip transforms for inertia metrics
        inertias = {'neg_inertia_release', 'neg_inertia_storage'}
        valid_metrics = []
        for m in base_valid:
            valid_metrics.append(m)
            if m not in inertias:
                valid_metrics.append(f'log_{m}')
                valid_metrics.append(f'Q20_{m}')  # calculated using <20th percentile obs
                valid_metrics.append(f'Q80_{m}')  # calculated using >80th percentile obs
                valid_metrics.append(f'Q20_log_{m}') # log + Q20
                valid_metrics.append(f'Q80_log_{m}') # log + Q80

        for m in metrics:
            if m not in valid_metrics:
                raise ValueError(f"Invalid metric: {m}\nValid metrics: {valid_metrics}")

        self.metrics = list(metrics)
        self.capacity_mg = capacity_mg
        self.inertia_tau = float(inertia_tau)
        self.inertia_scale_release = str(inertia_scale_release)
        self.inertia_release_scale_value = (
            float(inertia_release_scale_value) if inertia_release_scale_value is not None else None
        )
        self.inertia_scale_storage = str(inertia_scale_storage)
        self.inertia_storage_scale_value = (
            float(inertia_storage_scale_value) if inertia_storage_scale_value is not None else None
        )
                
        pass
    
    def validate_inputs(self, obs, sim):
        """
        Check that the obs and sim data are the same length.
        """
        if len(obs) != len(sim):
            raise ValueError("obs and sim data must be the same length.")
    
    
    def calculate(self, obs, sim):
        """
        Calculate the objective value based on the selected metric.
        """
        
        # make sure data is same length
        self.validate_inputs(obs, sim)

        obs_full = np.asarray(obs, dtype=float)
        sim_full = np.asarray(sim, dtype=float)
        
        objs = []
        for metric in self.metrics:

            log = ('log_' in metric)

            # Slice for Q20/Q80 on *obs* percentiles (for non-inertia metrics only)
            obs_m = obs_full
            sim_m = sim_full
            if metric.startswith('Q20_'):
                base = metric.replace('Q20_', '')
                if 'inertia' not in base:
                    use_idx = obs_full < np.percentile(obs_full, 20)
                    obs_m = obs_full[use_idx]
                    sim_m = sim_full[use_idx]
                metric = base
            elif metric.startswith('Q80_'):
                base = metric.replace('Q80_', '')
                if 'inertia' not in base:
                    use_idx = obs_full > np.percentile(obs_full, 80)
                    obs_m = obs_full[use_idx]
                    sim_m = sim_full[use_idx]
                metric = base
            if metric.startswith('log_'):
                metric = metric.replace('log_', '')

            # ---- INERTIA (use this calculator's sim) --------------------------
            if metric == 'neg_inertia_release':
                scale = self.inertia_scale_release
                scale_value = self.inertia_release_scale_value  
                val = self.policy_inertia(sim_full, tau=self.inertia_tau,
                                          scale=scale, scale_value=scale_value)
                objs.append(-val)
                continue
            if metric == 'neg_inertia_storage':
                # default to capacity scaling if provided (nice & comparable)
                scale = self.inertia_scale_storage
                scale_value = self.inertia_storage_scale_value
                if scale == 'value' and (scale_value is None or scale_value <= 0) and self.capacity_mg:
                    scale_value = float(self.capacity_mg)
                val = self.policy_inertia(sim_full, tau=self.inertia_tau,
                                          scale=scale, scale_value=scale_value)
                objs.append(-val)
                continue

            # ---- STANDARD METRICS --------------------------------------------
            if metric == 'neg_nse':
                o = self.nse(obs=obs_m, sim=sim_m, log=log)[0]
                objs.append(-o)
            elif metric == 'rmse':
                o = self.rmse(obs=obs_m, sim=sim_m, log=log)[0]
                objs.append(o)
            elif metric == 'neg_kge':
                o = self.kge(obs=obs_m, sim=sim_m, log=log)[0]
                objs.append(-o[0])   # KGE returns [KGE, cc, alpha, beta]
            elif metric == 'abs_pbias':
                o = self.pbias(obs=obs_m, sim=sim_m, log=log)[0]
                objs.append(abs(o))
            elif metric == 'nrmse':
                rmse_val = self.rmse(obs=obs_m, sim=sim_m, log=log)[0]
                if self.capacity_mg is None or self.capacity_mg <= 0:
                    objs.append(9999.0)
                else:
                    objs.append(rmse_val / float(self.capacity_mg))
            else:
                raise ValueError(f"Metric not handled: {metric}")
        
        objs = [float(o) for o in objs]
        
        return objs
    
    def nse(self, obs, sim, log=False):
        """
        Nash-Sutcliffe Efficiency (NSE).
        """
        if log:
            return he.evaluator(he.nse, sim, obs, transform='log')
        else:
            return he.evaluator(he.nse, sim, obs)
    
    def rmse(self, obs, sim, log=False):
        """
        Root Mean Squared Error (RMSE)
        """
        if log:
            return he.evaluator(he.rmse, sim, obs, transform='log')
        else:
            return he.evaluator(he.rmse, sim, obs)
    
    def kge(self, obs, sim, log=False):
        """
        Kling-Gupta Efficiency (KGE)
        
        Note: all 4 KGE components are returned, 
        but this is setup to keep only the total KGE.
        """
        if log:
            return he.evaluator(he.kge, sim, obs, transform='log')
        else:
            return he.evaluator(he.kge, sim, obs)
    
    def pbias(self, obs, sim, log=False):
        """
        Percent Bias (PBIAS)
        """
        if log:
            return he.evaluator(he.pbias, sim, obs, transform='log')
        else:
            return he.evaluator(he.pbias, sim, obs)
        
    @staticmethod
    def policy_inertia(u: np.ndarray, tau: float = 0.02,
                    scale: str = "range", scale_value: float | None = None) -> float:
        """
        Compute a **symmetric** inertia score on a time series u (release or storage).

        Concept
        -------
        We count the fraction of time steps where the absolute one-step change is “small.”
        Let u_t be the series at step t and define the one-step change
            d_t = |u_{t+1} - u_t|.
        For a chosen scale S and tolerance τ (tau), a step is "inertial" if d_t ≤ τ·S.

            inertia(u; τ, S) = (1 / (T-1)) * sum_{t=1..T-1}  1{ |u_{t+1}-u_t| ≤ τ·S }

        This **penalizes both increases and decreases** beyond the tolerance, unlike the
        one-sided (Quinn-style) version that only counts reductions.

        Interpretation
        --------------
        Returned value is in [0, 1]:
        • 1.0  → series is flat or changes only by small amounts (|Δ| ≤ τ·S) most of the time
        • 0.0  → frequent large step-to-step changes in either direction

        Scaling Options (S)
        -------------------
        The scale S sets the absolute size of what counts as a “small” change:

        - scale="range":
            S = max(u) - min(u)
            Adapts to the variability of u. Recommended default for both release and storage,
            giving a dimensionless tolerance relative to observed swings.

        - scale="max":
            S = max(u)
            Tolerance is a fraction of peak magnitude (useful if you think step changes
            should be judged relative to peaks).

        - scale="value":
            S = scale_value (must be > 0)
            External reference (e.g., storage capacity). If S is large relative to typical
            day-to-day changes, pick a correspondingly small τ to avoid inertia ~1.0.

        Practical Guidance
        ------------------
        • Release (scale="range"):
            Start with τ ≈ 0.01–0.03 (≈1–3% of the release swing).
        • Storage (scale="range"):
            Start with τ ≈ 0.002–0.006. If inertia saturates near 1.0, lower τ; if it hugs 0, raise τ.
        • If you need comparability across reservoirs irrespective of their variability,
        consider scale="max" (with a slightly larger τ) or "value" with an explicit S.

        Edge Cases
        ----------
        • len(u) < 2 → returns 1.0
        • S ≤ 0 (flat series) → returns 1.0

        Parameters
        ----------
        u : np.ndarray
            1D time series (release or storage).
        tau : float, default 0.02
            Fractional tolerance for a “small” change relative to S.
        scale : {"range", "max", "value"}, default "range"
            How S is defined (see above).
        scale_value : float | None
            Required when scale="value" (e.g., capacity in MG).

        Returns
        -------
        float
            Inertia score in [0, 1].
        """
        u = np.asarray(u, float).ravel()
        if u.size < 2:
            return 1.0

        if scale == "range":
            S = float(u.max() - u.min())
        elif scale == "max":
            S = float(u.max())
        elif scale == "value":
            if scale_value is None or scale_value <= 0:
                raise ValueError("scale_value must be > 0 when scale='value'.")
            S = float(scale_value)
        else:
            raise ValueError("scale must be 'range', 'max', or 'value'.")

        if S <= 0.0:
            return 1.0

        thr = float(tau * S)
        diffs = np.abs(np.diff(u))   
        return float((diffs <= thr).sum()) / float(u.size - 1)


    def get_metric_labels(self, prefix=None):
        """
        Get the metric labels for the selected metrics.
        """
        metric_labels = []
        
        for m in self.metrics:
            base = m.replace("neg_", "").replace("abs_", "").replace("log_", "")
            label = ""

            if 'nse' in m:
                label = "NSE (−)"
            elif 'rmse' in m:
                label = "RMSE"
            elif 'kge' in m:
                label = "KGE (−)"
            elif 'pbias' in m:
                label = "Bias Abs. %"
            elif 'nrmse' in m:
                label = "NRMSE"
            elif 'inertia_release' in m:
                label = "Inertia Release (−)"
            elif 'inertia_storage' in m:
                label = "Inertia Storage (−)"
            else:
                label = base.upper()

            # Add quantile prefix if needed
            if "Q20" in m:
                label = "Q20 " + label
            elif "Q80" in m:
                label = "Q80 " + label
            elif "log" in m:
                label = "Log " + label

            if prefix:
                label = f"{prefix} {label}"

            metric_labels.append(label)

        return metric_labels
    
    
    