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
                 inertia_storage_scale_value: float | None = None,
                 fdc_nq: int = 50,                 # number of quantiles for FDC error
                 fdc_normalize: str = "none",      # {"none","mean","max"} scaling before FDC
                 peak_k: int = 3,                  # compare top-k peaks
                 xcorr_demean: bool = True         # demean before correlation for stability
                 ):
        base_valid = [
            'neg_nse', 'rmse', 'neg_kge', 'abs_pbias', 'nrmse',
            'neg_inertia_release', 'neg_inertia_storage',
            'fdc_mse', 'fdc_ks', 'neg_xcorr0', 'abs_peak_lag'
        ]

        # Metrics that should NOT receive log/Q20/Q80 transforms
        no_transform = {
            'neg_inertia_release', 'neg_inertia_storage',
            'fdc_mse', 'fdc_ks', 'neg_xcorr0', 'abs_peak_lag'
        }

        valid_metrics = []
        for m in base_valid:
            valid_metrics.append(m)
            if m not in no_transform:
                valid_metrics.append(f'log_{m}')
                valid_metrics.append(f'Q20_{m}')
                valid_metrics.append(f'Q80_{m}')
                valid_metrics.append(f'Q20_log_{m}')
                valid_metrics.append(f'Q80_log_{m}')

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

        # NEW params
        self.fdc_nq = int(fdc_nq)
        self.fdc_normalize = str(fdc_normalize)
        self.peak_k = int(peak_k)
        self.xcorr_demean = bool(xcorr_demean)

        # keep a handle to the non-transform set for calculate()
        self._no_transform = no_transform

    def validate_inputs(self, obs, sim):
        if len(obs) != len(sim):
            raise ValueError("obs and sim data must be the same length.")

    def calculate(self, obs, sim):
        self.validate_inputs(obs, sim)
        obs_full = np.asarray(obs, dtype=float)
        sim_full = np.asarray(sim, dtype=float)

        objs = []
        for metric in self.metrics:
            log = ('log_' in metric)

            # Handle Q20/Q80 slicing (skip for "no_transform" metrics)
            obs_m = obs_full
            sim_m = sim_full
            if metric.startswith('Q20_'):
                base = metric.replace('Q20_', '')
                if base not in self._no_transform:
                    use_idx = obs_full < np.percentile(obs_full, 20)
                    obs_m = obs_full[use_idx]
                    sim_m = sim_full[use_idx]
                metric = base
            elif metric.startswith('Q80_'):
                base = metric.replace('Q80_', '')
                if base not in self._no_transform:
                    use_idx = obs_full > np.percentile(obs_full, 80)
                    obs_m = obs_full[use_idx]
                    sim_m = sim_full[use_idx]
                metric = base
            if metric.startswith('log_'):
                metric = metric.replace('log_', '')
                # log-transform only for metrics that accept transforms via hydroeval
                # FDC/XCORR/PEAK metrics ignore log flag internally

            # ---------------- INERTIA (unchanged) ----------------
            if metric == 'neg_inertia_release':
                scale = self.inertia_scale_release
                scale_value = self.inertia_release_scale_value
                val = self.policy_inertia(sim_full, tau=self.inertia_tau,
                                          scale=scale, scale_value=scale_value)
                objs.append(-val); continue
            if metric == 'neg_inertia_storage':
                scale = self.inertia_scale_storage
                scale_value = self.inertia_storage_scale_value
                if scale == 'value' and (scale_value is None or scale_value <= 0) and self.capacity_mg:
                    scale_value = float(self.capacity_mg)
                val = self.policy_inertia(sim_full, tau=self.inertia_tau,
                                          scale=scale, scale_value=scale_value)
                objs.append(-val); continue

            # ---------------- STANDARD METRICS -------------------
            if metric == 'neg_nse':
                o = self.nse(obs=obs_m, sim=sim_m, log=log)[0]; objs.append(-o)
            elif metric == 'rmse':
                o = self.rmse(obs=obs_m, sim=sim_m, log=log)[0]; objs.append(o)
            elif metric == 'neg_kge':
                o = self.kge(obs=obs_m, sim=sim_m, log=log)[0]; objs.append(-o[0])  # [KGE, cc, alpha, beta]
            elif metric == 'abs_pbias':
                o = self.pbias(obs=obs_m, sim=sim_m, log=log)[0]; objs.append(abs(o))
            elif metric == 'nrmse':
                rmse_val = self.rmse(obs=obs_m, sim=sim_m, log=log)[0]
                if self.capacity_mg is None or self.capacity_mg <= 0:
                    objs.append(9999.0)
                else:
                    objs.append(rmse_val / float(self.capacity_mg))

            # ---------------- NEW METRICS -----------------------
            elif metric == 'fdc_mse':
                val = self.fdc_mse(obs_full, sim_full,
                                   nq=self.fdc_nq, normalize=self.fdc_normalize)
                objs.append(val)
            elif metric == 'fdc_ks':
                val = self.fdc_ks(obs_full, sim_full)
                objs.append(val)
            elif metric == 'neg_xcorr0':
                val = self.neg_xcorr0(obs_full, sim_full, demean=self.xcorr_demean)
                objs.append(val)
            elif metric == 'abs_peak_lag':
                val = self.abs_peak_lag(obs_full, sim_full, k=self.peak_k)
                objs.append(val)
            else:
                raise ValueError(f"Metric not handled: {metric}")

        return [float(o) for o in objs]

    # ---------- classic metrics via hydroeval ----------
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
        

    # ---------- NEW: Flow Duration Curve errors ----------
    @staticmethod
    def _normalize_series(x: np.ndarray, mode: str) -> np.ndarray:
        if mode == "none":
            return x
        x = np.asarray(x, float)
        if mode == "mean":
            m = np.nanmean(x); return x / m if m != 0 else x
        if mode == "max":
            M = np.nanmax(x); return x / M if M != 0 else x
        raise ValueError("fdc_normalize must be one of {'none','mean','max'}")

    def fdc_mse(self, obs: np.ndarray, sim: np.ndarray, nq: int = 50,
                normalize: str = "none") -> float:
        """
        MSE between observed and simulated FDCs sampled at nq quantiles (0..1).
        """
        # optional normalization (helps scale invariance for different reservoirs)
        obs_n = self._normalize_series(obs, normalize)
        sim_n = self._normalize_series(sim, normalize)

        # quantile grid (exclude 0,1 to avoid extremes sensitivity)
        qs = np.linspace(0.01, 0.99, nq)
        f_obs = np.quantile(obs_n, qs, method="linear")
        f_sim = np.quantile(sim_n, qs, method="linear")
        dif = f_sim - f_obs
        return float(np.mean(dif * dif))

    def fdc_ks(self, obs: np.ndarray, sim: np.ndarray) -> float:
        """
        Kolmogorov–Smirnov D statistic between obs and sim distributions.
        """
        x = np.sort(np.asarray(obs, float))
        y = np.sort(np.asarray(sim, float))
        # build combined grid
        z = np.concatenate([x, y])
        z.sort()
        # empirical CDFs
        Fx = np.searchsorted(x, z, side='right') / x.size
        Fy = np.searchsorted(y, z, side='right') / y.size
        D = np.max(np.abs(Fx - Fy))
        return float(D)

    # ---------- NEW: Peak timing / correlation ----------
    @staticmethod
    def _top_k_indices(a: np.ndarray, k: int) -> np.ndarray:
        k = max(1, int(min(k, a.size)))
        # argpartition gets top-k unsorted; then sort descending by value and keep indices
        idx = np.argpartition(a, -k)[-k:]
        # order by time (ascending) to align sequences for lag calc
        return np.sort(idx)

    def abs_peak_lag(self, obs: np.ndarray, sim: np.ndarray, k: int = 3) -> float:
        """
        Mean absolute lag (in indices/time-steps) between top-k peaks of obs and sim.
        Peaks are taken as the k largest values in each series.
        """
        obs = np.asarray(obs, float).ravel()
        sim = np.asarray(sim, float).ravel()
        if obs.size != sim.size or obs.size == 0:
            return float('inf')
        io = self._top_k_indices(obs, k)
        is_ = self._top_k_indices(sim, k)
        # align by rank (i.e., 1st largest with 1st largest, etc.) using sorted-by-time lists
        # if lengths mismatch (edge cases), compare min length
        m = min(io.size, is_.size)
        if m == 0:
            return float('inf')
        return float(np.mean(np.abs(io[:m] - is_[:m])))

    def neg_xcorr0(self, obs: np.ndarray, sim: np.ndarray, demean: bool = True) -> float:
        """
        Negative Pearson correlation at lag 0 (so minimizing this maximizes correlation).
        """
        x = np.asarray(obs, float).ravel()
        y = np.asarray(sim, float).ravel()
        if demean:
            x = x - np.nanmean(x)
            y = y - np.nanmean(y)
        sx = np.nanstd(x); sy = np.nanstd(y)
        if sx == 0 or sy == 0:
            return 1.0  # worst case if no variability
        r = float(np.nanmean((x / sx) * (y / sy)))
        # clamp numerical jitter
        r = max(min(r, 1.0), -1.0)
        return -r

    # ---------- inertia (unchanged) ----------
    @staticmethod
    def policy_inertia(u: np.ndarray, tau: float = 0.02,
                       scale: str = "range", scale_value: float | None = None) -> float:
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
            elif m.endswith('inertia_release'):
                label = "Inertia Release (−)"
            elif m.endswith('inertia_storage'):
                label = "Inertia Storage (−)"
            elif 'fdc_mse' in m:
                label = "FDC MSE"
            elif 'fdc_ks' in m:
                label = "FDC KS"
            elif 'neg_xcorr0' in m:
                label = "Corr@Lag0 (−)"
            elif 'abs_peak_lag' in m:
                label = "Peak Lag |Δt|"
            else:
                label = base.upper()

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
    
    
    