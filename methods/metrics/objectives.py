"""
Used to calculate the objectives after Reservoir.run()
"""
import hydroeval as he

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
                 metrics):
        
        # all valid metrics
        
        valid_metrics = [
            'neg_nse', 'rmse', 
            'neg_kge', 'abs_pbias',
            ]
        
        for m in valid_metrics.copy():
            
            # log-transformed metrics
            valid_metrics.append(f'log_{m}')

            # low- and high-flow metrics
            valid_metrics.append(f'Q20_{m}') # calculated using <20th percentile obs
            valid_metrics.append(f'Q80_{m}') # calculated using >80th percentile obs
            
        # check that all metrics are valid
        for m in metrics:
            if m not in valid_metrics:
                raise ValueError(f"Invalid metric: {m}\nValid metrics: {valid_metrics}")

        self.metrics = metrics
                
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
        
        objs = []
        for metric in self.metrics:
            
            # Check if the metric is log-transformed
            log = True if 'log' in metric else False
            
            # subset the data for low- and high-flow metrics
            if 'Q20' in metric:
                obs = obs[obs < np.percentile(obs, 20)]
                sim = sim[obs < np.percentile(obs, 20)]
            elif 'Q80' in metric:
                obs = obs[obs > np.percentile(obs, 80)]
                sim = sim[obs > np.percentile(obs, 80)]
            
            # calculate the objective value based on the metric
            # add it to the list of objective outputs
            # metrics are transformed such that minimization is best
            if 'nse' in metric:
                o = self.nse(obs=obs, sim=sim, log=log)
                objs.append(-o)             
            elif 'rmse' in metric:
                o = self.rmse(obs=obs, sim=sim, log=log)
                obs.append(o)
            elif 'kge' in metric:
                o = self.kge(obs=obs, sim=sim, log=log)
                objs.append(-o[0])                
            elif 'pbias' in metric:
                o = self.pbias(obs=obs, sim=sim, log=log)
                objs.append(abs(o))
            else:
                raise ValueError(f"Invalid metric: {metric}")
        
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
        return he.evaluator(he.kge, sim, obs)[0]
    
    def pbias(self, obs, sim, log=False):
        """
        Percent Bias (PBIAS)
        """
        if log:
            return he.evaluator(he.pbias, sim, obs, transform='log')
        else:
            return he.evaluator(he.pbias, sim, obs)
    
    
    
    
    