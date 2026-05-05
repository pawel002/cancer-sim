import numpy as np
from typing import Callable, Any
from ..core.simulator import Simulator
from ..core.state import State

class ABCSampler:
    """
    Approximate Bayesian Computation (ABC) Rejection Sampler.
    Optimizes parameters theta = [rho, beta, K, D] by sampling priors,
    running the simulation, and comparing to observations.
    """
    def __init__(self, 
                 simulator_factory: Callable[[dict], Simulator],
                 initial_state_factory: Callable[[], State],
                 prior_sampler: Callable[[], dict],
                 observations: list[float],
                 obs_times: list[float]):
        """
        :param simulator_factory: Function that takes sampled params dict and returns a configured Simulator
        :param initial_state_factory: Function returning the initial State 
        :param prior_sampler: Function that returns a dictionary of sampled parameters
        :param observations: List of ground-truth summary statistics (e.g. mass)
        :param obs_times: Times corresponding to observations
        """
        self.simulator_factory = simulator_factory
        self.initial_state_factory = initial_state_factory
        self.prior_sampler = prior_sampler
        self.observations = np.array(observations)
        self.obs_times = np.array(obs_times)

    def sample(self, num_samples: int = 1000, keep_percentile: float = 15.0) -> list[dict]:
        results = []
        errors = []
        
        for _ in range(num_samples):
            params = self.prior_sampler()
            sim = self.simulator_factory(params)
            initial_state = self.initial_state_factory()
            
            # Run simulation
            history = sim.run(initial_state)
            
            # Extract simulated mass at observation times
            # Note: For simplicity, we assume history index matches time directly or we interpolate.
            # Here we just find the closest time steps.
            sim_times = np.array([state.t for state in history]) if hasattr(history[0], 't') else np.arange(len(history)) * sim.dt
            
            sim_obs = []
            for t in self.obs_times:
                idx = np.abs(sim_times - t).argmin()
                sim_obs.append(history[idx].mass)
                
            sim_obs = np.array(sim_obs)
            
            # Calculate RMSE
            rmse = np.linalg.norm(sim_obs - self.observations)
            errors.append(rmse)
            results.append((params, rmse))
            
        # Keep top percentile
        threshold = np.percentile(errors, keep_percentile)
        accepted_params = [res[0] for res in results if res[1] <= threshold]
        
        return accepted_params
