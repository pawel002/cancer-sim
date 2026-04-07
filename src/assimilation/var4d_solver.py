import numpy as np
from scipy.optimize import minimize
from typing import Callable, Any
from ..core.simulator import Simulator
from ..core.state import State

class Var4DSolver:
    """
    4D-Var Data Assimilation using L-BFGS-B optimization.
    """
    def __init__(self, 
                 simulator_factory: Callable[[list[float]], Simulator],
                 initial_state_factory: Callable[[], State],
                 observations: list[float],
                 obs_times: list[float],
                 param_bounds: list[tuple[float, float]]):
        """
        :param simulator_factory: Returns Simulator from a flat list/array of parameters [rho, beta, K, D]
        :param initial_state_factory: Returns initial state
        :param observations: Ground truth observations (e.g. mass)
        :param obs_times: Times of observations
        :param param_bounds: Bounds for optimization (min, max) for each parameter
        """
        self.simulator_factory = simulator_factory
        self.initial_state_factory = initial_state_factory
        self.observations = np.array(observations)
        self.obs_times = np.array(obs_times)
        self.param_bounds = param_bounds

    def _cost_functional(self, theta: np.ndarray) -> float:
        """
        J(theta) forward pass and loss evaluation.
        """
        sim = self.simulator_factory(theta)
        initial_state = self.initial_state_factory()
        
        history = sim.run(initial_state)
        
        sim_times = np.arange(len(history)) * sim.dt
        
        sim_obs = []
        for t in self.obs_times:
            idx = np.abs(sim_times - t).argmin()
            sim_obs.append(history[idx].mass)
            
        sim_obs = np.array(sim_obs)
        
        # Simple Sum of Squared Errors
        loss = np.sum((sim_obs - self.observations)**2)
        return loss

    def optimize(self, initial_guess: list[float]) -> dict:
        """
        Runs the L-BFGS-B optimizer.
        """
        res = minimize(
            self._cost_functional,
            x0=np.array(initial_guess),
            method='L-BFGS-B',
            bounds=self.param_bounds,
            options={'disp': True}
        )
        
        return {
            'success': res.success,
            'optimal_params': res.x,
            'final_loss': res.fun,
            'message': res.message
        }
