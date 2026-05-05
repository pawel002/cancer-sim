import torch
import torch.nn as nn
from typing import Callable
from ..base import BaseEngine
from ...core.state import State

class SuperNetEngine(BaseEngine):
    """
    SuperNet Engine. Fast rollout using Hybrid strategy.
    Equation: M_{n+1} = M_n + dt * (f_ODE + g_phi(s_n))
    """
    def __init__(self, 
                 rho: float, 
                 beta: float, 
                 H_eff: float, 
                 g_phi: nn.Module,
                 radiation_protocol: Callable[[float], float] | None = None):
        """
        :param g_phi: Neural network correcting the ODE physics.
        """
        self.rho = rho
        self.beta = beta
        self.H_eff = H_eff
        self.g_phi = g_phi
        self.radiation_protocol = radiation_protocol
        
        self.g_phi.eval()

    def step(self, current_state: State, t: float, dt: float) -> State:
        M_n = current_state.mass
        r_n = self.radiation_protocol(t) if self.radiation_protocol else 0.0
        
        f_ODE = self.rho * M_n * (1.0 - M_n) - self.beta * self.H_eff * r_n * M_n
        
        # Features s_n
        s_n = current_state.features
        if s_n is None:
            # Fallback to simple features
            s_n = torch.tensor([[M_n, r_n]], dtype=torch.float32)
        else:
            s_n = torch.tensor([s_n], dtype=torch.float32)
        
        with torch.no_grad():
            residual = self.g_phi(s_n).item()
            
        M_next = M_n + dt * (f_ODE + residual)
        
        return State(u=M_next, mass=M_next, features=current_state.features)
