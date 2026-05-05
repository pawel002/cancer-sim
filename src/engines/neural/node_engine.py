import torch
import torch.nn as nn
try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None

from typing import Callable
from ..base import BaseEngine
from ...core.state import State

class NODEPhysicsFunction(nn.Module):
    def __init__(self, rho: float, beta: float, H_eff: float, nn_correction: nn.Module):
        super().__init__()
        self.rho = rho
        self.beta = beta
        self.H_eff = H_eff
        self.nn_correction = nn_correction
        self.r = 0.0
        
    def forward(self, t, y):
        # y = [M, C]
        M = y[0]
        C = y[1]
        
        f_ODE_M = self.rho * M * (1.0 - M) - self.beta * self.H_eff * self.r * M
        
        # Features [M, C, r]
        # In a real batched training scenario, dimensionalities are handled properly.
        # Here we deal with scalar values for ODE integration string.
        state_tensor = torch.stack([M, C, torch.tensor(self.r, dtype=torch.float32)])
        correction = self.nn_correction(state_tensor)
        
        dM = f_ODE_M + correction[0]
        dC = correction[1]
        
        return torch.stack([dM, dC])

class NODEEngine(BaseEngine):
    """
    Neural ODE Tracking Mass and Center of Mass.
    """
    def __init__(self, 
                 physics_func: NODEPhysicsFunction, 
                 radiation_protocol: Callable[[float], float] | None = None):
        if odeint is None:
            raise ImportError("torchdiffeq is required for NODEEngine")
            
        self.physics_func = physics_func
        self.radiation_protocol = radiation_protocol

        self.physics_func.eval()

    def step(self, current_state: State, t: float, dt: float) -> State:
        M = current_state.mass
        C = current_state.center_of_mass if current_state.center_of_mass is not None else 0.0
        
        y0 = torch.tensor([M, C], dtype=torch.float32)
        
        if self.radiation_protocol:
            self.physics_func.r = self.radiation_protocol(t)
            
        t_span = torch.tensor([0.0, dt], dtype=torch.float32)
        
        with torch.no_grad():
            sol = odeint(self.physics_func, y0, t_span, method='euler')
            
        y_next = sol[-1]
        M_next = y_next[0].item()
        C_next = y_next[1].item()
        
        return State(u=M_next, mass=M_next, center_of_mass=C_next)
