import torch
import torch.nn as nn
from typing import Callable
from ..base import BaseEngine
from ...core.state import State

class MLPModel(nn.Module):
    """
    Standard MLP Architecture as specified.
    Linear(5, 32) -> ReLU -> Linear(32, 1)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class MLPEngine(BaseEngine):
    """
    MLP Time-Stepper Surrogate.
    Steps mass prediction completely using the ML model.
    """
    def __init__(self, 
                 model: nn.Module, 
                 rho: float, 
                 beta: float, 
                 radiation_protocol: Callable[[float], float] | None = None):
        self.model = model
        self.rho = rho
        self.beta = beta
        self.radiation_protocol = radiation_protocol
        
        self.model.eval()

    def step(self, current_state: State, t: float, dt: float) -> State:
        U_n = current_state.mass
        r_n = self.radiation_protocol(t) if self.radiation_protocol else 0.0
        
        # Inputs: [U_n, r_n, dt, rho, beta]
        x = torch.tensor([[U_n, r_n, dt, self.rho, self.beta]], dtype=torch.float32)
        
        with torch.no_grad():
            U_next = self.model(x).item()
            
        return State(u=U_next, mass=U_next)
