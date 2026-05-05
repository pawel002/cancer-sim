from typing import Callable
from .base import BaseEngine
from ..core.state import State

class ODEEngine(BaseEngine):
    """
    Spatially averaged surrogate ODE Engine.
    Uses a simple forward Euler step.
    """
    
    def __init__(self, 
                 rho: float, 
                 beta: float, 
                 H_eff: float = 1.0, 
                 radiation_protocol: Callable[[float], float] | None = None):
        """
        :param rho: Proliferation rate
        :param beta: Radiosensitivity
        :param H_eff: Effective global hypoxia
        :param radiation_protocol: Function that takes (t) and returns scalar dose r(t)
        """
        self.rho = rho
        self.beta = beta
        self.H_eff = H_eff
        self.radiation_protocol = radiation_protocol

    def step(self, current_state: State, t: float, dt: float) -> State:
        # Here u represents the scalar mass/density U(t)
        U = current_state.u
        
        r = self.radiation_protocol(t) if self.radiation_protocol else 0.0
        
        dU_dt = self.rho * U * (1.0 - U) - (self.beta * self.H_eff * r * U)
        U_next = U + dt * dU_dt
        
        return State(u=U_next, mass=U_next)
