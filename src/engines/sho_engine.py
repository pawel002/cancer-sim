import numpy as np
from scipy import ndimage
from typing import Callable
from .base import BaseEngine
from ..core.state import State

class SHOEngine(BaseEngine):
    """
    S-H-O (Normoxia, Hypoxia, Oxygen) Engine.
    Implements coupled reaction-diffusion system.
    """
    
    def __init__(self, 
                 D: float, 
                 rho: float, 
                 K: float, 
                 beta: float, 
                 dx: float, 
                 mu_0: float,
                 o_h: float,
                 radiation_protocol: Callable[[tuple, float], np.ndarray] | None = None):
        self.D_S = D
        self.D_H = D
        self.D_O = D * 10.0  # Oxygen diffuses faster
        
        self.rho = rho
        self.K = K
        self.beta = beta
        self.dx = dx
        self.inv_dx2 = 1.0 / (self.dx ** 2)
        
        self.mu_0 = mu_0
        self.o_h = o_h
        
        self.radiation_protocol = radiation_protocol

    def _mu(self, o: np.ndarray) -> np.ndarray:
        """Transition function mu(O)."""
        return self.mu_0 * np.maximum(0.0, 1.0 - o / self.o_h)

    def step(self, current_state: State, t: float, dt: float) -> State:
        # Require state to have these initialized
        S = current_state.normoxia.copy()
        H = current_state.hypoxia.copy()
        O = current_state.oxygen.copy()
        
        lap_S = ndimage.laplace(S, mode='constant', cval=0.0) * self.inv_dx2
        lap_H = ndimage.laplace(H, mode='constant', cval=0.0) * self.inv_dx2
        lap_O = ndimage.laplace(O, mode='constant', cval=1.0) * self.inv_dx2
        
        u = S + H
        mu_rate = self._mu(O)
        R = self.radiation_protocol(S.shape, t) if self.radiation_protocol else 0.0
        
        dS = (self.D_S * lap_S) + (self.rho * S * (1.0 - u / self.K)) - (mu_rate * S) - (self.beta * R * S)
        dH = (self.D_H * lap_H) + (mu_rate * S) - (self.beta * R * H * 0.5)
        
        q = 0.1 
        dO = (self.D_O * lap_O) - (q * u * O)
        
        S_next = S + dt * dS
        H_next = H + dt * dH
        O_next = O + dt * dO
        
        u_next = S_next + H_next
        dim = u_next.ndim
        mass = float(np.sum(u_next) * (self.dx ** dim))
        
        return State(u=u_next, mass=mass, normoxia=S_next, hypoxia=H_next, oxygen=O_next)
