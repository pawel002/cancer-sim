import numpy as np
from scipy import ndimage
from typing import Callable
from .base import BaseEngine
from ..core.state import State

class PDEEngine(BaseEngine):
    """
    1D/2D Partial Differential Equation engine using Finite Differences.
    Ground truth model.
    """
    
    def __init__(self, 
                 D: float, 
                 rho: float, 
                 K: float, 
                 beta: float, 
                 dx: float, 
                 radiation_protocol: Callable[[tuple, float], np.ndarray] | None = None, 
                 hypoxia_map: np.ndarray | None = None):
        """
        :param D: Diffusion coefficient
        :param rho: Proliferation rate
        :param K: Carrying capacity
        :param beta: Radiosensitivity
        :param dx: Spatial resolution
        :param radiation_protocol: Function that takes (shape, t) and returns R(x,t)
        :param hypoxia_map: Static hypoxia map H(x)
        """
        self.D = D
        self.rho = rho
        self.K = K
        self.beta = beta
        self.dx = dx
        self.inv_dx2 = 1.0 / (self.dx ** 2)
        
        self.radiation_protocol = radiation_protocol
        self.hypoxia_map = hypoxia_map

    def step(self, current_state: State, t: float, dt: float) -> State:
        u = current_state.u.copy()
        
        # Calculate Laplacian with zero boundary conditions (Dirichlet)
        lap = ndimage.laplace(u, mode='constant', cval=0.0) * self.inv_dx2
        
        # Hypoxia term
        H = self.hypoxia_map if self.hypoxia_map is not None else 1.0
        
        # Radiation term
        R = self.radiation_protocol(u.shape, t) if self.radiation_protocol else 0.0
        
        # PDE update
        du = (self.D * lap) + (self.rho * u * (1.0 - u / self.K)) - (self.beta * R * H * u)
        
        u_next = u + du * dt
        
        # Enforce Dirichlet BCs explicitly just in case
        if u_next.ndim == 2:
            u_next[0, :] = u_next[-1, :] = u_next[:, 0] = u_next[:, -1] = 0.0
        elif u_next.ndim == 3:
            u_next[0, :, :] = u_next[-1, :, :] = u_next[:, 0, :] = u_next[:, -1, :] = u_next[:, :, 0] = u_next[:, :, -1] = 0.0
            
        dim = u_next.ndim
        mass = float(np.sum(u_next) * (self.dx ** dim))
        
        return State(u=u_next, mass=mass)
