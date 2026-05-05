"""
PINN (Physics-Informed Neural Network) model and engine.
The model continuously approximates u_θ(t, x₁, x₂) ≈ u(t, x) for a fixed scenario.
At inference time the engine reconstructs the full 2-D field at each time step.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

from ...core.state import State
from ..base import BaseEngine


class PINNModel(nn.Module):
    """
    Continuous surrogate u_θ : (t, x₁, x₂) → [0, 1].
    Uses Tanh hidden activations and a Sigmoid output to stay in [0, 1].
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        :param pts: (N, 3) tensor with columns [t, x₁, x₂] already normalised.
        :return: (N, 1) predicted densities in [0, 1].
        """
        return self.net(pts)


class PINNEngine(BaseEngine):
    """
    Surrogate engine driven by a trained PINNModel.
    Evaluates u_θ(t, x) on the full spatial grid at every requested time step.
    Unlike step-based engines, the PINN is a direct function of time, so the
    engine ignores `current_state` and evaluates at the given `t`.
    """

    def __init__(
        self,
        model: PINNModel,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        dx: float,
        t_scale: float = 1.0,
        x_scale: float = 1.0,
        radiation_protocol: Callable[[float], float] | None = None,
    ) -> None:
        """
        :param model: Trained PINNModel.
        :param grid_x: 1-D array of x-coordinates (length N).
        :param grid_y: 1-D array of y-coordinates (length N).
        :param dx: Grid spacing.
        :param t_scale: Normalisation scale for time (divide t by this).
        :param x_scale: Normalisation scale for space (divide x by this).
        :param radiation_protocol: Stored for metadata / optional use.
        """
        self.model = model
        X, Y = np.meshgrid(grid_x, grid_y)
        self._x1_flat = torch.tensor(X.ravel() / x_scale, dtype=torch.float32).unsqueeze(1)
        self._x2_flat = torch.tensor(Y.ravel() / x_scale, dtype=torch.float32).unsqueeze(1)
        self._N = len(grid_x)
        self.dx = dx
        self.t_scale = t_scale
        self.radiation_protocol = radiation_protocol
        self.model.eval()

    def step(self, current_state: State, t: float, dt: float) -> State:
        n_pts = self._x1_flat.shape[0]
        t_col = torch.full((n_pts, 1), t / self.t_scale, dtype=torch.float32)
        pts = torch.cat([t_col, self._x1_flat, self._x2_flat], dim=1)

        with torch.no_grad():
            u_flat = self.model(pts).squeeze(1).numpy()

        u_next = u_flat.reshape(self._N, self._N)
        mass = float(np.sum(u_next) * self.dx ** 2)
        return State(u=u_next, mass=mass)
