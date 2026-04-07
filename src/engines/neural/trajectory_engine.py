"""
Trajectory Predictor — a conditional surrogate that predicts the full
normalised-mass trajectory U(t) in a single batched forward pass.

Architecture
------------
Input:  (ρ, β, D, amplitude, t_start, t_end, t)  ∈ ℝ⁷
Output: U(t) = M(t)/L²  ∈ [0, 1]

The model is trained across thousands of PDE simulations with randomised
physical parameters so that at inference time no time-stepping loop is
required — the entire trajectory is obtained in one call.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TrajectoryNet(nn.Module):
    """Conditional surrogate f_θ(params, t) → U(t)."""

    def __init__(self, n_params: int = 6, hidden: int = 256, n_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(n_params + 1, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, n_params+1)  last column is time t.
        :return:  (B,) predicted normalised mass.
        """
        return self.net(x).squeeze(-1)

    def predict_trajectory(
        self,
        params: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the full trajectory for a single parameter vector.

        :param params: (n_params,) — [ρ, β, D, amplitude, t_start, t_end].
        :param times:  (N_t,) — query times.
        :return:       (N_t,) — predicted U(t) values.
        """
        n_t = times.shape[0]
        p = params.unsqueeze(0).expand(n_t, -1)        # (N_t, n_params)
        t = times.unsqueeze(1)                          # (N_t, 1)
        x = torch.cat([p, t], dim=1)                    # (N_t, n_params+1)
        return self.forward(x)

    @torch.no_grad()
    def predict_trajectory_np(
        self,
        params: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """NumPy convenience wrapper."""
        p = torch.tensor(params, dtype=torch.float32)
        t = torch.tensor(times, dtype=torch.float32)
        return self.predict_trajectory(p, t).numpy()
