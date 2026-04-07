"""
Training pipeline for the SuperNet surrogate.

Strategy (PINN-as-teacher)
--------------------------
1. Roll out the trained PINN over the training window to get M_pinn(t_n).
2. Compute the target residual that is missing from the plain ODE:
       v_n = ( M_pinn(t_{n+1}) − M_pinn(t_n) ) / dt  −  f_ODE(M_pinn(t_n), r_n)
3. Train g_φ : [M_n, r_n] → v_n by minimising ∑‖g_φ(s_n) − v_n‖².
4. The trained g_φ is directly usable by SuperNetEngine.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..engines.neural.pinn_engine import PINNModel
from .generate_data import ScenarioConfig, make_scalar_rad

# ---------------------------------------------------------------------------
# Residual network g_φ
# ---------------------------------------------------------------------------

class ResidualNet(nn.Module):
    """
    Small MLP that learns the ODE correction term:
        g_φ : [M_n, r_n] → scalar residual
    """
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# PINN roll-out helpers
# ---------------------------------------------------------------------------

def _pinn_rollout_masses(
    model: PINNModel,
    cfg: ScenarioConfig,
    T_sc: float,
    L_sc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates PINN on the 2-D grid at every training time-step, integrates
    spatially to produce total mass M_pinn(t), then normalises to spatial
    average U_pinn = M / L² ∈ [0, 1] so residuals are computed in the same
    normalised units as the ODE formula.

    Returns (times, u_avg) where u_avg is the normalised spatial average.
    """
    n_steps_train = int((cfg.train_end - cfg.train_start) / cfg.dt)
    times = np.array([cfg.train_start + i * cfg.dt for i in range(n_steps_train + 1)])

    # Grid in normalised coords
    gx = cfg.grid_x / L_sc
    X_g, Y_g = np.meshgrid(gx, gx)
    x1_g = torch.tensor(X_g.ravel(), dtype=torch.float32).unsqueeze(1)
    x2_g = torch.tensor(Y_g.ravel(), dtype=torch.float32).unsqueeze(1)
    n_pts = x1_g.shape[0]

    u_avg = np.empty(len(times))
    model.eval()
    with torch.no_grad():
        for i, t_i in enumerate(times):
            t_col = torch.full((n_pts, 1), t_i / T_sc, dtype=torch.float32)
            pts = torch.cat([t_col, x1_g, x2_g], dim=1)
            u_flat = model(pts).squeeze(1).numpy()
            total_mass = float(np.sum(u_flat) * cfg.dx ** 2)
            u_avg[i] = total_mass / (cfg.L ** 2)   # normalise to [0, 1]

    return times, u_avg


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_supernet(
    cfg: ScenarioConfig,
    pinn_model: PINNModel,
    T_sc: float,
    L_sc: float,
    epochs: int = 500,
    lr: float = 1e-3,
    hidden: int = 32,
) -> ResidualNet:
    """
    Trains ResidualNet using PINN trajectories as supervisor.

    :param pinn_model: Trained PINNModel (eval mode).
    :param T_sc: Time normalisation scale used during PINN training.
    :param L_sc: Space normalisation scale used during PINN training.
    :return: Trained ResidualNet (eval mode).
    """
    print("[SuperNet] Computing PINN teacher trajectories…")
    times, M_pinn = _pinn_rollout_masses(pinn_model, cfg, T_sc, L_sc)

    rad_fn = make_scalar_rad(cfg)
    H_eff = cfg.H_eff_value()
    dt = cfg.dt

    # Build supervised dataset from teacher residuals
    M_n = M_pinn[:-1]
    M_np1 = M_pinn[1:]
    r_n = np.array([rad_fn(t) for t in times[:-1]], dtype=np.float32)

    f_ode = cfg.rho * M_n * (1.0 - M_n) - cfg.beta * H_eff * r_n * M_n
    target_v = (M_np1 - M_n) / dt - f_ode  # residual to learn

    # Torch tensors
    X_t = torch.tensor(
        np.stack([M_n, r_n], axis=1), dtype=torch.float32
    )
    y_t = torch.tensor(target_v, dtype=torch.float32).unsqueeze(1)

    g_phi = ResidualNet(hidden=hidden)
    opt = optim.Adam(g_phi.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = len(X_t)
    print(f"[SuperNet] Training on {n} residual samples for {epochs} epochs…")
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        pred = g_phi(X_t)
        loss = F.mse_loss(pred, y_t)
        loss.backward()
        opt.step()
        scheduler.step()
        if epoch % 100 == 0:
            print(f"[SuperNet]   Epoch {epoch}/{epochs}  Loss={loss.item():.4e}")

    g_phi.eval()
    print("[SuperNet] Training complete.")
    return g_phi


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_supernet(g_phi: ResidualNet, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"state_dict": g_phi.state_dict(), "hidden": g_phi.net[0].out_features},
        path,
    )
    print(f"[SuperNet] Model saved → {path}")


def load_supernet(path: str) -> ResidualNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    g_phi = ResidualNet(hidden=ckpt["hidden"])
    g_phi.load_state_dict(ckpt["state_dict"])
    g_phi.eval()
    return g_phi
