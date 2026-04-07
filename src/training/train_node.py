"""
Training pipeline for the Neural ODE surrogate.

Architecture
------------
State vector y(t) = [M(t), C(t)]  (mass, average centre-of-mass coordinate)

Dynamics:
  dM/dt = f_ODE(M, r(t)) + NN_mass(M, C, r)   hybrid: physics + learned correction
  dC/dt = NN_com(M, C, r)                       purely learned

Training uses a simple Euler rollout over the training window (avoids the
overhead of adjoint methods while still supporting gradient flow).
"""
from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .generate_data import (
    ScenarioConfig,
    compute_center_of_mass,
    make_scalar_rad,
    run_pde_scenario,
)

# ---------------------------------------------------------------------------
# Trainable NODE function (keeps radiation schedule externally injectable)
# ---------------------------------------------------------------------------

class TrainableNODE(nn.Module):
    """
    NODE vector field with physics baseline + learnable corrections.
    Compatible with the existing NODEPhysicsFunction interface for inference.
    """

    def __init__(
        self,
        rho: float,
        beta: float,
        H_eff: float,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.rho = rho
        self.beta = beta
        self.H_eff = H_eff

        # Shared encoder for both heads
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head_mass = nn.Linear(hidden, 1)
        self.head_com = nn.Linear(hidden, 1)

        # Radiation schedule: set once before training/inference
        self._rad_fn: Callable[[float], float] | None = None

    def set_radiation(self, rad_fn: Callable[[float], float]) -> None:
        self._rad_fn = rad_fn

    def _r(self, t: float) -> float:
        return self._rad_fn(t) if self._rad_fn is not None else 0.0

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        M, C = y[0:1], y[1:2]
        r_val = self._r(float(t))
        r = torch.tensor([r_val], dtype=torch.float32)

        f_ode = self.rho * M * (1.0 - M) - self.beta * self.H_eff * r * M

        feats = self.encoder(torch.cat([M, C, r]))
        dM = f_ode.squeeze() + self.head_mass(feats).squeeze()
        dC = self.head_com(feats).squeeze()
        return torch.stack([dM, dC])


# ---------------------------------------------------------------------------
# Custom Euler rollout (avoids torchdiffeq adjoint complexity in training)
# ---------------------------------------------------------------------------

def euler_rollout(
    func: TrainableNODE,
    y0: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """
    Integrates `func` with Euler method over `times`.
    Gradients flow through all steps for backpropagation.

    :param times: 1-D tensor of increasing time values.
    :return: (len(times), 2) trajectory tensor.
    """
    ys = [y0]
    for i in range(len(times) - 1):
        t_i = times[i]
        dt = float(times[i + 1] - times[i])
        dy = func(t_i, ys[-1])
        ys.append(ys[-1] + dt * dy)
    return torch.stack(ys)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_node(
    cfg: ScenarioConfig,
    epochs: int = 500,
    lr: float = 3e-3,
    seq_len: int = 40,
    n_seqs: int = 20,
    hidden: int = 32,
    w_com: float = 0.1,
) -> TrainableNODE:
    """
    Trains the NODE on PDE ground-truth M(t) and C(t) trajectories.

    :param seq_len:  Length of each training subsequence (in steps).
    :param n_seqs:   Number of subsequences sampled per epoch.
    :param w_com:    Weight for centre-of-mass loss term.
    :return: Trained TrainableNODE (in eval mode).
    """
    print("[NODE] Running PDE to generate training trajectory…")
    times_np, masses_np, u_fields = run_pde_scenario(cfg)
    coms_np = compute_center_of_mass(u_fields, cfg)

    # Normalise total mass to spatial average U = M / L² ∈ [0, 1]
    # so that the logistic ODE term ρU(1−U) is well-posed.
    masses_norm = masses_np / (cfg.L ** 2)

    # Select training window indices
    t_mask = (times_np >= cfg.train_start) & (times_np <= cfg.train_end)
    t_train = torch.tensor(times_np[t_mask], dtype=torch.float32)
    M_train = torch.tensor(masses_norm[t_mask], dtype=torch.float32)
    C_train = torch.tensor(coms_np[t_mask], dtype=torch.float32)

    H_eff = cfg.H_eff_value()
    rad_fn = make_scalar_rad(cfg)

    func = TrainableNODE(rho=cfg.rho, beta=cfg.beta, H_eff=H_eff, hidden=hidden)
    func.set_radiation(rad_fn)

    opt = optim.Adam(func.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    rng = np.random.default_rng(0)
    n_avail = len(t_train) - seq_len

    print(f"[NODE] Training for {epochs} epochs, {n_seqs} random sequences/epoch…")
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        total_loss = torch.tensor(0.0)

        for _ in range(n_seqs):
            start = int(rng.integers(0, max(1, n_avail)))
            end = start + seq_len
            t_seq = t_train[start:end]
            M_seq = M_train[start:end]
            C_seq = C_train[start:end]

            y0 = torch.tensor([M_seq[0].item(), C_seq[0].item()], dtype=torch.float32)
            pred = euler_rollout(func, y0, t_seq)

            loss_M = F.mse_loss(pred[:, 0], M_seq)
            loss_C = F.mse_loss(pred[:, 1], C_seq)
            total_loss = total_loss + loss_M + w_com * loss_C

        total_loss = total_loss / n_seqs
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(func.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if epoch % 100 == 0:
            print(f"[NODE]   Epoch {epoch}/{epochs}  Loss={total_loss.item():.4e}")

    func.eval()
    print("[NODE] Training complete.")
    return func


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_node(func: TrainableNODE, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": func.state_dict(),
            "rho": func.rho,
            "beta": func.beta,
            "H_eff": func.H_eff,
            "hidden": func.encoder[0].out_features,
        },
        path,
    )
    print(f"[NODE] Model saved → {path}")


def load_node(path: str, cfg: ScenarioConfig) -> TrainableNODE:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    func = TrainableNODE(
        rho=ckpt["rho"],
        beta=ckpt["beta"],
        H_eff=ckpt["H_eff"],
        hidden=ckpt["hidden"],
    )
    func.load_state_dict(ckpt["state_dict"])
    func.set_radiation(make_scalar_rad(cfg))
    func.eval()
    return func
