"""
Training pipeline for the PINN (Physics-Informed Neural Network) surrogate.

Loss function
-------------
L = w_pde  * L_pde   (PDE residual on interior collocation points)
  + w_ic   * L_ic    (initial condition)
  + w_bc   * L_bc    (Dirichlet BCs = 0 at domain boundary)
  + w_data * L_data  (match total-mass trajectory from PDE ground truth)

Inputs are normalised to [0, 1] before entering the network:
    t_norm = t / T_max,   x_norm = x / L
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.interpolate import RegularGridInterpolator

from ..engines.neural.pinn_engine import PINNModel
from .generate_data import (
    ScenarioConfig,
    make_gaussian_ic,
    make_hypoxia_map,
    make_scalar_rad,
    run_pde_scenario,
)

# ---------------------------------------------------------------------------
# Helper: evaluate hypoxia map H(x) at arbitrary (x1, x2) points
# ---------------------------------------------------------------------------

def _make_h_interpolator(cfg: ScenarioConfig) -> RegularGridInterpolator:
    H = make_hypoxia_map(cfg)
    x = cfg.grid_x
    # RegularGridInterpolator uses (row, col) ~ (y, x) order
    return RegularGridInterpolator((x, x), H, method="linear", bounds_error=False, fill_value=0.0)


# ---------------------------------------------------------------------------
# PDE residual (computed via autograd)
# ---------------------------------------------------------------------------

def _pde_residual_loss(
    model: PINNModel,
    t_c: torch.Tensor,    # (N, 1), requires_grad=True, normalised
    x1_c: torch.Tensor,   # (N, 1), requires_grad=True, normalised
    x2_c: torch.Tensor,   # (N, 1), requires_grad=True, normalised
    r_c: torch.Tensor,    # (N, 1), no grad, raw radiation values
    H_c: torch.Tensor,    # (N, 1), no grad, hypoxia values
    D: float,
    rho: float,
    K: float,
    beta: float,
    T_max: float,
    L: float,
) -> torch.Tensor:
    """Returns mean-squared PDE residual."""
    pts = torch.cat([t_c, x1_c, x2_c], dim=1)
    u = model(pts)  # (N, 1)

    # Time derivative (normalised → physical: ∂u/∂t = (1/T_max) * ∂u/∂t_norm)
    u_t_n = torch.autograd.grad(u.sum(), t_c, create_graph=True)[0]
    u_t = u_t_n / T_max

    # Spatial second derivatives (normalised → physical: Δu = (1/L)^2 * Δu_norm)
    u_x1_n = torch.autograd.grad(u.sum(), x1_c, create_graph=True)[0]
    u_x1x1_n = torch.autograd.grad(u_x1_n.sum(), x1_c, create_graph=True)[0]
    u_x2_n = torch.autograd.grad(u.sum(), x2_c, create_graph=True)[0]
    u_x2x2_n = torch.autograd.grad(u_x2_n.sum(), x2_c, create_graph=True)[0]
    lap = (u_x1x1_n + u_x2x2_n) / (L ** 2)

    # PDE residual: ∂u/∂t - D∇²u - ρu(1−u/K) + βr(t)H(x)u = 0
    R = u_t - D * lap - rho * u * (1.0 - u / K) + beta * r_c * H_c * u
    return torch.mean(R ** 2)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_pinn(
    cfg: ScenarioConfig,
    epochs: int = 2000,
    n_coll: int = 2000,
    n_ic: int = 500,
    n_bc: int = 400,
    n_data_times: int = 30,
    lr: float = 1e-3,
    w_pde: float = 1.0,
    w_ic: float = 10.0,
    w_bc: float = 5.0,
    w_data: float = 20.0,
    hidden_dim: int = 64,
    n_layers: int = 5,
    seed: int = 0,
) -> PINNModel:  # noqa: PLR0913
    """
    Trains a PINNModel for the canonical scenario defined by `cfg`.

    :return: Trained PINNModel (eval mode, normalised inputs).
    """
    print("[PINN] Running PDE ground-truth simulation…")
    times_np, masses_np, _ = run_pde_scenario(cfg)

    u0 = make_gaussian_ic(cfg)
    H_interp = _make_h_interpolator(cfg)
    rad_fn = make_scalar_rad(cfg)

    # Select training window
    t_mask = (times_np >= cfg.train_start) & (times_np <= cfg.train_end)
    times_train = times_np[t_mask]
    masses_train = masses_np[t_mask]

    # Pre-select times for the data loss
    data_idx = np.linspace(0, len(times_train) - 1, n_data_times, dtype=int)
    t_data = times_train[data_idx]
    M_data = masses_train[data_idx]

    # Normalisation scales
    T_sc = cfg.T_max
    L_sc = cfg.L

    # Pre-build grid in normalised coordinates for mass integration
    gx = cfg.grid_x / L_sc
    X_g, Y_g = np.meshgrid(gx, gx)
    x1_grid = torch.tensor(X_g.ravel(), dtype=torch.float32).unsqueeze(1)
    x2_grid = torch.tensor(Y_g.ravel(), dtype=torch.float32).unsqueeze(1)

    # IC: pre-interpolate u0 at random interior points
    rng = np.random.default_rng(seed)
    x1_ic_np = rng.uniform(0.0, 1.0, n_ic)
    x2_ic_np = rng.uniform(0.0, 1.0, n_ic)
    u0_interp = RegularGridInterpolator(
        (cfg.grid_x / L_sc, cfg.grid_x / L_sc), u0,
        method="linear", bounds_error=False, fill_value=0.0
    )
    u0_vals = u0_interp(np.stack([x1_ic_np, x2_ic_np], axis=1))

    pts_ic_const = torch.zeros(n_ic, 3, dtype=torch.float32)
    pts_ic_const[:, 1] = torch.tensor(x1_ic_np, dtype=torch.float32)
    pts_ic_const[:, 2] = torch.tensor(x2_ic_np, dtype=torch.float32)
    u0_target = torch.tensor(u0_vals, dtype=torch.float32).unsqueeze(1)

    # BC: fixed boundary points (quarter per side), random times
    def _bc_points(n: int) -> torch.Tensor:
        n4 = n // 4
        t_b = rng.uniform(cfg.train_start / T_sc, cfg.train_end / T_sc, n)
        # left, right, bottom, top
        x1_b = np.concatenate([
            np.zeros(n4), np.ones(n4),
            rng.uniform(0, 1, n4), rng.uniform(0, 1, n - 3 * n4)
        ])
        x2_b = np.concatenate([
            rng.uniform(0, 1, n4), rng.uniform(0, 1, n4),
            np.zeros(n4), np.ones(n - 3 * n4)
        ])
        return torch.tensor(
            np.stack([t_b, x1_b, x2_b], axis=1), dtype=torch.float32
        )

    model = PINNModel(hidden_dim=hidden_dim, n_layers=n_layers)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[1000, 1500], gamma=0.3)

    print(f"[PINN] Training for {epochs} epochs…")
    for epoch in range(1, epochs + 1):
        opt.zero_grad()

        # ── IC loss ────────────────────────────────────────────────────────
        u_ic_pred = model(pts_ic_const)
        loss_ic = F.mse_loss(u_ic_pred, u0_target)

        # ── BC loss ────────────────────────────────────────────────────────
        pts_bc = _bc_points(n_bc)
        u_bc = model(pts_bc)
        loss_bc = torch.mean(u_bc ** 2)

        # ── PDE residual loss ──────────────────────────────────────────────
        t_c_np = rng.uniform(cfg.train_start / T_sc, cfg.train_end / T_sc, n_coll)
        x1_c_np = rng.uniform(0.0, 1.0, n_coll)
        x2_c_np = rng.uniform(0.0, 1.0, n_coll)

        # Physical coordinates for r(t) and H(x)
        t_phys = t_c_np * T_sc
        x1_phys = x1_c_np * L_sc
        x2_phys = x2_c_np * L_sc
        r_vals = np.array([rad_fn(ti) for ti in t_phys], dtype=np.float32)
        H_pts = H_interp(np.stack([x1_phys, x2_phys], axis=1)).astype(np.float32)

        r_c = torch.tensor(r_vals, dtype=torch.float32).unsqueeze(1)
        H_c = torch.tensor(H_pts, dtype=torch.float32).unsqueeze(1)

        t_c = torch.tensor(t_c_np, dtype=torch.float32, requires_grad=True).unsqueeze(1)
        x1_c = torch.tensor(x1_c_np, dtype=torch.float32, requires_grad=True).unsqueeze(1)
        x2_c = torch.tensor(x2_c_np, dtype=torch.float32, requires_grad=True).unsqueeze(1)

        loss_pde = _pde_residual_loss(
            model, t_c, x1_c, x2_c, r_c, H_c,
            cfg.D, cfg.rho, cfg.K, cfg.beta, T_sc, L_sc
        )

        # ── Data (mass trajectory) loss ────────────────────────────────────
        loss_data = torch.tensor(0.0)
        for t_i, M_i in zip(t_data, M_data, strict=True):
            t_col = torch.full((cfg.N ** 2, 1), t_i / T_sc, dtype=torch.float32)
            pts_g = torch.cat([t_col, x1_grid, x2_grid], dim=1)
            u_g = model(pts_g)
            M_pred = u_g.sum() * cfg.dx ** 2
            loss_data = loss_data + (M_pred - float(M_i)) ** 2
        loss_data = loss_data / len(t_data)

        loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc + w_data * loss_data
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if epoch % 200 == 0:
            print(
                f"[PINN]   Epoch {epoch}/{epochs}  "
                f"pde={loss_pde.item():.3e}  ic={loss_ic.item():.3e}  "
                f"bc={loss_bc.item():.3e}  data={loss_data.item():.3e}"
            )

    model.eval()
    print("[PINN] Training complete.")
    return model


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_pinn(model: PINNModel, path: str, cfg: ScenarioConfig, n_layers: int = 5) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": model.net[0].out_features,  # type: ignore[union-attr]
            "n_layers": n_layers,
            "T_max": cfg.T_max,
            "L": cfg.L,
        },
        path,
    )
    print(f"[PINN] Model saved → {path}")


def load_pinn(path: str) -> tuple[PINNModel, float, float]:
    """Returns (model, T_max, L) where T_max and L are normalisation scales."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PINNModel(
        hidden_dim=ckpt.get("hidden_dim", 64),
        n_layers=ckpt.get("n_layers", 5),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, float(ckpt["T_max"]), float(ckpt["L"])
