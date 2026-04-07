"""
Training pipeline for the Trajectory Predictor surrogate.

Workflow
-------
1.  Sample N parameter vectors from the design space.
2.  For each, run the full PDE simulation and record the normalised mass
    trajectory U(t) = M(t) / L².
3.  Build a dataset of (params ⊕ t_i , U_i) pairs.
4.  Train the TrajectoryNet on this dataset.

The trained model can then predict *any* trajectory for *any* parameter
combination in a single batched forward pass (no time-stepping loop).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..core.protocols import uniform_spatial_radiation
from ..core.simulator import Simulator
from ..core.state import State
from ..engines.neural.trajectory_engine import TrajectoryNet
from ..engines.pde_engine import PDEEngine
from .generate_data import ScenarioConfig

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _run_single_pde(
    cfg: ScenarioConfig,
    rho: float,
    beta: float,
    D: float,
    amplitude: float,
    t_rad_start: float,
    t_rad_end: float,
) -> np.ndarray:
    """Run one PDE simulation and return the normalised mass trajectory."""
    x = cfg.grid_x
    X, Y = np.meshgrid(x, x)
    center = cfg.L / 2.0
    r_sq = (X - center) ** 2 + (Y - center) ** 2
    sigma = 1.5

    u0 = amplitude * np.exp(-r_sq / (2.0 * sigma ** 2))
    u0 = np.clip(u0, 0.0, 1.0)
    u0[0, :] = u0[-1, :] = u0[:, 0] = u0[:, -1] = 0.0

    H_map = 0.3 + 0.7 * np.exp(-r_sq / (cfg.L / 2.0) ** 2)

    def rad_spatial(shape: tuple[int, ...], t: float) -> np.ndarray:
        return uniform_spatial_radiation(shape, t, t_rad_start, t_rad_end)

    engine = PDEEngine(
        D=D, rho=rho, K=cfg.K, beta=beta, dx=cfg.dx,
        radiation_protocol=rad_spatial, hypoxia_map=H_map,
    )
    mass0 = float(np.sum(u0) * cfg.dx ** 2)
    state0 = State(u=u0, mass=mass0)
    history = Simulator(engine, cfg.dt, cfg.T_max).run(state0)

    masses = np.array([s.mass for s in history])
    return masses / (cfg.L ** 2)   # normalised spatial average


def generate_dataset(
    cfg: ScenarioConfig,
    n_samples: int = 3000,
    n_time_points: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate PDE training data with randomised parameters.

    Returns
    -------
    params      : (n_samples, 6)  [ρ, β, D, amplitude, t_s, t_e]
    trajectories: (n_samples, n_time_points)  normalised mass at query times
    query_times : (n_time_points,)  the time coordinates
    """
    rng = np.random.default_rng(seed)

    # Time query points (sub-sampled from full trajectory)
    n_full = cfg.n_steps + 1
    full_times = np.linspace(0.0, cfg.T_max, n_full)
    idx = np.linspace(0, n_full - 1, n_time_points, dtype=int)
    query_times = full_times[idx]

    params = np.empty((n_samples, 6), dtype=np.float32)
    trajectories = np.empty((n_samples, n_time_points), dtype=np.float32)

    print(f"[TrajData] Generating {n_samples} PDE simulations …")
    t0 = time.perf_counter()

    for i in range(n_samples):
        rho = float(rng.uniform(0.05, 0.6))
        beta = float(rng.uniform(0.1, 2.0))
        D = float(rng.uniform(0.002, 0.08))
        amplitude = float(rng.uniform(0.1, 1.0))
        t_s = float(rng.uniform(3.0, 20.0))
        t_e = float(t_s + rng.uniform(3.0, min(25.0, cfg.T_max - t_s - 1.0)))

        full_traj = _run_single_pde(cfg, rho, beta, D, amplitude, t_s, t_e)
        params[i] = [rho, beta, D, amplitude, t_s, t_e]
        trajectories[i] = full_traj[idx]

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (n_samples - i - 1)
            print(f"[TrajData]   {i+1}/{n_samples}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                  flush=True)

    elapsed = time.perf_counter() - t0
    print(f"[TrajData] Done in {elapsed:.1f}s  "
          f"({elapsed/n_samples*1000:.1f} ms/sim)")
    return params, trajectories, query_times


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_trajectory(
    cfg: ScenarioConfig,
    n_samples: int = 1500,
    n_time_points: int = 100,
    hidden: int = 128,
    n_layers: int = 3,
    epochs: int = 300,
    batch_size: int = 2048,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[TrajectoryNet, np.ndarray]:
    """
    Train the Trajectory Predictor.

    Returns the trained model and the query time-grid.
    """
    params_np, trajs_np, query_times = generate_dataset(
        cfg, n_samples=n_samples, n_time_points=n_time_points, seed=seed,
    )

    # Build (params ⊕ t, U) dataset  — shape: (n_samples * n_time_points, 7), (…,)
    n_s, n_t = trajs_np.shape
    p_rep = np.repeat(params_np, n_t, axis=0)                # (n_s*n_t, 6)
    t_rep = np.tile(query_times, n_s).reshape(-1, 1)          # (n_s*n_t, 1)
    X_np = np.hstack([p_rep, t_rep]).astype(np.float32)        # (n_s*n_t, 7)
    y_np = trajs_np.ravel().astype(np.float32)                 # (n_s*n_t,)

    # Normalise inputs (z-score on each column)
    X_mean = X_np.mean(axis=0)
    X_std  = X_np.std(axis=0) + 1e-8
    X_norm = (X_np - X_mean) / X_std

    X_t = torch.from_numpy(X_norm)
    y_t = torch.from_numpy(y_np)

    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True,
    )

    model = TrajectoryNet(n_params=6, hidden=hidden, n_layers=n_layers)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.MSELoss()

    print(f"[TrajNet] Training on {len(X_t):,} points for {epochs} epochs …")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()
        if epoch % 50 == 0:
            avg = total_loss / len(X_t)
            print(f"[TrajNet]   Epoch {epoch}/{epochs}  MSE={avg:.3e}",
                  flush=True)

    model.eval()
    print("[TrajNet] Training complete.")

    # Attach normalisation stats so we can use them at inference
    model.register_buffer("x_mean", torch.tensor(X_mean, dtype=torch.float32))
    model.register_buffer("x_std",  torch.tensor(X_std,  dtype=torch.float32))

    return model, query_times


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_trajectory_model(
    model: TrajectoryNet,
    query_times: np.ndarray,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": model.net[0].out_features,
            "n_layers": sum(1 for m in model.net if isinstance(m, nn.SiLU)),
            "x_mean": model.x_mean.numpy(),
            "x_std": model.x_std.numpy(),
            "query_times": query_times,
        },
        path,
    )
    print(f"[TrajNet] Saved → {path}")


def load_trajectory_model(path: str) -> tuple[TrajectoryNet, np.ndarray]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TrajectoryNet(
        n_params=6, hidden=ckpt["hidden"], n_layers=ckpt["n_layers"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["query_times"]


def predict_normalised(
    model: TrajectoryNet,
    params: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """
    Predict trajectory for a single parameter vector, applying
    the stored normalisation.
    """
    n_t = len(query_times)
    p_rep = np.repeat(params.reshape(1, -1), n_t, axis=0)   # (N_t, 6)
    t_col = query_times.reshape(-1, 1)                        # (N_t, 1)
    X = np.hstack([p_rep, t_col]).astype(np.float32)          # (N_t, 7)

    x_mean = model.x_mean.numpy()
    x_std  = model.x_std.numpy()
    X_norm = (X - x_mean) / x_std

    with torch.no_grad():
        pred = model(torch.from_numpy(X_norm)).numpy()
    return pred
