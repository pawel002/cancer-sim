"""
Training pipeline for the MLP time-stepper surrogate.
Trains a small MLP on ODE trajectories with randomised parameters, then saves
the checkpoint to disk for later use by MLPEngine.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..engines.neural.mlp_engine import MLPModel
from .generate_data import generate_mlp_data


def train_mlp(
    n_traj: int = 300,
    epochs: int = 300,
    batch_size: int = 1024,
    lr: float = 1e-3,
    T_train: float = 20.0,
    dt: float = 0.05,
    seed: int = 42,
) -> MLPModel:
    """
    Trains an MLPModel on multi-scenario ODE trajectory data.

    Returns the trained model (in eval mode).
    """
    print(f"[MLP] Generating training data ({n_traj} ODE trajectories)…")
    X_np, y_np = generate_mlp_data(n_traj=n_traj, T_train=T_train, dt=dt, seed=seed)
    print(f"[MLP]   {len(X_np):,} input-output pairs generated.")

    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y_np).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model = MLPModel()
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if epoch % 100 == 0:
            print(f"[MLP]   Epoch {epoch}/{epochs}  MSE={total / len(X_np):.2e}")

    model.eval()
    print("[MLP] Training complete.")
    return model


def save_mlp(model: MLPModel, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[MLP] Model saved → {path}")


def load_mlp(path: str) -> MLPModel:
    model = MLPModel()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
