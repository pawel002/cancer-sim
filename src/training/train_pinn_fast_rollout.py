"""
Offline training for the PINN-guided fast rollout surrogate.

The high-fidelity teacher is represented by macroscopic trajectories obtained
from the spatial solver used in the project.  In a clinical/production PINN
pipeline this function is the place where y_PINN(t)=∫u_PINN(x,t)dx would be
provided.  The training target follows Sec. 2.4 exactly:

    v_n = (U_teacher[n+1] - U_teacher[n]) / dt - f_RT(U_teacher[n], c_n)

Usage:
    uv run python src/training/train_pinn_fast_rollout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.methods.shared import DIST_NAMES, TIMES_FULL, DT, T_RAD_S, T_RAD_E
from src.methods.pde import run_pde
from src.methods.pinn import (
    PINNResidualMLP,
    CHECKPOINT_PATH,
    build_pinn_features,
    extract_pinn_scalars,
    f_rt,
)


def generate_training_data(N: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Build residual-derivative samples from all named IC×beam scenarios."""
    X_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []

    for ic in DIST_NAMES:
        for bm in DIST_NAMES:
            scalars = extract_pinn_scalars(ic, bm, N)
            _, U_teacher = run_pde(ic, bm, N=N)
            U_n = U_teacher[:-1].astype(np.float32)
            t_n = TIMES_FULL[:-1].astype(np.float32)
            c_n = np.array([1.0 if T_RAD_S <= t <= T_RAD_E else 0.0 for t in t_n], dtype=np.float32)
            teacher_derivative = ((U_teacher[1:] - U_teacher[:-1]) / DT).astype(np.float32)
            target = teacher_derivative - np.asarray(f_rt(U_n, c_n, scalars), dtype=np.float32)
            X_blocks.append(build_pinn_features(scalars, U_n, t_n).astype(np.float32))
            y_blocks.append(target.astype(np.float32))

    return np.vstack(X_blocks), np.concatenate(y_blocks)


def train(
    N: int = 50,
    n_epochs: int = 120,
    batch_size: int = 2048,
    lr: float = 1e-3,
    seed: int = 123,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> PINNResidualMLP:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[PINN-FR] Generating teacher residuals from named scenarios (N={N}) …")
    X_np, y_np = generate_training_data(N=N)
    print(f"[PINN-FR] Dataset: {len(X_np):,} samples × {X_np.shape[1]} features")

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    model = PINNResidualMLP()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, n_epochs + 1):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        scheduler.step()
        if epoch % 100 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{n_epochs}  MSE={total/len(X):.4e}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    print(f"[PINN-FR] Saved → {checkpoint_path}")
    model.eval()
    return model


if __name__ == "__main__":
    train()
