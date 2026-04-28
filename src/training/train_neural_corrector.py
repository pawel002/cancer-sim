"""
Offline training pipeline for the Neural Residual Corrector.

Generates random PDE + Galerkin trajectory pairs, computes residuals,
and trains a compact MLP to predict them from spatial scenario features.

The 16 named test scenarios (4 IC types × 4 beam types) are excluded from
training — only continuously-sampled random spatial configurations are used.

Usage:
    uv run python src/training/train_neural_corrector.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.methods.shared import (
    make_hypoxia, make_grid,
    TIMES_FULL,
    SIGMA_IC, SIGMA_BEAM, AMPL_IC, K_CAP, L,
    PEAK_CENTERS,
)
from src.methods.neural_corrector import (
    NeuralCorrector, build_input_tensor, extract_scenario_scalars, CHECKPOINT_PATH,
)
from src.methods.pde      import run_pde
from src.methods.galerkin import run_galerkin

# Exact IC/beam centre values used in the 16 test scenarios (hold-out set)
_TEST_CENTRES = {(cx, cy) for centers in PEAK_CENTERS.values() for cx, cy in centers}


# ─── random scenario generators ──────────────────────────────────────────
def _random_single_peak(rng: np.random.Generator) -> tuple[float, float]:
    """Sample a centre strictly outside the test-scenario grid points."""
    while True:
        cx = float(rng.uniform(L * 0.15, L * 0.85))
        cy = float(rng.uniform(L * 0.15, L * 0.85))
        if all(abs(cx - tx) > 0.5 and abs(cy - ty) > 0.5 for tx, ty in _TEST_CENTRES):
            return cx, cy


def make_custom_ic(centers: list[tuple[float, float]], N: int) -> np.ndarray:
    X, Y, _ = make_grid(N)
    n  = len(centers)
    u0 = np.zeros((N, N))
    for cx, cy in centers:
        u0 += (AMPL_IC / n) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * SIGMA_IC ** 2))
    u0 = np.clip(u0, 0.0, K_CAP)
    u0[0, :] = u0[-1, :] = u0[:, 0] = u0[:, -1] = 0.0
    return u0


def make_custom_beam(centers: list[tuple[float, float]], N: int) -> np.ndarray:
    X, Y, _ = make_grid(N)
    R0 = np.zeros((N, N))
    for bx, by in centers:
        R0 += np.exp(-((X - bx) ** 2 + (Y - by) ** 2) / (2 * SIGMA_BEAM ** 2))
    return np.clip(R0, 0.0, 1.0)


# ─── data generation ─────────────────────────────────────────────────────
def generate_training_data(
    n_scenarios: int = 200,
    N: int = 50,
    two_peak_frac: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (X, y) pairs for the neural corrector.

    X : (n_scenarios * T, 10)  — feature vectors
    y : (n_scenarios * T,)     — residual δ(t) = U_PDE(t) − U_Galerkin(t)
    """
    rng     = np.random.default_rng(seed)
    X_list, y_list = [], []
    n_two   = int(n_scenarios * two_peak_frac)
    n_single = n_scenarios - n_two
    H       = make_hypoxia(N)

    t0 = time.perf_counter()
    for idx in range(n_scenarios):
        # ── IC and beam centres ──────────────────────────────────────────
        use_two = idx >= n_single
        if use_two:
            cx1, cy1 = _random_single_peak(rng)
            # second peak roughly 4-6 units away
            angle = float(rng.uniform(0, 2 * np.pi))
            sep   = float(rng.uniform(3.5, 6.0))
            cx2   = float(np.clip(cx1 + sep * np.cos(angle), L * 0.15, L * 0.85))
            cy2   = float(np.clip(cy1 + sep * np.sin(angle), L * 0.15, L * 0.85))
            ic_ctr  = [(cx1, cy1), (cx2, cy2)]
        else:
            cx, cy  = _random_single_peak(rng)
            ic_ctr  = [(cx, cy)]

        # Beam can be single or two-peak independently
        use_two_bm = bool(rng.random() < two_peak_frac)
        if use_two_bm:
            bx1, by1 = _random_single_peak(rng)
            angle = float(rng.uniform(0, 2 * np.pi))
            sep   = float(rng.uniform(3.5, 6.0))
            bx2   = float(np.clip(bx1 + sep * np.cos(angle), L * 0.15, L * 0.85))
            by2   = float(np.clip(by1 + sep * np.sin(angle), L * 0.15, L * 0.85))
            bm_ctr = [(bx1, by1), (bx2, by2)]
        else:
            bx, by  = _random_single_peak(rng)
            bm_ctr  = [(bx, by)]

        u0 = make_custom_ic(ic_ctr,  N)
        R0 = make_custom_beam(bm_ctr, N)

        # ── simulate ────────────────────────────────────────────────────
        _, U_pde = run_pde("", "", N, u0=u0, H=H, R0=R0)
        _, U_gal = run_galerkin("", "", N, u0=u0, H=H, R0=R0)

        residual = (U_pde - U_gal).astype(np.float32)

        # ── features ────────────────────────────────────────────────────
        scalars = extract_scenario_scalars("", "", N, u0=u0, H=H, R0=R0)
        X_block = build_input_tensor(scalars, U_gal, TIMES_FULL).astype(np.float32)

        X_list.append(X_block)
        y_list.append(residual)

        if (idx + 1) % 20 == 0:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (idx + 1) * (n_scenarios - idx - 1)
            print(f"  [data gen] {idx+1}/{n_scenarios}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    print(f"  [data gen] Done in {time.perf_counter()-t0:.1f}s")
    return np.vstack(X_list), np.concatenate(y_list)


# ─── training ────────────────────────────────────────────────────────────
def train(
    n_scenarios: int = 200,
    n_epochs: int = 200,
    batch_size: int = 1024,
    lr: float = 1e-3,
    N: int = 50,
    checkpoint_path: Path = CHECKPOINT_PATH,
    seed: int = 42,
) -> NeuralCorrector:
    """
    Full training pipeline.

    Returns the trained NeuralCorrector model (also saved to checkpoint_path).
    """
    print(f"[NeuralCorrector] Generating {n_scenarios} training scenarios (N={N}) …")
    X_np, y_np = generate_training_data(n_scenarios=n_scenarios, N=N, seed=seed)
    print(f"[NeuralCorrector] Dataset: {len(X_np):,} samples × {X_np.shape[1]} features")

    torch.manual_seed(seed)
    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y_np)

    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=batch_size, shuffle=True)

    model     = NeuralCorrector()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    print(f"[NeuralCorrector] Training for {n_epochs} epochs …")
    model.train()
    for epoch in range(1, n_epochs + 1):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        scheduler.step()
        if epoch % 40 == 0:
            print(f"  Epoch {epoch}/{n_epochs}  MSE={total/len(X_t):.4e}", flush=True)

    model.eval()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    print(f"[NeuralCorrector] Saved → {checkpoint_path}")
    return model


if __name__ == "__main__":
    train()
