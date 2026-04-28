"""
Neural Residual Corrector (Section 3.3).

Uses the Galerkin surrogate as a physics-based baseline and trains a compact
MLP offline to correct the residual:

    δ(t) = U_PDE(t) − U_Galerkin(t)

The MLP receives a 10-dimensional feature vector encoding the current
Galerkin prediction, time, and fixed spatial summary statistics of the
scenario (IC and beam centres, their distance, effective beam overlap,
effective hypoxia).  These statistics are computed once per scenario from
initial fields — no spatial grid is needed at inference.

Corrected prediction:
    U_NC(t) = max( U_Galerkin(t) + g_φ(s_t), 0 )

Training is performed offline on random PDE simulations (see
src/training/train_neural_corrector.py).  At inference only Galerkin +
a single MLP forward pass per trajectory are required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.methods.shared import (
    make_ic, make_hypoxia, make_beam, make_grid,
    T_MAX, T_RAD_S, T_RAD_E, L,
)
from src.methods.galerkin import run_galerkin

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent.parent / "latex" / "figures" / "neural_corrector.pt"


# ─── network ─────────────────────────────────────────────────────────────
class NeuralCorrector(nn.Module):
    """
    MLP that predicts the residual δ(t) = U_PDE(t) − U_Galerkin(t).

    Input (10-dim):
      [U_gal, t/T, r(t), z_ic_x/L, z_ic_y/L,
       z_bm_x/L, z_bm_y/L, d/L, H_eff, R_eff]

    Output: scalar δ
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden: tuple[int, ...] = (64, 64, 32),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─── feature extraction ──────────────────────────────────────────────────
def extract_scenario_scalars(
    ic_name: str,
    beam_name: str,
    N: int = 50,
    *,
    u0: np.ndarray | None = None,
    H: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute fixed (time-independent) spatial summary statistics for a scenario.

    Returns dict with keys:
        z_ic_x, z_ic_y   – IC centre-of-mass (normalised by L)
        z_bm_x, z_bm_y  – beam centre-of-mass (normalised by L)
        d                – normalised IC–beam CoM distance
        H_eff            – IC-weighted mean hypoxia
        R_eff            – IC-weighted beam amplitude
    """
    if u0 is None:
        u0 = make_ic(ic_name, N)
    if H is None:
        H = make_hypoxia(N)
    if R0 is None:
        R0 = make_beam(beam_name, N)

    dx       = L / N
    X, Y, _  = make_grid(N)

    w_ic  = u0 * dx ** 2
    m_tot = np.sum(w_ic) + 1e-12
    z_ic_x = float(np.sum(X * w_ic) / m_tot) / L
    z_ic_y = float(np.sum(Y * w_ic) / m_tot) / L

    w_bm  = R0 * dx ** 2
    m_bm  = np.sum(w_bm) + 1e-12
    z_bm_x = float(np.sum(X * w_bm) / m_bm) / L
    z_bm_y = float(np.sum(Y * w_bm) / m_bm) / L

    d = float(np.sqrt((z_ic_x - z_bm_x) ** 2 + (z_ic_y - z_bm_y) ** 2))

    w = u0 / (np.sum(u0) + 1e-12)
    H_eff = float(np.sum(H  * w))
    R_eff = float(np.sum(R0 * w))

    return dict(z_ic_x=z_ic_x, z_ic_y=z_ic_y,
                z_bm_x=z_bm_x, z_bm_y=z_bm_y,
                d=d, H_eff=H_eff, R_eff=R_eff)


def build_input_tensor(
    scalars: dict[str, float],
    U_gal: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Assemble (T, 10) feature matrix.

    Columns: [U_gal, t/T, r(t), z_ic_x, z_ic_y, z_bm_x, z_bm_y, d, H_eff, R_eff]
    """
    T     = len(times)
    r_arr = np.array([1.0 if T_RAD_S <= t <= T_RAD_E else 0.0 for t in times],
                     dtype=np.float32)
    X = np.column_stack([
        U_gal.astype(np.float32),
        (times / T_MAX).astype(np.float32),
        r_arr,
        np.full(T, scalars["z_ic_x"], dtype=np.float32),
        np.full(T, scalars["z_ic_y"], dtype=np.float32),
        np.full(T, scalars["z_bm_x"], dtype=np.float32),
        np.full(T, scalars["z_bm_y"], dtype=np.float32),
        np.full(T, scalars["d"],      dtype=np.float32),
        np.full(T, scalars["H_eff"],  dtype=np.float32),
        np.full(T, scalars["R_eff"],  dtype=np.float32),
    ])
    return X   # (T, 10)


# ─── inference ───────────────────────────────────────────────────────────
def run_neural_corrector(
    ic_name: str,
    beam_name: str,
    N: int = 50,
    *,
    model: NeuralCorrector | None = None,
    checkpoint: Path = CHECKPOINT_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Neural Corrector inference.

    1. Run Galerkin → U_gal(t).
    2. Extract scenario scalars from initial fields.
    3. Predict residual δ(t) = g_φ(features_t).
    4. Return  U_NC(t) = max(U_gal + δ, 0).
    """
    # Load model if not supplied
    if model is None:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Neural corrector checkpoint not found at {checkpoint}.\n"
                "Run: uv run python src/training/train_neural_corrector.py"
            )
        ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = NeuralCorrector()
        model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Galerkin baseline
    times, U_gal = run_galerkin(ic_name, beam_name, N=N)

    # Scenario spatial summary
    scalars = extract_scenario_scalars(ic_name, beam_name, N)

    # Feature matrix → correction
    X_np  = build_input_tensor(scalars, U_gal, times)
    with torch.no_grad():
        delta = model(torch.from_numpy(X_np)).numpy()

    U_nc = np.maximum(U_gal + delta, 0.0)
    return times, U_nc
