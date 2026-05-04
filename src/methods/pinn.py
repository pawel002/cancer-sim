"""
PINN-guided fast rollout surrogate.

This module implements the low-dimensional PINN-guided rollout described in
cancer_sim.pdf, Sec. 2.4.  A high-fidelity teacher trajectory is used offline to
learn a residual derivative correction.  At inference time the method performs a
pure explicit ODE rollout:

    U[n+1] = U[n] + dt * ( f_RT(U[n], c[n]) + g_theta(s[n]) )

where U(t)=y(t)/L^2 is the observed tumour mass, f_RT is the known macroscopic
radiotherapy dynamics, and g_theta is a compact MLP correction that carries the
teacher's unresolved spatial information into a real-time rollout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.methods.shared import (
    make_ic,
    make_hypoxia,
    make_beam,
    make_grid,
    TIMES_FULL,
    T_MAX,
    N_STEPS,
    DT,
    T_RAD_S,
    T_RAD_E,
    RHO,
    K_CAP,
    BETA,
    L,
    L_SQ,
    D_COEF,
)

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent.parent / "latex" / "figures" / "pinn_fast_rollout.pt"


class PINNResidualMLP(nn.Module):
    """
    Residual derivative model g_theta(s).

    Input vector, 12 features:
      [U, t/T, c(t), dt/T, rho, beta, D, H_eff, R_eff,
       z_ic_x/L, z_beam_x/L, IC_beam_distance/L]

    Output: scalar derivative correction v(t), not a state correction.  This
    follows the PINN-guided fast-rollout target
        v_n = (U_teacher[n+1]-U_teacher[n])/dt - f_RT(U_teacher[n], c_n).
    """

    def __init__(self, input_dim: int = 12, hidden: tuple[int, ...] = (64, 64, 32)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.Tanh()]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def extract_pinn_scalars(
    ic_name: str,
    beam_name: str,
    N: int = 50,
    *,
    u0: np.ndarray | None = None,
    H: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute scenario-level statistics used by the fast PINN rollout."""
    if u0 is None:
        u0 = make_ic(ic_name, N)
    if H is None:
        H = make_hypoxia(N)
    if R0 is None:
        R0 = make_beam(beam_name, N)

    dx = L / N
    X, Y, _ = make_grid(N)

    w_ic = u0 * dx**2
    m_ic = float(np.sum(w_ic)) + 1e-12
    z_ic_x = float(np.sum(X * w_ic) / m_ic) / L
    z_ic_y = float(np.sum(Y * w_ic) / m_ic) / L

    w_bm = R0 * dx**2
    m_bm = float(np.sum(w_bm)) + 1e-12
    z_bm_x = float(np.sum(X * w_bm) / m_bm) / L
    z_bm_y = float(np.sum(Y * w_bm) / m_bm) / L

    d = float(np.sqrt((z_ic_x - z_bm_x) ** 2 + (z_ic_y - z_bm_y) ** 2))
    w = u0 / (np.sum(u0) + 1e-12)

    return {
        "U0": float(np.sum(u0) * dx**2) / L_SQ,
        "H_eff": float(np.sum(H * w)),
        "R_eff": float(np.sum(R0 * w)),
        "z_ic_x": z_ic_x,
        "z_ic_y": z_ic_y,
        "z_bm_x": z_bm_x,
        "z_bm_y": z_bm_y,
        "d": d,
    }


def f_rt(U: np.ndarray | float, c_t: np.ndarray | float, scalars: dict[str, float]) -> np.ndarray | float:
    """Known macroscopic radiotherapy core f_RT(U,c)."""
    return RHO * U * (1.0 - U / K_CAP) - BETA * scalars["H_eff"] * scalars["R_eff"] * c_t * U


def build_pinn_features(
    scalars: dict[str, float],
    U: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Assemble the feature matrix used by g_theta during training/inference."""
    c = np.array([1.0 if T_RAD_S <= t <= T_RAD_E else 0.0 for t in times], dtype=np.float32)
    T = len(times)
    return np.column_stack([
        U.astype(np.float32),
        (times / T_MAX).astype(np.float32),
        c,
        np.full(T, DT / T_MAX, dtype=np.float32),
        np.full(T, RHO, dtype=np.float32),
        np.full(T, BETA, dtype=np.float32),
        np.full(T, D_COEF, dtype=np.float32),
        np.full(T, scalars["H_eff"], dtype=np.float32),
        np.full(T, scalars["R_eff"], dtype=np.float32),
        np.full(T, scalars["z_ic_x"], dtype=np.float32),
        np.full(T, scalars["z_bm_x"], dtype=np.float32),
        np.full(T, scalars["d"], dtype=np.float32),
    ])


def run_pinn(
    ic_name: str,
    beam_name: str,
    N: int = 50,
    *,
    model: PINNResidualMLP | None = None,
    checkpoint: Path = CHECKPOINT_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    PINN-guided fast rollout inference.

    The checkpoint contains only the small residual MLP.  No spatial grid solve
    or high-fidelity teacher evaluation is performed during rollout.
    """
    scalars = extract_pinn_scalars(ic_name, beam_name, N)

    if model is None:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"PINN fast-rollout checkpoint not found at {checkpoint}.\n"
                "Run: uv run python src/training/train_pinn_fast_rollout.py"
            )
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = PINNResidualMLP()
        model.load_state_dict(ckpt["state_dict"])
    model.eval()

    U = float(scalars["U0"])
    out = [U]
    with torch.no_grad():
        for i in range(N_STEPS):
            t = TIMES_FULL[i]
            c_t = 1.0 if T_RAD_S <= t <= T_RAD_E else 0.0
            X = build_pinn_features(scalars, np.array([U], dtype=np.float32), np.array([t], dtype=np.float32))
            residual = float(model(torch.from_numpy(X)).item())
            dU = float(f_rt(U, c_t, scalars)) + residual
            U = max(U + DT * dU, 0.0)
            out.append(U)

    return TIMES_FULL, np.asarray(out, dtype=np.float64)
