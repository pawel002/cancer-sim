"""
ODE surrogate (Section 2.2).

dy/dt = ρ y (1 − y/K) − β H_eff R_eff(t) y

H_eff and R_eff are IC-weighted spatial means of H(x) and R₀(x).
"""
from __future__ import annotations

import numpy as np

from src.methods.shared import (
    make_ic, make_hypoxia, make_beam,
    TIMES_FULL, N_STEPS, DT, T_RAD_S, T_RAD_E,
    RHO, K_CAP, BETA, L, L_SQ,
)


def run_ode(
    ic_name: str,
    beam_name: str,
    N: int = 50,
    *,
    u0: np.ndarray | None = None,
    H: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    IC-weighted spatially-averaged ODE surrogate.

    Accepts optional pre-built arrays for training-data generation.
    """
    if u0 is None:
        u0 = make_ic(ic_name, N)
    if H is None:
        H = make_hypoxia(N)
    if R0 is None:
        R0 = make_beam(beam_name, N)

    dx    = L / N
    w     = u0 / (np.sum(u0) + 1e-12)
    H_eff = float(np.sum(H  * w))
    R_eff = float(np.sum(R0 * w))

    U   = float(np.sum(u0) * dx ** 2) / L_SQ
    out = [U]

    for i in range(N_STEPS):
        t   = i * DT
        r_t = R_eff if T_RAD_S <= t <= T_RAD_E else 0.0
        dU  = RHO * U * (1.0 - U / K_CAP) - BETA * H_eff * r_t * U
        U   = max(U + DT * dU, 0.0)
        out.append(U)

    return TIMES_FULL, np.array(out)
