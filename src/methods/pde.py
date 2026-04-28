"""
PDE ground truth — finite-difference reaction-diffusion solver.

∂u/∂t = D∇²u + ρu(1 − u/K) − β R(x,t) H(x) u
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import laplace as nd_laplace

from src.methods.shared import (
    make_ic, make_hypoxia, make_beam, make_grid,
    TIMES_FULL, N_STEPS, DT, T_RAD_S, T_RAD_E,
    D_COEF, RHO, K_CAP, BETA, L, L_SQ,
)


def run_pde(
    ic_name: str,
    beam_name: str,
    N: int = 50,
    *,
    u0: np.ndarray | None = None,
    H: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finite-difference ground truth.

    Accepts optional pre-built spatial arrays (u0, H, R0) for use by the
    neural-corrector training pipeline with custom random distributions.
    When names are given the arrays are built from PEAK_CENTERS.

    Returns
    -------
    times  : TIMES_FULL
    U      : normalised spatial average U(t) = mass(t) / L²
    """
    if u0 is None:
        u0 = make_ic(ic_name, N)
    if H is None:
        H = make_hypoxia(N)
    if R0 is None:
        R0 = make_beam(beam_name, N)

    dx = L / N
    u  = u0.copy()
    out = [float(np.sum(u) * dx ** 2) / L_SQ]

    for i in range(N_STEPS):
        t   = i * DT
        r_t = 1.0 if T_RAD_S <= t <= T_RAD_E else 0.0
        lap = nd_laplace(u, mode="constant", cval=0.0) / dx ** 2
        du  = D_COEF * lap + RHO * u * (1.0 - u / K_CAP) - BETA * R0 * H * u * r_t
        u   = np.clip(u + du * DT, 0.0, None)
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
        out.append(float(np.sum(u) * dx ** 2) / L_SQ)

    return TIMES_FULL, np.array(out)
