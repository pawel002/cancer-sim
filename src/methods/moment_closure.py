"""
Moment Closure surrogate (Section 3.1).

Tracks the tumour centre-of-mass z(t) initialised from the IC.
Radiation is evaluated at the fixed CoM position: R_eff(t) = R₀(z, t).

Key failure mode — Two-peak IC with misaligned beam:
  CoM sits between the tumour lobes; if the beam does not cover the CoM,
  the surrogate dramatically underestimates radiation exposure.
"""
from __future__ import annotations

import numpy as np

from src.methods.shared import (
    make_ic, make_hypoxia, make_beam, make_grid,
    TIMES_FULL, N_STEPS, DT, T_RAD_S, T_RAD_E,
    RHO, K_CAP, BETA, L, L_SQ,
)


def _bilinear(f: np.ndarray, xi: float, yi: float, dx: float) -> float:
    """Bilinear interpolation of a 2-D field at continuous coordinates (xi, yi)."""
    N  = f.shape[0]
    ix = float(np.clip(xi / dx, 0.0, N - 1 - 1e-9))
    iy = float(np.clip(yi / dx, 0.0, N - 1 - 1e-9))
    x0, y0 = int(ix), int(iy)
    x1, y1 = min(x0 + 1, N - 1), min(y0 + 1, N - 1)
    fx, fy = ix - x0, iy - y0
    return float(
        f[y0, x0] * (1 - fx) * (1 - fy)
        + f[y0, x1] * fx      * (1 - fy)
        + f[y1, x0] * (1 - fx) * fy
        + f[y1, x1] * fx       * fy
    )


def run_moment_closure(
    ic_name: str,
    beam_name: str,
    N: int = 50,
    *,
    u0: np.ndarray | None = None,
    H: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Centre-of-mass ODE surrogate.

    z is initialised from the IC and fixed (leading-order closure).
    H and R₀ are evaluated once at z.
    """
    if u0 is None:
        u0 = make_ic(ic_name, N)
    if H is None:
        H = make_hypoxia(N)
    if R0 is None:
        R0 = make_beam(beam_name, N)

    dx       = L / N
    X, Y, _  = make_grid(N)

    U     = float(np.sum(u0) * dx ** 2) / L_SQ
    w_ic  = u0 * dx ** 2
    m_tot = np.sum(w_ic) + 1e-12
    z_x   = float(np.sum(X * w_ic) / m_tot)
    z_y   = float(np.sum(Y * w_ic) / m_tot)

    H_z  = _bilinear(H,  z_x, z_y, dx)
    R0_z = _bilinear(R0, z_x, z_y, dx)

    out = [U]
    for i in range(N_STEPS):
        t   = i * DT
        r_t = 1.0 if T_RAD_S <= t <= T_RAD_E else 0.0
        dU  = RHO * U * (1.0 - U / K_CAP) - BETA * H_z * R0_z * r_t * U
        U   = max(U + DT * dU, 0.0)
        out.append(U)

    return TIMES_FULL, np.array(out)
