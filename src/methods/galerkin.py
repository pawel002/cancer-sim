"""
Fourier-Galerkin reduced-order model (Section 3.2).

Decomposes u(x,t) ≈ Σ_k a_k(t) ψ_k(x) using K = sqrt_K² sine-sine
Fourier modes and solves K coupled ODEs for the amplitudes.

The radiation coupling matrix R̃_{jk} = ∫ψ_j H R₀ ψ_k dA / ‖ψ_j‖²
is precomputed offline, embedding the full spatial structure of both the
hypoxia map and the beam.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from src.methods.shared import (
    make_ic, make_hypoxia, make_beam, make_grid,
    TIMES_FULL, T_MAX, T_RAD_S, T_RAD_E,
    D_COEF, RHO, K_CAP, BETA, L, L_SQ,
)


def build_basis(N: int, sqrt_K: int = 5) -> dict:
    """
    Build K = sqrt_K² Fourier basis functions satisfying Dirichlet BCs.

    ψ_{mn}(x,y) = sin(mπx/L) sin(nπy/L),  m,n ∈ {1,...,sqrt_K}.

    Returns
    -------
    dict with keys: bases (K,N,N), lambdas (K,), norm_sq (K,), M_k (K,), dx float
    """
    X, Y, dx = make_grid(N)
    bs, lam  = [], []
    for m in range(1, sqrt_K + 1):
        for n in range(1, sqrt_K + 1):
            psi = np.sin(m * np.pi * X / L) * np.sin(n * np.pi * Y / L)
            bs.append(psi)
            lam.append((m ** 2 + n ** 2) * np.pi ** 2 / L ** 2)
    bs  = np.array(bs)
    lam = np.array(lam)
    ns  = np.array([np.sum(b ** 2) * dx ** 2 for b in bs])
    Mk  = np.array([np.sum(b)      * dx ** 2 for b in bs])
    return dict(bases=bs, lambdas=lam, norm_sq=ns, M_k=Mk, dx=dx)


def build_radiation_matrix(
    basis: dict,
    H: np.ndarray,
    R0: np.ndarray,
) -> np.ndarray:
    """
    Precompute R̃_{jk} = ∫ψ_j H R₀ ψ_k dA / ‖ψ_j‖² (K×K matrix).

    This is the offline cost that scales as O(K² N²).
    """
    bs, ns, dx = basis["bases"], basis["norm_sq"], basis["dx"]
    K    = len(bs)
    Rjk  = np.zeros((K, K))
    for j in range(K):
        for k in range(K):
            Rjk[j, k] = np.sum(bs[j] * H * R0 * bs[k]) * dx ** 2
    return Rjk / ns[:, np.newaxis]


def run_galerkin(
    ic_name: str,
    beam_name: str,
    N: int = 50,
    sqrt_K: int = 5,
    *,
    u0: np.ndarray | None = None,
    H: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fourier-Galerkin ROM.

    Projection ODE:
        da_j/dt = −Dλ_j a_j + ρ(1 − U/K) a_j − β c(t) Σ_k R̃_{jk} a_k
    where U = Σ_k a_k M_k / L² is the normalised total mass.

    Accepts optional pre-built arrays for training-data generation.
    """
    if u0 is None:
        u0 = make_ic(ic_name, N)
    if H is None:
        H = make_hypoxia(N)
    if R0 is None:
        R0 = make_beam(beam_name, N)

    basis = build_basis(N, sqrt_K=sqrt_K)
    bs, lam, ns, Mk, dx = (
        basis["bases"], basis["lambdas"], basis["norm_sq"], basis["M_k"], basis["dx"]
    )
    Rt = build_radiation_matrix(basis, H, R0)
    a0 = np.array([np.sum(u0 * b) * dx ** 2 / ns[j] for j, b in enumerate(bs)])

    def rhs(t: float, a: np.ndarray) -> np.ndarray:
        U   = max(float(np.dot(Mk, a)) / L_SQ, 0.0)
        c_t = 1.0 if T_RAD_S <= t <= T_RAD_E else 0.0
        return -D_COEF * lam * a + RHO * (1.0 - U / K_CAP) * a - BETA * c_t * (Rt @ a)

    sol = solve_ivp(rhs, [0.0, T_MAX], a0, method="RK45",
                    t_eval=TIMES_FULL, rtol=1e-5, atol=1e-8)
    return sol.t, np.maximum(sol.y.T @ Mk, 0.0) / L_SQ
