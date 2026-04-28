"""
Shared physical constants, grid utilities, and spatial field factories.
All method modules import from here to guarantee a single source of truth.
"""
from __future__ import annotations

import numpy as np

# ─── physical constants ──────────────────────────────────────────────────
L          = 10.0
T_MAX      = 40.0
DT         = 0.05
D_COEF     = 0.01
RHO        = 0.01
K_CAP      = 1.0
BETA       = 0.7
T_RAD_S    = 10.0
T_RAD_E    = 30.0
SIGMA_IC   = 1.5
SIGMA_BEAM = 1.5    # narrow beam → clear spatial contrast
AMPL_IC    = 1
L_SQ       = L ** 2

N_STEPS    = int(T_MAX / DT)
TIMES_FULL = np.linspace(0.0, T_MAX, N_STEPS + 1)

# ─── named distribution centres ─────────────────────────────────────────
DIST_NAMES  = ["left", "right", "middle", "two_peaks"]
DIST_LABELS = ["Left", "Right", "Middle", "Two Peaks"]

PEAK_CENTERS: dict[str, list[tuple[float, float]]] = {
    "left":      [(L / 4,     L / 2)],
    "right":     [(3 * L / 4, L / 2)],
    "middle":    [(L / 2,     L / 2)],
    "two_peaks": [(L / 4,     L / 2), (3 * L / 4, L / 2)],
}


# ─── grid and field factories ────────────────────────────────────────────
def make_grid(N: int) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.linspace(0.0, L, N)
    X, Y = np.meshgrid(x, x)
    return X, Y, L / N


def make_ic(ic_name: str, N: int) -> np.ndarray:
    """Gaussian tumour IC centred on the chosen distribution's peak(s)."""
    X, Y, _ = make_grid(N)
    centers  = PEAK_CENTERS[ic_name]
    n        = len(centers)
    u0       = np.zeros((N, N))
    for cx, cy in centers:
        u0 += (AMPL_IC / n) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * SIGMA_IC ** 2))
    u0 = np.clip(u0, 0.0, K_CAP)
    u0[0, :] = u0[-1, :] = u0[:, 0] = u0[:, -1] = 0.0
    return u0


def make_hypoxia(N: int) -> np.ndarray:
    """Fixed tissue hypoxia map H(x) — radially decreasing from domain centre."""
    X, Y, _ = make_grid(N)
    cx, cy  = L / 2, L / 2
    return np.clip(0.3 + 0.7 * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (L / 2) ** 2), 0.3, 1.0)


def make_beam(beam_name: str, N: int) -> np.ndarray:
    """
    Gaussian beam profile.  Each sub-peak uses full amplitude 1.0 (independent
    radiation sources); result is clipped to [0, 1].

    For two-peak beam the mid-point between beams receives R₀ ≈ 0.25 (σ=1.5),
    creating the CoM-mismatch failure for the Moment Closure surrogate.
    """
    X, Y, _ = make_grid(N)
    R0       = np.zeros((N, N))
    for bx, by in PEAK_CENTERS[beam_name]:
        R0 += np.exp(-((X - bx) ** 2 + (Y - by) ** 2) / (2 * SIGMA_BEAM ** 2))
    return np.clip(R0, 0.0, 1.0)
