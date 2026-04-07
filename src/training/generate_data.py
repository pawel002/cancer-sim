"""
Canonical scenario configuration and data-generation helpers shared across all
training pipelines.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..core.protocols import standard_radiation_schedule, uniform_spatial_radiation
from ..core.simulator import Simulator
from ..core.state import State
from ..engines.ode_engine import ODEEngine
from ..engines.pde_engine import PDEEngine

# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """All physical/numerical parameters for the canonical benchmark scenario."""
    L: float = 10.0          # Domain side length [spatial units]
    N: int = 50               # Grid points per axis (N×N grid)
    dt: float = 0.05          # Time step [time units]
    T_max: float = 40.0       # Total simulation end time
    D: float = 0.01           # Diffusion coefficient
    rho: float = 0.3          # Proliferation rate
    K: float = 1.0            # Carrying capacity (normalised)
    beta: float = 0.8         # Radiosensitivity
    t_rad_start: float = 10.0 # Radiation window start
    t_rad_end: float = 30.0   # Radiation window end
    train_start: float = 0.0  # Training assimilation window start
    train_end: float = 20.0   # Training assimilation window end

    @property
    def dx(self) -> float:
        return self.L / self.N

    @property
    def n_steps(self) -> int:
        return int(self.T_max / self.dt)

    @property
    def grid_x(self) -> np.ndarray:
        return np.linspace(0.0, self.L, self.N)

    def H_eff_value(self) -> float:
        """Spatial average of the hypoxia map (used by scalar ODE models)."""
        return float(np.mean(make_hypoxia_map(self)))


# ---------------------------------------------------------------------------
# Spatial field factories
# ---------------------------------------------------------------------------

def make_gaussian_ic(
    cfg: ScenarioConfig,
    sigma: float = 1.5,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Creates a Gaussian initial condition centred in the domain."""
    x = cfg.grid_x
    X, Y = np.meshgrid(x, x)
    center = cfg.L / 2.0
    r_sq = (X - center) ** 2 + (Y - center) ** 2
    u0 = amplitude * np.exp(-r_sq / (2.0 * sigma ** 2))
    u0 = np.clip(u0, 0.0, 1.0)
    # Enforce Dirichlet BCs
    u0[0, :] = u0[-1, :] = u0[:, 0] = u0[:, -1] = 0.0
    return u0


def make_hypoxia_map(cfg: ScenarioConfig) -> np.ndarray:
    """Spatially-varying hypoxia map H(x) ∈ (0.3, 1.0]."""
    x = cfg.grid_x
    X, Y = np.meshgrid(x, x)
    center = cfg.L / 2.0
    r_sq = (X - center) ** 2 + (Y - center) ** 2
    return 0.3 + 0.7 * np.exp(-r_sq / (cfg.L / 2.0) ** 2)


# ---------------------------------------------------------------------------
# Protocol factories
# ---------------------------------------------------------------------------

def make_spatial_rad(cfg: ScenarioConfig) -> Callable[[tuple[int, ...], float], np.ndarray]:
    """Returns a spatial radiation protocol for the PDE engine."""
    def _rad(shape: tuple[int, ...], t: float) -> np.ndarray:
        return uniform_spatial_radiation(shape, t, cfg.t_rad_start, cfg.t_rad_end)
    return _rad


def make_scalar_rad(cfg: ScenarioConfig) -> Callable[[float], float]:
    """Returns a scalar radiation protocol for ODE/neural engines."""
    def _rad(t: float) -> float:
        return standard_radiation_schedule(t, cfg.t_rad_start, cfg.t_rad_end)
    return _rad


# ---------------------------------------------------------------------------
# Ground-truth simulation runners
# ---------------------------------------------------------------------------

def run_pde_scenario(
    cfg: ScenarioConfig,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Runs the PDE engine for the canonical scenario.

    Returns
    -------
    times   : (n_steps+1,) array of simulation times
    masses  : (n_steps+1,) array of total tumour mass
    u_fields: list of (N, N) density arrays, one per step
    """
    u0 = make_gaussian_ic(cfg)
    H_map = make_hypoxia_map(cfg)
    mass0 = float(np.sum(u0) * cfg.dx ** 2)

    engine = PDEEngine(
        D=cfg.D, rho=cfg.rho, K=cfg.K, beta=cfg.beta, dx=cfg.dx,
        radiation_protocol=make_spatial_rad(cfg),
        hypoxia_map=H_map,
    )
    state0 = State(u=u0, mass=mass0)
    sim = Simulator(engine, cfg.dt, cfg.T_max)
    history = sim.run(state0)

    n = len(history)
    times = np.array([i * cfg.dt for i in range(n)])
    masses = np.array([s.mass for s in history])
    u_fields = [s.u for s in history]
    return times, masses, u_fields


def run_ode_scenario(cfg: ScenarioConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the ODE surrogate for the canonical scenario.

    The ODE tracks the *spatial average* U(t) = M(t) / L² ∈ [0, 1].
    Returned ``masses`` are in the same normalised units as the ODE model.
    Multiply by L² to obtain total tumour mass for comparison with the PDE.

    Returns
    -------
    times  : (n_steps+1,) array
    masses : (n_steps+1,) normalised spatial-average array ∈ [0, 1]
    """
    u0 = make_gaussian_ic(cfg)
    mass0_total = float(np.sum(u0) * cfg.dx ** 2)
    # Spatial average: normalise to [0, 1] so the logistic term is well-posed
    u0_norm = mass0_total / (cfg.L ** 2)

    engine = ODEEngine(
        rho=cfg.rho,
        beta=cfg.beta,
        H_eff=cfg.H_eff_value(),
        radiation_protocol=make_scalar_rad(cfg),
    )
    state0 = State(u=u0_norm, mass=u0_norm)
    sim = Simulator(engine, cfg.dt, cfg.T_max)
    history = sim.run(state0)

    n = len(history)
    times = np.array([i * cfg.dt for i in range(n)])
    masses = np.array([s.mass for s in history])   # normalised spatial average
    return times, masses


def compute_center_of_mass(u_fields: list[np.ndarray], cfg: ScenarioConfig) -> np.ndarray:
    """
    Computes the average (x+y) centre-of-mass coordinate for each saved field.
    Used as the auxiliary tracking variable C(t) for the NODE.
    """
    x = cfg.grid_x
    X, Y = np.meshgrid(x, x)
    centers = np.empty(len(u_fields))
    for i, u in enumerate(u_fields):
        mass = np.sum(u) * cfg.dx ** 2
        if mass < 1e-10:
            centers[i] = cfg.L / 2.0
        else:
            cx = np.sum(X * u) * cfg.dx ** 2 / mass
            cy = np.sum(Y * u) * cfg.dx ** 2 / mass
            centers[i] = (cx + cy) / 2.0
    return centers


# ---------------------------------------------------------------------------
# MLP training data generator
# ---------------------------------------------------------------------------

def generate_mlp_data(
    n_traj: int = 300,
    T_train: float = 20.0,
    dt: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates (X, y) training pairs for the MLP time-stepper by running many
    ODE trajectories with randomised parameters.

    X columns: [U_n, r_n, dt, rho, beta]
    y         : U_{n+1}
    """
    rng = np.random.default_rng(seed)
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    steps = int(T_train / dt)

    for _ in range(n_traj):
        rho = float(rng.uniform(0.1, 0.6))
        beta = float(rng.uniform(0.2, 1.5))
        H_eff = float(rng.uniform(0.2, 1.0))
        U0 = float(rng.uniform(0.05, 0.6))
        t_s = float(rng.uniform(5.0, 15.0))
        t_e = t_s + float(rng.uniform(5.0, 15.0))

        def _make_rad(s: float, e: float) -> Callable[[float], float]:
            def _r(t: float) -> float:
                return standard_radiation_schedule(t, s, e)
            return _r

        rad = _make_rad(t_s, t_e)
        engine = ODEEngine(rho=rho, beta=beta, H_eff=H_eff, radiation_protocol=rad)
        hist = Simulator(engine, dt, T_train).run(State(u=U0, mass=U0))

        for i in range(steps):
            t_i = i * dt
            U_n = float(hist[i].mass)
            U_np1 = float(hist[i + 1].mass)
            r_n = rad(t_i)
            X_list.append(np.array([U_n, r_n, dt, rho, beta], dtype=np.float32))
            y_list.append(float(U_np1))

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    return X, y
