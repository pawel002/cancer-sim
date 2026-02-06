import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .sim_config import SimConfig


@dataclass
class SimState:
    """
    Snapshot of the simulation at a specific time t.
    Acts as a DTO for hooks.
    """

    t: float
    step_iteration: int
    u: np.ndarray  # Tumor density grid
    H: np.ndarray  # Hypoxia grid (static, but useful context for plots)
    config: SimConfig  # Reference to config for parameter access

    @property
    def total_mass(self) -> float:
        """Returns total tumor mass (integral of u)."""
        return np.sum(self.u) * (self.config.dx**self.config.DIM)


class SimEngine:
    """
    Manages the lifecycle of the simulation and the solver logic.
    """

    def __init__(self, config: SimConfig):
        self.cfg: SimConfig = config
        self.state: SimState | None = None
        self._hooks: list[Callable[[SimState], None]] = []

        self.inv_dx2 = 1 / (self.cfg.dx**2)
        self._initialize_state()

    def _initialize_state(self):
        """Sets up the initial u matrix and Hypoxia map."""

        axes = [np.linspace(0, self.cfg.L, self.cfg.N) for _ in range(self.cfg.DIM)]
        grids = np.meshgrid(*axes, indexing="ij")
        center = self.cfg.L / 2.0
        r_squared = sum((g - center) ** 2 for g in grids)

        u_init = 0.1 * np.exp(-r_squared / 2.0)
        H_init = np.exp(-r_squared / 5.0)

        self.state = SimState(
            t=0.0, step_iteration=0, u=u_init, H=H_init, config=self.cfg
        )

    def register_hook(
        self, callback: Callable[[SimState], None], every_n_steps: int = 1
    ):
        """
        Attaches a function to be called during the simulation loop.

        :param callback: Function taking SimState as input.
        :param every_n_steps: Throttle the hook (e.g., plot only every 10 frames).
        """

        def wrapped_hook(state: SimState):
            if state.step_iteration % every_n_steps == 0:
                callback(state)

        self._hooks.append(wrapped_hook)

    def _solve_step(self):
        u = self.state.u
        H = self.state.H
        t = self.state.t

        lap = np.zeros_like(u)
        for axis in range(u.ndim):
            lap += np.roll(u, 1, axis=axis) - 2 * u + np.roll(u, -1, axis=axis)
        lap *= self.inv_dx2

        rad_on = 1.0 if self.cfg.rad_start <= t <= self.cfg.rad_end else 0.0

        du = (
            (self.cfg.D_coeff * lap)
            + (self.cfg.rho * u * (1 - u / self.cfg.K))
            - (self.cfg.beta * rad_on * H * u)
        )

        u += du * self.cfg.dt

        if self.cfg.DIM == 2:
            u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0
        elif self.cfg.DIM == 3:
            u[0, :, :] = u[-1, :, :] = u[:, 0, :] = 0
            u[:, -1, :] = u[:, :, 0] = u[:, :, -1] = 0

    def run(self):
        print(f"Engine started. T_max={self.cfg.T_max}")

        total_steps = int(self.cfg.T_max / self.cfg.dt)
        total_time = 0.0

        for _ in range(total_steps):
            step_start_time = time.perf_counter()
            self._solve_step()
            total_time += time.perf_counter() - step_start_time

            self.state.t += self.cfg.dt
            self.state.step_iteration += 1

            for hook in self._hooks:
                hook(self.state)

        print(f"Simulation finished in {total_time:.2f}s")
