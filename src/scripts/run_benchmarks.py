import os
import sys
import numpy as np
import time

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.state import State
from src.core.simulator import Simulator
from src.core.protocols import standard_radiation_schedule, uniform_spatial_radiation
from src.engines.pde_engine import PDEEngine
from src.engines.ode_engine import ODEEngine

def main():
    dt = 0.01
    T_max = 20.0
    L = 10.0
    N = 100
    dx = L / N

    D = 0.05
    rho = 0.1
    K = 1.0
    beta = 0.5
    H_eff = 1.0

    # Initialization
    center = L / 2.0
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    r_sq = (X - center)**2 + (Y - center)**2
    
    u_init = 0.1 * np.exp(-r_sq / 2.0)
    H_map = np.exp(-r_sq / 5.0)
    mass_init = float(np.sum(u_init) * (dx**2))

    # Radiation configuration
    def rad_protocol_spatial(shape, t):
        return uniform_spatial_radiation(shape, t, start=10.0, end=15.0)

    def rad_protocol_scalar(t):
        return standard_radiation_schedule(t, start=10.0, end=15.0)

    pde_engine = PDEEngine(D=D, rho=rho, K=K, beta=beta, dx=dx, 
                           radiation_protocol=rad_protocol_spatial, hypoxia_map=H_map)

    ode_engine = ODEEngine(rho=rho, beta=beta, H_eff=H_eff, 
                           radiation_protocol=rad_protocol_scalar)

    # Initial states
    pde_state = State(u=u_init, mass=mass_init)
    ode_state = State(u=mass_init, mass=mass_init)

    def hook(step, t, state, sim):
        print(f"[t={t:.2f}] Mass = {state.mass:.4f}")

    print("Running PDE Benchmark...")
    pde_sim = Simulator(pde_engine, dt, T_max)
    t0 = time.time()
    pde_sim.run(pde_state, hook_fn=hook, m=500)
    print(f"PDE finished in {time.time() - t0:.2f}s")

    print("\nRunning ODE Benchmark...")
    ode_sim = Simulator(ode_engine, dt, T_max)
    t0 = time.time()
    ode_sim.run(ode_state, hook_fn=hook, m=500)
    print(f"ODE finished in {time.time() - t0:.2f}s")
    
if __name__ == "__main__":
    main()
