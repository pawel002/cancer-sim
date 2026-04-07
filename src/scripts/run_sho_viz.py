import os
import sys
import numpy as np

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.state import State
from src.core.simulator import Simulator
from src.core.protocols import uniform_spatial_radiation
from src.engines.sho_engine import SHOEngine
from src.visualize.hooks import MultiCompartmentGifHook

def main():
    dt = 0.002
    T_max = 20.0
    L = 10.0
    N = 100
    dx = L / N

    D = 0.05
    rho = 0.1
    K = 1.0
    beta = 0.5
    
    mu_0 = 0.5
    o_h = 0.5

    # Initialization
    center = L / 2.0
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    r_sq = (X - center)**2 + (Y - center)**2
    
    # Initial states
    S_init = 0.2 * np.exp(-r_sq / 2.0)
    H_init = 0.05 * np.exp(-r_sq / 1.5)
    u_init = S_init + H_init
    O_init = np.ones_like(S_init) * 1.0  # Fully oxygenated at t=0
    
    mass_init = float(np.sum(u_init) * (dx**2))

    # Target directories
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../experiments/sho_viz'))
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'sho_simulation.gif')

    def rad_protocol_spatial(shape, t):
        return uniform_spatial_radiation(shape, t, start=10.0, end=15.0)

    sho_engine = SHOEngine(D=D, rho=rho, K=K, beta=beta, dx=dx, mu_0=mu_0, o_h=o_h,
                           radiation_protocol=rad_protocol_spatial)
    
    sho_state = State(u=u_init, mass=mass_init, normoxia=S_init, hypoxia=H_init, oxygen=O_init)

    # Visualization Hook
    gif_hook = MultiCompartmentGifHook(save_path=save_path, fps=10)

    def combined_hook(step, t, state, sim):
        gif_hook(step, t, state, sim)

    print("Running SHO Benchmark with Multi-Compartment GIF Hooks...")
    sho_sim = Simulator(sho_engine, dt, T_max)
    sho_sim.run(sho_state, hook_fn=combined_hook, m=500)
    
    # Render final GIF
    gif_hook.render()
    print("SHO visualization complete.")
    
if __name__ == "__main__":
    main()
