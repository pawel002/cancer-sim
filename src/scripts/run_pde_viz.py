import os
import sys
import numpy as np

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.state import State
from src.core.simulator import Simulator
from src.core.protocols import uniform_spatial_radiation
from src.engines.pde_engine import PDEEngine
from src.visualize.hooks import GifHeatmapHook

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

    # Initialization
    center = L / 2.0
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    r_sq = (X - center)**2 + (Y - center)**2
    
    u_init = 0.1 * np.exp(-r_sq / 2.0)
    H_map = np.exp(-r_sq / 5.0)
    mass_init = float(np.sum(u_init) * (dx**2))

    # Target directories
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../experiments/pde_viz'))
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'pde_simulation.gif')

    def rad_protocol_spatial(shape, t):
        return uniform_spatial_radiation(shape, t, start=10.0, end=15.0)

    pde_engine = PDEEngine(D=D, rho=rho, K=K, beta=beta, dx=dx, 
                           radiation_protocol=rad_protocol_spatial, hypoxia_map=H_map)
    pde_state = State(u=u_init, mass=mass_init)

    # Visualization Hook
    # We record every 50 frames to make the gif compact but reasonably fluent.
    gif_hook = GifHeatmapHook(save_path=save_path, attr_name='u', title='PDE Density', fps=10, vmax=1.0)

    def combined_hook(step, t, state, sim):
        gif_hook(step, t, state, sim)

    print("Running PDE Benchmark with GIF Visualization Hooks...")
    pde_sim = Simulator(pde_engine, dt, T_max)
    pde_sim.run(pde_state, hook_fn=combined_hook, m=100)
    
    # Render final GIF
    gif_hook.render()
    print("PDE visualization complete.")
    
if __name__ == "__main__":
    main()
