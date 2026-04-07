import os
import sys
import numpy as np

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.state import State
from src.core.simulator import Simulator
from src.core.protocols import standard_radiation_schedule
from src.engines.ode_engine import ODEEngine
from src.visualize.hooks import LinearPlotHook

def main():
    dt = 0.01
    T_max = 20.0
    rho = 0.1
    beta = 0.5
    H_eff = 1.0
    mass_init = 0.1
    
    # Target directories
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../experiments/ode_viz'))
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'ode_mass_plot.png')

    def rad_protocol_scalar(t):
        return standard_radiation_schedule(t, start=10.0, end=15.0)

    ode_engine = ODEEngine(rho=rho, beta=beta, H_eff=H_eff, 
                           radiation_protocol=rad_protocol_scalar)

    ode_state = State(u=mass_init, mass=mass_init)

    # Initialize plotting hook
    plot_hook = LinearPlotHook(save_path=save_path, title="ODE Surrogate Tumor Mass (Radiation 10-15d)")
    
    # Wrapper hook to combine any custom logic with the plotter
    def combined_hook(step, t, state, sim):
        plot_hook(step, t, state, sim)

    print("Running ODE Benchmark with Visualization...")
    ode_sim = Simulator(ode_engine, dt, T_max)
    ode_sim.run(ode_state, hook_fn=combined_hook, m=20)  # Plot every 20 steps
    
    # Render final plot
    plot_hook.render()
    print("ODE visualization complete.")
    
if __name__ == "__main__":
    main()
