from sim_config import SimConfig
from sim_engine import SimEngine
from sim_hooks import GifPlotter, log_progress

if __name__ == "__main__":
    config = SimConfig(DIM=2, N=100, dt=0.05, T_max=150.0)
    engine = SimEngine(config)

    engine.register_hook(log_progress, every_n_steps=50)

    plotter = GifPlotter("experiments/experiment1")
    engine.register_hook(plotter.update, every_n_steps=10)

    engine.run()

    plotter.save_gif()
