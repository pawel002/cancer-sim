# Cancer Growth Simulation — Surrogate Benchmarking Environment

## Overview

This repository is a complete benchmarking framework for **performance-efficient
surrogate models** for simulating 2D tumour growth with spatial hypoxia under
fractionated radiotherapy.
The paper associated with this codebase is in `latex/main.tex`.

The ground-truth model is a reaction-diffusion PDE on a 2D grid.
Five surrogate methods are implemented and compared:

| Method | Type | Key idea |
|--------|------|----------|
| **ODE** | Analytical reduction | Spatial average of the PDE |
| **MLP** | Black-box ML | Learned time-stepper on ODE trajectories |
| **NODE** | Hybrid physics+ML | ODE baseline + learned correction, integrated as ODE |
| **PINN** | Physics-constrained ML | PDE residual loss + data loss |
| **SuperNet** | Teacher–student | PINN teacher → fast ODE student via residual distillation |

---

## Project Structure

```
cancer-sim/
├── src/
│   ├── core/               # Simulation framework
│   │   ├── state.py        # State dataclass (u, mass, oxygen, …)
│   │   ├── simulator.py    # Main loop: Simulator.run(initial_state, hook_fn)
│   │   └── protocols.py    # Radiation schedules and spatial masks
│   ├── engines/            # All simulation engines (Strategy pattern)
│   │   ├── base.py         # BaseEngine ABC – all engines implement step()
│   │   ├── pde_engine.py   # Finite-difference 2D PDE (ground truth)
│   │   ├── ode_engine.py   # Spatially-averaged ODE surrogate
│   │   ├── sho_engine.py   # Normoxia–Hypoxia–Oxygen coupled system
│   │   └── neural/
│   │       ├── mlp_engine.py       # MLP time-stepper
│   │       ├── node_engine.py      # Neural ODE (inference)
│   │       ├── pinn_engine.py      # PINN evaluator (full 2D field)
│   │       └── supernet_engine.py  # SuperNet (ODE + learned residual)
│   ├── training/           # Training pipelines for neural engines
│   │   ├── generate_data.py        # ScenarioConfig + PDE/ODE data runners
│   │   ├── train_mlp.py            # MLP training (ODE data, multi-scenario)
│   │   ├── train_node.py           # NODE training (Euler rollout on PDE data)
│   │   ├── train_pinn.py           # PINN training (PDE residual + mass data)
│   │   └── train_supernet.py       # SuperNet training (PINN teacher)
│   ├── assimilation/       # Data assimilation
│   │   ├── abc_sampler.py          # Approximate Bayesian Computation
│   │   └── var4d_solver.py         # 4D-Var (L-BFGS-B)
│   ├── visualize/
│   │   ├── hooks.py        # GifHeatmapHook, DualPanelGifHook, …
│   │   └── style.py        # Matplotlib scientific style
│   └── scripts/            # Executable entry points
│       ├── run_training.py         # Train all neural models
│       ├── run_comparison.py       # Run all engines + generate plots/GIF
│       ├── run_pde_viz.py          # PDE-only animated GIF
│       ├── run_ode_viz.py          # ODE mass trajectory plot
│       ├── run_sho_viz.py          # SHO model GIF
│       └── run_benchmarks.py       # PDE vs ODE timing benchmark
├── latex/
│   ├── main.tex            # 15-page scientific paper (LaTeX)
│   ├── refs.bib            # Bibliography (~30 references)
│   ├── Makefile            # Compile: make all
│   ├── generate_figures.py # Run benchmark + generate all paper figures
│   ├── results.json        # Numeric benchmark results
│   └── figures/            # Generated PNG figures for the paper
└── experiments/
    ├── models/             # Trained model checkpoints (.pt files)
    ├── comparison/         # Comparison plots and GIFs
    ├── pde_viz/
    ├── ode_viz/
    └── sho_viz/
```

---

## How to Run

### 1 — Install dependencies

```bash
uv sync
```

### 2 — Generate benchmark figures and train all models

```bash
uv run python latex/generate_figures.py
```

This (~2 minutes on CPU) trains all neural surrogates, runs all engines,
saves model checkpoints to `experiments/models/`, and saves all paper figures
to `latex/figures/`.

### 3 — Full-quality training (optional, slower)

```bash
uv run python -m src.scripts.run_training
```

Uses higher epoch counts (2000 for PINN) for better accuracy.

### 4 — Run comparison and create GIF

```bash
# Load trained models and produce comparison plot + dual-panel animated GIF
uv run python -m src.scripts.run_comparison

# Or auto-train + compare in one step:
uv run python -m src.scripts.run_comparison --train
```

Outputs:
- `experiments/comparison/mass_comparison.png` — mass trajectory comparison
- `experiments/comparison/evolution.gif` — animated GIF (PDE density + trajectories)

### 5 — Run individual visualisations

```bash
uv run python -m src.scripts.run_pde_viz    # 2D PDE animated GIF
uv run python -m src.scripts.run_sho_viz    # Normoxia/Hypoxia/Oxygen GIF
uv run python -m src.scripts.run_ode_viz    # ODE mass trajectory plot
```

### 6 — Compile the paper

```bash
# Requires LaTeX (MacTeX, TeX Live, or TinyTeX)
# Install TinyTeX: curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
cd latex && make
```

---

## Key Design Principles

### Strategy pattern — pluggable engines

Every engine implements a single method:
```python
def step(current_state: State, t: float, dt: float) -> State
```
The `Simulator` loop calls this method; hooks (visualisation, logging) are
called every `m` steps.

### Normalisation

The ODE-based surrogates (ODE, MLP, NODE, SuperNet) track the **normalised
spatial average** `U(t) = M(t) / L²  ∈ [0, 1]`, not the total mass.
This ensures the logistic term `ρU(1−U)` is well-posed.
The PINN outputs the full density field; its mass is normalised for comparison.

### Training window vs. extrapolation window

- **Training / assimilation window**: `[0, 20]` days — neural models are fitted here.
- **Extrapolation window**: `(20, 40]` days — tests generalisation.
  Shaded regions in all plots make this distinction explicit.

---

## Benchmark Results (fast run, 400 epochs)

| Method | RMSE (train) | RMSE (test) | Inf (ms/step) | Speedup | Train (s) |
|--------|-------------|------------|-------------|---------|-----------|
| ODE      | 1.44e-1 | 1.05e-2 | 0.054 | 0.7× | — |
| MLP      | 3.58e-1 | 4.61e-1 | 0.014 | 2.8× | 30.6 |
| NODE     | 8.88e-3 | 5.35e-2 | 0.047 | 0.8× | 52.3 |
| PINN     | 1.90e-2 | 2.04e-2 | 0.568 | 0.07× | 4.8 |
| SuperNet | 5.75e-2 | 4.78e-2 | 0.018 | 2.2× | 0.4 |

*RMSE is for normalised spatial average `U = M/L²`. Speedup relative to PDE
(0.039 ms/step on 50×50 grid). Speedup grows ~16× on a 200×200 grid.*

**Key takeaways:**
- **NODE** achieves lowest in-window RMSE (physics-constrained, best fit)
- **PINN** is most consistent across train/test (PDE residual regularises extrapolation)
- **MLP** extrapolates poorly (no physics constraint → drift outside training window)
- **SuperNet** is the fastest to train (0.4 s) and offers a good accuracy-efficiency balance
- **Scalar surrogates** become decisively faster than PDE on grids larger than ~100×100

---

## Extending the Framework

### Adding a new engine

```python
from src.engines.base import BaseEngine
from src.core.state import State

class MyEngine(BaseEngine):
    def step(self, current_state: State, t: float, dt: float) -> State:
        # ... your surrogate logic ...
        return State(u=u_next, mass=mass_next)
```

### Adding a new visualisation hook

```python
class MyHook:
    def __call__(self, step_idx, t, state, sim) -> bool:
        # called every m steps; return True to stop simulation
        ...
    def render(self):
        # called after simulation to save outputs
        ...
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Neural network training and inference |
| `torchdiffeq` | ODE integration via Neural ODE |
| `scipy` | PDE Laplacian, interpolation, optimisation (4D-Var) |
| `numpy` | Numerical arrays |
| `matplotlib` + `seaborn` | Visualisations |
| `imageio` | GIF compilation |
| `SALib` | Sensitivity analysis (Morris, Sobol) |
| `pydantic` | Configuration validation |
