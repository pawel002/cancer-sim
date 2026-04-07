"""
Orchestrates training of all surrogate models for the canonical scenario.

Usage
-----
    python -m src.scripts.run_training

Saves checkpoints under experiments/models/:
    mlp.pt        – MLP time-stepper
    node.pt       – Neural ODE
    pinn.pt       – PINN
    supernet.pt   – SuperNet residual network
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.training.generate_data import ScenarioConfig
from src.training.train_mlp import save_mlp, train_mlp
from src.training.train_node import save_node, train_node
from src.training.train_pinn import save_pinn, train_pinn
from src.training.train_supernet import save_supernet, train_supernet

MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../experiments/models")
)


def main() -> None:
    cfg = ScenarioConfig()

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Model checkpoints will be saved to: {MODEL_DIR}\n")
    print("=" * 60)
    print(f"Scenario: L={cfg.L}, N={cfg.N}, dt={cfg.dt}, T={cfg.T_max}")
    print(f"Physics:  ρ={cfg.rho}, β={cfg.beta}, D={cfg.D}, K={cfg.K}")
    print(f"Radiation: [{cfg.t_rad_start}, {cfg.t_rad_end}]")
    print(f"Training window: [{cfg.train_start}, {cfg.train_end}]")
    print("=" * 60, "\n")

    # ── MLP ────────────────────────────────────────────────────────────────
    mlp = train_mlp(
        n_traj=300, epochs=300, batch_size=1024, lr=1e-3,
        T_train=cfg.train_end, dt=cfg.dt,
    )
    save_mlp(mlp, os.path.join(MODEL_DIR, "mlp.pt"))
    print()

    # ── NODE ───────────────────────────────────────────────────────────────
    node_func = train_node(
        cfg, epochs=500, lr=3e-3, seq_len=40, n_seqs=20
    )
    save_node(node_func, os.path.join(MODEL_DIR, "node.pt"))
    print()

    # ── PINN ───────────────────────────────────────────────────────────────
    pinn_model = train_pinn(
        cfg, epochs=2000, n_coll=1500, n_data_times=30,
        lr=1e-3, w_pde=1.0, w_ic=10.0, w_bc=5.0, w_data=20.0,
        n_layers=5,
    )
    save_pinn(pinn_model, os.path.join(MODEL_DIR, "pinn.pt"), cfg, n_layers=5)
    print()

    # ── SuperNet (uses trained PINN as teacher) ────────────────────────────
    g_phi = train_supernet(
        cfg, pinn_model=pinn_model,
        T_sc=cfg.T_max, L_sc=cfg.L,
        epochs=500, lr=1e-3,
    )
    save_supernet(g_phi, os.path.join(MODEL_DIR, "supernet.pt"))
    print()

    print("=" * 60)
    print("All models trained and saved. Run run_comparison.py next.")
    print("=" * 60)


if __name__ == "__main__":
    main()
