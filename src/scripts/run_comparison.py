"""
Runs all surrogate engines against the PDE ground truth and produces:
  1. experiments/comparison/mass_comparison.png  – static trajectory plot
  2. experiments/comparison/evolution.gif        – dual-panel animated GIF
  3. Console metrics table (RMSE per engine in train / test windows)

Usage
-----
    python -m src.scripts.run_comparison [--train]

    --train   auto-trains all models before comparing (equivalent to running
              run_training.py first)
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse

import imageio
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.core.simulator import Simulator
from src.core.state import State
from src.training.generate_data import (
    ScenarioConfig,
    make_gaussian_ic,
    make_scalar_rad,
    run_ode_scenario,
    run_pde_scenario,
)
from src.visualize.style import apply_scientific_style

MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../experiments/models")
)
OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../experiments/comparison")
)


# ---------------------------------------------------------------------------
# Engine builders (lazy imports to avoid circular deps)
# ---------------------------------------------------------------------------

def _build_mlp_engine(cfg: ScenarioConfig) -> object:
    from src.engines.neural.mlp_engine import MLPEngine
    from src.training.train_mlp import load_mlp

    model = load_mlp(os.path.join(MODEL_DIR, "mlp.pt"))
    return MLPEngine(
        model=model,
        rho=cfg.rho,
        beta=cfg.beta,
        radiation_protocol=make_scalar_rad(cfg),
    )


def _build_node_engine(cfg: ScenarioConfig) -> object:
    from src.training.train_node import load_node
    return load_node(os.path.join(MODEL_DIR, "node.pt"), cfg)


def _build_pinn_engine(cfg: ScenarioConfig) -> object:
    from src.engines.neural.pinn_engine import PINNEngine
    from src.training.train_pinn import load_pinn

    model, T_sc, L_sc = load_pinn(os.path.join(MODEL_DIR, "pinn.pt"))
    grid_x = cfg.grid_x
    return PINNEngine(
        model=model,
        grid_x=grid_x,
        grid_y=grid_x,
        dx=cfg.dx,
        t_scale=T_sc,
        x_scale=L_sc,
        radiation_protocol=make_scalar_rad(cfg),
    )


def _build_supernet_engine(cfg: ScenarioConfig) -> object:
    from src.engines.neural.supernet_engine import SuperNetEngine
    from src.training.train_supernet import load_supernet

    g_phi = load_supernet(os.path.join(MODEL_DIR, "supernet.pt"))
    return SuperNetEngine(
        rho=cfg.rho,
        beta=cfg.beta,
        H_eff=cfg.H_eff_value(),
        g_phi=g_phi,
        radiation_protocol=make_scalar_rad(cfg),
    )


# ---------------------------------------------------------------------------
# Run a scalar-mass engine for the full simulation window
# ---------------------------------------------------------------------------

def run_scalar_engine(engine: object, cfg: ScenarioConfig, mass0: float) -> np.ndarray:
    """Returns mass array (n_steps+1,) for a scalar ODE-like engine."""
    state0 = State(u=mass0, mass=mass0)
    hist = Simulator(engine, cfg.dt, cfg.T_max).run(state0)  # type: ignore[arg-type]
    return np.array([s.mass for s in hist])


def run_node_rollout(func: object, cfg: ScenarioConfig, mass0: float, com0: float) -> np.ndarray:
    """Returns mass array for the TrainableNODE via custom Euler rollout."""
    import torch

    from src.training.train_node import euler_rollout

    n = cfg.n_steps
    times = np.linspace(0.0, cfg.T_max, n + 1)
    t_t = torch.tensor(times, dtype=torch.float32)
    y0 = torch.tensor([mass0, com0], dtype=torch.float32)

    with torch.no_grad():
        pred = euler_rollout(func, y0, t_t)  # type: ignore[arg-type]

    return pred[:, 0].numpy()


def run_pinn_engine(engine: object, cfg: ScenarioConfig, u0: np.ndarray) -> np.ndarray:
    """Returns mass array for the PINNEngine."""
    mass0 = float(np.sum(u0) * cfg.dx ** 2)
    state0 = State(u=u0, mass=mass0)
    hist = Simulator(engine, cfg.dt, cfg.T_max).run(state0)  # type: ignore[arg-type]
    return np.array([s.mass for s in hist])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    diff = pred[mask] - true[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


# ---------------------------------------------------------------------------
# Static comparison plot
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    "PDE":      "#2b2d42",
    "ODE":      "#8d99ae",
    "MLP":      "#ef233c",
    "NODE":     "#f77f00",
    "PINN":     "#4cc9f0",
    "SuperNet": "#7209b7",
}


def plot_comparison(
    times: np.ndarray,
    trajectories: dict[str, np.ndarray],
    cfg: ScenarioConfig,
    save_path: str,
) -> None:
    apply_scientific_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axvspan(
        cfg.train_start, cfg.train_end,
        alpha=0.10, color="#f4a261",
        label=f"Training window [{cfg.train_start}, {cfg.train_end}]"
    )
    ax.axvline(cfg.t_rad_start, color="#adb5bd", lw=1.0, ls=":", alpha=0.8)
    ax.axvline(cfg.t_rad_end,   color="#adb5bd", lw=1.0, ls=":", alpha=0.8)

    for name, masses in trajectories.items():
        lw = 2.5 if name == "PDE" else 1.8
        ls = "-" if name == "PDE" else "--"
        ax.plot(times, masses, color=COLORS.get(name, None),
                lw=lw, ls=ls, label=name, alpha=0.92)

    # Radiation annotation
    rad_patch = mpatches.Patch(color="#adb5bd", alpha=0.5, label="Radiation window")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [rad_patch], fontsize=9, loc="upper right")

    ax.set_xlabel("Time")
    ax.set_ylabel("Total tumour mass M(t)")
    ax.set_title("Surrogate comparison — PDE vs scalar surrogates")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Comparison plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Dual-panel GIF
# ---------------------------------------------------------------------------

def create_dual_gif(
    u_pde_fields: list[np.ndarray],
    times: np.ndarray,
    trajectories: dict[str, np.ndarray],
    cfg: ScenarioConfig,
    save_path: str,
    step: int = 15,
    fps: int = 8,
) -> None:
    """Creates dual-panel GIF: PDE density (left) + mass comparison (right)."""
    apply_scientific_style()
    frames_dir = os.path.join(os.path.dirname(save_path), ".frames_dual_tmp")
    os.makedirs(frames_dir, exist_ok=True)

    t_max_plot = times[-1]

    indices = list(range(0, len(times), step))
    print(f"Creating dual-panel GIF ({len(indices)} frames)…")

    for frame_i, idx in enumerate(indices):
        t_i = times[idx]
        u_i = u_pde_fields[idx]

        fig, (ax_map, ax_mass) = plt.subplots(1, 2, figsize=(13, 5))

        # Left: density map
        im = ax_map.imshow(
            u_i, origin="lower", vmin=0, vmax=1.0,
            cmap="inferno", interpolation="bilinear"
        )
        plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
        ax_map.set_title(f"PDE tumour density   t = {t_i:.1f}", fontsize=11)
        ax_map.set_xticks([])
        ax_map.set_yticks([])

        # Right: mass trajectories
        ax_mass.axvspan(cfg.train_start, cfg.train_end,
                        alpha=0.12, color="#f4a261", label="Training window")
        ax_mass.axvline(t_i, color="#333333", lw=1.0, ls="--", alpha=0.6)

        for name, masses in trajectories.items():
            mask = times <= t_i + 1e-9
            lw = 2.5 if name == "PDE" else 1.8
            ls = "-" if name == "PDE" else "--"
            ax_mass.plot(
                times[mask], masses[mask],
                color=COLORS.get(name, None),
                lw=lw, ls=ls, label=name, alpha=0.9,
            )

        ax_mass.set_xlabel("Time")
        ax_mass.set_ylabel("Total tumour mass")
        ax_mass.set_title("Mass trajectory comparison")
        ax_mass.legend(fontsize=8, loc="upper right")
        ax_mass.set_xlim(0, t_max_plot)
        ax_mass.set_ylim(bottom=0)
        ax_mass.grid(True, alpha=0.25)

        plt.tight_layout()
        frame_path = os.path.join(frames_dir, f"frame_{frame_i:05d}.png")
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)

    print("Compiling GIF…")
    with imageio.get_writer(save_path, mode="I", fps=fps) as writer:
        for frame_i in range(len(indices)):
            p = os.path.join(frames_dir, f"frame_{frame_i:05d}.png")
            writer.append_data(imageio.imread(p))

    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"GIF saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Surrogate model benchmark")
    parser.add_argument("--train", action="store_true", help="Train all models first")
    args = parser.parse_args()

    if args.train:
        from src.scripts.run_training import main as train_main
        train_main()

    cfg = ScenarioConfig()
    u0 = make_gaussian_ic(cfg)
    mass0 = float(np.sum(u0) * cfg.dx ** 2)

    # ── Ground-truth PDE ────────────────────────────────────────────────────
    print("\nRunning PDE ground truth…")
    t0 = time.perf_counter()
    times_pde, masses_pde, u_pde_fields = run_pde_scenario(cfg)
    print(f"  PDE done in {time.perf_counter() - t0:.1f}s")

    # ── ODE baseline ────────────────────────────────────────────────────────
    print("Running ODE baseline…")
    _, masses_ode = run_ode_scenario(cfg)

    trajectories: dict[str, np.ndarray] = {"PDE": masses_pde, "ODE": masses_ode}
    timings: dict[str, float] = {}

    # ── MLP ─────────────────────────────────────────────────────────────────
    if os.path.exists(os.path.join(MODEL_DIR, "mlp.pt")):
        print("Running MLP engine…")
        engine = _build_mlp_engine(cfg)
        t0 = time.perf_counter()
        trajectories["MLP"] = run_scalar_engine(engine, cfg, mass0)
        timings["MLP"] = time.perf_counter() - t0
    else:
        print("[WARN] mlp.pt not found – skipping MLP.")

    # ── NODE ────────────────────────────────────────────────────────────────
    if os.path.exists(os.path.join(MODEL_DIR, "node.pt")):
        print("Running NODE engine…")
        from src.training.generate_data import compute_center_of_mass
        com0 = float(compute_center_of_mass(u_pde_fields[:1], cfg)[0])
        func = _build_node_engine(cfg)
        t0 = time.perf_counter()
        trajectories["NODE"] = run_node_rollout(func, cfg, mass0, com0)
        timings["NODE"] = time.perf_counter() - t0
    else:
        print("[WARN] node.pt not found – skipping NODE.")

    # ── PINN ────────────────────────────────────────────────────────────────
    if os.path.exists(os.path.join(MODEL_DIR, "pinn.pt")):
        print("Running PINN engine…")
        engine_pinn = _build_pinn_engine(cfg)
        t0 = time.perf_counter()
        trajectories["PINN"] = run_pinn_engine(engine_pinn, cfg, u0)
        timings["PINN"] = time.perf_counter() - t0
    else:
        print("[WARN] pinn.pt not found – skipping PINN.")

    # ── SuperNet ────────────────────────────────────────────────────────────
    if os.path.exists(os.path.join(MODEL_DIR, "supernet.pt")):
        print("Running SuperNet engine…")
        engine_sn = _build_supernet_engine(cfg)
        t0 = time.perf_counter()
        trajectories["SuperNet"] = run_scalar_engine(engine_sn, cfg, mass0)
        timings["SuperNet"] = time.perf_counter() - t0
    else:
        print("[WARN] supernet.pt not found – skipping SuperNet.")

    # ── Metrics ─────────────────────────────────────────────────────────────
    t_arr = times_pde
    train_mask = (t_arr >= cfg.train_start) & (t_arr <= cfg.train_end)
    test_mask  = t_arr > cfg.train_end

    print("\n" + "=" * 64)
    print(f"{'Engine':<12} {'RMSE (train)':<18} {'RMSE (test)':<18} {'Time (s)'}")
    print("-" * 64)
    for name, masses in trajectories.items():
        if name == "PDE":
            continue
        r_tr = rmse(masses, masses_pde, train_mask)
        r_te = rmse(masses, masses_pde, test_mask)
        t_s  = timings.get(name, float("nan"))
        print(f"{name:<12} {r_tr:<18.4e} {r_te:<18.4e} {t_s:.3f}")
    print("=" * 64)

    # ── Static plot ─────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_comparison(
        times_pde, trajectories, cfg,
        save_path=os.path.join(OUT_DIR, "mass_comparison.png"),
    )

    # ── Dual-panel GIF ──────────────────────────────────────────────────────
    create_dual_gif(
        u_pde_fields, times_pde, trajectories, cfg,
        save_path=os.path.join(OUT_DIR, "evolution.gif"),
        step=20, fps=8,
    )


if __name__ == "__main__":
    main()
