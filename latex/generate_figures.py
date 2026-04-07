"""
Fast benchmark runner for the cancer-surrogate paper.
All surrogate methods are compared in *normalised* units (spatial average
U(t) = M(t) / L²  ∈ [0, 1]).  For PINN, total mass is divided by L² after
inference; for scalar engines the initial condition is the normalised mass.

Usage:  uv run python latex/generate_figures.py
Output: latex/figures/*.png   (paper figures)
        latex/results.json    (numeric results for tables)
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.training.generate_data import (
    ScenarioConfig,
    run_pde_scenario,
    run_ode_scenario,
    compute_center_of_mass,
    make_gaussian_ic,
    make_hypoxia_map,
    make_scalar_rad,
)
from src.training.train_mlp import train_mlp, save_mlp
from src.training.train_node import train_node, save_node, euler_rollout
from src.training.train_pinn import train_pinn, save_pinn
from src.training.train_supernet import train_supernet, save_supernet
from src.engines.neural.mlp_engine import MLPEngine
from src.engines.neural.pinn_engine import PINNEngine
from src.engines.neural.supernet_engine import SuperNetEngine
from src.core.state import State
from src.core.simulator import Simulator

# ── Output directories ────────────────────────────────────────────────────────
FIGS   = os.path.join(os.path.dirname(__file__), "figures")
MODELS = os.path.join(os.path.dirname(__file__), "..", "experiments", "models")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# ── Matplotlib style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 150,
    "savefig.dpi": 200, "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
})

COLORS = {
    "PDE":      "#2b2d42",
    "ODE":      "#6c757d",
    "MLP":      "#e63946",
    "NODE":     "#f4a261",
    "PINN":     "#2196f3",
    "SuperNet": "#7209b7",
}
MARKERS = {"PDE": None, "ODE": None, "MLP": "o", "NODE": "s", "PINN": "^", "SuperNet": "D"}

cfg = ScenarioConfig()
L_SQ = cfg.L ** 2        # Normalisation factor  M_total / L²  → spatial avg

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def run_scalar(engine: object, u0_norm: float) -> np.ndarray:
    """Run a scalar-mass engine initialised with normalised spatial average."""
    state0 = State(u=u0_norm, mass=u0_norm)
    hist = Simulator(engine, cfg.dt, cfg.T_max).run(state0)  # type: ignore[arg-type]
    return np.array([s.mass for s in hist])


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 – PDE ground truth
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1/6: PDE ground truth …")
t0 = time.perf_counter()
times_pde, masses_pde_total, u_fields = run_pde_scenario(cfg)
dt_pde = time.perf_counter() - t0

# All comparisons use normalised spatial average U = M/L²
masses_pde = masses_pde_total / L_SQ
t_pde_ms = dt_pde / cfg.n_steps * 1000
print(f"  done in {dt_pde:.2f}s  ({t_pde_ms:.4f} ms/step)"
      f"  M0={masses_pde_total[0]:.2f}  U0={masses_pde[0]:.4f}")

u0 = make_gaussian_ic(cfg)
mass0_total = float(np.sum(u0) * cfg.dx ** 2)
u0_norm = mass0_total / L_SQ            # ≈ 0.11 – normalised initial condition
coms = compute_center_of_mass(u_fields, cfg)
com0 = float(coms[0])

# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 – ODE baseline
# ─────────────────────────────────────────────────────────────────────────────
print("Step 2/6: ODE baseline …")
t0 = time.perf_counter()
_, masses_ode = run_ode_scenario(cfg)   # already normalised (spatial average)
t_ode_ms = (time.perf_counter() - t0) / cfg.n_steps * 1000
print(f"  done  U0_ode={masses_ode[0]:.4f}  U_ode_end={masses_ode[-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 – MLP
# ─────────────────────────────────────────────────────────────────────────────
print("Step 3/6: Training MLP …")
t0 = time.perf_counter()
mlp_model = train_mlp(
    n_traj=150, epochs=200, batch_size=512,
    T_train=cfg.train_end, dt=cfg.dt,
)
t_mlp_train = time.perf_counter() - t0
save_mlp(mlp_model, os.path.join(MODELS, "mlp.pt"))

mlp_engine = MLPEngine(model=mlp_model, rho=cfg.rho, beta=cfg.beta,
                        radiation_protocol=make_scalar_rad(cfg))
t0 = time.perf_counter()
masses_mlp = run_scalar(mlp_engine, u0_norm)
t_mlp_ms = (time.perf_counter() - t0) / cfg.n_steps * 1000

# ─────────────────────────────────────────────────────────────────────────────
#  Step 4 – NODE
# ─────────────────────────────────────────────────────────────────────────────
print("Step 4/6: Training NODE …")
t0 = time.perf_counter()
node_func = train_node(cfg, epochs=400, lr=3e-3, seq_len=40, n_seqs=20)
t_node_train = time.perf_counter() - t0
save_node(node_func, os.path.join(MODELS, "node.pt"))

n = cfg.n_steps
t_tensor = torch.tensor(np.linspace(0.0, cfg.T_max, n + 1), dtype=torch.float32)
y0_node = torch.tensor([u0_norm, com0], dtype=torch.float32)
t0 = time.perf_counter()
with torch.no_grad():
    pred_node = euler_rollout(node_func, y0_node, t_tensor)
t_node_ms = (time.perf_counter() - t0) / cfg.n_steps * 1000
masses_node = pred_node[:, 0].numpy()

# ─────────────────────────────────────────────────────────────────────────────
#  Step 5 – PINN
# ─────────────────────────────────────────────────────────────────────────────
print("Step 5/6: Training PINN …")
t0 = time.perf_counter()
pinn_model = train_pinn(
    cfg, epochs=400, n_coll=400, n_ic=200, n_bc=200, n_data_times=8,
    lr=1e-3, w_pde=1.0, w_ic=10.0, w_bc=5.0, w_data=20.0,
    hidden_dim=32, n_layers=3, seed=0,
)
t_pinn_train = time.perf_counter() - t0
save_pinn(pinn_model, os.path.join(MODELS, "pinn.pt"), cfg, n_layers=3)

pinn_engine = PINNEngine(
    model=pinn_model, grid_x=cfg.grid_x, grid_y=cfg.grid_x,
    dx=cfg.dx, t_scale=cfg.T_max, x_scale=cfg.L,
)
state0_pinn = State(u=u0, mass=mass0_total)
t0 = time.perf_counter()
hist_pinn = Simulator(pinn_engine, cfg.dt, cfg.T_max).run(state0_pinn)
t_pinn_ms = (time.perf_counter() - t0) / cfg.n_steps * 1000
# Normalise PINN total-mass output to spatial average for comparison
masses_pinn = np.array([s.mass for s in hist_pinn]) / L_SQ

# ─────────────────────────────────────────────────────────────────────────────
#  Step 6 – SuperNet
# ─────────────────────────────────────────────────────────────────────────────
print("Step 6/6: Training SuperNet …")
t0 = time.perf_counter()
g_phi = train_supernet(
    cfg, pinn_model=pinn_model, T_sc=cfg.T_max, L_sc=cfg.L, epochs=400,
)
t_sn_train = time.perf_counter() - t0
save_supernet(g_phi, os.path.join(MODELS, "supernet.pt"))

sn_engine = SuperNetEngine(
    rho=cfg.rho, beta=cfg.beta, H_eff=cfg.H_eff_value(),
    g_phi=g_phi, radiation_protocol=make_scalar_rad(cfg),
)
t0 = time.perf_counter()
masses_sn = run_scalar(sn_engine, u0_norm)
t_sn_ms = (time.perf_counter() - t0) / cfg.n_steps * 1000

# ─────────────────────────────────────────────────────────────────────────────
#  Collect & report
# ─────────────────────────────────────────────────────────────────────────────
trajectories: dict[str, np.ndarray] = {
    "PDE":      masses_pde,   # normalised spatial average
    "ODE":      masses_ode,
    "MLP":      masses_mlp,
    "NODE":     masses_node,
    "PINN":     masses_pinn,
    "SuperNet": masses_sn,
}
inf_ms: dict[str, float] = {
    "PDE": t_pde_ms, "ODE": t_ode_ms, "MLP": t_mlp_ms,
    "NODE": t_node_ms, "PINN": t_pinn_ms, "SuperNet": t_sn_ms,
}
train_s: dict[str, float] = {
    "MLP": t_mlp_train, "NODE": t_node_train,
    "PINN": t_pinn_train, "SuperNet": t_sn_train,
}

t_tr = (times_pde >= cfg.train_start) & (times_pde <= cfg.train_end)
t_te = times_pde > cfg.train_end

results: dict[str, dict] = {}
for name, m in trajectories.items():
    if name == "PDE":
        continue
    results[name] = {
        "rmse_train": rmse(m[t_tr], masses_pde[t_tr]),
        "rmse_test":  rmse(m[t_te], masses_pde[t_te]),
        "inf_ms":     inf_ms[name],
        "train_s":    train_s.get(name, 0.0),
        "speedup":    t_pde_ms / inf_ms[name],
    }

print("\n" + "=" * 74)
print(f"{'Method':<10} {'RMSE(train)':<14} {'RMSE(test)':<14} "
      f"{'Inf(ms/step)':<14} {'Speedup':<12} {'Train(s)'}")
print("-" * 74)
for n, r in results.items():
    print(f"{n:<10} {r['rmse_train']:<14.4e} {r['rmse_test']:<14.4e} "
          f"{r['inf_ms']:<14.4f} {r['speedup']:<12.1f} {r['train_s']:.1f}")
print("=" * 74)
print(f"\nPDE inference: {t_pde_ms:.4f} ms/step")

with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ═════════════════════════════════════════════════════════════════════════════

# ── Fig 1 – Scenario setup ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
im1 = ax1.imshow(u0, origin="lower", cmap="hot", extent=[0, cfg.L]*2,
                 interpolation="bilinear", vmin=0, vmax=1)
ax1.set_title("Initial condition $u_0(\\mathbf{x})$")
ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")
plt.colorbar(im1, ax=ax1, fraction=0.046)
im2 = ax2.imshow(make_hypoxia_map(cfg), origin="lower", cmap="Blues_r",
                 extent=[0, cfg.L]*2, interpolation="bilinear", vmin=0, vmax=1)
ax2.set_title("Hypoxia map $H(\\mathbf{x})$")
ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
plt.colorbar(im2, ax=ax2, fraction=0.046)
plt.suptitle("Simulation setup (canonical scenario)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_setup.png")); plt.close()
print("Saved fig_setup.png")

# ── Fig 2 – PDE snapshots ─────────────────────────────────────────────────────
snap_t = [0, 5, 10, 15, 20, 25, 30, 40]
snap_idx = [int(t / cfg.dt) for t in snap_t]
fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
for ax, idx, st in zip(axes.ravel(), snap_idx, snap_t):
    im = ax.imshow(u_fields[idx], origin="lower", vmin=0, vmax=1,
                   cmap="inferno", interpolation="bilinear",
                   extent=[0, cfg.L, 0, cfg.L])
    ax.set_title(f"$t={st}$", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.subplots_adjust(right=0.88, hspace=0.15, wspace=0.1)
cbar_ax = fig.add_axes([0.90, 0.15, 0.018, 0.7])
fig.colorbar(im, cax=cbar_ax).set_label("$u(\\mathbf{x},t)$", fontsize=9)
fig.suptitle("PDE ground-truth density snapshots  "
             "(irradiation: $t\\in[10,30]$)", fontsize=10, y=1.01)
plt.savefig(os.path.join(FIGS, "fig_pde_snapshots.png"), bbox_inches="tight")
plt.close(); print("Saved fig_pde_snapshots.png")

# ── Fig 3 – Mass trajectory comparison ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.axvspan(cfg.train_start, cfg.train_end, alpha=0.10, color="#f4a261",
           label=f"Assimilation window $[{cfg.train_start:.0f},{cfg.train_end:.0f}]$")
ax.axvspan(cfg.train_end, cfg.T_max, alpha=0.05, color="#2196f3")
ax.axvline(cfg.t_rad_start, color="#bbb", lw=0.8, ls=":", zorder=1)
ax.axvline(cfg.t_rad_end,   color="#bbb", lw=0.8, ls=":", zorder=1)
lss = {"PDE": "-", "ODE": "-.", "MLP": "--", "NODE": "--", "PINN": "--", "SuperNet": "--"}
for name, masses in trajectories.items():
    lw = 2.4 if name == "PDE" else 1.6
    ax.plot(times_pde, masses, color=COLORS[name], lw=lw, ls=lss[name],
            label=name, alpha=0.92)
ax.text((cfg.t_rad_start + cfg.t_rad_end) / 2, masses_pde.max() * 1.01,
        "Irradiation", ha="center", fontsize=7.5, color="#777")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Spatial average $U(t) = M(t)/L^2$")
ax.set_title("Mass trajectory comparison – all surrogate methods")
ax.legend(ncol=2, fontsize=8.5, loc="upper right")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_mass_comparison.png")); plt.close()
print("Saved fig_mass_comparison.png")

# ── Fig 4 – RMSE bar chart ────────────────────────────────────────────────────
methods = list(results.keys())
r_tr = [results[m]["rmse_train"] for m in methods]
r_te = [results[m]["rmse_test"]  for m in methods]
x = np.arange(len(methods))
w = 0.35
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - w/2, r_tr, w, label="Training window",
       color=[COLORS[m] for m in methods], alpha=0.7, edgecolor="k", lw=0.6)
ax.bar(x + w/2, r_te, w, label="Test window",
       color=[COLORS[m] for m in methods], alpha=1.0, edgecolor="k", lw=0.6, hatch="///")
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylabel("RMSE (normalised spatial average)")
ax.set_title("Accuracy: training vs. test window")
ax.legend(); ax.set_yscale("log"); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_rmse_bar.png")); plt.close()
print("Saved fig_rmse_bar.png")

# ── Fig 5 – Accuracy–efficiency frontier ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for name, r in results.items():
    ax.scatter(r["inf_ms"], r["rmse_test"], color=COLORS[name], s=90, zorder=5,
               edgecolor="k", lw=0.6, marker=MARKERS.get(name, "o"))
    ax.annotate(name, (r["inf_ms"], r["rmse_test"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8.5)
ax.scatter(t_pde_ms, 1e-10, color=COLORS["PDE"], s=110, marker="*", zorder=5,
           edgecolor="k", lw=0.6)
ax.annotate("PDE (ref)", (t_pde_ms, 1e-10),
            textcoords="offset points", xytext=(6, 6), fontsize=8.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Inference time per step (ms)  [log]")
ax.set_ylabel("RMSE in test window  [log]")
ax.set_title("Accuracy–efficiency frontier")
ax.grid(True, which="both", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_frontier.png")); plt.close()
print("Saved fig_frontier.png")

# ── Fig 6 – PINN spatial reconstruction ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
for col, st in enumerate([10, 20, 30]):
    idx = int(st / cfg.dt)
    u_true = u_fields[idx]
    state_dummy = State(u=u_true, mass=mass0_total)
    state_p = pinn_engine.step(state_dummy, float(st), cfg.dt)
    u_pinn = state_p.u
    vm = max(float(u_true.max()), 0.01)
    kw = dict(origin="lower", vmin=0, vmax=vm, cmap="inferno",
              interpolation="bilinear", extent=[0, cfg.L]*2)
    axes[0, col].imshow(u_true, **kw)
    axes[0, col].set_title(f"PDE  $t={st}$", fontsize=9)
    im2 = axes[1, col].imshow(u_pinn, **kw)
    axes[1, col].set_title(f"PINN  $t={st}$", fontsize=9)
    for row in range(2):
        axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
fig.subplots_adjust(right=0.87, hspace=0.15, wspace=0.12)
cbar_ax = fig.add_axes([0.89, 0.15, 0.018, 0.7])
fig.colorbar(im2, cax=cbar_ax).set_label("$u(\\mathbf{x},t)$", fontsize=9)
fig.suptitle("PINN spatial reconstruction vs. PDE ground truth", fontsize=10, y=1.02)
plt.savefig(os.path.join(FIGS, "fig_pinn_reconstruction.png"), bbox_inches="tight")
plt.close(); print("Saved fig_pinn_reconstruction.png")

# ── Fig 7 – Speedup bar chart ─────────────────────────────────────────────────
sp_names = [k for k in inf_ms if k != "PDE"]
speedups  = [t_pde_ms / inf_ms[k] for k in sp_names]
fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.barh(sp_names, speedups, color=[COLORS[k] for k in sp_names],
               edgecolor="k", lw=0.6)
ax.set_xscale("log")
ax.set_xlabel("Speedup vs PDE solver (log scale)")
ax.set_title("Inference speedup relative to PDE ground truth")
for bar, val in zip(bars, speedups):
    ax.text(max(val * 1.05, 1.1), bar.get_y() + bar.get_height()/2,
            f"{val:.0f}×", va="center", fontsize=8.5)
ax.grid(True, axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_speedup.png")); plt.close()
print("Saved fig_speedup.png")

# ── Fig 8 – Training-window illustration ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.fill_between(times_pde, 0, masses_pde, alpha=0.12, color=COLORS["PDE"])
ax.plot(times_pde, masses_pde, color=COLORS["PDE"], lw=2.0, label="PDE ground truth")
ax.axvspan(cfg.train_start, cfg.train_end, alpha=0.18, color="#f4a261",
           label=f"Assimilation window $[{cfg.train_start:.0f},{cfg.train_end:.0f}]$")
ax.axvspan(cfg.train_end, cfg.T_max, alpha=0.07, color="#2196f3",
           label=f"Extrapolation window $[{cfg.train_end:.0f},{cfg.T_max:.0f}]$")
ax.axvline(cfg.t_rad_start, color="#666", ls=":", lw=1.0)
ax.axvline(cfg.t_rad_end,   color="#666", ls=":", lw=1.0)
ax.text((cfg.t_rad_start + cfg.t_rad_end) / 2, 0.001, "Irradiation", ha="center",
        fontsize=8, color="#555")
ax.set_xlabel("Time (days)"); ax.set_ylabel("$U(t)$")
ax.set_title("Benchmark scenario: windows")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_scenario.png")); plt.close()
print("Saved fig_scenario.png")

print(f"\n✓ All {len(os.listdir(FIGS))} figures saved to {FIGS}")
print(f"✓ results.json saved")
