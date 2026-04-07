"""
Benchmark runner for the cancer-surrogate paper.
Trains all models (including the Trajectory Predictor), generates figures
and prints LaTeX-ready results.

Usage:  uv run python latex/generate_figures.py
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
    make_gaussian_ic,
    make_hypoxia_map,
    make_scalar_rad,
    run_ode_scenario,
    run_pde_scenario,
)
from src.training.train_mlp import train_mlp
from src.training.train_node import train_node, euler_rollout
from src.training.train_pinn import train_pinn
from src.training.train_supernet import train_supernet
from src.training.train_trajectory import (
    train_trajectory,
    predict_normalised,
    save_trajectory_model,
)
from src.training.generate_data import compute_center_of_mass
from src.engines.neural.mlp_engine import MLPEngine
from src.engines.neural.pinn_engine import PINNEngine
from src.engines.neural.supernet_engine import SuperNetEngine
from src.core.state import State
from src.core.simulator import Simulator

FIGS   = os.path.join(os.path.dirname(__file__), "figures")
MODELS = os.path.join(os.path.dirname(__file__), "..", "experiments", "models")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

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
    "PDE": "#2b2d42", "ODE": "#6c757d", "MLP": "#e63946",
    "NODE": "#f4a261", "PINN": "#2196f3", "SuperNet": "#7209b7",
    "TrajNet": "#2d6a4f",
}

cfg = ScenarioConfig()
L_SQ = cfg.L ** 2

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def run_scalar(engine: object, u0_norm: float) -> np.ndarray:
    state0 = State(u=u0_norm, mass=u0_norm)
    hist = Simulator(engine, cfg.dt, cfg.T_max).run(state0)  # type: ignore[arg-type]
    return np.array([s.mass for s in hist])

# ═══════════════════════════════════════════════════════════════════════════
#  1.  PDE ground truth
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1/7  PDE ground truth")
t0 = time.perf_counter()
times_pde, masses_pde_total, u_fields = run_pde_scenario(cfg)
dt_pde = time.perf_counter() - t0
masses_pde = masses_pde_total / L_SQ
t_pde_total_ms = dt_pde * 1000           # total PDE wall-clock
t_pde_step_ms  = dt_pde / cfg.n_steps * 1000
print(f"   {dt_pde*1000:.1f} ms total  ({t_pde_step_ms:.4f} ms/step)")

u0 = make_gaussian_ic(cfg)
mass0_total = float(np.sum(u0) * cfg.dx ** 2)
u0_norm = mass0_total / L_SQ
coms = compute_center_of_mass(u_fields, cfg)

# ═══════════════════════════════════════════════════════════════════════════
#  2.  ODE
# ═══════════════════════════════════════════════════════════════════════════
print("2/7  ODE")
t0 = time.perf_counter()
_, masses_ode = run_ode_scenario(cfg)
t_ode_total_ms = (time.perf_counter() - t0) * 1000

# ═══════════════════════════════════════════════════════════════════════════
#  3.  MLP
# ═══════════════════════════════════════════════════════════════════════════
print("3/7  MLP training")
mlp_model = train_mlp(n_traj=150, epochs=200, batch_size=512,
                       T_train=cfg.train_end, dt=cfg.dt)
mlp_engine = MLPEngine(model=mlp_model, rho=cfg.rho, beta=cfg.beta,
                        radiation_protocol=make_scalar_rad(cfg))
t0 = time.perf_counter()
masses_mlp = run_scalar(mlp_engine, u0_norm)
t_mlp_total_ms = (time.perf_counter() - t0) * 1000

# ═══════════════════════════════════════════════════════════════════════════
#  4.  NODE
# ═══════════════════════════════════════════════════════════════════════════
print("4/7  NODE training")
node_func = train_node(cfg, epochs=400, lr=3e-3, seq_len=40, n_seqs=20)
t_tensor = torch.tensor(np.linspace(0, cfg.T_max, cfg.n_steps+1),
                         dtype=torch.float32)
y0_node = torch.tensor([u0_norm, float(coms[0])], dtype=torch.float32)
t0 = time.perf_counter()
with torch.no_grad():
    masses_node = euler_rollout(node_func, y0_node, t_tensor)[:, 0].numpy()
t_node_total_ms = (time.perf_counter() - t0) * 1000

# ═══════════════════════════════════════════════════════════════════════════
#  5.  PINN
# ═══════════════════════════════════════════════════════════════════════════
print("5/7  PINN training")
pinn_model = train_pinn(cfg, epochs=400, n_coll=400, n_ic=200, n_bc=200,
                         n_data_times=8, hidden_dim=32, n_layers=3)
pinn_engine = PINNEngine(model=pinn_model, grid_x=cfg.grid_x,
                          grid_y=cfg.grid_x, dx=cfg.dx,
                          t_scale=cfg.T_max, x_scale=cfg.L)
state0_pinn = State(u=u0, mass=mass0_total)
t0 = time.perf_counter()
hist_pinn = Simulator(pinn_engine, cfg.dt, cfg.T_max).run(state0_pinn)
t_pinn_total_ms = (time.perf_counter() - t0) * 1000
masses_pinn = np.array([s.mass for s in hist_pinn]) / L_SQ

# ═══════════════════════════════════════════════════════════════════════════
#  6.  SuperNet
# ═══════════════════════════════════════════════════════════════════════════
print("6/7  SuperNet training")
g_phi = train_supernet(cfg, pinn_model=pinn_model, T_sc=cfg.T_max,
                        L_sc=cfg.L, epochs=400)
sn_engine = SuperNetEngine(rho=cfg.rho, beta=cfg.beta,
                            H_eff=cfg.H_eff_value(), g_phi=g_phi,
                            radiation_protocol=make_scalar_rad(cfg))
t0 = time.perf_counter()
masses_sn = run_scalar(sn_engine, u0_norm)
t_sn_total_ms = (time.perf_counter() - t0) * 1000

# ═══════════════════════════════════════════════════════════════════════════
#  7.  Trajectory Predictor (main contribution)
# ═══════════════════════════════════════════════════════════════════════════
print("7/7  Trajectory Predictor training (data generation + training)")
traj_model, query_times = train_trajectory(
    cfg, n_samples=1500, n_time_points=100,
    hidden=128, n_layers=3, epochs=300, batch_size=2048,
)
save_trajectory_model(traj_model, query_times,
                       os.path.join(MODELS, "trajnet.pt"))

# Predict the canonical scenario
canon_params = np.array([cfg.rho, cfg.beta, cfg.D, 0.8,
                          cfg.t_rad_start, cfg.t_rad_end], dtype=np.float32)
t0 = time.perf_counter()
for _ in range(100):                       # average over 100 calls
    masses_traj = predict_normalised(traj_model, canon_params, query_times)
t_traj_total_ms = (time.perf_counter() - t0) / 100 * 1000

# Interpolate to the full PDE time-grid for RMSE comparison
masses_traj_full = np.interp(times_pde, query_times, masses_traj)

# ── Generalization test (unseen parameter combos) ────────────────────────
rng = np.random.default_rng(999)
n_test = 100
test_rmses = np.empty(n_test)
for i in range(n_test):
    rho_t  = float(rng.uniform(0.05, 0.6))
    beta_t = float(rng.uniform(0.1, 2.0))
    D_t    = float(rng.uniform(0.002, 0.08))
    amp_t  = float(rng.uniform(0.1, 1.0))
    ts_t   = float(rng.uniform(3, 20))
    te_t   = float(ts_t + rng.uniform(3, min(25, cfg.T_max - ts_t - 1)))

    from src.training.train_trajectory import _run_single_pde
    true_traj = _run_single_pde(cfg, rho_t, beta_t, D_t, amp_t, ts_t, te_t)
    n_full = len(true_traj)
    idx_sub = np.linspace(0, n_full - 1, len(query_times), dtype=int)
    true_sub = true_traj[idx_sub]

    test_p = np.array([rho_t, beta_t, D_t, amp_t, ts_t, te_t], dtype=np.float32)
    pred_sub = predict_normalised(traj_model, test_p, query_times)
    test_rmses[i] = rmse(pred_sub, true_sub)

print(f"\n[TrajNet] Generalization test ({n_test} unseen scenarios):")
print(f"   Median RMSE = {np.median(test_rmses):.4e}")
print(f"   Mean   RMSE = {np.mean(test_rmses):.4e}")
print(f"   P95    RMSE = {np.percentile(test_rmses, 95):.4e}")

# ═══════════════════════════════════════════════════════════════════════════
#  Collect results
# ═══════════════════════════════════════════════════════════════════════════

trajectories = {
    "PDE": masses_pde, "ODE": masses_ode, "MLP": masses_mlp,
    "NODE": masses_node, "PINN": masses_pinn, "SuperNet": masses_sn,
    "TrajNet": masses_traj_full,
}
total_ms = {
    "PDE": t_pde_total_ms, "ODE": t_ode_total_ms, "MLP": t_mlp_total_ms,
    "NODE": t_node_total_ms, "PINN": t_pinn_total_ms,
    "SuperNet": t_sn_total_ms, "TrajNet": t_traj_total_ms,
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
        "total_ms":   total_ms[name],
        "speedup":    t_pde_total_ms / total_ms[name],
    }

results["TrajNet"]["generalization_median"] = float(np.median(test_rmses))
results["TrajNet"]["generalization_p95"]    = float(np.percentile(test_rmses, 95))

print("\n" + "=" * 78)
print(f"{'Method':<10} {'RMSE(train)':<14} {'RMSE(test)':<14} "
      f"{'Total(ms)':<14} {'Speedup'}")
print("-" * 78)
for n, r in results.items():
    print(f"{n:<10} {r['rmse_train']:<14.4e} {r['rmse_test']:<14.4e} "
          f"{r['total_ms']:<14.2f} {r['speedup']:.0f}×")
print("=" * 78)

with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════════════════

# ── Fig 1: Setup ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
im1 = ax1.imshow(u0, origin="lower", cmap="hot",
                 extent=[0, cfg.L]*2, interpolation="bilinear", vmin=0, vmax=1)
ax1.set_title(r"Initial condition $u_0(\mathbf{x})$")
ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")
plt.colorbar(im1, ax=ax1, fraction=0.046)
im2 = ax2.imshow(make_hypoxia_map(cfg), origin="lower", cmap="Blues_r",
                 extent=[0, cfg.L]*2, interpolation="bilinear", vmin=0, vmax=1)
ax2.set_title(r"Hypoxia map $H(\mathbf{x})$")
ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
plt.colorbar(im2, ax=ax2, fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_setup.png")); plt.close()

# ── Fig 2: PDE snapshots ─────────────────────────────────────────────────
snap_t = [0, 5, 10, 15, 20, 25, 30, 40]
snap_idx = [int(t / cfg.dt) for t in snap_t]
fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
for ax, idx, st in zip(axes.ravel(), snap_idx, snap_t):
    im = ax.imshow(u_fields[idx], origin="lower", vmin=0, vmax=1,
                   cmap="inferno", interpolation="bilinear",
                   extent=[0, cfg.L]*2)
    ax.set_title(f"$t={st}$", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.subplots_adjust(right=0.88, hspace=0.15, wspace=0.1)
cbar_ax = fig.add_axes([0.90, 0.15, 0.018, 0.7])
fig.colorbar(im, cax=cbar_ax).set_label(r"$u(\mathbf{x},t)$", fontsize=9)
fig.suptitle("PDE ground-truth density snapshots "
             r"(irradiation: $t\in[10,30]$)", fontsize=10, y=1.01)
plt.savefig(os.path.join(FIGS, "fig_pde_snapshots.png"),
            bbox_inches="tight"); plt.close()

# ── Fig 3: Mass trajectory comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.axvspan(cfg.train_start, cfg.train_end, alpha=0.10, color="#f4a261",
           label=r"Training window $[0, 20]$")
ax.axvspan(cfg.train_end, cfg.T_max, alpha=0.05, color="#2196f3")
ax.axvline(cfg.t_rad_start, color="#bbb", lw=0.8, ls=":")
ax.axvline(cfg.t_rad_end,   color="#bbb", lw=0.8, ls=":")
lss = {"PDE": "-", "ODE": "-.", "MLP": "--", "NODE": "--",
       "PINN": "--", "SuperNet": "--", "TrajNet": "-"}
for name in ["PDE", "ODE", "MLP", "NODE", "PINN", "SuperNet", "TrajNet"]:
    m = trajectories[name]
    lw = 2.4 if name in ("PDE", "TrajNet") else 1.4
    ax.plot(times_pde, m, color=COLORS[name], lw=lw, ls=lss[name],
            label=name, alpha=0.9)
ax.set_xlabel("Time (days)")
ax.set_ylabel(r"Spatial average $U(t)$")
ax.set_title("Mass trajectory comparison — all methods")
ax.legend(ncol=2, fontsize=8); ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_mass_comparison.png")); plt.close()

# ── Fig 4: RMSE bar chart ────────────────────────────────────────────────
methods = list(results.keys())
r_tr = [results[m]["rmse_train"] for m in methods]
r_te = [results[m]["rmse_test"]  for m in methods]
x = np.arange(len(methods))
w = 0.35
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(x - w/2, r_tr, w, label="Training window",
       color=[COLORS[m] for m in methods], alpha=0.7, edgecolor="k", lw=0.5)
ax.bar(x + w/2, r_te, w, label="Test window",
       color=[COLORS[m] for m in methods], alpha=1.0, edgecolor="k",
       lw=0.5, hatch="///")
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylabel("RMSE (normalised spatial average)")
ax.set_title("Accuracy: training vs.\ extrapolation window")
ax.legend(); ax.set_yscale("log"); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_rmse_bar.png")); plt.close()

# ── Fig 5: Speedup bar chart (total trajectory time) ─────────────────────
sp_names = [k for k in results]
speedups = [results[k]["speedup"] for k in sp_names]
fig, ax = plt.subplots(figsize=(6.5, 3.5))
bars = ax.barh(sp_names, speedups, color=[COLORS[k] for k in sp_names],
               edgecolor="k", lw=0.5)
ax.set_xscale("log")
ax.set_xlabel(r"Speedup vs PDE (full trajectory, log scale)")
ax.set_title("Wall-clock speedup — full 40-day trajectory")
ax.axvline(1.0, color="k", ls="--", lw=0.7, alpha=0.4)
for bar, val in zip(bars, speedups):
    label = f"{val:.0f}×" if val >= 1 else f"{val:.2f}×"
    ax.text(max(val * 1.08, 0.15), bar.get_y() + bar.get_height()/2,
            label, va="center", fontsize=8.5)
ax.grid(True, axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_speedup.png")); plt.close()

# ── Fig 6: Accuracy–efficiency frontier ──────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for name, r in results.items():
    ax.scatter(r["total_ms"], r["rmse_test"], color=COLORS[name], s=90,
               zorder=5, edgecolor="k", lw=0.6)
    ax.annotate(name, (r["total_ms"], r["rmse_test"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8.5)
ax.scatter(t_pde_total_ms, 1e-10, color=COLORS["PDE"], s=110, marker="*",
           zorder=5, edgecolor="k", lw=0.6)
ax.annotate("PDE (ref)", (t_pde_total_ms, 1e-10),
            textcoords="offset points", xytext=(6, 6), fontsize=8.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Total inference time for full trajectory (ms) — log")
ax.set_ylabel("RMSE in test window — log")
ax.set_title("Accuracy–efficiency frontier")
ax.grid(True, which="both", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_frontier.png")); plt.close()

# ── Fig 7: TrajNet generalization histogram ──────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(test_rmses, bins=30, color=COLORS["TrajNet"], edgecolor="k",
        lw=0.5, alpha=0.85)
ax.axvline(np.median(test_rmses), color="k", ls="--", lw=1.2,
           label=f"Median = {np.median(test_rmses):.3e}")
ax.axvline(np.percentile(test_rmses, 95), color="#e63946", ls="--", lw=1.2,
           label=f"95th pct = {np.percentile(test_rmses, 95):.3e}")
ax.set_xlabel("RMSE (normalised spatial average)")
ax.set_ylabel("Count")
ax.set_title("TrajNet generalization — 200 unseen parameter combinations")
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_trajnet_generalization.png")); plt.close()

# ── Fig 8: Scenario illustration ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.fill_between(times_pde, 0, masses_pde, alpha=0.12, color=COLORS["PDE"])
ax.plot(times_pde, masses_pde, color=COLORS["PDE"], lw=2.0,
        label="PDE ground truth")
ax.axvspan(cfg.train_start, cfg.train_end, alpha=0.18, color="#f4a261",
           label=r"Training window $[0,20]$")
ax.axvspan(cfg.train_end, cfg.T_max, alpha=0.07, color="#2196f3",
           label=r"Extrapolation window $(20,40]$")
ax.axvline(cfg.t_rad_start, color="#666", ls=":", lw=1.0)
ax.axvline(cfg.t_rad_end,   color="#666", ls=":", lw=1.0)
ax.set_xlabel("Time (days)"); ax.set_ylabel("$U(t)$")
ax.set_title("Benchmark scenario"); ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_scenario.png")); plt.close()

# ── Fig 9: PINN spatial reconstruction ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
for col, st in enumerate([10, 20, 30]):
    idx = int(st / cfg.dt)
    u_true = u_fields[idx]
    state_p = pinn_engine.step(State(u=u_true, mass=mass0_total),
                                float(st), cfg.dt)
    vm = max(float(u_true.max()), 0.01)
    kw = dict(origin="lower", vmin=0, vmax=vm, cmap="inferno",
              interpolation="bilinear", extent=[0, cfg.L]*2)
    axes[0, col].imshow(u_true, **kw)
    axes[0, col].set_title(f"PDE  $t={st}$", fontsize=9)
    im2 = axes[1, col].imshow(state_p.u, **kw)
    axes[1, col].set_title(f"PINN  $t={st}$", fontsize=9)
    for r in range(2):
        axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
fig.subplots_adjust(right=0.87, hspace=0.15, wspace=0.12)
cbar_ax = fig.add_axes([0.89, 0.15, 0.018, 0.7])
fig.colorbar(im2, cax=cbar_ax).set_label(r"$u(\mathbf{x},t)$", fontsize=9)
fig.suptitle("PINN reconstruction vs.\ PDE", fontsize=10, y=1.02)
plt.savefig(os.path.join(FIGS, "fig_pinn_reconstruction.png"),
            bbox_inches="tight"); plt.close()

print(f"\n=  All figures saved to {FIGS}")
print(f"=  results.json saved")
