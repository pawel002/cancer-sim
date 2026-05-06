"""
Experiment orchestrator: 4 IC × 4 beam distributions.

Key comparison
--------------
The 4×4 trajectory matrix focuses on the three most relevant methods:
  PDE            – finite-difference ground truth
  ODE            – spatially-averaged surrogate (Section 2.2)
  ODECorrector   – ODE + offline MLP residual correction (Section 3.3 variant)

Additional methods (Galerkin, NeuralCorrector based on Galerkin) are kept
available and included in the accuracy table and timing figures for reference,
but excluded from the main 4×4 plot for clarity.

Usage:
    uv run python latex/run_experiments.py
    uv run python latex/run_experiments.py --skip-train   # use saved checkpoints
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import seaborn as sns
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 250, "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
})

from src.methods                   import NeuralCorrector
from src.methods.pde               import run_pde
from src.methods.ode               import run_ode
from src.methods.moment_closure    import run_moment_closure
from src.methods.galerkin          import run_galerkin
from src.methods.neural_corrector  import run_neural_corrector, CHECKPOINT_PATH
from src.methods.ode_corrector     import run_ode_corrector, ODE_CHECKPOINT_PATH
from src.methods.shared            import (
    DIST_NAMES, DIST_LABELS, TIMES_FULL, T_MAX, T_RAD_S, T_RAD_E, L,
    make_ic, make_hypoxia, make_beam, make_grid,
    K_CAP, AMPL_IC, D_COEF, RHO, BETA, N_STEPS, DT,
)
from src.training.train_neural_corrector import train as train_gal_corrector
from src.training.train_ode_corrector    import train as train_ode_corrector

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# ─── visual style ────────────────────────────────────────────────────────
COLORS = {
    "PDE":             "#2b2d42",
    "ODE":             "#6c757d",
    "ODECorrector":    "#e07b39",   # orange — ODE+MLP (main new method)
    "Moment":          "#a8c5da",
    "Galerkin":        "#2196f3",
    "NeuralCorrector": "#2d6a4f",
}
LINESTYLES = {
    "PDE":             "-",
    "ODE":             "-.",
    "ODECorrector":    "--",
    "Moment":          ":",
    "Galerkin":        ":",
    "NeuralCorrector": (0, (4, 1, 1, 1)),
}
# Methods shown in the 4×4 trajectory matrix
PLOT_METHODS  = ["PDE", "ODE", "ODECorrector"]
PLOT_LABELS   = ["PDE (ground truth)", "ODE", "ODE + MLP corrector"]
# All methods available for accuracy / timing figures
ALL_METHODS   = ["PDE", "ODE", "ODECorrector", "Galerkin", "NeuralCorrector"]
ALL_LABELS    = ["PDE", "ODE", "ODE+MLP", "Galerkin", "Gal.+MLP"]
SURROGATES    = ["ODE", "ODECorrector", "Galerkin", "NeuralCorrector"]
SURR_LABELS   = ["ODE", "ODE+MLP", "Galerkin", "Gal.+MLP"]


# ═══════════════════════════════════════════════════════════════════════════
# Run all 16 × 5 simulations
# ═══════════════════════════════════════════════════════════════════════════
def run_all(
    N: int = 50,
    nc_gal_model: NeuralCorrector | None = None,
    nc_ode_model: NeuralCorrector | None = None,
) -> dict:
    """results[ic][beam][method] = U(t) array."""
    results = {ic: {bm: {} for bm in DIST_NAMES} for ic in DIST_NAMES}
    n_total = len(DIST_NAMES) ** 2 * len(ALL_METHODS)
    done    = 0

    for ic in DIST_NAMES:
        for bm in DIST_NAMES:
            print(f"  IC={ic:<10}  beam={bm:<10}", end="  ", flush=True)

            # PDE
            _, v = run_pde(ic, bm, N=N)
            results[ic][bm]["PDE"] = v
            done += 1

            # ODE
            _, v = run_ode(ic, bm, N=N)
            results[ic][bm]["ODE"] = v
            done += 1

            # ODE + MLP
            _, v = run_ode_corrector(ic, bm, N=N, model=nc_ode_model)
            results[ic][bm]["ODECorrector"] = v
            done += 1

            # Galerkin
            _, v = run_galerkin(ic, bm, N=N)
            results[ic][bm]["Galerkin"] = v
            done += 1

            # Galerkin + MLP
            _, v = run_neural_corrector(ic, bm, N=N, model=nc_gal_model)
            results[ic][bm]["NeuralCorrector"] = v
            done += 1

            print(f"[{done}/{n_total}]")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: 4×4 trajectory matrix  (PDE / ODE / ODE+MLP)
# ═══════════════════════════════════════════════════════════════════════════
def fig_4x4(results: dict, outpath: Path) -> None:
    """
    4 rows (IC distribution) × 4 columns (beam distribution).
    Each panel shows PDE, ODE, and ODE+MLP trajectories overlaid.
    """
    fig, axes = plt.subplots(
        4, 4, figsize=(15, 12),
        sharex=True, sharey="row",
        gridspec_kw={"hspace": 0.10, "wspace": 0.07},
    )
    fig.suptitle(
        r"$U(t) = y(t)/L^2$ — rows: tumour IC, columns: beam distribution"
        "\n(PDE · ODE · ODE+MLP corrector;  shaded = radiation window $[10,30]$)",
        fontsize=10.5, y=1.01,
    )

    for c, dlabel in enumerate(DIST_LABELS):
        axes[0, c].set_title(f"Beam: {dlabel}", fontsize=8.8, pad=4)
    for r, dlabel in enumerate(DIST_LABELS):
        axes[r, 0].set_ylabel(f"IC: {dlabel}\n$U(t)$", fontsize=7.8)

    for r, ic in enumerate(DIST_NAMES):
        for c, bm in enumerate(DIST_NAMES):
            ax = axes[r, c]
            ax.axvspan(T_RAD_S, T_RAD_E, alpha=0.09, color="#e07b39", zorder=0)

            for mname, mlabel in zip(PLOT_METHODS, PLOT_LABELS):
                v  = results[ic][bm][mname]
                lw = 2.0 if mname == "PDE" else 1.5
                ax.plot(TIMES_FULL[:len(v)], v,
                        color=COLORS[mname], lw=lw,
                        ls=LINESTYLES[mname], label=mlabel)

            ax.grid(True, alpha=0.18)
            ax.set_xlim(0, T_MAX)
            ax.tick_params(labelsize=7)
            if r == 3:
                ax.set_xlabel("Time", fontsize=7.5)

    axes[0, 3].legend(fontsize=8, loc="upper right", ncol=1)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Timing vs grid size N
# ═══════════════════════════════════════════════════════════════════════════
def fig_timing(
    outpath: Path,
    nc_ode_model: NeuralCorrector | None = None,
    nc_gal_model: NeuralCorrector | None = None,
) -> None:
    from src.methods.galerkin         import build_basis, build_radiation_matrix
    from src.methods.neural_corrector import extract_scenario_scalars, build_input_tensor
    import torch

    grid_sizes = [10, 15, 20, 30, 40, 50, 70, 100]
    ic, bm     = "middle", "middle"
    times_dict: dict[str, list[float]] = {m: [] for m in ALL_METHODS}

    print(f"\nTiming sweep (IC='{ic}', beam='{bm}'):")
    for N in grid_sizes:
        print(f"  N={N} ...", end=" ", flush=True)

        t0=time.perf_counter(); run_pde(ic,bm,N=N);            times_dict["PDE"].append(time.perf_counter()-t0)

        t0=time.perf_counter(); _, U_ode=run_ode(ic,bm,N=N);  times_dict["ODE"].append(time.perf_counter()-t0)

        # ODE corrector = ODE + MLP forward pass
        if nc_ode_model is not None:
            scalars = extract_scenario_scalars(ic, bm, N)
            X_np    = build_input_tensor(scalars, U_ode, TIMES_FULL)
            t0=time.perf_counter()
            with torch.no_grad(): nc_ode_model(torch.from_numpy(X_np))
            times_dict["ODECorrector"].append(times_dict["ODE"][-1] + (time.perf_counter()-t0))
        else:
            times_dict["ODECorrector"].append(times_dict["ODE"][-1] + 0.001)

        t0=time.perf_counter(); _, U_gal=run_galerkin(ic,bm,N=N); times_dict["Galerkin"].append(time.perf_counter()-t0)

        # Galerkin corrector = Galerkin + MLP forward pass
        if nc_gal_model is not None:
            scalars = extract_scenario_scalars(ic, bm, N)
            X_np    = build_input_tensor(scalars, U_gal, TIMES_FULL)
            t0=time.perf_counter()
            with torch.no_grad(): nc_gal_model(torch.from_numpy(X_np))
            times_dict["NeuralCorrector"].append(times_dict["Galerkin"][-1] + (time.perf_counter()-t0))
        else:
            times_dict["NeuralCorrector"].append(times_dict["Galerkin"][-1] + 0.001)

        print("  ".join(f"{m[:4]}={times_dict[m][-1]*1e3:.0f}ms" for m in ALL_METHODS))

    N_arr = np.array(grid_sizes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for mname, mlabel in zip(ALL_METHODS, ALL_LABELS):
        t_arr = np.array(times_dict[mname]) * 1e3
        ax1.loglog(N_arr, t_arr, marker="o", color=COLORS[mname], lw=1.6, label=mlabel)
        ax2.semilogy(N_arr, np.array(times_dict["PDE"])*1e3 / (t_arr + 0.01),
                     marker="o", color=COLORS[mname], lw=1.6, label=mlabel)

    ax1.set_xlabel("Grid size $N$ (per axis)"); ax1.set_ylabel("Wall-clock time (ms)")
    ax1.set_title("Computation time vs. grid size"); ax1.legend(fontsize=8); ax1.grid(True, which="both", alpha=0.25)
    ax2.axhline(1.0, color="#bbb", ls="--", lw=0.8)
    ax2.set_xlabel("Grid size $N$ (per axis)"); ax2.set_ylabel("Speedup vs. PDE")
    ax2.set_title("Speedup vs. PDE ground truth"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Accuracy table — 4×4 RMSE heatmaps for all surrogates
# ═══════════════════════════════════════════════════════════════════════════
def fig_accuracy_table(results: dict, outpath: Path) -> dict:
    n_d = len(DIST_NAMES)
    rmse_grids = {m: np.zeros((n_d, n_d)) for m in SURROGATES}
    metrics: dict = {}

    for i, ic in enumerate(DIST_NAMES):
        metrics[ic] = {}
        for j, bm in enumerate(DIST_NAMES):
            pde_m = results[ic][bm]["PDE"]
            metrics[ic][bm] = {}
            for mname in SURROGATES:
                pred  = results[ic][bm][mname]
                mn    = min(len(pde_m), len(pred))
                diff  = pred[:mn] - pde_m[:mn]
                rmse  = float(np.sqrt(np.mean(diff**2)))
                denom = float(np.sqrt(np.mean(pde_m[:mn]**2))) + 1e-12
                rmse_grids[mname][i, j] = rmse
                metrics[ic][bm][mname]  = {"rmse": rmse, "relative_error": rmse/denom}

    vmax = max(g.max() for g in rmse_grids.values()) * 1.05
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        "RMSE vs. PDE  (rows = IC distribution, cols = beam distribution)",
        fontsize=10.5,
    )
    for ax, mname, slabel in zip(axes, SURROGATES, SURR_LABELS):
        mat = rmse_grids[mname]
        im  = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(n_d)); ax.set_xticklabels(DIST_LABELS, fontsize=8.5)
        ax.set_yticks(range(n_d)); ax.set_yticklabels(DIST_LABELS, fontsize=8.5)
        ax.set_xlabel("Beam", fontsize=9); ax.set_ylabel("IC", fontsize=9)
        ax.set_title(slabel, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("RMSE", fontsize=8)
        for ii in range(n_d):
            for jj in range(n_d):
                v   = mat[ii, jj]
                clr = "white" if v > vmax * 0.62 else "black"
                ax.text(jj, ii, f"{v:.3f}", ha="center", va="center",
                        fontsize=7.5, color=clr)

    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath.name}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Publication-quality figure — 3 selected scenarios
# ═══════════════════════════════════════════════════════════════════════════
def fig_publication(results: dict, outpath: Path) -> None:
    from scipy.ndimage import laplace as nd_laplace

    CASES = [
        ("middle",    "middle",    "Aligned (Middle IC + Middle beam)"),
        ("left",      "right",     "Misaligned (Left IC + Right beam)"),
        ("two_peaks", "two_peaks", "Two-peak: ODE vs ODE+MLP"),
    ]
    N_pub = 80

    fig = plt.figure(figsize=(13, 9.5))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.38, wspace=0.30,
                            left=0.08, right=0.97, top=0.92, bottom=0.06)

    fig.text(0.24, 0.955, r"Initial density $u_0(\mathbf{x})$" + "\n(cyan = beam)",
             ha="center", fontsize=9.8, fontweight="bold")
    fig.text(0.52, 0.955, r"Final PDE density $u(\mathbf{x},\,T{=}40)$" + "\n(cyan = beam)",
             ha="center", fontsize=9.8, fontweight="bold")
    fig.text(0.81, 0.955, r"$U(t) = y(t)/L^2$ — all methods",
             ha="center", fontsize=9.8, fontweight="bold")

    last_im = None
    for row, (ic_n, bm_n, title) in enumerate(CASES):
        u0_p = make_ic(ic_n, N_pub)
        H_p  = make_hypoxia(N_pub)
        R0_p = make_beam(bm_n, N_pub)
        X_p, Y_p, dx_p = make_grid(N_pub)

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(u0_p, origin="lower", cmap="hot",
                   extent=[0, L, 0, L], vmin=0, vmax=AMPL_IC, interpolation="bilinear")
        ax0.contour(X_p, Y_p, R0_p, levels=[0.25, 0.60, 0.90],
                    colors="cyan", linewidths=0.8, alpha=0.9)
        ax0.set_title(title, fontsize=9, pad=3, loc="left", fontweight="bold")
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_ylabel(r"$x_2$", fontsize=8)

        # Final PDE field
        u_p = u0_p.copy()
        for i in range(N_STEPS):
            t_i = i * DT
            r_t = 1.0 if T_RAD_S <= t_i <= T_RAD_E else 0.0
            lap = nd_laplace(u_p, mode="constant", cval=0.0) / dx_p**2
            du  = D_COEF*lap + RHO*u_p*(1-u_p/K_CAP) - BETA*R0_p*H_p*u_p*r_t
            u_p = np.clip(u_p + du*DT, 0.0, None)
            u_p[0,:]=u_p[-1,:]=u_p[:,0]=u_p[:,-1]=0.0

        ax1 = fig.add_subplot(gs[row, 1])
        last_im = ax1.imshow(u_p, origin="lower", cmap="hot",
                              extent=[0, L, 0, L], vmin=0, vmax=AMPL_IC,
                              interpolation="bilinear")
        ax1.contour(X_p, Y_p, R0_p, levels=[0.25, 0.60, 0.90],
                    colors="cyan", linewidths=0.8, alpha=0.9)
        ax1.set_xticks([]); ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[row, 2])
        ax2.axvspan(T_RAD_S, T_RAD_E, alpha=0.09, color="#e07b39", zorder=0)
        for mname, mlabel in zip(ALL_METHODS, ALL_LABELS):
            v   = results[ic_n][bm_n][mname]
            lw  = 2.2 if mname == "PDE" else 1.5
            ax2.plot(TIMES_FULL[:len(v)], v,
                     color=COLORS[mname], lw=lw, ls=LINESTYLES[mname], label=mlabel)
        if row == 0:
            ax2.legend(fontsize=7.5, ncol=2, loc="upper right")
        ax2.set_xlim(0, T_MAX)
        ax2.grid(True, alpha=0.2)
        ax2.set_ylabel(r"$U(t)$", fontsize=8)
        ax2.tick_params(labelsize=8)
        if row == 2:
            ax2.set_xlabel("Time", fontsize=8.5)

    cbar_ax = fig.add_axes([0.015, 0.12, 0.012, 0.73])
    fig.colorbar(last_im, cax=cbar_ax).set_label(r"$u(\mathbf{x},t)$", fontsize=8)
    plt.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Load existing checkpoints instead of re-training")
    args = parser.parse_args()

    print("=" * 70)
    print("Cancer-surrogate experiments: 4 IC × 4 beam × PDE/ODE/ODE+MLP")
    print("=" * 70)

    import torch

    def _load_or_train(name: str, ckpt: Path, train_fn) -> NeuralCorrector:
        if args.skip_train and ckpt.exists():
            print(f"  Loading {name} from {ckpt.name} …")
            c = torch.load(ckpt, map_location="cpu", weights_only=False)
            m = NeuralCorrector(); m.load_state_dict(c["state_dict"]); m.eval()
            return m
        print(f"  Training {name} …")
        return train_fn(n_scenarios=200, n_epochs=200, N=50)

    print("\n[1/6] Neural corrector models …")
    nc_ode_model = _load_or_train("ODE Corrector",     ODE_CHECKPOINT_PATH, train_ode_corrector)
    nc_gal_model = _load_or_train("Galerkin Corrector", CHECKPOINT_PATH,    train_gal_corrector)

    print("\n[2/6] Running all 16 IC×beam combinations × 5 methods (N=50) …")
    t0      = time.perf_counter()
    results = run_all(N=50, nc_gal_model=nc_gal_model, nc_ode_model=nc_ode_model)
    print(f"  Total: {time.perf_counter()-t0:.1f}s")

    print("\n[3/6] Generating 4×4 trajectory matrix (PDE / ODE / ODE+MLP) …")
    fig_4x4(results, FIGDIR / "fig_4x4_trajectories.png")

    print("\n[4/6] Timing sweep across grid sizes …")
    fig_timing(FIGDIR / "fig_timing.png",
               nc_ode_model=nc_ode_model, nc_gal_model=nc_gal_model)

    print("\n[5/6] Generating accuracy table (all surrogates) …")
    metrics = fig_accuracy_table(results, FIGDIR / "fig_accuracy_table.png")

    print("\n[6/6] Generating publication figure …")
    fig_publication(results, FIGDIR / "fig_publication.png")

    out_json = Path(__file__).parent / "results_experiments.json"
    with open(out_json, "w") as f:
        json.dump({
            "trajectories": {
                ic: {bm: {m: v.tolist() for m, v in results[ic][bm].items()}
                     for bm in DIST_NAMES}
                for ic in DIST_NAMES
            },
            "accuracy": metrics,
        }, f, indent=2)
    print(f"\nResults → {out_json}")

    print("\n" + "=" * 72)
    print(f"{'Case':<12}  {'ODE':>8}  {'ODE+MLP':>10}  {'Galerkin':>10}  {'Gal+MLP':>10}")
    print("-" * 72)
    for d in DIST_NAMES:
        row = metrics[d][d]
        print(f"{d:<12}  "
              f"{row.get('ODE',{}).get('rmse', float('nan')):>8.4f}  "
              f"{row.get('ODECorrector',{}).get('rmse', float('nan')):>10.4f}  "
              f"{row.get('Galerkin',{}).get('rmse', float('nan')):>10.4f}  "
              f"{row.get('NeuralCorrector',{}).get('rmse', float('nan')):>10.4f}")
    print("=" * 72)
    print("\nAll done — figures in latex/figures/")
