"""
Experiment script: 4 IC distributions × 4 beam distributions × 4 methods.

The 4×4 trajectory matrix has:
  rows    – 4 initial tumour density distributions  (Left, Right, Middle, Two Peaks)
  columns – 4 radiation beam distributions          (Left, Right, Middle, Two Peaks)
  content – all 4 method trajectories overlaid per cell  →  64 curves total

This cross-product design reveals whether the surrogates embed spatial
information: on the diagonal (tumour aligned with beam) radiation is strong;
off-diagonal cases create spatial mismatches that differentiate methods.

The hypoxia map H(x) is a FIXED tissue property (radially decreasing from
the domain centre), independent of the tumour's location.

Methods (Sections 2.2, 3.1, 3.2):
  PDE            – finite-difference ground truth
  ODE            – spatially-averaged macroscopic surrogate
  Moment Closure – tracks tumour centre-of-mass; evaluates R₀(z, t)
  Galerkin/POD   – Fourier-mode projection with precomputed R̃_{jk} matrix

All outputs: U(t) = y(t)/L²  (normalised spatial average).

Usage:  uv run python latex/run_experiments.py
"""
from __future__ import annotations

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
from scipy.integrate import solve_ivp
from scipy.ndimage import laplace as nd_laplace

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

import seaborn as sns
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "figure.dpi": 150,
    "savefig.dpi": 250, "savefig.bbox": "tight",
    "lines.linewidth": 1.6,
})

COLORS     = {"PDE":"#2b2d42", "ODE":"#6c757d", "Moment":"#e07b39", "Galerkin":"#2196f3"}
LINESTYLES = {"PDE":"-",       "ODE":"-.",       "Moment":"--",      "Galerkin":":"}
METHOD_NAMES  = ["PDE", "ODE", "Moment", "Galerkin"]
METHOD_LABELS = ["PDE", "ODE", "Moment Cl.", "Galerkin"]

# ═══════════════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════════════
L          = 10.0
T_MAX      = 40.0
DT         = 0.05
D_COEF     = 0.01
RHO        = 0.05
K_CAP      = 1.0
BETA       = 0.8    # chosen so aligned equilibrium U_eq ≈ 0.46 (non-zero, visible)
T_RAD_S    = 10.0
T_RAD_E    = 30.0
SIGMA_IC   = 1.5    # tumour IC Gaussian half-width
SIGMA_BEAM = 1.5    # radiation beam half-width (narrow → clear spatial contrast)
AMPL_IC    = 1.5      # peak density of initial condition  (< K_CAP = 1)
L_SQ       = L ** 2

# With H_eff ≈ 0.7 (centre hypoxia) and aligned beam (R=1):
#   U_eq ≈ RHO / (RHO + BETA*H*R) = 0.3 / (0.3 + 0.5*0.7) = 0.46
# → radiation creates a visible but non-zero equilibrium; tumour is not annihilated.
# For two-peak IC with two-peak beam, CoM lies between the beams:
#   R_CoM ≈ exp(-0.5) ≈ 0.61  (σ=1.5, distance L/4)
#   BETA*H*R_CoM = 0.5*0.7*0.61 = 0.21 < RHO=0.3 → Moment Closure predicts GROWTH
# while ODE (R_eff=1.0) and Galerkin correctly predict suppression.

N_STEPS    = int(T_MAX / DT)
TIMES_FULL = np.linspace(0.0, T_MAX, N_STEPS + 1)

# ═══════════════════════════════════════════════════════════════════════════
# Spatial distribution centres
# ═══════════════════════════════════════════════════════════════════════════
DIST_NAMES  = ["left", "right", "middle", "two_peaks"]
DIST_LABELS = ["Left", "Right", "Middle", "Two Peaks"]

PEAK_CENTERS: dict[str, list[tuple[float, float]]] = {
    "left":      [(L / 4,     L / 2)],
    "right":     [(3 * L / 4, L / 2)],
    "middle":    [(L / 2,     L / 2)],
    "two_peaks": [(L / 4,     L / 2), (3 * L / 4, L / 2)],
}


# ═══════════════════════════════════════════════════════════════════════════
# Field factories (IC, hypoxia, beam are fully independent)
# ═══════════════════════════════════════════════════════════════════════════
def make_grid(N: int) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.linspace(0.0, L, N)
    X, Y = np.meshgrid(x, x)
    return X, Y, L / N


def make_ic(ic_name: str, N: int) -> np.ndarray:
    """Gaussian tumour density IC centred on the chosen distribution."""
    X, Y, _ = make_grid(N)
    centers  = PEAK_CENTERS[ic_name]
    n        = len(centers)
    u0       = np.zeros((N, N))
    for cx, cy in centers:
        u0 += (AMPL_IC / n) * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*SIGMA_IC**2))
    u0 = np.clip(u0, 0.0, K_CAP)
    u0[0, :] = u0[-1, :] = u0[:, 0] = u0[:, -1] = 0.0
    return u0


def make_hypoxia(N: int) -> np.ndarray:
    """Fixed tissue hypoxia map H(x) — radially decreasing from domain centre."""
    X, Y, _ = make_grid(N)
    cx, cy  = L / 2, L / 2
    return np.clip(0.3 + 0.7 * np.exp(-((X-cx)**2 + (Y-cy)**2) / (L/2)**2), 0.3, 1.0)


def make_beam(beam_name: str, N: int) -> np.ndarray:
    """
    Gaussian radiation beam profile R₀(x).

    Each sub-beam has full amplitude 1.0 (independent sources at full power).
    For two-peak beam:  R₀ is high (≈1) at each of the two lobe centres
    and low (≈0.25 for σ=1.5, separation=L/2) midway between them.
    This creates the Moment-Closure failure: the tumour CoM sits between
    the two beams → R₀(z) ≈ 0.25 → BETA·H·R₀ < ρ → Moment predicts growth
    while ODE (R_eff≈1) and Galerkin both correctly predict suppression.
    """
    X, Y, _ = make_grid(N)
    R0       = np.zeros((N, N))
    for bx, by in PEAK_CENTERS[beam_name]:
        R0 += np.exp(-((X-bx)**2 + (Y-by)**2) / (2*SIGMA_BEAM**2))
    return np.clip(R0, 0.0, 1.0)   # clip sum of Gaussians to physical [0,1]


# ═══════════════════════════════════════════════════════════════════════════
# PDE ground truth
# ═══════════════════════════════════════════════════════════════════════════
def run_pde(
    ic_name: str, beam_name: str, N: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    u0 = make_ic(ic_name, N)
    H  = make_hypoxia(N)
    R0 = make_beam(beam_name, N)
    dx = L / N
    u  = u0.copy()
    out = [float(np.sum(u) * dx**2) / L_SQ]
    for i in range(N_STEPS):
        t   = i * DT
        r_t = 1.0 if T_RAD_S <= t <= T_RAD_E else 0.0
        lap = nd_laplace(u, mode="constant", cval=0.0) / dx**2
        du  = D_COEF * lap + RHO * u * (1.0 - u/K_CAP) - BETA * R0 * H * u * r_t
        u   = np.clip(u + du * DT, 0.0, None)
        u[0,:] = u[-1,:] = u[:,0] = u[:,-1] = 0.0
        out.append(float(np.sum(u) * dx**2) / L_SQ)
    return TIMES_FULL, np.array(out)


# ═══════════════════════════════════════════════════════════════════════════
# ODE surrogate  (Section 2.2)
# ═══════════════════════════════════════════════════════════════════════════
def run_ode(
    ic_name: str, beam_name: str, N: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    H_eff, R_eff = IC-weighted means of H, R₀ respectively.
    K_eff = ⟨u²⟩/⟨u⟩ corrects the leading spatial non-linearity error.
    """
    u0 = make_ic(ic_name, N)
    H  = make_hypoxia(N)
    R0 = make_beam(beam_name, N)
    dx = L / N
    w      = u0 / (np.sum(u0) + 1e-12)
    H_eff  = float(np.sum(H  * w))
    R_eff  = float(np.sum(R0 * w))
    U      = float(np.sum(u0) * dx**2) / L_SQ
    out    = [U]
    for i in range(N_STEPS):
        t   = i * DT
        r_t = R_eff if T_RAD_S <= t <= T_RAD_E else 0.0
        dU  = RHO * U * (1.0 - U / K_CAP) - BETA * H_eff * r_t * U
        U   = max(U + DT * dU, 0.0)
        out.append(U)
    return TIMES_FULL, np.array(out)


# ═══════════════════════════════════════════════════════════════════════════
# Moment Closure  (Section 3.1)
# ═══════════════════════════════════════════════════════════════════════════
def _bilinear(f: np.ndarray, xi: float, yi: float, dx: float) -> float:
    N  = f.shape[0]
    ix = float(np.clip(xi / dx, 0.0, N - 1 - 1e-9))
    iy = float(np.clip(yi / dx, 0.0, N - 1 - 1e-9))
    x0, y0 = int(ix), int(iy)
    x1, y1 = min(x0+1, N-1), min(y0+1, N-1)
    fx, fy = ix-x0, iy-y0
    return float(f[y0,x0]*(1-fx)*(1-fy) + f[y0,x1]*fx*(1-fy)
               + f[y1,x0]*(1-fx)*fy    + f[y1,x1]*fx*fy)


def run_moment_closure(
    ic_name: str, beam_name: str, N: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates R₀(z(t), t) where z is the tumour centre-of-mass.

    For single-peak IC aligned with the beam: z ≈ beam centre → R₀(z)≈1.
    For two-peak IC with a two-peak beam: z = midpoint between lobes ≈ (L/2, L/2),
    which lies midway between the two sub-beams → R₀(z) ≈ exp(-0.5) ≈ 0.25,
    severely underestimating the actual radiation each lobe receives (≈1.0).
    """
    u0 = make_ic(ic_name, N)
    H  = make_hypoxia(N)
    R0 = make_beam(beam_name, N)
    dx = L / N
    X, Y, _ = make_grid(N)
    U     = float(np.sum(u0) * dx**2) / L_SQ
    w_ic  = u0 * dx**2
    m_tot = np.sum(w_ic) + 1e-12
    z_x   = float(np.sum(X * w_ic) / m_tot)
    z_y   = float(np.sum(Y * w_ic) / m_tot)
    H_z   = _bilinear(H,  z_x, z_y, dx)
    R0_z  = _bilinear(R0, z_x, z_y, dx)
    out   = [U]
    for i in range(N_STEPS):
        t   = i * DT
        r_t = 1.0 if T_RAD_S <= t <= T_RAD_E else 0.0
        dU  = RHO * U * (1.0 - U / K_CAP) - BETA * H_z * R0_z * r_t * U
        U   = max(U + DT * dU, 0.0)
        out.append(U)
    return TIMES_FULL, np.array(out)


# ═══════════════════════════════════════════════════════════════════════════
# Fourier-Galerkin  (Section 3.2)
# ═══════════════════════════════════════════════════════════════════════════
def _build_basis(N: int, sqrt_K: int = 5) -> dict:
    X, Y, dx = make_grid(N)
    bs, lam = [], []
    for m in range(1, sqrt_K+1):
        for n in range(1, sqrt_K+1):
            psi = np.sin(m*np.pi*X/L) * np.sin(n*np.pi*Y/L)
            bs.append(psi)
            lam.append((m**2+n**2)*np.pi**2/L**2)
    bs  = np.array(bs)
    lam = np.array(lam)
    ns  = np.array([np.sum(b**2)*dx**2 for b in bs])
    Mk  = np.array([np.sum(b)*dx**2    for b in bs])
    return dict(bases=bs, lambdas=lam, norm_sq=ns, M_k=Mk, dx=dx)


def run_galerkin(
    ic_name: str, beam_name: str, N: int = 50, sqrt_K: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The precomputed coupling matrix R̃_{jk} = ∫ψ_j H R₀ ψ_k dA / ‖ψ_j‖² fully
    embeds both H(x) and R₀(x), so each (ic_name, beam_name) combination gets
    its own coupling matrix without additional PDE solves.
    """
    u0 = make_ic(ic_name, N)
    H  = make_hypoxia(N)
    R0 = make_beam(beam_name, N)
    fb  = _build_basis(N, sqrt_K=sqrt_K)
    bs, lam, ns, Mk, dx = fb["bases"], fb["lambdas"], fb["norm_sq"], fb["M_k"], fb["dx"]
    K   = len(bs)
    Rjk = np.zeros((K, K))
    for j in range(K):
        for k in range(K):
            Rjk[j, k] = np.sum(bs[j] * H * R0 * bs[k]) * dx**2
    Rt  = Rjk / ns[:, np.newaxis]
    a0  = np.array([np.sum(u0 * b) * dx**2 / ns[j] for j, b in enumerate(bs)])

    def rhs(t: float, a: np.ndarray) -> np.ndarray:
        U   = max(float(np.dot(Mk, a)) / L_SQ, 0.0)
        c_t = 1.0 if T_RAD_S <= t <= T_RAD_E else 0.0
        return -D_COEF*lam*a + RHO*(1.0 - U/K_CAP)*a - BETA*c_t*(Rt@a)

    sol = solve_ivp(rhs, [0.0, T_MAX], a0, method="RK45",
                    t_eval=TIMES_FULL, rtol=1e-5, atol=1e-8)
    return sol.t, np.maximum(sol.y.T @ Mk, 0.0) / L_SQ


# ═══════════════════════════════════════════════════════════════════════════
# Run all 16 ic × beam combinations for all 4 methods
# ═══════════════════════════════════════════════════════════════════════════
def run_all(N: int = 50) -> dict:
    """
    Returns results[ic_name][beam_name][method_name] = U(t) array.
    """
    results: dict = {ic: {bm: {} for bm in DIST_NAMES} for ic in DIST_NAMES}
    n_total = 16 * 4
    done    = 0

    for ic in DIST_NAMES:
        for bm in DIST_NAMES:
            print(f"  IC={ic:<10}  beam={bm:<10}", end="  ", flush=True)
            for method, fn in [("PDE",     run_pde),
                                ("ODE",     run_ode),
                                ("Moment",  run_moment_closure),
                                ("Galerkin",run_galerkin)]:
                t0 = time.perf_counter()
                _, v = fn(ic, bm, N=N)
                results[ic][bm][method] = v
                done += 1
            print(f"[{done}/{n_total}]")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: 4×4 trajectory matrix  (all 4 methods overlaid per cell)
# ═══════════════════════════════════════════════════════════════════════════
def fig_4x4(results: dict, outpath: Path) -> None:
    """
    Rows    = 4 IC distributions (tumour shape)
    Columns = 4 beam distributions (radiation pattern)
    Content = 4 method curves overlaid; PDE is the thick reference.
    """
    fig, axes = plt.subplots(
        4, 4, figsize=(15, 12),
        sharex=True,
        sharey="row",
        gridspec_kw={"hspace": 0.12, "wspace": 0.08},
    )
    fig.suptitle(
        r"$U(t) = y(t)/L^2$ — rows: tumour IC distribution, "
        r"columns: beam distribution, 4 methods per panel"
        "\n(shaded = radiation window $[10,30]$)",
        fontsize=11, y=1.01,
    )

    # Column headers (beam names)
    for c, dlabel in enumerate(DIST_LABELS):
        axes[0, c].set_title(f"Beam: {dlabel}", fontsize=9, pad=5)

    # Row labels (IC names)
    for r, dlabel in enumerate(DIST_LABELS):
        axes[r, 0].set_ylabel(f"IC: {dlabel}\n$U(t)$", fontsize=8)

    for r, ic in enumerate(DIST_NAMES):
        for c, bm in enumerate(DIST_NAMES):
            ax = axes[r, c]
            ax.axvspan(T_RAD_S, T_RAD_E, alpha=0.09, color="#e07b39", zorder=0)

            for mname, mlabel in zip(METHOD_NAMES, METHOD_LABELS):
                v  = results[ic][bm][mname]
                lw = 2.0 if mname == "PDE" else 1.2
                ax.plot(TIMES_FULL[:len(v)], v,
                        color=COLORS[mname], lw=lw,
                        ls=LINESTYLES[mname], label=mlabel)

            ax.grid(True, alpha=0.18)
            ax.set_xlim(0, T_MAX)
            ax.tick_params(labelsize=7)
            if r == 3:
                ax.set_xlabel("Time", fontsize=8)

    # Single legend on top-right panel
    axes[0, 3].legend(fontsize=7.5, loc="upper right", ncol=1)

    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Timing vs grid size N
# ═══════════════════════════════════════════════════════════════════════════
def fig_timing(outpath: Path) -> None:
    grid_sizes = [10, 15, 20, 30, 40, 50, 70, 100]
    ic, bm     = "middle", "middle"     # representative diagonal case
    pde_t, ode_t, mc_t, gal_pre_t, gal_ode_t = [], [], [], [], []

    print(f"\nTiming sweep (IC='{ic}', beam='{bm}'):")
    for N in grid_sizes:
        print(f"  N={N} ...", end=" ", flush=True)
        t0=time.perf_counter(); run_pde(ic,bm,N=N);            pde_t.append(time.perf_counter()-t0)
        t0=time.perf_counter(); run_ode(ic,bm,N=N);            ode_t.append(time.perf_counter()-t0)
        t0=time.perf_counter(); run_moment_closure(ic,bm,N=N); mc_t.append( time.perf_counter()-t0)

        u0 = make_ic(ic, N)
        H  = make_hypoxia(N)
        R0 = make_beam(bm, N)
        fb  = _build_basis(N, sqrt_K=5)
        bs, lam, ns, Mk, dx = fb["bases"], fb["lambdas"], fb["norm_sq"], fb["M_k"], fb["dx"]
        K   = len(bs)
        t0=time.perf_counter()
        Rjk = np.zeros((K,K))
        for j in range(K):
            for k in range(K):
                Rjk[j,k]=np.sum(bs[j]*H*R0*bs[k])*dx**2
        Rt=Rjk/ns[:,np.newaxis]
        gal_pre_t.append(time.perf_counter()-t0)

        a0=np.array([np.sum(u0*b)*dx**2/ns[j] for j,b in enumerate(bs)])
        def _rhs(t,a,_lam=lam,_Rt=Rt,_Mk=Mk):
            U=max(float(np.dot(_Mk,a))/L_SQ,0.0)
            c_t=1.0 if T_RAD_S<=t<=T_RAD_E else 0.0
            return -D_COEF*_lam*a+RHO*(1-U/K_CAP)*a-BETA*c_t*(_Rt@a)

        t0=time.perf_counter()
        solve_ivp(_rhs,[0.0,T_MAX],a0,method="RK45",t_eval=TIMES_FULL,rtol=1e-5,atol=1e-8)
        gal_ode_t.append(time.perf_counter()-t0)
        print(f"PDE={pde_t[-1]*1e3:.0f}ms  ODE={ode_t[-1]*1e3:.0f}ms  "
              f"MC={mc_t[-1]*1e3:.0f}ms  Gal={(gal_pre_t[-1]+gal_ode_t[-1])*1e3:.0f}ms")

    N_arr = np.array(grid_sizes)
    t_gal = np.array(gal_pre_t) + np.array(gal_ode_t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.loglog(N_arr, np.array(pde_t)*1e3,     "o-",  color=COLORS["PDE"],     label="PDE",                lw=2.0)
    ax1.loglog(N_arr, np.array(ode_t)*1e3,     "s-.", color=COLORS["ODE"],     label="ODE",                lw=1.6)
    ax1.loglog(N_arr, np.array(mc_t)*1e3,      "^--", color=COLORS["Moment"],  label="Moment Closure",     lw=1.6)
    ax1.loglog(N_arr, t_gal*1e3,               "D:",  color=COLORS["Galerkin"],label="Galerkin (total)",   lw=1.6)
    ax1.loglog(N_arr, np.array(gal_ode_t)*1e3, "d--", color=COLORS["Galerkin"],label="Galerkin (ODE only)",lw=1.2,alpha=0.5)
    ax1.set_xlabel("Grid size $N$ (per axis)")
    ax1.set_ylabel("Wall-clock time (ms, log scale)")
    ax1.set_title("Absolute computation time vs. grid size")
    ax1.legend(fontsize=8); ax1.grid(True, which="both", alpha=0.25)

    ax2.semilogy(N_arr, np.array(pde_t)/np.array(ode_t),"s-.", color=COLORS["ODE"],    label="ODE",           lw=1.6)
    ax2.semilogy(N_arr, np.array(pde_t)/np.array(mc_t), "^--", color=COLORS["Moment"], label="Moment Closure",lw=1.6)
    ax2.semilogy(N_arr, np.array(pde_t)/t_gal,          "D:",  color=COLORS["Galerkin"],label="Galerkin",     lw=1.6)
    ax2.axhline(1.0, color="#bbb", ls="--", lw=0.8)
    ax2.set_xlabel("Grid size $N$ (per axis)")
    ax2.set_ylabel("Speedup relative to PDE")
    ax2.set_title("Speedup vs. PDE ground truth")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Accuracy table — 4×4 heatmap per method pair
# ═══════════════════════════════════════════════════════════════════════════
def fig_accuracy_table(results: dict, outpath: Path) -> dict:
    """
    Three 4×4 heatmaps (one per non-PDE method) showing RMSE vs. PDE
    for every (IC, beam) combination.
    """
    methods  = ["ODE", "Moment", "Galerkin"]
    metrics: dict = {}
    rmse_grids = {m: np.zeros((4, 4)) for m in methods}

    for i, ic in enumerate(DIST_NAMES):
        metrics[ic] = {}
        for j, bm in enumerate(DIST_NAMES):
            pde_m = results[ic][bm]["PDE"]
            metrics[ic][bm] = {}
            for mname in methods:
                pred = results[ic][bm][mname]
                mn   = min(len(pde_m), len(pred))
                diff = pred[:mn] - pde_m[:mn]
                rmse = float(np.sqrt(np.mean(diff**2)))
                denom= float(np.sqrt(np.mean(pde_m[:mn]**2))) + 1e-12
                rmse_grids[mname][i, j] = rmse
                metrics[ic][bm][mname]  = {"rmse": rmse, "relative_error": rmse/denom}

    vmax = max(g.max() for g in rmse_grids.values()) * 1.05

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("RMSE vs. PDE ground truth  (rows = IC distribution, "
                 "cols = beam distribution)", fontsize=10.5)

    for ax, mname in zip(axes, methods):
        mat = rmse_grids[mname]
        im  = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(4)); ax.set_xticklabels(DIST_LABELS, fontsize=8.5)
        ax.set_yticks(range(4)); ax.set_yticklabels(DIST_LABELS, fontsize=8.5)
        ax.set_xlabel("Beam distribution", fontsize=9)
        ax.set_ylabel("IC distribution",   fontsize=9)
        ax.set_title(mname, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("RMSE", fontsize=8)
        for ii in range(4):
            for jj in range(4):
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
# Figure 4: Publication-quality figure
# Shows three contrasting cases: diagonal (aligned), off-diagonal (misaligned),
# and the two-peak pathological case for Moment Closure.
# ═══════════════════════════════════════════════════════════════════════════
def fig_publication(results: dict, outpath: Path) -> None:
    """
    3-column × 3-row layout highlighting the most informative scenarios.

    Selected cases:
      Row 0: IC=middle,    beam=middle    (aligned, all methods work)
      Row 1: IC=left,      beam=right     (fully misaligned)
      Row 2: IC=two_peaks, beam=two_peaks (Moment Closure fails at CoM)

    Each row: (initial density + beam contours | final PDE field | trajectories)
    """
    CASES = [
        ("middle",    "middle",    "Aligned (Middle IC, Middle beam)"),
        ("left",      "right",     "Misaligned (Left IC, Right beam)"),
        ("two_peaks", "two_peaks", "Two-peak: Moment Closure CoM error"),
    ]
    N_pub = 80

    fig = plt.figure(figsize=(13, 9.5))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.38, wspace=0.30,
                            left=0.08, right=0.97, top=0.92, bottom=0.06)

    fig.text(0.24, 0.955, r"Initial density $u_0(\mathbf{x})$"
             "\n(cyan = beam)",           ha="center", fontsize=9.8, fontweight="bold")
    fig.text(0.52, 0.955, r"Final PDE density $u(\mathbf{x},\,T{=}40)$"
             "\n(cyan = beam)",           ha="center", fontsize=9.8, fontweight="bold")
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
                   extent=[0,L,0,L], vmin=0, vmax=AMPL_IC, interpolation="bilinear")
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
                              extent=[0,L,0,L], vmin=0, vmax=AMPL_IC,
                              interpolation="bilinear")
        ax1.contour(X_p, Y_p, R0_p, levels=[0.25, 0.60, 0.90],
                    colors="cyan", linewidths=0.8, alpha=0.9)
        ax1.set_xticks([]); ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[row, 2])
        ax2.axvspan(T_RAD_S, T_RAD_E, alpha=0.09, color="#e07b39", zorder=0)
        for mname, mlabel in zip(METHOD_NAMES, METHOD_LABELS):
            v   = results[ic_n][bm_n][mname]
            lw  = 2.2 if mname == "PDE" else 1.5
            ax2.plot(TIMES_FULL[:len(v)], v,
                     color=COLORS[mname], lw=lw, ls=LINESTYLES[mname], label=mlabel)
        if row == 0:
            ax2.legend(fontsize=8, ncol=2, loc="upper right")
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
    print("=" * 70)
    print("Cancer-surrogate experiments: 4 IC × 4 beam × 4 methods = 64 curves")
    print("=" * 70)

    print("\n[1/5] Running all 16 IC×beam combinations × 4 methods (N=50) …")
    t_all0   = time.perf_counter()
    results  = run_all(N=50)
    t_all    = time.perf_counter() - t_all0
    print(f"  Total wall-clock: {t_all:.1f}s")

    print("\n[2/5] Generating 4×4 trajectory matrix …")
    fig_4x4(results, FIGDIR / "fig_4x4_trajectories.png")

    print("\n[3/5] Timing sweep across grid sizes …")
    fig_timing(FIGDIR / "fig_timing.png")

    print("\n[4/5] Generating accuracy table …")
    metrics = fig_accuracy_table(results, FIGDIR / "fig_accuracy_table.png")

    print("\n[5/5] Generating publication figure …")
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

    # Summary: diagonal (aligned) and off-diagonal worst cases
    print("\nDiagonal (aligned) RMSE — tumour IC = beam distribution:")
    print(f"  {'Case':<12}  {'ODE':>8}  {'Moment':>10}  {'Galerkin':>10}")
    for d in DIST_NAMES:
        row = metrics[d][d]
        print(f"  {d:<12}  "
              f"{row.get('ODE',{}).get('rmse',float('nan')):>8.4f}  "
              f"{row.get('Moment',{}).get('rmse',float('nan')):>10.4f}  "
              f"{row.get('Galerkin',{}).get('rmse',float('nan')):>10.4f}")
    print("\nAll done — figures in latex/figures/")
