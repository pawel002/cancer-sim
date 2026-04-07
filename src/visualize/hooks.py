import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import imageio
from typing import Any
from ..core.state import State
from ..core.simulator import Simulator

class LinearPlotHook:
    """
    Hook to plot scalar simulation variables (e.g. mass, center_of_mass) over time.
    """
    def __init__(self, save_path: str, title: str = "Tumor Mass Over Time"):
        self.save_path = save_path
        self.title = title
        self.times = []
        self.masses = []

    def __call__(self, step_idx: int, t: float, current_state: State, sim: Simulator) -> bool:
        self.times.append(t)
        self.masses.append(current_state.mass)
        return False
        
    def render(self):
        """Generates the final plot and saves it."""
        from .style import apply_scientific_style
        apply_scientific_style()
        
        plt.figure()
        plt.plot(self.times, self.masses, label="Tumor Mass", color='#D9381E')
        plt.xlabel("Time (days)")
        plt.ylabel("Mass")
        plt.title(self.title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Ensure dir exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        plt.savefig(self.save_path)
        print(f"Plot saved to {self.save_path}")
        plt.close()


class GifHeatmapHook:
    """
    Hook to plot 2D density/matrix values and save them as an animated GIF.
    """
    def __init__(self, 
                 save_path: str, 
                 attr_name: str = 'u', 
                 title: str = "Tumor Density", 
                 fps: int = 10,
                 cmap: str = 'inferno',
                 vmax: float = 1.0):
        self.save_path = save_path
        self.attr_name = attr_name
        self.title = title
        self.fps = fps
        self.cmap = cmap
        self.vmax = vmax
        
        # Temporary directory for frames
        self.frames_dir = os.path.join(os.path.dirname(self.save_path), ".frames_tmp")
        os.makedirs(self.frames_dir, exist_ok=True)
        
        self.frame_paths = []

    def __call__(self, step_idx: int, t: float, current_state: State, sim: Simulator) -> bool:
        from .style import apply_scientific_style
        apply_scientific_style()
        
        data = getattr(current_state, self.attr_name)
        if data is None:
            return False
            
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(data, origin="lower", vmin=0, vmax=self.vmax, cmap=self.cmap)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{self.title} | t={t:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        frame_filename = os.path.join(self.frames_dir, f"frame_{step_idx:05d}.png")
        fig.savefig(frame_filename)
        self.frame_paths.append(frame_filename)
        
        # clear memory
        plt.close(fig)
        return False

    def render(self):
        """Compiles the collected frame images into a GIF."""
        if not self.frame_paths:
            print("No frames were captured for GIF.")
            return

        print(f"Compiling {len(self.frame_paths)} frames into {self.save_path}...")
        
        with imageio.get_writer(self.save_path, mode="I", fps=self.fps) as writer:
            for filename in self.frame_paths:
                image = imageio.imread(filename)
                writer.append_data(image)
        
        print(f"GIF compiled and saved to {self.save_path}")
        
        # Cleanup
        shutil.rmtree(self.frames_dir, ignore_errors=True)

class MultiCompartmentGifHook:
    """
    Hook specifically for models like SHO with multiple spatial compartments.
    Plots Normoxia, Hypoxia, and total Density side-by-side.
    """
    def __init__(self, save_path: str, fps: int = 10):
        self.save_path = save_path
        self.fps = fps
        self.frames_dir = os.path.join(os.path.dirname(self.save_path), ".frames_multi_tmp")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.frame_paths = []

    def __call__(self, step_idx: int, t: float, current_state: State, sim: Simulator) -> bool:
        from .style import apply_scientific_style
        apply_scientific_style()
        
        S = current_state.normoxia
        H = current_state.hypoxia
        u = current_state.u
        
        if S is None or H is None:
            return False
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Tumor Density
        im0 = axes[0].imshow(u, origin="lower", cmap='inferno', vmin=0, vmax=1.0)
        axes[0].set_title(f"Total Density | t={t:.2f}")
        fig.colorbar(im0, ax=axes[0])
        
        # Normoxia
        im1 = axes[1].imshow(S, origin="lower", cmap='viridis', vmin=0, vmax=1.0)
        axes[1].set_title("Normoxia (S)")
        fig.colorbar(im1, ax=axes[1])
        
        # Hypoxia
        im2 = axes[2].imshow(H, origin="lower", cmap='magma', vmin=0, vmax=1.0)
        axes[2].set_title("Hypoxia (H)")
        fig.colorbar(im2, ax=axes[2])
        
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.tight_layout()
        frame_filename = os.path.join(self.frames_dir, f"frame_{step_idx:05d}.png")
        fig.savefig(frame_filename)
        self.frame_paths.append(frame_filename)
        plt.close(fig)
        
        return False

    def render(self):
        if not self.frame_paths:
            return

        print(f"Compiling {len(self.frame_paths)} multi-compartment frames into {self.save_path}...")
        with imageio.get_writer(self.save_path, mode="I", fps=self.fps) as writer:
            for filename in self.frame_paths:
                image = imageio.imread(filename)
                writer.append_data(image)
        shutil.rmtree(self.frames_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Dual-panel hook: 2-D density heatmap  +  live mass-trajectory comparison
# ---------------------------------------------------------------------------

class DualPanelGifHook:
    """
    Hook that creates an animated GIF with two panels per frame:
      Left  – 2-D tumour-density heatmap (current PDE state)
      Right – mass-trajectory comparison (all pre-computed engines) drawn up to
              the current time, with the training/assimilation window shaded.

    Pass pre-computed reference trajectories in the constructor so the hook can
    draw all curves simultaneously without re-running the other engines.
    """

    def __init__(
        self,
        save_path: str,
        ref_trajectories: dict[str, tuple[Any, Any]],
        train_start: float,
        train_end: float,
        fps: int = 8,
        vmax: float = 1.0,
        cmap: str = "inferno",
        colors: dict[str, str] | None = None,
    ) -> None:
        """
        :param ref_trajectories: mapping  name → (times_array, masses_array)
                                 for every engine that should appear on the
                                 right panel.  The PDE ground truth should be
                                 included under the key ``"PDE"``.
        :param train_start: Left edge of the shaded assimilation window.
        :param train_end:   Right edge of the shaded assimilation window.
        :param colors:      Optional per-engine colour overrides.
        """
        self.save_path = save_path
        self.ref_trajectories = ref_trajectories
        self.train_start = train_start
        self.train_end = train_end
        self.fps = fps
        self.vmax = vmax
        self.cmap = cmap

        _default_colors: dict[str, str] = {
            "PDE":      "#2b2d42",
            "ODE":      "#8d99ae",
            "MLP":      "#ef233c",
            "NODE":     "#f77f00",
            "PINN":     "#4cc9f0",
            "SuperNet": "#7209b7",
        }
        self.colors: dict[str, str] = {**_default_colors, **(colors or {})}

        self.frames_dir = os.path.join(os.path.dirname(save_path), ".frames_dual_tmp")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.frame_paths: list[str] = []

    def __call__(
        self, step_idx: int, t: float, current_state: State, sim: Simulator
    ) -> bool:
        from .style import apply_scientific_style
        apply_scientific_style()

        u = current_state.u
        if u is None or np.ndim(u) < 2:
            return False

        fig, (ax_map, ax_mass) = plt.subplots(1, 2, figsize=(13, 5))

        # ── Left: density heatmap ─────────────────────────────────────────
        im = ax_map.imshow(
            u, origin="lower", vmin=0, vmax=self.vmax,
            cmap=self.cmap, interpolation="bilinear"
        )
        plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
        ax_map.set_title(f"Tumour density   t = {t:.1f}", fontsize=11)
        ax_map.set_xticks([])
        ax_map.set_yticks([])

        # ── Right: mass trajectory ────────────────────────────────────────
        # Shaded training window
        ax_mass.axvspan(
            self.train_start, self.train_end,
            alpha=0.12, color="#f4a261", label="Training window"
        )
        # Vertical line at current time
        ax_mass.axvline(t, color="#333333", lw=1.0, ls="--", alpha=0.6)

        for name, (ts, ms) in self.ref_trajectories.items():
            ts_arr = np.asarray(ts)
            ms_arr = np.asarray(ms)
            mask = ts_arr <= t + 1e-9
            lw = 2.5 if name == "PDE" else 1.8
            ls = "-" if name == "PDE" else "--"
            ax_mass.plot(
                ts_arr[mask], ms_arr[mask],
                color=self.colors.get(name, None),
                lw=lw, ls=ls, label=name, alpha=0.9
            )

        ax_mass.set_xlabel("Time")
        ax_mass.set_ylabel("Total tumour mass")
        ax_mass.set_title("Mass trajectory comparison")
        ax_mass.legend(fontsize=8, loc="upper right")
        ax_mass.set_xlim(left=0)
        ax_mass.grid(True, alpha=0.25)

        plt.tight_layout()
        frame_path = os.path.join(self.frames_dir, f"frame_{step_idx:05d}.png")
        fig.savefig(frame_path, dpi=100)
        self.frame_paths.append(frame_path)
        plt.close(fig)
        return False

    def render(self) -> None:
        if not self.frame_paths:
            print("DualPanelGifHook: no frames captured.")
            return
        print(f"Compiling {len(self.frame_paths)} dual-panel frames → {self.save_path}…")
        with imageio.get_writer(self.save_path, mode="I", fps=self.fps) as writer:
            for p in self.frame_paths:
                writer.append_data(imageio.imread(p))
        shutil.rmtree(self.frames_dir, ignore_errors=True)
        print(f"GIF saved → {self.save_path}")
