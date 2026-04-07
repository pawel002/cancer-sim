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
