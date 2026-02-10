import os
import shutil

import imageio
import matplotlib.pyplot as plt
from sim_engine import SimStatePDE


def log_progress(state: SimStatePDE):
    mass = state.total_mass
    print(f"[Step {state.step_iteration}] t={state.t:.2f} | Tumor Mass: {mass:.4f}")


class GifPlotter:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.frames_dir = os.path.join(self.save_path, "frames")

        # if directory exists, clear it
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)

        os.makedirs(self.frames_dir)

        self.fig, self.ax = plt.subplots()
        self.im = None
        self.frame_paths: list[str] = []
        self.frame_count = 0

    def update(self, state: SimStatePDE):
        if self.im is None:
            self.im = self.ax.imshow(
                state.u, origin="lower", vmin=0, vmax=1, cmap="inferno"
            )
            plt.colorbar(self.im)
            self.title = self.ax.set_title(f"t={state.t:.2f}")
        else:
            self.im.set_data(state.u)
            self.title.set_text(f"t={state.t:.2f}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        filename = os.path.join(self.frames_dir, f"frame_{self.frame_count:05d}.png")

        self.fig.savefig(filename)
        self.frame_paths.append(filename)
        self.frame_count += 1

    def save_gif(self, fps: int = 10):
        gif_path = os.path.join(self.save_path, "simulation.gif")
        print(f"Compiling {len(self.frame_paths)} frames into {gif_path}...")

        with imageio.get_writer(gif_path, mode="I", fps=fps) as writer:
            for filename in self.frame_paths:
                image = imageio.imread(filename)
                writer.append_data(image)

        print("GIF saved successfully.")
