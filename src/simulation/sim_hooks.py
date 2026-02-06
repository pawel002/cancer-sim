import matplotlib.pyplot as plt
from sim_engine import SimState


def log_progress(state: SimState):
    mass = state.total_mass
    print(f"[Step {state.step_iteration}] t={state.t:.2f} | Tumor Mass: {mass:.4f}")


class LivePlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.im = None
        plt.ion()

    def update(self, state: SimState):
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
