import numpy as np
import matplotlib.pyplot as plt
import time


def run_simulation():
    # ==========================================
    # 1. CONFIGURATION & PARAMETERS
    # ==========================================
    DIM = 2  # Set to 2 for 2D, 3 for 3D

    # Grid parameters
    L = 10.0  # Domain size (cm)
    N = 100  # Grid points per dimension
    dx = L / N
    dt = 0.01  # Time step (days) - Must satisfy CFL condition!
    T_max = 20.0  # Total simulation time (days)

    # Physical Parameters
    D_coeff = 0.05  # Diffusion coefficient (cm^2/day)
    rho = 0.1  # Proliferation rate (1/day)
    K = 1.0  # Carrying capacity
    beta = 0.5  # Radiosensitivity

    # Radiation Schedule
    # Apply radiation only at specific time windows (e.g., day 10 to 15)
    rad_start_time = 10.0
    rad_end_time = 15.0

    # ==========================================
    # 2. INITIALIZATION
    # ==========================================
    shape = tuple([N] * DIM)  # (N, N) or (N, N, N)
    u = np.zeros(shape)

    # Coordinate grids (for defining H(x) and initial u)
    axes = [np.linspace(0, L, N) for _ in range(DIM)]
    grids = np.meshgrid(*axes, indexing="ij")  # grids[0]=x, grids[1]=y, ...

    # Center of domain
    center = L / 2.0

    # Distance from center (r)
    r_squared = sum((g - center) ** 2 for g in grids)

    # Initial Tumor: Gaussian blob
    u = 0.1 * np.exp(-r_squared / 2.0)

    # Hypoxia Function H(x):
    # Assume hypoxia is high in the center (necrotic core logic)
    # H goes from 1 (max hypoxia) at center to 0 at edges.
    H = np.exp(-r_squared / 5.0)

    # ==========================================
    # 3. SOLVER FUNCTIONS
    # ==========================================

    def laplacian(field, dx):
        """Calculates discrete Laplacian using finite differences."""
        lap = np.zeros_like(field)
        inv_dx2 = 1 / (dx**2)

        # We roll the array to access neighbors (u_{i+1}, u_{i-1}, etc.)
        # This effectively uses periodic boundary conditions for simplicity,
        # but for a tumor in center of large domain, Dirichlet=0 is implied effectively.

        for axis in range(field.ndim):
            lap += (
                np.roll(field, +1, axis=axis)
                - 2 * field
                + np.roll(field, -1, axis=axis)
            ) * inv_dx2

        return lap

    def get_radiation_dose(t, spatial_grids):
        """
        R(x, t): Radiation dose.
        Returns 1.0 if within treatment window, 0.0 otherwise.
        Can be made spatially varying if needed.
        """
        if rad_start_time <= t <= rad_end_time:
            return 1.0  # Constant dose of 1 Gy (normalized)
        return 0.0

    # ==========================================
    # 4. MAIN LOOP
    # ==========================================
    print(f"Starting {DIM}D Simulation...")
    print(f"Grid: {shape}, Steps: {int(T_max / dt)}")

    start_time = time.time()

    t = 0.0
    steps = int(T_max / dt)

    for step in range(steps):
        # 1. Diffusion: D * Laplacian(u)
        diff_term = D_coeff * laplacian(u, dx)

        # 2. Proliferation: rho * u * (1 - u/K)
        prolif_term = rho * u * (1 - u / K)

        # 3. Radiation Kill: - beta * R(t) * H(x) * u
        R_val = get_radiation_dose(t, grids)
        kill_term = -beta * R_val * H * u

        # Update u
        du_dt = diff_term + prolif_term + kill_term
        u = u + du_dt * dt

        # Boundary handling (force zero at edges to simulate isolated dish)
        # Slicing syntax works for n-dims
        # In a real rigorous solver, we'd handle BCs inside Laplacian,
        # but this prevents "wrapping" artifacts from np.roll
        if DIM == 2:
            u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0
        elif DIM == 3:
            u[0, :, :] = u[-1, :, :] = u[:, 0, :] = u[:, -1, :] = u[:, :, 0] = u[
                :, :, -1
            ] = 0

        t += dt

    end_time = time.time()
    computation_cost = end_time - start_time

    print("=" * 40)
    print("Simulation Complete.")
    print(f"Computation Cost: {computation_cost:.4f} seconds")
    print("=" * 40)

    # ==========================================
    # 5. VISUALIZATION
    # ==========================================
    if DIM == 2:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.title("Final Tumor Density u(x)")
        im = plt.imshow(u, extent=[0, L, 0, L], origin="lower", cmap="inferno")
        plt.colorbar(im)

        plt.subplot(1, 2, 2)
        plt.title("Hypoxia Map H(x)")
        im2 = plt.imshow(H, extent=[0, L, 0, L], origin="lower", cmap="Blues")
        plt.colorbar(im2)

        plt.tight_layout()
        plt.show()

    elif DIM == 3:
        # For 3D, we visualize the middle slice
        mid_idx = N // 2

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.title(f"Tumor Density (Z-Slice {mid_idx})")
        im = plt.imshow(
            u[:, :, mid_idx], extent=[0, L, 0, L], origin="lower", cmap="inferno"
        )
        plt.colorbar(im)

        plt.subplot(1, 2, 2)
        plt.title(f"Hypoxia Map (Z-Slice {mid_idx})")
        im2 = plt.imshow(
            H[:, :, mid_idx], extent=[0, L, 0, L], origin="lower", cmap="Blues"
        )
        plt.colorbar(im2)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run_simulation()
