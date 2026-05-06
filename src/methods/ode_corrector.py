"""
ODE Neural Residual Corrector.

Uses the ODE surrogate as the physics baseline and trains a compact MLP
to correct the residual:

    δ(t) = U_PDE(t) − U_ODE(t)

Same 10-dimensional feature vector and MLP architecture as the Galerkin
corrector (neural_corrector.py), with U_ode replacing U_gal as the first
feature.  This lets us test whether embedding spatial IC/beam information
through the feature vector alone — without Fourier-mode coupling — is
sufficient to match PDE accuracy at ODE speed.

Corrected prediction:
    U_NC_ODE(t) = max( U_ODE(t) + g_φ(s_t), 0 )
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.methods.neural_corrector import (
    NeuralCorrector,
    extract_scenario_scalars,
    build_input_tensor,
)
from src.methods.ode import run_ode

ODE_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "latex" / "figures" / "ode_corrector.pt"
)


def run_ode_corrector(
    ic_name: str,
    beam_name: str,
    N: int = 50,
    *,
    model: NeuralCorrector | None = None,
    checkpoint: Path = ODE_CHECKPOINT_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ODE + Neural Corrector inference.

    1. Run ODE surrogate → U_ode(t).
    2. Extract spatial scenario scalars (IC/beam CoM, misalignment d, H_eff, R_eff).
    3. Build 10-dim feature matrix: [U_ode, t/T, r(t), spatial scalars].
    4. Predict residual δ(t) = g_φ(features).
    5. Return (times, clip(U_ode + δ, 0, ∞)).
    """
    if model is None:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"ODE corrector checkpoint not found at {checkpoint}.\n"
                "Run: uv run python src/training/train_ode_corrector.py"
            )
        ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = NeuralCorrector()
        model.load_state_dict(ckpt["state_dict"])
    model.eval()

    times, U_ode = run_ode(ic_name, beam_name, N=N)

    scalars = extract_scenario_scalars(ic_name, beam_name, N)
    X_np    = build_input_tensor(scalars, U_ode, times)

    with torch.no_grad():
        delta = model(torch.from_numpy(X_np)).numpy()

    return times, np.maximum(U_ode + delta, 0.0)
