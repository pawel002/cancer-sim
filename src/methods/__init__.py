"""
Method registry — every surrogate is importable from this package.
"""
from src.methods.pde               import run_pde
from src.methods.ode               import run_ode
from src.methods.moment_closure    import run_moment_closure
from src.methods.galerkin          import run_galerkin
from src.methods.neural_corrector  import run_neural_corrector, NeuralCorrector
from src.methods.pinn              import run_pinn, PINNResidualMLP

METHOD_REGISTRY: dict[str, object] = {
    "PDE":             run_pde,
    "ODE":             run_ode,
    "Moment":          run_moment_closure,
    "Galerkin":        run_galerkin,
    "NeuralCorrector": run_neural_corrector,
    "PINN":            run_pinn,
}

__all__ = [
    "run_pde", "run_ode", "run_moment_closure",
    "run_galerkin", "run_neural_corrector", "run_pinn",
    "NeuralCorrector", "PINNResidualMLP", "METHOD_REGISTRY",
]
