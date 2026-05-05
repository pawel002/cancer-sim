import numpy as np
from dataclasses import dataclass
from typing import Any

@dataclass
class State:
    """
    A data class holding the current state of the simulation.
    """
    u: Any  # Tumor density (could be scalar U(t) or grid u(x,t))
    mass: float  # Total tumor mass M(t)
    
    # Optional components used by different engines
    oxygen: Any = None
    normoxia: Any = None
    hypoxia: Any = None
    center_of_mass: Any = None
    
    # Used for SuperNet residual learning baseline
    features: Any = None
