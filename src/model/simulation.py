from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """
    Represents the configuration of simulation.

    Attributes:

    """

    L: float = 10.0
    N: int = 1


if __name__ == "__main__":
    s = SimulationConfig()
