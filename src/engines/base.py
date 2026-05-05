from abc import ABC, abstractmethod
from ..core.state import State

class BaseEngine(ABC):
    """
    Abstract base class for all simulation engines.
    """
    
    @abstractmethod
    def step(self, current_state: State, t: float, dt: float) -> State:
        """
        Advances the simulation state by one time step `dt`.
        
        :param current_state: The current State object
        :param t: The current time t
        :param dt: Time step size
        :return: A new State object representing time t + dt
        """
        pass
