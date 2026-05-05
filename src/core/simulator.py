from typing import Any, Callable
from .state import State

class Simulator:
    """
    The main orchestrator loop for cancer simulations.
    It takes an initialized Engine as a strategy to step the simulation forward.
    """
    
    def __init__(self, engine: Any, dt: float, t_end: float):
        """
        :param engine: Instance of BaseEngine subclass
        :param dt: Time step size
        :param t_end: Total time
        """
        self.engine = engine
        self.dt = dt
        self.t_end = t_end
        self.history: list[State] = []

    def run(self, initial_state: State, hook_fn: Callable = None, m: int = 10) -> list[State]:
        """
        Runs the simulation loop.
        
        :param initial_state: The starting state
        :param hook_fn: A callable executed every m-th step
        :param m: Frequency of hook execution
        """
        current_state = initial_state
        self.history.append(current_state)
        
        steps = int(self.t_end / self.dt)
        for step_idx in range(1, steps + 1):
            t = step_idx * self.dt
            
            # The Engine calculates the next state
            current_state = self.engine.step(current_state, t, self.dt)
            self.history.append(current_state)
            
            # The requested m-th step hook
            if hook_fn and step_idx % m == 0:
                # E.g. to early stop, hook_fn can raise a custom exception 
                # or we can check its return value.
                stop = hook_fn(step_idx, t, current_state, self)
                if stop:
                    break
                
        return self.history
