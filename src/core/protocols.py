"""
A collection of common radiation schedules and spatial protocols.
"""
import numpy as np

def standard_radiation_schedule(t: float, start: float, end: float, dose: float = 1.0) -> float:
    """
    Standard block radiation protocol r(t).
    
    :param t: Current time
    :param start: Start time of radiation therapy
    :param end: End time of radiation therapy
    :param dose: Delivery intensity/dose
    """
    return dose if start <= t <= end else 0.0

def uniform_spatial_radiation(x_shape: tuple, t: float, start: float, end: float, dose: float = 1.0) -> np.ndarray:
    """
    Uniform field spatial radiation R(x,t).
    
    :param x_shape: Shape of the spatial domain
    :param t: Current time
    :param start: Start time of radiation therapy
    :param end: End time of radiation therapy
    :param dose: Delivery intensity/dose
    """
    rad_t = standard_radiation_schedule(t, start, end, dose)
    return np.full(x_shape, rad_t)
