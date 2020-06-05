"""
    utils
    ~~~~~
"""
import numpy as np


def empty_array(shape=(0,)):
    """Create empty array.
    """
    size = np.prod(shape)
    if size != 0:
        raise ValueError(f"Size ({size}) of empty array must be zero.")
    return np.array([]).reshape(shape)
