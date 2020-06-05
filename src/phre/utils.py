"""
    utils
    ~~~~~
"""
from typing import Tuple, Union
import numpy as np


def empty_array(shape=(0,)):
    """Create empty array.
    """
    size = np.prod(shape)
    if size != 0:
        raise ValueError(f"Size ({size}) of empty array must be zero.")
    return np.array([]).reshape(shape)


def distance(old: Union[np.ndarray, Tuple[np.ndarray]],
             new: Union[np.ndarray, Tuple[np.ndarray]],
             rel: bool = False,
             check_inputs: bool = True) -> float:
    """Compute the distance between old and new

    Args:
        old (Union[np.ndarray, Tuple[np.ndarray]]): Old vectors or tuple of old vectors.
        new (Union[np.ndarray, Tuple[np.ndarray]]): New vectors or tuple of new vectors.
        rel (bool, optional): If `True` compute the relative distance. Defaults to False.
        check_input(bool, optional):
            If `True`, go through a list of assertion to check the inputs.

    Returns:
        float: Distance.
    """
    old = old if isinstance(old, tuple) else tuple(old)
    new = new if isinstance(new, tuple) else tuple(new)

    if check_inputs:
        assert len(old) == len(new)
        assert all([isinstance(vec, np.ndarray) for vec in old])
        assert all([isinstance(vec, np.ndarray) for vec in new])
        assert all([len(vec_old) == len(vec_new)
                    for vec_old, vec_new in zip(old, new)])

    dists = np.array([
        np.linalg.norm(vec_old - vec_new)
        for vec_old, vec_new in zip(old, new)
    ])
    if rel:
        dists /= np.array([
            np.linalg.norm(vec_new)
            for vec_new in new
        ])

    return np.sum(dists)
