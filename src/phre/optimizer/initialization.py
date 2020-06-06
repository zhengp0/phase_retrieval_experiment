"""
    initialization
    ~~~~~~~~~~~~~~
"""
import numpy as np


def find_initial_image(obs_mat: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Intialization scheme to find good initial image.

    Args:
        obs_mat (np.ndarray): Observation mapping.
        obs (np.ndarray): Observations

    Returns:
        np.ndarray: Initial image.
    """
    obs_avg = np.mean(obs)
    selected_index = obs < 0.5*obs_avg

    selected_mat = (obs_mat.T*selected_index.astype(float)).dot(obs_mat)
    eig_vals, eig_vecs = np.linalg.eig(selected_mat)

    initial_image = eig_vecs[:, np.argmin(eig_vals)]*np.sqrt(obs_avg)

    return initial_image
