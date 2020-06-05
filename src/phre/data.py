"""
    data module
    ~~~~~~~~~~~
"""
from typing import Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from .utils import empty_array


@dataclass
class ImageData:
    image: np.ndarray = field(default_factory=empty_array)
    image_shape: Tuple[int, int] = field(init=False)
    image_size: int = field(init=False)
    image_vec: np.ndarray = field(init=False)

    def __post_init__(self):
        self.image_shape = self.image.shape
        self.image_size = self.image.size
        self.image_vec = self.image.ravel()

    def show_image(self):
        """Plot the image.
        """
        plt.imshow(self.image)


def measure_data(data: ImageData,
                 num_obs: int,
                 obs_std: float,
                 seed: Union[int, None] = None,
                 normalize_obs_mat: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Create phase retrieval measurements from ImageData.

    Args:
        data (ImageData): Image data object.
        num_obs (int): Number of the observations.
        obs_std (float): Measurement error standard deviation.
        seed (Union[int, None], optional): Random seed. Defaults to None.
        normalize_obs_mat (bool):
            If `True` normalize the observation mapping.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Return the measurement mapping and measurements.
    """
    np.random.seed(seed)

    obs_mat = np.random.randn(num_obs, data.image_size)
    if normalize_obs_mat:
        obs_mat = obs_mat/np.linalg.norm(obs_mat, axis=0)

    obs_err = np.random.randn(num_obs)*obs_std

    obs = obs_mat.dot(data.image_vec) + obs_err
    obs = np.abs(obs)

    return obs_mat, obs
