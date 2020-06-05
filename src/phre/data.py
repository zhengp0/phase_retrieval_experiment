"""
    data module
    ~~~~~~~~~~~
"""
from typing import Tuple
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from .utils import empty_array


@dataclass
class ImageData:
    image: np.ndarray = field(default_factory=empty_array)
    image_shape: Tuple[int, int] = field(init=False)
    image_vec: np.ndarray = field(init=False)

    def __post_init__(self):
        self.image_shape = self.image.shape
        self.image_vec = self.image.ravel()

    def show_image(self):
        """Plot the image.
        """
        plt.imshow(self.image)
