"""
    optimizer
    ~~~~~~~~~
"""
import numpy as np


class Optimizer:
    """Optimizer parent class.
    """

    def __init__(self,
                 obs_mat: np.ndarray,
                 obs: np.ndarray,
                 obj_type: str = 'soft'):
        """Construtor of Optimizer.

        Args:
            obs_mat (np.ndarray):
                Observation mapping, from original image to observation.
            obs (np.ndarray): Observations.
            obj_type (str, optional):
                Objective type, when `soft` using the 1 norm, when `hard` using
                indicator function.
        """
        self.obs_mat = obs_mat
        self.obs = obs
        self.obj_type = obj_type

        self.num_obs = self.obs_mat.shape[0]
        self.image_size = self.obs_mat.shape[1]

    def phase_retrieval(self, **fit_options):
        """Optimization based phase retrieval algorithm.

        Args:
            fit_options (dict): A dictionary contains for fit options.

        Raises:
            NotImplementedError:
                phase_retrieval need to be implemented by subclass.
        """
        raise NotImplementedError("phase_retrieval need to be implemented by"
                                  "subclasses.")

    def _soft_prox(self, v: np.ndarray, alpha: float) -> np.ndarray:
        """Prox for 1 norm objective.
        argmin_w |||w| - obs||_1 + 1/(2*alpha)||w - v||_2^2

        Args:
            v (np.ndarray): Reference point.
            alpha (float): Prox parameter.

        Returns:
            np.ndarray: Solution of the prox operator.
        """
        sign_v = np.sign(v)
        v *= sign_v

        w = self.obs.copy()
        r_index = v > self.obs + alpha
        l_index = v < self.obs - alpha

        w[r_index] = v[r_index] - alpha
        w[l_index] = v[l_index] + alpha

        w *= sign_v

        return w

    def _hard_prox(self, v: np.ndarray) -> np.ndarray:
        """Prox for indicator objective.
        argmin_w 1_[|w| == obs] + 1/2||w - v||_2^2

        Args:
            v (np.ndarray): Reference point.

        Returns:
            np.ndarray: Solution of the prox operator.
        """
        sign_v = np.sign(v)
        w = self.obs.copy()
        w *= sign_v
        return w

    def objective(self, x: np.ndarray) -> float:
        """Objective function value.

        Args:
            x (np.ndarray): Image variable.

        Returns:
            float: Objective function value.
        """
        return np.sum(np.abs(np.abs(self.obs_mat.dot(x)) - self.obs))
