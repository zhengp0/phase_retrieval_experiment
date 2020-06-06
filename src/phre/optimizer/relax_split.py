"""
    relax_split
    ~~~~~~~~~~~
"""
from typing import Tuple, Union
import numpy as np
from phre.utils import distance


class RelaxSplit:
    """Relax and split optimizer.
    """

    def __init__(self,
                 obs_mat: np.ndarray,
                 obs: np.ndarray,
                 nu: float = 1.0):
        """Constructor of relax and split solver.

        Args:
            obs_mat (np.ndarray):
                Observation mapping, from original image to observation.
            nu (float, optional):
                Hyper-parameter of relax and split solver, controls how close
                to the original (before relax) formulation. The smaller the `nu`
                the closer to the original problem. Defaults to 1.0.
        """
        self.obs_mat = obs_mat
        self.obs = obs
        self.nu = nu

        self.num_obs = self.obs_mat.shape[0]
        self.image_size = self.obs_mat.shape[1]

    def phase_retrieval(self,
                        init_x: Union[np.ndarray, None] = None,
                        init_w: Union[np.ndarray, None] = None,
                        tol: float = 1e-6,
                        max_iter: int = 100,
                        verbose: bool = False) -> np.ndarray:
        """Phase retrieval algorithm.

        Args:
            init_x (Union[np.ndarray, None], optional):
                Initialization of the image of the optimizatoin problem.
                If `None`, start from zero vector. Defaults to None.
            init_w (Union[np.ndarray, None], optional):
                Initialization of the measurements of the optimization problem.
                If `None`, start from `self.obs`. Defualts to None.
            tol (float, optional):
                Tolerance for convergence of the optimization problem.
            max_iter (int, optional):
                Maximum number of iterations.
            verbose (bool, optional):
                If `True` print out the optimization convergence information.
                Defaults to False.

        Returns:
            np.ndarray: Final result.
        """
        x, w = self.initialize_vars(init_x=init_x, init_w=init_w)

        for iter_counter in range(1, max_iter + 1):
            x_new = self.step_x(w)
            w_new = self.step_w(x_new)

            err = distance((x, w), (x_new, w_new),
                           rel=True, check_inputs=False)

            np.copyto(x, x_new)
            np.copyto(w, w_new)

            if verbose:
                obj = self.objective(x, w)
                print(f"iter {iter_counter:5}, obj {obj:.2e}, err {err:.2e}")

            if err < tol:
                break

        return x

    def objective(self, x: np.ndarray, w: np.ndarray) -> float:
        """Objective function value.

        Args:
            x (np.ndarray): Image variable.
            w (np.ndarray): Pseudo observation variable.

        Returns:
            float: Objective function value.
        """
        v = self.obs_mat.dot(x)
        val = np.sum(np.abs(np.abs(w) - self.obs))
        val += 0.5*np.sum((v - w)**2)/self.nu
        return val

    def initialize_vars(self,
                        init_x: Union[np.ndarray, None] = None,
                        init_w: Union[np.ndarray, None] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Intialize variables.

        Args:
            init_x (Union[np.ndarray, None], optional):
                Initialization of the image of the optimizatoin problem.
                If `None`, start from zero vector. Defaults to None.
            init_w (Union[np.ndarray, None], optional):
                Initialization of the measurements of the optimization problem.
                If `None`, start from `self.obs`. Defualts to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Initial x and w variable.
        """
        # initialization
        if init_x is None:
            init_x = np.zeros(self.image_size)
        else:
            init_x = init_x.copy()

        if init_x.size != self.image_size:
            raise ValueError(
                f"invalid initial value for x, size ({init_x.size}) should"
                "be consistent with image_size({self.image_size})"
            )

        if init_w is None:
            init_w = self.obs.copy()
        else:
            init_w = init_w.copy()

        if init_w.size != self.num_obs:
            raise ValueError(
                f"Invalid initial value for w, size ({init_w.size}) should"
                "be consistent with num_obs ({self.num_obs})"
            )

        return init_x, init_w

    def step_x(self, w: np.ndarray) -> np.ndarray:
        """Optimization x step.

        Args:
            w (np.ndarray): Pseudo observation vector.

        Returns:
            np.ndarray: Optimal image vector based on `w`.
        """
        return np.linalg.solve(self.obs_mat.T.dot(self.obs_mat),
                               self.obs_mat.T.dot(w))

    def step_w(self, x: np.ndarray) -> np.ndarray:
        """Optimization w step.

        Args:
            x (np.ndarray): Image vector.

        Returns:
            np.ndarray: Optimal pseudo observation vector.
        """
        v = self.obs_mat.dot(x)
        sign_v = np.sign(v)
        v *= sign_v

        w = self.obs.copy()
        r_index = v > self.obs + self.nu
        l_index = v < self.obs - self.nu

        w[r_index] = v[r_index] - self.nu
        w[l_index] = v[l_index] + self.nu

        w *= sign_v

        return w
