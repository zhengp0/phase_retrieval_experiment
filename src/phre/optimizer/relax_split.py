"""
    relax_split
    ~~~~~~~~~~~
"""
from typing import Tuple, Union
import numpy as np
from phre.utils import distance
from .optimizer import Optimizer


class RelaxSplit(Optimizer):
    """Relax and split optimizer.
    """

    def __init__(self,
                 obs_mat: np.ndarray,
                 obs: np.ndarray,
                 obj_type: str = 'soft',
                 nu: float = 1.0):
        """Constructor of relax and split solver.

        Args:
            nu (float, optional):
                Hyper-parameter of relax and split solver, controls how close
                to the original (before relax) formulation. The smaller the `nu`
                the closer to the original problem. Defaults to 1.0.
        """
        super().__init__(obs_mat, obs, obj_type=obj_type)
        self.nu = nu
        self._cached_mat = self.obs_mat.T.dot(self.obs_mat)
        self.default_fit_options = dict(
            init_x=None,
            init_w=None,
            tol=1e-6,
            max_iter=100,
            verbose=False
        )

    def phase_retrieval(self, **fit_options) -> np.ndarray:
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
        self.reset_solver_info()
        fit_options = {**self.default_fit_options, **fit_options}
        x, w = self.initialize_vars(init_x=fit_options['init_x'],
                                    init_w=fit_options['init_w'])

        for iter_counter in range(1, fit_options['max_iter'] + 1):
            x_new = self.step_x(w)
            w_new = self.step_w(x_new)

            err = distance((x, w), (x_new, w_new),
                           rel=True, check_inputs=False)
            obj = self.objective(x_new)

            np.copyto(x, x_new)
            np.copyto(w, w_new)

            self.record_solver_info(obj=obj, err=err)

            if fit_options['verbose']:
                print(f"iter {iter_counter:5}, obj {obj: .2e}, err {err:.2e}")

            if err < fit_options['tol']:
                break

        return x

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
        return np.linalg.solve(self._cached_mat, self.obs_mat.T.dot(w))

    def step_w(self, x: np.ndarray) -> np.ndarray:
        """Optimization w step.

        Args:
            x (np.ndarray): Image vector.

        Returns:
            np.ndarray: Optimal pseudo observation vector.
        """
        v = self.obs_mat.dot(x)
        if self.obj_type == 'soft':
            w = self._soft_prox(v, self.nu)
        else:
            w = self._hard_prox(v)
        return w
