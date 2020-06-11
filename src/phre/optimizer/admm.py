"""
    admm
    ~~~~
"""
from typing import Tuple, Union
import numpy as np
from phre.utils import distance
from .optimizer import Optimizer


class ADMM(Optimizer):
    """ADMM Optimizer.
    """

    def __init__(self,
                 obs_mat: np.ndarray,
                 obs: np.ndarray,
                 obj_type: str = 'soft',
                 rho: float = 1.0):
        """Constructor of ADMM solver.

        Args:
            obs_mat (np.ndarray):
                Observation mapping, from original image to observation.
            obs (np.ndarray): Observations.
            rho (float, optional):
                Hyper-parameter of ADMM algorithm, regularization for the
                Lagrangian. Defualt to 1.0.
        """
        super().__init__(obs_mat, obs, obj_type=obj_type)
        self.rho = rho
        self._cached_mat = self.obs_mat.T.dot(self.obs_mat)
        self.default_fit_options = dict(
            init_x=None,
            init_w=None,
            init_d=None,
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
            init_d (Union[np.ndarray, None], optional):
                Initialization of the dual variable. If `None`, start from zero
                vector. Defaults to None.
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
        x, w, d = self.initialize_vars(init_x=fit_options['init_x'],
                                       init_w=fit_options['init_w'],
                                       init_d=fit_options['init_d'])

        for iter_counter in range(1, fit_options['max_iter'] + 1):
            x_new = self.step_x(w, d)
            w_new = self.step_w(x_new, d)
            d_new = self.step_d(x_new, w_new, d)

            err = distance((x, w, d), (x_new, w_new, d_new),
                           rel=True, check_inputs=False)
            obj = self.objective(x_new)

            np.copyto(x, x_new)
            np.copyto(w, w_new)
            np.copyto(d, d_new)

            self.record_solver_info(obj=obj, err=err)

            if fit_options['verbose']:
                primal_feas = np.linalg.norm(self.obs_mat.dot(x) - w)
                print(f"iter {iter_counter:5}, obj {obj: .2e}, err {err:.2e}, "
                      f"primal_feas {primal_feas:.2e}")

            if err < fit_options['tol']:
                break

        return x

    def initialize_vars(self,
                        init_x: Union[np.ndarray, None] = None,
                        init_w: Union[np.ndarray, None] = None,
                        init_d: Union[np.ndarray, None] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Intialize variables.

        Args:
            init_x (Union[np.ndarray, None], optional):
                Initialization of the image of the optimizatoin problem.
                If `None`, start from zero vector. Defaults to None.
            init_w (Union[np.ndarray, None], optional):
                Initialization of the measurements of the optimization problem.
                If `None`, start from `self.obs`. Defualts to None.
            init_d (Union[np.ndarray, None], optional):
                Initialization of the dual variable. If `None`, start from zero
                vector. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Initial x, w and d variable.
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

        if init_d is None:
            init_d = np.zeros(self.num_obs)
        else:
            init_d = init_d.copy()

        if init_d.size != self.num_obs:
            raise ValueError(
                f"Invalid initial value for d, size ({init_d.size}) should"
                "be consistent with num_obs ({self.num_obs})"
            )

        return init_x, init_w, init_d

    def step_x(self, w: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Optimization x step.

        Args:
            w (np.ndarray): Pseudo observation vector.
            d (np.ndarray): Dual varaible.

        Returns:
            np.ndarray: Optimal image at the step.
        """
        return np.linalg.solve(self._cached_mat, self.obs_mat.T.dot(w - d))

    def step_w(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Optimization w step.

        Args:
            x (np.ndarray): Image vector.
            d (np.ndarray): Dual variable.

        Returns:
            np.ndarray: Optimal pseudo observation vector at the step.
        """
        v = self.obs_mat.dot(x) + d
        if self.obj_type == 'soft':
            w = self._soft_prox(v, 1.0/self.rho)
        else:
            w = self._hard_prox(v)

        return w

    def step_d(self, x: np.ndarray, w: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Optimiation d step.

        Args:
            x (np.ndarray): Image vector.
            w (np.ndarray): Pesudo image obersvation vector.
            d (np.ndarray): Dual variable.

        Returns:
            np.ndarray: Optimal dual variable at the step.
        """
        return d + self.obs_mat.dot(x) - w
