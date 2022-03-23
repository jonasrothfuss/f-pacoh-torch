import numpy as np
import math
import warnings
import scipy.optimize
import copy
import inspect

from typing import Callable, Tuple, Optional, List
from collections import OrderedDict


class Solver:

    def __init__(self, domain, initial_x=None):
        self._domain = domain
        self.initial_x = initial_x

    def minimize(self, f) -> Tuple[np.ndarray, float]:
        """
            optimize f over domain
            if self.requires_gradients = True, fun should return a tuple of (y,grad)
         """
        raise NotImplementedError

    @property
    def requires_gradients(self) -> bool:
        return False

    @property
    def requires_safety(self) -> bool:
        raise NotImplementedError


class CandidateSolver(Solver):

    def __init__(self, domain, candidates):
        """
        Deviates from parent argument interface.
        Args:
            domain:
            candidates:
        """
        super(CandidateSolver, self).__init__(domain)
        self.candidates = candidates

    def minimize(self, f: Callable) -> Tuple[np.ndarray, float]:
        # res = []
        # for c in self.candidates:
        #     res.append(f(c))
        res = f(self.candidates)
        index = np.argmin(res)
        best_x = self.candidates[index]
        return best_x, res[index]

    @property
    def requires_gradients(self) -> bool:
        return False

    @property
    def requires_safety(self) -> bool:
        return False


class FiniteDomainSolver(CandidateSolver):
    def __init__(self, domain):
        super().__init__(domain, domain.points)


class GridSolver(CandidateSolver):

    def __init__(self, domain, points_per_dimension=None):
        arrays = [np.linspace(l, u, points_per_dimension).reshape(points_per_dimension, 1) for (l, u) in
                  zip(domain.l, domain.u)]
        grid = cartesian(arrays)
        super(GridSolver, self).__init__(domain, candidates=grid)


class EvolutionarySolver(Solver):

    def __init__(self, domain, survival_rate=0.95, num_particles_per_d2=100, max_iter_per_d=30, initial_x=None,
                 num_iter_success=100, atol: float = 1e-8, random_state=None):
        super().__init__(domain, initial_x=initial_x)

        # store init arguments
        _local_vars = locals()
        self._init_args = OrderedDict([(k, _local_vars[k]) for k in inspect.signature(self.__init__).parameters.keys()])

        self.d = domain.d
        self.domain = domain
        self._rds = np.random if random_state is None else random_state

        self.num_particles = self.d**2 * num_particles_per_d2
        self.max_iter = self.d * max_iter_per_d
        self.survival_rate = survival_rate if survival_rate is None else survival_rate
        self._num_iter_success = num_iter_success
        self._atol = atol

        self.max_sampling_radius = (self.domain.u - self.domain.l) * self.d / self.num_particles

    def minimize(self, f) -> Tuple[np.ndarray, float]:
        best_f = 1e8 * np.ones(self.num_particles)
        particles = self._rds.uniform(self.domain.l, self.domain.u, size=(self.num_particles, self.domain.d))
        _best_f_last = 1e8
        _counter_no_improvement = 0
        for i in range(self.max_iter):
            temp = 1 - i / self.max_iter
            if i > 0:
                pertubation = temp * self.max_sampling_radius * self._uniform_sample_unit_ball()
                particle_proposal = np.clip(particles + pertubation, self.domain.l, self.domain.u)
            else:
                particle_proposal = particles

            # make sure that the proposal is in the domain
            particle_proposal = np.clip(particle_proposal, self.domain.l, self.domain.u)

            y = f(particle_proposal).reshape((self.num_particles))
            improvement_cond = y < best_f
            best_f = np.where(improvement_cond, y, best_f)
            particles = np.where(improvement_cond[:, None], particle_proposal, particles)

            # sort by fitness, eliminate worst particles and duplicate best particles
            sorted_idx = np.argsort(best_f)
            num_elimins = math.ceil(self.num_particles * (1-self.survival_rate))
            new_idx = np.concatenate([sorted_idx[:-num_elimins], sorted_idx[:num_elimins]])
            best_f = best_f[new_idx]
            particles = particles[new_idx]
            _best_f_all = np.min(best_f)
            if _best_f_last - self._atol <= _best_f_all:
                _counter_no_improvement += 1
            else:
                _counter_no_improvement = 0
            if _counter_no_improvement >= self._num_iter_success:
                break  # stop the optimization if there was no improvement for 5 consecutive iterations
            _best_f_last = min(_best_f_last, _best_f_all)
            if i >= (self.max_iter - 1):
                warnings.warn(f'EvolutionarySolver has reached the maximum number of {self.max_iter} iterations.')

        best_idx = np.argmin(best_f)
        return particles[best_idx], best_f[best_idx]

    def _uniform_sample_unit_ball(self) -> np.ndarray:
        # sample from unit sphere
        x = self._rds.normal(size=(self.num_particles, self.d))
        x = x / np.linalg.norm(x, axis=-1)[:, None]

        # transform in unit ball
        u = self._rds.uniform(0, 1, size=(self.num_particles, ))
        return x * (u**(1/self.d))[:, None]

    @property
    def requires_gradients(self) -> bool:
        return False

    @property
    def requires_safety(self) -> bool:
        return False

    def get_instance_with_double_effort(self) -> Solver:
        new_args = copy.copy(self._init_args)
        new_args['random_state'] = np.random.RandomState(self._rds.randint(0, 10 ** 8))
        new_args['num_particles_per_d2'] *= 2
        new_solver_instance = self.__class__(**new_args)
        assert 2 * self.num_particles == new_solver_instance.num_particles
        return new_solver_instance


class DualAnnealingSolver(Solver):
    """
    Simulated Annealing Solver
    """

    def __init__(self, domain, max_iter: int = 5000, random_state=None):
        super().__init__(domain)
        self.max_iter = max_iter
        self._rds = np.random if random_state is None else random_state

    def minimize(self, f):
        def f_single_x(x):
            return float(f(x.reshape(1, -1)))
        result = scipy.optimize.dual_annealing(f_single_x,
                                               bounds=np.stack([self._domain.l, self._domain.u], axis=1),
                                               maxiter=self.max_iter,
                                               seed=self._rds)
        return result.x, result.fun

    @property
    def requires_gradients(self):
        return False

    @property
    def requires_safety(self):
        return False


class DoubleSolverWrapper:
    """
    Wraps a solver and minimizes the provided function with it twice. Only if the difference of both
    independent solutions is smaller than the tolerance, it returns the minimum, otherwise it repeats the
    minimizations, potentially with increased effort.
    """

    def __init__(self, solver: Solver,
                 atol: float = 1e-3,
                 max_repeats: int = 5,
                 double_effort_at_rerun: bool = True,
                 global_effort_doubling_threshold: float = 0.5,
                 throw_precision_error: bool = False):
        self._solver = solver
        self._atol = atol
        self._max_repeats = max_repeats
        self._double_effort_at_rerun = double_effort_at_rerun
        self._solver_runs_counter = 0
        self._precision_failure_counter = 0
        self.global_effort_doubling_threshold = global_effort_doubling_threshold
        self._throw_precision_error = throw_precision_error

        if self._double_effort_at_rerun:
            assert hasattr(self._solver, 'get_instance_with_double_effort')

    def minimize(self, f: Callable):
        # check whether to double the solver's effort permanently makes sense
        check_failure_rate = self._double_effort_at_rerun and self._solver_runs_counter > 10
        if check_failure_rate and self._failure_rate() > self.global_effort_doubling_threshold:
            self._double_effort_globally()

        _solver = self._solver
        _best_f_all = np.inf
        _best_x_all = None
        self._solver_runs_counter += 1

        for i in range(self._max_repeats):
            x, y = _solver.minimize(f)
            x2, y2 = _solver.minimize(f)
            _best_x_iter, _best_f_iter = (x, y) if y < y2 else (x2, y2)
            if _best_f_iter < _best_f_all:
                _best_x_all, _best_f_all = _best_x_iter, _best_f_iter
            diff = np.abs(y-y2)
            if diff <= self._atol:
                return _best_x_all, _best_f_all
            else:
                msg = (f'Solutions of independent solvers of the DoubleSolverWrapper differ by {diff}. ' +
                       'Repeating minimization with doubled effort.'
                        if self._double_effort_at_rerun else 'Repeating minimization.')
                warnings.warn(msg)
                print(msg)
                self._precision_failure_counter += 1
                if self._double_effort_at_rerun:
                    _solver = _solver.get_instance_with_double_effort()

        msg = ('DoubleSolverWrapper did not obtain independent solutions in the specified '
               f'absolute tolerance of {self._atol} with {self._max_repeats} repetitions.')
        if self._throw_precision_error:
            raise RuntimeError(msg)
        else:
            msg += ' Returning the best solution so far.'
            warnings.warn(msg)
            print(msg)
            return _best_x_all, _best_f_all

    @property
    def requires_gradients(self):
        return self._solver.requires_gradients

    @property
    def requires_safety(self):
        return self._solver.requires_safety

    def _double_effort_globally(self) -> None:
        msg = "Doubled effort of solver permanently"
        warnings.warn(msg)
        print(msg)
        self._solver_runs_counter = 0
        self._precision_failure_counter = 0
        self._solver = self._solver.get_instance_with_double_effort()

    def _failure_rate(self) -> float:
        if self._solver_runs_counter > 0:
            return self._precision_failure_counter / self._solver_runs_counter
        else:
            return 0.0


""" helpers """


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
            1-D arrays to form the cartesian product of.
    out : ndarray
            Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
                 [1, 4, 7],
                 [1, 5, 6],
                 [1, 5, 7],
                 [2, 4, 6],
                 [2, 4, 7],
                 [2, 5, 6],
                 [2, 5, 7],
                 [3, 4, 6],
                 [3, 4, 7],
                 [3, 5, 6],
                 [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    m = int(m)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out