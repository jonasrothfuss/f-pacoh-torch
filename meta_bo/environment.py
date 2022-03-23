import numpy as np
import os
import time
import json
import glob

from meta_bo.domain import ContinuousDomain, DiscreteDomain
from meta_bo.solver import EvolutionarySolver
from config import BASE_DIR, DATA_DIR
from typing import Optional, Dict, List, Tuple

from functools import cached_property


class Environment:
    domain = None

    def __init__(self):
        self.tmax = None
        self._x0 = None
        self._t = 0

    @property
    def name(self):
        return f"{type(self).__module__}.{type(self).__name__}"

    def evaluate(self, x):
        raise NotImplementedError


class BenchmarkEnvironment(Environment):
    has_constraint = None

    def __init__(self, noise_std: float = 0.0, noise_std_constr: float = 0.0,
                 random_state: Optional[np.random.RandomState] = None):
        super().__init__()
        self.min_value = None

        self._rds = np.random if random_state is None else random_state
        self.noise_std = noise_std
        self.noise_std_constr = noise_std_constr

    def f(self, x: np.ndarray) -> np.ndarray:
        """
        Function to be implemented by actual benchmark.
        """
        raise NotImplementedError

    def q_constraint(self, x: np.ndarray) -> np.ndarray:
        """ constraint function"""
        raise NotImplementedError

    def evaluate(self, x: np.ndarray, x_bp: Optional[np.ndarray] = None) -> Dict:
        self._t += 1
        evaluation = {'x': x, 't': self._t}
        evaluation['y_exact'] = np.asscalar(self.f(x))
        evaluation['y_min'] = self.min_value

        evaluation['y_std'] = self.noise_std
        evaluation['y'] = evaluation['y_exact'] + self.noise_std * self._rds.normal(0, 1)

        if self.has_constraint:
            evaluation['q_excact'] = np.asscalar(self.q_constraint(x))
            evaluation['q_std'] = self.noise_std_constr
            evaluation['q'] = evaluation['q_excact'] + self.noise_std_constr * self._rds.normal(0, 1)

        if x_bp is not None:
            evaluation['x_bp'] = x_bp
            evaluation['y_exact_bp'] = np.asscalar(self.f(x_bp))
            evaluation['y_bp'] = evaluation['y_exact'] + self.noise_std * self._rds.normal(0, 1)

        return evaluation

    def _determine_minimum(self, num_particles_per_d2: int = 2000, max_iter_per_d: int = 500):
        if isinstance(self.domain, ContinuousDomain):
            solver = EvolutionarySolver(self.domain, num_particles_per_d2=num_particles_per_d2,
                                        max_iter_per_d=max_iter_per_d)
            solution = solver.minimize(lambda x: self.f(x))
            return solution[1]
        elif isinstance(self.domain, DiscreteDomain):
            return np.argmin(self.f(self.domain.points))

    def generate_uniform_data(self, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self.domain, ContinuousDomain):
            x = self._rds.uniform(self.domain.l, self.domain.u, size=(num_points, self.domain.d))
        elif isinstance(self.domain, DiscreteDomain):
            x = self._rds.choice(self.domain.points, num_points, replace=True)
        else:
            raise AssertionError
        y = self.f(x) + self.noise_std * self._rds.normal(0, 1, num_points)
        return x, y

    @cached_property
    def normalization_stats(self) -> Dict:
        if isinstance(self.domain, ContinuousDomain):
            x_points = self._rds.uniform(self.domain.l, self.domain.u, size=(1000 * self.domain.d**2, self.domain.d))
        elif isinstance(self.domain, DiscreteDomain):
            x_points = self.domain.points
        else:
            raise NotImplementedError
        ys = self.f(x_points)
        y_min, y_max = np.min(ys), np.max(ys)
        stats = {
            'x_mean': np.mean(x_points, axis=0),
            'x_std': np.std(x_points, axis=0),
            'y_mean': (y_max + y_min) / 2.,
            'y_std': (y_max - y_min) / 5.0
        }
        return stats

    @cached_property
    def normalization_stats_constr(self) -> Dict:
        assert self.has_constraint
        if isinstance(self.domain, ContinuousDomain):
            x_points = self._rds.uniform(self.domain.l, self.domain.u, size=(1000 * self.domain.d**2, self.domain.d))
        elif isinstance(self.domain, DiscreteDomain):
            x_points = self.domain.points
        else:
            raise NotImplementedError
        ys = self.q_constraint(x_points)
        y_min, y_max = np.min(ys), np.max(ys)
        stats = {
            'x_mean': np.mean(x_points, axis=0),
            'x_std': np.std(x_points, axis=0),
            'y_mean': (y_max + y_min) / 2.,
            'y_std': (y_max - y_min) / 5.0
        }
        return stats


class BraninEnvironment(BenchmarkEnvironment):
    domain = ContinuousDomain(np.array([-5., 0.]), np.array([10., 15.]))
    has_constraint = True

    def __init__(self, params: Optional[Dict] = None, random_state: Optional[np.random.RandomState] = None):
        super().__init__(noise_std=2.0, random_state=random_state)
        if params is not None:
            assert set(params.keys()) == {'a', 'b', 'c', 'r', 's', 't', 'constr_a', 'constr_b'}
            self._params = params
            self._construct_f_from_params(self._params)
            self.min_value = 0.397887
        else:
            self._params = {'a': 1.0, 'b': 5.1 / (4*np.pi**2), 'c': 5./np.pi, 'r': 6., 's': 10, 't': 1/(8*np.pi),
                            'constr_a': 2.0, 'constr_b': 1.0}
            self._construct_f_from_params(self._params)
            self.min_value = self._determine_minimum()
        self._construct_q_from_params(self._params)

    def _construct_f_from_params(self, params: Dict):
        def fun(x):
            x = np.array(x)
            assert x.ndim <= 2
            if x.ndim == 2:
                x1, x2 = x[:, 0], x[:, 1]
            else:
                x1, x2 = x[0], x[1]
            return (params['a'] * (x2 - params['b'] * x1 ** 2 + params['c'] * x1 - params['r']) ** 2 +
                    params['s'] * (1 - params['t']) * np.cos(x1) + params['s'])
        self.f = fun

    def _construct_q_from_params(self, params: Dict):
        def q(x):
            x = np.array(x)
            assert x.ndim <= 2
            if x.ndim == 2:
                x1, x2 = x[:, 0], x[:, 1]
            else:
                x1, x2 = x[0], x[1]

            x1 = x1 / 5. - 1
            x2 = x2 /  5. - 1
            return params['constr_a'] * x1**2 - 1.05 * x1**4 + x1**6 / 6. + x1 * params['constr_b'] * x2 + x2**2 - 1.
        self.q_constraint = q


class MixtureEnvironment(BenchmarkEnvironment):
    domain = ContinuousDomain(np.array([-10.]), np.array([10.]))
    has_constraint = True

    def __init__(self, params: Dict = None, constr_params: Dict = None,
                 random_state: Optional[np.random.RandomState] = None):
        super().__init__(noise_std=0.02, random_state=random_state)
        if params is not None:
            assert set(params.keys()) == {'loc1', 'loc2', 'loc3', 'scales'}
            self._params = params
        else:
            self._params = {'loc1': -2, 'loc2': 3, 'loc3': -8, 'scales': np.ones(3)}
        self._construct_f_from_params(self._params)
        self.min_value = self._determine_minimum()

        if constr_params is not None:
            assert set(params.keys()) == {'loc1', 'loc2', 'loc3', 'scales'}
            self._constr_params = constr_params
        else:
            self._constr_params = {'loc1': -5, 'loc2': 8, 'scales': np.ones(2)}
        self._construct_q_from_params(self._constr_params)

    def _construct_f_from_params(self, params):
        def fun(x):
            d = self.domain.d
            x = np.reshape(x, (x.shape[0], d))
            cauchy1 = 1 / (np.pi * (1 + (np.linalg.norm(x - params['loc1'], axis=-1) / d) ** 2))
            gaussian = 1 / np.sqrt(2 * np.pi) * np.exp(
                - 0.5 * (np.linalg.norm(x - params['loc2'], axis=-1) / (d * 2)) ** 2)
            cauchy3 = 1 / (np.pi * (1 + (np.linalg.norm(x - params['loc3'], axis=-1) / (d * 4)) ** 2))
            return - params['scales'][0] * 2. * cauchy1 - 1.5 * params['scales'][1] * gaussian - \
                   params['scales'][2] * 1.8 * cauchy3 - 1
        self.f = fun

    def _construct_q_from_params(self, params):
        def constr_fun(x):
            d = self.domain.d
            x = np.reshape(x, (x.shape[0], d))
            cauchy1 = 1 / (np.pi * (1 + (np.linalg.norm(x - params['loc1'], axis=-1) / d) ** 2))
            gaussian = 1 / np.sqrt(2 * np.pi) * np.exp(
                - 0.5 * (np.linalg.norm(x - params['loc2'], axis=-1) / (d * 2)) ** 2)
            #cauchy3 = 1 / (np.pi * (1 + (np.linalg.norm(x - params['loc3'], axis=-1) / (d * 4)) ** 2))
            return params['scales'][0] * 2. * cauchy1 + 2 * params['scales'][1] * gaussian - 0.4
        self.q_constraint = constr_fun


class DatasetBanditEnvironment(BenchmarkEnvironment):

    def __init__(self, data_x: np.ndarray, data_y: np.ndarray, data_q: Optional[np.ndarray] = None,
                 noise_std: float = 0.0, random_state: Optional[np.random.RandomState] = None):
        super().__init__(noise_std=noise_std, noise_std_constr=0.0, random_state=random_state)

        assert data_x.shape[0] == data_y.shape[0]
        assert data_y.ndim == 1 or (data_y.ndim == 2 and data_y.shape[1] == 1)
        self.domain = DiscreteDomain(points=data_x)
        self._data_x = data_x
        self._data_y = data_y.reshape((-1,))
        if data_q is None:
            self.has_constraint = False
            self._data_q = None
        else:
            assert data_q.shape[0] == data_x.shape[0]
            self._data_q = data_q
            self.has_constraint = True

        self.min_value = np.min(self._data_y)

    def f(self, x: np.ndarray):
        idx = self._find_matching_idx(x)
        y = self._data_y[idx]
        return y

    def q_constraint(self, x: np.ndarray):
        assert self.has_constraint, 'Environment has no contraint'
        idx = self._find_matching_idx(x)
        y = self._data_q[idx]
        return y

    def _find_matching_idx(self, x: np.ndarray) -> int:
        if x.ndim == 2 and x.shape[0] == 1:
            x = x.reshape(x.shape[-1])
        assert x.ndim == 1, 'can only query one point at a time'

        matching_indices = np.where(np.prod(x == self._data_x, axis=-1))[0]
        if len(matching_indices) == 1:
            idx = matching_indices[0]
        else:
            idx = np.argmin(np.linalg.norm(x - self._data_x, axis=-1))
        return idx

    @property
    def normalization_stats(self):
        y_min, y_max = np.min(self._data_y), np.max(self._data_y)
        stats = {
            'x_mean': np.mean(self._data_x, axis=0),
            'x_std': np.std(self._data_x, axis=0),
            'y_mean': (y_max + y_min) / 2.,
            'y_std': (y_max - y_min) / 5.0
        }
        return stats
