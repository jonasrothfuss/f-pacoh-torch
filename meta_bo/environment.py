import numpy as np
from .domain import ContinuousDomain, DiscreteDomain
from .solver import EvolutionarySolver

from functools import lru_cache

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

    def __init__(self, noise_std=0.0, random_state=None):
        super().__init__()
        self.min_value = None
        self._rds = np.random if random_state is None else random_state
        self.noise_std = noise_std

    def f(self, x):
        """
        Function to be implemented by actual benchmark.
        """
        raise NotImplementedError

    def evaluate(self, x, x_bp=None):
        self._t += 1
        evaluation = {'x': x, 't': self._t}
        evaluation['y_exact'] = np.asscalar(self.f(x))
        evaluation['y_min'] = self.min_value

        evaluation['y_std'] = self.noise_std
        evaluation['y'] = evaluation['y_exact'] + self.noise_std * self._rds.normal(0, 1)

        if x_bp is not None:
            evaluation['x_bp'] = x_bp
            evaluation['y_exact_bp'] = np.asscalar(self.f(x_bp))
            evaluation['y_bp'] = evaluation['y_exact'] + self.noise_std * self._rds.normal(0, 1)

        return evaluation

    def _determine_minimum(self, num_particles_per_d=2000, num_iter_per_d=500):
        if isinstance(self.domain, ContinuousDomain):
            solver = EvolutionarySolver(self.domain, num_particles_per_d=num_particles_per_d, num_iter_per_d=num_iter_per_d)
            solution = solver.minimize(lambda x: self.f(x))
            return solution[1]
        elif isinstance(self.domain, DiscreteDomain):
            return np.argmin(self.f(self.domain.points))

    @property
    def normalization_stats(self):
        if isinstance(self.domain, ContinuousDomain):
            x_points = np.random.uniform(self.domain.l, self.domain.u, size=(1000 * self.domain.d**2, self.domain.d))
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

class BraninEnvironment(BenchmarkEnvironment):
    domain = ContinuousDomain(np.array([-5, 0]), np.array([10, 15]))

    def __init__(self, params=None, random_state=None):
        super().__init__(noise_std=2.0, random_state=random_state)
        if params is not None:
            assert set(params.keys()) == {'a', 'b', 'c', 'r', 's', 't'}
            self._params = params
            self._construct_f_from_params(self._params)
            self.min_value = 0.397887
        else:
            self._params = {'a': 1.0, 'b': 5.1 / (4*np.pi**2), 'c': 5./np.pi, 'r': 6., 's': 10, 't': 1/(8*np.pi)}
            self._construct_f_from_params(self._params)
            self.min_value = self._determine_minimum()

    def _construct_f_from_params(self, params):
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


class MixtureEnvironment(BenchmarkEnvironment):
    domain = ContinuousDomain(np.array([-10]), np.array([10]))

    def __init__(self, params=None, random_state=None):
        super().__init__(noise_std=0.02, random_state=random_state)
        if params is not None:
            assert set(params.keys()) == {'loc1', 'loc2', 'loc3', 'scales'}
            self._params = params
        else:
            self._params = {'loc1': -2, 'loc2': 3, 'loc3': -8, 'scales': np.ones(3)}
        self._construct_f_from_params(self._params)
        self.min_value = self._determine_minimum()


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

