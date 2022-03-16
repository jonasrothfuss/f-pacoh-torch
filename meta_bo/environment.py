import numpy as np
import os
import time
import json
import glob

from meta_bo.domain import ContinuousDomain, DiscreteDomain
from meta_bo.solver import EvolutionarySolver
from config import BASE_DIR, DATA_DIR
from typing import Optional, Dict, List

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

    def __init__(self, noise_std=0.0, noise_std_constr=0.0, random_state=None):
        super().__init__()
        self.min_value = None

        self._rds = np.random if random_state is None else random_state
        self.noise_std = noise_std
        self.noise_std_constr = noise_std_constr

    def f(self, x):
        """
        Function to be implemented by actual benchmark.
        """
        raise NotImplementedError

    def q_constraint(self, x):
        """ constraint function"""
        raise NotImplementedError

    def evaluate(self, x, x_bp=None):
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

    def _determine_minimum(self, num_particles_per_d2=2000, max_iter_per_d=500):
        if isinstance(self.domain, ContinuousDomain):
            solver = EvolutionarySolver(self.domain, num_particles_per_d2=num_particles_per_d2,
                                        max_iter_per_d=max_iter_per_d)
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

    @property
    def normalization_stats_constr(self):
        assert self.has_constraint
        if isinstance(self.domain, ContinuousDomain):
            x_points = np.random.uniform(self.domain.l, self.domain.u, size=(1000 * self.domain.d**2, self.domain.d))
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
        self.q = q


class MixtureEnvironment(BenchmarkEnvironment):
    domain = ContinuousDomain(np.array([-10.]), np.array([10.]))
    has_constraint = True

    def __init__(self, params=None, constr_params=None, random_state=None):
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


class ArgusSimEnvironment(Environment):
    domain = ContinuousDomain(np.array([50., 300., 500.]), np.array([400., 1200., 4000.]))
    has_constraint = True
    default_params = {'Ts': 5e-5,  # Sampling time simulation in s
                      'Ctime': 1e-3,  # Sampling time function generator (reference position - RPOS) in s
                      'stepsize': 0.1,  # RPOS stepsize in m (max. feasible 0.2)
                      'jerk': 1e3,  # RPOS max. jerk in m/(s^3)
                      'acc': 10,  # RPOS max. acceleration in m/(s^2)
                      'vmax': 0.1,  # RPOS maximum velocity in m/s
                      'SLAFF': 0}  # Acceleration feedforward gain
    optim_params = [
        'SLPKP',  # Proportional gain position controller (nominal 200)
        'SLVKP', # Proportional gain velocity controller (nominal 600)
        'SLVKI' # Integral gain velocity controller (nominal 1000)
    ]
    _default_max_TV = 1.0

    def __init__(self, params=None, constr_params=None, matlab_enginge=None, logspace_y=False, random_state=None):
        super().__init__()

        self.logspace_y = logspace_y

        # deal with params
        assert params is None or set(params.keys()) <= set(self.default_params.keys())
        assert len(self.optim_params) == self.domain.d
        self.params = self.default_params
        if params is not None:
            self.params.update(params)

        # constrain params
        if constr_params is None:
            self._max_TV = self._default_max_TV
        else:
            self._max_TV = constr_params['max_TV']

        if matlab_enginge is None:
            self.matlab_engine = self._setup_matlab_enginge()
        else:
            self.matlab_engine = matlab_enginge

    @property
    def d(self):
        return len(self.optim_params)

    def evaluate(self, x, x_bp=None):
        assert x.shape == (self.d,) or x.shape == (1, self.d)
        self._t += 1
        optim_param_dict = self._x_param_dict_map(x)
        from argus_sim.Argus_sim import RunSim_Argus
        T_settle, TV = RunSim_Argus(self.matlab_engine, {**self.params, **optim_param_dict})
        if self.logspace_y:
            T_settle = np.log(T_settle)
        evaluation = {'x': x, 't': self._t, 'y': T_settle, 'q': - TV + self._max_TV}

        if x_bp is not None:
            optim_param_dict_bp = self._x_param_dict_map(x_bp)
            T_settle_bp, TV_bp = RunSim_Argus(self.matlab_engine, {**self.params, **optim_param_dict_bp})
            if self.logspace_y:
                T_settle_bp = np.log(T_settle_bp)
            evaluation.update({'x_bp': x_bp, 'y_bp': T_settle_bp, 'q_bp': TV_bp})

        return evaluation

    def _setup_matlab_enginge(self):
        import matlab
        import matlab.engine
        os.chdir(os.path.join(BASE_DIR, 'argus_sim'))  # TODO: This might be dangerous, any better idea how to solve this?
        t_eng_start = time.time()
        matlab_engine = matlab.engine.start_matlab()
        matlab_engine.Argus_Parameters(nargout=0)
        print("Engine setup time: ", time.time() - t_eng_start, "s")
        return matlab_engine

    def _x_param_dict_map(self, x):
        x = x.squeeze()
        return dict([(param_name, x[i]) for i, param_name in enumerate(self.optim_params)])

    @property
    def normalization_stats(self):
        if self.logspace_y:
            y_min, y_max = -5., 2.
        else:
            y_min, y_max = 0., 5.
        stats = {
            'x_mean': (self.domain.l + self.domain.u) / 2.0,
            'x_std': (self.domain.u - self.domain.l) / 5.0,
            'y_mean': np.array((y_max + y_min) / 2.),
            'y_std': np.array((y_max - y_min) / 5.0)
        }
        return stats


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


class ArgusDatatsetEnvironment(DatasetBanditEnvironment):

    _argus_domain_data_dir = os.path.join(DATA_DIR, 'uniform_domain_data')
    has_constraint = True
    optim_params = [
        'SLPKP',  # Proportional gain position controller (nominal 200)
        'SLVKP', # Proportional gain velocity controller (nominal 600)
        'SLVKI' # Integral gain velocity controller (nominal 1000)
    ]
    _max_TV = 1.0

    def __init__(self, params: Optional[Dict] = None, logspace_y: bool = False,
                 random_state: Optional[np.random.RandomState] = None):
        self.logspace_y = logspace_y
        self._all_domain_data = self._load_argus_domain_data()

        if params is None:
            stepsize = self._available_stepsizes[len(self._available_stepsizes) // 2]
        else:
            stepsize = params['stepsize']
            assert stepsize in self._available_stepsizes

        domain_data = self._all_domain_data[stepsize]
        super().__init__(data_x=domain_data['x'], data_y=domain_data['y'], data_q=domain_data['q'],
                         noise_std=0.0, random_state=random_state)

    @cached_property
    def _available_stepsizes(self) -> List[int]:
        return sorted(list(set(self._all_domain_data.keys())))

    def f(self, x: np.ndarray):
        if self.logspace_y:
            np.log(super().f(x))
        else:
            return super().f(x)

    @staticmethod
    def _load_argus_domain_data():
        all_data_dict = {}
        for path in glob.glob(os.path.join(ArgusDatatsetEnvironment._argus_domain_data_dir, '*.json')):
            with open(path, 'r') as f:
                data_dict = json.load(f)
                del data_dict['evals']['t']
                stepsize = data_dict['params']['stepsize']
                if stepsize in all_data_dict:
                    for k, v in data_dict['evals'].items():
                        all_data_dict[stepsize][k] = np.concatenate([all_data_dict[stepsize][k],
                                                                     np.array(v)], axis=0)
                else:
                    all_data_dict[stepsize] = {k: np.array(v) for k, v in data_dict['evals'].items()}
        return all_data_dict
