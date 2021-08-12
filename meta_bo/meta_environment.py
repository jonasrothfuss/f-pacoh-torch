import numpy as np
import time
import os

from .environment import MixtureEnvironment, BraninEnvironment, ArgusSimEnvironment
from .domain import ContinuousDomain, DiscreteDomain
from config import BASE_DIR

class MetaEnvironment:

    def __init__(self, random_state=None):
        self._rds = np.random if random_state is None else random_state

    def sample_env_param(self):
        raise NotImplementedError

    def sample_env_params(self, num_envs):
        return [self.sample_env_param() for _ in range(num_envs)]

    def sample_envs(self, num_envs):
        pass


class MetaBenchmarkEnvironment(MetaEnvironment):
    env_class = None

    def sample_env(self):
        return self.env_class(params=self.sample_env_param(), random_state=self._rds)

    def sample_envs(self, num_envs):
        param_list = self.sample_env_params(num_envs)
        return [self.env_class(params=params, random_state=self._rds) for params in param_list]

    def generate_uniform_meta_train_data(self, num_tasks, num_points_per_task):
        envs = self.sample_envs(num_tasks)
        meta_data = []
        for env in envs:
            if isinstance(env.domain, ContinuousDomain):
                x = self._rds.uniform(env.domain.l, env.domain.u,
                                             size=(num_points_per_task, env.domain.d))
            elif isinstance(env.domain, DiscreteDomain):
                x = self._rds.choice(env.domain.points, num_points_per_task, replace=True)
            else:
                raise AssertionError
            y = env.f(x) + env.noise_std * self._rds.normal(0, 1, num_points_per_task)
            meta_data.append((x,y))
        return meta_data

    def generate_uniform_meta_valid_data(self, num_tasks, num_points_context, num_points_test):
        meta_data = self.generate_uniform_meta_train_data(num_tasks, num_points_context+num_points_test)
        meta_valid_data = [(x[:num_points_context], y[:num_points_context],
                            x[num_points_context:], y[num_points_context:]) for x, y in meta_data]
        return meta_valid_data

    @property
    def domain(self):
        return self.env_class.domain

    @property
    def normalization_stats(self):
        meta_data = self.generate_uniform_meta_train_data(20, 1000 * self.domain.d**2)

        if isinstance(self.domain, DiscreteDomain):
            x_points = self.domain.points
        else:
            x_points = np.concatenate([y for x, y in meta_data], axis=0)
        y_concat = np.concatenate([y for x, y in meta_data], axis=0)
        y_min, y_max = np.min(y_concat), np.max(y_concat)
        stats = {
            'x_mean': np.mean(x_points, axis=0),
            'x_std': np.std(x_points, axis=0),
            'y_mean': (y_max + y_min) / 2.,
            'y_std': (y_max - y_min) / 5.0
        }
        return stats


class RandomBraninMetaEnv(MetaBenchmarkEnvironment):
    env_class = BraninEnvironment

    def sample_env_param(self):
        param_dict = {'a': self._rds.uniform(0.5, 1.5),
                      'b': self._rds.uniform(0.1, 0.15),
                      'c': self._rds.uniform(1, 2),
                      'r': self._rds.uniform(5, 7),
                      's': self._rds.uniform(8, 12),
                      't': self._rds.uniform(0.03, 0.05)}
        return param_dict

class RandomMixtureMetaEnv(MetaBenchmarkEnvironment):
    env_class = MixtureEnvironment

    def sample_env_param(self):
        d = 1
        param_dict = {
            'scales': self._rds.uniform(0.6 * np.ones(3), 1.4 * np.ones(3)),
            'loc1': self._rds.normal(-2 * np.ones((d,)), 0.3 * np.ones((d,))),
            'loc2': self._rds.normal(3 * np.ones((d,)), 0.3 * np.ones((d,))),
            'loc3': self._rds.normal(-8 * np.ones((d,)), 0.3 * np.ones((d,))),
        }
        return param_dict

class ArgusSimMetaEnv(MetaBenchmarkEnvironment):
    env_class = ArgusSimEnvironment

    def __init__(self, logspace_y=False, random_state=None):
        super().__init__(random_state=random_state)
        self._rds = np.random if random_state is None else random_state
        self.logspace_y = logspace_y
        self.matlab_engine = self._setup_matlab_enginge()

    def sample_env_param(self):
        param_dict = {
            'stepsize': np.exp(self._rds.uniform(-5, -1)) # sample in log-space between 10um and 100mm
        }
        return param_dict

    def sample_envs(self, num_envs):
        param_list = self.sample_env_params(num_envs)
        return [self.env_class(params=params, matlab_enginge=self.matlab_engine,
                              logspace_y=self.logspace_y) for params in param_list]

    def sample_env(self):
        return self.env_class(params=self.sample_env_param(), matlab_enginge=self.matlab_engine,
                              logspace_y=self.logspace_y)

    def generate_uniform_meta_train_data(self, num_tasks, num_points_per_task):
        envs = self.sample_envs(num_tasks)
        meta_data = []
        for env in envs:
            if isinstance(env.domain, ContinuousDomain):
                x = self._rds.uniform(env.domain.l, env.domain.u,
                                             size=(num_points_per_task, env.domain.d))
                y = np.array([env.evaluate(x[i, :])['y'] for i in range(num_points_per_task)])
            meta_data.append((x, y))
        return meta_data

    def _setup_matlab_enginge(self):
        os.chdir(os.path.join(BASE_DIR, 'argus_sim'))  # TODO: This might be dangerous, any better idea how to solve this?
        import matlab
        import matlab.engine
        t_eng_start = time.time()
        matlab_engine = matlab.engine.start_matlab()
        matlab_engine.Argus_Parameters(nargout=0)
        print("Engine setup time: ", time.time() - t_eng_start, "s")
        return matlab_engine

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