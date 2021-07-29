import numpy as np

from .environment import MixtureEnvironment, BraninEnvironment
from .domain import ContinuousDomain, DiscreteDomain

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

