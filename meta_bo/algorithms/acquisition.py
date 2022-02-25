from meta_bo.domain import DiscreteDomain, ContinuousDomain
from meta_bo.solver import FiniteDomainSolver, EvolutionarySolver

import numpy as np

class AcquisitionAlgorithm:
    """
    Algorithm which is defined through an acquisition function.
    """

    def __init__(self, model, domain, x0=None, solver=None, random_state=None):
        super().__init__()

        self.model = model
        self.domain = domain
        self.t = 0
        self._x0 = x0
        self._rds = np.random if random_state is None else random_state
        self.solver = self._get_solver(domain=self.domain) if solver is None else solver

    def acquisition(self, x):
        raise NotImplementedError

    def add_data(self, X, y):
        self.model.add_data(X, y)

    def next(self):
        if self.t == 0:
            if self._x0 is not None:
                x = self._x0
            else:
                x = self.domain.default_x0
        else:
            x, _ = self.solver.minimize(self.acquisition)
        self.t += 1
        return x

    def best_predicted(self):
        x_bp, _ = self.solver.minimize(lambda x: self.model.predict_mean_std(x)[0])
        return x_bp

    def _get_solver(self, domain):
        if isinstance(domain, DiscreteDomain):
            return FiniteDomainSolver(domain)
        elif isinstance(domain, ContinuousDomain):
            return EvolutionarySolver(domain, num_particles_per_d=100, max_iter_per_d=50)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['solver']
        return self_dict

class UCB(AcquisitionAlgorithm):

    def __init__(self, model, domain, beta=2.0, **kwargs):
        super().__init__(model, domain, **kwargs)
        self.beta = beta

    def acquisition(self, x):
        pred_mean, pred_std = self.model.predict_mean_std(x)
        return pred_mean - self.beta * pred_std  # since we minimize f - we want to minimize the LCB


class GooseUCB(UCB):

    def __init__(self, model_target, model_constr, domain, beta=2.0, **kwargs):
        super().__init__(model_target, domain, beta=2.0, **kwargs)
        self.beta = beta
        self.model_constr = model_constr

    def add_data(self, X, y, q):
        self.model.add_data(X, y)
        self.model_constr.add_data(X, q)