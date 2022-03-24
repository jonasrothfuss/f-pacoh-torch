from meta_bo.domain import DiscreteDomain, ContinuousDomain, Domain
from meta_bo.solver import Solver, FiniteDomainSolver, EvolutionarySolver, DoubleSolverWrapper
from meta_bo.models.abstract import RegressionModel
import numpy as np

from typing import Optional, Tuple, List

class AcquisitionAlgorithm:
    """
    Algorithm which is defined through an acquisition function.
    """

    def __init__(self, model: RegressionModel, domain: Domain, x0: np.ndarray = None,
                 solver: Optional[Solver] = None, random_state: Optional[np.random.RandomState] = None):
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
        x_bp, _ = self.solver.minimize(lambda x: self.model.predict(x, return_density=False)[0])
        return x_bp

    def _get_solver(self, domain):
        if isinstance(domain, DiscreteDomain):
            return FiniteDomainSolver(domain)
        elif isinstance(domain, ContinuousDomain):
            return DoubleSolverWrapper(solver=EvolutionarySolver(domain, num_particles_per_d2=500,
                                                                 survival_rate=0.98,
                                                                 max_iter_per_d=200, random_state=self._rds),
                                       atol=1e-3, max_repeats=4, double_effort_at_rerun=True,
                                       throw_precision_error=False)


    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['solver']
        return self_dict


class UCB(AcquisitionAlgorithm):

    def __init__(self, model: RegressionModel, domain: Domain, beta: float = 2.0, **kwargs):
        """

        Args:
            model: the probabilistic regression model to use to compute the ucb
            domain: optimization domain
            beta: UCB beta. The UCB is condtructed as UCB = mean + beta * std
            **kwargs:
        """
        super().__init__(model, domain, **kwargs)
        self.beta = beta

    def acquisition(self, x):
        pred_mean, pred_std = self.model.predict(x, return_density=False)
        return pred_mean - self.beta * pred_std  # since we minimize f - we want to minimize the LCB
