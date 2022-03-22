from meta_bo.domain import DiscreteDomain, ContinuousDomain, Domain
from meta_bo.solver import FiniteDomainSolver, EvolutionarySolver, DoubleSolverWrapper
from meta_bo.models.abstract import RegressionModel
import numpy as np
# asdsadasd

from typing import Optional, Tuple, List

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

    def __init__(self, model, domain, beta=2.0, **kwargs):
        super().__init__(model, domain, **kwargs)
        self.beta = beta

    def acquisition(self, x):
        pred_mean, pred_std = self.model.predict_mean_std(x)
        return pred_mean - self.beta * pred_std  # since we minimize f - we want to minimize the LCB


class GooseUCB(UCB):

    def __init__(self, model_target: RegressionModel, model_constr: RegressionModel, domain: Domain,
                 beta: float = 2.0, beta_constr: Optional[float] = None, epsilon: float = 0.1, **kwargs):
        super().__init__(model_target, domain, beta=2.0, **kwargs)
        self.beta_constr = beta if beta_constr is None else beta_constr
        self.model_constr = model_constr
        assert epsilon >= 0., 'epsilon must be positive'
        self.epsilon = epsilon
        assert isinstance(domain, DiscreteDomain)
        assert self._x0 is not None, 'x0 (safe set) must be provided'

    def add_data(self, X: np.ndarray, y: float, q: float) -> None:
        super().add_data(X, y)
        self.model_constr.add_data(X, q)

    def next(self):
        if self.t == 0:
            self.t += 1
            return self._x0
        pess_safe_set, opt_safe_set, uncertain_set, informative_expanders_set = self.get_safe_sets()
        x_ucb = self._find_ucb_candidate(opt_safe_set)
        if self._is_in_pessimistic_safe_set(x_ucb)[0]:
            x = x_ucb
        else: # expand
            if len(informative_expanders_set) > 0:
                idx_expander = np.argmin(np.linalg.norm(informative_expanders_set - x_ucb, axis=-1))
                x = informative_expanders_set[idx_expander]
            else:
                x = self._find_ucb_candidate(pess_safe_set)
        self.t += 1
        return x

    def _is_in_pessimistic_safe_set(self, x: np.ndarray) -> List[bool]:
        return self._pessimistic_cond(*self.model_constr.predict(x, include_obs_noise=False))

    def _find_ucb_candidate(self, admissible_points: np.ndarray):
        idx = np.argmin(self.acquisition(admissible_points))
        return admissible_points[idx]

    def _pessimistic_cond(self, q_mean: np.ndarray, q_std: np.ndarray) -> List[bool]:
        return q_mean + self.beta_constr * q_std <= 0

    def _optimistic_cond(self, q_mean: np.ndarray, q_std: np.ndarray) -> List[bool]:
        return q_mean - self.beta_constr * q_std + self.epsilon <= 0

    def get_safe_sets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.array]:
        q_mean, q_std = self.model_constr.predict(self.domain.points, include_obs_noise=False)
        pess_cond = self._pessimistic_cond(q_mean, q_std)
        opt_cond = self._optimistic_cond(q_mean, q_std)
        pessimistic_safe_set_idx = np.where(pess_cond)[0]
        optimistic_safe_set_idx = np.where(opt_cond)[0]
        uncertain_safety_set_idx = np.where(np.logical_and(opt_cond, np.logical_not(pess_cond)))[0]
        sufficient_info_cond = np.where(2 * self.beta_constr * q_std[pessimistic_safe_set_idx] >= self.epsilon)[0]
        informative_expanders_idx = pessimistic_safe_set_idx[sufficient_info_cond]

        pessimistic_safe_set = self.domain.points[pessimistic_safe_set_idx]
        optimistic_safe_set = self.domain.points[optimistic_safe_set_idx]
        uncertain_safety_set = self.domain.points[uncertain_safety_set_idx]
        informative_expanders = self.domain.points[informative_expanders_idx]
        return pessimistic_safe_set, optimistic_safe_set, uncertain_safety_set, informative_expanders
