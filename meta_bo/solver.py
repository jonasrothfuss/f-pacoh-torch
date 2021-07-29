import numpy as np
import math

class Solver:

    def __init__(self, domain, initial_x=None):
        self._domain = domain
        self.initial_x = initial_x

    def minimize(self, f):
        """
            optimize f over domain
            if self.requires_gradients = True, fun should return a tuple of (y,grad)
         """
        raise NotImplementedError

    @property
    def requires_gradients(self):
        return False

    @property
    def requires_safety(self):
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

    def minimize(self, f):
        # res = []
        # for c in self.candidates:
        #     res.append(f(c))
        res = f(self.candidates)
        index = np.argmin(res)
        best_x = self.candidates[index]
        return best_x, res[index]

    @property
    def requires_gradients(self):
        return False

    @property
    def requires_safety(self):
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

    def __init__(self, domain, survival_rate=0.9, num_particles_per_d=100, num_iter_per_d=30, initial_x=None,
                 random_state=None):
        super().__init__(domain, initial_x=initial_x)
        self.d = domain.d
        self.domain = domain
        self._rds = np.random if random_state is None else random_state

        self.num_particles = self.d * (num_particles_per_d if num_particles_per_d is None else num_particles_per_d)
        self.num_iter = self.d * (num_iter_per_d if num_iter_per_d is None else num_iter_per_d)
        self.survival_rate = survival_rate if survival_rate is None else survival_rate

        self.max_sampling_radius = (self.domain.u - self.domain.l) * self.d / self.num_particles

    def minimize(self, f):
        best_f = 1e8 * np.ones(self.num_particles)
        particles = self._rds.uniform(self.domain.l, self.domain.u, size=(self.num_particles, self.domain.d))
        for i in range(self.num_iter):
            temp = 1 - i / self.num_iter
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


        best_idx = np.argmin(best_f)

        return particles[best_idx], best_f[best_idx]

    def _uniform_sample_unit_ball(self):
        # sample from unit sphere
        x = self._rds.normal(size=(self.num_particles, self.d))
        x = x / np.linalg.norm(x, axis=-1)[:, None]

        # transform in unit ball
        u = self._rds.uniform(0, 1, size=(self.num_particles, ))
        return x * (u**(1/self.d))[:, None]

    @property
    def requires_gradients(self):
        return False

    @property
    def requires_safety(self):
        return False


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