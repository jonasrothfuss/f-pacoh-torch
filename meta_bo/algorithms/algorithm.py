from meta_bo.domain import ContinuousDomain
import warnings
import numpy as np

class Algorithm:
    """
    Base class for algorithms.
    """

    def __init__(self):
        self.name = type(self).__name__

    def initialize(self, **kwargs):
        """
        Called to initialize the algorithm. Resets the algorithm, discards previous data.

        Args:
            **kwargs: Arbitrary keyword arguments, received from environment.initialize().
            domain (DiscreteDomain, ContinousDomain): Mandatory argument. Domain of the environment
            initial_evaluation: (Optional) Initial evaluation taken from the environment

        """
        self.domain = kwargs.get("domain")
        self.x0 = kwargs.get("x0", None)
        self._exit = False
        self.t = 0

        # # TODO maybe we need the constraints
        # self.lower_bound_objective_value = kwargs.get("lower_bound_objective_value", None)
        # self.num_constraints = kwargs.get("num_constraints", None)

        self._best_x = None
        self._best_y = -10e10

    def _next(self, context=None):
        """
        Called by next(), uxsed to get proposed parameter.
        Opposed to ``next()``, does return only x, not a tuple.
        Returns: parameter x
        """
        raise NotImplementedError

    def next(self, context=None):
        """
        Called to get next evaluation point from the algorithm.
        By default uses  self._next() to get a proposed parameter, and creates additional_data

        Returns: Tuple (x, additional_data), where x is the proposed parameter, and additional_data is np 1-dim array of dtype self.dtype

        """
        if context is None:
            # call without context (algorithm might not allow context argument)
            next_x = self._next()
        else:
            # call with context
            next_x = self._next(context)

        if isinstance(next_x, tuple):
            x = next_x[0]
            additional_data = next_x[1]
        else:
            x = next_x
            additional_data = {}
        additional_data['t'] = self.t
        self.t += 1

        # for continuous domains, check if x is inside box
        if isinstance(self.domain, ContinuousDomain):
            if (x > self.domain.u).any() or (x < self.domain.l).any():
                warnings.warn('Point outside domain. Projecting back into box.'
                              f'\nx is {x}, with limits {self.domain.l}, {self.domain.u}')
                x = np.maximum(np.minimum(x, self.domain.u), self.domain.l)

        return x, additional_data

    def add_data(self, data):
        """
        Add observation data to the algorithm.

        Args:
            data: TBD

        """
        if data['y'] < self._best_y:
            self._best_y = data['y']
            self._best_x = data['x']

        self.collected_data.append(data)

    def finalize(self):
        return {'initial_data' : self.collected_data,
                'best_x' : self.best_predicted()}

    def best_predicted(self):
        """
        If implemented, this should returns a point in the domain, which is currently believed to be best
        Returns:
        """
        return self._best_x


