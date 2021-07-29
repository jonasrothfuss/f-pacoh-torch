import numpy as np

class ContinuousDomain:

    def __init__(self, l, u):
        assert l.ndim == u.ndim == 1 and l.shape[0] == u.shape[0]
        assert np.all(l < u)
        self._l = l
        self._u = u
        self._range = self._u - self._l
        self._d = l.shape[0]
        self._bounds = np.vstack((self._l, self._u)).T

    @property
    def l(self):
        return self._l

    @property
    def u(self):
        return self._u

    @property
    def bounds(self):
        return self._bounds

    @property
    def range(self):
        return self._range

    @property
    def d(self):
        return self._d

    def normalize(self, x):
        return (x - self._l) / self._range

    def denormalize(self, x):
        return x * self._range + self._l

    def project(self, X):
        """
        Project X into domain rectangle.
        """
        return np.minimum(np.maximum(X, self.l), self.u)

    @property
    def is_continuous(self):
        return True

    @property
    def default_x0(self):
        return self._l + self._range / 2
        # use random initial point
        # return np.random.uniform(low=self.l, high=self.u, size=(1, self.d))

class DiscreteDomain:

    def __init__(self, points, d=None):
        if points.ndim == 1:
            points = np.expand_dims(points, axis=-1)
        assert points.ndim == 2
        self._points = points
        if d is None:
            self._d = points[0].shape[0]
        else:
            self._d = d

    @property
    def points(self):
        return self._points

    @property
    def d(self):
        return self._d

    @property
    def num_points(self):
        return len(self._points)

    def normalize(self, x):
        return x

    def denormalize(self, x):
        return x

    @property
    def is_continuous(self):
        return False

    @property
    def default_x0(self):
        return self.points[0]
