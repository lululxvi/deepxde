import numpy as np

from .geometry import Geometry
from .sampler import sample
from .. import config


class Interval(Geometry):
    def __init__(self, l, r):
        super().__init__(1, (np.array([l]), np.array([r])), r - l)
        self.l, self.r = l, r

    def inside(self, x):
        return np.logical_and(self.l <= x, x <= self.r).flatten()

    def on_boundary(self, x):
        return np.any(np.isclose(x, [self.l, self.r]), axis=-1)

    def distance2boundary(self, x, dirn):
        return x - self.l if dirn < 0 else self.r - x

    def mindist2boundary(self, x):
        return min(np.amin(x - self.l), np.amin(self.r - x))

    def boundary_normal(self, x):
        return -np.isclose(x, self.l).astype(config.real(np)) + np.isclose(x, self.r)

    def uniform_points(self, n, boundary=True):
        if boundary:
            return np.linspace(self.l, self.r, num=n, dtype=config.real(np))[:, None]
        return np.linspace(
            self.l, self.r, num=n + 1, endpoint=False, dtype=config.real(np)
        )[1:, None]

    def log_uniform_points(self, n, boundary=True):
        eps = 0 if self.l > 0 else np.finfo(config.real(np)).eps
        l = np.log(self.l + eps)
        r = np.log(self.r + eps)
        if boundary:
            x = np.linspace(l, r, num=n, dtype=config.real(np))[:, None]
        else:
            x = np.linspace(l, r, num=n + 1, endpoint=False, dtype=config.real(np))[
                1:, None
            ]
        return np.exp(x) - eps

    def random_points(self, n, random="pseudo"):
        x = sample(n, 1, random)
        return (self.diam * x + self.l).astype(config.real(np))

    def uniform_boundary_points(self, n):
        if n == 1:
            return np.array([[self.l]]).astype(config.real(np))
        xl = np.full((n // 2, 1), self.l).astype(config.real(np))
        xr = np.full((n - n // 2, 1), self.r).astype(config.real(np))
        return np.vstack((xl, xr))

    def random_boundary_points(self, n, random="pseudo"):
        if n == 2:
            return np.array([[self.l], [self.r]]).astype(config.real(np))
        return np.random.choice([self.l, self.r], n)[:, None].astype(config.real(np))

    def periodic_point(self, x, component=0):
        tmp = np.copy(x)
        tmp[np.isclose(x, self.l)] = self.r
        tmp[np.isclose(x, self.r)] = self.l
        return tmp

    def background_points(self, x, dirn, dist2npt, shift):
        """
        Args:
            dirn: -1 (left), or 1 (right), or 0 (both direction).
            dist2npt: A function which converts distance to the number of extra
                points (not including x).
            shift: The number of shift.
        """

        def background_points_left():
            dx = x[0] - self.l
            n = max(dist2npt(dx), 1)
            h = dx / n
            pts = x[0] - np.arange(-shift, n - shift + 1, dtype=config.real(np)) * h
            return pts[:, None]

        def background_points_right():
            dx = self.r - x[0]
            n = max(dist2npt(dx), 1)
            h = dx / n
            pts = x[0] + np.arange(-shift, n - shift + 1, dtype=config.real(np)) * h
            return pts[:, None]

        return (
            background_points_left()
            if dirn < 0
            else background_points_right()
            if dirn > 0
            else np.vstack((background_points_left(), background_points_right()))
        )
