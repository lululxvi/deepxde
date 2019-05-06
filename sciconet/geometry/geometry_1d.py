from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from SALib.sample import sobol_sequence

from .geometry import Geometry
from .. import config


class Interval(Geometry):
    def __init__(self, l, r):
        super(Interval, self).__init__("Interval", 1, r - l)
        self.l, self.r = l, r

    def in_domain(self, x):
        return self.l <= x[0] <= self.r

    def on_boundary(self, x):
        return x[0] == self.l or x[0] == self.r

    def distance2boundary(self, x, dirn):
        return x[0] - self.l if dirn < 0 else self.r - x[0]

    def mindist2boundary(self, x):
        return min(np.amin(x - self.l), np.amin(self.r - x))

    def uniform_points(self, n, boundary):
        if boundary:
            return np.linspace(self.l, self.r, num=n, dtype=config.real(np))[:, None]
        return np.linspace(
            self.l, self.r, num=n + 1, endpoint=False, dtype=config.real(np)
        )[1:, None]

    def log_uniform_points(self, n, boundary):
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

    def random_points(self, n, random):
        if random == "pseudo":
            x = np.random.rand(n, 1)
        elif random == "sobol":
            x = sobol_sequence.sample(n + 1, 1)[1:]
        return self.diam * x + self.l

    def uniform_boundary_points(self, n):
        if n == 1:
            return np.array([[self.l]])
        if n == 2:
            return np.array(([[self.l], [self.r]]))
        raise ValueError("Invalid boundary point number %d" % n)

    def random_boundary_points(self, n, random=None):
        if n == 1:
            return np.array([np.random.choice([self.l, self.r], 1)])
        if n == 2:
            return np.array(([[self.l], [self.r]]))
        raise ValueError("Invalid boundary point number %d" % n)

    def background_points(self, x, dirn, dist2npt, shift):
        """
        dirn: -1 --> left, 1 --> right, 0 --> both direction
        dist2npt: a function which converts distance to the number of extra
                  points (not including x)
        shift: the number of shift
        """

        def background_points_left():
            dx = x[0] - self.l
            n = max(dist2npt(dx), 1)
            h = dx / n
            pts = x[0] - np.arange(-shift, n - shift + 1) * h
            return pts[:, None]

        def background_points_right():
            dx = self.r - x[0]
            n = max(dist2npt(dx), 1)
            h = dx / n
            pts = x[0] + np.arange(-shift, n - shift + 1) * h
            return pts[:, None]

        return (
            background_points_left()
            if dirn < 0
            else background_points_right()
            if dirn > 0
            else np.vstack((background_points_left(), background_points_right()))
        )
