from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from .geometry_1d import Interval


class TimeDomain(Interval):
    def __init__(self, t0, t1):
        super(TimeDomain, self).__init__(t0, t1)
        self.t0 = t0
        self.t1 = t1

    def on_initial(self, t):
        return np.isclose(t[0], self.t0)


class GeometryXTime(object):
    def __init__(self, geometry, timedomain):
        self.geometry = geometry
        self.timedomain = timedomain

    def on_boundary(self, x):
        return self.geometry.on_boundary(x[:-1])

    def on_initial(self, x):
        return self.timedomain.on_initial(x[-1:])

    def boundary_normal(self, x):
        n = self.geometry.boundary_normal(x[:-1])
        return np.append(n, 0)

    def uniform_points(self, n, boundary=True):
        """Uniform points on the spatio-temporal domain.

        Geometry volume ~ diam ^ dim
        Time volume ~ diam
        """
        nx = int(
            (n * self.geometry.diam ** self.geometry.dim / self.timedomain.diam) ** 0.5
        )
        nt = int(np.ceil(n / nx))
        x = self.geometry.uniform_points(nx, boundary=boundary)
        t = self.timedomain.uniform_points(nt, boundary=boundary)
        xt = []
        for ti in t:
            xt.append(np.hstack((x, np.full([nx, 1], ti[0]))))
        xt = np.vstack(xt)
        if n != len(xt):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(xt))
            )
        return xt

    def random_points(self, n, random="pseudo"):
        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.random_points(n, random=random)
        return np.hstack((x, t))

    # def uniform_boundary_points(self, nx, nt):
    #     x = self.geometry.uniform_boundary_points(nx)
    #     t = self.timedomain.uniform_points(nt, True)
    #     xt = []
    #     for ti in t:
    #         xt.append(np.hstack((x, np.full([nx, 1], ti[0]))))
    #     return np.vstack(xt)

    def random_boundary_points(self, n, random="pseudo"):
        x = self.geometry.random_boundary_points(n, random=random)
        t = self.timedomain.random_points(n, random=random)
        return np.hstack((x, t))

    def uniform_initial_points(self, n):
        x = self.geometry.uniform_points(n, True)
        t = self.timedomain.t0
        return np.hstack((x, np.full([n, 1], t)))

    def random_initial_points(self, n, random="pseudo"):
        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.t0
        return np.hstack((x, np.full([n, 1], t)))

    def periodic_point(self, x, component):
        xp = self.geometry.periodic_point(x[:-1])
        return np.append(xp, x[-1])
