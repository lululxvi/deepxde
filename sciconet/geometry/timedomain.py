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


class GeometryXTime(object):
    def __init__(self, geometry, timedomain):
        self.geometry = geometry
        self.timedomain = timedomain

    def uniform_points(self, nx, nt, boundary):
        x = self.geometry.uniform_points(nx, boundary)
        t = self.timedomain.uniform_points(nt, boundary)
        xt = []
        for ti in t:
            xt.append(np.hstack((x, np.full([nx, 1], ti[0]))))
        return np.vstack(xt)

    def random_points(self, n, random="pseudo"):
        x = self.geometry.random_points(n, random)
        t = self.timedomain.random_points(n, random)
        return np.hstack((x, t))

    def uniform_boundary_points(self, nx, nt):
        x = self.geometry.uniform_boundary_points(nx)
        t = self.timedomain.uniform_points(nt, True)
        xt = []
        for ti in t:
            xt.append(np.hstack((x, np.full([nx, 1], ti[0]))))
        return np.vstack(xt)

    def uniform_initial_points(self, n):
        x = self.geometry.uniform_points(n, False)
        t = self.timedomain.t0
        return np.hstack((x, np.full([n, 1], t)))
