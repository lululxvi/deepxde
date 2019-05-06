from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import numpy as np


class Geometry(object):
    def __init__(self, idstr, dim, diam):
        self.idstr = idstr
        self.dim = dim
        self.diam = diam

    @abc.abstractmethod
    def in_domain(self, x):
        return False

    @abc.abstractmethod
    def on_boundary(self, x):
        return False

    @abc.abstractmethod
    def distance2boundary(self, x, dirn):
        return 0

    @abc.abstractmethod
    def mindist2boundary(self, x):
        return 0

    @abc.abstractmethod
    def uniform_points(self, n, boundary):
        return np.array([])

    @abc.abstractmethod
    def random_points(self, n, random):
        return np.array([])

    @abc.abstractmethod
    def uniform_boundary_points(self, n):
        return np.array([])

    @abc.abstractmethod
    def random_boundary_points(self, n, random):
        return np.array([])

    @abc.abstractmethod
    def background_points(self, x, dirn, dist2npt, shift):
        return np.array([])
