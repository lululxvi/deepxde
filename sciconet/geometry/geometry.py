from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class Geometry(object):
    def __init__(self, idstr, dim, diam):
        self.idstr = idstr
        self.dim = dim
        self.diam = diam

    @abc.abstractmethod
    def in_domain(self, x):
        raise NotImplementedError("{}.in_domain to be implemented".format(self.idstr))

    @abc.abstractmethod
    def on_boundary(self, x):
        raise NotImplementedError("{}.on_boundary to be implemented".format(self.idstr))

    @abc.abstractmethod
    def distance2boundary(self, x, dirn):
        raise NotImplementedError(
            "{}.distance2boundary to be implemented".format(self.idstr)
        )

    @abc.abstractmethod
    def mindist2boundary(self, x):
        raise NotImplementedError(
            "{}.mindist2boundary to be implemented".format(self.idstr)
        )

    @abc.abstractmethod
    def uniform_points(self, n, boundary):
        raise NotImplementedError(
            "{}.uniform_points to be implemented".format(self.idstr)
        )

    @abc.abstractmethod
    def random_points(self, n, random):
        raise NotImplementedError(
            "{}.random_points to be implemented".format(self.idstr)
        )

    @abc.abstractmethod
    def uniform_boundary_points(self, n):
        raise NotImplementedError(
            "{}.uniform_boundary_points to be implemented".format(self.idstr)
        )

    @abc.abstractmethod
    def random_boundary_points(self, n, random):
        raise NotImplementedError(
            "{}.random_boundary_points to be implemented".format(self.idstr)
        )

    @abc.abstractmethod
    def background_points(self, x, dirn, dist2npt, shift):
        raise NotImplementedError(
            "{}.background_points to be implemented".format(self.idstr)
        )
