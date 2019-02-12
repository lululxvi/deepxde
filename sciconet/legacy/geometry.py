# author: Lu Lu
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import itertools

import numpy as np
from SALib.sample import sobol_sequence
from scipy import stats
from sklearn import preprocessing

from nnlearn import config


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


class Interval(Geometry):
    def __init__(self, l, r):
        super(Interval, self).__init__('Interval', 1, r - l)
        self.l, self.r = l, r

    def in_domain(self, x):
        return self.l <= x[0] <= self.r

    def on_boundary(self, x):
        return x[0] == self.l or x[0] == self.r

    def distance2boundary(self, x, dirn):
        return x[0] - self.l if dirn < 0 else \
            self.r - x[0]

    def mindist2boundary(self, x):
        return min(np.amin(x - self.l), np.amin(self.r - x))

    def uniform_points(self, n, boundary):
        if boundary:
            return np.linspace(self.l, self.r, num=n,
                               dtype=config.real(np))[:, None]
        return np.linspace(self.l, self.r, num=n+1, endpoint=False,
                           dtype=config.real(np))[1:, None]

    def random_points(self, n, random):
        if random == 'pseudo':
            x = np.random.rand(n, 1)
        elif random == 'sobol':
            x = sobol_sequence.sample(n + 1, 1)[1:]
        return self.diam * x + self.l

    def uniform_boundary_points(self, n):
        if n == 1:
            return np.array([[self.l]])
        if n == 2:
            return np.array(([[self.l], [self.r]]))
        raise ValueError('Invalid boundary point number %d' % n)

    def random_boundary_points(self, n, random=None):
        if n == 1:
            return np.array([np.random.choice([self.l, self.r], 1)])
        if n == 2:
            return np.array(([[self.l], [self.r]]))
        raise ValueError('Invalid boundary point number %d' % n)

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
            pts = x[0] - np.arange(-shift, n-shift+1) * h
            return pts[:, None]

        def background_points_right():
            dx = self.r - x[0]
            n = max(dist2npt(dx), 1)
            h = dx / n
            pts = x[0] + np.arange(-shift, n-shift+1) * h
            return pts[:, None]

        return background_points_left() if dirn < 0 else \
            background_points_right() if dirn > 0 else \
            np.vstack((background_points_left(), background_points_right()))


class Disk(Geometry):
    def __init__(self, center, radius):
        super(Disk, self).__init__('Disk', 2, 2 * radius)
        self.center, self.radius = center, radius

        self._r2 = radius**2

    def in_domain(self, x):
        return np.linalg.norm(x - self.center) <= self.radius

    def on_boundary(self, x):
        return np.isclose(np.linalg.norm(x - self.center), self.radius)

    def distance2boundary_unitdirn(self, x, dirn):
        """https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        """
        xc = x - self.center
        ad = np.dot(xc, dirn)
        return -ad + (ad**2 - np.dot(xc, xc) + self._r2)**0.5

    def distance2boundary(self, x, dirn):
        return self.distance2boundary_unitdirn(x, dirn / np.linalg.norm(dirn))

    def mindist2boundary(self, x):
        return np.amin(self.radius - np.linalg.norm(x - self.center, axis=1))

    def uniform_points(self, n, boundary):
        raise NotImplementedError("disk.uniform_points to be implemented")

    def random_points(self, n, random):
        """http://mathworld.wolfram.com/DiskPointPicking.html
        """
        if random == 'pseudo':
            rng = np.random.rand(n, 2)
        elif random == 'sobol':
            rng = sobol_sequence.sample(n, 2)
        r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
        x, y = np.cos(theta), np.sin(theta)
        return self.radius * (np.sqrt(r) * np.vstack((x, y))).T + self.center

    def uniform_boundary_points(self, n):
        theta = np.linspace(0, 2*np.pi, num=n, endpoint=False)
        X = np.vstack((np.cos(theta), np.sin(theta))).T
        return self.radius * X + self.center

    def random_boundary_points(self, n, random):
        if random == 'pseudo':
            u = np.random.rand(n, 1)
        elif random == 'sobol':
            u = sobol_sequence.sample(n, 1)
        theta = 2*np.pi * u
        X = np.hstack((np.cos(theta), np.sin(theta)))
        return self.radius * X + self.center

    def background_points(self, x, dirn, dist2npt, shift):
        dirn = dirn / np.linalg.norm(dirn)
        dx = self.distance2boundary_unitdirn(x, -dirn)
        n = max(dist2npt(dx), 1)
        h = dx / n
        pts = x - np.arange(-shift, n-shift+1)[:, None] * h * dirn
        return pts


class Hypercube(Geometry):
    def __init__(self, xmin, xmax):
        if len(xmin) != len(xmax):
            raise ValueError('Dimensions of xmin and xmax do not match.')
        if np.any(np.array(xmin) >= np.array(xmax)):
            raise ValueError('xmin >= xmax')

        self.xmin, self.xmax = np.array(xmin), np.array(xmax)
        super(Hypercube, self).__init__('Hypercube', len(xmin),
                                        np.linalg.norm(self.xmax - self.xmin))

    def in_domain(self, x):
        raise NotImplementedError("Hypercube.in_domain to be implemented")

    def on_boundary(self, x):
        raise NotImplementedError("Hypercube.on_boundary to be implemented")

    def distance2boundary(self, x, dirn):
        raise NotImplementedError(
            "Hypercube.distance2boundary to be implemented")

    def mindist2boundary(self, x):
        raise NotImplementedError(
            "Hypercube.mindist2boundary to be implemented")

    def uniform_points(self, n, boundary):
        n1 = int(np.ceil(n ** (1 / self.dim)))
        xi = []
        for i in range(self.dim):
            if boundary:
                xi.append(np.linspace(self.xmin[i], self.xmax[i], num=n1))
            else:
                xi.append(np.linspace(
                    self.xmin[i], self.xmax[i], num=n1+1, endpoint=False)[1:])
        x = np.array(list(itertools.product(*xi)))
        if n != len(x):
            print('Warning: {} points required, but {} points sampled.'.format(
                n, len(x)))
        return x

    def random_points(self, n, random):
        if random == 'pseudo':
            x = np.random.rand(n, self.dim)
        elif random == 'sobol':
            x = sobol_sequence.sample(n + 1, self.dim)[1:]
        return (self.xmax - self.xmin) * x + self.xmin

    def uniform_boundary_points(self, n):
        raise NotImplementedError(
            "Hypercube.uniform_boundary_points to be implemented")

    def random_boundary_points(self, n, random):
        raise NotImplementedError(
            "Hypercube.random_boundary_points to be implemented")

    def background_points(self, x, dirn, dist2npt, shift):
        raise NotImplementedError(
            "Hypercube.background_points to be implemented")


class Hypersphere(Geometry):
    def __init__(self, center, radius):
        super(Hypersphere, self).__init__('Hypersphere', len(center), 2*radius)
        self.center, self.radius = center, radius

        self._r2 = radius**2

    def in_domain(self, x):
        return np.linalg.norm(x - self.center) <= self.radius

    def on_boundary(self, x):
        return np.isclose(np.linalg.norm(x - self.center), self.radius)

    def distance2boundary_unitdirn(self, x, dirn):
        """https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        """
        xc = x - self.center
        ad = np.dot(xc, dirn)
        return -ad + (ad**2 - np.dot(xc, xc) + self._r2)**0.5

    def distance2boundary(self, x, dirn):
        return self.distance2boundary_unitdirn(x, dirn / np.linalg.norm(dirn))

    def mindist2boundary(self, x):
        return np.amin(self.radius - np.linalg.norm(x - self.center, axis=1))

    def uniform_points(self, n, boundary):
        raise NotImplementedError(
            "Hypersphere.uniform_points to be implemented")

    def random_points(self, n, random):
        """https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
        """
        if random == 'pseudo':
            U = np.random.rand(n, 1)
            X = np.random.normal(size=(n, self.dim))
        elif random == 'sobol':
            rng = sobol_sequence.sample(n + 1, self.dim + 1)[1:]
            U, X = rng[:, 0:1], rng[:, 1:]
            X = stats.norm.ppf(X)
        X = preprocessing.normalize(X)
        X = U**(1/self.dim) * X
        return self.radius * X + self.center

    def uniform_boundary_points(self, n):
        raise NotImplementedError(
            "Hypersphere.uniform_boundary_points to be implemented")

    def random_boundary_points(self, n, random):
        """http://mathworld.wolfram.com/HyperspherePointPicking.html
        """
        if random == 'pseudo':
            X = np.random.normal(size=(n, self.dim))
        elif random == 'sobol':
            U = sobol_sequence.sample(n + 1, self.dim)[1:]
            X = stats.norm.ppf(U)
        X = preprocessing.normalize(X)
        return self.radius * X + self.center

    def background_points(self, x, dirn, dist2npt, shift):
        dirn = dirn / np.linalg.norm(dirn)
        dx = self.distance2boundary_unitdirn(x, -dirn)
        n = max(dist2npt(dx), 1)
        h = dx / n
        pts = x - np.arange(-shift, n-shift+1)[:, None] * h * dirn
        return pts
