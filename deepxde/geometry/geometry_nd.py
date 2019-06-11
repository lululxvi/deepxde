from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from SALib.sample import sobol_sequence
from scipy import stats
from sklearn import preprocessing

from .geometry import Geometry


class Hypercube(Geometry):
    def __init__(self, xmin, xmax):
        if len(xmin) != len(xmax):
            raise ValueError("Dimensions of xmin and xmax do not match.")
        if np.any(np.array(xmin) >= np.array(xmax)):
            raise ValueError("xmin >= xmax")

        self.xmin, self.xmax = np.array(xmin), np.array(xmax)
        super(Hypercube, self).__init__(
            len(xmin), (self.xmin, self.xmax), np.linalg.norm(self.xmax - self.xmin)
        )

    def inside(self, x):
        return np.all(x >= self.xmin) and np.all(x <= self.xmax)

    def on_boundary(self, x):
        return self.inside(x) and (
            np.any(np.isclose(x, self.xmin)) or np.any(np.isclose(x, self.xmax))
        )

    def boundary_normal(self, x):
        n = np.zeros(self.dim)
        for i, xi in enumerate(x):
            if np.isclose(xi, self.xmin[i]):
                n[i] = -1
                break
            if np.isclose(xi, self.xmax[i]):
                n[i] = 1
                break
        return n

    def uniform_points(self, n, boundary=True):
        n1 = int(np.ceil(n ** (1 / self.dim)))
        xi = []
        for i in range(self.dim):
            if boundary:
                xi.append(np.linspace(self.xmin[i], self.xmax[i], num=n1))
            else:
                xi.append(
                    np.linspace(self.xmin[i], self.xmax[i], num=n1 + 1, endpoint=False)[
                        1:
                    ]
                )
        x = np.array(list(itertools.product(*xi)))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_points(self, n, random="pseudo"):
        if random == "pseudo":
            x = np.random.rand(n, self.dim)
        elif random == "sobol":
            x = sobol_sequence.sample(n + 1, self.dim)[1:]
        return (self.xmax - self.xmin) * x + self.xmin

    def periodic_point(self, x, component):
        y = np.copy(x)
        if np.isclose(y[component], self.xmin[component]):
            y[component] += self.xmax[component] - self.xmin[component]
        elif np.isclose(y[component], self.xmax[component]):
            y[component] -= self.xmax[component] - self.xmin[component]
        return y


class Hypersphere(Geometry):
    def __init__(self, center, radius):
        self.center, self.radius = np.array(center), radius
        super(Hypersphere, self).__init__(
            len(center), (self.center - radius, self.center + radius), 2 * radius
        )

        self._r2 = radius ** 2

    def inside(self, x):
        return np.linalg.norm(x - self.center) <= self.radius

    def on_boundary(self, x):
        return np.isclose(np.linalg.norm(x - self.center), self.radius)

    def distance2boundary_unitdirn(self, x, dirn):
        """https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        """
        xc = x - self.center
        ad = np.dot(xc, dirn)
        return -ad + (ad ** 2 - np.dot(xc, xc) + self._r2) ** 0.5

    def distance2boundary(self, x, dirn):
        return self.distance2boundary_unitdirn(x, dirn / np.linalg.norm(dirn))

    def mindist2boundary(self, x):
        return np.amin(self.radius - np.linalg.norm(x - self.center, axis=1))

    def random_points(self, n, random="pseudo"):
        """https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
        """
        if random == "pseudo":
            U = np.random.rand(n, 1)
            X = np.random.normal(size=(n, self.dim))
        elif random == "sobol":
            rng = sobol_sequence.sample(n + 1, self.dim + 1)[1:]
            U, X = rng[:, 0:1], rng[:, 1:]
            X = stats.norm.ppf(X)
        X = preprocessing.normalize(X)
        X = U ** (1 / self.dim) * X
        return self.radius * X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        """http://mathworld.wolfram.com/HyperspherePointPicking.html
        """
        if random == "pseudo":
            X = np.random.normal(size=(n, self.dim))
        elif random == "sobol":
            U = sobol_sequence.sample(n + 1, self.dim)[1:]
            X = stats.norm.ppf(U)
        X = preprocessing.normalize(X)
        return self.radius * X + self.center

    def background_points(self, x, dirn, dist2npt, shift):
        dirn = dirn / np.linalg.norm(dirn)
        dx = self.distance2boundary_unitdirn(x, -dirn)
        n = max(dist2npt(dx), 1)
        h = dx / n
        pts = x - np.arange(-shift, n - shift + 1)[:, None] * h * dirn
        return pts
