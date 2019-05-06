from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from SALib.sample import sobol_sequence

from .geometry import Geometry


class Disk(Geometry):
    def __init__(self, center, radius):
        super(Disk, self).__init__("Disk", 2, 2 * radius)
        self.center, self.radius = center, radius

        self._r2 = radius ** 2

    def in_domain(self, x):
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

    def random_points(self, n, random):
        """http://mathworld.wolfram.com/DiskPointPicking.html
        """
        if random == "pseudo":
            rng = np.random.rand(n, 2)
        elif random == "sobol":
            rng = sobol_sequence.sample(n, 2)
        r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
        x, y = np.cos(theta), np.sin(theta)
        return self.radius * (np.sqrt(r) * np.vstack((x, y))).T + self.center

    def uniform_boundary_points(self, n):
        theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        X = np.vstack((np.cos(theta), np.sin(theta))).T
        return self.radius * X + self.center

    def random_boundary_points(self, n, random):
        if random == "pseudo":
            u = np.random.rand(n, 1)
        elif random == "sobol":
            u = sobol_sequence.sample(n, 1)
        theta = 2 * np.pi * u
        X = np.hstack((np.cos(theta), np.sin(theta)))
        return self.radius * X + self.center

    def background_points(self, x, dirn, dist2npt, shift):
        dirn = dirn / np.linalg.norm(dirn)
        dx = self.distance2boundary_unitdirn(x, -dirn)
        n = max(dist2npt(dx), 1)
        h = dx / n
        pts = x - np.arange(-shift, n - shift + 1)[:, None] * h * dirn
        return pts
