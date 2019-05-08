from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from SALib.sample import sobol_sequence

from .geometry import Geometry
from .geometry_nd import Hypercube


class Disk(Geometry):
    def __init__(self, center, radius):
        super(Disk, self).__init__(2, 2 * radius)
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


class Rectangle(Hypercube):
    def __init__(self, xmin, xmax):
        """
        Args:
            xmin: Coordinate of bottom left corner.
            xmax: Coordinate of top right corner.
        """
        super(Rectangle, self).__init__(xmin, xmax)
        self.perimeter = 2 * np.sum(self.xmax - self.xmin)

    def uniform_boundary_points(self, n):
        nx, ny = np.ceil(n / self.perimeter * (self.xmax - self.xmin)).astype(int)
        xbot = np.hstack(
            (
                np.linspace(self.xmin[0], self.xmax[0], num=nx, endpoint=False)[
                    :, None
                ],
                np.full([nx, 1], self.xmin[1]),
            )
        )
        yrig = np.hstack(
            (
                np.full([ny, 1], self.xmax[0]),
                np.linspace(self.xmin[1], self.xmax[1], num=ny, endpoint=False)[
                    :, None
                ],
            )
        )
        xtop = np.hstack(
            (
                np.linspace(self.xmin[0], self.xmax[0], num=nx + 1)[1:, None],
                np.full([nx, 1], self.xmax[1]),
            )
        )
        ylef = np.hstack(
            (
                np.full([ny, 1], self.xmin[0]),
                np.linspace(self.xmin[1], self.xmax[1], num=ny + 1)[1:, None],
            )
        )
        x = np.vstack((xbot, yrig, xtop, ylef))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x_corner = np.vstack(
            (
                self.xmin,
                [self.xmax[0], self.xmin[1]],
                self.xmax,
                [self.xmin[0], self.xmax[1]],
            )
        )
        n -= 4
        if n <= 0:
            return x_corner

        l1 = self.xmax[0] - self.xmin[0]
        l2 = l1 + self.xmax[1] - self.xmin[1]
        l3 = l2 + l1
        if random == "sobol":
            u = np.ravel(sobol_sequence.sample(n + 4, 1))[2:]
            u = u[np.logical_not(np.isclose(u, l1 / self.perimeter))]
            u = u[np.logical_not(np.isclose(u, l3 / self.perimeter))]
            u = u[:n, None]
        else:
            u = np.random.rand(n, 1)
        u *= self.perimeter
        x = []
        for l in u:
            if l < l1:
                x.append([self.xmin[0] + l, self.xmin[1]])
            elif l < l2:
                x.append([self.xmax[0], self.xmin[1] + l - l1])
            elif l < l3:
                x.append([self.xmax[0] - l + l2, self.xmax[1]])
            else:
                x.append([self.xmin[0], self.xmax[1] - l + l3])
        return np.vstack((x_corner, x))


class Triangle(Geometry):
    def __init__(self, x1, x2, x3):
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.x3 = np.array(x3)
        self.v12 = self.x2 - self.x1
        self.v23 = self.x3 - self.x2
        self.v31 = self.x1 - self.x3
        self.l12 = np.linalg.norm(self.v12)
        self.l23 = np.linalg.norm(self.v23)
        self.l31 = np.linalg.norm(self.v31)
        self.n12 = self.v12 / self.l12
        self.n23 = self.v23 / self.l23
        self.n31 = self.v31 / self.l31
        self.perimeter = self.l12 + self.l23 + self.l31
        super(Triangle, self).__init__(
            2,
            self.l12
            * self.l23
            * self.l31
            / (
                self.perimeter
                * (self.l12 + self.l23 - self.l31)
                * (self.l23 + self.l31 - self.l12)
                * (self.l31 + self.l12 - self.l23)
            )
            ** 0.5,
        )

    def random_points(self, n, random="pseudo"):
        """There are two methods for triangle point picking.
        Method 1 (used here):
            https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
        Method 2:
            http://mathworld.wolfram.com/TrianglePointPicking.html
            https://hbfs.wordpress.com/2010/10/05/random-points-in-a-triangle-generating-random-sequences-ii/
            https://stackoverflow.com/questions/19654251/random-point-inside-triangle-inside-java
        """
        sqrt_r1 = np.sqrt(np.random.rand(n, 1))
        r2 = np.random.rand(n, 1)
        return (
            (1 - sqrt_r1) * self.x1
            + sqrt_r1 * (1 - r2) * self.x2
            + r2 * sqrt_r1 * self.x3
        )

    def uniform_boundary_points(self, n):
        density = n / self.perimeter
        x12 = (
            np.linspace(0, 1, num=int(np.ceil(density * self.l12)), endpoint=False)[
                :, None
            ]
            * self.v12
            + self.x1
        )
        x23 = (
            np.linspace(0, 1, num=int(np.ceil(density * self.l23)), endpoint=False)[
                :, None
            ]
            * self.v23
            + self.x2
        )
        x31 = (
            np.linspace(0, 1, num=int(np.ceil(density * self.l31)), endpoint=False)[
                :, None
            ]
            * self.v31
            + self.x3
        )
        x = np.vstack((x12, x23, x31))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x_corner = np.vstack((self.x1, self.x2, self.x3))
        n -= 3
        if n <= 0:
            return x_corner

        if random == "sobol":
            u = np.ravel(sobol_sequence.sample(n + 3, 1))[1:]
            u = u[np.logical_not(np.isclose(u, self.l12 / self.perimeter))]
            u = u[np.logical_not(np.isclose(u, (self.l12 + self.l23) / self.perimeter))]
            u = u[:n, None]
        else:
            u = np.random.rand(n, 1)
        u *= self.perimeter
        x = []
        for l in u:
            if l < self.l12:
                x.append(l * self.n12 + self.x1)
            elif l < self.l12 + self.l23:
                x.append((l - self.l12) * self.n23 + self.x2)
            else:
                x.append((l - self.l12 - self.l23) * self.n31 + self.x3)
        return np.vstack((x_corner, x))
