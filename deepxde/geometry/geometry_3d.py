from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from .geometry_2d import Rectangle
from .geometry_nd import Hypercube, Hypersphere


class Cuboid(Hypercube):
    """
    Args:
        xmin: Coordinate of bottom left corner.
        xmax: Coordinate of top right corner.
    """

    def __init__(self, xmin, xmax):
        super(Cuboid, self).__init__(xmin, xmax)
        dx = self.xmax - self.xmin
        self.area = 2 * np.sum(dx * np.roll(dx, 2))

    def random_boundary_points(self, n, random="pseudo"):
        # TODO: Remove the corners, whose normal derivative is not well defined.
        x_corner = np.vstack(
            (
                self.xmin,
                [self.xmin[0], self.xmax[1], self.xmin[2]],
                [self.xmax[0], self.xmax[1], self.xmin[2]],
                [self.xmax[0], self.xmin[1], self.xmin[2]],
                self.xmax,
                [self.xmin[0], self.xmax[1], self.xmax[2]],
                [self.xmin[0], self.xmin[1], self.xmax[2]],
                [self.xmax[0], self.xmin[1], self.xmax[2]],
            )
        )
        if n <= 8:
            return x_corner[np.random.choice(8, size=n, replace=False)]

        pts = [x_corner]
        density = (n - 8) / self.area
        rect = Rectangle(self.xmin[:-1], self.xmax[:-1])
        for z in [self.xmin[-1], self.xmax[-1]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((u, np.full((len(u), 1), z))))
        rect = Rectangle(self.xmin[::2], self.xmax[::2])
        for y in [self.xmin[1], self.xmax[1]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((u[:, 0:1], np.full((len(u), 1), y), u[:, 1:])))
        rect = Rectangle(self.xmin[1:], self.xmax[1:])
        for x in [self.xmin[0], self.xmax[0]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((np.full((len(u), 1), x), u)))
        pts = np.vstack(pts)
        if len(pts) > n:
            return pts[np.random.choice(len(pts), size=n, replace=False)]
        return pts

    def uniform_boundary_points(self, n):
        # TODO: Remove the corners, whose normal derivative is not well defined.
        h = (self.area / n) ** 0.5
        nx, ny, nz = np.ceil((self.xmax - self.xmin) / h).astype(int) + 1
        x = np.linspace(self.xmin[0], self.xmax[0], num=nx)
        y = np.linspace(self.xmin[1], self.xmax[1], num=ny)
        z = np.linspace(self.xmin[2], self.xmax[2], num=nz)

        pts = []
        for v in [self.xmin[-1], self.xmax[-1]]:
            u = list(itertools.product(x, y))
            pts.append(np.hstack((u, np.full((len(u), 1), v))))
        if nz > 2:
            for v in [self.xmin[1], self.xmax[1]]:
                u = np.array(list(itertools.product(x, z[1:-1])))
                pts.append(np.hstack((u[:, 0:1], np.full((len(u), 1), v), u[:, 1:])))
        if ny > 2 and nz > 2:
            for v in [self.xmin[0], self.xmax[0]]:
                u = list(itertools.product(y[1:-1], z[1:-1]))
                pts.append(np.hstack((np.full((len(u), 1), v), u)))
        pts = np.vstack(pts)
        if n != len(pts):
            print(
                "Warning: {} points required, but {} points sampled.".format(
                    n, len(pts)
                )
            )
        return pts


class Sphere(Hypersphere):
    """
    Args:
        center: Center of the sphere.
        radius: Radius of the sphere.
    """

    def __init__(self, center, radius):
        super(Sphere, self).__init__(center, radius)
