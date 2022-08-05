import itertools

import numpy as np

from .geometry_1d import Interval
from .geometry_2d import Rectangle
from .geometry_3d import Cuboid
from .geometry_nd import Hypercube
from .. import config


class TimeDomain(Interval):
    def __init__(self, t0, t1):
        super().__init__(t0, t1)
        self.t0 = t0
        self.t1 = t1

    def on_initial(self, t):
        return np.isclose(t, self.t0).flatten()


class GeometryXTime:
    def __init__(self, geometry, timedomain):
        self.geometry = geometry
        self.timedomain = timedomain
        self.dim = geometry.dim + timedomain.dim

    def on_boundary(self, x):
        return self.geometry.on_boundary(x[:, :-1])

    def on_initial(self, x):
        return self.timedomain.on_initial(x[:, -1:])

    def boundary_normal(self, x):
        _n = self.geometry.boundary_normal(x[:, :-1])
        return np.hstack([_n, np.zeros((len(_n), 1))])

    def uniform_points(self, n, boundary=True):
        """Uniform points on the spatio-temporal domain.

        Geometry volume ~ bbox.
        Time volume ~ diam.
        """
        nx = int(
            np.ceil(
                (
                    n
                    * np.prod(self.geometry.bbox[1] - self.geometry.bbox[0])
                    / self.timedomain.diam
                )
                ** 0.5
            )
        )
        nt = int(np.ceil(n / nx))
        x = self.geometry.uniform_points(nx, boundary=boundary)
        nx = len(x)
        if boundary:
            t = self.timedomain.uniform_points(nt, boundary=True)
        else:
            t = np.linspace(
                self.timedomain.t1,
                self.timedomain.t0,
                num=nt,
                endpoint=False,
                dtype=config.real(np),
            )[:, None]
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
        if isinstance(self.geometry, Interval):
            geom = Rectangle(
                [self.geometry.l, self.timedomain.t0],
                [self.geometry.r, self.timedomain.t1],
            )
            return geom.random_points(n, random=random)

        if isinstance(self.geometry, Rectangle):
            geom = Cuboid(
                [self.geometry.xmin[0], self.geometry.xmin[1], self.timedomain.t0],
                [self.geometry.xmax[0], self.geometry.xmax[1], self.timedomain.t1],
            )
            return geom.random_points(n, random=random)

        if isinstance(self.geometry, (Cuboid, Hypercube)):
            geom = Hypercube(
                np.append(self.geometry.xmin, self.timedomain.t0),
                np.append(self.geometry.xmax, self.timedomain.t1),
            )
            return geom.random_points(n, random=random)

        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.random_points(n, random=random)
        t = np.random.permutation(t)
        return np.hstack((x, t))

    def uniform_boundary_points(self, n):
        """Uniform boundary points on the spatio-temporal domain.

        Geometry surface area ~ bbox.
        Time surface area ~ diam.
        """
        if self.geometry.dim == 1:
            nx = 2
        else:
            s = 2 * sum(
                map(
                    lambda l: l[0] * l[1],
                    itertools.combinations(
                        self.geometry.bbox[1] - self.geometry.bbox[0], 2
                    ),
                )
            )
            nx = int((n * s / self.timedomain.diam) ** 0.5)
        nt = int(np.ceil(n / nx))
        x = self.geometry.uniform_boundary_points(nx)
        nx = len(x)
        t = np.linspace(
            self.timedomain.t1,
            self.timedomain.t0,
            num=nt,
            endpoint=False,
            dtype=config.real(np),
        )
        xt = []
        for ti in t:
            xt.append(np.hstack((x, np.full([nx, 1], ti))))
        xt = np.vstack(xt)
        if n != len(xt):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(xt))
            )
        return xt

    def random_boundary_points(self, n, random="pseudo"):
        x = self.geometry.random_boundary_points(n, random=random)
        t = self.timedomain.random_points(n, random=random)
        t = np.random.permutation(t)
        return np.hstack((x, t))

    def uniform_initial_points(self, n):
        x = self.geometry.uniform_points(n, True)
        t = self.timedomain.t0
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return np.hstack((x, np.full([len(x), 1], t, dtype=config.real(np))))

    def random_initial_points(self, n, random="pseudo"):
        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.t0
        return np.hstack((x, np.full([n, 1], t, dtype=config.real(np))))

    def periodic_point(self, x, component):
        xp = self.geometry.periodic_point(x[:, :-1], component)
        return np.hstack([xp, x[:, -1:]])
