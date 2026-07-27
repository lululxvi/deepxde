import numpy as np

from . import geometry
from .. import config


class CSGUnion(geometry.Geometry):
    """Construct an object by CSG Union."""

    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError(
                "{} | {} failed (dimensions do not match).".format(
                    geom1.idstr, geom2.idstr
                )
            )
        super().__init__(
            geom1.dim,
            (
                np.minimum(geom1.bbox[0], geom2.bbox[0]),
                np.maximum(geom1.bbox[1], geom2.bbox[1]),
            ),
            geom1.diam + geom2.diam,
        )
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return np.logical_or(self.geom1.inside(x), self.geom2.inside(x))

    def on_boundary(self, x):
        # Prevent shared overlapping boundaries from being excluded
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x) | self.geom2.on_boundary(x)),
            np.logical_and(self.geom2.on_boundary(x), ~self.geom1.inside(x) | self.geom1.on_boundary(x)),
        )

    def boundary_normal(self, x):
        # Mutually exclusive overlap handling to prevent doubling normal vectors
        mask1 = np.logical_and(
            self.geom1.on_boundary(x), ~self.geom2.inside(x) | self.geom2.on_boundary(x)
        )
        mask2 = np.logical_and(self.geom2.on_boundary(x), ~self.geom1.inside(x))
        
        return mask1[:, np.newaxis] * self.geom1.boundary_normal(x) + \
               mask2[:, np.newaxis] * self.geom2.boundary_normal(x)

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:
            tmp = (
                np.random.rand(n, self.dim) * (self.bbox[1] - self.bbox[0])
                + self.bbox[0]
            )
            tmp = tmp[self.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            # geom1 claims points on the shared overlapping boundary
            geom1_boundary_points = geom1_boundary_points[
                ~self.geom2.inside(geom1_boundary_points) | self.geom2.on_boundary(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            # geom2 discards points on overlap to prevent double-density
            geom2_boundary_points = geom2_boundary_points[
                ~self.geom1.inside(geom2_boundary_points)
            ]

            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))
            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def periodic_point(self, x, component):
        x = np.copy(x)
        # geom1 handles the overlap mapping
        on_boundary_geom1 = np.logical_and(
            self.geom1.on_boundary(x), ~self.geom2.inside(x) | self.geom2.on_boundary(x)
        )
        x[on_boundary_geom1] = self.geom1.periodic_point(x, component)[
            on_boundary_geom1
        ]
        
        # geom2 ignores overlap to prevent mapping the same point twice
        on_boundary_geom2 = np.logical_and(
            self.geom2.on_boundary(x), ~self.geom1.inside(x)
        )
        x[on_boundary_geom2] = self.geom2.periodic_point(x, component)[
            on_boundary_geom2
        ]
        return x


class CSGDifference(geometry.Geometry):
    """Construct an object by CSG Difference."""

    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError(
                "{} - {} failed (dimensions do not match).".format(
                    geom1.idstr, geom2.idstr
                )
            )
        super().__init__(geom1.dim, geom1.bbox, geom1.diam)
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return np.logical_and(self.geom1.inside(x), ~self.geom2.inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x) & ~self.geom2.on_boundary(x)),
            np.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x) & ~self.geom1.on_boundary(x)),
        )

    def boundary_normal(self, x):
        return np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x))[
            :, np.newaxis
        ] * self.geom1.boundary_normal(x) + np.logical_and(
            self.geom1.inside(x), self.geom2.on_boundary(x)
        )[
            :, np.newaxis
        ] * -self.geom2.boundary_normal(
            x
        )

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[~self.geom2.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[
                ~self.geom2.inside(geom1_boundary_points) & ~self.geom2.on_boundary(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[
                self.geom1.inside(geom2_boundary_points) & ~self.geom1.on_boundary(geom2_boundary_points)
            ]

            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def periodic_point(self, x, component):
        x = np.copy(x)
        on_boundary_geom1 = np.logical_and(
            self.geom1.on_boundary(x), ~self.geom2.inside(x)
        )
        x[on_boundary_geom1] = self.geom1.periodic_point(x, component)[
            on_boundary_geom1
        ]
        return x


class CSGIntersection(geometry.Geometry):
    """Construct an object by CSG Intersection."""

    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError(
                "{} & {} failed (dimensions do not match).".format(
                    geom1.idstr, geom2.idstr
                )
            )
        super().__init__(
            geom1.dim,
            (
                np.maximum(geom1.bbox[0], geom2.bbox[0]),
                np.minimum(geom1.bbox[1], geom2.bbox[1]),
            ),
            min(geom1.diam, geom2.diam),
        )
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return np.logical_and(self.geom1.inside(x), self.geom2.inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), self.geom2.inside(x)),
            np.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x)),
        )

    def boundary_normal(self, x):
        return np.logical_and(self.geom1.on_boundary(x), self.geom2.inside(x))[
            :, np.newaxis
        ] * self.geom1.boundary_normal(x) + np.logical_and(
            self.geom1.inside(x), self.geom2.on_boundary(x)
        )[
            :, np.newaxis
        ] * self.geom2.boundary_normal(
            x
        )

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[self.geom2.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[
                self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[
                self.geom1.inside(geom2_boundary_points)
            ]

            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def periodic_point(self, x, component):
        x = np.copy(x)
        on_boundary_geom1 = np.logical_and(
            self.geom1.on_boundary(x), self.geom2.inside(x)
        )
        x[on_boundary_geom1] = self.geom1.periodic_point(x, component)[
            on_boundary_geom1
        ]
        on_boundary_geom2 = np.logical_and(
            self.geom2.on_boundary(x), self.geom1.inside(x)
        )
        x[on_boundary_geom2] = self.geom2.periodic_point(x, component)[
            on_boundary_geom2
        ]
        return x
