import itertools

import brainstate as bst
import jax.numpy as jnp

from .base import GeometryExperimental
from .geometry_1d import Interval
from .geometry_2d import Rectangle
from .geometry_3d import Cuboid
from .geometry_nd import Hypercube
from ..utils import isclose


class TimeDomain(Interval):
    """
    Represents a time domain interval.

    This class extends the Interval class to specifically handle time domains.
    It provides functionality to check if a given time point is at the initial time.

    Attributes:
        t0 (jnp.ndarray): The start time of the domain.
        t1 (jnp.ndarray): The end time of the domain.
    """

    def __init__(self, t0, t1):
        """
        Initialize the TimeDomain.

        Parameters:
            t0 (float or jnp.ndarray): The start time of the domain.
            t1 (float or jnp.ndarray): The end time of the domain.
        """
        super().__init__(t0, t1)
        self.t0 = jnp.asarray(t0, dtype=bst.environ.dftype())
        self.t1 = jnp.asarray(t1, dtype=bst.environ.dftype())

    def on_initial(self, t):
        """
        Check if the given time point is at the initial time (t0).

        Parameters:
            t (jnp.ndarray): The time point(s) to check.

        Returns:
            jnp.ndarray: A boolean array indicating whether each time point is at the initial time.
        """
        return isclose(t, self.t0).flatten()


class GeometryXTime(GeometryExperimental):
    """
    Represents a geometry combined with a time domain for spatio-temporal problems.

    This class extends GeometryExperimental to handle both spatial and temporal dimensions.
    """

    def __init__(self, geometry, timedomain):
        """
        Initialize the GeometryXTime object.

        Parameters:
            geometry (GeometryExperimental): The spatial geometry.
            timedomain (TimeDomain): The time domain.
        """
        self.geometry = geometry
        self.timedomain = timedomain
        super().__init__(
            geometry.dim + timedomain.dim,
            geometry.bbox + timedomain.bbox,
            min(geometry.diam, timedomain.diam),
        )

    def inside(self, x):
        """
        Check if points are inside the spatio-temporal domain.

        Parameters:
            x (jnp.ndarray): Array of points to check.

        Returns:
            jnp.ndarray: Boolean array indicating whether each point is inside the domain.
        """
        return jnp.logical_and(
            self.geometry.inside(x[:, :-1]), self.timedomain.inside(x[:, -1:])
        )

    def on_boundary(self, x):
        """
        Check if points are on the spatial boundary of the domain.

        Parameters:
            x (jnp.ndarray): Array of points to check.

        Returns:
            jnp.ndarray: Boolean array indicating whether each point is on the boundary.
        """
        return self.geometry.on_boundary(x[:, :-1])

    def on_initial(self, x):
        """
        Check if points are at the initial time of the domain.

        Parameters:
            x (jnp.ndarray): Array of points to check.

        Returns:
            jnp.ndarray: Boolean array indicating whether each point is at the initial time.
        """
        return self.timedomain.on_initial(x[:, -1:])

    def boundary_normal(self, x):
        """
        Compute the boundary normal vectors for given points.

        Parameters:
            x (jnp.ndarray): Array of points on the boundary.

        Returns:
            jnp.ndarray: Array of boundary normal vectors.
        """
        _n = self.geometry.boundary_normal(x[:, :-1])
        return jnp.hstack([_n, jnp.zeros((len(_n), 1))])

    def uniform_points(self, n, boundary=True):
        """
        Generate uniform points in the spatio-temporal domain.

        Parameters:
            n (int): Number of points to generate.
            boundary (bool): Whether to include boundary points.

        Returns:
            jnp.ndarray: Array of uniformly distributed points.
        """
        nx = int(
            jnp.ceil(
                (
                    n
                    * jnp.prod(self.geometry.bbox[1] - self.geometry.bbox[0])
                    / self.timedomain.diam
                )
                ** 0.5
            )
        )
        nt = int(jnp.ceil(n / nx))
        x = self.geometry.uniform_points(nx, boundary=boundary)
        nx = len(x)
        if boundary:
            t = self.timedomain.uniform_points(nt, boundary=True)
        else:
            t = jnp.linspace(
                self.timedomain.t1,
                self.timedomain.t0,
                num=nt,
                endpoint=False,
                dtype=bst.environ.dftype(),
            )[:, None]
        xt = []
        for ti in t:
            xt.append(jnp.hstack((x, jnp.full([nx, 1], ti[0]))))
        xt = jnp.vstack(xt)
        if n != len(xt):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(xt))
            )
        return xt

    def random_points(self, n, random="pseudo"):
        """
        Generate random points in the spatio-temporal domain.

        Parameters:
            n (int): Number of points to generate.
            random (str): Type of random number generation ("pseudo" or "sobol").
        """
        if isinstance(self.geometry, (Cuboid, Hypercube)):
            geom = Hypercube(
                jnp.append(self.geometry.xmin, self.timedomain.t0),
                jnp.append(self.geometry.xmax, self.timedomain.t1),
            )
            return geom.random_points(n, random=random)

        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.random_points(n, random=random)
        t = bst.random.permutation(t)
        return jnp.hstack((x, t))

    def uniform_boundary_points(self, n):
        """
        Generate uniform points on the boundary of the spatio-temporal domain.

        Parameters:
            n (int): Number of boundary points to generate.

        Returns:
            jnp.ndarray: Array of uniformly distributed boundary points.
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
        nt = int(jnp.ceil(n / nx))
        x = self.geometry.uniform_boundary_points(nx)
        nx = len(x)
        t = jnp.linspace(
            self.timedomain.t1,
            self.timedomain.t0,
            num=nt,
            endpoint=False,
            dtype=bst.environ.dftype(),
        )
        xt = []
        for ti in t:
            xt.append(jnp.hstack((x, jnp.full([nx, 1], ti))))
        xt = jnp.vstack(xt)
        if n != len(xt):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(xt))
            )
        return xt

    def random_boundary_points(self, n, random="pseudo"):
        """
        Generate random points on the boundary of the spatio-temporal domain.

        Parameters:
            n (int): Number of boundary points to generate.
            random (str): Type of random number generation ("pseudo" or "sobol").

        Returns:
            jnp.ndarray: Array of randomly distributed boundary points.
        """
        x = self.geometry.random_boundary_points(n, random=random)
        t = self.timedomain.random_points(n, random=random)
        t = bst.random.permutation(t)
        return jnp.hstack((x, t))

    def uniform_initial_points(self, n):
        """
        Generate uniform points at the initial time of the spatio-temporal domain.

        Parameters:
            n (int): Number of initial points to generate.

        Returns:
            jnp.ndarray: Array of uniformly distributed initial points.
        """
        x = self.geometry.uniform_points(n, True)
        t = self.timedomain.t0
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return jnp.hstack((x, jnp.full([len(x), 1], t, dtype=bst.environ.dftype())))

    def random_initial_points(self, n, random="pseudo"):
        """
        Generate random points at the initial time of the spatio-temporal domain.

        Parameters:
            n (int): Number of initial points to generate.
            random (str): Type of random number generation ("pseudo" or "sobol").

        Returns:
            jnp.ndarray: Array of randomly distributed initial points.
        """
        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.t0
        return jnp.hstack((x, jnp.full([n, 1], t, dtype=bst.environ.dftype())))

    def periodic_point(self, x, component):
        """
        Map points to their periodic counterparts in the spatial domain.

        Parameters:
            x (jnp.ndarray): Array of points to map.
            component (int): The spatial component for which to apply periodicity.

        Returns:
            jnp.ndarray: Array of mapped periodic points.
        """
        xp = self.geometry.periodic_point(x[:, :-1], component)
        return jnp.hstack([xp, x[:, -1:]])
