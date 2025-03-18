import itertools
from typing import Literal

import brainstate as bst
import jax
import jax.numpy as jnp
from scipy import stats
from sklearn import preprocessing

from deepxde.geometry.sampler import sample
from deepxde.experimental import utils
from .base import GeometryExperimental as Geometry
from ..utils import isclose


class Hypercube(Geometry):
    """
    Represents a hypercube geometry in N-dimensional space.

    This class defines a hypercube with specified minimum and maximum coordinates
    for each dimension.
    """

    def __init__(self, xmin, xmax):
        """
        Initialize a Hypercube object.

        Args:
            xmin (array-like): Minimum coordinates for each dimension.
            xmax (array-like): Maximum coordinates for each dimension.

        Raises:
            ValueError: If dimensions of xmin and xmax do not match or if xmin >= xmax.
        """
        if len(xmin) != len(xmax):
            raise ValueError("Dimensions of xmin and xmax do not match.")

        self.xmin = jnp.array(xmin, dtype=bst.environ.dftype())
        self.xmax = jnp.array(xmax, dtype=bst.environ.dftype())
        if jnp.any(self.xmin >= self.xmax):
            raise ValueError("xmin >= xmax")

        self.side_length = self.xmax - self.xmin
        super().__init__(
            len(xmin),
            (self.xmin, self.xmax),
            jnp.linalg.norm(self.side_length)
        )
        self.volume = jnp.prod(self.side_length)

    def inside(self, x):
        """
        Check if points are inside the hypercube.

        Args:
            x (array-like): Points to check.

        Returns:
            array-like: Boolean array indicating whether each point is inside the hypercube.
        """
        mod = utils.smart_numpy(x)
        return mod.logical_and(mod.all(x >= self.xmin, axis=-1),
                               mod.all(x <= self.xmax, axis=-1))

    def on_boundary(self, x):
        """
        Check if points are on the boundary of the hypercube.

        Args:
            x (array-like): Points to check.

        Returns:
            array-like: Boolean array indicating whether each point is on the boundary.
        """
        mod = utils.smart_numpy(x)
        if x.ndim == 0:
            _on_boundary = mod.logical_or(mod.isclose(x, self.xmin),
                                          mod.isclose(x, self.xmax))
        else:
            _on_boundary = mod.logical_or(
                mod.any(mod.isclose(x, self.xmin), axis=-1),
                mod.any(mod.isclose(x, self.xmax), axis=-1),
            )
        return mod.logical_and(self.inside(x), _on_boundary)

    def boundary_normal(self, x):
        """
        Compute the normal vectors at boundary points.

        Args:
            x (array-like): Points on the boundary.

        Returns:
            array-like: Normal vectors at the given points.
        """
        mod = utils.smart_numpy(x)
        _n = -mod.isclose(x, self.xmin).astype(bst.environ.dftype()) + mod.isclose(x, self.xmax)
        # For vertices, the normal is averaged for all directions
        idx = mod.count_nonzero(_n, axis=-1) > 1
        _n = jax.vmap(lambda idx_, n_: jax.numpy.where(idx_, n_ / mod.linalg.norm(n_, keepdims=True), n_))(idx, _n)
        return mod.asarray(_n)

    def uniform_points(self, n, boundary=True):
        """
        Generate uniformly distributed points in the hypercube.

        Args:
            n (int): Number of points to generate.
            boundary (bool): Whether to include boundary points.

        Returns:
            array-like: Uniformly distributed points.
        """
        dx = (self.volume / n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = int(jnp.ceil(self.side_length[i] / dx))
            if boundary:
                xi.append(
                    jnp.linspace(
                        self.xmin[i], self.xmax[i], num=ni, dtype=bst.environ.dftype()
                    )
                )
            else:
                xi.append(
                    jnp.linspace(
                        self.xmin[i],
                        self.xmax[i],
                        num=ni + 1,
                        endpoint=False,
                        dtype=bst.environ.dftype(),
                    )[1:]
                )
        x = jnp.array(list(itertools.product(*xi)))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def uniform_boundary_points(self, n):
        """
        Generate uniformly distributed points on the boundary of the hypercube.

        Args:
            n (int): Number of points to generate.

        Returns:
            array-like: Uniformly distributed boundary points.
        """
        points_per_face = max(1, n // (2 * self.dim))
        points = []
        for d in range(self.dim):
            for boundary in [self.xmin[d], self.xmax[d]]:
                xi = []
                for i in range(self.dim):
                    if i == d:
                        xi.append(jnp.array([boundary], dtype=bst.environ.dftype()))
                    else:
                        ni = int(jnp.ceil(points_per_face ** (1 / (self.dim - 1))))
                        xi.append(
                            jnp.linspace(
                                self.xmin[i],
                                self.xmax[i],
                                num=ni + 1,
                                endpoint=False,
                                dtype=bst.environ.dftype(),
                            )[1:]
                        )
                face_points = jnp.array(list(itertools.product(*xi)))
                points.append(face_points)
        points = jnp.vstack(points)
        if n != len(points):
            print(
                "Warning: {} points required, but {} points sampled.".format(
                    n, len(points)
                )
            )
        return points

    def random_points(self, n, random="pseudo"):
        """
        Generate random points inside the hypercube.

        Args:
            n (int): Number of points to generate.
            random (str): Type of random number generation ("pseudo" or other).

        Returns:
            array-like: Randomly generated points.
        """
        x = sample(n, self.dim, random)
        return (self.xmax - self.xmin) * x + self.xmin

    def random_boundary_points(self, n, random="pseudo"):
        """
        Generate random points on the boundary of the hypercube.

        Args:
            n (int): Number of points to generate.
            random (str): Type of random number generation ("pseudo" or other).

        Returns:
            array-like: Randomly generated boundary points.
        """
        x = sample(n, self.dim, random)
        # Randomly pick a dimension
        rand_dim = bst.random.randint(self.dim, size=n)
        # Replace value of the randomly picked dimension with the nearest boundary value (0 or 1)
        x[jnp.arange(n), rand_dim] = jnp.round(x[jnp.arange(n), rand_dim])
        return (self.xmax - self.xmin) * x + self.xmin

    def periodic_point(self, x, component):
        """
        Map points to their periodic counterparts on the opposite face of the hypercube.

        Args:
            x (array-like): Points to map.
            component (int): The dimension along which to apply periodicity.

        Returns:
            array-like: Mapped periodic points.
        """
        y = jnp.copy(x)
        _on_xmin = isclose(y[:, component], self.xmin[component])
        _on_xmax = isclose(y[:, component], self.xmax[component])
        y[:, component][_on_xmin] = self.xmax[component]
        y[:, component][_on_xmax] = self.xmin[component]
        return y

    def boundary_constraint_factor(
        self,
        x,
        smoothness: Literal["C0", "C0+", "Cinf"] = "C0",
        where: None = None,
        inside: bool = True,
    ):
        """
        Compute the hard constraint factor at x for the boundary.

        This function is used for the hard-constraint methods in Physics-Informed Neural Networks (PINNs).
        The hard constraint factor satisfies the following properties:

        - The function is zero on the boundary and positive elsewhere.
        - The function is at least continuous.

        In the ansatz `boundary_constraint_factor(x) * NN(x) + boundary_condition(x)`, when `x` is on the boundary,
        `boundary_constraint_factor(x)` will be zero, making the ansatz be the boundary condition, which in
        turn makes the boundary condition a "hard constraint".

        Args:
            x: A 2D array of shape (n, dim), where `n` is the number of points and
                `dim` is the dimension of the geometry. Note that `x` should be a tensor type
                of backend (e.g., `tf.Tensor` or `torch.Tensor`), not a numpy array.
            smoothness (string, optional): A string to specify the smoothness of the distance function,
                e.g., "C0", "C0+", "Cinf". "C0" is the least smooth, "Cinf" is the most smooth.
                Default is "C0".

                - C0
                The distance function is continuous but may not be non-differentiable.
                But the set of non-differentiable points should have measure zero,
                which makes the probability of the collocation point falling in this set be zero.

                - C0+
                The distance function is continuous and differentiable almost everywhere. The
                non-differentiable points can only appear on boundaries. If the points in `x` are
                all inside or outside the geometry, the distance function is smooth.

                - Cinf
                The distance function is continuous and differentiable at any order on any
                points. This option may result in a polynomial of HIGH order.

                - WARNING
                In current implementation,
                numerical underflow may happen for high dimensionalities
                when `smoothness="C0+"` or `smoothness="Cinf"`.

            where (string, optional): This option is currently not supported for Hypercube.
            inside (bool, optional): The `x` is either inside or outside the geometry.
                The cases where there are both points inside and points
                outside the geometry are NOT allowed. NOTE: currently only support `inside=True`.

        Returns:
            A tensor of a type determined by the backend, which will have a shape of (n, 1).
            Each element in the tensor corresponds to the computed distance value for the respective point in `x`.
        """
        if where is not None:
            raise ValueError("where is currently not supported for Hypercube")
        if smoothness not in ["C0", "C0+", "Cinf"]:
            raise ValueError("smoothness must be one of C0, C0+, Cinf")
        if not inside:
            raise ValueError("inside=False is not supported for Hypercube")

        if not hasattr(self, "self.xmin_tensor"):
            self.xmin_tensor = jnp.asarray(self.xmin)
            self.xmax_tensor = jnp.asarray(self.xmax)

        dist_l = jnp.abs(
            (x - self.xmin_tensor) / (self.xmax_tensor - self.xmin_tensor) * 2
        )
        dist_r = jnp.abs(
            (x - self.xmax_tensor) / (self.xmax_tensor - self.xmin_tensor) * 2
        )
        if smoothness == "C0":
            dist_l = jnp.min(dist_l, axis=-1, keepdims=True)
            dist_r = jnp.min(dist_r, axis=-1, keepdims=True)
            return jnp.minimum(dist_l, dist_r)
        # TODO: fix potential numerical underflow
        dist_l = jnp.prod(dist_l, axis=-1, keepdims=True)
        dist_r = jnp.prod(dist_r, dim=-1, keepdims=True)
        return dist_l * dist_r


class Hypersphere(Geometry):
    """
    Represents a hypersphere geometry in N-dimensional space.

    This class defines a hypersphere with a specified center and radius.
    """

    def __init__(self, center, radius):
        """
        Initialize a Hypersphere object.

        Args:
            center (array-like): Coordinates of the center of the hypersphere.
            radius (float): Radius of the hypersphere.
        """
        self.center = jnp.array(center, dtype=bst.environ.dftype())
        self.radius = radius
        super().__init__(
            len(center), (self.center - radius, self.center + radius), 2 * radius
        )

        self._r2 = radius ** 2

    def inside(self, x):
        """
        Check if points are inside the hypersphere.

        Args:
            x (array-like): Points to check.

        Returns:
            array-like: Boolean array indicating whether each point is inside the hypersphere.
        """
        return jnp.linalg.norm(x - self.center, axis=-1) <= self.radius

    def on_boundary(self, x):
        """
        Check if points are on the boundary of the hypersphere.

        Args:
            x (array-like): Points to check.

        Returns:
            array-like: Boolean array indicating whether each point is on the boundary.
        """
        return isclose(jnp.linalg.norm(x - self.center, axis=-1), self.radius)

    def distance2boundary_unitdirn(self, x, dirn):
        """
        Compute the distance from points to the boundary along a unit direction.

        Args:
            x (array-like): Points to compute distance from.
            dirn (array-like): Unit direction vector.

        Returns:
            array-like: Distances from points to the boundary along the given direction.
        """
        xc = x - self.center
        ad = jnp.dot(xc, dirn)
        return (-ad + (ad ** 2 - jnp.sum(xc * xc, axis=-1) + self._r2) ** 0.5).astype(
            bst.environ.dftype()
        )

    def distance2boundary(self, x, dirn):
        """
        Compute the distance from points to the boundary along a given direction.

        Args:
            x (array-like): Points to compute distance from.
            dirn (array-like): Direction vector (will be normalized).

        Returns:
            array-like: Distances from points to the boundary along the given direction.
        """
        return self.distance2boundary_unitdirn(x, dirn / jnp.linalg.norm(dirn))

    def mindist2boundary(self, x):
        """
        Compute the minimum distance from points to the boundary.

        Args:
            x (array-like): Points to compute distance from.

        Returns:
            array-like: Minimum distances from points to the boundary.
        """
        return jnp.amin(self.radius - jnp.linalg.norm(x - self.center, axis=-1))

    def boundary_constraint_factor(
        self, x, smoothness: Literal["C0", "C0+", "Cinf"] = "C0+"
    ):
        """
        Compute the boundary constraint factor for given points.

        Args:
            x (array-like): Points to compute the factor for.
            smoothness (str): Smoothness of the constraint factor. Options are "C0", "C0+", or "Cinf".

        Returns:
            array-like: Boundary constraint factors for the given points.
        """
        if smoothness not in ["C0", "C0+", "Cinf"]:
            raise ValueError("smoothness must be one of C0, C0+, Cinf")

        if not hasattr(self, "self.center_tensor"):
            self.center_tensor = jnp.asarray(self.center)
            self.radius_tensor = jnp.asarray(self.radius)

        dist = jnp.linalg.norm(x - self.center_tensor, axis=-1, keepdims=True) - self.radius
        if smoothness == "Cinf":
            dist = jnp.square(dist)
        else:
            dist = jnp.abs(dist)
        return dist

    def boundary_normal(self, x):
        """
        Compute the normal vectors at boundary points.

        Args:
            x (array-like): Points on the boundary.

        Returns:
            array-like: Normal vectors at the given points.
        """
        _n = x - self.center
        l = jnp.linalg.norm(_n, axis=-1, keepdims=True)
        _n = _n / l * isclose(l, self.radius)
        return _n

    def random_points(self, n, random="pseudo"):
        """
        Generate random points inside the hypersphere.

        Args:
            n (int): Number of points to generate.
            random (str): Type of random number generation ("pseudo" or other).

        Returns:
            array-like: Randomly generated points.
        """
        if random == "pseudo":
            U = bst.random.rand(n, 1).astype(bst.environ.dftype())
            X = bst.random.normal(size=(n, self.dim)).astype(bst.environ.dftype())
        else:
            rng = sample(n, self.dim + 1, random)
            U, X = rng[:, 0:1], rng[:, 1:]
            X = stats.norm.ppf(X).astype(bst.environ.dftype())
        X = preprocessing.normalize(X)
        X = U ** (1 / self.dim) * X
        return self.radius * X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        """
        Generate random points on the boundary of the hypersphere.

        Args:
            n (int): Number of points to generate.
            random (str): Type of random number generation ("pseudo" or other).

        Returns:
            array-like: Randomly generated boundary points.
        """
        if random == "pseudo":
            X = bst.random.normal(size=(n, self.dim)).astype(bst.environ.dftype())
        else:
            U = sample(n, self.dim, random)
            X = stats.norm.ppf(U).astype(bst.environ.dftype())
        X = preprocessing.normalize(X)
        return self.radius * X + self.center

    def background_points(self, x, dirn, dist2npt, shift):
        """
        Generate background points along a direction from given points.

        Args:
            x (array-like): Starting points.
            dirn (array-like): Direction vector.
            dist2npt (callable): Function to determine number of points based on distance.
            shift (float): Shift factor for point generation.

        Returns:
            array-like: Generated background points.
        """
        dirn = dirn / jnp.linalg.norm(dirn)
        dx = self.distance2boundary_unitdirn(x, -dirn)
        n = max(dist2npt(dx), 1)
        h = dx / n
        pts = (
            x
            - jnp.arange(-shift, n - shift + 1, dtype=bst.environ.dftype())[:, None]
            * h
            * dirn
        )
        return pts
