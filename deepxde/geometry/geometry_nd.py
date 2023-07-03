import itertools
from typing import Literal

import numpy as np
from scipy import stats
from sklearn import preprocessing

from .geometry import Geometry
from .sampler import sample
from .. import backend as bkd
from .. import config
from ..utils import isclose


class Hypercube(Geometry):
    def __init__(self, xmin, xmax):
        if len(xmin) != len(xmax):
            raise ValueError("Dimensions of xmin and xmax do not match.")

        self.xmin = np.array(xmin, dtype=config.real(np))
        self.xmax = np.array(xmax, dtype=config.real(np))
        if np.any(self.xmin >= self.xmax):
            raise ValueError("xmin >= xmax")

        self.side_length = self.xmax - self.xmin
        super().__init__(
            len(xmin), (self.xmin, self.xmax), np.linalg.norm(self.side_length)
        )
        self.volume = np.prod(self.side_length)

    def inside(self, x):
        return np.logical_and(
            np.all(x >= self.xmin, axis=-1), np.all(x <= self.xmax, axis=-1)
        )

    def on_boundary(self, x):
        _on_boundary = np.logical_or(
            np.any(isclose(x, self.xmin), axis=-1),
            np.any(isclose(x, self.xmax), axis=-1),
        )
        return np.logical_and(self.inside(x), _on_boundary)

    def boundary_normal(self, x):
        _n = -isclose(x, self.xmin).astype(config.real(np)) + isclose(x, self.xmax)
        # For vertices, the normal is averaged for all directions
        idx = np.count_nonzero(_n, axis=-1) > 1
        if np.any(idx):
            print(
                f"Warning: {self.__class__.__name__} boundary_normal called on vertices. "
                "You may use PDE(..., exclusions=...) to exclude the vertices."
            )
            l = np.linalg.norm(_n[idx], axis=-1, keepdims=True)
            _n[idx] /= l
        return _n

    def uniform_points(self, n, boundary=True):
        dx = (self.volume / n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = int(np.ceil(self.side_length[i] / dx))
            if boundary:
                xi.append(
                    np.linspace(
                        self.xmin[i], self.xmax[i], num=ni, dtype=config.real(np)
                    )
                )
            else:
                xi.append(
                    np.linspace(
                        self.xmin[i],
                        self.xmax[i],
                        num=ni + 1,
                        endpoint=False,
                        dtype=config.real(np),
                    )[1:]
                )
        x = np.array(list(itertools.product(*xi)))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_points(self, n, random="pseudo"):
        x = sample(n, self.dim, random)
        return (self.xmax - self.xmin) * x + self.xmin

    def random_boundary_points(self, n, random="pseudo"):
        x = sample(n, self.dim, random)
        # Randomly pick a dimension
        rand_dim = np.random.randint(self.dim, size=n)
        # Replace value of the randomly picked dimension with the nearest boundary value (0 or 1)
        x[np.arange(n), rand_dim] = np.round(x[np.arange(n), rand_dim])
        return (self.xmax - self.xmin) * x + self.xmin

    def periodic_point(self, x, component):
        y = np.copy(x)
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
        """Compute the hard constraint factor at x for the boundary.

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
            self.xmin_tensor = bkd.as_tensor(self.xmin)
            self.xmax_tensor = bkd.as_tensor(self.xmax)

        dist_l = bkd.abs(
            (x - self.xmin_tensor) / (self.xmax_tensor - self.xmin_tensor) * 2
        )
        dist_r = bkd.abs(
            (x - self.xmax_tensor) / (self.xmax_tensor - self.xmin_tensor) * 2
        )
        if smoothness == "C0":
            dist_l = bkd.min(dist_l, dim=-1, keepdims=True)
            dist_r = bkd.min(dist_r, dim=-1, keepdims=True)
            return bkd.minimum(dist_l, dist_r)
        # TODO: fix potential numerical underflow
        dist_l = bkd.prod(dist_l, dim=-1, keepdims=True)
        dist_r = bkd.prod(dist_r, dim=-1, keepdims=True)
        return dist_l * dist_r


class Hypersphere(Geometry):
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=config.real(np))
        self.radius = radius
        super().__init__(
            len(center), (self.center - radius, self.center + radius), 2 * radius
        )

        self._r2 = radius**2

    def inside(self, x):
        return np.linalg.norm(x - self.center, axis=-1) <= self.radius

    def on_boundary(self, x):
        return isclose(np.linalg.norm(x - self.center, axis=-1), self.radius)

    def distance2boundary_unitdirn(self, x, dirn):
        # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        xc = x - self.center
        ad = np.dot(xc, dirn)
        return (-ad + (ad**2 - np.sum(xc * xc, axis=-1) + self._r2) ** 0.5).astype(
            config.real(np)
        )

    def distance2boundary(self, x, dirn):
        return self.distance2boundary_unitdirn(x, dirn / np.linalg.norm(dirn))

    def mindist2boundary(self, x):
        return np.amin(self.radius - np.linalg.norm(x - self.center, axis=-1))

    def boundary_constraint_factor(
        self, x, smoothness: Literal["C0", "C0+", "Cinf"] = "C0+"
    ):
        if smoothness not in ["C0", "C0+", "Cinf"]:
            raise ValueError("smoothness must be one of C0, C0+, Cinf")

        if not hasattr(self, "self.center_tensor"):
            self.center_tensor = bkd.as_tensor(self.center)
            self.radius_tensor = bkd.as_tensor(self.radius)

        dist = bkd.norm(x - self.center_tensor, axis=-1, keepdims=True) - self.radius
        if smoothness == "Cinf":
            dist = bkd.square(dist)
        else:
            dist = bkd.abs(dist)
        return dist

    def boundary_normal(self, x):
        _n = x - self.center
        l = np.linalg.norm(_n, axis=-1, keepdims=True)
        _n = _n / l * isclose(l, self.radius)
        return _n

    def random_points(self, n, random="pseudo"):
        # https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
        if random == "pseudo":
            U = np.random.rand(n, 1).astype(config.real(np))
            X = np.random.normal(size=(n, self.dim)).astype(config.real(np))
        else:
            rng = sample(n, self.dim + 1, random)
            U, X = rng[:, 0:1], rng[:, 1:]  # Error if X = [0, 0, ...]
            X = stats.norm.ppf(X).astype(config.real(np))
        X = preprocessing.normalize(X)
        X = U ** (1 / self.dim) * X
        return self.radius * X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        # http://mathworld.wolfram.com/HyperspherePointPicking.html
        if random == "pseudo":
            X = np.random.normal(size=(n, self.dim)).astype(config.real(np))
        else:
            U = sample(n, self.dim, random)  # Error for [0, 0, ...] or [0.5, 0.5, ...]
            X = stats.norm.ppf(U).astype(config.real(np))
        X = preprocessing.normalize(X)
        return self.radius * X + self.center

    def background_points(self, x, dirn, dist2npt, shift):
        dirn = dirn / np.linalg.norm(dirn)
        dx = self.distance2boundary_unitdirn(x, -dirn)
        n = max(dist2npt(dx), 1)
        h = dx / n
        pts = (
            x
            - np.arange(-shift, n - shift + 1, dtype=config.real(np))[:, None]
            * h
            * dirn
        )
        return pts
