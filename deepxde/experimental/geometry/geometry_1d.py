# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================

from typing import Literal, Union

import brainstate as bst
import jax.numpy as jnp

from deepxde.geometry.sampler import sample
from deepxde.experimental import utils
from .base import GeometryExperimental as Geometry


class Interval(Geometry):
    """
    Represents a 1D interval geometry.

    This class defines an interval [l, r] and provides various methods for
    working with points within and on the boundary of this interval.
    """

    def __init__(self, l, r):
        """
        Initialize the Interval object.

        Args:
            l (float): The left endpoint of the interval.
            r (float): The right endpoint of the interval.
        """
        super().__init__(
            1,
            (jnp.array([l], dtype=bst.environ.dftype()),
             jnp.array([r], dtype=bst.environ.dftype())),
            r - l
        )
        self.l, self.r = l, r

    def inside(self, x):
        """
        Check if points are inside the interval.

        Args:
            x (array-like): The points to check.

        Returns:
            array: Boolean array indicating whether each point is inside the interval.
        """
        mod = utils.smart_numpy(x)
        return mod.logical_and(self.l <= x, x <= self.r).flatten()

    def on_boundary(self, x):
        """
        Check if points are on the boundary of the interval.

        Args:
            x (array-like): The points to check.

        Returns:
            array: Boolean array indicating whether each point is on the boundary.
        """
        mod = utils.smart_numpy(x)
        return mod.any(mod.isclose(x, jnp.array([self.l, self.r], dtype=bst.environ.dftype())), axis=-1)

    def distance2boundary(self, x, dirn):
        """
        Compute the distance from points to the boundary in a specified direction.

        Args:
            x (array-like): The points to compute distance for.
            dirn (int): Direction indicator (-1 for left, 1 for right).

        Returns:
            array: Distances to the boundary.
        """
        return x - self.l if dirn < 0 else self.r - x

    def mindist2boundary(self, x):
        """
        Compute the minimum distance from points to the boundary.

        Args:
            x (array-like): The points to compute distance for.

        Returns:
            float: Minimum distance to the boundary.
        """
        mod = utils.smart_numpy(x)
        return min(mod.amin(x - self.l), mod.amin(self.r - x))

    def boundary_constraint_factor(
        self,
        x,
        smoothness: Literal["C0", "C0+", "Cinf"] = "C0+",
        where: Union[None, Literal["left", "right"]] = None,
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
                Default is "C0+".

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

            where (string, optional): A string to specify which part of the boundary to compute the distance,
                e.g., "left", "right". If `None`, compute the distance to the whole boundary. Default is `None`.

        Returns:
            A tensor of a type determined by the backend, which will have a shape of (n, 1).
            Each element in the tensor corresponds to the computed distance value for the respective point in `x`.
        """

        if where not in [None, "left"]:
            raise ValueError("where must be None or left")
        if smoothness not in ["C0", "C0+", "Cinf"]:
            raise ValueError("smoothness must be one of C0, C0+, Cinf")

        # To convert self.l and self.r to tensor,
        # and avoid repeated conversion in the loop
        if not hasattr(self, "self.l_tensor"):
            self.l_tensor = jnp.asarray(self.l)
            self.r_tensor = jnp.asarray(self.r)

        dist_l = dist_r = None
        if where != "right":
            dist_l = jnp.abs((x - self.l_tensor) / (self.r_tensor - self.l_tensor) * 2)
        if where != "left":
            dist_r = jnp.abs((x - self.r_tensor) / (self.r_tensor - self.l_tensor) * 2)

        if where is None:
            if smoothness == "C0":
                return jnp.minimum(dist_l, dist_r)
            if smoothness == "C0+":
                return dist_l * dist_r
            return jnp.square(dist_l * dist_r)
        if where == "left":
            if smoothness == "Cinf":
                dist_l = jnp.square(dist_l)
            return dist_l
        if smoothness == "Cinf":
            dist_r = jnp.square(dist_r)
        return dist_r

    def boundary_normal(self, x):
        """
        Compute the normal vector at boundary points.

        Args:
            x (array-like): The points to compute normal vectors for.

        Returns:
            array: Normal vectors at the given points.
        """
        mod = utils.smart_numpy(x)
        return -mod.isclose(x, self.l).astype(bst.environ.dftype()) + mod.isclose(x, self.r)

    def uniform_points(self, n, boundary=True):
        """
        Generate uniformly distributed points in the interval.

        Args:
            n (int): Number of points to generate.
            boundary (bool): Whether to include boundary points.

        Returns:
            array: Uniformly distributed points.
        """
        if boundary:
            return jnp.linspace(self.l, self.r, num=n, dtype=bst.environ.dftype())[:, None]
        return jnp.linspace(
            self.l, self.r, num=n + 1, endpoint=False, dtype=bst.environ.dftype()
        )[1:, None]

    def log_uniform_points(self, n, boundary=True):
        """
        Generate logarithmically uniformly distributed points in the interval.

        Args:
            n (int): Number of points to generate.
            boundary (bool): Whether to include boundary points.

        Returns:
            array: Logarithmically uniformly distributed points.
        """
        eps = 0 if self.l > 0 else jnp.finfo(bst.environ.dftype()).eps
        l = jnp.log(self.l + eps)
        r = jnp.log(self.r + eps)
        if boundary:
            x = jnp.linspace(l, r, num=n, dtype=bst.environ.dftype())[:, None]
        else:
            x = jnp.linspace(l, r, num=n + 1, endpoint=False, dtype=bst.environ.dftype())[
                1:, None
                ]
        return jnp.exp(x) - eps

    def random_points(self, n, random="pseudo"):
        """
        Generate random points in the interval.

        Args:
            n (int): Number of points to generate.
            random (str): Type of random number generation ("pseudo" or other).

        Returns:
            array: Randomly distributed points.
        """
        x = sample(n, 1, random)
        return (self.diam * x + self.l).astype(bst.environ.dftype())

    def uniform_boundary_points(self, n):
        """
        Generate uniformly distributed points on the boundary.

        Args:
            n (int): Number of points to generate.

        Returns:
            array: Uniformly distributed boundary points.
        """
        if n == 1:
            return jnp.array([[self.l]]).astype(bst.environ.dftype())
        xl = jnp.full((n // 2, 1), self.l).astype(bst.environ.dftype())
        xr = jnp.full((n - n // 2, 1), self.r).astype(bst.environ.dftype())
        return jnp.vstack((xl, xr))

    def random_boundary_points(self, n, random="pseudo"):
        """
        Generate random points on the boundary.

        Args:
            n (int): Number of points to generate.
            random (str): Type of random number generation ("pseudo" or other).

        Returns:
            array: Randomly distributed boundary points.
        """
        if n == 2:
            return jnp.array([[self.l], [self.r]]).astype(bst.environ.dftype())
        return bst.random.choice([self.l, self.r], n)[:, None].astype(bst.environ.dftype())

    def periodic_point(self, x, component=0):
        """
        Map points to their periodic equivalents within the interval.

        Args:
            x (array-like): Points to map.
            component (int): Component to apply periodicity to (unused in 1D).

        Returns:
            array: Mapped periodic points.
        """
        tmp = jnp.copy(x)
        tmp[utils.isclose(x, self.l)] = self.r
        tmp[utils.isclose(x, self.r)] = self.l
        return tmp

    def background_points(self, x, dirn, dist2npt, shift):
        """
        Generate background points based on a given point and direction.

        Args:
            x (array-like): Reference point.
            dirn (int): Direction (-1 for left, 1 for right, 0 for both).
            dist2npt (callable): Function to convert distance to number of points.
            shift (int): Number of points to shift.

        Returns:
            array: Generated background points.
        """

        def background_points_left():
            dx = x[0] - self.l
            n = max(dist2npt(dx), 1)
            h = dx / n
            pts = x[0] - jnp.arange(-shift, n - shift + 1, dtype=bst.environ.dftype()) * h
            return pts[:, None]

        def background_points_right():
            dx = self.r - x[0]
            n = max(dist2npt(dx), 1)
            h = dx / n
            pts = x[0] + jnp.arange(-shift, n - shift + 1, dtype=bst.environ.dftype()) * h
            return pts[:, None]

        return (
            background_points_left()
            if dirn < 0
            else background_points_right()
            if dirn > 0
            else jnp.vstack((background_points_left(), background_points_right()))
        )
