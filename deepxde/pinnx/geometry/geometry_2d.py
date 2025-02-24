# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================

__all__ = ["Disk", "Ellipse", "Polygon", "Rectangle", "StarShaped", "Triangle"]

from typing import Union, Literal

import brainstate as bst
import jax.numpy as jnp
from scipy import spatial

from deepxde.geometry.sampler import sample
from deepxde.pinnx import utils
from deepxde.utils.internal import vectorize
from .base import GeometryPINNx as Geometry
from .geometry_nd import Hypercube, Hypersphere
from ..utils import isclose


class Disk(Hypersphere):
    """
    A class representing a disk in 2D space, inheriting from Hypersphere.
    """

    def inside(self, x):
        """
        Check if points are inside the disk.

        Args:
            x (array-like): The coordinates of points to check.

        Returns:
            array-like: Boolean array indicating whether each point is inside the disk.
        """
        mod = utils.smart_numpy(x)
        return mod.linalg.norm(x - self.center, axis=-1) <= self.radius

    def on_boundary(self, x):
        """
        Check if points are on the boundary of the disk.

        Args:
            x (array-like): The coordinates of points to check.

        Returns:
            array-like: Boolean array indicating whether each point is on the disk's boundary.
        """
        mod = utils.smart_numpy(x)
        return mod.isclose(mod.linalg.norm(x - self.center, axis=-1), self.radius)

    def distance2boundary_unitdirn(self, x, dirn):
        """
        Calculate the distance from points to the disk boundary in a given unit direction.

        Args:
            x (array-like): The coordinates of points.
            dirn (array-like): The unit direction vector.

        Returns:
            array-like: Distances from points to the disk boundary in the given direction.
        """
        mod = utils.smart_numpy(x)
        xc = x - self.center
        ad = jnp.dot(xc, dirn)
        return (-ad + (ad ** 2 - mod.sum(xc * xc, axis=-1) + self._r2) ** 0.5).astype(bst.environ.dftype())

    def distance2boundary(self, x, dirn):
        """
        Calculate the distance from points to the disk boundary in a given direction.

        Args:
            x (array-like): The coordinates of points.
            dirn (array-like): The direction vector (not necessarily unit).

        Returns:
            array-like: Distances from points to the disk boundary in the given direction.
        """
        mod = utils.smart_numpy(x)
        return self.distance2boundary_unitdirn(x, dirn / mod.linalg.norm(dirn))

    def mindist2boundary(self, x):
        """
        Calculate the minimum distance from points to the disk boundary.

        Args:
            x (array-like): The coordinates of points.

        Returns:
            array-like: Minimum distances from points to the disk boundary.
        """
        mod = utils.smart_numpy(x)
        return mod.amin(self.radius - mod.linalg.norm(x - self.center, axis=1))

    def boundary_normal(self, x):
        """
        Calculate the unit normal vector to the disk boundary at given points.

        Args:
            x (array-like): The coordinates of points on the disk boundary.

        Returns:
            array-like: Unit normal vectors to the disk boundary at the given points.
        """
        mod = utils.smart_numpy(x)
        _n = x - self.center
        l = mod.linalg.norm(_n, axis=-1, keepdims=True)
        _n = _n / l * mod.isclose(l, self.radius)
        return _n

    def random_points(self, n, random="pseudo"):
        """
        Generate random points inside the disk.

        Args:
            n (int): Number of points to generate.
            random (str, optional): Method for generating random numbers. Defaults to "pseudo".

        Returns:
            array-like: Coordinates of randomly generated points inside the disk.
        """
        rng = sample(n, 2, random)
        r, theta = rng[:, 0], 2 * jnp.pi * rng[:, 1]
        x, y = jnp.cos(theta), jnp.sin(theta)
        return self.radius * (jnp.sqrt(r) * jnp.vstack((x, y))).T + self.center

    def uniform_boundary_points(self, n):
        """
        Generate uniformly distributed points on the disk boundary.

        Args:
            n (int): Number of points to generate.

        Returns:
            array-like: Coordinates of uniformly distributed points on the disk boundary.
        """
        theta = jnp.linspace(0, 2 * jnp.pi, num=n, endpoint=False)
        X = jnp.vstack((jnp.cos(theta), jnp.sin(theta))).T
        return self.radius * X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        """
        Generate random points on the disk boundary.

        Args:
            n (int): Number of points to generate.
            random (str, optional): Method for generating random numbers. Defaults to "pseudo".

        Returns:
            array-like: Coordinates of randomly generated points on the disk boundary.
        """
        u = sample(n, 1, random)
        theta = 2 * jnp.pi * u
        X = jnp.hstack((jnp.cos(theta), jnp.sin(theta)))
        return self.radius * X + self.center

    def background_points(self, x, dirn, dist2npt, shift):
        """
        Generate background points along a line passing through given points.

        Args:
            x (array-like): The coordinates of points.
            dirn (array-like): The direction vector.
            dist2npt (callable): Function to determine number of points based on distance.
            shift (float): Shift factor for point generation.

        Returns:
            array-like: Coordinates of generated background points.
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


class Ellipse(Geometry):
    """
    A class representing an ellipse in 2D space.

    This class inherits from the Geometry class and provides methods for working with ellipses,
    including generating points on the boundary and inside the ellipse, and computing distances.

    Args:
        center (array-like): The coordinates of the center of the ellipse.
        semimajor (float): The length of the semimajor axis of the ellipse.
        semiminor (float): The length of the semiminor axis of the ellipse.
        angle (float, optional): The rotation angle of the ellipse in radians. Defaults to 0.
            A positive angle rotates the ellipse clockwise about the center,
            while a negative angle rotates it counterclockwise.
    """

    def __init__(self, center, semimajor, semiminor, angle=0):
        self.center = jnp.array(center, dtype=bst.environ.dftype())
        self.semimajor = semimajor
        self.semiminor = semiminor
        self.angle = angle
        self.c = (semimajor ** 2 - semiminor ** 2) ** 0.5

        self.focus1 = jnp.array(
            [
                center[0] - self.c * jnp.cos(angle),
                center[1] + self.c * jnp.sin(angle),
            ],
            dtype=bst.environ.dftype(),
        )
        self.focus2 = jnp.array(
            [
                center[0] + self.c * jnp.cos(angle),
                center[1] - self.c * jnp.sin(angle),
            ],
            dtype=bst.environ.dftype(),
        )
        self.rotation_mat = jnp.array(
            [[jnp.cos(-angle), -jnp.sin(-angle)], [jnp.sin(-angle), jnp.cos(-angle)]]
        )
        (
            self.theta_from_arc_length,
            self.total_arc,
        ) = self._theta_from_arc_length_constructor()
        super().__init__(
            2, (self.center - semimajor, self.center + semiminor), 2 * self.c
        )

    def on_boundary(self, x):
        """
        Check if points are on the boundary of the ellipse.

        Args:
            x (array-like): The coordinates of points to check.

        Returns:
            array-like: Boolean array indicating whether each point is on the ellipse's boundary.
        """
        d1 = jnp.linalg.norm(x - self.focus1, axis=-1)
        d2 = jnp.linalg.norm(x - self.focus2, axis=-1)
        return isclose(d1 + d2, 2 * self.semimajor)

    def inside(self, x):
        """
        Check if points are inside the ellipse.

        Args:
            x (array-like): The coordinates of points to check.

        Returns:
            array-like: Boolean array indicating whether each point is inside the ellipse.
        """
        d1 = jnp.linalg.norm(x - self.focus1, axis=-1)
        d2 = jnp.linalg.norm(x - self.focus2, axis=-1)
        return d1 + d2 <= 2 * self.semimajor

    def _ellipse_arc(self):
        """
        Calculate the cumulative arc length of the ellipse.

        Returns:
            tuple: A tuple containing:
                - theta (array-like): Angle values.
                - cumulative_distance (array-like): Cumulative distance at each theta.
                - c (float): Total arc length of the ellipse.
        """
        # Divide the interval [0 , theta] into n steps at regular angles
        theta = jnp.linspace(0, 2 * jnp.pi, 10000)
        coords = jnp.array(
            [self.semimajor * jnp.cos(theta), self.semiminor * jnp.sin(theta)]
        )
        # Compute vector distance between each successive point
        coords_diffs = jnp.diff(coords)
        # Compute the full arc
        delta_r = jnp.linalg.norm(coords_diffs, axis=0)
        cumulative_distance = jnp.concatenate(([0], jnp.cumsum(delta_r)))
        c = jnp.sum(delta_r)
        return theta, cumulative_distance, c

    def _theta_from_arc_length_constructor(self):
        """
        Construct a function that returns the angle associated with a given cumulative arc length.

        Returns:
            tuple: A tuple containing:
                - f (callable): A function that takes an arc length and returns the corresponding angle.
                - total_arc (float): The total arc length of the ellipse.
        """
        theta, cumulative_distance, total_arc = self._ellipse_arc()

        # Construct the inverse arc length function
        def f(s):
            return jnp.interp(s, cumulative_distance, theta)

        return f, total_arc

    def random_points(self, n, random="pseudo"):
        """
        Generate random points inside the ellipse.

        Args:
            n (int): Number of points to generate.
            random (str, optional): Method for generating random numbers. Defaults to "pseudo".

        Returns:
            array-like: Coordinates of randomly generated points inside the ellipse.
        """
        # http://mathworld.wolfram.com/DiskPointPicking.html
        rng = sample(n, 2, random)
        r, theta = rng[:, 0], 2 * jnp.pi * rng[:, 1]
        x, y = self.semimajor * jnp.cos(theta), self.semiminor * jnp.sin(theta)
        X = jnp.sqrt(r) * jnp.vstack((x, y))
        return jnp.matmul(self.rotation_mat, X).T + self.center

    def uniform_boundary_points(self, n):
        """
        Generate uniformly distributed points on the ellipse boundary.

        Args:
            n (int): Number of points to generate.

        Returns:
            array-like: Coordinates of uniformly distributed points on the ellipse boundary.
        """
        # https://codereview.stackexchange.com/questions/243590/generate-random-points-on-perimeter-of-ellipse
        u = jnp.linspace(0, 1, num=n, endpoint=False).reshape((-1, 1))
        theta = self.theta_from_arc_length(u * self.total_arc)
        X = jnp.hstack((self.semimajor * jnp.cos(theta), self.semiminor * jnp.sin(theta)))
        return jnp.matmul(self.rotation_mat, X.T).T + self.center

    def random_boundary_points(self, n, random="pseudo"):
        """
        Generate random points on the ellipse boundary.

        Args:
            n (int): Number of points to generate.
            random (str, optional): Method for generating random numbers. Defaults to "pseudo".

        Returns:
            array-like: Coordinates of randomly generated points on the ellipse boundary.
        """
        u = sample(n, 1, random)
        theta = self.theta_from_arc_length(u * self.total_arc)
        X = jnp.hstack((self.semimajor * jnp.cos(theta), self.semiminor * jnp.sin(theta)))
        return jnp.matmul(self.rotation_mat, X.T).T + self.center

    def boundary_constraint_factor(
        self, x, smoothness: Literal["C0", "C0+", "Cinf"] = "C0+"
    ):
        """
        Compute the boundary constraint factor for given points.

        This function calculates a factor that represents how close points are to the ellipse boundary.
        The factor is zero on the boundary and positive elsewhere.

        Args:
            x (array-like): The coordinates of points to evaluate.
            smoothness (str, optional): The smoothness of the constraint factor. 
                Must be one of "C0", "C0+", or "Cinf". Defaults to "C0+".

        Returns:
            array-like: The computed boundary constraint factors for the input points.

        Raises:
            ValueError: If smoothness is not one of the allowed values.
        """
        if smoothness not in ["C0", "C0+", "Cinf"]:
            raise ValueError("`smoothness` must be one of C0, C0+, Cinf")

        if not hasattr(self, "self.focus1_tensor"):
            self.focus1_tensor = jnp.asarray(self.focus1)
            self.focus2_tensor = jnp.asarray(self.focus2)

        d1 = jnp.linalg.norm(x - self.focus1_tensor, axis=-1, keepdims=True)
        d2 = jnp.linalg.norm(x - self.focus2_tensor, axis=-1, keepdims=True)
        dist = d1 + d2 - 2 * self.semimajor

        if smoothness == "Cinf":
            dist = jnp.square(dist)
        else:
            dist = jnp.abs(dist)

        return dist


class Rectangle(Hypercube):
    """
    A class representing a rectangle in 2D space.

    This class inherits from the Hypercube class and provides methods for working with rectangles,
    including generating points on the boundary and inside the rectangle, and computing distances.

    Args:
        xmin: Coordinate of bottom left corner.
        xmax: Coordinate of top right corner.
    """

    def __init__(self, xmin, xmax):
        """
        Initialize a Rectangle object.

        Args:
            xmin (array-like): Coordinate of the bottom left corner.
            xmax (array-like): Coordinate of the top right corner.
        """
        super().__init__(xmin, xmax)
        self.perimeter = 2 * jnp.sum(self.xmax - self.xmin)
        self.area = jnp.prod(self.xmax - self.xmin)

    def uniform_boundary_points(self, n):
        """
        Generate uniformly distributed points on the rectangle boundary.

        Args:
            n (int): Number of points to generate.

        Returns:
            array-like: Coordinates of uniformly distributed points on the rectangle boundary.
        """
        nx, ny = jnp.ceil(n / self.perimeter * (self.xmax - self.xmin)).astype(int)
        xbot = jnp.hstack(
            (
                jnp.linspace(self.xmin[0], self.xmax[0], num=nx, endpoint=False)[
                :, None
                ],
                jnp.full([nx, 1], self.xmin[1]),
            )
        )
        yrig = jnp.hstack(
            (
                jnp.full([ny, 1], self.xmax[0]),
                jnp.linspace(self.xmin[1], self.xmax[1], num=ny, endpoint=False)[
                :, None
                ],
            )
        )
        xtop = jnp.hstack(
            (
                jnp.linspace(self.xmin[0], self.xmax[0], num=nx + 1)[1:, None],
                jnp.full([nx, 1], self.xmax[1]),
            )
        )
        ylef = jnp.hstack(
            (
                jnp.full([ny, 1], self.xmin[0]),
                jnp.linspace(self.xmin[1], self.xmax[1], num=ny + 1)[1:, None],
            )
        )
        x = jnp.vstack((xbot, yrig, xtop, ylef))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_boundary_points(self, n, random="pseudo"):
        """
        Generate random points on the rectangle boundary.

        Args:
            n (int): Number of points to generate.
            random (str, optional): Method for generating random numbers. Defaults to "pseudo".

        Returns:
            array-like: Coordinates of randomly generated points on the rectangle boundary.
        """
        l1 = self.xmax[0] - self.xmin[0]
        l2 = l1 + self.xmax[1] - self.xmin[1]
        l3 = l2 + l1
        u = jnp.ravel(sample(n + 2, 1, random))
        # Remove the possible points very close to the corners
        u = u[jnp.logical_not(isclose(u, l1 / self.perimeter))]
        u = u[jnp.logical_not(isclose(u, l3 / self.perimeter))]
        u = u[:n]

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
        return jnp.vstack(x)

    @staticmethod
    def is_valid(vertices):
        """
        Check if the geometry is a valid Rectangle.

        Args:
            vertices (array-like): An array of 4 vertices defining the rectangle.

        Returns:
            bool: True if the geometry is a valid rectangle, False otherwise.
        """
        return (
            len(vertices) == 4
            and isclose(jnp.prod(vertices[1] - vertices[0]), 0)
            and isclose(jnp.prod(vertices[2] - vertices[1]), 0)
            and isclose(jnp.prod(vertices[3] - vertices[2]), 0)
            and isclose(jnp.prod(vertices[0] - vertices[3]), 0)
        )


class StarShaped(Geometry):
    """Star-shaped 2d domain, i.e., a geometry whose boundary is parametrized in polar coordinates as:

    $$
    r(theta) := r_0 + sum_{i = 1}^N [a_i cos( i theta) + b_i sin(i theta) ],  theta in [0,2 pi].
    $$

    For more details, refer to:
    `Hiptmair et al. Large deformation shape uncertainty quantification in acoustic
    scattering. Adv Comp Math, 2018.
    <https://link.springer.com/article/10.1007/s10444-018-9594-8>`_

    Args:
        center: Center of the domain.
        radius: 0th-order term of the parametrization (r_0).
        coeffs_cos: i-th order coefficients for the i-th cos term (a_i).
        coeffs_sin: i-th order coefficients for the i-th sin term (b_i).
    """

    def __init__(self, center, radius, coeffs_cos, coeffs_sin):
        self.center = jnp.array(center, dtype=bst.environ.dftype())
        self.radius = radius
        self.coeffs_cos = coeffs_cos
        self.coeffs_sin = coeffs_sin
        max_radius = radius + jnp.sum(coeffs_cos) + jnp.sum(coeffs_sin)
        super().__init__(
            2,
            (self.center - max_radius, self.center + max_radius),
            2 * max_radius,
        )

    def _r_theta(self, theta):
        """Define the parametrization r(theta) at angles theta."""
        result = self.radius * jnp.ones(theta.shape)
        for i, (coeff_cos, coeff_sin) in enumerate(
            zip(self.coeffs_cos, self.coeffs_sin), start=1
        ):
            result += coeff_cos * jnp.cos(i * theta) + coeff_sin * jnp.sin(i * theta)
        return result

    def _dr_theta(self, theta):
        """Evalutate the polar derivative r'(theta) at angles theta"""
        result = jnp.zeros(theta.shape)
        for i, (coeff_cos, coeff_sin) in enumerate(
            zip(self.coeffs_cos, self.coeffs_sin), start=1
        ):
            result += -coeff_cos * i * jnp.sin(i * theta) + coeff_sin * i * jnp.cos(
                i * theta
            )
        return result

    def inside(self, x):
        r, theta = polar(x - self.center)
        r_theta = self._r_theta(theta)
        return r_theta >= r

    def on_boundary(self, x):
        r, theta = polar(x - self.center)
        r_theta = self._r_theta(theta)
        return isclose(jnp.linalg.norm(r_theta - r), 0)

    def boundary_normal(self, x):
        _, theta = polar(x - self.center)
        dr_theta = self._dr_theta(theta)
        r_theta = self._r_theta(theta)

        dxt = jnp.vstack(
            (
                dr_theta * jnp.cos(theta) - r_theta * jnp.sin(theta),
                dr_theta * jnp.sin(theta) + r_theta * jnp.cos(theta),
            )
        ).T
        norm = jnp.linalg.norm(dxt, axis=-1, keepdims=True)
        dxt /= norm
        return jnp.array([dxt[:, 1], -dxt[:, 0]]).T

    def random_points(self, n, random="pseudo"):
        x = jnp.empty((0, 2), dtype=bst.environ.dftype())
        vbbox = self.bbox[1] - self.bbox[0]
        while len(x) < n:
            x_new = sample(n, 2, sampler="pseudo") * vbbox + self.bbox[0]
            x = jnp.vstack((x, x_new[self.inside(x_new)]))
        return x[:n]

    def uniform_boundary_points(self, n):
        theta = jnp.linspace(0, 2 * jnp.pi, num=n, endpoint=False)
        r_theta = self._r_theta(theta)
        X = jnp.vstack((r_theta * jnp.cos(theta), r_theta * jnp.sin(theta))).T
        return X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        u = sample(n, 1, random)
        theta = 2 * jnp.pi * u
        r_theta = self._r_theta(theta)
        X = jnp.hstack((r_theta * jnp.cos(theta), r_theta * jnp.sin(theta)))
        return X + self.center


class Triangle(Geometry):
    """Triangle.

    The order of vertices can be in a clockwise or counterclockwise direction. The
    vertices will be re-ordered in counterclockwise (right hand rule).
    """

    def __init__(self, x1, x2, x3):
        self.area = polygon_signed_area([x1, x2, x3])
        # Clockwise
        if self.area < 0:
            self.area = -self.area
            x2, x3 = x3, x2

        self.x1 = jnp.array(x1, dtype=bst.environ.dftype())
        self.x2 = jnp.array(x2, dtype=bst.environ.dftype())
        self.x3 = jnp.array(x3, dtype=bst.environ.dftype())

        self.v12 = self.x2 - self.x1
        self.v23 = self.x3 - self.x2
        self.v31 = self.x1 - self.x3
        self.l12 = jnp.linalg.norm(self.v12)
        self.l23 = jnp.linalg.norm(self.v23)
        self.l31 = jnp.linalg.norm(self.v31)
        self.n12 = self.v12 / self.l12
        self.n23 = self.v23 / self.l23
        self.n31 = self.v31 / self.l31
        self.n12_normal = clockwise_rotation_90(self.n12)
        self.n23_normal = clockwise_rotation_90(self.n23)
        self.n31_normal = clockwise_rotation_90(self.n31)
        self.perimeter = self.l12 + self.l23 + self.l31

        super().__init__(
            2,
            (jnp.minimum(x1, jnp.minimum(x2, x3)), jnp.maximum(x1, jnp.maximum(x2, x3))),
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

    def inside(self, x):
        # https://stackoverflow.com/a/2049593/12679294
        _sign = jnp.hstack(
            [
                jnp.cross(self.v12, x - self.x1)[:, jnp.newaxis],
                jnp.cross(self.v23, x - self.x2)[:, jnp.newaxis],
                jnp.cross(self.v31, x - self.x3)[:, jnp.newaxis],
            ]
        )
        return ~jnp.logical_and(jnp.any(_sign > 0, axis=-1), jnp.any(_sign < 0, axis=-1))

    def on_boundary(self, x):
        l1 = jnp.linalg.norm(x - self.x1, axis=-1)
        l2 = jnp.linalg.norm(x - self.x2, axis=-1)
        l3 = jnp.linalg.norm(x - self.x3, axis=-1)
        return jnp.any(
            isclose(
                [l1 + l2 - self.l12, l2 + l3 - self.l23, l3 + l1 - self.l31],
                0,
            ),
            axis=0,
        )

    def boundary_normal(self, x):
        l1 = jnp.linalg.norm(x - self.x1, axis=-1, keepdims=True)
        l2 = jnp.linalg.norm(x - self.x2, axis=-1, keepdims=True)
        l3 = jnp.linalg.norm(x - self.x3, axis=-1, keepdims=True)
        on12 = isclose(l1 + l2, self.l12)
        on23 = isclose(l2 + l3, self.l23)
        on31 = isclose(l3 + l1, self.l31)
        # Check points on the vertexes
        if jnp.any(jnp.count_nonzero(jnp.hstack([on12, on23, on31]), axis=-1) > 1):
            raise ValueError(
                "{}.boundary_normal do not accept points on the vertexes.".format(
                    self.__class__.__name__
                )
            )
        return self.n12_normal * on12 + self.n23_normal * on23 + self.n31_normal * on31

    def random_points(self, n, random="pseudo"):
        # There are two methods for triangle point picking.
        # Method 1 (used here):
        # - https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
        # Method 2:
        # - http://mathworld.wolfram.com/TrianglePointPicking.html
        # - https://hbfs.wordpress.com/2010/10/05/random-points-in-a-triangle-generating-random-sequences-ii/
        # - https://stackoverflow.com/questions/19654251/random-point-inside-triangle-inside-java
        sqrt_r1 = jnp.sqrt(bst.random.rand(n, 1))
        r2 = bst.random.rand(n, 1)
        return (
            (1 - sqrt_r1) * self.x1
            + sqrt_r1 * (1 - r2) * self.x2
            + r2 * sqrt_r1 * self.x3
        )

    def uniform_boundary_points(self, n):
        density = n / self.perimeter
        x12 = (
            jnp.linspace(0, 1, num=int(jnp.ceil(density * self.l12)), endpoint=False)[
            :, None
            ]
            * self.v12
            + self.x1
        )
        x23 = (
            jnp.linspace(0, 1, num=int(jnp.ceil(density * self.l23)), endpoint=False)[
            :, None
            ]
            * self.v23
            + self.x2
        )
        x31 = (
            jnp.linspace(0, 1, num=int(jnp.ceil(density * self.l31)), endpoint=False)[
            :, None
            ]
            * self.v31
            + self.x3
        )
        x = jnp.vstack((x12, x23, x31))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_boundary_points(self, n, random="pseudo"):
        u = jnp.ravel(sample(n + 2, 1, random))
        # Remove the possible points very close to the corners
        u = u[jnp.logical_not(isclose(u, self.l12 / self.perimeter))]
        u = u[jnp.logical_not(isclose(u, (self.l12 + self.l23) / self.perimeter))]
        u = u[:n]

        u *= self.perimeter
        x = []
        for l in u:
            if l < self.l12:
                x.append(l * self.n12 + self.x1)
            elif l < self.l12 + self.l23:
                x.append((l - self.l12) * self.n23 + self.x2)
            else:
                x.append((l - self.l12 - self.l23) * self.n31 + self.x3)
        return jnp.vstack(x)

    def boundary_constraint_factor(
        self,
        x,
        smoothness: Literal["C0", "C0+", "Cinf"] = "C0+",
        where: Union[None, Literal["x1-x2", "x1-x3", "x2-x3"]] = None,
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

            where (string, optional): A string to specify which part of the boundary to compute the distance.
                If `None`, compute the distance to the whole boundary. 
                "x1-x2" indicates the line segment with vertices x1 and x2 (after reordered). Default is `None`.

        Returns:
            A tensor of a type determined by the backend, which will have a shape of (n, 1).
            Each element in the tensor corresponds to the computed distance value for the respective point in `x`.
        """

        if where not in [None, "x1-x2", "x1-x3", "x2-x3"]:
            raise ValueError("where must be one of None, x1-x2, x1-x3, x2-x3")
        if smoothness not in ["C0", "C0+", "Cinf"]:
            raise ValueError("smoothness must be one of C0, C0+, Cinf")

        if not hasattr(self, "self.x1_tensor"):
            self.x1_tensor = jnp.asarray(self.x1)
            self.x2_tensor = jnp.asarray(self.x2)
            self.x3_tensor = jnp.asarray(self.x3)

        diff_x1_x2 = diff_x1_x3 = diff_x2_x3 = None
        if where not in ["x1-x3", "x2-x3"]:
            diff_x1_x2 = (
                jnp.linalg.norm(x - self.x1_tensor, axis=-1, keepdims=True)
                + jnp.linalg.norm(x - self.x2_tensor, axis=-1, keepdims=True)
                - self.l12
            )
        if where not in ["x1-x2", "x2-x3"]:
            diff_x1_x3 = (
                jnp.linalg.norm(x - self.x1_tensor, axis=-1, keepdims=True)
                + jnp.linalg.norm(x - self.x3_tensor, axis=-1, keepdims=True)
                - self.l31
            )
        if where not in ["x1-x2", "x1-x3"]:
            diff_x2_x3 = (
                jnp.linalg.norm(x - self.x2_tensor, axis=-1, keepdims=True)
                + jnp.linalg.norm(x - self.x3_tensor, axis=-1, keepdims=True)
                - self.l23
            )

        if where is None:
            if smoothness == "C0":
                return jnp.minimum(jnp.minimum(diff_x1_x2, diff_x1_x3), diff_x2_x3)
            return diff_x1_x2 * diff_x1_x3 * diff_x2_x3
        if where == "x1-x2":
            return diff_x1_x2
        if where == "x1-x3":
            return diff_x1_x3
        return diff_x2_x3


class Polygon(Geometry):
    """
    Represents a simple polygon geometry.

    This class creates a polygon object from a set of vertices. The vertices can be provided
    in either clockwise or counterclockwise order, and will be reordered to counterclockwise
    (right-hand rule) if necessary.

    Args:
        vertices (list or array-like): A sequence of (x, y) coordinates defining the vertices
            of the polygon. The order can be clockwise or counterclockwise.

    Raises:
        ValueError: If the polygon is a triangle (use Triangle class instead) or
            if the polygon is a rectangle (use Rectangle class instead).

    Attributes:
        vertices (jnp.ndarray): Array of vertex coordinates.
        area (float): Signed area of the polygon.
        diagonals (jnp.ndarray): Square matrix of distances between vertices.
        nvertices (int): Number of vertices in the polygon.
        perimeter (float): Perimeter of the polygon.
        bbox (jnp.ndarray): Bounding box of the polygon.
        segments (jnp.ndarray): Vectors representing the edges of the polygon.
        normal (jnp.ndarray): Normal vectors for each edge of the polygon.

    Note:
        This class inherits from the Geometry base class and implements several
        methods for working with polygons, including checking if points are inside
        or on the boundary, and generating random points within or on the boundary
        of the polygon.
    """

    def __init__(self, vertices):
        self.vertices = jnp.array(vertices, dtype=bst.environ.dftype())
        if len(vertices) == 3:
            raise ValueError("The polygon is a triangle. Use Triangle instead.")
        if Rectangle.is_valid(self.vertices):
            raise ValueError("The polygon is a rectangle. Use Rectangle instead.")

        self.area = polygon_signed_area(self.vertices)
        # Clockwise
        if self.area < 0:
            self.area = -self.area
            self.vertices = jnp.flipud(self.vertices)

        self.diagonals = spatial.distance.squareform(
            spatial.distance.pdist(self.vertices)
        )
        super().__init__(
            2,
            (jnp.amin(self.vertices, axis=0), jnp.amax(self.vertices, axis=0)),
            jnp.max(self.diagonals),
        )
        self.nvertices = len(self.vertices)
        self.perimeter = jnp.sum(
            [self.diagonals[i, i + 1] for i in range(-1, self.nvertices - 1)]
        )
        self.bbox = jnp.array(
            [jnp.min(self.vertices, axis=0), jnp.max(self.vertices, axis=0)]
        )

        self.segments = self.vertices[1:] - self.vertices[:-1]
        self.segments = jnp.vstack((self.vertices[0] - self.vertices[-1], self.segments))
        self.normal = clockwise_rotation_90(self.segments.T).T
        self.normal = self.normal / jnp.linalg.norm(self.normal, axis=1).reshape(-1, 1)

    def inside(self, x):
        def wn_PnPoly(P, V):
            """Winding number algorithm.

            https://en.wikipedia.org/wiki/Point_in_polygon
            http://geomalgorithms.com/a03-_inclusion.html

            Args:
                P: A point.
                V: Vertex points of a polygon.

            Returns:
                wn: Winding number (=0 only if P is outside polygon).
            """
            wn = jnp.zeros(len(P))  # Winding number counter

            # Repeat the first vertex at end
            # Loop through all edges of the polygon
            for i in range(-1, self.nvertices - 1):  # Edge from V[i] to V[i+1]
                tmp = jnp.all(
                    jnp.hstack(
                        [
                            V[i, 1] <= P[:, 1:2],  # Start y <= P[1]
                            V[i + 1, 1] > P[:, 1:2],  # An upward crossing
                            is_left(V[i], V[i + 1], P) > 0,  # P left of edge
                        ]
                    ),
                    axis=-1,
                )
                wn[tmp] += 1  # Have a valid up intersect
                tmp = jnp.all(
                    jnp.hstack(
                        [
                            V[i, 1] > P[:, 1:2],  # Start y > P[1]
                            V[i + 1, 1] <= P[:, 1:2],  # A downward crossing
                            is_left(V[i], V[i + 1], P) < 0,  # P right of edge
                        ]
                    ),
                    axis=-1,
                )
                wn[tmp] -= 1  # Have a valid down intersect
            return wn

        return wn_PnPoly(x, self.vertices) != 0

    def on_boundary(self, x):
        _on = jnp.zeros(shape=len(x), dtype=int)
        for i in range(-1, self.nvertices - 1):
            l1 = jnp.linalg.norm(self.vertices[i] - x, axis=-1)
            l2 = jnp.linalg.norm(self.vertices[i + 1] - x, axis=-1)
            _on[isclose(l1 + l2, self.diagonals[i, i + 1])] += 1
        return _on > 0

    @vectorize(excluded=[0], signature="(n)->(n)")
    def boundary_normal(self, x):
        for i in range(self.nvertices):
            if is_on_line_segment(self.vertices[i - 1], self.vertices[i], x):
                return self.normal[i]
        return jnp.array([0, 0])

    def random_points(self, n, random="pseudo"):
        x = jnp.empty((0, 2), dtype=bst.environ.dftype())
        vbbox = self.bbox[1] - self.bbox[0]
        while len(x) < n:
            x_new = sample(n, 2, sampler="pseudo") * vbbox + self.bbox[0]
            x = jnp.vstack((x, x_new[self.inside(x_new)]))
        return x[:n]

    def uniform_boundary_points(self, n):
        density = n / self.perimeter
        x = []
        for i in range(-1, self.nvertices - 1):
            x.append(
                jnp.linspace(
                    0,
                    1,
                    num=int(jnp.ceil(density * self.diagonals[i, i + 1])),
                    endpoint=False,
                )[:, None]
                * (self.vertices[i + 1] - self.vertices[i])
                + self.vertices[i]
            )
        x = jnp.vstack(x)
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_boundary_points(self, n, random="pseudo"):
        u = jnp.ravel(sample(n + self.nvertices, 1, random))
        # Remove the possible points very close to the corners
        l = 0
        for i in range(0, self.nvertices - 1):
            l += self.diagonals[i, i + 1]
            u = u[jnp.logical_not(isclose(u, l / self.perimeter))]
        u = u[:n]
        u *= self.perimeter
        u.sort()

        x = []
        i = -1
        l0 = 0
        l1 = l0 + self.diagonals[i, i + 1]
        v = (self.vertices[i + 1] - self.vertices[i]) / self.diagonals[i, i + 1]
        for l in u:
            if l > l1:
                i += 1
                l0, l1 = l1, l1 + self.diagonals[i, i + 1]
                v = (self.vertices[i + 1] - self.vertices[i]) / self.diagonals[i, i + 1]
            x.append((l - l0) * v + self.vertices[i])
        return jnp.vstack(x)


def polygon_signed_area(vertices):
    """The (signed) area of a simple polygon.

    If the vertices are in the counterclockwise direction, then the area is positive; if
    they are in the clockwise direction, the area is negative.

    Shoelace formula: https://en.wikipedia.org/wiki/Shoelace_formula
    """
    x, y = zip(*vertices)
    x = jnp.array(list(x) + [x[0]])
    y = jnp.array(list(y) + [y[0]])
    return 0.5 * (jnp.sum(x[:-1] * y[1:]) - jnp.sum(x[1:] * y[:-1]))


def clockwise_rotation_90(v):
    """Rotate a vector of 90 degrees clockwise about the origin."""
    return jnp.array([v[1], -v[0]])


def is_left(P0, P1, P2):
    """Test if a point is Left|On|Right of an infinite line.

    See: the January 2001 Algorithm "Area of 2D and 3D Triangles and Polygons".

    Args:
        P0: One point in the line.
        P1: One point in the line.
        P2: A array of point to be tested.

    Returns:
        >0 if P2 left of the line through P0 and P1, =0 if P2 on the line, <0 if P2
        right of the line.
    """
    return jnp.cross(P1 - P0, P2 - P0, axis=-1).reshape((-1, 1))


def is_rectangle(vertices):
    """Check if the geometry is a rectangle.

    https://stackoverflow.com/questions/2303278/find-if-4-points-on-a-plane-form-a-rectangle/2304031

    1. Find the center of mass of corner points: cx=(x1+x2+x3+x4)/4, cy=(y1+y2+y3+y4)/4
    2. Test if square of distances from center of mass to all 4 corners are equal
    """
    if len(vertices) != 4:
        return False

    c = jnp.mean(vertices, axis=0)
    d = jnp.sum((vertices - c) ** 2, axis=1)
    return jnp.allclose(d, jnp.full(4, d[0]))


def is_on_line_segment(P0, P1, P2):
    """Test if a point is between two other points on a line segment.

    Args:
        P0: One point in the line.
        P1: One point in the line.
        P2: The point to be tested.

    References:
        https://stackoverflow.com/questions/328107
    """
    v01 = P1 - P0
    v02 = P2 - P0
    v12 = P2 - P1
    return (
        # check that P2 is almost on the line P0 P1
        isclose(jnp.cross(v01, v02) / jnp.linalg.norm(v01), 0)
        # check that projection of P2 to line is between P0 and P1
        and v01 @ v02 >= 0
        and v01 @ v12 <= 0
    )
    # Not between P0 and P1, but close to P0 or P1
    # or isclose(np.linalg.norm(v02), 0)  # check whether P2 is close to P0
    # or isclose(np.linalg.norm(v12), 0)  # check whether P2 is close to P1


def polar(x):
    """Get the polar coordinated for a 2d vector in cartesian coordinates."""
    r = jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    theta = jnp.arctan2(x[:, 1], x[:, 0])
    return r, theta
