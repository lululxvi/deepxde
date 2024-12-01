__all__ = ["Disk", "Ellipse", "Polygon", "Rectangle", "StarShaped", "Triangle"]

from typing import Union, Literal

import numpy as np
from scipy import spatial

from .geometry import Geometry
from .geometry_nd import Hypercube, Hypersphere
from .sampler import sample
from .. import backend as bkd
from .. import config
from ..utils import isclose, vectorize


class Disk(Hypersphere):
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
        return np.amin(self.radius - np.linalg.norm(x - self.center, axis=1))

    def boundary_normal(self, x):
        _n = x - self.center
        l = np.linalg.norm(_n, axis=-1, keepdims=True)
        _n = _n / l * isclose(l, self.radius)
        return _n

    def random_points(self, n, random="pseudo"):
        # http://mathworld.wolfram.com/DiskPointPicking.html
        rng = sample(n, 2, random)
        r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
        x, y = np.cos(theta), np.sin(theta)
        return self.radius * (np.sqrt(r) * np.vstack((x, y))).T + self.center

    def uniform_boundary_points(self, n):
        theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        X = np.vstack((np.cos(theta), np.sin(theta))).T
        return self.radius * X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        u = sample(n, 1, random)
        theta = 2 * np.pi * u
        X = np.hstack((np.cos(theta), np.sin(theta)))
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


class Ellipse(Geometry):
    """Ellipse.

    Args:
        center: Center of the ellipse.
        semimajor: Semimajor of the ellipse.
        semiminor: Semiminor of the ellipse.
        angle: Rotation angle of the ellipse. A positive angle rotates the ellipse
            clockwise about the center and a negative angle rotates the ellipse
            counterclockwise about the center.
    """

    def __init__(self, center, semimajor, semiminor, angle=0):
        self.center = np.array(center, dtype=config.real(np))
        self.semimajor = semimajor
        self.semiminor = semiminor
        self.angle = angle
        self.c = (semimajor**2 - semiminor**2) ** 0.5

        self.focus1 = np.array(
            [
                center[0] - self.c * np.cos(angle),
                center[1] + self.c * np.sin(angle),
            ],
            dtype=config.real(np),
        )
        self.focus2 = np.array(
            [
                center[0] + self.c * np.cos(angle),
                center[1] - self.c * np.sin(angle),
            ],
            dtype=config.real(np),
        )
        self.rotation_mat = np.array(
            [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
        )
        (
            self.theta_from_arc_length,
            self.total_arc,
        ) = self._theta_from_arc_length_constructor()
        super().__init__(
            2, (self.center - semimajor, self.center + semiminor), 2 * self.c
        )

    def on_boundary(self, x):
        d1 = np.linalg.norm(x - self.focus1, axis=-1)
        d2 = np.linalg.norm(x - self.focus2, axis=-1)
        return isclose(d1 + d2, 2 * self.semimajor)

    def inside(self, x):
        d1 = np.linalg.norm(x - self.focus1, axis=-1)
        d2 = np.linalg.norm(x - self.focus2, axis=-1)
        return d1 + d2 <= 2 * self.semimajor

    def _ellipse_arc(self):
        """Cumulative arc length of ellipse with given dimensions. Returns theta values,
        distance cumulated at each theta, and total arc length.
        """
        # Divide the interval [0 , theta] into n steps at regular angles
        theta = np.linspace(0, 2 * np.pi, 10000)
        coords = np.array(
            [self.semimajor * np.cos(theta), self.semiminor * np.sin(theta)]
        )
        # Compute vector distance between each successive point
        coords_diffs = np.diff(coords)
        # Compute the full arc
        delta_r = np.linalg.norm(coords_diffs, axis=0)
        cumulative_distance = np.concatenate(([0], np.cumsum(delta_r)))
        c = np.sum(delta_r)
        return theta, cumulative_distance, c

    def _theta_from_arc_length_constructor(self):
        """Constructs a function that returns the angle associated with a given
        cumulative arc length for given ellipse.
        """
        theta, cumulative_distance, total_arc = self._ellipse_arc()

        # Construct the inverse arc length function
        def f(s):
            return np.interp(s, cumulative_distance, theta)

        return f, total_arc

    def random_points(self, n, random="pseudo"):
        # http://mathworld.wolfram.com/DiskPointPicking.html
        rng = sample(n, 2, random)
        r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
        x, y = self.semimajor * np.cos(theta), self.semiminor * np.sin(theta)
        X = np.sqrt(r) * np.vstack((x, y))
        return np.matmul(self.rotation_mat, X).T + self.center

    def uniform_boundary_points(self, n):
        # https://codereview.stackexchange.com/questions/243590/generate-random-points-on-perimeter-of-ellipse
        u = np.linspace(0, 1, num=n, endpoint=False).reshape((-1, 1))
        theta = self.theta_from_arc_length(u * self.total_arc)
        X = np.hstack((self.semimajor * np.cos(theta), self.semiminor * np.sin(theta)))
        return np.matmul(self.rotation_mat, X.T).T + self.center

    def random_boundary_points(self, n, random="pseudo"):
        u = sample(n, 1, random)
        theta = self.theta_from_arc_length(u * self.total_arc)
        X = np.hstack((self.semimajor * np.cos(theta), self.semiminor * np.sin(theta)))
        return np.matmul(self.rotation_mat, X.T).T + self.center

    def boundary_constraint_factor(
        self, x, smoothness: Literal["C0", "C0+", "Cinf"] = "C0+"
    ):
        if smoothness not in ["C0", "C0+", "Cinf"]:
            raise ValueError("`smoothness` must be one of C0, C0+, Cinf")

        if not hasattr(self, "self.focus1_tensor"):
            self.focus1_tensor = bkd.as_tensor(self.focus1)
            self.focus2_tensor = bkd.as_tensor(self.focus2)

        d1 = bkd.norm(x - self.focus1_tensor, axis=-1, keepdims=True)
        d2 = bkd.norm(x - self.focus2_tensor, axis=-1, keepdims=True)
        dist = d1 + d2 - 2 * self.semimajor

        if smoothness == "Cinf":
            dist = bkd.square(dist)
        else:
            dist = bkd.abs(dist)

        return dist


class Rectangle(Hypercube):
    """
    Args:
        xmin: Coordinate of bottom left corner.
        xmax: Coordinate of top right corner.
    """

    def __init__(self, xmin, xmax):
        super().__init__(xmin, xmax)
        self.perimeter = 2 * np.sum(self.xmax - self.xmin)
        self.area = np.prod(self.xmax - self.xmin)

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
        l1 = self.xmax[0] - self.xmin[0]
        l2 = l1 + self.xmax[1] - self.xmin[1]
        l3 = l2 + l1
        u = np.ravel(sample(n + 2, 1, random))
        # Remove the possible points very close to the corners
        u = u[np.logical_not(isclose(u, l1 / self.perimeter))]
        u = u[np.logical_not(isclose(u, l3 / self.perimeter))]
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
        return np.vstack(x)

    def _boundary_constraint_factor_inside(
        self,
        x,
        where: Union[None, Literal["left", "right", "bottom", "top"]] = None,
        smoothness: Literal["C0", "C0+", "Cinf"] = "C0+",
    ):
        """(Internal use only) Compute the hard constraint factor at `x` for the boundary.
        The points in `x` are assumed to live inside the geometry.

        This function is a helper function used internally by the `boundary_constraint_factor` function.
        It should not be called directly in most cases.
        """

        if not hasattr(self, "self.xmin_tensor"):
            self.xmin_tensor = bkd.as_tensor(self.xmin)
            self.xmax_tensor = bkd.as_tensor(self.xmax)
        if where not in ["right", "top"]:
            dist_l = bkd.abs(
                (x - self.xmin_tensor) / (self.xmax_tensor - self.xmin_tensor) * 2
            )
        if where not in ["left", "bottom"]:
            dist_r = bkd.abs(
                (x - self.xmax_tensor) / (self.xmax_tensor - self.xmin_tensor) * 2
            )

        if where == "left":
            return dist_l[:, 0:1]
        if where == "right":
            return dist_r[:, 0:1]
        if where == "bottom":
            return dist_l[:, 1:]
        if where == "top":
            return dist_r[:, 1:]

        if smoothness == "C0":
            dist_l = bkd.min(dist_l, dim=-1, keepdims=True)
            dist_r = bkd.min(dist_r, dim=-1, keepdims=True)
            return bkd.minimum(dist_l, dist_r)
        dist_l = bkd.prod(dist_l, dim=-1, keepdims=True)
        dist_r = bkd.prod(dist_r, dim=-1, keepdims=True)
        return dist_l * dist_r

    def boundary_constraint_factor(
        self,
        x,
        smoothness: Literal["C0", "C0+", "Cinf"] = "C0+",
        where: Union[None, Literal["left", "right", "bottom", "top"]] = None,
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
                "left": x[0] = xmin[0], "right": x[0] = xmax[0], "bottom": x[1] = xmin[1], "top": x[1] = xmax[1]. 
                If `None`, compute the distance to the whole boundary. Default is `None`.
            inside (bool, optional): The `x` is either inside or outside the geometry.
                The cases where there are both points inside and points
                outside the geometry are NOT allowed. Default is `True`.

        Returns:
            A tensor of a type determined by the backend, which will have a shape of (n, 1).
            Each element in the tensor corresponds to the computed distance value for the respective point in `x`.
        """
        if where not in [None, "left", "right", "bottom", "top"]:
            raise ValueError("where must be one of None, left, right, bottom, top")
        if smoothness not in ["C0", "C0+", "Cinf"]:
            raise ValueError("smoothness must be one of C0, C0+, Cinf")
        if self.dim != 2:
            raise ValueError("self.dim must be 2")

        if inside:
            return self._boundary_constraint_factor_inside(x, where, smoothness)

        if not hasattr(self, "self.x11_tensor"):
            self.x11_tensor = bkd.as_tensor(self.xmin)
            self.x22_tensor = bkd.as_tensor(self.xmax)
            self.x12_tensor = bkd.as_tensor([self.xmin[0], self.xmax[1]])
            self.x21_tensor = bkd.as_tensor([self.xmax[0], self.xmin[1]])

        dist_left = dist_right = dist_bottom = dist_top = None
        if where is None or where == "left":
            dist_left = bkd.abs(
                bkd.norm(x - self.x11_tensor, axis=-1, keepdims=True)
                + bkd.norm(x - self.x12_tensor, axis=-1, keepdims=True)
                - (self.xmax[1] - self.xmin[1])
            )
        if where is None or where == "right":
            dist_right = bkd.abs(
                bkd.norm(x - self.x21_tensor, axis=-1, keepdims=True)
                + bkd.norm(x - self.x22_tensor, axis=-1, keepdims=True)
                - (self.xmax[1] - self.xmin[1])
            )
        if where is None or where == "bottom":
            dist_bottom = bkd.abs(
                bkd.norm(x - self.x11_tensor, axis=-1, keepdims=True)
                + bkd.norm(x - self.x21_tensor, axis=-1, keepdims=True)
                - (self.xmax[0] - self.xmin[0])
            )
        if where is None or where == "top":
            dist_top = bkd.abs(
                bkd.norm(x - self.x12_tensor, axis=-1, keepdims=True)
                + bkd.norm(x - self.x22_tensor, axis=-1, keepdims=True)
                - (self.xmax[0] - self.xmin[0])
            )

        if where == "left":
            return dist_left
        if where == "right":
            return dist_right
        if where == "bottom":
            return dist_bottom
        if where == "top":
            return dist_top
        if smoothness == "C0":
            return bkd.minimum(
                bkd.minimum(dist_left, dist_right), bkd.minimum(dist_bottom, dist_top)
            )
        return dist_left * dist_right * dist_bottom * dist_top

    @staticmethod
    def is_valid(vertices):
        """Check if the geometry is a Rectangle."""
        return (
            len(vertices) == 4
            and isclose(np.prod(vertices[1] - vertices[0]), 0)
            and isclose(np.prod(vertices[2] - vertices[1]), 0)
            and isclose(np.prod(vertices[3] - vertices[2]), 0)
            and isclose(np.prod(vertices[0] - vertices[3]), 0)
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
        self.center = np.array(center, dtype=config.real(np))
        self.radius = radius
        self.coeffs_cos = coeffs_cos
        self.coeffs_sin = coeffs_sin
        max_radius = radius + np.sum(coeffs_cos) + np.sum(coeffs_sin)
        super().__init__(
            2,
            (self.center - max_radius, self.center + max_radius),
            2 * max_radius,
        )

    def _r_theta(self, theta):
        """Define the parametrization r(theta) at angles theta."""
        result = self.radius * np.ones(theta.shape)
        for i, (coeff_cos, coeff_sin) in enumerate(
            zip(self.coeffs_cos, self.coeffs_sin), start=1
        ):
            result += coeff_cos * np.cos(i * theta) + coeff_sin * np.sin(i * theta)
        return result

    def _dr_theta(self, theta):
        """Evalutate the polar derivative r'(theta) at angles theta"""
        result = np.zeros(theta.shape)
        for i, (coeff_cos, coeff_sin) in enumerate(
            zip(self.coeffs_cos, self.coeffs_sin), start=1
        ):
            result += -coeff_cos * i * np.sin(i * theta) + coeff_sin * i * np.cos(
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
        return isclose(np.linalg.norm(r_theta - r), 0)

    def boundary_normal(self, x):
        _, theta = polar(x - self.center)
        dr_theta = self._dr_theta(theta)
        r_theta = self._r_theta(theta)

        dxt = np.vstack(
            (
                dr_theta * np.cos(theta) - r_theta * np.sin(theta),
                dr_theta * np.sin(theta) + r_theta * np.cos(theta),
            )
        ).T
        norm = np.linalg.norm(dxt, axis=-1, keepdims=True)
        dxt /= norm
        return np.array([dxt[:, 1], -dxt[:, 0]]).T

    def random_points(self, n, random="pseudo"):
        x = np.empty((0, 2), dtype=config.real(np))
        vbbox = self.bbox[1] - self.bbox[0]
        while len(x) < n:
            x_new = sample(n, 2, sampler="pseudo") * vbbox + self.bbox[0]
            x = np.vstack((x, x_new[self.inside(x_new)]))
        return x[:n]

    def uniform_boundary_points(self, n):
        theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        r_theta = self._r_theta(theta)
        X = np.vstack((r_theta * np.cos(theta), r_theta * np.sin(theta))).T
        return X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        u = sample(n, 1, random)
        theta = 2 * np.pi * u
        r_theta = self._r_theta(theta)
        X = np.hstack((r_theta * np.cos(theta), r_theta * np.sin(theta)))
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

        self.x1 = np.array(x1, dtype=config.real(np))
        self.x2 = np.array(x2, dtype=config.real(np))
        self.x3 = np.array(x3, dtype=config.real(np))

        self.v12 = self.x2 - self.x1
        self.v23 = self.x3 - self.x2
        self.v31 = self.x1 - self.x3
        self.l12 = np.linalg.norm(self.v12)
        self.l23 = np.linalg.norm(self.v23)
        self.l31 = np.linalg.norm(self.v31)
        self.n12 = self.v12 / self.l12
        self.n23 = self.v23 / self.l23
        self.n31 = self.v31 / self.l31
        self.n12_normal = clockwise_rotation_90(self.n12)
        self.n23_normal = clockwise_rotation_90(self.n23)
        self.n31_normal = clockwise_rotation_90(self.n31)
        self.perimeter = self.l12 + self.l23 + self.l31

        super().__init__(
            2,
            (np.minimum(x1, np.minimum(x2, x3)), np.maximum(x1, np.maximum(x2, x3))),
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
        _sign = np.hstack(
            [
                np.cross(self.v12, x - self.x1)[:, np.newaxis],
                np.cross(self.v23, x - self.x2)[:, np.newaxis],
                np.cross(self.v31, x - self.x3)[:, np.newaxis],
            ]
        )
        return ~np.logical_and(np.any(_sign > 0, axis=-1), np.any(_sign < 0, axis=-1))

    def on_boundary(self, x):
        l1 = np.linalg.norm(x - self.x1, axis=-1)
        l2 = np.linalg.norm(x - self.x2, axis=-1)
        l3 = np.linalg.norm(x - self.x3, axis=-1)
        return np.any(
            isclose(
                [l1 + l2 - self.l12, l2 + l3 - self.l23, l3 + l1 - self.l31],
                0,
            ),
            axis=0,
        )

    def boundary_normal(self, x):
        l1 = np.linalg.norm(x - self.x1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(x - self.x2, axis=-1, keepdims=True)
        l3 = np.linalg.norm(x - self.x3, axis=-1, keepdims=True)
        on12 = isclose(l1 + l2, self.l12)
        on23 = isclose(l2 + l3, self.l23)
        on31 = isclose(l3 + l1, self.l31)
        # Check points on the vertexes
        if np.any(np.count_nonzero(np.hstack([on12, on23, on31]), axis=-1) > 1):
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
        u = np.ravel(sample(n + 2, 1, random))
        # Remove the possible points very close to the corners
        u = u[np.logical_not(isclose(u, self.l12 / self.perimeter))]
        u = u[np.logical_not(isclose(u, (self.l12 + self.l23) / self.perimeter))]
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
        return np.vstack(x)

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
            self.x1_tensor = bkd.as_tensor(self.x1)
            self.x2_tensor = bkd.as_tensor(self.x2)
            self.x3_tensor = bkd.as_tensor(self.x3)

        diff_x1_x2 = diff_x1_x3 = diff_x2_x3 = None
        if where not in ["x1-x3", "x2-x3"]:
            diff_x1_x2 = (
                bkd.norm(x - self.x1_tensor, axis=-1, keepdims=True)
                + bkd.norm(x - self.x2_tensor, axis=-1, keepdims=True)
                - self.l12
            )
        if where not in ["x1-x2", "x2-x3"]:
            diff_x1_x3 = (
                bkd.norm(x - self.x1_tensor, axis=-1, keepdims=True)
                + bkd.norm(x - self.x3_tensor, axis=-1, keepdims=True)
                - self.l31
            )
        if where not in ["x1-x2", "x1-x3"]:
            diff_x2_x3 = (
                bkd.norm(x - self.x2_tensor, axis=-1, keepdims=True)
                + bkd.norm(x - self.x3_tensor, axis=-1, keepdims=True)
                - self.l23
            )

        if where is None:
            if smoothness == "C0":
                return bkd.minimum(bkd.minimum(diff_x1_x2, diff_x1_x3), diff_x2_x3)
            return diff_x1_x2 * diff_x1_x3 * diff_x2_x3
        if where == "x1-x2":
            return diff_x1_x2
        if where == "x1-x3":
            return diff_x1_x3
        return diff_x2_x3


class Polygon(Geometry):
    """Simple polygon.

    Args:
        vertices: The order of vertices can be in a clockwise or counterclockwise
            direction. The vertices will be re-ordered in counterclockwise (right hand
            rule).
    """

    def __init__(self, vertices):
        self.vertices = np.array(vertices, dtype=config.real(np))
        if len(vertices) == 3:
            raise ValueError("The polygon is a triangle. Use Triangle instead.")
        if Rectangle.is_valid(self.vertices):
            raise ValueError("The polygon is a rectangle. Use Rectangle instead.")

        self.area = polygon_signed_area(self.vertices)
        # Clockwise
        if self.area < 0:
            self.area = -self.area
            self.vertices = np.flipud(self.vertices)

        self.diagonals = spatial.distance.squareform(
            spatial.distance.pdist(self.vertices)
        )
        super().__init__(
            2,
            (np.amin(self.vertices, axis=0), np.amax(self.vertices, axis=0)),
            np.max(self.diagonals),
        )
        self.nvertices = len(self.vertices)
        self.perimeter = np.sum(
            [self.diagonals[i, i + 1] for i in range(-1, self.nvertices - 1)]
        )
        self.bbox = np.array(
            [np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)]
        )

        self.segments = self.vertices[1:] - self.vertices[:-1]
        self.segments = np.vstack((self.vertices[0] - self.vertices[-1], self.segments))
        self.normal = clockwise_rotation_90(self.segments.T).T
        self.normal = self.normal / np.linalg.norm(self.normal, axis=1).reshape(-1, 1)

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
            wn = np.zeros(len(P))  # Winding number counter

            # Repeat the first vertex at end
            # Loop through all edges of the polygon
            for i in range(-1, self.nvertices - 1):  # Edge from V[i] to V[i+1]
                tmp = np.all(
                    np.hstack(
                        [
                            V[i, 1] <= P[:, 1:2],  # Start y <= P[1]
                            V[i + 1, 1] > P[:, 1:2],  # An upward crossing
                            is_left(V[i], V[i + 1], P) > 0,  # P left of edge
                        ]
                    ),
                    axis=-1,
                )
                wn[tmp] += 1  # Have a valid up intersect
                tmp = np.all(
                    np.hstack(
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
        _on = np.zeros(shape=len(x), dtype=int)
        for i in range(-1, self.nvertices - 1):
            l1 = np.linalg.norm(self.vertices[i] - x, axis=-1)
            l2 = np.linalg.norm(self.vertices[i + 1] - x, axis=-1)
            _on[isclose(l1 + l2, self.diagonals[i, i + 1])] += 1
        return _on > 0

    @vectorize(excluded=[0], signature="(n)->(n)")
    def boundary_normal(self, x):
        for i in range(self.nvertices):
            if is_on_line_segment(self.vertices[i - 1], self.vertices[i], x):
                return self.normal[i]
        return np.array([0, 0])

    def random_points(self, n, random="pseudo"):
        x = np.empty((0, 2), dtype=config.real(np))
        vbbox = self.bbox[1] - self.bbox[0]
        while len(x) < n:
            x_new = sample(n, 2, sampler="pseudo") * vbbox + self.bbox[0]
            x = np.vstack((x, x_new[self.inside(x_new)]))
        return x[:n]

    def uniform_boundary_points(self, n):
        density = n / self.perimeter
        x = []
        for i in range(-1, self.nvertices - 1):
            x.append(
                np.linspace(
                    0,
                    1,
                    num=int(np.ceil(density * self.diagonals[i, i + 1])),
                    endpoint=False,
                )[:, None]
                * (self.vertices[i + 1] - self.vertices[i])
                + self.vertices[i]
            )
        x = np.vstack(x)
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_boundary_points(self, n, random="pseudo"):
        u = np.ravel(sample(n + self.nvertices, 1, random))
        # Remove the possible points very close to the corners
        l = 0
        for i in range(0, self.nvertices - 1):
            l += self.diagonals[i, i + 1]
            u = u[np.logical_not(isclose(u, l / self.perimeter))]
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
        return np.vstack(x)


def polygon_signed_area(vertices):
    """The (signed) area of a simple polygon.

    If the vertices are in the counterclockwise direction, then the area is positive; if
    they are in the clockwise direction, the area is negative.

    Shoelace formula: https://en.wikipedia.org/wiki/Shoelace_formula
    """
    x, y = zip(*vertices)
    x = np.array(list(x) + [x[0]])
    y = np.array(list(y) + [y[0]])
    return 0.5 * (np.sum(x[:-1] * y[1:]) - np.sum(x[1:] * y[:-1]))


def clockwise_rotation_90(v):
    """Rotate a vector of 90 degrees clockwise about the origin."""
    return np.array([v[1], -v[0]])


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
    return np.cross(P1 - P0, P2 - P0, axis=-1).reshape((-1, 1))


def is_rectangle(vertices):
    """Check if the geometry is a rectangle.

    https://stackoverflow.com/questions/2303278/find-if-4-points-on-a-plane-form-a-rectangle/2304031

    1. Find the center of mass of corner points: cx=(x1+x2+x3+x4)/4, cy=(y1+y2+y3+y4)/4
    2. Test if square of distances from center of mass to all 4 corners are equal
    """
    if len(vertices) != 4:
        return False

    c = np.mean(vertices, axis=0)
    d = np.sum((vertices - c) ** 2, axis=1)
    return np.allclose(d, np.full(4, d[0]))


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
        isclose(np.cross(v01, v02) / np.linalg.norm(v01), 0)
        # check that projection of P2 to line is between P0 and P1
        and v01 @ v02 >= 0
        and v01 @ v12 <= 0
    )
    # Not between P0 and P1, but close to P0 or P1
    # or isclose(np.linalg.norm(v02), 0)  # check whether P2 is close to P0
    # or isclose(np.linalg.norm(v12), 0)  # check whether P2 is close to P1


def polar(x):
    """Get the polar coordinated for a 2d vector in cartesian coordinates."""
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    theta = np.arctan2(x[:, 1], x[:, 0])
    return r, theta
