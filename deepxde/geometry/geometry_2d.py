__all__ = ["Disk", "Polygon", "Rectangle", "Triangle"]

import numpy as np
from scipy import spatial

from .geometry import Geometry
from .geometry_nd import Hypercube
from .sampler import sample
from .. import config
from ..utils import vectorize


class Disk(Geometry):
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=config.real(np))
        self.radius = radius
        super().__init__(2, (self.center - radius, self.center + radius), 2 * radius)

        self._r2 = radius**2

    def inside(self, x):
        return np.linalg.norm(x - self.center, axis=-1) <= self.radius

    def on_boundary(self, x):
        return np.isclose(np.linalg.norm(x - self.center, axis=-1), self.radius)

    def distance2boundary_unitdirn(self, x, dirn):
        # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        xc = x - self.center
        ad = np.dot(xc, dirn)
        return (-ad + (ad ** 2 - np.sum(xc * xc, axis=-1) + self._r2) ** 0.5).astype(config.real(np))

    def distance2boundary(self, x, dirn):
        return self.distance2boundary_unitdirn(x, dirn / np.linalg.norm(dirn))

    def mindist2boundary(self, x):
        return np.amin(self.radius - np.linalg.norm(x - self.center, axis=1))

    def boundary_normal(self, x):
        _n = x - self.center
        l = np.linalg.norm(_n, axis=-1, keepdims=True)
        _n = _n / l * np.isclose(l, self.radius)
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
        pts = x - np.arange(-shift, n - shift + 1, dtype=config.real(np))[:, None] * h * dirn
        return pts


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
        u = u[np.logical_not(np.isclose(u, l1 / self.perimeter))]
        u = u[np.logical_not(np.isclose(u, l3 / self.perimeter))]
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

    @staticmethod
    def is_valid(vertices):
        """Check if the geometry is a Rectangle."""
        return (
            len(vertices) == 4
            and np.isclose(np.prod(vertices[1] - vertices[0]), 0)
            and np.isclose(np.prod(vertices[2] - vertices[1]), 0)
            and np.isclose(np.prod(vertices[3] - vertices[2]), 0)
            and np.isclose(np.prod(vertices[0] - vertices[3]), 0)
        )


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
            np.isclose(
                [l1 + l2 - self.l12, l2 + l3 - self.l23, l3 + l1 - self.l31],
                0,
                atol=1e-6,
            ),
            axis=0,
        )

    def boundary_normal(self, x):
        l1 = np.linalg.norm(x - self.x1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(x - self.x2, axis=-1, keepdims=True)
        l3 = np.linalg.norm(x - self.x3, axis=-1, keepdims=True)
        on12 = np.isclose(l1 + l2, self.l12)
        on23 = np.isclose(l2 + l3, self.l23)
        on31 = np.isclose(l3 + l1, self.l31)
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
        u = u[np.logical_not(np.isclose(u, self.l12 / self.perimeter))]
        u = u[np.logical_not(np.isclose(u, (self.l12 + self.l23) / self.perimeter))]
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
        _on = np.zeros(shape=len(x), dtype=np.int)
        for i in range(-1, self.nvertices - 1):
            l1 = np.linalg.norm(self.vertices[i] - x, axis=-1)
            l2 = np.linalg.norm(self.vertices[i + 1] - x, axis=-1)
            _on[np.isclose(l1 + l2, self.diagonals[i, i + 1])] += 1
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
            u = u[np.logical_not(np.isclose(u, l / self.perimeter))]
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
        np.isclose(np.cross(v01, v02) / np.linalg.norm(v01), 0, atol=1e-6)
        # check that projection of P2 to line is between P0 and P1
        and v01 @ v02 >= 0
        and v01 @ v12 <= 0
    )
    # Not between P0 and P1, but close to P0 or P1
    # or np.isclose(np.linalg.norm(v02), 0, atol=1e-6)  # check whether P2 is close to P0
    # or np.isclose(np.linalg.norm(v12), 0, atol=1e-6)  # check whether P2 is close to P1
