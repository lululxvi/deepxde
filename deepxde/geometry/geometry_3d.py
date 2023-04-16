import itertools

import numpy as np

from .geometry_2d import Rectangle
from .geometry_nd import Hypercube, Hypersphere, Literal, Union, bkd


class Cuboid(Hypercube):
    """
    Args:
        xmin: Coordinate of bottom left corner.
        xmax: Coordinate of top right corner.
    """

    def __init__(self, xmin, xmax):
        super().__init__(xmin, xmax)
        dx = self.xmax - self.xmin
        self.area = 2 * np.sum(dx * np.roll(dx, 2))

    def random_boundary_points(self, n, random="pseudo"):
        pts = []
        density = n / self.area
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

    def approxdist2boundary(self, x, where: Union[
            None, Literal["back", "front", "left", "right", 
                        "bottom", "top"]] = None,
        smoothness: Literal["L", "M", "H"] = "M",
        inside: bool = True):
        """
        `inside`: `x` is either inside or outside the geometry.
        The case where there are both points inside and points
        outside the geometry is NOT allowed.

        NOTE: currently only support `inside=True`.

        See `Geometry.approxdist2boundary()` for more info on args.
        """
        
        assert smoothness in ["L", "M", "H"], "smoothness must be one of L, M, H"
        assert self.dim == 3
        assert inside, "inside=False is not supported for Cuboid"

        if not hasattr(self, "self.xmin_tensor"):
            self.xmin_tensor = bkd.as_tensor(self.xmin)
            self.xmax_tensor = bkd.as_tensor(self.xmax)

        dist_l = bkd.abs((x - self.xmin_tensor) /
                        (self.xmax_tensor - self.xmin_tensor) * 2)
        dist_r = bkd.abs((x - self.xmax_tensor) /
                        (self.xmax_tensor - self.xmin_tensor) * 2)
        
        if where == "back":
            return dist_l[:, 0:1]
        elif where == "front":
            return dist_r[:, 0:1]
        elif where == "left":
            return dist_l[:, 1:2]
        elif where == "right":
            return dist_r[:, 1:2]
        elif where == "bottom":
            return dist_l[:, 2:]
        elif where == "top":
            return dist_r[:, 2:]

        if smoothness == "L":
            dist_l = bkd.min(dist_l, dim=-1, keepdims=True)
            dist_r = bkd.min(dist_r, dim=-1, keepdims=True)
            return bkd.minimum(dist_l, dist_r)
        else:
            dist_l = bkd.prod(dist_l, dim=-1, keepdims=True)
            dist_r = bkd.prod(dist_r, dim=-1, keepdims=True)
            return dist_l * dist_r


class Sphere(Hypersphere):
    """
    Args:
        center: Center of the sphere.
        radius: Radius of the sphere.
    """
