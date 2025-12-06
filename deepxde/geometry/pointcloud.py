import numpy as np

from .geometry import Geometry
from .. import config
from ..data import BatchSampler
from ..utils import PointSet


class PointCloud(Geometry):
    """A geometry represented by a point cloud, i.e., a set of points in space.

    Args:
        points: A 2-D NumPy array. If `boundary_points` is not provided, `points` can
            include points both inside the geometry or on the boundary; if `boundary_points`
            is provided, `points` includes only points inside the geometry.
        boundary_points: A 2-D NumPy array.
        boundary_normals: A 2-D NumPy array.
    """

    def __init__(self, points, boundary_points=None, boundary_normals=None):
        self.points = np.asarray(points, dtype=config.real(np))
        self.num_points = len(self.points)
        self._point_set = PointSet(self.points)
        self.boundary_points = None
        self.boundary_normals = None
        all_points = self.points
        if boundary_points is not None:
            self.boundary_points = np.asarray(boundary_points, dtype=config.real(np))
            self.num_boundary_points = len(self.boundary_points)
            all_points = np.vstack((self.points, self.boundary_points))
            self.boundary_sampler = BatchSampler(self.num_boundary_points, shuffle=True)
            self._boundary_point_set = PointSet(self.boundary_points)
            if boundary_normals is not None:
                self.boundary_normals = np.asarray(
                    boundary_normals, dtype=config.real(np)
                )
                self._boundary_normal_query = self._boundary_point_set.values_to_func(self.boundary_normals, default_value=None)
        super().__init__(
            len(self.points[0]),
            (np.amin(all_points, axis=0), np.amax(all_points, axis=0)),
            np.inf,
        )
        self.sampler = BatchSampler(self.num_points, shuffle=True)

    def inside(self, x):
        return self._point_set.inside(x)

    def on_boundary(self, x):
        if self.boundary_points is None:
            raise ValueError("boundary_points must be defined to test on_boundary")
        return self._boundary_point_set.inside(x)

    def boundary_normal(self, x):
        if self.boundary_normals is None:
            raise ValueError(
                "boundary_normals must be defined for boundary_normal"
            )
        return self._boundary_normal_query(x)
    
    def random_points(self, n, random="pseudo"):
        if n <= self.num_points:
            indices = self.sampler.get_next(n)
            return self.points[indices]

        x = np.tile(self.points, (n // self.num_points, 1))
        indices = self.sampler.get_next(n % self.num_points)
        return np.vstack((x, self.points[indices]))

    def random_boundary_points(self, n, random="pseudo"):
        if self.boundary_points is None:
            raise ValueError("boundary_points must be defined to test on_boundary")
        if n <= self.num_boundary_points:
            indices = self.boundary_sampler.get_next(n)
            return self.boundary_points[indices]

        x = np.tile(self.boundary_points, (n // self.num_boundary_points, 1))
        indices = self.boundary_sampler.get_next(n % self.num_boundary_points)
        return np.vstack((x, self.boundary_points[indices]))
