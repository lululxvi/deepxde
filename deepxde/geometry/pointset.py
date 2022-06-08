import numpy as np

from .geometry import Geometry
from ..data import BatchSampler


class PointSet(Geometry):
    """A geometry defined by a set of points.

    Args:
        points: A NumPy array of shape (`N`, `dx`). A list of `dx`-dim points.
    """

    def __init__(self, points):
        self.points = np.asarray(points)
        super().__init__(
            len(points[0]),
            (np.amin(self.points, axis=0), np.amax(self.points, axis=0)),
            np.inf,
        )

        self.sampler = BatchSampler(len(self.points), shuffle=True)

    def inside(self, x):
        raise NotImplementedError("dde.geometry.PointSet doesn't support inside.")

    def on_boundary(self, x):
        raise NotImplementedError("dde.geometry.PointSet doesn't support on_boundary.")

    def random_points(self, n, random="pseudo"):
        if n > len(self.points):
            raise ValueError(f"dde.geometry.PointSet doesn't have {n} points.")

        indices = self.sampler.get_next(n)
        return self.points[indices]

    def random_boundary_points(self, n, random="pseudo"):
        raise NotImplementedError(
            "dde.geometry.PointSet doesn't support random_boundary_points."
        )
