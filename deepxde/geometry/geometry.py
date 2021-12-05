import abc

import numpy as np


class Geometry(abc.ABC):
    def __init__(self, dim, bbox, diam):
        self.dim = dim
        self.bbox = bbox
        self.diam = min(diam, np.linalg.norm(bbox[1] - bbox[0]))
        self.idstr = type(self).__name__

    @abc.abstractmethod
    def inside(self, x):
        """Check if x is inside the geometry (including the boundary)."""

    @abc.abstractmethod
    def on_boundary(self, x):
        """Check if x is on the geometry boundary."""

    def distance2boundary(self, x, dirn):
        raise NotImplementedError(
            "{}.distance2boundary to be implemented".format(self.idstr)
        )

    def mindist2boundary(self, x):
        raise NotImplementedError(
            "{}.mindist2boundary to be implemented".format(self.idstr)
        )

    def boundary_normal(self, x):
        """Compute the unit normal at x for Neumann or Robin boundary conditions."""
        raise NotImplementedError(
            "{}.boundary_normal to be implemented".format(self.idstr)
        )

    def uniform_points(self, n, boundary=True):
        """Compute the equispaced point locations in the geometry."""
        print(
            "Warning: {}.uniform_points not implemented. Use random_points instead.".format(
                self.idstr
            )
        )
        return self.random_points(n)

    @abc.abstractmethod
    def random_points(self, n, random="pseudo"):
        """Compute the random point locations in the geometry."""

    def uniform_boundary_points(self, n):
        """Compute the equispaced point locations on the boundary."""
        print(
            "Warning: {}.uniform_boundary_points not implemented. Use random_boundary_points instead.".format(
                self.idstr
            )
        )
        return self.random_boundary_points(n)

    @abc.abstractmethod
    def random_boundary_points(self, n, random="pseudo"):
        """Compute the random point locations on the boundary."""

    def periodic_point(self, x, component):
        """Compute the periodic image of x for periodic boundary condition."""
        raise NotImplementedError(
            "{}.periodic_point to be implemented".format(self.idstr)
        )

    def background_points(self, x, dirn, dist2npt, shift):
        raise NotImplementedError(
            "{}.background_points to be implemented".format(self.idstr)
        )

    def union(self, other):
        """CSG Union."""
        from . import csg

        return csg.CSGUnion(self, other)

    def __or__(self, other):
        """CSG Union."""
        from . import csg

        return csg.CSGUnion(self, other)

    def difference(self, other):
        """CSG Difference."""
        from . import csg

        return csg.CSGDifference(self, other)

    def __sub__(self, other):
        """CSG Difference."""
        from . import csg

        return csg.CSGDifference(self, other)

    def intersection(self, other):
        """CSG Intersection."""
        from . import csg

        return csg.CSGIntersection(self, other)

    def __and__(self, other):
        """CSG Intersection."""
        from . import csg

        return csg.CSGIntersection(self, other)
