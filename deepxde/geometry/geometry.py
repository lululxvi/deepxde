from __future__ import annotations

import abc
from typing import Callable, Literal
from numbers import Number

import numpy as np
from numpy.typing import NDArray

from ..types import Tensor, TensorOrTensors

class Geometry(abc.ABC):
    def __init__(self, dim: int, bbox: NDArray[np.float_], diam: Number):
        self.dim = dim
        self.bbox = bbox
        self.diam = min(diam, np.linalg.norm(bbox[1] - bbox[0]))
        self.idstr = type(self).__name__

    @abc.abstractmethod
    def inside(self, x: NDArray[np.float_]) -> NDArray[np.bool_]:
        """Check if x is inside the geometry (including the boundary)."""

    @abc.abstractmethod
    def on_boundary(self, x: NDArray[np.float_]) -> NDArray[np.bool_]:
        """Check if x is on the geometry boundary."""

    def distance2boundary(self, x: NDArray[np.float_], dirn: Number) -> NDArray[np.float_]:
        raise NotImplementedError(
            "{}.distance2boundary to be implemented".format(self.idstr)
        )

    def mindist2boundary(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        raise NotImplementedError(
            "{}.mindist2boundary to be implemented".format(self.idstr)
        )

    def boundary_constraint_factor(
        self, x: NDArray[np.float_], smoothness: Literal["C0", "C0+", "Cinf"] = "C0+"
    ) -> Tensor:
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

        Returns:
            A tensor of a type determined by the backend, which will have a shape of (n, 1).
            Each element in the tensor corresponds to the computed distance value for the respective point in `x`.
        """
        raise NotImplementedError(
            "{}.boundary_constraint_factor to be implemented".format(self.idstr)
        )

    def boundary_normal(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Compute the unit normal at x for Neumann or Robin boundary conditions."""
        raise NotImplementedError(
            "{}.boundary_normal to be implemented".format(self.idstr)
        )

    def uniform_points(self, n: int, boundary: bool = True)-> NDArray[np.float_]:
        """Compute the equispaced point locations in the geometry."""
        print(
            "Warning: {}.uniform_points not implemented. Use random_points instead.".format(
                self.idstr
            )
        )
        return self.random_points(n)

    @abc.abstractmethod
    def random_points(self, n: int, random: str = "pseudo") -> NDArray[np.float_]:
        """Compute the random point locations in the geometry."""

    def uniform_boundary_points(self, n: int) -> NDArray[np.float_]:
        """Compute the equispaced point locations on the boundary."""
        print(
            "Warning: {}.uniform_boundary_points not implemented. Use random_boundary_points instead.".format(
                self.idstr
            )
        )
        return self.random_boundary_points(n)

    @abc.abstractmethod
    def random_boundary_points(self, n: int, random: str = "pseudo") -> NDArray[np.float_]:
        """Compute the random point locations on the boundary."""

    def periodic_point(self, x: NDArray[np.float_], component: int | list[int]) -> NDArray[np.float_]:
        """Compute the periodic image of x for periodic boundary condition."""
        raise NotImplementedError(
            "{}.periodic_point to be implemented".format(self.idstr)
        )

    def background_points(self, x: NDArray[np.float_], dirn: Number, dist2npt: Callable[[NDArray[np.float_]], int], shift: int) -> NDArray[np.float_]:
        raise NotImplementedError(
            "{}.background_points to be implemented".format(self.idstr)
        )

    def union(self, other: Geometry) -> Geometry:
        """CSG Union."""
        from . import csg

        return csg.CSGUnion(self, other)

    def __or__(self, other: Geometry) -> Geometry:
        """CSG Union."""
        from . import csg

        return csg.CSGUnion(self, other)

    def difference(self, other: Geometry) -> Geometry:
        """CSG Difference."""
        from . import csg

        return csg.CSGDifference(self, other)

    def __sub__(self, other: Geometry) -> Geometry:
        """CSG Difference."""
        from . import csg

        return csg.CSGDifference(self, other)

    def intersection(self, other: Geometry) -> Geometry:
        """CSG Intersection."""
        from . import csg

        return csg.CSGIntersection(self, other)

    def __and__(self, other: Geometry) -> Geometry:
        """CSG Intersection."""
        from . import csg

        return csg.CSGIntersection(self, other)
