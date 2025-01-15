# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


import abc
from typing import Dict, Union
from typing import Literal, Sequence

import brainstate as bst
import brainunit as u
import jax.numpy as jnp
import numpy as np

from pinnx import utils

__all__ = [
    'AbstractGeometry',
    'Geometry',
    'CSGUnion',
    'CSGDifference',
    'CSGIntersection'
]


class AbstractGeometry(abc.ABC):
    def __init__(self, dim: int):
        assert isinstance(dim, int), "dim must be an integer"
        self.dim = dim
        self.idstr = type(self).__name__

    @abc.abstractmethod
    def inside(self, x) -> np.ndarray[bool]:
        """
        Check if x is inside the geometry (including the boundary).

        Args:
            x: A 2D array of shape (n, dim), where `n` is the number of points and
               `dim` is the dimension of the geometry.

        Returns:
            A boolean array of shape (n,) where each element is True if the point is inside the geometry.
        """

    @abc.abstractmethod
    def on_boundary(self, x) -> np.ndarray[bool]:
        """
        Check if x is on the geometry boundary.

        Args:
            x: A 2D array of shape (n, dim), where `n` is the number of points and
               `dim` is the dimension of the geometry.

        Returns:
            A boolean array of shape (n,) where each element is True if the point is on the boundary.
        """

    def distance2boundary(self, x, dirn):
        """
        Compute the distance to the boundary.

        Args:
            x: A 2D array of shape (n, dim), where `n` is the number of points and
                `dim` is the dimension of the geometry.
            dirn: A 2D array of shape (n, dim), where `n` is the number of points and
                `dim` is the dimension of the geometry. The direction of the distance
                computation. If `dirn` is not provided, the distance is computed in the
                normal direction.
        """
        raise NotImplementedError("{}.distance2boundary to be implemented".format(self.idstr))

    def mindist2boundary(self, x):
        """
        Compute the minimum distance to the boundary.

        Args:
            x: A 2D array of shape (n, dim), where `n` is the number of points and
                `dim` is the dimension of the geometry.
        """
        raise NotImplementedError("{}.mindist2boundary to be implemented".format(self.idstr))

    def boundary_constraint_factor(
        self,
        x,
        smoothness: Literal["C0", "C0+", "Cinf"] = "C0+"
    ):
        """
        Compute the hard constraint factor at x for the boundary.

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
        raise NotImplementedError("{}.boundary_constraint_factor to be implemented".format(self.idstr))

    def boundary_normal(self, x):
        """
        Compute the unit normal at x for Neumann or Robin boundary conditions.

        Args:
            x: A 2D array of shape (n, dim), where `n` is the number of points and
               `dim` is the dimension of the geometry.
        """
        raise NotImplementedError("{}.boundary_normal to be implemented".format(self.idstr))

    @abc.abstractmethod
    def uniform_points(self, n, boundary: bool = True) -> np.ndarray:
        """
        Compute the equispaced point locations in the geometry.

        Args:
            n: The number of points.
            boundary: If True, include the boundary points.
        """

    @abc.abstractmethod
    def random_points(self, n, random: str = "pseudo") -> np.ndarray:
        """
        Compute the random point locations in the geometry.

        Args:
            n: The number of points.
            random: The random distribution. One of the following: "pseudo" (pseudorandom),
                "LHS" (Latin hypercube sampling), "Halton" (Halton sequence),
                "Hammersley" (Hammersley sequence), or "Sobol" (Sobol sequence
        """

    @abc.abstractmethod
    def uniform_boundary_points(self, n) -> np.ndarray:
        """
        Compute the equispaced point locations on the boundary.

        Args:
            n: The number of points.
        """

    @abc.abstractmethod
    def random_boundary_points(self, n, random: str = "pseudo") -> np.ndarray:
        """Compute the random point locations on the boundary."""

    def periodic_point(self, x, component):
        """
        Compute the periodic image of x for periodic boundary condition.

        Args:
            x: A 2D array of shape (n, dim), where `n` is the number of points and
                `dim` is the dimension of the geometry.
            component: The component of the periodic direction.
        """
        raise NotImplementedError("{}.periodic_point to be implemented".format(self.idstr))

    def background_points(self, x, dirn, dist2npt, shift):
        """
        Compute the background points for the collocation points.

        Args:
            x: A 2D array of shape (n, dim), where `n` is the number of points and
                `dim` is the dimension of the geometry.
            dirn: The direction of the background points. One of the following: -1 (left),
                or 1 (right), or 0 (both direction).
            dist2npt: A function which converts distance to the number of extra
                points (not including x).
            shift: The number of shift.
        """
        raise NotImplementedError("{}.background_points to be implemented".format(self.idstr))

    def to_dict_point(self, *names, **kw_names):
        """
        Convert the geometry to a dictionary geometry.

        Args:
            names: The names of the coordinates.
            kw_names: The names of the coordinates and their physical units.
        """
        return DictPointGeometry(self, *names, **kw_names)


class Geometry(AbstractGeometry):
    """
    Base class for defining geometries.

    Args:
        dim: The dimension of the geometry.
        bbox: The bounding box of the geometry.
        diam: The diameter of the geometry.
    """

    def __init__(
        self,
        dim: int,
        bbox: Sequence,
        diam: float,
    ):
        super().__init__(dim)
        self.bbox = bbox
        self.diam = np.minimum(diam, np.linalg.norm(bbox[1] - bbox[0]))

    def uniform_points(self, n, boundary: bool = True) -> np.ndarray:
        """
        Compute the equi-spaced point locations in the geometry.

        Args:
            n: The number of points.
            boundary: If True, include the boundary points.
        """
        print("Warning: {}.uniform_points not implemented. Use random_points instead.".format(self.idstr))
        return self.random_points(n)

    def uniform_boundary_points(self, n) -> np.ndarray:
        """
        Compute the equi-spaced point locations on the boundary.

        Args:
            n: The number of points.
        """
        print("Warning: {}.uniform_boundary_points not implemented. "
              "Use random_boundary_points instead.".format(self.idstr))
        return self.random_boundary_points(n)

    def union(self, other):
        """
        CSG Union.

        Args:
            other: The other geometry object.
        """
        return self.__or__(other)

    def __or__(self, other):
        """CSG Union."""
        return CSGUnion(self, other)

    def difference(self, other):
        """
        CSG Difference.

        Args:
            other: The other geometry object.
        """
        return self.__sub__(other)

    def __sub__(self, other):
        """CSG Difference."""
        return CSGDifference(self, other)

    def intersection(self, other):
        """
        CSG Intersection.

        Args:
            other: The other geometry object.
        """
        return self.__and__(other)

    def __and__(self, other):
        """CSG Intersection."""
        return CSGIntersection(self, other)


class CSGUnion(Geometry):
    """
    Construct an object by CSG Union.

    Args:
        geom1: The first geometry object.
        geom2: The second geometry object.
    """

    def __init__(self, geom1, geom2):
        assert isinstance(geom1, Geometry), "geom1 must be an instance of Geometry"
        assert isinstance(geom2, Geometry), "geom2 must be an instance of Geometry"
        if geom1.dim != geom2.dim:
            raise ValueError("{} | {} failed (dimensions do not match).".format(geom1.idstr, geom2.idstr))
        super().__init__(
            geom1.dim,
            (
                np.minimum(geom1.bbox[0], geom2.bbox[0]),
                np.maximum(geom1.bbox[1], geom2.bbox[1]),
            ),
            geom1.diam + geom2.diam,
        )
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        mod = utils.smart_numpy(x)
        return mod.logical_or(self.geom1.inside(x), self.geom2.inside(x))

    def on_boundary(self, x):
        mod = utils.smart_numpy(x)
        return mod.logical_or(
            mod.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x)),
            mod.logical_and(self.geom2.on_boundary(x), ~self.geom1.inside(x)),
        )

    def boundary_normal(self, x):
        mod = utils.smart_numpy(x)
        return (
            mod.logical_and(self.geom1.on_boundary(x),
                            ~self.geom2.inside(x))[:, np.newaxis] * self.geom1.boundary_normal(x)
            +
            mod.logical_and(self.geom2.on_boundary(x),
                            ~self.geom1.inside(x))[:, np.newaxis] * self.geom2.boundary_normal(x)
        )

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=bst.environ.dftype())
        i = 0
        while i < n:
            tmp = (
                np.random.rand(n, self.dim) * (self.bbox[1] - self.bbox[0])
                + self.bbox[0]
            )
            tmp = tmp[self.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i: i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=bst.environ.dftype())
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[~self.geom2.inside(geom1_boundary_points)]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[~self.geom1.inside(geom2_boundary_points)]

            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i: i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def periodic_point(self, x, component):
        x = np.copy(x)
        on_boundary_geom1 = np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x))
        x[on_boundary_geom1] = self.geom1.periodic_point(x, component)[on_boundary_geom1]
        on_boundary_geom2 = np.logical_and(self.geom2.on_boundary(x), ~self.geom1.inside(x))
        x[on_boundary_geom2] = self.geom2.periodic_point(x, component)[on_boundary_geom2]
        return x


class CSGDifference(Geometry):
    """
    Construct an object by CSG Difference.

    Args:
        geom1: The first geometry object.
        geom2: The second geometry object.
    """

    def __init__(self, geom1, geom2):
        assert isinstance(geom1, Geometry), "geom1 must be an instance of Geometry"
        assert isinstance(geom2, Geometry), "geom2 must be an instance of Geometry"
        if geom1.dim != geom2.dim:
            raise ValueError("{} - {} failed (dimensions do not match).".format(geom1.idstr, geom2.idstr))
        super().__init__(geom1.dim, geom1.bbox, geom1.diam)
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        mod = utils.smart_numpy(x)
        return mod.logical_and(self.geom1.inside(x), ~self.geom2.inside(x))

    def on_boundary(self, x):
        mod = utils.smart_numpy(x)
        return mod.logical_or(
            mod.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x)),
            mod.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x)),
        )

    def boundary_normal(self, x):
        mod = utils.smart_numpy(x)
        return (
            mod.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x))[:, np.newaxis] *
            self.geom1.boundary_normal(x)
            +
            mod.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x))[:, np.newaxis] *
            -self.geom2.boundary_normal(x)
        )

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=bst.environ.dftype())
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[~self.geom2.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i: i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=bst.environ.dftype())
        i = 0
        while i < n:

            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[~self.geom2.inside(geom1_boundary_points)]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[self.geom1.inside(geom2_boundary_points)]

            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i: i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def periodic_point(self, x, component):
        x = np.copy(x)
        on_boundary_geom1 = np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x))
        x[on_boundary_geom1] = self.geom1.periodic_point(x, component)[on_boundary_geom1]
        return x


class CSGIntersection(Geometry):
    """
    Construct an object by CSG Intersection.

    Args:
        geom1: The first geometry object.
        geom2: The second geometry object.
    """

    def __init__(self, geom1, geom2):
        assert isinstance(geom1, Geometry), "geom1 must be an instance of Geometry"
        assert isinstance(geom2, Geometry), "geom2 must be an instance of Geometry"
        if geom1.dim != geom2.dim:
            raise ValueError("{} & {} failed (dimensions do not match).".format(geom1.idstr, geom2.idstr))
        super().__init__(
            geom1.dim,
            (
                np.maximum(geom1.bbox[0], geom2.bbox[0]),
                np.minimum(geom1.bbox[1], geom2.bbox[1]),
            ),
            min(geom1.diam, geom2.diam),
        )
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        mod = utils.smart_numpy(x)
        return mod.logical_and(self.geom1.inside(x), self.geom2.inside(x))

    def on_boundary(self, x):
        mod = utils.smart_numpy(x)
        return mod.logical_or(
            mod.logical_and(self.geom1.on_boundary(x), self.geom2.inside(x)),
            mod.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x)),
        )

    def boundary_normal(self, x):
        mod = utils.smart_numpy(x)
        return (
            mod.logical_and(self.geom1.on_boundary(x), self.geom2.inside(x))[:, np.newaxis] *
            self.geom1.boundary_normal(x)
            +
            mod.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x))[:, np.newaxis] *
            self.geom2.boundary_normal(x)
        )

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=bst.environ.dftype())
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[self.geom2.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i: i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=bst.environ.dftype())
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[self.geom2.inside(geom1_boundary_points)]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[self.geom1.inside(geom2_boundary_points)]

            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i: i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def periodic_point(self, x, component):
        x = np.copy(x)
        on_boundary_geom1 = np.logical_and(self.geom1.on_boundary(x), self.geom2.inside(x))
        x[on_boundary_geom1] = self.geom1.periodic_point(x, component)[on_boundary_geom1]
        on_boundary_geom2 = np.logical_and(self.geom2.on_boundary(x), self.geom1.inside(x))
        x[on_boundary_geom2] = self.geom2.periodic_point(x, component)[on_boundary_geom2]
        return x


def quantity_to_array(quantity: Union[np.ndarray, jnp.ndarray, u.Quantity], unit: u.Unit):
    if isinstance(quantity, u.Quantity):
        return quantity.to(unit).magnitude
    else:
        assert unit.is_unitless, "The unit should be unitless."
    return quantity


def array_to_quantity(array: Union[np.ndarray, jnp.ndarray], unit: u.Unit):
    return u.math.maybe_decimal(u.Quantity(array, unit=unit))


class DictPointGeometry(AbstractGeometry):
    """
    Convert a geometry to a dictionary geometry.
    """

    def __init__(self, geom: AbstractGeometry, *names, **kw_names):
        super().__init__(geom.dim)

        self.geom = geom
        for name in names:
            assert isinstance(name, str), "The name should be a string."
        kw_names = {key: u.UNITLESS if unit is None else unit for key, unit in kw_names.items()}
        for key, unit in kw_names.items():
            assert isinstance(key, str), "The name should be a string."
            assert isinstance(unit, u.Unit), "The unit should be a brainunit.Unit."
        self.name2unit = {name: u.UNITLESS for name in names}
        self.name2unit.update(kw_names)
        if len(self.name2unit) != geom.dim:
            raise ValueError("The number of names should match the dimension of the geometry. "
                             "But got {} names and {} dimensions.".format(len(self.name2unit), geom.dim))

    def arr_to_dict(self, x: bst.typing.ArrayLike) -> Dict[str, bst.typing.ArrayLike]:
        return {name: array_to_quantity(x[..., i], unit)
                for i, (name, unit) in enumerate(self.name2unit.items())}

    def dict_to_arr(self, x: Dict[str, bst.typing.ArrayLike]) -> bst.typing.ArrayLike:
        arrs = [quantity_to_array(x[name], unit) for name, unit in self.name2unit.items()]
        mod = utils.smart_numpy(arrs[0])
        return mod.stack(arrs, axis=-1)

    def inside(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]) -> np.ndarray[bool]:
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        return self.geom.inside(x)

    def on_initial(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]) -> np.ndarray:
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        return self.geom.on_initial(x)

    def on_boundary(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]) -> np.ndarray[bool]:
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        return self.geom.on_boundary(x)

    def distance2boundary(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict], dirn: int) -> np.ndarray:
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        return self.geom.distance2boundary(x, dirn)

    def mindist2boundary(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]) -> np.ndarray:
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        return self.geom.mindist2boundary(x)

    def boundary_constraint_factor(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict], **kw) -> np.ndarray:
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        return self.geom.boundary_constraint_factor(x, **kw)

    def boundary_normal(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]) -> Dict[str, bst.typing.ArrayLike]:
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        normal = self.geom.boundary_normal(x)
        return self.arr_to_dict(normal)

    def uniform_points(self, n, boundary: bool = True) -> Dict[str, bst.typing.ArrayLike]:
        points = self.geom.uniform_points(n, boundary=boundary)
        return self.arr_to_dict(points)

    def random_points(self, n, random="pseudo") -> Dict[str, bst.typing.ArrayLike]:
        points = self.geom.random_points(n, random=random)
        return self.arr_to_dict(points)

    def uniform_boundary_points(self, n) -> Dict[str, bst.typing.ArrayLike]:
        points = self.geom.uniform_boundary_points(n)
        return self.arr_to_dict(points)

    def random_boundary_points(self, n, random: str = "pseudo") -> Dict[str, bst.typing.ArrayLike]:
        points = self.geom.random_boundary_points(n, random=random)
        return self.arr_to_dict(points)

    def periodic_point(self, x, component: Union[str, int]) -> Dict[str, bst.typing.ArrayLike]:
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        if isinstance(component, str):
            component = list(self.name2unit.keys()).index(component)
        assert isinstance(component, int), f"The component should be an integer or a string. But got {component}."
        x = self.geom.periodic_point(x, component)
        return self.arr_to_dict(x)

    def background_points(self, x, dirn, dist2npt, shift) -> Dict[str, bst.typing.ArrayLike]:
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        points = self.geom.background_points(x, dirn, dist2npt, shift)
        return self.arr_to_dict(points)

    def random_initial_points(self, n: int, random: str = "pseudo") -> Dict[str, bst.typing.ArrayLike]:
        points = self.geom.random_initial_points(n, random=random)
        return self.arr_to_dict(points)

    def uniform_initial_points(self, n: int) -> Dict[str, bst.typing.ArrayLike]:
        points = self.geom.uniform_initial_points(n)
        return self.arr_to_dict(points)
