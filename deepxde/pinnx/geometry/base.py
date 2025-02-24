# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


from typing import Dict, Union

import brainstate as bst
import brainunit as u
import jax.numpy as jnp
import numpy as np

from deepxde.geometry.geometry import Geometry
from deepxde.pinnx import utils

__all__ = [
    'GeometryPINNx',
    'DictPointGeometry',
]


class GeometryPINNx(Geometry):
    """
    A base class for geometries in the PINNx (Physics-Informed Neural Networks Extended) framework.

    This class extends the functionality of the base Geometry class to provide additional
    features specific to the PINNx framework. It serves as a foundation for creating
    more specialized geometry classes that can work with dictionary-based point representations
    and unit-aware computations.

    Attributes:
        Inherits all attributes from the Geometry base class.

    Methods:
        to_dict_point(*names, **kw_names):
            Converts the geometry to a dictionary-based point representation.

    Note:
        This class is designed to be subclassed for specific geometry implementations
        in the PINNx framework. It provides a bridge between the standard Geometry
        representations and the more flexible, unit-aware representations used in PINNx.

    Example:
        class CustomGeometry(GeometryPINNx):
            def __init__(self, dim, bbox, diam):
                super().__init__(dim, bbox, diam)
                # Additional initialization specific to CustomGeometry

            # Implement other required methods

        # Usage
        custom_geom = CustomGeometry(dim=2, bbox=[0, 1, 0, 1], diam=1.414)
        dict_geom = custom_geom.to_dict_point('x', 'y', z=u.meter)
    """
    def to_dict_point(self, *names, **kw_names):
        """
        Convert the geometry to a dictionary geometry.

        This method creates a DictPointGeometry object, which represents the geometry
        using named coordinates and their associated units.

        Args:
            *names (str): Variable length argument list of coordinate names.
                          These are assumed to be unitless.
            **kw_names (dict): Arbitrary keyword arguments where keys are coordinate names
                                and values are their corresponding units.

        Returns:
            DictPointGeometry: A new geometry object that represents the current geometry
                               using a dictionary-based structure with named coordinates
                               and units.

        Raises:
            ValueError: If the number of provided names doesn't match the dimension of the geometry.

        Note:
            If a coordinate is specified in both *names and **kw_names, the unit from **kw_names will be used.
        """
        return DictPointGeometry(self, *names, **kw_names)


def quantity_to_array(quantity: Union[np.ndarray, jnp.ndarray, u.Quantity], unit: u.Unit):
    """
    Convert a quantity to an array with specified units.

    This function takes a quantity (which can be a numpy array, JAX array, or a Quantity object)
    and converts it to an array with the specified units. If the input is already a Quantity,
    it is converted to the specified unit and its magnitude is returned. If the input is an array,
    it is returned as-is, but only if the specified unit is unitless.

    Parameters:
    -----------
    quantity : Union[np.ndarray, jnp.ndarray, u.Quantity]
        The input quantity to be converted. Can be a numpy array, JAX array, or a Quantity object.
    unit : u.Unit
        The target unit for conversion. If the input is not a Quantity, this must be unitless.

    Returns:
    --------
    Union[np.ndarray, jnp.ndarray]
        The magnitude of the quantity in the specified units, returned as an array.

    Raises:
    -------
    AssertionError
        If the input is not a Quantity and the specified unit is not unitless.
    """
    if isinstance(quantity, u.Quantity):
        return quantity.to(unit).magnitude
    else:
        assert unit.is_unitless, "The unit should be unitless."
    return quantity


def array_to_quantity(array: Union[np.ndarray, jnp.ndarray], unit: u.Unit):
    """
    Convert an array to a Quantity object with specified units.

    This function takes an array (either numpy or JAX) and a unit, and returns
    a Quantity object representing the array with the given unit.

    Parameters:
    -----------
    array : Union[np.ndarray, jnp.ndarray]
        The input array to be converted to a Quantity. Can be either a numpy array
        or a JAX array.
    unit : u.Unit
        The unit to be associated with the array values.

    Returns:
    --------
    u.Quantity
        A Quantity object representing the input array with the specified unit.
        The returned object may be a decimal representation if appropriate.
    """
    return u.math.maybe_decimal(u.Quantity(array, unit=unit))


class DictPointGeometry(GeometryPINNx):
    """
    A class that converts a standard Geometry object to a dictionary-based geometry representation.

    This class extends GeometryPINNx to provide a more flexible, named coordinate system
    with unit awareness. It wraps an existing Geometry object and allows access to its
    methods while providing additional functionality for working with named coordinates.

    Attributes:
        geom (Geometry): The original geometry object being wrapped.
        name2unit (dict): A dictionary mapping coordinate names to their corresponding units.

    Methods:
        arr_to_dict(x): Convert an array to a dictionary of named quantities.
        dict_to_arr(x): Convert a dictionary of named quantities to an array.
        inside(x): Check if points are inside the geometry.
        on_initial(x): Check if points are on the initial boundary.
        on_boundary(x): Check if points are on the boundary of the geometry.
        distance2boundary(x, dirn): Calculate the distance to the boundary in a specific direction.
        mindist2boundary(x): Calculate the minimum distance to the boundary.
        boundary_constraint_factor(x, **kw): Calculate the boundary constraint factor.
        boundary_normal(x): Calculate the boundary normal vectors.
        uniform_points(n, boundary): Generate uniformly distributed points in the geometry.
        random_points(n, random): Generate random points in the geometry.
        uniform_boundary_points(n): Generate uniformly distributed points on the boundary.
        random_boundary_points(n, random): Generate random points on the boundary.
        periodic_point(x, component): Find the periodic point for a given point and component.
        background_points(x, dirn, dist2npt, shift): Generate background points.
        random_initial_points(n, random): Generate random initial points.
        uniform_initial_points(n): Generate uniformly distributed initial points.

    The class provides a bridge between array-based and dictionary-based representations
    of geometric points, allowing for more intuitive and unit-aware manipulations of
    geometric data in the context of physics-informed neural networks.

    Example:
        geom = Geometry(dim=2, bbox=[0, 1, 0, 1], diam=1.414)
        dict_geom = DictPointGeometry(geom, 'x', y=u.meter)
        points = dict_geom.uniform_points(100)
        # points will be a dictionary with keys 'x' (unitless) and 'y' (in meters)
    """

    def __init__(self, geom: Geometry, *names, **kw_names):
        """
        Initialize a DictPointGeometry object.

        Args:
            geom (Geometry): The base geometry object to be converted.
            *names (str): Variable length argument list of coordinate names (assumed to be unitless).
            **kw_names (dict): Arbitrary keyword arguments where keys are coordinate names
                                and values are their corresponding units.

        Raises:
            ValueError: If the number of provided names doesn't match the dimension of the geometry.
        """
        super().__init__(geom.dim, geom.bbox, geom.diam)

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
        """
        Convert an array to a dictionary of named quantities.

        Args:
            x (ArrayLike): The input array to be converted.

        Returns:
            Dict[str, ArrayLike]: A dictionary where keys are coordinate names and values are quantities.
        """
        return {name: array_to_quantity(x[..., i], unit)
                for i, (name, unit) in enumerate(self.name2unit.items())}

    def dict_to_arr(self, x: Dict[str, bst.typing.ArrayLike]) -> bst.typing.ArrayLike:
        """
        Convert a dictionary of named quantities to an array.

        Args:
            x (Dict[str, ArrayLike]): The input dictionary to be converted.

        Returns:
            ArrayLike: The resulting array.
        """
        arrs = [quantity_to_array(x[name], unit) for name, unit in self.name2unit.items()]
        mod = utils.smart_numpy(arrs[0])
        return mod.stack(arrs, axis=-1)

    def inside(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]) -> np.ndarray[bool]:
        """
        Check if points are inside the geometry.

        Args:
            x (Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]): The points to check.

        Returns:
            np.ndarray[bool]: Boolean array indicating whether each point is inside the geometry.
        """
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        return self.geom.inside(x)

    def on_initial(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]) -> np.ndarray:
        """
        Check if points are on the initial boundary.

        Args:
            x (Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]): The points to check.

        Returns:
            np.ndarray: Array indicating whether each point is on the initial boundary.
        """
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        return self.geom.on_initial(x)

    def on_boundary(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]) -> np.ndarray[bool]:
        """
        Check if points are on the boundary of the geometry.

        Args:
            x (Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]): The points to check.

        Returns:
            np.ndarray[bool]: Boolean array indicating whether each point is on the boundary.
        """
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        return self.geom.on_boundary(x)

    def distance2boundary(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict], dirn: int) -> np.ndarray:
        """
        Calculate the distance to the boundary in a specific direction.

        Args:
            x (Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]): The points to calculate from.
            dirn (int): The direction to calculate the distance.

        Returns:
            np.ndarray: Array of distances to the boundary.
        """
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        return self.geom.distance2boundary(x, dirn)

    def mindist2boundary(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]) -> np.ndarray:
        """
        Calculate the minimum distance to the boundary.

        Args:
            x (Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]): The points to calculate from.

        Returns:
            np.ndarray: Array of minimum distances to the boundary.
        """
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        return self.geom.mindist2boundary(x)

    def boundary_constraint_factor(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict], **kw) -> np.ndarray:
        """
        Calculate the boundary constraint factor.

        Args:
            x (Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]): The points to calculate for.
            **kw: Additional keyword arguments.

        Returns:
            np.ndarray: Array of boundary constraint factors.
        """
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        return self.geom.boundary_constraint_factor(x, **kw)

    def boundary_normal(self, x: Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]) -> Dict[str, bst.typing.ArrayLike]:
        """
        Calculate the boundary normal vectors.

        Args:
            x (Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]): The points to calculate for.

        Returns:
            Dict[str, ArrayLike]: Dictionary of boundary normal vectors.
        """
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        normal = self.geom.boundary_normal(x)
        return self.arr_to_dict(normal)

    def uniform_points(self, n, boundary: bool = True) -> Dict[str, bst.typing.ArrayLike]:
        """
        Generate uniformly distributed points in the geometry.

        Args:
            n (int): Number of points to generate.
            boundary (bool, optional): Whether to include boundary points. Defaults to True.

        Returns:
            Dict[str, ArrayLike]: Dictionary of generated points.
        """
        points = self.geom.uniform_points(n, boundary=boundary)
        return self.arr_to_dict(points)

    def random_points(self, n, random="pseudo") -> Dict[str, bst.typing.ArrayLike]:
        """
        Generate random points in the geometry.

        Args:
            n (int): Number of points to generate.
            random (str, optional): Type of random number generation. Defaults to "pseudo".

        Returns:
            Dict[str, ArrayLike]: Dictionary of generated points.
        """
        points = self.geom.random_points(n, random=random)
        return self.arr_to_dict(points)

    def uniform_boundary_points(self, n) -> Dict[str, bst.typing.ArrayLike]:
        """
        Generate uniformly distributed points on the boundary.

        Args:
            n (int): Number of points to generate.

        Returns:
            Dict[str, ArrayLike]: Dictionary of generated boundary points.
        """
        points = self.geom.uniform_boundary_points(n)
        return self.arr_to_dict(points)

    def random_boundary_points(self, n, random: str = "pseudo") -> Dict[str, bst.typing.ArrayLike]:
        """
        Generate random points on the boundary.

        Args:
            n (int): Number of points to generate.
            random (str, optional): Type of random number generation. Defaults to "pseudo".

        Returns:
            Dict[str, ArrayLike]: Dictionary of generated boundary points.
        """
        points = self.geom.random_boundary_points(n, random=random)
        return self.arr_to_dict(points)

    def periodic_point(self, x, component: Union[str, int]) -> Dict[str, bst.typing.ArrayLike]:
        """
        Find the periodic point for a given point and component.

        Args:
            x (Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]): The point to find the periodic point for.
            component (Union[str, int]): The component to consider for periodicity.

        Returns:
            Dict[str, ArrayLike]: Dictionary of the periodic point.

        Raises:
            AssertionError: If the component is not an integer or a string.
        """
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        if isinstance(component, str):
            component = list(self.name2unit.keys()).index(component)
        assert isinstance(component, int), f"The component should be an integer or a string. But got {component}."
        x = self.geom.periodic_point(x, component)
        return self.arr_to_dict(x)

    def background_points(self, x, dirn, dist2npt, shift) -> Dict[str, bst.typing.ArrayLike]:
        """
        Generate background points.

        Args:
            x (Union[np.ndarray, jnp.ndarray, u.Quantity, Dict]): The reference points.
            dirn: The direction for generating background points.
            dist2npt: The distance to number of points mapping.
            shift: The shift to apply.

        Returns:
            Dict[str, ArrayLike]: Dictionary of generated background points.
        """
        if isinstance(x, dict):
            x = self.dict_to_arr(x)
        points = self.geom.background_points(x, dirn, dist2npt, shift)
        return self.arr_to_dict(points)

    def random_initial_points(self, n: int, random: str = "pseudo") -> Dict[str, bst.typing.ArrayLike]:
        """
        Generate random initial points.

        Args:
            n (int): Number of points to generate.
            random (str, optional): Type of random number generation. Defaults to "pseudo".

        Returns:
            Dict[str, ArrayLike]: Dictionary of generated initial points.
        """
        points = self.geom.random_initial_points(n, random=random)
        return self.arr_to_dict(points)

    def uniform_initial_points(self, n: int) -> Dict[str, bst.typing.ArrayLike]:
        """
        Generate uniformly distributed initial points.

        Args:
            n (int): Number of points to generate.

        Returns:
            Dict[str, ArrayLike]: Dictionary of generated initial points.
        """
        points = self.geom.uniform_initial_points(n)
        return self.arr_to_dict(points)
