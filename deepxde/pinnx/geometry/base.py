# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


import abc
from typing import Dict, Union, Literal, Sequence

import brainstate as bst
import brainunit as u
import jax.numpy as jnp
import numpy as np

from deepxde.pinnx import utils
from deepxde.geometry.geometry import AbstractGeometry

__all__ = [
    'AbstractGeometry',
]


class GeometryPINNx(AbstractGeometry):

    def to_dict_point(self, *names, **kw_names):
        """
        Convert the geometry to a dictionary geometry.

        Args:
            names: The names of the coordinates.
            kw_names: The names of the coordinates and their physical units.
        """
        return DictPointGeometry(self, *names, **kw_names)





def quantity_to_array(quantity: Union[np.ndarray, jnp.ndarray, u.Quantity], unit: u.Unit):
    if isinstance(quantity, u.Quantity):
        return quantity.to(unit).magnitude
    else:
        assert unit.is_unitless, "The unit should be unitless."
    return quantity


def array_to_quantity(array: Union[np.ndarray, jnp.ndarray], unit: u.Unit):
    return u.math.maybe_decimal(u.Quantity(array, unit=unit))


class DictPointGeometry(GeometryPINNx):
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
