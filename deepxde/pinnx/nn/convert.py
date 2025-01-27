from typing import Dict

import brainstate as bst
import brainunit as u

__all__ = [
    'DictToArray',
    'ArrayToDict',
]


def dict_to_array(
    d: Dict[str, bst.typing.ArrayLike],
    axis: int = 1
):
    """
    Convert a dictionary to an array.

    Args:
        d (dict): The dictionary.
        axis (int): The axis to concatenate.

    Returns:
        ndarray: The array.
    """
    keys = tuple(d.keys())
    return u.math.stack([d[key] for key in keys], axis=axis)


class DictToArray(bst.nn.Module):
    """
    DictToArray layer, scaling the input data according to the given units, and merging them into an array.

    Args:
        axis (int): The axis to concatenate.
        **units: The units for each input. The unit should be the instance of ``brainunit.Unit``, but it can be None.
    """

    def __init__(self, axis: int = -1, **units):
        super().__init__()

        # axis
        assert isinstance(axis, int), f"DictToArray axis must be an integer. Please check the input values."
        self.axis = axis

        # unit scale
        self.units = units
        for val in units.values():
            assert isinstance(val, u.Unit) or val is None, (f"DictToArray values must be a unit or None. "
                                                            "Please check the input values.")

        self.in_size = len(units)
        self.out_size = len(units)

    def update(self, x: Dict[str, bst.typing.ArrayLike]):
        assert set(x.keys()) == set(self.units.keys()), (f"DictToArray keys mismatch. "
                                                         f"{set(x.keys())} != {set(self.units.keys())}.")

        # scale the input
        x_dict = dict()
        for key in self.units.keys():
            val = x[key]
            if isinstance(self.units[key], u.Unit):
                assert (isinstance(val, u.Quantity) or self.units[key].dim == u.DIMENSIONLESS), (
                    f"DictToArray values must be a quantity. "
                    "Please check the input values.")
                x_dict[key] = val.to_decimal(self.units[key]) if isinstance(val, u.Quantity) else val
            else:
                x_dict[key] = u.maybe_decimal(val)

        # convert to array
        arr = dict_to_array(x_dict, axis=self.axis)
        return arr


class ArrayToDict(bst.nn.Module):
    """
    Output layer, splitting the output data into a dict and assign the corresponding units.

    Args:
        axis (int): The axis to split the output data.
        **units: The units of the output data. The unit should be the instance
          of ``brainunit.Unit``, but it can be None.
    """

    def __init__(self, axis: int = -1, **units):
        super().__init__()

        assert isinstance(axis, int), f"Output axis must be an integer. "
        self.axis = axis
        self.units = units
        for val in units.values():
            assert isinstance(val, u.Unit) or val is None, (f"Input values must be a unit or None. "
                                                            "Please check the input values.")
        self.in_size = len(units)
        self.out_size = len(units)

    def update(self, arr: bst.typing.ArrayLike) -> Dict[str, bst.typing.ArrayLike]:
        assert arr.shape[self.axis] == len(self.units), (f"The number of columns of x must be "
                                                         f"equal to the number of units. "
                                                         f"Got {arr.shape[self.axis]} != {len(self.units)}. "
                                                         "Please check the input values.")
        shape = list(arr.shape)
        shape.pop(self.axis)
        xs = u.math.split(arr, len(self.units), axis=self.axis)

        keys = tuple(self.units.keys())
        units = tuple(self.units.values())
        res = dict()
        for key, unit, x in zip(keys, units, xs):
            res[key] = u.math.squeeze(x, axis=self.axis)
            if unit is not None:
                res[key] *= unit
        return res
