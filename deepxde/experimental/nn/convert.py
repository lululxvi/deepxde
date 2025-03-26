from typing import Dict

import brainstate as bst
import brainunit as u

__all__ = [
    "DictToArray",
    "ArrayToDict",
]


def dict_to_array(d: Dict[str, bst.typing.ArrayLike], axis: int = 1):
    """
    Convert a dictionary of array-like values to a single concatenated array.

    This function takes a dictionary where each value is an array-like object,
    and concatenates all these arrays along the specified axis to create a
    single output array.

    Args:
        d (Dict[str, bst.typing.ArrayLike]): A dictionary where keys are strings
            and values are array-like objects (e.g., numpy arrays, lists, etc.).
        axis (int, optional): The axis along which the arrays should be concatenated.
            Default is 1.

    Returns:
        ndarray: A single array containing all the input arrays concatenated
            along the specified axis. The order of concatenation is determined
            by the order of the keys in the input dictionary.

    Example:
        >>> d = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        >>> dict_to_array(d)
        array([[1, 4],
               [2, 5],
               [3, 6]])
    """
    keys = tuple(d.keys())
    return u.math.stack([d[key] for key in keys], axis=axis)


class DictToArray(bst.nn.Module):
    """
    DictToArray layer, scaling the input data according to the given units, and merging them into an array.

    This layer takes a dictionary of array-like inputs, scales them according to specified units,
    and concatenates them into a single array along a specified axis.

    Args:
        axis (int, optional): The axis along which to concatenate the input arrays. Defaults to -1.
        **units: Keyword arguments specifying the units for each input. Each unit should be an
                 instance of ``brainunit.Unit`` or None.

    Attributes:
        axis (int): The axis along which concatenation is performed.
        units (dict): A dictionary mapping input keys to their corresponding units.
        in_size (int): The number of input elements (length of units dictionary).
        out_size (int): The number of output elements (same as in_size).
    """

    def __init__(self, axis: int = -1, **units):
        super().__init__()

        # axis
        assert isinstance(
            axis, int
        ), f"DictToArray axis must be an integer. Please check the input values."
        self.axis = axis

        # unit scale
        self.units = units
        for val in units.values():
            assert isinstance(val, u.Unit) or val is None, (
                f"DictToArray values must be a unit or None. "
                "Please check the input values."
            )

        self.in_size = len(units)
        self.out_size = len(units)

    def update(self, x: Dict[str, bst.typing.ArrayLike]):
        """
        Scales the input dictionary values according to their units and concatenates them into an array.

        Args:
            x (Dict[str, bst.typing.ArrayLike]): A dictionary of input arrays to be scaled and concatenated.
                The keys should match those specified in the units dictionary during initialization.

        Returns:
            ndarray: A single array containing all the scaled input arrays concatenated along the specified axis.

        Raises:
            AssertionError: If the input dictionary keys don't match the units dictionary keys,
                            or if the input values are not of the expected type (Quantity or dimensionless).
        """
        assert set(x.keys()) == set(self.units.keys()), (
            f"DictToArray keys mismatch. "
            f"{set(x.keys())} != {set(self.units.keys())}."
        )

        # scale the input
        x_dict = dict()
        for key in self.units.keys():
            val = x[key]
            if isinstance(self.units[key], u.Unit):
                assert (
                    isinstance(val, u.Quantity)
                    or self.units[key].dim == u.DIMENSIONLESS
                ), (
                    f"DictToArray values must be a quantity. "
                    "Please check the input values."
                )
                x_dict[key] = (
                    val.to_decimal(self.units[key])
                    if isinstance(val, u.Quantity)
                    else val
                )
            else:
                x_dict[key] = u.maybe_decimal(val)

        # convert to array
        arr = dict_to_array(x_dict, axis=self.axis)
        return arr


class ArrayToDict(bst.nn.Module):
    """
    Output layer, splitting the output data into a dict and assign the corresponding units.

    This class takes an input array and splits it into a dictionary, where each key-value pair
    represents a specific output with its corresponding unit.

    Args:
        axis (int, optional): The axis along which to split the output data. Defaults to -1.
        **units: Keyword arguments specifying the units for each output. Each unit should be an
                 instance of ``brainunit.Unit`` or None.

    Attributes:
        axis (int): The axis along which splitting is performed.
        units (dict): A dictionary mapping output keys to their corresponding units.
        in_size (int): The number of input elements (length of units dictionary).
        out_size (int): The number of output elements (same as in_size).
    """

    def __init__(self, axis: int = -1, **units):
        super().__init__()

        assert isinstance(axis, int), f"Output axis must be an integer. "
        self.axis = axis
        self.units = units
        for val in units.values():
            assert isinstance(val, u.Unit) or val is None, (
                f"Input values must be a unit or None. "
                "Please check the input values."
            )
        self.in_size = len(units)
        self.out_size = len(units)

    def update(self, arr: bst.typing.ArrayLike) -> Dict[str, bst.typing.ArrayLike]:
        """
        Splits the input array into a dictionary and assigns the corresponding units.

        This method takes an input array, splits it along the specified axis, and creates
        a dictionary where each key-value pair represents a specific output with its
        corresponding unit.

        Args:
            arr (bst.typing.ArrayLike): The input array to be split and converted into a dictionary.

        Returns:
            Dict[str, bst.typing.ArrayLike]: A dictionary where keys are the output names and
            values are the corresponding split arrays, potentially with units applied.

        Raises:
            AssertionError: If the shape of the input array along the specified axis doesn't
                            match the number of units provided during initialization.
        """
        assert arr.shape[self.axis] == len(self.units), (
            f"The number of columns of x must be "
            f"equal to the number of units. "
            f"Got {arr.shape[self.axis]} != {len(self.units)}. "
            "Please check the input values."
        )
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
