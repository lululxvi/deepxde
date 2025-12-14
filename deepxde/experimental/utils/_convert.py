from typing import Sequence, Dict

import brainstate as bst
import brainunit as u
import numpy as np

__all__ = [
    "array_to_dict",
    "dict_to_array",
]


def array_to_dict(
    x: bst.typing.ArrayLike, names: Sequence[str], keep_dim: bool = False
):
    """
    Convert args to a dictionary.

    """
    if x.shape[-1] != len(names):
        raise ValueError(
            "The number of columns of x must be equal to the number of names."
        )

    if keep_dim:
        return {key: x[..., i : i + 1] for i, key in enumerate(names)}
    else:
        return {key: x[..., i] for i, key in enumerate(names)}


def dict_to_array(d: Dict[str, bst.typing.ArrayLike], keep_dim: bool = False):
    """
    Convert a dictionary to an array.

    Args:
        d (dict): The dictionary.
        keep_dim (bool): Whether to keep the dimension.

    Returns:
        ndarray: The array.
    """
    keys = tuple(d.keys())
    if isinstance(d[keys[0]], np.ndarray):
        if keep_dim:
            return np.concatenate([d[key] for key in keys], axis=-1)
        else:
            return np.stack([d[key] for key in keys], axis=-1)
    else:
        if keep_dim:
            return u.math.concatenate([d[key] for key in keys], axis=-1)
        else:
            return u.math.stack([d[key] for key in keys], axis=-1)
