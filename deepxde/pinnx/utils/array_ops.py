# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


from typing import Sequence

import brainstate as bst
import brainunit as u
import jax
import numpy as np


def is_tensor(obj):
    return isinstance(obj, (jax.Array, u.Quantity, np.ndarray))


def istensorlist(values):
    return any(map(is_tensor, values))


def convert_to_array(value: Sequence):
    """Convert a list of numpy arrays or tensors to a numpy array or a tensor."""
    if istensorlist(value):
        return np.stack(value, axis=0)
    return np.array(value, dtype=bst.environ.dftype())


def hstack(tup):
    if not is_tensor(tup[0]) and isinstance(tup[0], list) and tup[0] == []:
        tup = list(tup)
        if istensorlist(tup[1:]):
            tup[0] = np.asarray([], dtype=bst.environ.dftype())
        else:
            tup[0] = np.array([], dtype=bst.environ.dftype())
    return np.concatenate(tup, 0) if is_tensor(tup[0]) else np.hstack(tup)


def zero_padding(array, pad_width):
    # SparseTensor
    if isinstance(array, (list, tuple)) and len(array) == 3:
        indices, values, dense_shape = array
        indices = [(i + pad_width[0][0], j + pad_width[1][0]) for i, j in indices]
        dense_shape = (
            dense_shape[0] + sum(pad_width[0]),
            dense_shape[1] + sum(pad_width[1]),
        )
        return indices, values, dense_shape
    return np.pad(array, pad_width)
