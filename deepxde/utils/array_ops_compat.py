"""Operations which handle numpy and tensorflow.compat.v1 automatically."""

import numpy as np

from .. import backend as bkd
from .. import config
from ..backend import is_tensor, tf


def istensorlist(values):
    return any(map(is_tensor, values))


def convert_to_array(value):
    """Convert a list of numpy arrays or tensors to a numpy array or a tensor."""
    if istensorlist(value):
        # TODO: use concat instead of stack as paddle now use shape [1,]
        # for 0-D tensor, it will be solved soon.
        if bkd.backend_name == "paddle":
            return bkd.concat(value, axis=0)
        return bkd.stack(value, axis=0)
    value = np.array(value)
    if value.dtype != config.real(np):
        return value.astype(config.real(np))
    return value


def hstack(tup):
    if not is_tensor(tup[0]) and tup[0] == []:
        tup = list(tup)
        if istensorlist(tup[1:]):
            tup[0] = bkd.as_tensor([], dtype=config.real(bkd.lib))
        else:
            tup[0] = np.array([], dtype=config.real(np))
    return bkd.concat(tup, 0) if is_tensor(tup[0]) else np.hstack(tup)


def roll(a, shift, axis):
    return tf.roll(a, shift, axis) if is_tensor(a) else np.roll(a, shift, axis=axis)


def split_in_rank(array):
    """Split given array into continuous subarray according to world size and rank.

    Args:
        array (array or Tensor): Array to be split.

    Returns:
        array or Tensor: Split array or Tensor.
    """
    if config.world_size <= 1:
        return array

    n_total = len(array)
    if n_total % config.world_size > 0:
        raise ValueError(
            f"The data length({n_total}) must be an "
            f"integer multiple of world_size({config.world_size})."
        )
    n_split = n_total // config.world_size
    beg = n_split * config.rank
    end = n_split * (config.rank + 1)
    return array[beg: end]
