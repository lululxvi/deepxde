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
    if is_tensor(array):
        return tf.pad(array, tf.constant(pad_width))
    return np.pad(array, pad_width)


def _padding_array(array, nprocs):
    """Padding an array with the last value of array, so as to be divided by nprocs

    Args:
        array (array or Tensor): Array to be padded
        nprocs (int): Number of world_size

    Returns:
        array or Tensor: Padded array or Tensor
    """
    # pad with npad elements, %nprocs at last in case of nprocs=1
    npad = (nprocs - len(array) % nprocs) % nprocs
    if npad == 0:
        return array
    if isinstance(array, np.ndarray):
        datapad = array[-1, :].reshape([-1, array[-1, :].shape[0]])
        for _ in range(npad):
            array = np.append(array, datapad, axis=0)
    elif bkd.is_tensor(array):
        element_shape = array[0].shape[2:]
        datapad = array[-1].reshape([-1, *element_shape])
        for _ in range(npad):
            array = bkd.concat([array, datapad], axis=0)
    else:
        raise NotImplementedError(f"Array of type `{type(array)}` is not supported now.")
    return array


def _sub(array, nprocs, n):
    """Get continuous subset of array or Tensor according to rank n

    Args:
        array (array or Tensor): array to be subdivided
        nprocs (int): Number of world_size
        n (int): Rank of current device

    Returns:
        array or Tensor: Subdivided array or Tensor
    """
    N = len(array)
    subN = N // nprocs
    s = subN * n
    e = subN * (n + 1)
    return array[s: e]


def sub_with_padding(array):
    """
    Get continuous subset of an padded array or Tensor according to rank n
    Args:
        array (array or Tensor): Array to be sharded

    Returns:
        array or Tensor: Subdivided array or Tensor
    """
    if config.world_size <= 1:
        return array
    array_pad = _padding_array(array, config.world_size)
    return _sub(array_pad, config.world_size, config.rank)
