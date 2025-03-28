"""jax backend implementation"""
import jax
import jax.numpy as jnp
import numpy as np


lib = jax


def data_type_dict():
    return {
        "float16": jnp.float16,
        "float32": jnp.float32,
        "float64": jnp.float64,
        "uint8": jnp.uint8,
        "int8": jnp.int8,
        "int16": jnp.int16,
        "int32": jnp.int32,
        "int64": jnp.int64,
        "bool": jnp.bool_,
    }


def is_tensor(obj):
    return isinstance(obj, jnp.ndarray)


def shape(input_array):
    return input_array.shape


def ndim(input_array):
    return input_array.ndim


def transpose(tensor, axes=None):
    return jnp.transpose(tensor, axes=axes)


def reshape(tensor, shape):
    return jnp.reshape(tensor, shape)


class Variable:
    def __init__(self, initial_value, dtype=None):
        self._value = jnp.array(initial_value, dtype=dtype)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value


def as_tensor(data, dtype=None):
    if isinstance(data, jnp.ndarray):
        if dtype is None or data.dtype == dtype:
            return data
        return data.astype(dtype)
    return jnp.asarray(data, dtype=dtype)


def from_numpy(np_array):
    return jnp.asarray(np_array)


def to_numpy(input_tensor):
    return np.asarray(input_tensor)


def concat(values, axis):
    return jnp.concatenate(values, axis=axis)


def stack(values, axis):
    return jnp.stack(values, axis=axis)


def elu(x):
    return jax.nn.elu(x)


def relu(x):
    return jax.nn.relu(x)


def selu(x):
    return jax.nn.selu(x)


def sigmoid(x):
    return jax.nn.sigmoid(x)


def silu(x):
    return jax.nn.silu(x)


def sin(x):
    return jnp.sin(x)


def cos(x):
    return jnp.cos(x)


def square(x):
    return jnp.square(x)


# pylint: disable=redefined-builtin
def abs(x):
    return jnp.abs(x)


def minimum(x, y):
    return jnp.minimum(x, y)


def tanh(x):
    return jnp.tanh(x)


def mean(input_tensor, dim, keepdims=False):
    return jnp.mean(input_tensor, axis=dim, keepdims=keepdims)


def reduce_mean(input_tensor):
    return jnp.mean(input_tensor)


def sum(input_tensor, dim, keepdims=False):
    return jnp.sum(input_tensor, axis=dim, keepdims=keepdims)


def reduce_sum(input_tensor):
    return jnp.sum(input_tensor)


def prod(input_tensor, dim, keepdims=False):
    return jnp.prod(input_tensor, axis=dim, keepdims=keepdims)


def reduce_prod(input_tensor):
    return jnp.prod(input_tensor)


# pylint: disable=redefined-builtin
def min(input_tensor, dim, keepdims=False):
    return jnp.min(input_tensor, axis=dim, keepdims=keepdims)


def reduce_min(input_tensor):
    return jnp.min(input_tensor)


# pylint: disable=redefined-builtin
def max(input_tensor, dim, keepdims=False):
    return jnp.max(input_tensor, axis=dim, keepdims=keepdims)


def reduce_max(input_tensor):
    return jnp.max(input_tensor)


def norm(tensor, ord=None, axis=None, keepdims=False):
    return jnp.linalg.norm(tensor, ord=ord, axis=axis, keepdims=keepdims)


def zeros(shape, dtype):
    return jnp.zeros(shape, dtype=dtype)


def zeros_like(input_tensor):
    return jnp.zeros_like(input_tensor)


def l1_regularization(l1):
    return lambda params: l1 * jnp.sum(jnp.concatenate([jnp.abs(w).flatten() for w in params]))


def l2_regularization(l2):
    return lambda params: l2 * jnp.sum(jnp.concatenate([jnp.square(w).flatten() for w in params]))


def l1_l2_regularization(l1, l2):
    return lambda params: l1_regularization(l1)(params) + l2_regularization(l2)(params)
