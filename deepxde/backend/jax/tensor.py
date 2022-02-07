"""jax backend implementation"""
import jax
import jax.numpy as jnp
import numpy as np


lib = jax

# TODO:support jax.numpy.float64, which is not automatically enabled by default, and will be truncated to jax.numpy.float32 for now.
# See https://github.com/google/jax#current-gotchas for more.
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
    return list(input_array.shape)


def ndim(input_array):
    return jnp.ndim(input_array)


def Variable(initial_value, dtype=None):
    # TODO: add variable class to jax
    raise NotImplementedError(
        "Variable functionality to be implemented for jax backend"
    )


def as_tensor(data, dtype=None):
    if isinstance(data, jnp.ndarray):
        if dtype is None or data.dtype == dtype:
            return data
        return data.astype(dtype=dtype)
    return jnp.asarray(data, dtype=dtype)


def from_numpy(np_array):
    # jax.numpy.array, jax.numpy.asarray, jax.device_put work without memory copy.
    # jax.numpy.array and jax.numpy.asarray can transform numpy array to different dtype, at a cost, however, with memory copy.
    # jax.numpy.array and jax.numpy.asarray create arrays on JAX's default device, while jax.device_put on designated device.
    # np_array in np.float64 is automatically converted to jax.DeviceArray in jnp.float32
    return jax.device_put(np_array)


def to_numpy(input_tensor):
    return np.asarray(input_tensor)


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


def square(x):
    return jnp.square(x)


def tanh(x):
    return jax.nn.tanh(x)


def mean(input_tensor, dim, keepdims=False):
    return jnp.mean(input_tensor, axis=dim, keepdims=keepdims)


def reduce_mean(input_tensor):
    return jnp.mean(input_tensor)


def sum(input_tensor, dim, keepdims=False):
    return jnp.sum(input_tensor, axis=dim, keepdims=keepdims)


def reduce_sum(input_tensor):
    return jnp.sum(input_tensor)


def zeros(shape, dtype):
    return jnp.zeros(shape, dtype=dtype)


def zeros_like(input_tensor):
    return jnp.zeros_like(input_tensor)
