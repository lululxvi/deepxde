"""tensorflow backend implementation"""
from packaging.version import Version

import tensorflow as tf
import tensorflow_probability as tfp


if Version(tf.__version__) < Version("2.2.0"):
    raise RuntimeError("DeepXDE requires TensorFlow>=2.2.0.")
if Version(tfp.__version__) < Version("0.10.0"):
    raise RuntimeError("DeepXDE requires TensorFlow Probability>=0.10.0.")

lib = tf


def data_type_dict():
    return {
        "float16": tf.float16,
        "float32": tf.float32,
        "float64": tf.float64,
        "uint8": tf.uint8,
        "int8": tf.int8,
        "int16": tf.int16,
        "int32": tf.int32,
        "int64": tf.int64,
        "bool": tf.bool,
    }


def is_gpu_available():
    return bool(tf.config.list_physical_devices("GPU"))


def is_tensor(obj):
    return tf.is_tensor(obj)


def shape(input_tensor):
    return input_tensor.shape.as_list()


def ndim(input_tensor):
    return len(input_tensor.shape)


def transpose(tensor, axes=None):
    return tf.transpose(tensor, perm=axes)


def reshape(tensor, shape):
    return tf.reshape(tensor, shape)


def Variable(initial_value, dtype=None):
    return tf.Variable(initial_value=initial_value, trainable=True, dtype=dtype)


def as_tensor(data, dtype=None):
    if tf.is_tensor(data):
        if dtype is None or data.dtype == dtype:
            return data
        return tf.cast(data, dtype)
    return tf.convert_to_tensor(data, dtype=dtype)


def from_numpy(np_array):
    # Do memory copy:
    # https://stackoverflow.com/questions/47519802/does-tensorflow-convert-to-tensor-do-memory-copy
    # To avoid memory copy, use implicit conversion, but memory copy is still possible.
    # https://www.tensorflow.org/tutorials/customization/basics#numpy_compatibility
    return tf.convert_to_tensor(np_array)


def to_numpy(input_tensor):
    return input_tensor.numpy()


def concat(values, axis):
    return tf.concat(values, axis)


def stack(values, axis):
    return tf.stack(values, axis)


def expand_dims(tensor, axis):
    return tf.expand_dims(tensor, axis)


def reverse(tensor, axis):
    return tf.reverse(tensor, axis)


def roll(tensor, shift, axis):
    return tf.roll(tensor, shift, axis)


def lgamma(x):
    return tf.math.lgamma(x)


def elu(x):
    return tf.nn.elu(x)


def relu(x):
    return tf.nn.relu(x)


def selu(x):
    return tf.nn.selu(x)


def sigmoid(x):
    return tf.math.sigmoid(x)


def silu(x):
    return tf.keras.activations.swish(x)


def sin(x):
    return tf.math.sin(x)


def cos(x):
    return tf.math.cos(x)


def exp(x):
    return tf.math.exp(x)


def square(x):
    return tf.math.square(x)


# pylint: disable=redefined-builtin
def abs(x):
    return tf.math.abs(x)


def minimum(x, y):
    return tf.math.minimum(x, y)


def tanh(x):
    return tf.math.tanh(x)


def pow(x, y):
    return tf.math.pow(x, y)


def mean(input_tensor, dim, keepdims=False):
    return tf.math.reduce_mean(input_tensor, axis=dim, keepdims=keepdims)


def reduce_mean(input_tensor):
    return tf.math.reduce_mean(input_tensor)


def sum(input_tensor, dim, keepdims=False):
    return tf.math.reduce_sum(input_tensor, axis=dim, keepdims=keepdims)


def reduce_sum(input_tensor):
    return tf.math.reduce_sum(input_tensor)


def prod(input_tensor, dim, keepdims=False):
    return tf.math.reduce_prod(input_tensor, axis=dim, keepdims=keepdims)


def reduce_prod(input_tensor):
    return tf.math.reduce_prod(input_tensor)


# pylint: disable=redefined-builtin
def min(input_tensor, dim, keepdims=False):
    return tf.math.reduce_min(input_tensor, axis=dim, keepdims=keepdims)


def reduce_min(input_tensor):
    return tf.math.reduce_min(input_tensor)


# pylint: disable=redefined-builtin
def max(input_tensor, dim, keepdims=False):
    return tf.math.reduce_max(input_tensor, axis=dim, keepdims=keepdims)


def reduce_max(input_tensor):
    return tf.math.reduce_max(input_tensor)


def norm(tensor, ord=None, axis=None, keepdims=False):
    if ord is None:
        ord = "euclidean"
    return tf.norm(tensor, ord=ord, axis=axis, keepdims=keepdims)


def zeros(shape, dtype):
    return tf.zeros(shape, dtype=dtype)


def zeros_like(input_tensor):
    return tf.zeros_like(input_tensor)


def matmul(x, y):
    return tf.linalg.matmul(x, y)
