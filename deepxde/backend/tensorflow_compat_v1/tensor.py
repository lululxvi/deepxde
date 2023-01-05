"""tensorflow.compat.v1 backend implementation"""
from packaging.version import Version

import tensorflow.compat.v1 as tf


if Version(tf.__version__) < Version("2.7.0"):
    raise RuntimeError("DeepXDE requires TensorFlow>=2.7.0.")


# The major changes from TensorFlow 1.x to TensorFlow 2.x are:
# 1. Eager execution: enable_eager_execution(), disable_eager_execution()
# 2. Resource variables: enable_resource_variables(), disable_resource_variables()
# 3. Tensor shapes: enable_v2_tensorshape(), disable_v2_tensorshape()
# 4. Control flow: enable_control_flow_v2(), disable_control_flow_v2()
# 5. Tensors comparison: enable_tensor_equality(), disable_tensor_equality()
# 6. Some internal uses of tf.data symbols
# For more details, see
# - https://www.tensorflow.org/guide/migrate
# - the source code of disable_v2_behavior()

# We can simply disable all TensorFlow 2.x behaviors by disable_v2_behavior(), but some
# features in TensorFlow 2.x are useful such as `Tensor shapes`. Actually we use `Tensor
# shapes` in DeepXDE.
tf.disable_v2_behavior()
tf.enable_v2_tensorshape()

# In terms of functionality, we only need to disable eager mode.
# tf.disable_eager_execution()
# It hurts performance a lot (only in some cases?) if enabling tensor equality.
# tf.disable_tensor_equality()
# It hurts performance a little (only in some cases?) if enabling resource variables.
# tf.disable_resource_variables()
# It hurts performance a little (only in some cases?) if enabling control flow.
# tf.disable_control_flow_v2()


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


def size(tensor):
    return tf.get_static_value(tf.size(tensor)).item()


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


def sparse_tensor(indices, values, shape):
    return tf.sparse.SparseTensor(indices, values, shape)


def from_numpy(np_array):
    # Do memory copy:
    # https://stackoverflow.com/questions/47519802/does-tensorflow-convert-to-tensor-do-memory-copy
    # To avoid memory copy, use implicit conversion, but memory copy is still possible.
    # https://www.tensorflow.org/tutorials/customization/basics#numpy_compatibility
    return tf.convert_to_tensor(np_array)


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


def sparse_dense_matmul(x, y):
    return tf.sparse.sparse_dense_matmul(x, y)
