"""tensorflow backend implementation"""
from __future__ import absolute_import

from distutils.version import LooseVersion

import tensorflow as tf


if LooseVersion(tf.__version__) < LooseVersion("2.2.0"):
    raise RuntimeError("DeepXDE requires tensorflow>=2.2.0.")


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


def is_tensor(obj):
    return tf.is_tensor(obj)


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


def square(x):
    return tf.math.square(x)


def tanh(x):
    return tf.math.tanh(x)


def mean(input_tensor, dim, keepdims=False):
    return tf.math.reduce_mean(input_tensor, axis=dim, keepdims=keepdims)


def reduce_mean(input_tensor):
    return tf.math.reduce_mean(input_tensor)
