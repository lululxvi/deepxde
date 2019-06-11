from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def linear(x):
    return x


def get(identifier):
    if identifier is None:
        return linear
    if isinstance(identifier, str):
        return {
            "elu": tf.nn.elu,
            "relu": tf.nn.relu,
            "selu": tf.nn.selu,
            "sigmoid": tf.nn.sigmoid,
            "sin": tf.sin,
            "tanh": tf.nn.tanh,
        }[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(
            "Could not interpret activation function identifier:", identifier
        )
