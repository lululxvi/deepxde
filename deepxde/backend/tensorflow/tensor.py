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
