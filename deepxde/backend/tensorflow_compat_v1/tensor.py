"""tensorflow.compat.v1 backend implementation"""
from __future__ import absolute_import

from distutils.version import LooseVersion

import tensorflow.compat.v1 as tf


if LooseVersion(tf.__version__) < LooseVersion("2.2.0"):
    raise RuntimeError("DeepXDE requires tensorflow>=2.2.0.")


def is_tensor(obj):
    return tf.is_tensor(obj)
