from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


_BACKEND = "tensorflow.compat.v1"
_VERSION = tf.__version__


print("Using backend: tensorflow.compat.v1\n")
# Disable TF eager mode
tf = tf.compat.v1
tf.disable_v2_behavior()


def backend():
    """Returns the name and version of the current backend, e.g., ("tensorflow", 1.14.0).

    Returns:
        tuple: A ``tuple`` of the name and version of the backend DeepXDE is currently using.

    Examples:

    .. code-block:: python

        >>> dde.backend.backend()
        ("tensorflow", 1.14.0)
    """
    return _BACKEND, _VERSION
