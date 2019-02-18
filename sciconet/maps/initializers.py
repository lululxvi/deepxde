from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get(identifier):
    identifiers = {
        "He normal": tf.variance_scaling_initializer(scale=2.0),
        "LeCun normal": tf.variance_scaling_initializer(),
        "Glorot normal": tf.glorot_normal_initializer(),
        "Glorot uniform": tf.glorot_uniform_initializer(),
        "Orthogonal": tf.orthogonal_initializer(),
    }

    if isinstance(identifier, str):
        return identifiers[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(
            "Could not interpret initializer identifier: " + str(identifier)
        )
